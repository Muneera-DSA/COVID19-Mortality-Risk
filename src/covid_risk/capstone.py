"""Part 1: audit of the original capstone notebook on its own data.

    python -m covid_risk.capstone            # writes reports/capstone_audit.{md,json}

1. Facts about the extract (duplicates, codes, columns that are copies of each other).
2. The original notebook's pipeline, re-implemented faithfully, reproduces its published
   numbers (accuracy 0.82, ROC-AUC 0.90).
3. One flaw is removed at a time; every variant is scored on the SAME untouched 20% test set of
   real patients, so each change in the metrics is attributable to that one fix.
4. A corrected model on the extract (confirmed COVID only, triage features only).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, average_precision_score
from sklearn.model_selection import train_test_split

from . import config, features, models, stats

PLACEHOLDER_DATE = "9999-99-99"
CAPSTONE_COMORBIDITIES = {"DIABETES": "diabetes", "COPD": "copd", "ASTHMA": "asthma", "INMSUPR": "immunosuppression",
                          "HIPERTENSION": "hypertension", "OTHER_DISEASE": "other_disease",
                          "CARDIOVASCULAR": "cardiovascular", "OBESITY": "obesity", "RENAL_CHRONIC": "chronic_kidney",
                          "TOBACCO": "smoking"}
LEAKY_ORIGINAL = ["ICU", "PATIENT_TYPE_ORDINAL", "INTUBED_ORDINAL"]
# CLASSIFICATION_FINAL contains code 2 ("confirmed by committee"), which is assigned only to deaths
OUTCOME_CODED = ["CLASSIFICATION_FINAL", "CLASSIFICATION_FINAL_SEVERITY_SEVERE"]


def load(path: Path = config.CAPSTONE_CSV) -> pd.DataFrame:
    raw = pd.read_csv(path)
    raw["DIED"] = (raw.DATE_DIED.astype(str) != PLACEHOLDER_DATE).astype(int)
    return raw


# ---------------------------------------------------------------------- facts

def facts(raw: pd.DataFrame) -> dict:
    no_outcome = raw.drop(columns=["DIED"])
    dup = no_outcome.duplicated()
    by_outcome = {}
    for d, name in ((0, "survivors"), (1, "deaths")):
        sub = no_outcome[raw.DIED == d]
        sizes = sub.groupby(list(sub.columns), dropna=False).size()
        by_outcome[name] = {"rows": int(len(sub)), "distinct": int(len(sizes)),
                            "share_with_identical_row": float(sizes[sizes > 1].sum() / len(sub)),
                            "largest_identical_group": int(sizes.max())}
    dedup = raw[~dup]
    com_score_with_99 = raw[[c for c in CAPSTONE_COMORBIDITIES if c != "TOBACCO"] + ["PNEUMONIA"]].sum(axis=1)
    rate = raw.groupby("CLASIFFICATION_FINAL").DIED.agg(["mean", "size"])
    return {
        "rows": int(len(raw)), "deaths": int(raw.DIED.sum()), "death_rate": float(raw.DIED.mean()),
        "exact_duplicate_rows": int(dup.sum()), "duplicates_by_outcome": by_outcome,
        "after_dropping_duplicates": {"rows": int(len(dedup)), "death_rate": float(dedup.DIED.mean()),
                                      "survivors_removed": int((dup & (raw.DIED == 0)).sum()),
                                      "deaths_removed": int((dup & (raw.DIED == 1)).sum())},
        "icu_identical_to_hospitalised": bool(((raw.PATIENT_TYPE - 1) == raw.ICU).all()),
        "intubation_code3_equals_outpatient": bool(((raw.INTUBED == 3) == (raw.PATIENT_TYPE == 1)).all()),
        "death_rate_intubation_unknown": float(raw.loc[raw.INTUBED == 4, "DIED"].mean()),
        "pneumonia_missing_code_99": int((raw.PNEUMONIA == 99).sum()),
        "max_comorbidity_score_original_formula": int(com_score_with_99.max()),
        "rows_with_score_inflated_by_99": int((raw.PNEUMONIA == 99).sum()),
        "death_rate_by_classification": {int(k): {"rate": float(v["mean"]), "n": int(v["size"])} for k, v in rate.iterrows()},
        "death_dates": [str(pd.to_datetime(raw.loc[raw.DIED == 1, "DATE_DIED"], format="%d-%m-%Y").min().date()),
                        str(pd.to_datetime(raw.loc[raw.DIED == 1, "DATE_DIED"], format="%d-%m-%Y").max().date())],
        "medical_unit_values": sorted(int(x) for x in raw.MEDICAL_UNIT.unique()),
    }


# --------------------------------------------- the original notebook, faithfully

def original_features(raw: pd.DataFrame, dedup: bool) -> tuple[pd.DataFrame, np.ndarray]:
    """Feature engineering exactly as in the notebook (cells 12-203), including its errors."""
    df = raw.drop(columns=["DIED"]).rename(columns={"CLASIFFICATION_FINAL": "CLASSIFICATION_FINAL",
                                                    "HIPERTENSION": "HYPERTENSION"})
    if dedup:
        df = df.drop_duplicates()
    df = df.copy()
    df["DIED"] = (df.DATE_DIED.astype(str) != PLACEHOLDER_DATE).astype(int)
    com = ["DIABETES", "COPD", "ASTHMA", "INMSUPR", "HYPERTENSION", "CARDIOVASCULAR", "RENAL_CHRONIC",
           "OTHER_DISEASE", "OBESITY", "PNEUMONIA"]                 # PNEUMONIA codes 1/2/99 summed in
    df["COMORBIDITY_SCORE"] = df[com].sum(axis=1)
    df["PREGNANT"] = df.PREGNANT.replace({2: 0, 3: 0, 4: 0})
    q1, q3 = df.AGE.quantile([0.25, 0.75])
    df["AGE"] = df.AGE.clip(q1 - 1.5 * (q3 - q1), q3 + 1.5 * (q3 - q1))   # IQR capping of age
    df["COMORBIDITY_SCORE_LOG"] = np.log(df.COMORBIDITY_SCORE + 1)
    df["COMORBIDITY_SCORE_SQRT"] = np.sqrt(df.COMORBIDITY_SCORE)
    df["AGE_GROUP"] = pd.cut(df.AGE, [0, 18, 40, 60, np.inf], labels=["0-18", "19-40", "41-60", "61+"], right=False)
    df["PNEUMONIA_ORDINAL"] = df.PNEUMONIA.map({1: 2, 2: 1, 99: np.nan}).fillna(df.PNEUMONIA.mode()[0])
    df["PATIENT_TYPE_ORDINAL"] = df.PATIENT_TYPE
    df["INTUBED_ORDINAL"] = df.INTUBED.map({1: 4, 2: 3, 3: 2, 4: 1})
    df["CLASSIFICATION_FINAL_SEVERITY"] = np.where(df.CLASSIFICATION_FINAL <= 4, "MILD to MODERATE", "SEVERE")
    df = pd.get_dummies(df, columns=["AGE_GROUP"], prefix="AGEGROUP")
    X = df.drop(columns=["PNEUMONIA", "PATIENT_TYPE", "INTUBED", "AGE", "USMER", "MEDICAL_UNIT", "DIED", "DATE_DIED",
                         "COMORBIDITY_SCORE"])
    X = pd.get_dummies(X, drop_first=True).astype(float)
    return X, df.DIED.to_numpy()


def original_model(seed: int = config.SEED) -> GradientBoostingClassifier:
    return GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=seed)


def plain_smote(X: np.ndarray, y: np.ndarray, seed: int = config.SEED):
    """imblearn's default SMOTE (k=5, all features interpolated), as used in the notebook."""
    return models.smote_nc(np.asarray(X, float), np.asarray(y), continuous=list(range(np.asarray(X).shape[1])), seed=seed)


# ----------------------------------------------------------- corrected cohort

def to_cohort(raw: pd.DataFrame) -> pd.DataFrame:
    """Correct decoding of the extract (official catalogue meanings) into the shared English columns."""
    out = pd.DataFrame({"age": raw.AGE.astype(float), "female": (raw.SEX == 1).astype(int),
                        "pregnant": (raw.PREGNANT == 1).astype(int)})
    for src, dst in CAPSTONE_COMORBIDITIES.items():
        out[dst] = raw[src].astype(int)
    out["pneumonia"] = (raw.PNEUMONIA == 1).astype(int)
    out["pneumonia_missing"] = (raw.PNEUMONIA == 99).astype(int)
    out["hospitalised"] = (raw.PATIENT_TYPE == 2).astype(int)
    out["icu"] = raw.ICU.astype(int)
    out["intubated"] = (raw.INTUBED == 1).astype(int)
    out["classification"] = raw.CLASIFFICATION_FINAL.astype(int)
    out["died"] = raw.DIED.astype(int)
    features.check_cohort(out)
    return out


# ------------------------------------------------------------------ evaluation

def score(y: np.ndarray, p: np.ndarray, n_boot: int, seed: int = 0) -> dict:
    auc = stats.delong_auc(y, p)
    pr = stats.bootstrap_metric(lambda i: average_precision_score(y[i], p[i]), len(y), n_boot, seed)
    cal = stats.calibration_intercept_slope(y, p)
    return {"n": int(len(y)), "death_rate": float(y.mean()), "accuracy_at_0.5": float(accuracy_score(y, p >= 0.5)),
            "roc_auc": auc["value"], "roc_auc_ci": auc["ci"], "pr_auc": pr["value"], "pr_auc_ci": pr["ci"],
            "brier": stats.brier(y, p), "ece": stats.expected_calibration_error(y, p),
            "mean_predicted_risk": cal["mean_predicted"], "calibration_intercept": cal["intercept"],
            "calibration_slope": cal["slope"], "share_predicted_to_die": float((p >= 0.5).mean())}


def flaw_by_flaw(raw: pd.DataFrame, n_boot: int = 300, seed: int = config.SEED) -> dict:
    out = {}
    # S0: the notebook as published: de-duplicate everything, then split
    X0, y0 = original_features(raw, dedup=True)
    tr, te = train_test_split(np.arange(len(y0)), test_size=0.2, random_state=seed, stratify=y0)
    Xs, ys = plain_smote(X0.iloc[tr].to_numpy(), y0[tr], seed)
    m0 = original_model(seed).fit(Xs, ys)
    p0 = m0.predict_proba(X0.iloc[te].to_numpy())[:, 1]
    out["S0_original_as_published"] = score(y0[te], p0, n_boot)
    out["S0_original_as_published"]["features"] = list(X0.columns)

    # one fixed test set of real patients for every corrected variant
    idx = np.arange(len(raw))
    tr_i, te_i = train_test_split(idx, test_size=0.2, random_state=seed, stratify=raw.DIED)
    train_raw, test_raw = raw.iloc[tr_i], raw.iloc[te_i]
    Xte_full, yte = original_features(test_raw, dedup=False)

    def fit_original(train_frame, dedup, smote=True, drop=()):
        Xtr, ytr = original_features(train_frame, dedup=dedup)
        Xtr = Xtr.drop(columns=list(drop))
        cols = list(Xtr.columns)
        Xa = Xtr.to_numpy()
        if smote:
            Xa, ytr = plain_smote(Xa, ytr, seed)
        m = original_model(seed).fit(Xa, ytr)
        return m.predict_proba(Xte_full.reindex(columns=cols, fill_value=0).to_numpy())[:, 1]

    out["S1_original_model_on_real_patients"] = score(yte, fit_original(train_raw, dedup=True), n_boot)
    out["S2_keep_all_patients"] = score(yte, fit_original(train_raw, dedup=False), n_boot)
    out["S3_remove_post_triage_leakage"] = score(yte, fit_original(train_raw, False, drop=LEAKY_ORIGINAL), n_boot)
    out["S4_no_smote"] = score(yte, fit_original(train_raw, False, smote=False, drop=LEAKY_ORIGINAL), n_boot)
    out["S4b_remove_outcome_defined_code"] = score(yte, fit_original(train_raw, False, smote=False,
                                                                     drop=LEAKY_ORIGINAL + OUTCOME_CODED), n_boot)

    # S5: correct decoding + continuous age, same population and test set, gradient boosting
    cohort = to_cohort(raw)
    feats = features.triage_features(cohort)
    m5 = models.build("hgb", feats).fit(features.matrix(cohort.iloc[tr_i], feats), cohort.died.iloc[tr_i])
    out["S5_correct_coding_and_age"] = score(yte, m5.predict_proba(features.matrix(cohort.iloc[te_i], feats))[:, 1], n_boot)
    out["S5_correct_coding_and_age"]["features"] = feats

    # leakage size on the corrected features: add the post-triage columns back
    leak_feats = feats + ["hospitalised", "intubated"]          # icu is an exact copy of hospitalised
    m6 = models.build("hgb", leak_feats).fit(features.matrix(cohort.iloc[tr_i], leak_feats), cohort.died.iloc[tr_i])
    p6 = m6.predict_proba(features.matrix(cohort.iloc[te_i], leak_feats))[:, 1]
    out["S6_corrected_plus_post_triage"] = score(yte, p6, n_boot)
    p5 = m5.predict_proba(features.matrix(cohort.iloc[te_i], feats))[:, 1]
    out["leakage_test"] = stats.delong_paired(yte, p6, p5)
    out["leakage_pr_auc"] = stats.bootstrap_diff(average_precision_score, yte, p6, p5, n_boot, seed)
    return out


def corrected_confirmed(raw: pd.DataFrame, n_boot: int = 300, seed: int = config.SEED) -> dict:
    """Corrected study on the extract: confirmed COVID (codes 1, 3) only; three models."""
    cohort = to_cohort(raw)
    cohort = cohort[cohort.classification.isin(config.CONFIRMED_CODES)].reset_index(drop=True)
    feats = features.triage_features(cohort)
    tr, te = train_test_split(np.arange(len(cohort)), test_size=0.2, random_state=seed, stratify=cohort.died)
    y = cohort.died.to_numpy()
    res, probs = {"n_patients": int(len(cohort)), "death_rate": float(y.mean()), "features": feats}, {}
    for name, fs in (("age_only", ["age"]), ("logistic", feats), ("hgb", feats)):
        m = models.build(name, fs).fit(features.matrix(cohort.iloc[tr], fs), y[tr])
        probs[name] = m.predict_proba(features.matrix(cohort.iloc[te], fs))[:, 1]
        res[name] = score(y[te], probs[name], n_boot)
        res[name]["capacity"] = [stats.capture_at_capacity(y[te], probs[name], s) for s in config.CAPACITY_SHARES]
    res["logistic_vs_age_only"] = stats.delong_paired(y[te], probs["logistic"], probs["age_only"])
    res["hgb_vs_logistic"] = stats.delong_paired(y[te], probs["hgb"], probs["logistic"])
    res["hgb_vs_logistic_pr_auc"] = stats.bootstrap_diff(average_precision_score, y[te], probs["hgb"], probs["logistic"],
                                                         n_boot, seed)
    return res


def to_markdown(r: dict) -> str:
    f, ff = r["facts"], r["flaw_by_flaw"]
    d = f["duplicates_by_outcome"]
    L = ["# Audit of the original capstone notebook", "",
         "Generated by `python -m covid_risk.capstone` from `data/covid_capstone.csv.gz`.", "",
         "## The extract", "",
         f"- {f['rows']:,} patients, {f['deaths']:,} deaths (**{f['death_rate']:.1%}**); deaths dated "
         f"{f['death_dates'][0]} to {f['death_dates'][1]}.",
         f"- {f['exact_duplicate_rows']:,} rows are exact copies of another row. They are different patients with the same "
         f"profile: {d['survivors']['share_with_identical_row']:.1%} of survivors share a row with someone else "
         f"(largest group: {d['survivors']['largest_identical_group']} identical people), against "
         f"{d['deaths']['share_with_identical_row']:.1%} of deaths (each death has its own date).",
         f"- Dropping them removed {f['after_dropping_duplicates']['survivors_removed']:,} survivors but only "
         f"{f['after_dropping_duplicates']['deaths_removed']:,} deaths, raising the death rate to "
         f"**{f['after_dropping_duplicates']['death_rate']:.1%}**.",
         f"- `ICU` is identical to `PATIENT_TYPE` (hospitalised) in every row: {f['icu_identical_to_hospitalised']}. "
         f"Intubation code 3 marks exactly the outpatients: {f['intubation_code3_equals_outpatient']}.",
         f"- Intubation status 'unknown' (code 4): death rate {f['death_rate_intubation_unknown']:.1%}, a recording artefact.",
         f"- Pneumonia code 99 (missing) in {f['pneumonia_missing_code_99']:,} rows; summed into the notebook's comorbidity "
         f"score it creates scores up to {f['max_comorbidity_score_original_formula']}.",
         "- Death rate by `CLASIFFICATION_FINAL` (1-3 = confirmed COVID; 2 = confirmed by committee, a code used only "
         "for deaths; 4-7 = not confirmed):", "",
         "| Code | Patients | Death rate |", "|---|---|---|"]
    L += [f"| {k} | {v['n']:,} | {v['rate']:.1%} |" for k, v in sorted(f["death_rate_by_classification"].items())]
    rows = [("S0", "Original notebook as published (de-duplicated data, its own test split)", "S0_original_as_published"),
            ("S1", "Same model, scored on real patients (untouched 20% test set)", "S1_original_model_on_real_patients"),
            ("S2", "Keep all patients when training", "S2_keep_all_patients"),
            ("S3", "Remove ICU / intubation / hospitalisation", "S3_remove_post_triage_leakage"),
            ("S4", "Remove SMOTE", "S4_no_smote"),
            ("S4b", "Remove `CLASSIFICATION_FINAL` (code 2 is given only to deaths)", "S4b_remove_outcome_defined_code"),
            ("S5", "Correct coding (pneumonia, pregnancy), continuous age, histogram gradient boosting", "S5_correct_coding_and_age"),
            ("S6", "S5 + hospitalisation and intubation (leakage check)", "S6_corrected_plus_post_triage")]
    L += ["", "## Flaw by flaw", "",
          "S1-S6 are scored on the same 20% of real patients (death rate "
          f"{ff['S1_original_model_on_real_patients']['death_rate']:.1%}). 95% CIs: DeLong for ROC-AUC, bootstrap for PR-AUC.",
          "", "| Step | Change | ROC-AUC | PR-AUC | Accuracy at 0.5 | Mean predicted risk | Calibration slope |",
          "|---|---|---|---|---|---|---|"]
    for sid, text, key in rows:
        s = ff[key]
        L.append(f"| {sid} | {text} | {s['roc_auc']:.3f} ({s['roc_auc_ci'][0]:.3f}-{s['roc_auc_ci'][1]:.3f}) | "
                 f"{s['pr_auc']:.3f} ({s['pr_auc_ci'][0]:.3f}-{s['pr_auc_ci'][1]:.3f}) | {s['accuracy_at_0.5']:.3f} | "
                 f"{s['mean_predicted_risk']:.1%} (observed {s['death_rate']:.1%}) | {s['calibration_slope']:.2f} |")
    lt, lp = ff["leakage_test"], ff["leakage_pr_auc"]
    L += ["", f"Leakage: adding hospitalisation and intubation raises ROC-AUC by {lt['diff']:+.3f} "
          f"(CI {lt['ci'][0]:+.3f} to {lt['ci'][1]:+.3f}, DeLong p = {lt['p']:.2g}) and PR-AUC by {lp['value']:+.3f} "
          f"(CI {lp['ci'][0]:+.3f} to {lp['ci'][1]:+.3f}).",
          f"S0 used {len(ff['S0_original_as_published']['features'])} features: "
          + ", ".join(f"`{c}`" for c in ff["S0_original_as_published"]["features"]) + "."]
    c = r["corrected_confirmed"]
    L += ["", "## Corrected model on the extract (confirmed COVID, codes 1 and 3)", "",
          f"{c['n_patients']:,} patients, death rate {c['death_rate']:.1%}; triage features only; 80/20 stratified split.", "",
          "| Model | ROC-AUC | PR-AUC | Brier | Calibration slope | Deaths caught in top 10% / 20% |", "|---|---|---|---|---|---|"]
    for name, label in (("age_only", "Age only (baseline)"), ("logistic", "Logistic regression"), ("hgb", "Gradient boosting")):
        s = c[name]
        cap = {round(x["share_flagged"], 2): x["deaths_captured"] for x in s["capacity"]}
        L.append(f"| {label} | {s['roc_auc']:.3f} ({s['roc_auc_ci'][0]:.3f}-{s['roc_auc_ci'][1]:.3f}) | "
                 f"{s['pr_auc']:.3f} ({s['pr_auc_ci'][0]:.3f}-{s['pr_auc_ci'][1]:.3f}) | {s['brier']:.3f} | "
                 f"{s['calibration_slope']:.2f} | {cap[0.1]:.1%} / {cap[0.2]:.1%} |")
    a, b, bp = c["logistic_vs_age_only"], c["hgb_vs_logistic"], c["hgb_vs_logistic_pr_auc"]
    L += ["", f"- Logistic vs age only: ROC-AUC {a['diff']:+.3f} (CI {a['ci'][0]:+.3f} to {a['ci'][1]:+.3f}), p = {a['p']:.2g}.",
          f"- Gradient boosting vs logistic: ROC-AUC {b['diff']:+.3f} (CI {b['ci'][0]:+.3f} to {b['ci'][1]:+.3f}), p = {b['p']:.2g}; "
          f"PR-AUC {bp['value']:+.3f} (CI {bp['ci'][0]:+.3f} to {bp['ci'][1]:+.3f})."]
    return "\n".join(L) + "\n"


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", type=Path, default=config.CAPSTONE_CSV)
    ap.add_argument("--reports-dir", type=Path, default=config.REPORTS_DIR)
    ap.add_argument("--n-boot", type=int, default=300)
    args = ap.parse_args(argv)
    raw = load(args.data)
    r = {"facts": facts(raw), "flaw_by_flaw": flaw_by_flaw(raw, args.n_boot), "corrected_confirmed": corrected_confirmed(raw, args.n_boot)}
    args.reports_dir.mkdir(parents=True, exist_ok=True)
    (args.reports_dir / "capstone_audit.json").write_text(json.dumps(r, indent=2, default=float))
    md = to_markdown(r)
    (args.reports_dir / "capstone_audit.md").write_text(md)
    print(md)
    return r


if __name__ == "__main__":
    main()
