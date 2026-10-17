"""Part 2 experiments on the official cohort. Each stage writes results/<stage>.json and is skipped
when that file exists (delete it, or use --redo, to recompute).

  eda        descriptive analysis: mortality by age, month, state, condition; crude odds ratios
  benchmark  screen of 13+ model families by 5-fold CV (LazyPredict done properly: no test set)
  select     tuning + 5-fold CV on development data: 3 models x 4 imbalance strategies
  temporal   fit on 2020, test on 2021 (external in time): discrimination, calibration, decisions
  iecv       internal-external cross-validation by state (each state held out in turn)
  leakage    add post-triage variables (hospitalisation, ICU, intubation): how much is inflated?
  subgroups  performance and calibration by sex, age band, health-care sector, pneumonia
  explain    SHAP for the chosen model (and gradient boosting) + adjusted odds ratios
  monitoring rolling month-by-month recalibration of the intercept (previous month only)

    python -m covid_risk.experiments --cohort artifacts/cohort.pkl --stages select temporal ...
"""

from __future__ import annotations

import argparse
import itertools
import json
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy import stats as sps
from scipy.special import expit, logit
from sklearn.metrics import average_precision_score, log_loss, roc_auc_score
from sklearn.model_selection import StratifiedKFold

from . import config, features, models, stats

STAGES = ["eda", "benchmark", "select", "temporal", "iecv", "leakage", "subgroups", "explain", "monitoring"]
IMBALANCE = ["none", "class_weight", "oversample", "smote"]
HGB_GRID = {"learning_rate": [0.05, 0.1], "max_leaf_nodes": [15, 31, 63], "min_samples_leaf": [100, 500]}
LOGISTIC_GRID = {"C": [0.01, 0.1, 1.0, 10.0]}


# ------------------------------------------------------------------- helpers

def _json(obj) -> str:
    return json.dumps(obj, indent=2, default=lambda o: o.item() if hasattr(o, "item") else str(o))


def _subsample(df: pd.DataFrame, n: int, seed: int) -> pd.DataFrame:
    return df.iloc[stats.stratified_subsample(df.died.to_numpy(), n, seed)].reset_index(drop=True)


def fold_metrics(y, p) -> dict:
    cal = stats.calibration_intercept_slope(y, p)
    return {"roc_auc": float(roc_auc_score(y, p)), "pr_auc": float(average_precision_score(y, p)),
            "brier": stats.brier(y, p), "log_loss": float(log_loss(y, np.clip(p, 1e-6, 1 - 1e-6))),
            "cal_intercept": cal["intercept"], "cal_slope": cal["slope"]}


def corrected_t(diffs: np.ndarray, n_train: int, n_test: int) -> float:
    """Nadeau & Bengio (2003) corrected resampled t-test for CV score differences. Two-sided p."""
    k = len(diffs)
    var = np.var(diffs, ddof=1)
    if var == 0:
        return 1.0 if np.mean(diffs) == 0 else 0.0
    t = np.mean(diffs) / np.sqrt((1 / k + n_test / n_train) * var)
    return float(2 * sps.t.sf(abs(t), k - 1))


def full_metrics(y, p, n_boot: int, seed: int = 0) -> dict:
    """Test-set metrics with CIs. ROC-AUC: DeLong on all rows; PR-AUC: bootstrap on a stratified subsample."""
    y, p = np.asarray(y), np.asarray(p, float)
    auc = stats.delong_auc(y, p)
    keep = stats.stratified_subsample(y, config.BOOT_MAX_ROWS, seed)
    ys, ps = y[keep], p[keep]
    pr = stats.bootstrap_metric(lambda i: average_precision_score(ys[i], ps[i]), len(ys), n_boot, seed)
    cal = stats.calibration_intercept_slope(y, p)
    return {"n": int(len(y)), "deaths": int(y.sum()), "death_rate": float(y.mean()),
            "roc_auc": auc["value"], "roc_auc_ci": auc["ci"], "pr_auc": pr["value"], "pr_auc_ci": pr["ci"],
            "brier": stats.brier(y, p), "ece": stats.expected_calibration_error(y, p),
            "calibration": cal, "reliability": stats.reliability_curve(y, p, 10),
            "capacity": [stats.capture_at_capacity(y, p, s) for s in config.CAPACITY_SHARES],
            "net_benefit": stats.net_benefit(y, p, config.DECISION_THRESHOLDS),
            "decision_curve": stats.net_benefit(y, p, np.round(np.arange(0.01, 0.51, 0.01), 2))}


def feats_for(model: str, triage: list[str]) -> list[str]:
    return ["age"] if model == "age_only" else triage


# ------------------------------------------------------------------- stages

def stage_select(dev: pd.DataFrame, out: Path, seed: int, n_sample: int = config.SELECTION_SAMPLE) -> dict:
    sub = _subsample(dev, n_sample, seed)
    triage = features.triage_features(sub)
    y = sub.died.to_numpy()
    res = {"n_sample": int(len(sub)), "death_rate": float(y.mean()), "features": triage, "tuning": {}, "cv": {}}

    # 1) tuning (3-fold, no imbalance correction), metric: PR-AUC, tie-break log loss
    tune_folds = list(StratifiedKFold(3, shuffle=True, random_state=seed).split(sub, y))
    for name, grid in (("logistic", LOGISTIC_GRID), ("hgb", HGB_GRID)):
        X = features.matrix(sub, triage)
        rows = []
        for values in itertools.product(*grid.values()):
            params = dict(zip(grid.keys(), values))
            pr, ll = [], []
            for tr, te in tune_folds:
                m = models.build(name, triage, params=params, seed=seed).fit(X[tr], y[tr])
                p = m.predict_proba(X[te])[:, 1]
                pr.append(average_precision_score(y[te], p))
                ll.append(log_loss(y[te], np.clip(p, 1e-6, 1 - 1e-6)))
            rows.append({"params": params, "pr_auc": float(np.mean(pr)), "log_loss": float(np.mean(ll))})
            print(f"[tune] {name} {params}: PR-AUC {np.mean(pr):.4f}")
        best = max(rows, key=lambda r: (round(r["pr_auc"], 4), -r["log_loss"]))
        res["tuning"][name] = {"grid": rows, "best": best["params"]}

    # 2) 5-fold CV: every model x imbalance strategy, tuned parameters
    folds = list(StratifiedKFold(config.CV_FOLDS, shuffle=True, random_state=seed + 1).split(sub, y))
    configs = [("age_only", "none")] + [(m, i) for m in ("logistic", "hgb") for i in IMBALANCE]
    for name, imb in configs:
        fs = feats_for(name, triage)
        X = features.matrix(sub, fs)
        params = res["tuning"].get(name, {}).get("best", {})
        per_fold, t0 = [], time.time()
        for tr, te in folds:
            m = models.build(name, fs, imb, params, seed).fit(X[tr], y[tr])
            per_fold.append(fold_metrics(y[te], m.predict_proba(X[te])[:, 1]))
        key = f"{name}|{imb}"
        res["cv"][key] = {"folds": per_fold, **{k: float(np.mean([f[k] for f in per_fold])) for k in per_fold[0]},
                          **{k + "_sd": float(np.std([f[k] for f in per_fold], ddof=1)) for k in per_fold[0]}}
        print(f"[cv] {key}: PR-AUC {res['cv'][key]['pr_auc']:.4f}  ROC-AUC {res['cv'][key]['roc_auc']:.4f}  "
              f"slope {res['cv'][key]['cal_slope']:.2f}  ({time.time() - t0:.0f}s)")

    # 3) tests: imbalance strategies vs none (gradient boosting and logistic), models vs each other
    n_tr = int(len(sub) * (config.CV_FOLDS - 1) / config.CV_FOLDS)
    n_te = len(sub) - n_tr
    tests = {}
    for name in ("logistic", "hgb"):
        base = np.array([f["pr_auc"] for f in res["cv"][f"{name}|none"]["folds"]])
        for imb in IMBALANCE[1:]:
            alt = np.array([f["pr_auc"] for f in res["cv"][f"{name}|{imb}"]["folds"]])
            tests[f"{name}|{imb}_vs_none"] = {"diff_pr_auc": float((alt - base).mean()), "p": corrected_t(alt - base, n_tr, n_te),
                                             "diff_cal_slope": float(np.mean([f["cal_slope"] for f in res["cv"][f"{name}|{imb}"]["folds"]]) - 1)}
    res["tests"] = tests

    # 4) choice: best mean PR-AUC; imbalance correction only if it is significantly better than none
    best_model = max(("logistic", "hgb"), key=lambda m: res["cv"][f"{m}|none"]["pr_auc"])
    imb_choice = "none"
    for imb in IMBALANCE[1:]:
        t = tests[f"{best_model}|{imb}_vs_none"]
        if t["diff_pr_auc"] > 0 and t["p"] < 0.05:
            imb_choice = imb
    res["choice"] = {"model": best_model, "imbalance": imb_choice, "params": res["tuning"][best_model]["best"],
                     "rule": "highest mean CV PR-AUC; imbalance correction only if significantly better than none (corrected t-test)"}
    (out / "select.json").write_text(_json(res))
    return res


def _fit(name: str, imb: str, params: dict, df: pd.DataFrame, fs: list[str], seed: int):
    return models.build(name, fs, imb, params, seed).fit(features.matrix(df, fs), df.died.to_numpy())


def stage_temporal(dev, test, out: Path, art: Path, sel: dict, seed: int, n_boot: int) -> dict:
    triage = sel["features"]
    choice = sel["choice"]
    res, preds = {"choice": choice, "dev": {"n": int(len(dev)), "death_rate": float(dev.died.mean())}}, {}
    (art / "models").mkdir(parents=True, exist_ok=True)
    for name in ("age_only", "logistic", "hgb"):
        fs = feats_for(name, triage)
        imb = choice["imbalance"] if name == choice["model"] else "none"
        params = sel["tuning"].get(name, {}).get("best", {})
        t0 = time.time()
        m = _fit(name, imb, params, dev, fs, seed)
        joblib.dump({"model": m, "features": fs}, art / "models" / f"{name}.joblib")
        preds[name] = m.predict_proba(features.matrix(test, fs))[:, 1]
        res[name] = full_metrics(test.died.to_numpy(), preds[name], n_boot, seed)
        print(f"[temporal] {name}: ROC-AUC {res[name]['roc_auc']:.4f} PR-AUC {res[name]['pr_auc']:.4f} ({time.time() - t0:.0f}s)")
    y = test.died.to_numpy()
    res["logistic_vs_age_only"] = stats.delong_paired(y, preds["logistic"], preds["age_only"])
    res["hgb_vs_logistic"] = stats.delong_paired(y, preds["hgb"], preds["logistic"])
    res["hgb_vs_logistic_pr_auc"] = stats.bootstrap_diff(average_precision_score, y, preds["hgb"], preds["logistic"],
                                                         n_boot, seed, config.BOOT_MAX_ROWS)
    final = preds[choice["model"]]
    # month by month: drift in discrimination and calibration
    monthly = []
    for month, idx in test.groupby("month").indices.items():
        yy, pp = y[idx], final[idx]
        if yy.sum() >= 20 and (1 - yy).sum() >= 20:
            monthly.append({"month": month, "n": int(len(idx)), "observed": float(yy.mean()), "predicted": float(pp.mean()),
                            "roc_auc": float(roc_auc_score(yy, pp))})
    res["monthly"] = monthly
    # simple update: re-estimate the intercept on the first test month, apply to the rest
    first = test.month == test.month.min()
    if first.any() and (~first).any():
        shift = stats.calibration_intercept_slope(y[first.to_numpy()], final[first.to_numpy()])["intercept"]
        rest = (~first).to_numpy()
        updated = expit(logit(np.clip(final[rest], 1e-6, 1 - 1e-6)) + shift)
        res["recalibration"] = {"fitted_on": str(test.month.min()), "intercept_shift": shift,
                                "before": stats.calibration_intercept_slope(y[rest], final[rest]),
                                "after": stats.calibration_intercept_slope(y[rest], updated),
                                "ece_before": stats.expected_calibration_error(y[rest], final[rest]),
                                "ece_after": stats.expected_calibration_error(y[rest], updated)}
    cols = ["id", "month", "state", "sector", "female", "age", "pneumonia", "died"]
    pred = test[cols].copy()
    for k, v in preds.items():
        pred[f"p_{k}"] = v.astype(np.float32)
    (art / "predictions").mkdir(parents=True, exist_ok=True)
    pred.to_pickle(art / "predictions" / "test_predictions.pkl")
    (out / "temporal.json").write_text(_json(res))
    return res


def stage_iecv(dev, out: Path, sel: dict, seed: int, n_sample: int = config.IECV_SAMPLE) -> dict:
    sub = _subsample(dev, n_sample, seed + 7)
    c = sel["choice"]
    fs = feats_for(c["model"], sel["features"])
    rows = []
    for state in sorted(sub.state.unique()):
        held = sub.state == state
        yy = sub.died[held].to_numpy()
        if yy.sum() < 20 or (1 - yy).sum() < 20:
            rows.append({"state": int(state), "n": int(held.sum()), "deaths": int(yy.sum()), "skipped": True})
            continue
        m = _fit(c["model"], c["imbalance"], c["params"], sub[~held], fs, seed)
        p = m.predict_proba(features.matrix(sub[held], fs))[:, 1]
        auc = stats.delong_auc(yy, p)
        cal = stats.calibration_intercept_slope(yy, p)
        rows.append({"state": int(state), "n": int(held.sum()), "deaths": int(yy.sum()), "observed": float(yy.mean()),
                     "predicted": float(p.mean()), "roc_auc": auc["value"], "roc_auc_ci": auc["ci"],
                     "pr_auc": float(average_precision_score(yy, p)), "cal_intercept": cal["intercept"], "cal_slope": cal["slope"]})
        print(f"[iecv] state {state:2d}: n={held.sum():6d} ROC-AUC {auc['value']:.3f}")
    ok = [r for r in rows if not r.get("skipped")]
    aucs = np.array([r["roc_auc"] for r in ok])
    res = {"n_sample": int(len(sub)), "states": rows,
           "summary": {"n_states": len(ok), "median_auc": float(np.median(aucs)), "min_auc": float(aucs.min()),
                       "max_auc": float(aucs.max()), "iqr_auc": [float(np.quantile(aucs, .25)), float(np.quantile(aucs, .75))],
                       "median_cal_slope": float(np.median([r["cal_slope"] for r in ok])),
                       "range_cal_intercept": [float(min(r["cal_intercept"] for r in ok)), float(max(r["cal_intercept"] for r in ok))]}}
    (out / "iecv.json").write_text(_json(res))
    return res


def stage_leakage(dev, test, out: Path, art: Path, sel: dict, seed: int, n_boot: int) -> dict:
    c = sel["choice"]
    pred = pd.read_pickle(art / "predictions" / "test_predictions.pkl")
    base = pred[f"p_{c['model']}"].to_numpy()
    y = test.died.to_numpy()
    res = {}
    for label, extra in (("plus_hospitalised", ["hospitalised"]), ("plus_all_post_triage", config.POST_TRIAGE_FEATURES)):
        fs = sel["features"] + extra
        m = _fit(c["model"], c["imbalance"], c["params"], dev, fs, seed)
        p = m.predict_proba(features.matrix(test, fs))[:, 1]
        res[label] = {"features_added": extra, "roc_auc": float(roc_auc_score(y, p)), "pr_auc": float(average_precision_score(y, p)),
                      "vs_triage_model": stats.delong_paired(y, p, base),
                      "pr_auc_diff": stats.bootstrap_diff(average_precision_score, y, p, base, n_boot, seed, config.BOOT_MAX_ROWS)}
        print(f"[leakage] {label}: ROC-AUC {res[label]['roc_auc']:.4f}")
    (out / "leakage.json").write_text(_json(res))
    return res


MAIN_SECTORS = (4, 12, 6, 9, 3)


def stage_subgroups(out: Path, art: Path, sel: dict) -> dict:
    pred = pd.read_pickle(art / "predictions" / "test_predictions.pkl")
    p_col = f"p_{sel['choice']['model']}"
    groups = {
        "sex": pred.female.map({1: "female", 0: "male"}),
        "age": pd.cut(pred.age, [0, 40, 60, 80, 121], right=False, labels=["0-39", "40-59", "60-79", "80+"]).astype(str),
        "sector": pred.sector.map(lambda s: config.SECTORS[int(s)] if int(s) in MAIN_SECTORS else "Other"),
        "pneumonia": pred.pneumonia.map({1: "pneumonia", 0: "no pneumonia"}),
    }
    res = {}
    for gname, g in groups.items():
        rows = []
        for level, idx in pred.groupby(g.values).indices.items():
            yy, pp = pred.died.to_numpy()[idx], pred[p_col].to_numpy()[idx]
            row = {"group": str(level), "n": int(len(idx)), "deaths": int(yy.sum()), "observed": float(yy.mean()),
                   "predicted": float(pp.mean())}
            if yy.sum() >= 20 and (1 - yy).sum() >= 20:
                auc = stats.delong_auc(yy, pp)
                row.update({"roc_auc": auc["value"], "roc_auc_ci": auc["ci"], "roc_auc_se": auc["se"],
                            "cal_intercept": stats.calibration_intercept_slope(yy, pp)["intercept"]})
            rows.append(row)
        res[gname] = rows
    f = next(r for r in res["sex"] if r["group"] == "female")
    m = next(r for r in res["sex"] if r["group"] == "male")
    z = (f["roc_auc"] - m["roc_auc"]) / np.sqrt(f["roc_auc_se"] ** 2 + m["roc_auc_se"] ** 2)
    res["sex_auc_test"] = {"diff_female_minus_male": f["roc_auc"] - m["roc_auc"], "z": float(z), "p": float(2 * sps.norm.sf(abs(z)))}
    (out / "subgroups.json").write_text(_json(res))
    return res


def linear_shap(pipe, X: np.ndarray, background: np.ndarray, fs: list[str]) -> tuple[np.ndarray, float]:
    """Exact interventional SHAP for the logistic model (additive in the original features).

    The model is logit = b + sum_j g_j(x_j), with g_j a spline (age, delay) or a linear term, so the
    SHAP value of feature j is g_j(x_j) - mean over the background of g_j. Returns SHAP values (log-odds)
    per ORIGINAL feature and the base value.
    """
    pre, lr = pipe.named_steps["pre"], pipe.named_steps["lr"]
    Z, Zb = pre.transform(X), pre.transform(background)
    contrib = (Z - Zb.mean(axis=0)) * lr.coef_.ravel()
    cont, other = models._spline_columns(fs)
    sl_spline, sl_rest = pre.output_indices_["spline"], pre.output_indices_["rest"]
    k = (sl_spline.stop - sl_spline.start) // max(len(cont), 1)
    sv = np.zeros((len(X), len(fs)))
    for i, j in enumerate(cont):
        sv[:, j] = contrib[:, sl_spline.start + i * k: sl_spline.start + (i + 1) * k].sum(axis=1)
    for i, j in enumerate(other):
        sv[:, j] = contrib[:, sl_rest.start + i]
    base = float(lr.intercept_[0] + (Zb.mean(axis=0) * lr.coef_.ravel()).sum())
    return sv, base


def explain_model(name: str, art: Path, Xe: np.ndarray, background: np.ndarray, y_e: np.ndarray, seed: int) -> dict:
    bundle = joblib.load(art / "models" / f"{name}.joblib")
    model, mfs = bundle["model"], bundle["features"]
    est = getattr(model, "estimator_", model)
    out = {"model": name, "features": mfs}
    if name == "logistic":
        sv, base = linear_shap(est, Xe, background, mfs)
        out["method"] = "exact linear SHAP (additive logistic model, interventional, development background)"
        lin = est.decision_function(Xe)
        try:   # cross-check against the shap library on the transformed design matrix
            import shap
            pre = est.named_steps["pre"]
            Zb = pre.transform(background)
            ex = shap.LinearExplainer(est.named_steps["lr"], shap.maskers.Independent(Zb, max_samples=len(Zb)))
            lib = np.asarray(ex.shap_values(pre.transform(Xe))).sum(axis=1)
            out["shap_library_check_max_diff"] = float(np.max(np.abs(lib - sv.sum(axis=1))))
        except ImportError:
            out["shap_library_check_max_diff"] = None
    elif name == "hgb":
        try:
            import shap
        except ImportError:
            from sklearn.inspection import permutation_importance
            pi = permutation_importance(est, Xe, y_e, scoring="roc_auc", n_repeats=5, random_state=seed)
            out.update({"method": "permutation importance (shap not installed)",
                        "mean_abs": dict(zip(mfs, pi.importances_mean.astype(float).tolist()))})
            return out
        explainer = shap.TreeExplainer(est)
        sv = np.asarray(explainer.shap_values(Xe))
        if sv.ndim == 3:
            sv = sv[..., 1] if sv.shape[-1] == 2 else sv[1]
        base = float(np.ravel(explainer.expected_value)[-1])
        out["method"] = f"shap.TreeExplainer {shap.__version__}"
        lin = np.ravel(est.decision_function(Xe))
    else:
        raise ValueError(name)
    out["mean_abs"] = dict(zip(mfs, np.abs(sv).mean(axis=0).astype(float).tolist()))
    out["expected_value"] = base
    # completeness: base value + sum of SHAP values must equal the model's log-odds output
    out["additivity_max_error"] = float(np.max(np.abs(sv.sum(axis=1) + base - np.ravel(lin))))
    np.savez_compressed(art / f"shap_{name}.npz", shap=sv.astype(np.float32), X=Xe, features=np.array(mfs))
    return out


def stage_explain(dev, test, out: Path, art: Path, sel: dict, seed: int, n_explain: int = 3000) -> dict:
    res = {}
    fs = sel["features"]
    # adjusted odds ratios: unpenalised logistic regression, age per 10 years, development data
    sub = _subsample(dev, config.SELECTION_SAMPLE, seed + 3)
    cols = list(fs)
    X = sub[cols].to_numpy(float)
    X[:, cols.index("age")] = X[:, cols.index("age")] / 10
    names = ["age (per 10 years)" if c == "age" else c for c in cols]
    ors = stats.logistic_odds_ratios(X, sub.died.to_numpy(), names)
    adj = stats.holm({o["feature"]: o["p"] for o in ors})
    for o in ors:
        o["p_holm"] = adj[o["feature"]]
    res["odds_ratios"] = {"n": int(len(sub)), "rows": ors}
    # SHAP on a stratified sample of 2021 test patients: the chosen model first, gradient boosting too
    idx = stats.stratified_subsample(test.died.to_numpy(), n_explain, seed)
    bg = features.matrix(_subsample(dev, 2000, seed + 5), fs)
    Xe = features.matrix(test.iloc[idx], fs)
    chosen = sel["choice"]["model"]
    res["shap"] = explain_model(chosen, art, Xe, bg, test.died.to_numpy()[idx], seed)
    if chosen != "hgb":
        res["shap_hgb"] = explain_model("hgb", art, Xe, bg, test.died.to_numpy()[idx], seed)
    (out / "explain.json").write_text(_json(res))
    return res


def _band(interval) -> str:
    return f"{int(interval.left)}+" if interval.right >= 121 else f"{int(interval.left)}-{int(interval.right) - 1}"


def stage_eda(dev, test, out: Path) -> dict:
    """Descriptive analysis of the cohort (numbers for the EDA figures and tables)."""
    df = pd.concat([dev, test], ignore_index=True)
    res = {}
    band = pd.cut(df.age, list(range(0, 95, 5)) + [121], right=False)
    g = df.groupby([band, "period"], observed=True).died.agg(["mean", "size"]).reset_index()
    res["by_age"] = [{"age_band": _band(r.age), "period": r.period,
                      "death_rate": float(r["mean"]), "n": int(r["size"])} for _, r in g.iterrows()]
    m = df.groupby("month").died.agg(["mean", "size"])
    res["by_month"] = [{"month": k, "death_rate": float(v["mean"]), "n": int(v["size"])} for k, v in m.iterrows()]
    rows = []
    for f in [*config.COMORBIDITY_EN.values(), "pneumonia", "female", "pregnant"]:
        a = dev[dev[f] == 1].died
        b = dev[dev[f] == 0].died
        t = np.array([[a.sum(), len(a) - a.sum()], [b.sum(), len(b) - b.sum()]], float) + 0.5
        lor = np.log(t[0, 0] * t[1, 1] / (t[0, 1] * t[1, 0]))
        se = np.sqrt((1 / t).sum())
        rows.append({"feature": f, "prevalence": float(dev[f].mean()), "death_rate_with": float(a.mean()) if len(a) else float("nan"),
                     "death_rate_without": float(b.mean()), "crude_odds_ratio": float(np.exp(lor)),
                     "ci": (float(np.exp(lor - 1.96 * se)), float(np.exp(lor + 1.96 * se))),
                     "median_age_with": float(dev[dev[f] == 1].age.median()) if len(a) else float("nan")})
    res["conditions"] = rows
    st = dev.groupby("state").died.agg(["mean", "size"])
    res["by_state"] = [{"state": int(k), "death_rate": float(v["mean"]), "n": int(v["size"])} for k, v in st.iterrows()]
    res["delay_days"] = {"median": float(dev.days_symptoms_to_visit.median()),
                         "death_rate_by_delay": {str(k): float(v) for k, v in
                                                 dev.groupby(pd.cut(dev.days_symptoms_to_visit, [-1, 2, 5, 9, 31], labels=["0-2", "3-5", "6-9", "10-30"]),
                                                             observed=True).died.mean().items()}}
    res["n_conditions"] = {str(k): {"share": float(v["size"] / len(dev)), "death_rate": float(v["mean"])} for k, v in
                           dev.assign(nc=dev[list(config.COMORBIDITY_EN.values())].sum(axis=1).clip(upper=4)).groupby("nc").died.agg(["mean", "size"]).iterrows()}
    bands = pd.cut(dev.age, [0, 20, 40, 50, 60, 70, 80, 121], right=False)
    sa = dev.groupby([bands, "female"], observed=True).died.mean().unstack()
    res["by_sex_age"] = [{"age_band": _band(k), "male": float(v.get(0, np.nan)),
                          "female": float(v.get(1, np.nan))} for k, v in sa.iterrows()]
    sec = dev.groupby("sector").died.agg(["mean", "size"])
    res["by_sector"] = [{"sector": config.SECTORS.get(int(k), str(k)), "death_rate": float(v["mean"]), "n": int(v["size"])}
                        for k, v in sec.iterrows() if v["size"] >= 100]
    care = dev.groupby(bands, observed=True)[["hospitalised", "icu", "intubated", "died"]].mean()
    res["care_by_age"] = [{"age_band": _band(k), **{c: float(v[c]) for c in care.columns}}
                          for k, v in care.iterrows()]
    res["missing_records"] = {f: {"share": float(dev[f].mean()), "death_rate_if_missing": float(dev.loc[dev[f] == 1, "died"].mean()) if dev[f].any() else None,
                                  "death_rate_otherwise": float(dev.loc[dev[f] == 0, "died"].mean())}
                              for f in ("comorbidity_missing", "pneumonia_missing")}
    cols = [*config.COMORBIDITY_EN.values(), "pneumonia", "female"]
    phi = dev[cols].astype(float).corr()
    res["correlation"] = {"features": cols, "matrix": phi.round(4).values.tolist(),
                          "max_abs_offdiag": float(np.max(np.abs(phi.values - np.eye(len(cols)))))}
    res["age_summary"] = {"survived": dev.loc[dev.died == 0, "age"].describe().round(2).to_dict(),
                          "died": dev.loc[dev.died == 1, "age"].describe().round(2).to_dict()}
    (out / "eda.json").write_text(_json(res))
    return res


def benchmark_models(seed: int) -> dict:
    """A broad screen of model families (what LazyPredict does, but scored by cross-validation on
    training data only, with probability metrics). Default-ish settings; the two strongest
    families are tuned properly in the `select` stage."""
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
    from sklearn.dummy import DummyClassifier
    from sklearn.ensemble import AdaBoostClassifier, ExtraTreesClassifier, HistGradientBoostingClassifier, RandomForestClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.naive_bayes import BernoulliNB, GaussianNB
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.neural_network import MLPClassifier
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.tree import DecisionTreeClassifier
    m = {
        "Prior only (no information)": DummyClassifier(strategy="prior"),
        "Logistic regression": make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000)),
        "LDA": make_pipeline(StandardScaler(), LinearDiscriminantAnalysis()),
        "QDA (regularised)": make_pipeline(StandardScaler(), QuadraticDiscriminantAnalysis(reg_param=0.1)),
        "Gaussian naive Bayes": GaussianNB(),
        "Bernoulli naive Bayes": BernoulliNB(),
        "k-nearest neighbours (k=100)": make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=100)),
        "Decision tree (depth 8)": DecisionTreeClassifier(max_depth=8, min_samples_leaf=50, random_state=seed),
        "Random forest": RandomForestClassifier(n_estimators=200, min_samples_leaf=20, n_jobs=-1, random_state=seed),
        "Extra trees": ExtraTreesClassifier(n_estimators=200, min_samples_leaf=20, n_jobs=-1, random_state=seed),
        "AdaBoost": AdaBoostClassifier(n_estimators=200, random_state=seed),
        "Histogram gradient boosting": HistGradientBoostingClassifier(random_state=seed),
        "Neural network (MLP 32-16)": make_pipeline(StandardScaler(), MLPClassifier((32, 16), early_stopping=True, max_iter=200, random_state=seed)),
    }
    try:
        from xgboost import XGBClassifier
        m["XGBoost"] = XGBClassifier(n_estimators=300, learning_rate=0.1, max_depth=6, n_jobs=-1, random_state=seed, verbosity=0)
    except ImportError:
        pass
    try:
        from lightgbm import LGBMClassifier
        m["LightGBM"] = LGBMClassifier(n_estimators=300, learning_rate=0.05, random_state=seed, verbose=-1)
    except ImportError:
        pass
    return m


def stage_benchmark(dev, out: Path, seed: int, n_sample: int = config.BENCHMARK_SAMPLE) -> dict:
    sub = _subsample(dev, n_sample, seed + 11)
    fs = features.triage_features(sub)
    X, y = features.matrix(sub, fs), sub.died.to_numpy()
    folds = list(StratifiedKFold(config.CV_FOLDS, shuffle=True, random_state=seed + 1).split(X, y))
    rows = []
    for name, model in benchmark_models(seed).items():
        per, t0 = [], time.time()
        try:
            for tr, te in folds:
                from sklearn.base import clone
                p = clone(model).fit(X[tr], y[tr]).predict_proba(X[te])[:, 1]
                per.append({"roc_auc": float(roc_auc_score(y[te], p)), "pr_auc": float(average_precision_score(y[te], p)),
                            "brier": stats.brier(y[te], p)})
        except Exception as exc:  # noqa: BLE001 - a failing model is reported, not hidden
            rows.append({"model": name, "error": f"{type(exc).__name__}: {exc}"[:200]})
            continue
        row = {"model": name, "seconds": round(time.time() - t0, 1),
               **{k: float(np.mean([f[k] for f in per])) for k in per[0]}, **{k + "_sd": float(np.std([f[k] for f in per], ddof=1)) for k in per[0]}}
        rows.append(row)
        print(f"[benchmark] {name:32s} ROC-AUC {row['roc_auc']:.4f} PR-AUC {row['pr_auc']:.4f} ({row['seconds']:.0f}s)")
    ok = [r for r in rows if "error" not in r]
    best = max(ok, key=lambda r: r["pr_auc"])
    res = {"n_sample": int(len(sub)), "death_rate": float(y.mean()), "features": fs, "folds": config.CV_FOLDS,
           "models": sorted(rows, key=lambda r: -r.get("pr_auc", -1)), "best": best["model"]}
    (out / "benchmark.json").write_text(_json(res))
    return res


def rolling_recalibration(pred: pd.DataFrame, p_col: str) -> dict:
    """Monitoring as a hospital would do it: each month, re-estimate only the calibration intercept on the
    PREVIOUS month's outcomes and apply it to the current month (no refitting, no look-ahead)."""
    pred = pred.sort_values("month")
    months = sorted(pred.month.unique())
    p_all = np.clip(pred[p_col].to_numpy(float), 1e-6, 1 - 1e-6)
    y_all = pred.died.to_numpy()
    upd = p_all.copy()
    rows = []
    idx_by_month = {m: np.flatnonzero(pred.month.to_numpy() == m) for m in months}
    for k, m in enumerate(months):
        i = idx_by_month[m]
        if k > 0:
            prev = idx_by_month[months[k - 1]]
            shift = stats.calibration_intercept_slope(y_all[prev], p_all[prev])["intercept"]
            upd[i] = expit(logit(p_all[i]) + shift)
        else:
            shift = 0.0
        rows.append({"month": m, "n": int(len(i)), "observed": float(y_all[i].mean()), "predicted": float(p_all[i].mean()),
                     "updated": float(upd[i].mean()), "shift_applied": float(shift)})
    later = np.concatenate([idx_by_month[m] for m in months[1:]])
    yl = y_all[later]
    return {"months": rows,
            "before": {"ece": stats.expected_calibration_error(yl, p_all[later]), **stats.calibration_intercept_slope(yl, p_all[later])},
            "after": {"ece": stats.expected_calibration_error(yl, upd[later]), **stats.calibration_intercept_slope(yl, upd[later])},
            "roc_auc_unchanged": True}


def stage_monitoring(out: Path, art: Path, sel: dict) -> dict:
    full = art / "predictions" / "test_predictions.pkl"
    sample = out / "test_predictions_sample.csv.gz"
    if full.exists():
        pred, source = pd.read_pickle(full), "all 2021 test patients"
    elif sample.exists():
        pred, source = pd.read_csv(sample, dtype={"month": str}), f"the published stratified sample of 2021 predictions ({sample.name})"
    else:
        raise FileNotFoundError("no test predictions found; run the temporal stage first")
    res = {"source": source, "n": int(len(pred)), "model": sel["choice"]["model"],
           **rolling_recalibration(pred, f"p_{sel['choice']['model']}")}
    (out / "monitoring.json").write_text(_json(res))
    return res


# ---------------------------------------------------------------------- main

def run(cohort_path: Path, results: Path, art: Path, stages: list[str], redo: list[str], seed: int, n_boot: int,
        n_select: int = config.SELECTION_SAMPLE, n_iecv: int = config.IECV_SAMPLE, n_bench: int = config.BENCHMARK_SAMPLE) -> dict:
    results.mkdir(parents=True, exist_ok=True)
    art.mkdir(parents=True, exist_ok=True)
    for s in redo:
        (results / f"{s}.json").unlink(missing_ok=True)
    dev = test = None
    if any(s not in ("subgroups", "monitoring") and not (results / f"{s}.json").exists() for s in stages):
        df = pd.read_pickle(cohort_path)
        dev = df[df.period == "development"].reset_index(drop=True)
        test = df[df.period == "test"].reset_index(drop=True)
        del df
        print(f"development: {len(dev):,} patients ({dev.died.mean():.1%} died); test: {len(test):,} ({test.died.mean():.1%})")
    done = lambda s: (results / f"{s}.json").exists()  # noqa: E731
    load = lambda s: json.loads((results / f"{s}.json").read_text())  # noqa: E731
    out = {}
    for s in stages:
        if done(s):
            print(f"[{s}] already done, skipping")
            out[s] = load(s)
            continue
        t0 = time.time()
        sel = out.get("select") or (load("select") if done("select") else None)
        if s == "eda":
            out[s] = stage_eda(dev, test, results)
        elif s == "benchmark":
            out[s] = stage_benchmark(dev, results, seed, n_bench)
        elif s == "select":
            out[s] = stage_select(dev, results, seed, n_select)
        elif s == "temporal":
            out[s] = stage_temporal(dev, test, results, art, sel, seed, n_boot)
        elif s == "iecv":
            out[s] = stage_iecv(dev, results, sel, seed, n_iecv)
        elif s == "leakage":
            out[s] = stage_leakage(dev, test, results, art, sel, seed, n_boot)
        elif s == "subgroups":
            out[s] = stage_subgroups(results, art, sel)
        elif s == "explain":
            out[s] = stage_explain(dev, test, results, art, sel, seed)
        elif s == "monitoring":
            out[s] = stage_monitoring(results, art, sel)
        print(f"[{s}] finished in {time.time() - t0:.0f}s")
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cohort", type=Path, default=config.ARTIFACTS_DIR / "cohort.pkl")
    ap.add_argument("--results-dir", type=Path, default=config.RESULTS_DIR)
    ap.add_argument("--artifacts-dir", type=Path, default=config.ARTIFACTS_DIR)
    ap.add_argument("--stages", nargs="+", default=STAGES)
    ap.add_argument("--redo", nargs="*", default=[])
    ap.add_argument("--n-boot", type=int, default=config.N_BOOT)
    ap.add_argument("--quick", action="store_true", help="small samples and few bootstraps (pipeline check)")
    args = ap.parse_args(argv)
    kw = {"n_select": 20_000, "n_iecv": 20_000, "n_bench": 10_000} if args.quick else {}
    return run(args.cohort, args.results_dir, args.artifacts_dir, args.stages, args.redo, config.SEED,
               50 if args.quick else args.n_boot, **kw)


if __name__ == "__main__":
    main()
