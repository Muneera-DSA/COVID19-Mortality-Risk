"""reports/results.md and figures, generated only from results/*.json (no numbers typed by hand).

    python -m covid_risk.report --results-dir results --reports-dir reports
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.stats import norm

from . import config, stats

BLUE, ORANGE, GREY, GREEN = "#2a78d6", "#e8743b", "#8c8c8c", "#2f9e6b"
MODEL_LABEL = {"age_only": "Age only (baseline)", "logistic": "Logistic regression", "hgb": "Gradient boosting"}
FEATURE_LABEL = {"age": "Age", "female": "Female", "pregnant": "Pregnant", "diabetes": "Diabetes", "copd": "COPD",
                 "asthma": "Asthma", "immunosuppression": "Immunosuppression", "hypertension": "Hypertension",
                 "other_disease": "Other chronic disease", "cardiovascular": "Cardiovascular disease", "obesity": "Obesity",
                 "chronic_kidney": "Chronic kidney disease", "smoking": "Smoking", "pneumonia": "Pneumonia at attendance",
                 "pneumonia_missing": "Pneumonia not recorded", "comorbidity_missing": "A comorbidity not recorded",
                 "days_symptoms_to_visit": "Days from symptoms to attendance", "age (per 10 years)": "Age (per 10 years)"}


def _load(results: Path) -> dict:
    return {p.stem: json.loads(p.read_text()) for p in results.glob("*.json")}


def _p(p: float) -> str:
    return "<1e-300" if p == 0 else f"{p:.2g}"


def _ci(v, ci, fmt=".3f") -> str:
    return f"{v:{fmt}} ({ci[0]:{fmt}}–{ci[1]:{fmt}})"


# ------------------------------------------------------------------ hypotheses

def hypotheses(r: dict) -> list[dict]:
    H = []
    t, sel = r.get("temporal"), r.get("select")
    if t:
        a, b = t["logistic_vs_age_only"], t["hgb_vs_logistic"]
        H.append({"id": "H1", "text": "Triage features add to age alone (logistic vs age-only, 2021 test)",
                  "evidence": f"ROC-AUC {a['diff']:+.3f} (CI {a['ci'][0]:+.3f} to {a['ci'][1]:+.3f})", "p": a["p"], "ok": a["diff"] > 0})
        H.append({"id": "H2", "text": "Gradient boosting beats logistic regression (2021 test)",
                  "evidence": f"ROC-AUC {b['diff']:+.3f} (CI {b['ci'][0]:+.3f} to {b['ci'][1]:+.3f})", "p": b["p"], "ok": b["diff"] > 0})
        cal = t[t["choice"]["model"]]["calibration"]
        z = cal["intercept"] / cal["intercept_se"]
        H.append({"id": "H4", "text": "Calibration drifts from 2020 to 2021 (calibration-in-the-large ≠ 0)",
                  "evidence": f"intercept {cal['intercept']:+.3f}; predicted {cal['mean_predicted']:.1%} vs observed {cal['observed_rate']:.1%}",
                  "p": float(2 * norm.sf(abs(z))), "ok": True})
    if sel:
        m = sel["choice"]["model"]
        for k, label in (("class_weight", "a"), ("oversample", "b"), ("smote", "c")):
            tt = sel["tests"][f"{m}|{k}_vs_none"]
            H.append({"id": f"H3{label}", "text": f"{k.replace('_', ' ').capitalize()} improves PR-AUC ({MODEL_LABEL[m].lower()}, 5-fold CV)",
                      "evidence": f"PR-AUC {tt['diff_pr_auc']:+.4f}; calibration slope {1 + tt['diff_cal_slope']:.2f}",
                      "p": tt["p"], "ok": tt["diff_pr_auc"] > 0})
    if r.get("leakage"):
        lk = r["leakage"]["plus_all_post_triage"]["vs_triage_model"]
        H.append({"id": "H5", "text": "Post-triage variables (hospitalisation, ICU, intubation) inflate ROC-AUC",
                  "evidence": f"{lk['auc_a']:.3f} vs {lk['auc_b']:.3f} ({lk['diff']:+.3f})", "p": lk["p"], "ok": lk["diff"] > 0})
    if r.get("subgroups"):
        s = r["subgroups"]["sex_auc_test"]
        H.append({"id": "H6", "text": "Discrimination differs between women and men",
                  "evidence": f"ROC-AUC difference (F − M) {s['diff_female_minus_male']:+.3f}", "p": s["p"], "ok": True})
    adj = stats.holm({h["id"]: h["p"] for h in H})
    for h in H:
        h["p_holm"] = adj[h["id"]]
        h["verdict"] = "supported" if (h["p_holm"] < 0.05 and h["ok"]) else "not supported"
    return sorted(H, key=lambda h: h["id"])


# ---------------------------------------------------------------------- figures

def figures(r: dict, out: Path, artifacts: Path | None = None) -> list[str]:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False, "figure.dpi": 130})
    out.mkdir(parents=True, exist_ok=True)
    made = []
    t = r.get("temporal")
    if t:
        fig, ax = plt.subplots(figsize=(5.2, 4.6))
        ax.plot([0, 1], [0, 1], ls="--", color=GREY, lw=1, label="perfect calibration")
        for name, color in (("age_only", GREY), ("logistic", BLUE), ("hgb", ORANGE)):
            rel = t[name]["reliability"]
            ax.plot(rel["predicted"], rel["observed"], "o-", color=color, ms=4, label=MODEL_LABEL[name])
        top = max(max(t[n]["reliability"]["observed"] + t[n]["reliability"]["predicted"]) for n in ("logistic", "hgb")) * 1.05
        ax.set(xlim=(0, top), ylim=(0, top), xlabel="Predicted 30-day mortality", ylabel="Observed 30-day mortality",
               title="Calibration on 2021 patients (deciles)")
        ax.legend(frameon=False, fontsize=8)
        fig.tight_layout()
        fig.savefig(out / "calibration_2021.png")
        plt.close(fig)
        made.append("calibration_2021.png")

        mo = t["monthly"]
        fig, ax = plt.subplots(2, 1, figsize=(7, 5), sharex=True)
        x = range(len(mo))
        ax[0].plot(x, [m["observed"] for m in mo], "o-", color="black", ms=3, label="observed")
        ax[0].plot(x, [m["predicted"] for m in mo], "o-", color=ORANGE, ms=3, label="predicted (model fitted on 2020)")
        ax[0].set(ylabel="30-day mortality", title="Drift across 2021 waves")
        ax[0].legend(frameon=False, fontsize=8)
        ax[1].plot(x, [m["roc_auc"] for m in mo], "o-", color=BLUE, ms=3)
        ax[1].set(ylabel="ROC-AUC")
        ax[1].set_xticks(list(x), [m["month"] for m in mo], rotation=45, ha="right")
        fig.tight_layout()
        fig.savefig(out / "drift_2021.png")
        plt.close(fig)
        made.append("drift_2021.png")

        dc = t[t["choice"]["model"]]["decision_curve"]
        fig, ax = plt.subplots(figsize=(5.6, 4))
        ax.plot(dc["threshold"], dc["model"], color=ORANGE, label="use the model")
        ax.plot(dc["threshold"], dc["treat_all"], color=GREY, ls="--", label="prioritise everyone")
        ax.axhline(0, color="black", lw=0.8, label="prioritise no one")
        ax.set(ylim=(-0.02, max(dc["model"]) * 1.15), xlabel="Risk threshold for priority care", ylabel="Net benefit",
               title="Decision curve, 2021 patients")
        ax.legend(frameon=False, fontsize=8)
        fig.tight_layout()
        fig.savefig(out / "decision_curve.png")
        plt.close(fig)
        made.append("decision_curve.png")

    mon = r.get("monitoring")
    if mon:
        mo = mon["months"]
        fig, ax = plt.subplots(figsize=(7, 3.4))
        x = range(len(mo))
        ax.plot(x, [m["observed"] for m in mo], "o-", color="black", ms=3, label="observed")
        ax.plot(x, [m["predicted"] for m in mo], "o--", color=GREY, ms=3, label="2020 model, unchanged")
        ax.plot(x, [m["updated"] for m in mo], "o-", color=ORANGE, ms=3, label="with rolling monthly recalibration")
        ax.set_xticks(list(x), [m["month"] for m in mo], rotation=45, ha="right")
        ax.set(ylabel="30-day mortality", title="Keeping the model calibrated through 2021")
        ax.legend(frameon=False, fontsize=8)
        fig.tight_layout()
        fig.savefig(out / "monitoring.png")
        plt.close(fig)
        made.append("monitoring.png")

    iv = r.get("iecv")
    if iv:
        rows = sorted([s for s in iv["states"] if not s.get("skipped")], key=lambda s: s["roc_auc"])
        fig, ax = plt.subplots(figsize=(6, max(3, 0.22 * len(rows) + 1)))
        y = np.arange(len(rows))
        ax.errorbar([s["roc_auc"] for s in rows], y,
                    xerr=[[s["roc_auc"] - s["roc_auc_ci"][0] for s in rows], [s["roc_auc_ci"][1] - s["roc_auc"] for s in rows]],
                    fmt="o", color=BLUE, ms=3, lw=1)
        ax.axvline(iv["summary"]["median_auc"], color=ORANGE, ls="--", lw=1, label="median")
        ax.set_yticks(y, [config.STATES.get(s["state"], str(s["state"])) for s in rows], fontsize=7)
        ax.set(xlabel="ROC-AUC when the state is held out", title="Each state tested by a model that never saw it")
        ax.legend(frameon=False, fontsize=8)
        fig.tight_layout()
        fig.savefig(out / "states_iecv.png")
        plt.close(fig)
        made.append("states_iecv.png")

    eda = r.get("eda")
    if eda:
        fig, ax = plt.subplots(1, 2, figsize=(10, 3.8))
        for period, color in (("development", BLUE), ("test", ORANGE)):
            rows = [x for x in eda["by_age"] if x["period"] == period]
            ax[0].plot([x["age_band"] for x in rows], [x["death_rate"] for x in rows], "o-", ms=3, color=color,
                       label="2020" if period == "development" else "2021")
        ax[0].set(ylabel="30-day mortality", title="Mortality by age")
        ax[0].tick_params(axis="x", rotation=90, labelsize=7)
        ax[0].legend(frameon=False)
        mo = eda["by_month"]
        ax2 = ax[1].twinx()
        ax2.bar(range(len(mo)), [m["n"] for m in mo], color=GREY, alpha=0.35)
        ax2.set_ylabel("patients", color=GREY)
        ax[1].plot(range(len(mo)), [m["death_rate"] for m in mo], "o-", ms=3, color="black", zorder=3)
        ax[1].set_zorder(ax2.get_zorder() + 1)
        ax[1].patch.set_visible(False)
        ax[1].set_xticks(range(len(mo)), [m["month"] for m in mo], rotation=90, fontsize=7)
        ax[1].set(ylabel="30-day mortality", title="Patients and mortality by month of attendance")
        fig.tight_layout()
        fig.savefig(out / "eda_age_month.png")
        plt.close(fig)
        made.append("eda_age_month.png")
        cond = sorted(eda["conditions"], key=lambda c: c["crude_odds_ratio"])
        fig, ax = plt.subplots(figsize=(6.4, 4.6))
        y = np.arange(len(cond))
        ax.barh(y - 0.2, [c["death_rate_with"] for c in cond], 0.4, color=ORANGE, label="with the condition")
        ax.barh(y + 0.2, [c["death_rate_without"] for c in cond], 0.4, color=GREY, label="without")
        ax.set_yticks(y, [FEATURE_LABEL.get(c["feature"], c["feature"]) for c in cond], fontsize=8)
        ax.set(xlabel="30-day mortality (2020, unadjusted)", title="Mortality with and without each condition")
        ax.legend(frameon=False, fontsize=8)
        fig.tight_layout()
        fig.savefig(out / "eda_conditions.png")
        plt.close(fig)
        made.append("eda_conditions.png")

    if eda and "by_state" in eda:
        st = sorted(eda["by_state"], key=lambda x: x["death_rate"])
        fig, ax = plt.subplots(figsize=(6, 0.2 * len(st) + 1.2))
        ax.barh([config.STATES.get(x["state"], str(x["state"])) for x in st], [x["death_rate"] for x in st], color=BLUE)
        ax.set(xlabel="30-day mortality (2020)", title="Mortality by state of the health unit")
        ax.tick_params(axis="y", labelsize=7)
        fig.tight_layout()
        fig.savefig(out / "eda_states.png")
        plt.close(fig)
        made.append("eda_states.png")
    if eda and "correlation" in eda:
        c = eda["correlation"]
        fig, ax = plt.subplots(figsize=(6, 5.2))
        im = ax.imshow(np.array(c["matrix"]), cmap="coolwarm", vmin=-0.5, vmax=0.5)
        labels = [FEATURE_LABEL.get(f, f) for f in c["features"]]
        ax.set_xticks(range(len(labels)), labels, rotation=90, fontsize=7)
        ax.set_yticks(range(len(labels)), labels, fontsize=7)
        ax.set_title("Correlation between conditions (phi)")
        fig.colorbar(im, ax=ax, shrink=0.8)
        fig.tight_layout()
        fig.savefig(out / "eda_correlation.png")
        plt.close(fig)
        made.append("eda_correlation.png")
    if eda and "by_sex_age" in eda:
        rows = eda["by_sex_age"]
        fig, ax = plt.subplots(figsize=(5.6, 3.6))
        x = np.arange(len(rows))
        ax.bar(x - 0.2, [r_["male"] for r_ in rows], 0.4, color=BLUE, label="men")
        ax.bar(x + 0.2, [r_["female"] for r_ in rows], 0.4, color=ORANGE, label="women")
        ax.set_xticks(x, [r_["age_band"] for r_ in rows])
        ax.set(xlabel="Age", ylabel="30-day mortality", title="Mortality by sex within age bands (2020)")
        ax.legend(frameon=False)
        fig.tight_layout()
        fig.savefig(out / "eda_sex_age.png")
        plt.close(fig)
        made.append("eda_sex_age.png")

    bm = r.get("benchmark")
    if bm:
        rows = [m for m in bm["models"] if "error" not in m][::-1]
        fig, ax = plt.subplots(figsize=(6.4, 0.32 * len(rows) + 1.2))
        ax.barh([m["model"] for m in rows], [m["pr_auc"] for m in rows], xerr=[m["pr_auc_sd"] for m in rows], color=BLUE)
        ax.set(xlabel="PR-AUC, 5-fold CV on 2020 data (mean ± SD)", title="Screening of model families")
        ax.tick_params(axis="y", labelsize=8)
        fig.tight_layout()
        fig.savefig(out / "benchmark.png")
        plt.close(fig)
        made.append("benchmark.png")

    ex = r.get("explain")
    if ex:
        sh = ex["shap"]
        imp = sorted(sh["mean_abs"].items(), key=lambda kv: kv[1])
        fig, ax = plt.subplots(figsize=(5.6, 4.6))
        ax.barh([FEATURE_LABEL.get(k, k) for k, _ in imp], [v for _, v in imp], color=ORANGE)
        is_shap = "permutation" not in sh["method"]
        ax.set(xlabel="Mean |SHAP value| (log-odds)" if is_shap else "Permutation importance (ROC-AUC drop)",
               title=f"What drives the {MODEL_LABEL.get(sh.get('model', 'hgb'), '').lower()}")
        fig.tight_layout()
        fig.savefig(out / "shap_importance.png")
        plt.close(fig)
        made.append("shap_importance.png")
        npz = (artifacts / f"shap_{sh.get('model', 'hgb')}.npz") if artifacts else None
        if npz is not None and npz.exists():
            d = np.load(npz, allow_pickle=False)
            sv, X, names = d["shap"], d["X"], [str(n) for n in d["features"]]
            order = np.argsort(np.abs(sv).mean(axis=0))[::-1][:10][::-1]
            rng = np.random.default_rng(0)
            fig, ax = plt.subplots(figsize=(6.4, 4.8))
            for row, j in enumerate(order):
                x = X[:, j].astype(float)
                lo, hi = np.nanpercentile(x, 1), np.nanpercentile(x, 99)
                c = np.clip((x - lo) / (hi - lo if hi > lo else 1), 0, 1)
                ax.scatter(sv[:, j], row + rng.normal(0, 0.12, len(x)), c=c, cmap="coolwarm", s=4, alpha=0.6, linewidths=0)
            ax.axvline(0, color=GREY, lw=0.8)
            ax.set_yticks(range(len(order)), [FEATURE_LABEL.get(names[j], names[j]) for j in order], fontsize=8)
            ax.set(xlabel="SHAP value (effect on log-odds of death)", title="SHAP summary, 2021 patients (red = high value)")
            fig.tight_layout()
            fig.savefig(out / "shap_summary.png")
            plt.close(fig)
            made.append("shap_summary.png")
            if "age" in names:
                j = names.index("age")
                fig, ax = plt.subplots(figsize=(5.2, 3.6))
                ax.scatter(X[:, j], sv[:, j], s=4, alpha=0.5, color=BLUE, linewidths=0)
                ax.set(xlabel="Age", ylabel="SHAP value for age", title="How age changes the predicted risk")
                fig.tight_layout()
                fig.savefig(out / "shap_age.png")
                plt.close(fig)
                made.append("shap_age.png")

    if ex:
        ors = sorted([o for o in ex["odds_ratios"]["rows"] if np.isfinite(o["ci"][0]) and np.isfinite(o["ci"][1])], key=lambda o: o["odds_ratio"])
        fig, ax = plt.subplots(figsize=(5.6, 4.8))
        y = np.arange(len(ors))
        ax.errorbar([o["odds_ratio"] for o in ors], y, xerr=[[o["odds_ratio"] - o["ci"][0] for o in ors], [o["ci"][1] - o["odds_ratio"] for o in ors]],
                    fmt="o", color=BLUE, ms=3, lw=1)
        ax.axvline(1, color=GREY, ls="--", lw=1)
        ax.set_xscale("log")
        lo, hi = min(o["ci"][0] for o in ors), max(o["ci"][1] for o in ors)
        ticks = [t for t in (0.25, 0.5, 1, 2, 4, 8, 16, 32) if lo / 1.5 <= t <= hi * 1.5]
        ax.set_xticks(ticks, [f"{t:g}" for t in ticks])
        ax.minorticks_off()
        ax.set_yticks(y, [FEATURE_LABEL.get(o["feature"], o["feature"]) for o in ors], fontsize=8)
        ax.set(xlabel="Adjusted odds ratio of 30-day death (log scale)", title="Risk factors, mutually adjusted (2020)")
        fig.tight_layout()
        fig.savefig(out / "odds_ratios.png")
        plt.close(fig)
        made.append("odds_ratios.png")
    return made


def capstone_figure(audit: dict, out: Path) -> str | None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    ff = audit["flaw_by_flaw"]
    steps = [("S0", "S0_original_as_published"), ("S1", "S1_original_model_on_real_patients"), ("S2", "S2_keep_all_patients"),
             ("S3", "S3_remove_post_triage_leakage"), ("S4", "S4_no_smote"), ("S4b", "S4b_remove_outcome_defined_code"),
             ("S5", "S5_correct_coding_and_age")]
    steps = [(s, k) for s, k in steps if k in ff]
    fig, ax = plt.subplots(1, 2, figsize=(9, 3.4))
    x = np.arange(len(steps))
    ax[0].bar(x, [ff[k]["pr_auc"] for _, k in steps], color=[GREY] + [BLUE] * (len(steps) - 1))
    ax[0].set(ylabel="PR-AUC", title="Ranking of deaths", ylim=(0, 1))
    ax[1].bar(x - 0.2, [ff[k]["mean_predicted_risk"] for _, k in steps], 0.4, color=ORANGE, label="mean predicted risk")
    ax[1].bar(x + 0.2, [ff[k]["death_rate"] for _, k in steps], 0.4, color="black", label="observed death rate")
    ax[1].set(title="Calibration", ylim=(0, 0.6))
    ax[1].legend(frameon=False, fontsize=8)
    for a in ax:
        a.set_xticks(x, [s for s, _ in steps])
        a.spines[["top", "right"]].set_visible(False)
    fig.suptitle("Original notebook (S0) and the effect of each correction (S1–S5)", fontsize=10)
    fig.tight_layout()
    out.mkdir(parents=True, exist_ok=True)
    fig.savefig(out / "capstone_flaws.png", dpi=130)
    plt.close(fig)
    return "capstone_flaws.png"


# ------------------------------------------------------------------- markdown

def to_markdown(r: dict, H: list[dict], cohort: dict | None) -> str:
    L = ["# Results (generated by `python -m covid_risk.report`; do not edit by hand)", ""]
    if cohort:
        L += ["## Cohort", "", "| Step | Patients |", "|---|---|"]
        L += [f"| {s['step']} | {s['n']:,} |" for s in cohort["flow"]]
        L += ["", "| | Development (2020) | Test (2021) |", "|---|---|---|"]
        d, t = cohort["periods"].get("development"), cohort["periods"].get("test")
        if d and t:
            L += [f"| Attended | {d['attended_from']} to {d['attended_to']} | {t['attended_from']} to {t['attended_to']} |",
                  f"| Patients | {d['patients']:,} | {t['patients']:,} |",
                  f"| 30-day deaths | {d['deaths_30d']:,} ({d['death_rate_30d']:.1%}) | {t['deaths_30d']:,} ({t['death_rate_30d']:.1%}) |",
                  f"| Age, median (IQR) | {d['age_median']:.0f} ({d['age_iqr'][0]:.0f}–{d['age_iqr'][1]:.0f}) | "
                  f"{t['age_median']:.0f} ({t['age_iqr'][0]:.0f}–{t['age_iqr'][1]:.0f}) |",
                  f"| Female | {d['female']:.1%} | {t['female']:.1%} |", f"| Pneumonia at attendance | {d['pneumonia']:.1%} | {t['pneumonia']:.1%} |",
                  f"| Hospitalised | {d['hospitalised']:.1%} | {t['hospitalised']:.1%} |", f"| States | {d['states']} | {t['states']} |"]
            L += [f"| {k.replace('_', ' ').capitalize()} | {d['prevalence'][k]:.1%} | {t['prevalence'][k]:.1%} |" for k in d["prevalence"]]
    dq = r.get("data_quality")
    if dq:
        L += ["", "## Data quality (raw confirmed records, before cleaning)", "",
              f"{dq['records_checked']:,} confirmed records checked. {dq['note']}", "",
              "| Field | Records in scope | Missing / unknown | Share | Codes outside the catalogue |", "|---|---|---|---|---|"]
        L += [f"| {c['field']} | {c['scope']:,} | {c['missing']:,} | {c['share_missing']:.2%} | {c['outside_catalogue']:,} |" for c in dq["completeness"]]
        L += ["", "| Consistency / plausibility check | Records | Share |", "|---|---|---|"]
        L += [f"| {c['check']} | {c['records']:,} | {c['share']:.3%} |" for c in dq["checks"]]
        L += ["", "Shift between 2020 (development) and 2021 (test): standardised mean difference (|SMD| > 0.1 is a notable shift).", "",
              "| Feature | 2020 | 2021 | SMD |", "|---|---|---|---|"]
        L += [f"| {FEATURE_LABEL.get(d['feature'], d['feature'])} | {d['development']:.3f} | {d['test']:.3f} | {d['smd']:+.3f} |"
              for d in dq["drift_2020_to_2021"]]
    eda = r.get("eda")
    if eda:
        L += ["", "## Exploratory analysis (2020 development data unless stated)", "",
              "![Mortality by age and month](figures/eda_age_month.png)", "", "![Conditions](figures/eda_conditions.png)", "",
              "| Condition | Prevalence | Mortality with | Mortality without | Crude odds ratio (95% CI) | Median age with |", "|---|---|---|---|---|---|"]
        for c in sorted(eda["conditions"], key=lambda c: -c["crude_odds_ratio"]):
            L.append(f"| {FEATURE_LABEL.get(c['feature'], c['feature'])} | {c['prevalence']:.1%} | {c['death_rate_with']:.1%} | {c['death_rate_without']:.1%} | "
                     f"{_ci(c['crude_odds_ratio'], c['ci'], '.2f')} | {c['median_age_with']:.0f} |")
        L += ["", "Crude odds ratios are confounded by age (most conditions are more common in older patients); adjusted odds ratios are in the Explanations section.",
              "", "| Number of chronic conditions | Share of patients | 30-day mortality |", "|---|---|---|"]
        L += [f"| {k if k != '4' else '4+'} | {v['share']:.1%} | {v['death_rate']:.1%} |" for k, v in eda["n_conditions"].items()]
        if "by_sex_age" in eda:
            L += ["", "![Sex and age](figures/eda_sex_age.png)", "", "| Age | Men | Women |", "|---|---|---|"]
            L += [f"| {x['age_band']} | {x['male']:.1%} | {x['female']:.1%} |" for x in eda["by_sex_age"]]
            L += ["", "| Age | Hospitalised | ICU | Intubated | 30-day mortality |", "|---|---|---|---|---|"]
            L += [f"| {x['age_band']} | {x['hospitalised']:.1%} | {x['icu']:.1%} | {x['intubated']:.1%} | {x['died']:.1%} |" for x in eda["care_by_age"]]
            L += ["", "| Health-care sector | Patients | 30-day mortality |", "|---|---|---|"]
            L += [f"| {x['sector']} | {x['n']:,} | {x['death_rate']:.1%} |" for x in sorted(eda["by_sector"], key=lambda x: -x["n"])]
            L += ["", f"Days from symptoms to attendance: median {eda['delay_days']['median']:.0f}; mortality by delay: "
                  + ", ".join(f"{k} days {v:.1%}" for k, v in eda["delay_days"]["death_rate_by_delay"].items()) + "."]
            mr = eda["missing_records"]
            L += ["", "Records with missing information: " + "; ".join(
                f"{FEATURE_LABEL.get(k, k)}: {v['share']:.1%} of patients, mortality {v['death_rate_if_missing']:.1%} vs {v['death_rate_otherwise']:.1%}"
                for k, v in mr.items() if v["death_rate_if_missing"] is not None) + ". Missingness is kept as its own feature rather than imputed."]
            L += ["", "![States](figures/eda_states.png)", "", "![Correlation](figures/eda_correlation.png)", "",
                  f"Largest correlation between two conditions: |phi| = {eda['correlation']['max_abs_offdiag']:.2f} (no problematic overlap between predictors)."
                  if eda["correlation"]["max_abs_offdiag"] < 0.7 else
                  f"Largest correlation between two conditions: |phi| = {eda['correlation']['max_abs_offdiag']:.2f}: strongly overlapping predictors."]
    bm = r.get("benchmark")
    if bm:
        L += ["", "## Screening many model families (5-fold CV, development data only)", "",
              f"Sample of {bm['n_sample']:,} patients; near-default settings. This replaces LazyPredict, which ranks models on the test set. "
              "The two strongest families are then tuned in the next step.", "",
              "![Benchmark](figures/benchmark.png)", "", "| Model | ROC-AUC | PR-AUC | Brier | Fit + predict time (5 folds) |", "|---|---|---|---|---|"]
        for m in bm["models"]:
            if "error" in m:
                L.append(f"| {m['model']} | failed: {m['error']} | | | |")
            else:
                L.append(f"| {m['model']} | {m['roc_auc']:.4f} ± {m['roc_auc_sd']:.4f} | {m['pr_auc']:.4f} ± {m['pr_auc_sd']:.4f} | {m['brier']:.4f} | {m['seconds']:.0f}s |")
    sel = r.get("select")
    if sel:
        L += ["", "## Model selection (development data, 5-fold cross-validation)", "",
              f"Stratified sample of {sel['n_sample']:,} patients (death rate {sel['death_rate']:.1%}). Tuned on 3 folds by PR-AUC: "
              + "; ".join(f"{MODEL_LABEL[k]} {v['best']}" for k, v in sel["tuning"].items()) + ".", "",
              "| Model | Imbalance handling | ROC-AUC | PR-AUC | Brier | Calibration intercept | Calibration slope |", "|---|---|---|---|---|---|---|"]
        for key, v in sel["cv"].items():
            m, imb = key.split("|")
            L.append(f"| {MODEL_LABEL[m]} | {imb.replace('_', ' ')} | {v['roc_auc']:.4f} ± {v['roc_auc_sd']:.4f} | {v['pr_auc']:.4f} ± {v['pr_auc_sd']:.4f} | "
                     f"{v['brier']:.4f} | {v['cal_intercept']:+.2f} | {v['cal_slope']:.2f} |")
        c = sel["choice"]
        L += ["", f"**Chosen:** {MODEL_LABEL[c['model']]}, imbalance handling: {c['imbalance']}, parameters {c['params']}. Rule: {c['rule']}."]
    t = r.get("temporal")
    if t:
        L += ["", "## Temporal validation: fitted on 2020, tested on 2021", "",
              f"Development: {t['dev']['n']:,} patients. Test: {t['hgb']['n']:,} patients, {t['hgb']['deaths']:,} deaths ({t['hgb']['death_rate']:.1%}). "
              "ROC-AUC CI: DeLong; PR-AUC CI: bootstrap.", "",
              "| Model | ROC-AUC | PR-AUC | Brier | ECE | Calibration intercept | Calibration slope |", "|---|---|---|---|---|---|---|"]
        for name in ("age_only", "logistic", "hgb"):
            s = t[name]
            L.append(f"| {MODEL_LABEL[name]} | {_ci(s['roc_auc'], s['roc_auc_ci'])} | {_ci(s['pr_auc'], s['pr_auc_ci'])} | {s['brier']:.4f} | "
                     f"{s['ece']:.4f} | {s['calibration']['intercept']:+.3f} | {s['calibration']['slope']:.3f} |")
        b = t["hgb_vs_logistic_pr_auc"]
        L += ["", f"Gradient boosting vs logistic, PR-AUC: {b['value']:+.4f} (CI {b['ci'][0]:+.4f} to {b['ci'][1]:+.4f})."]
        ch = t["choice"]["model"]
        s = t[ch]
        L += ["", f"### Using the {MODEL_LABEL[ch].lower()} to prioritise patients (2021)", "",
              "| Patients flagged (highest risk first) | Share of all deaths among them | Death rate among flagged | Risk cut-off |", "|---|---|---|---|"]
        L += [f"| top {c['share_flagged']:.0%} | {c['deaths_captured']:.1%} | {c['death_rate_in_flagged']:.1%} | {c['risk_cutoff']:.1%} |" for c in s["capacity"]]
        nb = s["net_benefit"]
        L += ["", "| Risk threshold | Net benefit: model | Net benefit: prioritise everyone |", "|---|---|---|"]
        L += [f"| {th:.0%} | {m:.4f} | {a:.4f} |" for th, m, a in zip(nb["threshold"], nb["model"], nb["treat_all"])]
        if "recalibration" in t:
            rc = t["recalibration"]
            L += ["", f"**Recalibration:** re-estimating only the intercept on {rc['fitted_on']} (shift {rc['intercept_shift']:+.3f}) changes "
                  f"calibration-in-the-large on later months from {rc['before']['intercept']:+.3f} to {rc['after']['intercept']:+.3f} "
                  f"(ECE {rc['ece_before']:.4f} → {rc['ece_after']:.4f})."]
        L += ["", "| Month | Patients | Observed | Predicted | ROC-AUC |", "|---|---|---|---|---|"]
        L += [f"| {m['month']} | {m['n']:,} | {m['observed']:.1%} | {m['predicted']:.1%} | {m['roc_auc']:.3f} |" for m in t["monthly"]]
    iv = r.get("iecv")
    if iv:
        s = iv["summary"]
        L += ["", "## Geographic validation: each state held out in turn (development data)", "",
              f"Sample of {iv['n_sample']:,} patients; {s['n_states']} states evaluated. ROC-AUC median {s['median_auc']:.3f} "
              f"(IQR {s['iqr_auc'][0]:.3f}–{s['iqr_auc'][1]:.3f}; range {s['min_auc']:.3f}–{s['max_auc']:.3f}). Calibration slope median "
              f"{s['median_cal_slope']:.2f}; calibration intercept range {s['range_cal_intercept'][0]:+.2f} to {s['range_cal_intercept'][1]:+.2f}.", "",
              "| State | Patients | Observed | Predicted | ROC-AUC (95% CI) | Calibration intercept |", "|---|---|---|---|---|---|"]
        for st in sorted(iv["states"], key=lambda x: x["state"]):
            name = config.STATES.get(st["state"], str(st["state"]))
            if st.get("skipped"):
                L.append(f"| {name} | {st['n']:,} | – | – | too few deaths | – |")
            else:
                L.append(f"| {name} | {st['n']:,} | {st['observed']:.1%} | {st['predicted']:.1%} | {_ci(st['roc_auc'], st['roc_auc_ci'])} | {st['cal_intercept']:+.2f} |")
    mon = r.get("monitoring")
    if mon:
        bf, af = mon["before"], mon["after"]
        L += ["", "## Monitoring: rolling monthly recalibration (2021)", "",
              f"Each month, only the calibration intercept is re-estimated on the previous month's outcomes and applied to the current month "
              f"(no refitting, no look-ahead). Computed on {mon['source']} ({mon['n']:,} patients). Ranking (ROC-AUC) is unchanged by construction.", "",
              f"Months 2-{len(mon['months'])}: calibration-in-the-large {bf['intercept']:+.3f} → {af['intercept']:+.3f}; "
              f"ECE {bf['ece']:.4f} → {af['ece']:.4f}; mean predicted {bf['mean_predicted']:.2%} → {af['mean_predicted']:.2%} (observed {bf['observed_rate']:.2%}).", "",
              "![Monitoring](figures/monitoring.png)", "",
              "| Month | Observed | Predicted (2020 model) | Predicted after rolling update |", "|---|---|---|---|"]
        L += [f"| {m['month']} | {m['observed']:.1%} | {m['predicted']:.1%} | {m['updated']:.1%} |" for m in mon["months"]]
    lk = r.get("leakage")
    if lk:
        L += ["", "## Leakage check (2021 test)", "", "| Model | ROC-AUC | PR-AUC | ROC-AUC change vs triage model |", "|---|---|---|---|"]
        base = lk["plus_hospitalised"]["vs_triage_model"]["auc_b"]
        L.append(f"| Triage features only (deployable) | {base:.4f} | – | – |")
        for k, label in (("plus_hospitalised", "+ hospitalised"), ("plus_all_post_triage", "+ hospitalised, ICU, intubated")):
            v = lk[k]
            L.append(f"| {label} | {v['roc_auc']:.4f} | {v['pr_auc']:.4f} | {v['vs_triage_model']['diff']:+.4f} (p {_p(v['vs_triage_model']['p'])}) |")
    sg = r.get("subgroups")
    if sg:
        L += ["", "## Subgroups (2021 test)", "", "| Group | Patients | Deaths | Observed | Predicted | ROC-AUC (95% CI) |", "|---|---|---|---|---|---|"]
        for gname in ("sex", "age", "sector", "pneumonia"):
            for row in sg[gname]:
                auc = _ci(row["roc_auc"], row["roc_auc_ci"]) if "roc_auc" in row else "too few deaths"
                L.append(f"| {gname}: {row['group']} | {row['n']:,} | {row['deaths']:,} | {row['observed']:.1%} | {row['predicted']:.1%} | {auc} |")
    ex = r.get("explain")
    if ex:
        sh = ex["shap"]
        add = f"; additivity check max error {sh['additivity_max_error']:.1e}" if "additivity_max_error" in sh else ""
        head = "Mean \\|SHAP\\| (log-odds)" if "permutation" not in sh["method"] else "Permutation importance (ROC-AUC drop)"
        L += ["", "## Explanations", "", f"Model explained: {MODEL_LABEL.get(sh.get('model', 'hgb'), '')} (the chosen model). Method: {sh['method']}{add}.", "",
              f"| Feature | {head} |", "|---|---|"]
        L += [f"| {FEATURE_LABEL.get(k, k)} | {v:.4f} |" for k, v in sorted(sh["mean_abs"].items(), key=lambda kv: -kv[1])]
        if sh.get("shap_library_check_max_diff") is not None:
            L += ["", f"Cross-check against the `shap` library (LinearExplainer): max difference {sh['shap_library_check_max_diff']:.1e}."]
        L += ["", "![SHAP summary](figures/shap_summary.png)", "", "![SHAP age](figures/shap_age.png)"]
        if "shap_hgb" in ex:
            g = ex["shap_hgb"]
            L += ["", f"Gradient boosting, for comparison ({g['method']}): " + ", ".join(
                f"{FEATURE_LABEL.get(k, k)} {v:.3f}" for k, v in sorted(g["mean_abs"].items(), key=lambda kv: -kv[1])[:6]) + "."]
        L += ["", f"Adjusted odds ratios (unpenalised logistic regression, {ex['odds_ratios']['n']:,} development patients; Holm-adjusted p):", "",
              "| Factor | Odds ratio (95% CI) | p (Holm) |", "|---|---|---|"]
        est = [o for o in ex["odds_ratios"]["rows"] if np.isfinite(o["ci"][0]) and np.isfinite(o["ci"][1])]
        L += [f"| {FEATURE_LABEL.get(o['feature'], o['feature'])} | {_ci(o['odds_ratio'], o['ci'], '.2f')} | {_p(o['p_holm'])} |"
              for o in sorted(est, key=lambda o: -o["odds_ratio"])]
        L += [f"| {FEATURE_LABEL.get(o['feature'], o['feature'])} | not estimable (no such patients in the sample) | – |"
              for o in ex["odds_ratios"]["rows"] if o not in est]
    if H:
        L += ["", "## Hypothesis tests (Holm-adjusted across this table)", "", "| # | Hypothesis | Evidence | p (Holm) | Verdict |", "|---|---|---|---|---|"]
        L_note = ["", "With millions of patients, very small differences reach statistical significance. Judge each supported hypothesis by the size of "
                  "its effect in the Evidence column, not by the p-value alone."]
        L += [f"| {h['id']} | {h['text']} | {h['evidence']} | {_p(h['p_holm'])} | {h['verdict']} |" for h in H]
        L += L_note
    return "\n".join(L) + "\n"


def build(results: Path, reports: Path, artifacts: Path | None = None) -> dict:
    r = _load(results)
    cohort = None
    for cand in [results / "cohort_summary.json", *([artifacts / "cohort_summary.json"] if artifacts else [])]:
        if cand.exists():
            cohort = json.loads(cand.read_text())
            break
    r.pop("cohort_summary", None)
    H = hypotheses(r)
    figs = figures(r, reports / "figures", artifacts)
    audit = reports / "capstone_audit.json"
    if audit.exists():
        figs.append(capstone_figure(json.loads(audit.read_text()), reports / "figures"))
    md = to_markdown(r, H, cohort)
    reports.mkdir(parents=True, exist_ok=True)
    (reports / "results.md").write_text(md)
    (reports / "results.json").write_text(json.dumps({"hypotheses": H, "figures": figs}, indent=2))
    return {"hypotheses": H, "figures": figs, "markdown": md}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--results-dir", type=Path, default=config.RESULTS_DIR)
    ap.add_argument("--reports-dir", type=Path, default=config.REPORTS_DIR)
    ap.add_argument("--artifacts-dir", type=Path, default=config.ARTIFACTS_DIR)
    args = ap.parse_args(argv)
    out = build(args.results_dir, args.reports_dir, args.artifacts_dir)
    print(out["markdown"])


if __name__ == "__main__":
    main()
