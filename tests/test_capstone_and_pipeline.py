import functools
import json
import tempfile
import zipfile
from pathlib import Path

from covid_risk import capstone, pipeline
from synthetic import make_capstone_csv, make_official_zip


def test_capstone_facts_and_original_features(tmp_path):
    raw = capstone.load(make_capstone_csv(tmp_path / "cap.csv", n=2000))
    f = capstone.facts(raw)
    assert f["icu_identical_to_hospitalised"] and f["intubation_code3_equals_outpatient"]
    assert f["exact_duplicate_rows"] > 0
    assert f["after_dropping_duplicates"]["death_rate"] > f["death_rate"]          # de-duplication removes survivors
    assert f["death_rate_by_classification"][2]["rate"] == 1.0
    X, y = capstone.original_features(raw, dedup=True)
    assert {"ICU", "PATIENT_TYPE_ORDINAL", "INTUBED_ORDINAL", "CLASSIFICATION_FINAL"} <= set(X.columns)
    assert "AGE" not in X.columns
    cohort = capstone.to_cohort(raw)
    assert cohort.pneumonia_missing.sum() == (raw.PNEUMONIA == 99).sum()
    assert not ((cohort.pregnant == 1) & (cohort.female == 0)).any()


def test_capstone_audit_runs_end_to_end(tmp_path):
    path = make_capstone_csv(tmp_path / "cap.csv", n=2500)
    r = capstone.main(["--data", str(path), "--reports-dir", str(tmp_path), "--n-boot", "20"])
    ff = r["flaw_by_flaw"]
    assert ff["S0_original_as_published"]["death_rate"] > ff["S1_original_model_on_real_patients"]["death_rate"]
    assert "leakage_test" in ff and (tmp_path / "capstone_audit.md").exists()


@functools.lru_cache(maxsize=1)
def _run():
    w = Path(tempfile.mkdtemp())
    z20 = make_official_zip(w / "raw" / "COVID19MEXICO2020.zip", n=12000, seed=1, start="2020-03-01", end="2020-12-31", updated="2021-01-10")
    z21 = make_official_zip(w / "raw" / "COVID19MEXICO2021.zip", n=12000, seed=2, start="2020-06-01", end="2021-12-31", updated="2022-01-05")
    cap = make_capstone_csv(w / "cap.csv", n=2000)
    pipeline.main(["--workdir", str(w), "--files", str(z20), str(z21), "--capstone", str(cap), "--quick"])
    return w


def test_pipeline_produces_every_result_and_a_bundle():
    w = _run()
    for s in ("eda", "benchmark", "select", "temporal", "iecv", "leakage", "subgroups", "explain", "cohort_summary", "data_quality"):
        assert (w / "results" / f"{s}.json").exists(), s
    md = (w / "reports" / "results.md").read_text()
    for h in ("H1", "H2", "H3a", "H4", "H5", "H6"):
        assert f"| {h} |" in md
    names = zipfile.ZipFile(w / "results_bundle.zip").namelist()
    assert "reports/results.md" in names and "results/test_predictions_sample.csv.gz" in names
    assert "reports/figures/capstone_flaws.png" in names
    for fig in ("eda_age_month.png", "eda_conditions.png", "eda_states.png", "benchmark.png", "shap_summary.png", "shap_age.png"):
        assert f"reports/figures/{fig}" in names, fig
    for section in ("## Data quality", "## Exploratory analysis", "## Screening many model families", "## Explanations"):
        assert section in md, section


def test_benchmark_screens_many_models_and_explanations_add_up():
    w = _run()
    b = json.loads((w / "results" / "benchmark.json").read_text())
    ok = [m for m in b["models"] if "error" not in m]
    assert len(ok) >= 12
    prior = next(m for m in ok if m["model"].startswith("Prior"))
    assert all(m["roc_auc"] > prior["roc_auc"] for m in ok if m is not prior)       # every model beats no information
    e = json.loads((w / "results" / "explain.json").read_text())
    assert e["shap"]["model"] in ("logistic", "hgb") and e["shap"]["additivity_max_error"] < 1e-3
    q = json.loads((w / "results" / "data_quality.json").read_text())
    assert q["records_checked"] > 0 and {c["check"] for c in q["checks"]} >= {"male recorded as pregnant", "outpatient recorded in ICU"}


def test_pipeline_signal_and_leakage_direction():
    w = _run()
    t = json.loads((w / "results" / "temporal.json").read_text())
    assert t["logistic"]["roc_auc"] > t["age_only"]["roc_auc"] > 0.7          # synthetic outcome depends on age + comorbidities
    lk = json.loads((w / "results" / "leakage.json").read_text())
    assert lk["plus_all_post_triage"]["vs_triage_model"]["diff"] > 0


def test_pipeline_resumes_without_recomputing():
    w = _run()
    before = (w / "results" / "select.json").stat().st_mtime
    pipeline.main(["--workdir", str(w), "--stages", "select", "temporal", "--quick"])
    assert (w / "results" / "select.json").stat().st_mtime == before
