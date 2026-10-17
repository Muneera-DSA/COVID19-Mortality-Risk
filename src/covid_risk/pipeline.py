"""Run the whole study (designed for one Google Colab CPU session; resumable).

    python -m covid_risk.pipeline --workdir /content/drive/MyDrive/covid_risk --raw-dir /content/data

Stages (finished work in --workdir is skipped, so a disconnected session can simply rerun):
  data        download the 2020 and 2021 official files; build the cohort  (~10-15 min)
  audit       Part 1: audit of the original capstone notebook (needs data/covid_capstone.csv)
  select, temporal, iecv, leakage, subgroups, explain   (see covid_risk.experiments)
  report      reports/results.md and figures
  bundle      results_bundle.zip: reports, results JSON and a sample of test predictions
"""

from __future__ import annotations

import argparse
import shutil
import zipfile
from pathlib import Path

import pandas as pd

from . import capstone, config, experiments, official, report, stats

ALL = ["data", "audit", *experiments.STAGES, "report", "bundle"]


def bundle(workdir: Path, results: Path, reports: Path, art: Path, n_sample: int = 200_000) -> Path:
    pred = art / "predictions" / "test_predictions.pkl"
    if pred.exists():
        df = pd.read_pickle(pred)
        keep = stats.stratified_subsample(df.died.to_numpy(), n_sample, config.SEED)
        df.iloc[keep].drop(columns=["id"]).to_csv(results / "test_predictions_sample.csv.gz", index=False, float_format="%.5f")
    target = workdir / "results_bundle.zip"
    with zipfile.ZipFile(target, "w", zipfile.ZIP_DEFLATED) as z:
        for base, name in ((results, "results"), (reports, "reports")):
            for f in base.rglob("*"):
                if f.is_file() and f.suffix in {".json", ".md", ".png", ".gz"}:
                    z.write(f, Path(name) / f.relative_to(base))
    return target


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--workdir", type=Path, default=config.PROJECT_ROOT)
    ap.add_argument("--raw-dir", type=Path, default=None, help="where the official zips go (default <workdir>/data/official_raw)")
    ap.add_argument("--files", type=Path, nargs="*", help="official zip/csv files already on disk (skip download)")
    ap.add_argument("--capstone", type=Path, default=config.CAPSTONE_CSV)
    ap.add_argument("--stages", nargs="+", default=ALL)
    ap.add_argument("--redo", nargs="*", default=[])
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args(argv)

    w = args.workdir
    art, results, reports = w / "artifacts", w / "results", w / "reports"
    for d in (art, results, reports):
        d.mkdir(parents=True, exist_ok=True)
    cohort = art / "cohort.pkl"

    if "data" in args.stages:
        if "data" in args.redo:
            cohort.unlink(missing_ok=True)
        official.main(["--raw-dir", str(args.raw_dir or w / "data" / "official_raw"), "--out", str(cohort),
                       *(["--files", *map(str, args.files)] if args.files else [])])
        for name in ("cohort_summary.json", "data_quality.json"):
            if (art / name).exists():
                shutil.copy(art / name, results / name)
    if "audit" in args.stages:
        if args.capstone.exists() and (not (reports / "capstone_audit.json").exists() or "audit" in args.redo):
            capstone.main(["--data", str(args.capstone), "--reports-dir", str(reports)] + (["--n-boot", "50"] if args.quick else []))
        elif not args.capstone.exists():
            print(f"[audit] {args.capstone} not found; skipping Part 1")
    exp_stages = [s for s in args.stages if s in experiments.STAGES]
    if exp_stages:
        experiments.main(["--cohort", str(cohort), "--results-dir", str(results), "--artifacts-dir", str(art),
                          "--stages", *exp_stages, "--redo", *[s for s in args.redo if s in experiments.STAGES]]
                         + (["--quick"] if args.quick else []))
    if "report" in args.stages:
        out = report.build(results, reports, art)
        print(out["markdown"])
    if "bundle" in args.stages:
        print(f"Results bundle: {bundle(w, results, reports, art)}")


if __name__ == "__main__":
    main()
