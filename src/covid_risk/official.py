"""Download the official Mexican COVID-19 line list and build the study cohort.

    python -m covid_risk.official --raw-dir data/official_raw --out artifacts/cohort.pkl

Cohort rules (each step's row count is reported, TRIPOD-style):
  1. records from all yearly files; a record appearing in several files keeps its most recent
     version (latest FECHA_ACTUALIZACION: most complete outcome)
  2. confirmed COVID-19: CLASIFICACION_FINAL 1 (clinical-epidemiological) or 3 (laboratory/antigen).
     Code 2 ("confirmed by committee") is excluded: it is assigned only to deaths.
  3. valid age (0-110), sex and attendance date
  4. attended in the development or test period, with >= 30 days of follow-up before the file date
  5. outcome: death within 30 days of attendance; deaths dated before attendance are dropped
     as recording errors
"""

from __future__ import annotations

import argparse
import gzip
import json
import shutil
import time
import urllib.request
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

from . import config, features

CHUNK = 1_000_000


# ---------------------------------------------------------------------- download

def download(year: int, raw_dir: Path, url: str | None = None, retries: int = 3) -> Path:
    raw_dir.mkdir(parents=True, exist_ok=True)
    url = url or config.OFFICIAL_URLS[year]
    target = raw_dir / Path(url).name
    if target.exists() and zipfile.is_zipfile(target):
        print(f"{target.name}: already downloaded ({target.stat().st_size / 1e6:.0f} MB)")
        return target
    for attempt in range(1, retries + 1):
        try:
            print(f"Downloading {url} (attempt {attempt}) ...")
            tmp = target.with_suffix(".part")
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 (research download)"})
            with urllib.request.urlopen(req, timeout=120) as r, open(tmp, "wb") as f:
                shutil.copyfileobj(r, f, length=1 << 22)
            if not zipfile.is_zipfile(tmp):
                raise OSError("downloaded file is not a zip archive")
            tmp.rename(target)
            print(f"  saved {target} ({target.stat().st_size / 1e6:.0f} MB)")
            return target
        except Exception as exc:  # noqa: BLE001 - reported to the user with a manual fallback
            print(f"  failed: {type(exc).__name__}: {exc}")
            time.sleep(5 * attempt)
    raise SystemExit(f"Could not download {url}.\nDownload it manually from {config.OFFICIAL_PORTAL} "
                     f"and put the zip file in {raw_dir}.")


# -------------------------------------------------------------------------- read

QUALITY_FIELDS = ["SEXO", "EDAD", "EMBARAZO", *config.COMORBIDITIES, "NEUMONIA", "TIPO_PACIENTE", "UCI", "INTUBADO",
                  "FECHA_INGRESO", "FECHA_SINTOMAS", "ENTIDAD_UM", "SECTOR"]
CATALOGUE = {"SEXO": {1, 2, 99}, "TIPO_PACIENTE": {1, 2, 99}, "NEUMONIA": {1, 2, 97, 98, 99}, "UCI": {1, 2, 97, 98, 99},
             "INTUBADO": {1, 2, 97, 98, 99}, "EMBARAZO": {1, 2, 97, 98, 99}, **{c: {1, 2, 97, 98, 99} for c in config.COMORBIDITIES}}


def quality_counts(c: pd.DataFrame) -> dict[str, int]:
    """Data-quality counters on the raw codes of confirmed records (codes 1-3), before any cleaning."""
    c = c[c.CLASIFICACION_FINAL.isin(["1", "2", "3"])]
    num = {k: pd.to_numeric(c[k], errors="coerce") for k in QUALITY_FIELDS if k in c and not k.startswith("FECHA")}
    dates = {k: pd.to_datetime(c[k], format="%Y-%m-%d", errors="coerce") for k in ("FECHA_INGRESO", "FECHA_SINTOMAS", "FECHA_ACTUALIZACION")}
    female, hosp, outp = num["SEXO"] == 1, num["TIPO_PACIENTE"] == 2, num["TIPO_PACIENTE"] == 1
    q = {"records": len(c)}
    for k, v in num.items():
        if k == "EDAD":
            q["missing|EDAD"] = int((v.isna() | (v < 0) | (v > 120)).sum())
            continue
        scope = female if k == "EMBARAZO" else hosp if k in ("UCI", "INTUBADO") else pd.Series(True, index=c.index)
        q[f"scope|{k}"] = int(scope.sum())
        q[f"missing|{k}"] = int((scope & (v.isna() | v.isin([98, 99]) | ((v == 97) & (k not in ("SEXO", "TIPO_PACIENTE"))))).sum())
        if k in CATALOGUE:
            q[f"outside_catalogue|{k}"] = int((v.notna() & ~v.isin(list(CATALOGUE[k]))).sum())
    for k in ("FECHA_INGRESO", "FECHA_SINTOMAS"):
        q[f"missing|{k}"] = int(dates[k].isna().sum())
    q["inconsistent|outpatient recorded in ICU"] = int((outp & (num["UCI"] == 1)).sum())
    q["inconsistent|outpatient recorded as intubated"] = int((outp & (num["INTUBADO"] == 1)).sum())
    q["inconsistent|male recorded as pregnant"] = int(((num["SEXO"] == 2) & (num["EMBARAZO"] == 1)).sum())
    q["inconsistent|symptoms after attendance"] = int((dates["FECHA_SINTOMAS"] > dates["FECHA_INGRESO"]).sum())
    q["inconsistent|symptoms > 30 days before attendance"] = int(((dates["FECHA_INGRESO"] - dates["FECHA_SINTOMAS"]).dt.days > 30).sum())
    q["inconsistent|attendance after file date"] = int((dates["FECHA_INGRESO"] > dates["FECHA_ACTUALIZACION"]).sum())
    q["implausible|age over 110"] = int((num["EDAD"] > 110).sum())
    return q


def _clean_chunk(c: pd.DataFrame) -> pd.DataFrame:
    """One raw chunk (all columns as text) -> cohort columns. Confirmed codes 1, 2, 3 are kept here."""
    c = c[c.CLASIFICACION_FINAL.isin(["1", "2", "3"])]
    code = lambda col: pd.to_numeric(c[col], errors="coerce").fillna(99).astype(np.int16)  # noqa: E731
    date = lambda col: pd.to_datetime(c[col], format="%Y-%m-%d", errors="coerce")      # noqa: E731
    out = pd.DataFrame({"id": c.ID_REGISTRO.values}, index=c.index)
    out["updated"] = date("FECHA_ACTUALIZACION")
    out["attended"] = date("FECHA_INGRESO")
    out["symptoms"] = date("FECHA_SINTOMAS")
    out["death_date"] = date("FECHA_DEF")             # 9999-99-99 / -001-11-30 -> NaT (alive)
    out["age"] = pd.to_numeric(c.EDAD, errors="coerce")
    sex = code("SEXO")
    out["sex_code"] = sex
    out["female"] = (sex == 1).astype(np.int8)
    out["pregnant"] = (code("EMBARAZO") == config.YES).astype(np.int8)
    missing_any = np.zeros(len(c), bool)
    for src, dst in config.COMORBIDITY_EN.items():
        v = code(src)
        out[dst] = (v == config.YES).astype(np.int8)
        missing_any |= v.isin(list(config.MISSING_CODES)).to_numpy()
    out["comorbidity_missing"] = missing_any.astype(np.int8)
    pneu = code("NEUMONIA")
    out["pneumonia"] = (pneu == config.YES).astype(np.int8)
    out["pneumonia_missing"] = pneu.isin(list(config.MISSING_CODES)).astype(np.int8)
    out["hospitalised"] = (code("TIPO_PACIENTE") == 2).astype(np.int8)
    out["icu"] = (code("UCI") == config.YES).astype(np.int8) if "UCI" in c else np.int8(0)
    out["intubated"] = (code("INTUBADO") == config.YES).astype(np.int8)
    out["state"] = code("ENTIDAD_UM")
    out["sector"] = code("SECTOR")
    out["usmer"] = (code("ORIGEN") == 1).astype(np.int8)
    out["classification"] = code("CLASIFICACION_FINAL")
    return out.reset_index(drop=True)


def read_file(path: Path, chunksize: int = CHUNK) -> pd.DataFrame:
    """Read one zip (or csv) of the line list in chunks, keeping confirmed cases only."""
    if zipfile.is_zipfile(path):
        z = zipfile.ZipFile(path)
        name = next(n for n in z.namelist() if n.lower().endswith(".csv"))
        handle = z.open(name)
    elif str(path).endswith(".gz"):
        handle = gzip.open(path, "rb")
    else:
        handle = open(path, "rb")
    parts, total, quality = [], 0, {}
    with handle:
        reader = pd.read_csv(handle, dtype=str, encoding="utf-8-sig", encoding_errors="replace",
                             usecols=lambda col: col.strip('"﻿') in config.OFFICIAL_COLUMNS, chunksize=chunksize)
        for chunk in reader:
            chunk.columns = [col.strip('"﻿') for col in chunk.columns]
            total += len(chunk)
            parts.append(_clean_chunk(chunk))
            for k, v in quality_counts(chunk).items():
                quality[k] = quality.get(k, 0) + v
            print(f"  {path.name}: {total:,} rows read, {sum(len(p) for p in parts):,} confirmed", end="\r")
    print()
    df = pd.concat(parts, ignore_index=True)
    df.attrs["rows_read"] = total
    df.attrs["quality"] = quality
    return df


# ------------------------------------------------------------------------ cohort

def build_cohort(frames: list[pd.DataFrame], rows_read: int) -> tuple[pd.DataFrame, list[dict]]:
    flow = [{"step": "records in the yearly files", "n": int(rows_read)}]
    df = pd.concat(frames, ignore_index=True)
    flow.append({"step": "confirmed or committee-classified (codes 1-3)", "n": int(len(df))})
    df = df.sort_values("updated").drop_duplicates("id", keep="last")
    flow.append({"step": "unique patients (latest version of each record)", "n": int(len(df))})
    df = df[df.classification.isin(config.CONFIRMED_CODES)]
    flow.append({"step": "confirmed COVID-19 (codes 1, 3); code 2 excluded (assigned only to deaths)", "n": int(len(df))})
    ok = df.age.between(0, 110) & df.sex_code.isin([1, 2]) & df.attended.notna()
    df = df[ok]
    flow.append({"step": "valid age, sex and attendance date", "n": int(len(df))})
    dev = df.attended.between(*config.DEV_PERIOD)
    test = df.attended.between(*config.TEST_PERIOD)
    follow = df.attended <= df.updated - pd.Timedelta(days=config.OUTCOME_DAYS)
    df = df[(dev | test) & follow].copy()
    flow.append({"step": f"attended {config.DEV_PERIOD[0]}..{config.TEST_PERIOD[1]} with >= {config.OUTCOME_DAYS} days follow-up",
                 "n": int(len(df))})
    days = (df.death_date - df.attended).dt.days
    bad = df.death_date.notna() & (days < 0)
    df = df[~bad].copy()
    days = days[~bad]
    flow.append({"step": "death not dated before attendance", "n": int(len(df))})
    df["died"] = (df.death_date.notna() & (days <= config.OUTCOME_DAYS)).astype(np.int8)
    df["died_any_time"] = df.death_date.notna().astype(np.int8)
    df["period"] = np.where(df.attended.between(*config.DEV_PERIOD), "development", "test")
    df["month"] = df.attended.dt.to_period("M").astype(str)
    delay = (df.attended - df.symptoms).dt.days
    df["days_symptoms_to_visit"] = delay.clip(0, 30).fillna(delay.median()).astype(np.float32)
    df["age"] = df.age.astype(np.float32)
    df = df.drop(columns=["sex_code", "updated"]).reset_index(drop=True)
    features.check_cohort(df)
    return df, flow


def summary(df: pd.DataFrame, flow: list[dict]) -> dict:
    by = {}
    for period, g in df.groupby("period"):
        by[period] = {"patients": int(len(g)), "deaths_30d": int(g.died.sum()), "death_rate_30d": float(g.died.mean()),
                      "attended_from": str(g.attended.min().date()), "attended_to": str(g.attended.max().date()),
                      "age_median": float(g.age.median()), "age_iqr": [float(g.age.quantile(.25)), float(g.age.quantile(.75))],
                      "female": float(g.female.mean()), "hospitalised": float(g.hospitalised.mean()),
                      "pneumonia": float(g.pneumonia.mean()), "states": int(g.state.nunique()),
                      "prevalence": {k: float(g[k].mean()) for k in config.COMORBIDITY_EN.values()}}
    return {"flow": flow, "periods": by}


def quality_report(counters: list[dict], frames_rows: int, cohort: pd.DataFrame) -> dict:
    """Completeness, validity, consistency, uniqueness and drift, from raw counters and the final cohort."""
    q: dict[str, int] = {}
    for c in counters:
        for k, v in c.items():
            q[k] = q.get(k, 0) + v
    n = q.get("records", 0)
    completeness = []
    for k, v in q.items():
        if k.startswith("missing|"):
            field = k.split("|", 1)[1]
            scope = q.get(f"scope|{field}", n)
            completeness.append({"field": field, "scope": scope, "missing": v, "share_missing": v / scope if scope else 0.0,
                                 "outside_catalogue": q.get(f"outside_catalogue|{field}", 0)})
    checks = [{"check": k.split("|", 1)[1], "kind": k.split("|", 1)[0], "records": v, "share": v / n if n else 0.0}
              for k, v in q.items() if k.split("|", 1)[0] in ("inconsistent", "implausible")]
    dev, test = cohort[cohort.period == "development"], cohort[cohort.period == "test"]
    drift = []
    for f in [*features.triage_features(cohort), "died"]:
        a, b = dev[f].astype(float), test[f].astype(float)
        pooled = np.sqrt((a.var() + b.var()) / 2)
        drift.append({"feature": f, "development": float(a.mean()), "test": float(b.mean()),
                      "smd": float((b.mean() - a.mean()) / pooled) if pooled > 0 else 0.0})
    return {"records_checked": n, "records_read": frames_rows, "completeness": sorted(completeness, key=lambda r: -r["share_missing"]),
            "checks": checks, "drift_2020_to_2021": sorted(drift, key=lambda r: -abs(r["smd"])),
            "note": "Counts are on confirmed records (codes 1-3) in the raw files, before de-duplication and exclusions. "
                    "'Missing' includes 98 (unknown) and 99 (not specified), and 97 (not applicable) where the field applies."}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--raw-dir", type=Path, default=config.DATA_DIR / "official_raw")
    ap.add_argument("--out", type=Path, default=config.ARTIFACTS_DIR / "cohort.pkl")
    ap.add_argument("--years", type=int, nargs="+", default=list(config.OFFICIAL_URLS))
    ap.add_argument("--files", type=Path, nargs="*", help="use these local files instead of downloading")
    args = ap.parse_args(argv)
    if args.out.exists():
        print(f"{args.out} exists; delete it to rebuild")
        return pd.read_pickle(args.out)
    paths = args.files or [download(y, args.raw_dir) for y in args.years]
    frames, rows = [], 0
    for p in paths:
        f = read_file(p)
        rows += f.attrs["rows_read"]
        frames.append(f)
    cohort, flow = build_cohort(frames, rows)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    cohort.to_pickle(args.out)
    (args.out.parent / "cohort_summary.json").write_text(json.dumps(summary(cohort, flow), indent=2))
    (args.out.parent / "data_quality.json").write_text(json.dumps(quality_report([f.attrs["quality"] for f in frames], rows, cohort), indent=2))
    for s in flow:
        print(f"{s['n']:>12,}  {s['step']}")
    return cohort


if __name__ == "__main__":
    main()
