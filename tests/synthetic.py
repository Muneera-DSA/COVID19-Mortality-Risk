"""Synthetic data in the exact formats of the two real sources (no real patients).

- make_official_zip: a zip with one CSV in the Ministry of Health line-list layout (all 40 columns,
  official codes, UTF-8 with BOM, quoted header), covering 2020-2021 and several states.
- make_capstone_csv: the 21-column capstone extract layout.
Outcomes follow a known logistic model in age, comorbidities and pneumonia, so tests can check
that models recover signal; nothing here says anything about real accuracy.
"""

from __future__ import annotations

import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

OFFICIAL_HEADER = ["FECHA_ACTUALIZACION", "ID_REGISTRO", "ORIGEN", "SECTOR", "ENTIDAD_UM", "SEXO", "ENTIDAD_NAC",
                   "ENTIDAD_RES", "MUNICIPIO_RES", "TIPO_PACIENTE", "FECHA_INGRESO", "FECHA_SINTOMAS", "FECHA_DEF",
                   "INTUBADO", "NEUMONIA", "EDAD", "NACIONALIDAD", "EMBARAZO", "HABLA_LENGUA_INDIG", "INDIGENA",
                   "DIABETES", "EPOC", "ASMA", "INMUSUPR", "HIPERTENSION", "OTRA_COM", "CARDIOVASCULAR", "OBESIDAD",
                   "RENAL_CRONICA", "TABAQUISMO", "OTRO_CASO", "TOMA_MUESTRA_LAB", "RESULTADO_LAB",
                   "TOMA_MUESTRA_ANTIGENO", "RESULTADO_ANTIGENO", "CLASIFICACION_FINAL", "MIGRANTE",
                   "PAIS_NACIONALIDAD", "PAIS_ORIGEN", "UCI"]
COMORB = ["DIABETES", "EPOC", "ASMA", "INMUSUPR", "HIPERTENSION", "OTRA_COM", "CARDIOVASCULAR", "OBESIDAD",
          "RENAL_CRONICA", "TABAQUISMO"]


def _patients(n: int, rng: np.random.Generator, start: str, end: str, n_states: int):
    age = np.clip(rng.gamma(4.5, 10, n), 0, 105).astype(int)
    female = rng.random(n) < 0.5
    com = {c: rng.random(n) < (0.03 + 0.25 * (age / 100) * (c in ("DIABETES", "HIPERTENSION"))) for c in COMORB}
    com["OBESIDAD"] = rng.random(n) < 0.18
    pneumonia = rng.random(n) < (0.05 + 0.4 * age / 100)
    lin = -6.2 + 0.065 * age + 0.8 * com["DIABETES"] + 0.9 * com["RENAL_CRONICA"] + 1.6 * pneumonia - 0.3 * female
    state = rng.integers(1, n_states + 1, n)
    lin += 0.15 * (state % 3)
    died = rng.random(n) < 1 / (1 + np.exp(-lin))
    days = (pd.Timestamp(end) - pd.Timestamp(start)).days
    attended = pd.Timestamp(start) + pd.to_timedelta(rng.integers(0, days, n), unit="D")
    symptoms = attended - pd.to_timedelta(rng.integers(0, 8, n), unit="D")
    death = attended + pd.to_timedelta(rng.integers(0, 40, n), unit="D")
    hosp = died | (pneumonia & (rng.random(n) < 0.7)) | (rng.random(n) < 0.05)
    return dict(age=age, female=female, com=com, pneumonia=pneumonia, died=died, state=state,
                attended=attended, symptoms=symptoms, death=death, hosp=hosp)


def official_frame(n: int = 4000, seed: int = 0, start: str = "2020-03-01", end: str = "2021-11-30",
                   updated: str = "2022-01-05", n_states: int = 6) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    p = _patients(n, rng, start, end, n_states)
    yn = lambda b: np.where(b, "1", "2")  # noqa: E731
    df = pd.DataFrame({c: "99" for c in OFFICIAL_HEADER}, index=range(n))
    df["FECHA_ACTUALIZACION"] = updated
    df["ID_REGISTRO"] = [f"{seed:02d}{i:07x}" for i in range(n)]
    df["ORIGEN"] = rng.choice(["1", "2"], n)
    df["SECTOR"] = rng.choice(["4", "12", "6", "9"], n, p=[0.5, 0.35, 0.1, 0.05])
    df["ENTIDAD_UM"] = [f"{s:02d}" for s in p["state"]]
    df["SEXO"] = np.where(p["female"], "1", "2")
    df["TIPO_PACIENTE"] = np.where(p["hosp"], "2", "1")
    df["FECHA_INGRESO"] = p["attended"].strftime("%Y-%m-%d")
    df["FECHA_SINTOMAS"] = p["symptoms"].strftime("%Y-%m-%d")
    df["FECHA_DEF"] = np.where(p["died"], p["death"].strftime("%Y-%m-%d"), "9999-99-99")
    df["INTUBADO"] = np.where(p["hosp"], yn(p["died"] & (rng.random(n) < 0.4)), "97")
    df["UCI"] = np.where(p["hosp"], yn(p["died"] & (rng.random(n) < 0.3)), "97")
    df["NEUMONIA"] = np.where(rng.random(n) < 0.01, "99", yn(p["pneumonia"]))
    df["EDAD"] = p["age"].astype(str)
    df["EMBARAZO"] = np.where(p["female"], yn((p["age"] > 18) & (p["age"] < 40) & (rng.random(n) < 0.05)), "97")
    for c in COMORB:
        df[c] = np.where(rng.random(n) < 0.005, "98", yn(p["com"][c]))
    df["CLASIFICACION_FINAL"] = rng.choice(["3", "1", "7", "6", "2"], n, p=[0.6, 0.05, 0.25, 0.08, 0.02])
    df.loc[df.CLASIFICACION_FINAL == "2", "FECHA_DEF"] = df.loc[df.CLASIFICACION_FINAL == "2", "FECHA_INGRESO"]
    return df


def make_official_zip(path: Path, **kw) -> Path:
    df = official_frame(**kw)
    path.parent.mkdir(parents=True, exist_ok=True)
    csv = df.to_csv(index=False, quoting=1)           # quoted header, like the official files
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr(path.stem + ".csv", "﻿" + csv)
    return path


def make_capstone_csv(path: Path, n: int = 3000, seed: int = 0) -> Path:
    rng = np.random.default_rng(seed)
    p = _patients(n, rng, "2020-03-01", "2021-04-30", 4)
    cap = {"DIABETES": "DIABETES", "COPD": "EPOC", "ASTHMA": "ASMA", "INMSUPR": "INMUSUPR", "HIPERTENSION": "HIPERTENSION",
           "OTHER_DISEASE": "OTRA_COM", "CARDIOVASCULAR": "CARDIOVASCULAR", "OBESITY": "OBESIDAD",
           "RENAL_CHRONIC": "RENAL_CRONICA", "TOBACCO": "TABAQUISMO"}
    df = pd.DataFrame({
        "USMER": rng.choice([1, 2], n), "MEDICAL_UNIT": rng.choice([1, 2, 3, 4], n, p=[.01, .01, .1, .88]),
        "SEX": np.where(p["female"], 1, 2), "PATIENT_TYPE": np.where(p["hosp"], 2, 1),
        "DATE_DIED": np.where(p["died"], p["death"].strftime("%d-%m-%Y"), "9999-99-99"),
        "INTUBED": np.where(p["hosp"], np.where(p["died"] & (rng.random(n) < .4), 1, 2), 3),
        "PNEUMONIA": np.where(rng.random(n) < .02, 99, np.where(p["pneumonia"], 1, 2)), "AGE": p["age"],
        "PREGNANT": np.where(p["female"], np.where((p["age"] > 18) & (p["age"] < 40) & (rng.random(n) < .05), 1, 2), 3),
    })
    for k, v in cap.items():
        df[k] = p["com"][v].astype(int)
    df["CLASIFFICATION_FINAL"] = rng.choice([3, 1, 7, 6, 2, 5], n, p=[.6, .05, .15, .12, .03, .05])
    df.loc[df.CLASIFFICATION_FINAL == 2, "DATE_DIED"] = "01-06-2020"
    df["ICU"] = (df.PATIENT_TYPE == 2).astype(int)
    survivors = df[df.DATE_DIED == "9999-99-99"]
    df = pd.concat([df, survivors.sample(frac=0.3, random_state=seed)], ignore_index=True)   # identical profiles
    order = ["USMER", "MEDICAL_UNIT", "SEX", "PATIENT_TYPE", "DATE_DIED", "INTUBED", "PNEUMONIA", "AGE", "PREGNANT",
             "DIABETES", "COPD", "ASTHMA", "INMSUPR", "HIPERTENSION", "OTHER_DISEASE", "CARDIOVASCULAR", "OBESITY",
             "RENAL_CHRONIC", "TOBACCO", "CLASIFFICATION_FINAL", "ICU"]
    path.parent.mkdir(parents=True, exist_ok=True)
    df[order].to_csv(path, index=False)
    return path
