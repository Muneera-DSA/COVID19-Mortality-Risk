import numpy as np
import pandas as pd

from covid_risk import config, official
from synthetic import make_official_zip, official_frame


def _build(tmp_path, frame):
    path = tmp_path / "COVID19MEXICO.csv"
    frame.to_csv(path, index=False, quoting=1)
    f = official.read_file(path, chunksize=500)
    return official.build_cohort([f], f.attrs["rows_read"])


def test_reads_official_zip_and_decodes_codes(tmp_path):
    z = make_official_zip(tmp_path / "COVID19MEXICO2020.zip", n=1500, seed=3)
    f = official.read_file(z, chunksize=400)
    raw = official_frame(n=1500, seed=3)
    assert f.attrs["rows_read"] == 1500
    assert set(f.classification) <= {1, 2, 3}
    assert len(f) == raw.CLASIFICACION_FINAL.isin(["1", "2", "3"]).sum()
    assert not ((f.pregnant == 1) & (f.female == 0)).any()
    conf = raw[raw.CLASIFICACION_FINAL.isin(["1", "2", "3"])].reset_index(drop=True)
    assert (f.diabetes.to_numpy() == (conf.DIABETES == "1").to_numpy()).all()
    assert (f.comorbidity_missing.to_numpy() == conf[["DIABETES", "EPOC", "ASMA", "INMUSUPR", "HIPERTENSION", "OTRA_COM",
                                                      "CARDIOVASCULAR", "OBESIDAD", "RENAL_CRONICA", "TABAQUISMO"]]
            .isin(["97", "98", "99"]).any(axis=1).to_numpy()).all()


def test_cohort_rules(tmp_path):
    fr = official_frame(n=8, seed=0, updated="2021-12-31")
    fr["CLASIFICACION_FINAL"] = "3"
    fr["EDAD"] = "50"
    fr["SEXO"] = "1"
    fr["FECHA_INGRESO"] = "2020-06-01"
    fr["FECHA_DEF"] = ["2020-06-10", "2020-07-10", "9999-99-99", "2020-05-20", "2020-06-05", "9999-99-99", "9999-99-99", "9999-99-99"]
    fr.loc[5, "CLASIFICACION_FINAL"] = "2"                        # committee-confirmed: excluded
    fr.loc[6, "FECHA_INGRESO"] = "2021-12-20"                      # < 30 days follow-up: excluded
    fr.loc[7, "EDAD"] = "130"                                      # invalid age: excluded
    cohort, flow = _build(tmp_path, fr)
    ids = dict(zip(cohort.id, cohort.died))
    assert ids[fr.ID_REGISTRO[0]] == 1                             # died on day 9
    assert ids[fr.ID_REGISTRO[1]] == 0                             # died on day 39: not a 30-day death
    assert cohort.loc[cohort.id == fr.ID_REGISTRO[1], "died_any_time"].item() == 1
    assert ids[fr.ID_REGISTRO[2]] == 0
    assert fr.ID_REGISTRO[3] not in ids                            # death before attendance: dropped
    assert fr.ID_REGISTRO[5] not in ids and fr.ID_REGISTRO[6] not in ids and fr.ID_REGISTRO[7] not in ids
    assert [s["n"] for s in flow] == sorted([s["n"] for s in flow], reverse=True)


def test_duplicate_records_keep_latest_version(tmp_path):
    a = official_frame(n=3, seed=1, updated="2021-01-10", start="2020-05-01", end="2020-05-02")
    b = a.copy()
    a["CLASIFICACION_FINAL"] = b["CLASIFICACION_FINAL"] = "3"
    a["EDAD"] = b["EDAD"] = "60"
    a["FECHA_DEF"] = "9999-99-99"
    b["FECHA_ACTUALIZACION"] = "2021-06-01"
    b["FECHA_DEF"] = (pd.to_datetime(b.FECHA_INGRESO) + pd.Timedelta(days=5)).dt.strftime("%Y-%m-%d")
    cohort, _ = _build(tmp_path, pd.concat([a, b]))
    assert len(cohort) == 3 and cohort.died.sum() == 3


def test_periods_and_delay_feature(tmp_path):
    cohort, _ = _build(tmp_path, official_frame(n=3000, seed=5))
    assert set(cohort.period) == {"development", "test"}
    dev = cohort[cohort.period == "development"]
    assert dev.attended.min() >= pd.Timestamp(config.DEV_PERIOD[0]) and dev.attended.max() <= pd.Timestamp(config.DEV_PERIOD[1])
    assert cohort.days_symptoms_to_visit.between(0, 30).all()
    assert np.isfinite(cohort.days_symptoms_to_visit).all()
