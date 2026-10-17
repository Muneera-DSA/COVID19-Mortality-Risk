"""Paths, data codes and study settings in one place."""

from __future__ import annotations

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
REPORTS_DIR = PROJECT_ROOT / "reports"
RESULTS_DIR = PROJECT_ROOT / "results"
CAPSTONE_CSV = DATA_DIR / "covid_capstone.csv.gz"

SEED = 42

# ------------------------------------------------------------------ official data
# Mexican Ministry of Health (Direccion General de Epidemiologia) open COVID-19 line list.
# One frozen file per year; addresses as used by the CRAN package covidmx (get_sites_covid()).
OFFICIAL_URLS = {
    2020: "https://datosabiertos.salud.gob.mx/gobmx/salud/datos_abiertos/historicos/2020/COVID19MEXICO2020.zip",
    2021: "https://datosabiertos.salud.gob.mx/gobmx/salud/datos_abiertos/historicos/2021/COVID19MEXICO2021.zip",
}
OFFICIAL_PORTAL = "https://www.gob.mx/salud/documentos/datos-abiertos-bases-historicas-direccion-general-de-epidemiologia"

# Catalogue SI_NO: 1 yes, 2 no, 97 not applicable, 98 unknown, 99 not specified.
YES, NO = 1, 2
MISSING_CODES = {97, 98, 99}

# CLASIFICACION_FINAL: 1 confirmed by clinical-epidemiological association, 2 confirmed by a
# committee (applies ONLY to deaths, so it is selected on the outcome), 3 laboratory/antigen
# confirmed, 4 invalid, 5 not processed, 6 suspected, 7 negative.
CONFIRMED_CODES = (1, 3)
OUTCOME_DEFINED_CODE = 2

COMORBIDITIES = ["DIABETES", "EPOC", "ASMA", "INMUSUPR", "HIPERTENSION", "OTRA_COM",
                 "CARDIOVASCULAR", "OBESIDAD", "RENAL_CRONICA", "TABAQUISMO"]
COMORBIDITY_EN = {"DIABETES": "diabetes", "EPOC": "copd", "ASMA": "asthma", "INMUSUPR": "immunosuppression",
                  "HIPERTENSION": "hypertension", "OTRA_COM": "other_disease", "CARDIOVASCULAR": "cardiovascular",
                  "OBESIDAD": "obesity", "RENAL_CRONICA": "chronic_kidney", "TABAQUISMO": "smoking"}
OFFICIAL_COLUMNS = ["FECHA_ACTUALIZACION", "ID_REGISTRO", "ORIGEN", "SECTOR", "ENTIDAD_UM", "SEXO", "TIPO_PACIENTE",
                    "FECHA_INGRESO", "FECHA_SINTOMAS", "FECHA_DEF", "INTUBADO", "NEUMONIA", "EDAD", "EMBARAZO",
                    *COMORBIDITIES, "CLASIFICACION_FINAL", "UCI"]

# ------------------------------------------------------------------ study design
OUTCOME_DAYS = 30                       # death within 30 days of first attendance
DEV_PERIOD = ("2020-03-01", "2020-12-31")   # development: first wave
TEST_PERIOD = ("2021-01-01", "2021-11-30")  # temporal external test: later waves (Delta, vaccination)

# Features known when the patient first presents (triage)
TRIAGE_FEATURES = ["age", "female", "pregnant", *COMORBIDITY_EN.values(), "pneumonia", "pneumonia_missing",
                   "comorbidity_missing", "days_symptoms_to_visit"]
# Recorded after the triage decision: used only to measure leakage
POST_TRIAGE_FEATURES = ["hospitalised", "icu", "intubated"]

# Model-selection CV and tuning run on a stratified subsample (the full cohort is >1M rows)
SELECTION_SAMPLE = 200_000
BENCHMARK_SAMPLE = 100_000
IECV_SAMPLE = 400_000
CV_FOLDS = 5
N_BOOT = 300
BOOT_MAX_ROWS = 400_000          # bootstrap on a stratified subsample of very large test sets
CAPACITY_SHARES = (0.05, 0.10, 0.20, 0.30)   # share of patients flagged for priority care
DECISION_THRESHOLDS = (0.05, 0.10, 0.20, 0.30)

# ENTIDAD_UM catalogue (official "Catálogo de ENTIDADES")
STATES = {1: "Aguascalientes", 2: "Baja California", 3: "Baja California Sur", 4: "Campeche", 5: "Coahuila", 6: "Colima",
          7: "Chiapas", 8: "Chihuahua", 9: "Mexico City", 10: "Durango", 11: "Guanajuato", 12: "Guerrero", 13: "Hidalgo",
          14: "Jalisco", 15: "State of Mexico", 16: "Michoacán", 17: "Morelos", 18: "Nayarit", 19: "Nuevo León", 20: "Oaxaca",
          21: "Puebla", 22: "Querétaro", 23: "Quintana Roo", 24: "San Luis Potosí", 25: "Sinaloa", 26: "Sonora", 27: "Tabasco",
          28: "Tamaulipas", 29: "Tlaxcala", 30: "Veracruz", 31: "Yucatán", 32: "Zacatecas"}
# SECTOR catalogue (main institutions)
SECTORS = {1: "Red Cross", 2: "DIF", 3: "State", 4: "IMSS", 5: "IMSS-Bienestar", 6: "ISSSTE", 7: "Municipal", 8: "PEMEX",
           9: "Private", 10: "SEDENA", 11: "SEMAR", 12: "SSA (Health Ministry)", 13: "University"}
