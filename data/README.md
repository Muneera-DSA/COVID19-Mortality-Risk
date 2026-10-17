# Data

## 1. Official Mexican COVID-19 line list (Part 2, main study)

Not stored here: it is downloaded by `python -m covid_risk.official` (or the Colab notebook).

- **Source:** Secretaría de Salud, Dirección General de Epidemiología. Open data on COVID-19 cases notified to the epidemiological surveillance system (SISVER), one frozen file per year:
  - `COVID19MEXICO2020.zip`
  - `COVID19MEXICO2021.zip`
- **Address:** `https://datosabiertos.salud.gob.mx/gobmx/salud/datos_abiertos/historicos/<year>/COVID19MEXICO<year>.zip`. This is the address used by the CRAN package `covidmx`. If it moves, see the [historical-data page](https://www.gob.mx/salud/documentos/datos-abiertos-bases-historicas-direccion-general-de-epidemiologia).
- **Coding:** official catalogues.
  - `SI_NO`: 1 yes, 2 no, 97 not applicable, 98 unknown, 99 not specified.
  - `CLASIFICACION_FINAL`: 1–3 confirmed. Code 2 is used only for deaths. 4–7 are not confirmed.
- **Licence:** Mexican federal open data, *Libre Uso MX* (free use with attribution).

## 2. `covid_capstone.csv.gz` (Part 1, audit of the original notebook)

The 199,999-row extract used by the original capstone notebook, gzipped. It is derived from the same official line list, but someone modified it:
- `ICU` is a copy of `PATIENT_TYPE`;
- the medical units are collapsed to 4 values;
- comorbidities are recoded to 0/1.

Its sampling procedure is undocumented. Its death rate (20.7%) is high for confirmed cases, which suggests enrichment.

It is kept here so that `python -m covid_risk.capstone` reproduces the audit exactly.

## 3. `official_sample.csv.gz`: a small sample of the official files

58,964 records (attended July–October 2021) in the exact official format, used by `notebooks/walkthrough.ipynb` to run the cleaning, validation and feature-engineering code live on real records.

It is the public example file distributed with the R package `covidmx` (MIT licence, © Instituto Mexicano del Seguro Social, 2021), which in turn comes from the Secretaría de Salud open data (*Libre Uso MX*).
