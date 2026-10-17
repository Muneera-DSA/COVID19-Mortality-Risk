# Model card: COVID-19 30-day mortality risk at first attendance

Format: Mitchell et al., *Model Cards for Model Reporting* (2019). The reporting follows the TRIPOD checklist for prediction models where it applies. Performance figures are copied from `reports/results.md` only.

## Model details
- **Type:** the model chosen by cross-validation from three candidates: age-only logistic regression, logistic regression with an age spline, and histogram gradient boosting. The choice and its rule are recorded in `results/select.json`.
- **Output:** probability of death within 30 days of first attendance.
- **Training data:** confirmed COVID-19 patients in Mexico's official open line list, attended March–December 2020.
- **Author:** Muneera Mohamed. Portfolio research project; not deployed.

## Intended use
- Research and education: showing how a clinical risk model should be validated (over time and place), checked for leakage and calibrated.
- At most, decision support for prioritising monitoring of confirmed COVID-19 patients at first contact, with clinicians deciding.

## Out-of-scope uses
- Diagnosis of COVID-19, or use for patients without a confirmed infection.
- Any automated admission, triage or treatment decision.
- Allocation of scarce care in a way that could disadvantage groups (older people, people with disabilities) without a clinical and ethical review.
- Settings outside Mexico, or periods with different variants, vaccination coverage or testing policies, without re-validation and re-calibration.

## Inputs
Age, sex, pregnancy, diabetes, COPD, asthma, immunosuppression, hypertension, other chronic disease, cardiovascular disease, obesity, chronic kidney disease, smoking, pneumonia, and days from symptom onset to attendance, plus flags for missing values.

**Not used as predictors:** hospitalisation, ICU admission and intubation. They are decided after triage and would leak the outcome; the leakage check measures how much they would inflate the scores.

## Performance
Copied from `reports/results.md`. The chosen model is tuned histogram gradient boosting, trained on 1,511,556 patients (2020).

**Temporal test** (2,410,830 patients attended January–November 2021, 5.7% died within 30 days):

| Metric | Value (95% CI) |
|---|---|
| ROC-AUC | 0.949 (0.949–0.950) |
| PR-AUC | 0.546 (0.539–0.553) |
| Brier / ECE | 0.0334 / 0.0043 |
| Calibration intercept / slope | −0.049 / 1.070 |
| Deaths among the 10% / 20% highest-risk | 78.1% / 92.3% |
| Monthly ROC-AUC range | 0.937–0.958 |

**Other checks:**
- **States:** each state held out in turn gives median ROC-AUC 0.927 (range 0.872–0.962), with the calibration intercept ranging from −0.46 to +0.39.
- **Subgroups:** ROC-AUC 0.953 for women and 0.945 for men; 0.780 for ages 80+; 0.705 among patients with pneumonia.
- **Sector calibration:** private units 8.3% predicted vs 4.1% observed; IMSS 5.5% vs 6.8%.

## Known limitations and risks
- **Under-recorded deaths:** the official line list misses some deaths (excess-mortality studies estimate a large shortfall), so absolute risks may be underestimated.
- **Testing policy:** who gets tested changed over time, which changes the confirmed population (case mix). The temporal test measures the effect, but cannot remove it.
- **Missing clinical data:** no vital signs, oxygen saturation, laboratory values or vaccination status. Those are strong predictors that the line list lacks.
- **"Pneumonia at attendance":** may sometimes be recorded later in the stay. A model without it is a natural sensitivity check.
- **Recalibration:** calibration drifts between waves; a single fixed correction made it worse. A rolling monthly intercept update helped. Re-estimate at least the intercept on recent local data before any use.
- **Institution and state:** calibration differs by health-care sector and state (sector is not an input). Local recalibration is required.
- **Weak within high-risk groups:** ROC-AUC 0.78 among the over-80s and 0.71 among patients with pneumonia. Do not use the model to rank patients within these groups.
- **Associations, not causes:** smoking and asthma appear protective, a known artefact of surveillance data.

## Ethical considerations
- The data are de-identified public surveillance records.
- Risk scores are dominated by age, so a "highest risk first" policy prioritises older patients. That may be clinically appropriate, but it must be an explicit decision.
- Subgroup results are reported so that unequal performance is visible.
