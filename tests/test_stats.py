import numpy as np
from scipy.special import expit
from sklearn.metrics import roc_auc_score

from covid_risk import stats


def _data(n=4000, seed=0):
    rng = np.random.default_rng(seed)
    x = rng.normal(size=n)
    p = expit(-2 + 1.5 * x)
    y = (rng.random(n) < p).astype(int)
    return x, p, y


def test_delong_matches_sklearn_and_ci_contains_it():
    x, p, y = _data()
    d = stats.delong_auc(y, p)
    assert abs(d["value"] - roc_auc_score(y, p)) < 1e-12
    assert d["ci"][0] < d["value"] < d["ci"][1]


def test_delong_paired_detects_a_better_model_and_is_antisymmetric():
    x, p, y = _data()
    noisy = p + np.random.default_rng(1).normal(scale=0.3, size=len(p))
    ab, ba = stats.delong_paired(y, p, noisy), stats.delong_paired(y, noisy, p)
    assert ab["diff"] > 0 and ab["p"] < 1e-3
    assert abs(ab["diff"] + ba["diff"]) < 1e-12 and abs(ab["p"] - ba["p"]) < 1e-12


def test_calibration_of_true_probabilities_is_ideal():
    _, p, y = _data(20000)
    c = stats.calibration_intercept_slope(y, p)
    assert abs(c["intercept"]) < 3 * c["intercept_se"] + 0.02
    assert abs(c["slope"] - 1) < 3 * c["slope_se"] + 0.02
    over = stats.calibration_intercept_slope(y, np.clip(p * 2, 0, 0.99))
    assert over["intercept"] < -0.3                      # over-prediction -> negative intercept
    assert stats.expected_calibration_error(y, p) < 0.02


def test_odds_ratios_recover_known_coefficients():
    rng = np.random.default_rng(3)
    X = np.column_stack([rng.normal(size=30000), rng.random(30000) < 0.3])
    y = (rng.random(30000) < expit(-1 + 0.7 * X[:, 0] + 1.2 * X[:, 1])).astype(int)
    rows = stats.logistic_odds_ratios(X, y, ["x", "b"])
    assert abs(np.log(rows[0]["odds_ratio"]) - 0.7) < 0.05 and abs(np.log(rows[1]["odds_ratio"]) - 1.2) < 0.08
    assert rows[0]["ci"][0] < rows[0]["odds_ratio"] < rows[0]["ci"][1]


def test_net_benefit_treat_all_formula_and_capacity():
    y = np.array([1, 0, 0, 0, 1, 0, 0, 0, 0, 0])
    p = np.array([.9, .1, .2, .1, .8, .1, .3, .1, .1, .1])
    nb = stats.net_benefit(y, p, [0.2])
    assert abs(nb["treat_all"][0] - (0.2 - 0.8 * 0.25)) < 1e-12
    assert abs(nb["model"][0] - (2 / 10 - 2 / 10 * 0.25)) < 1e-12        # flags .9, .8, .3, .2: 2 deaths, 2 survivors
    cap = stats.capture_at_capacity(y, p, 0.2)
    assert cap["deaths_captured"] == 1.0 and cap["death_rate_in_flagged"] == 1.0


def test_subsample_keeps_class_balance_and_bootstrap_brackets_point():
    y = np.r_[np.ones(1000), np.zeros(9000)].astype(int)
    idx = stats.stratified_subsample(y, 2000)
    assert len(idx) == 2000 and abs(y[idx].mean() - 0.1) < 1e-9
    x, p, yy = _data(3000)
    b = stats.bootstrap_metric(lambda i: roc_auc_score(yy[i], p[i]), len(yy), 100)
    assert b["ci"][0] <= b["value"] <= b["ci"][1]


def test_holm_is_monotone_and_capped():
    adj = stats.holm({"a": 0.01, "b": 0.04, "c": 0.5})
    assert abs(adj["a"] - 0.03) < 1e-12 and abs(adj["b"] - 0.08) < 1e-12 and abs(adj["c"] - 0.5) < 1e-12
    assert stats.holm({"a": 0.4, "b": 0.5})["a"] == 0.8 and stats.holm({"a": 0.6, "b": 0.7})["a"] == 1.0
