import numpy as np

from covid_risk import models

FEATS = ["age", "female", "diabetes", "pneumonia", "days_symptoms_to_visit"]


def _xy(n=3000, seed=0):
    rng = np.random.default_rng(seed)
    X = np.column_stack([rng.integers(0, 100, n), rng.random(n) < .5, rng.random(n) < .15, rng.random(n) < .2,
                         rng.integers(0, 15, n)]).astype(float)
    lin = -6 + 0.06 * X[:, 0] + 0.8 * X[:, 2] + 1.5 * X[:, 3]
    y = (rng.random(n) < 1 / (1 + np.exp(-lin))).astype(int)
    return X, y


def test_smote_nc_balances_and_keeps_binary_features_binary():
    X, y = _xy()
    Xr, yr = models.smote_nc(X, y, continuous=[0, 4], seed=1)
    assert (yr == 1).sum() == (yr == 0).sum()
    new = Xr[len(X):]
    assert set(np.unique(new[:, [1, 2, 3]])) <= {0.0, 1.0}            # no "0.37 diabetic"
    pos = X[y == 1]
    assert new[:, 0].min() >= pos[:, 0].min() and new[:, 0].max() <= pos[:, 0].max()


def test_random_oversample_duplicates_only_minority():
    X, y = _xy()
    Xr, yr = models.random_oversample(X, y, seed=0)
    assert (yr == 1).sum() == (yr == 0).sum() and (yr == 0).sum() == (y == 0).sum()


def test_every_model_and_imbalance_strategy_fits_and_predicts_probabilities():
    X, y = _xy()
    for name in ("age_only", "logistic", "hgb"):
        for imb in ("none", "class_weight", "oversample", "smote"):
            fs = ["age"] if name == "age_only" else FEATS
            Xm = X[:, :1] if name == "age_only" else X
            p = models.build(name, fs, imb).fit(Xm, y).predict_proba(Xm)[:, 1]
            assert p.shape == (len(y),) and np.all((p >= 0) & (p <= 1)), (name, imb)


def test_resampling_shifts_mean_risk_but_plain_model_does_not():
    X, y = _xy(6000)
    plain = models.build("logistic", FEATS).fit(X, y).predict_proba(X)[:, 1]
    over = models.build("logistic", FEATS, "oversample").fit(X, y).predict_proba(X)[:, 1]
    assert abs(plain.mean() - y.mean()) < 0.01
    assert over.mean() > y.mean() + 0.1          # trained on 50/50 data -> inflated risk


def test_linear_shap_is_exact_and_matches_the_shap_library():
    from covid_risk import experiments
    X, y = _xy(4000)
    pipe = models.build("logistic", FEATS).fit(X, y)
    sv, base = experiments.linear_shap(pipe, X[:300], X[1000:3000], FEATS)
    assert np.max(np.abs(sv.sum(axis=1) + base - pipe.decision_function(X[:300]))) < 1e-8
    assert np.abs(sv).mean(axis=0).argmax() == FEATS.index("pneumonia") or np.abs(sv).mean(axis=0).argmax() == FEATS.index("age")
    try:
        import shap
    except ImportError:
        return
    pre, lr = pipe.named_steps["pre"], pipe.named_steps["lr"]
    Zb = pre.transform(X[1000:3000])
    ex = shap.LinearExplainer(lr, shap.maskers.Independent(Zb, max_samples=len(Zb)))
    lib = np.asarray(ex.shap_values(pre.transform(X[:300])))
    assert np.max(np.abs(lib.sum(axis=1) - sv.sum(axis=1))) < 1e-6


def test_rolling_recalibration_corrects_a_drifting_intercept_without_look_ahead():
    import pandas as pd
    from scipy.special import expit

    from covid_risk import experiments
    rng = np.random.default_rng(4)
    rows = []
    for k, shift in enumerate([0.0, -0.5, -0.5, -0.5, -0.5]):        # true risk drops after month 1; the model does not know
        lp = rng.normal(-3, 1.2, 20000)
        y = rng.random(20000) < expit(lp + shift)
        rows.append(pd.DataFrame({"month": f"2021-0{k + 1}", "died": y.astype(int), "p_hgb": expit(lp)}))
    pred = pd.concat(rows, ignore_index=True)
    r = experiments.rolling_recalibration(pred, "p_hgb")
    assert r["months"][0]["shift_applied"] == 0.0                       # first month: nothing to learn from yet
    assert abs(r["after"]["intercept"]) < abs(r["before"]["intercept"]) / 2
    assert r["after"]["ece"] < r["before"]["ece"]
