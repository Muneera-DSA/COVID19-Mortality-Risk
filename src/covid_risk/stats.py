"""Statistics used for every claim in the reports.

- delong_auc / delong_paired : AUC with CI and the paired test for two models on the same patients
  (DeLong et al. 1988; fast algorithm of Sun & Xu 2014, vectorised).
- bootstrap_metric / bootstrap_paired : percentile CIs by resampling patients.
- calibration_intercept_slope : calibration-in-the-large and slope (Van Calster et al. 2019).
- expected_calibration_error, brier, net_benefit (decision-curve analysis, Vickers & Elkin 2006).
- logistic_odds_ratios : adjusted odds ratios with Wald CIs (IRLS, no extra dependency).
- holm : family-wise correction.
"""

from __future__ import annotations

from typing import Callable

import numpy as np
from scipy import stats
from scipy.special import expit, logit


# ----------------------------------------------------------------------- DeLong

def _delong_components(y: np.ndarray, scores: np.ndarray):
    y = np.asarray(y).astype(bool)
    pos, neg = scores[:, y], scores[:, ~y]
    m, n = pos.shape[1], neg.shape[1]
    tx = np.apply_along_axis(stats.rankdata, 1, pos)
    ty = np.apply_along_axis(stats.rankdata, 1, neg)
    tz = np.apply_along_axis(stats.rankdata, 1, np.hstack([pos, neg]))
    aucs = (tz[:, :m].sum(axis=1) / m - (m + 1) / 2) / n
    v01 = (tz[:, :m] - tx) / n
    v10 = 1 - (tz[:, m:] - ty) / m
    sx, sy = np.atleast_2d(np.cov(v01)), np.atleast_2d(np.cov(v10))
    return aucs, sx / m + sy / n


def delong_auc(y, score, conf: float = 0.95) -> dict:
    aucs, cov = _delong_components(np.asarray(y), np.asarray(score, float)[None, :])
    auc, se = float(aucs[0]), float(np.sqrt(cov[0, 0]))
    z = stats.norm.ppf(0.5 + conf / 2)
    return {"value": auc, "se": se, "ci": (max(0.0, auc - z * se), min(1.0, auc + z * se))}


def delong_paired(y, score_a, score_b) -> dict:
    """H0: AUC(a) == AUC(b) on the same patients. Two-sided."""
    s = np.vstack([np.asarray(score_a, float), np.asarray(score_b, float)])
    aucs, cov = _delong_components(np.asarray(y), s)
    var = cov[0, 0] + cov[1, 1] - 2 * cov[0, 1]
    diff = float(aucs[0] - aucs[1])
    se = float(np.sqrt(var)) if var > 0 else 0.0
    z = diff / se if se > 0 else 0.0
    return {"auc_a": float(aucs[0]), "auc_b": float(aucs[1]), "diff": diff,
            "ci": (diff - 1.96 * se, diff + 1.96 * se), "z": float(z), "p": float(2 * stats.norm.sf(abs(z)))}


# --------------------------------------------------------------------- bootstrap

def stratified_subsample(y, max_rows: int, seed: int = 0) -> np.ndarray:
    """Indices of a class-stratified subsample (all rows if already small enough)."""
    y = np.asarray(y)
    if len(y) <= max_rows:
        return np.arange(len(y))
    rng = np.random.default_rng(seed)
    out = []
    for c in np.unique(y):
        idx = np.flatnonzero(y == c)
        out.append(rng.choice(idx, int(round(max_rows * len(idx) / len(y))), replace=False))
    return np.sort(np.concatenate(out))


def bootstrap_metric(metric: Callable[[np.ndarray], float], n: int, n_boot: int = 500, seed: int = 0,
                     conf: float = 0.95) -> dict:
    """Percentile CI for metric(idx); `metric` receives row indices (resampled with replacement)."""
    rng = np.random.default_rng(seed)
    point = float(metric(np.arange(n)))
    vals = np.array([metric(rng.integers(0, n, n)) for _ in range(n_boot)], float)
    vals = vals[np.isfinite(vals)]
    if len(vals) == 0:
        return {"value": point, "ci": (float("nan"), float("nan")), "p_le_0": float("nan")}
    a = (1 - conf) / 2
    return {"value": point, "ci": (float(np.quantile(vals, a)), float(np.quantile(vals, 1 - a))),
            "p_le_0": float((vals <= 0).mean())}


def bootstrap_diff(metric: Callable[[np.ndarray, np.ndarray], float], y, pa, pb, n_boot: int = 500,
                   seed: int = 0, max_rows: int | None = None) -> dict:
    """CI for metric(y, pa) - metric(y, pb), paired by patient. Two-sided bootstrap p (sign of draws)."""
    y, pa, pb = np.asarray(y), np.asarray(pa), np.asarray(pb)
    if max_rows:
        keep = stratified_subsample(y, max_rows, seed)
        y, pa, pb = y[keep], pa[keep], pb[keep]
    point = metric(y, pa) - metric(y, pb)
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(n_boot):
        i = rng.integers(0, len(y), len(y))
        vals.append(metric(y[i], pa[i]) - metric(y[i], pb[i]))
    vals = np.asarray(vals)
    p = 2 * min((vals <= 0).mean(), (vals >= 0).mean())
    return {"value": float(point), "ci": (float(np.quantile(vals, 0.025)), float(np.quantile(vals, 0.975))),
            "p": float(max(min(p, 1.0), 1 / n_boot))}


# ------------------------------------------------------------------ calibration

def _fit_logistic(X: np.ndarray, y: np.ndarray, offset: np.ndarray | None = None, max_iter: int = 50):
    """Unpenalised logistic regression by IRLS. Returns coefficients and their covariance."""
    X = np.asarray(X, float)
    y = np.asarray(y, float)
    off = np.zeros(len(y)) if offset is None else offset
    beta = np.zeros(X.shape[1])
    for _ in range(max_iter):
        p = expit(X @ beta + off)
        w = np.clip(p * (1 - p), 1e-12, None)
        hess = X.T @ (X * w[:, None])
        step = np.linalg.solve(hess + 1e-10 * np.eye(len(beta)), X.T @ (y - p))
        beta = beta + step
        if np.max(np.abs(step)) < 1e-8:
            break
    p = expit(X @ beta + off)
    hess = X.T @ (X * np.clip(p * (1 - p), 1e-12, None)[:, None])
    return beta, np.linalg.pinv(hess)


def calibration_intercept_slope(y, p) -> dict:
    """Intercept (calibration-in-the-large, slope fixed at 1) and slope. Ideal: 0 and 1."""
    y = np.asarray(y, float)
    lp = logit(np.clip(np.asarray(p, float), 1e-6, 1 - 1e-6))
    a, cov_a = _fit_logistic(np.ones((len(y), 1)), y, offset=lp)
    b, cov_b = _fit_logistic(np.column_stack([np.ones(len(y)), lp]), y)
    return {"intercept": float(a[0]), "intercept_se": float(np.sqrt(cov_a[0, 0])),
            "slope": float(b[1]), "slope_se": float(np.sqrt(cov_b[1, 1])),
            "observed_rate": float(y.mean()), "mean_predicted": float(np.mean(p))}


def expected_calibration_error(y, p, n_bins: int = 15) -> float:
    """ECE on equal-width probability bins for the positive class."""
    y, p = np.asarray(y, float), np.asarray(p, float)
    bins = np.minimum((p * n_bins).astype(int), n_bins - 1)
    ece = 0.0
    for b in np.unique(bins):
        m = bins == b
        ece += m.mean() * abs(y[m].mean() - p[m].mean())
    return float(ece)


def brier(y, p) -> float:
    return float(np.mean((np.asarray(p, float) - np.asarray(y, float)) ** 2))


def reliability_curve(y, p, n_bins: int = 10) -> dict:
    """Observed vs predicted risk in equal-size (quantile) bins."""
    y, p = np.asarray(y, float), np.asarray(p, float)
    order = np.argsort(p)
    groups = np.array_split(order, n_bins)
    return {"predicted": [float(p[g].mean()) for g in groups], "observed": [float(y[g].mean()) for g in groups]}


def net_benefit(y, p, thresholds) -> dict:
    """Decision-curve analysis: net benefit of acting on p >= t, of treating all, and of none (0)."""
    y, p = np.asarray(y, int), np.asarray(p, float)
    n, prev = len(y), y.mean()
    out = {"threshold": [], "model": [], "treat_all": []}
    for t in thresholds:
        act = p >= t
        tp, fp = (act & (y == 1)).sum() / n, (act & (y == 0)).sum() / n
        w = t / (1 - t)
        out["threshold"].append(float(t))
        out["model"].append(float(tp - fp * w))
        out["treat_all"].append(float(prev - (1 - prev) * w))
    return out


def capture_at_capacity(y, p, share: float) -> dict:
    """Flag the top `share` of patients by risk: what share of all deaths do they include?"""
    y, p = np.asarray(y, int), np.asarray(p, float)
    k = max(1, int(round(share * len(y))))
    top = np.argsort(-p, kind="stable")[:k]
    return {"share_flagged": share, "deaths_captured": float(y[top].sum() / max(y.sum(), 1)),
            "death_rate_in_flagged": float(y[top].mean()), "risk_cutoff": float(p[top].min())}


# ---------------------------------------------------------------- odds ratios

def logistic_odds_ratios(X: np.ndarray, y, names: list[str]) -> list[dict]:
    """Adjusted odds ratios (unpenalised logistic regression with intercept) and 95% Wald CIs."""
    Xi = np.column_stack([np.ones(len(X)), np.asarray(X, float)])
    beta, cov = _fit_logistic(Xi, np.asarray(y, float))
    se = np.sqrt(np.diag(cov))
    out = []
    for k, name in enumerate(names, start=1):
        z = beta[k] / se[k]
        out.append({"feature": name, "odds_ratio": float(np.exp(beta[k])),
                    "ci": (float(np.exp(beta[k] - 1.96 * se[k])), float(np.exp(beta[k] + 1.96 * se[k]))),
                    "p": float(2 * stats.norm.sf(abs(z)))})
    return out


def holm(pvalues: dict[str, float]) -> dict[str, float]:
    """Holm-Bonferroni adjusted p-values."""
    items = sorted(pvalues.items(), key=lambda kv: kv[1])
    m, running, out = len(items), 0.0, {}
    for rank, (k, p) in enumerate(items):
        running = max(running, min(1.0, (m - rank) * p))
        out[k] = running
    return out
