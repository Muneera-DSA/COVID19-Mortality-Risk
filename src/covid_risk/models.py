"""Model factories and imbalance strategies.

Models (each must beat the previous one to justify its complexity):
  age_only   logistic regression on a spline of age: the "clinical rule of thumb" baseline
  logistic   logistic regression: age spline + sex, pregnancy, comorbidities, pneumonia, delay
  hgb        histogram gradient boosting (captures interactions such as age x kidney disease)

Imbalance strategies (deaths are a minority; tested, not assumed):
  none          fit on the data as it is
  class_weight  balanced class weights
  oversample    random duplication of deaths until classes are equal
  smote         SMOTE-NC style: interpolate continuous features, majority vote of the k
                neighbours for binary features (plain SMOTE would invent values such as
                "0.37 diabetic")
All resampling happens inside fit(), so in cross-validation it only ever touches the
training folds.
"""

from __future__ import annotations

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import NearestNeighbors
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import SplineTransformer, StandardScaler

from . import config

CONTINUOUS = ("age", "days_symptoms_to_visit")


def _spline_columns(features: list[str]) -> tuple[list[int], list[int]]:
    cont = [i for i, f in enumerate(features) if f in CONTINUOUS]
    other = [i for i in range(len(features)) if i not in cont]
    return cont, other


def age_only(seed: int = config.SEED, class_weight=None):
    return Pipeline([("spline", SplineTransformer(n_knots=6, degree=3)),
                     ("lr", LogisticRegression(C=1.0, max_iter=2000, class_weight=class_weight))])


def logistic(features: list[str], C: float = 1.0, seed: int = config.SEED, class_weight=None):
    cont, other = _spline_columns(features)
    pre = ColumnTransformer([("spline", SplineTransformer(n_knots=6, degree=3), cont),
                             ("rest", StandardScaler(), other)])
    return Pipeline([("pre", pre), ("lr", LogisticRegression(C=C, max_iter=3000, class_weight=class_weight))])


def hgb(learning_rate: float = 0.1, max_leaf_nodes: int = 31, min_samples_leaf: int = 200,
        l2_regularization: float = 1.0, seed: int = config.SEED, class_weight=None):
    return HistGradientBoostingClassifier(learning_rate=learning_rate, max_leaf_nodes=max_leaf_nodes,
                                          min_samples_leaf=min_samples_leaf, l2_regularization=l2_regularization,
                                          max_iter=500, early_stopping=True, validation_fraction=0.1,
                                          n_iter_no_change=20, class_weight=class_weight, random_state=seed)


# ------------------------------------------------------------------ resampling

def random_oversample(X: np.ndarray, y: np.ndarray, seed: int = 0):
    rng = np.random.default_rng(seed)
    cls, cnt = np.unique(y, return_counts=True)
    minority = cls[cnt.argmin()]
    extra = rng.choice(np.flatnonzero(y == minority), cnt.max() - cnt.min(), replace=True)
    return np.vstack([X, X[extra]]), np.concatenate([y, y[extra]])


def smote_nc(X: np.ndarray, y: np.ndarray, continuous: list[int], k: int = 5, seed: int = 0):
    rng = np.random.default_rng(seed)
    X = np.asarray(X, float)
    cls, cnt = np.unique(y, return_counts=True)
    minority = cls[cnt.argmin()]
    need = int(cnt.max() - cnt.min())
    Xm = X[y == minority]
    if need == 0 or len(Xm) <= k:
        return X, y
    scale = Xm.std(axis=0)
    scale[scale == 0] = 1
    nn = NearestNeighbors(n_neighbors=k + 1).fit(Xm / scale)
    neigh = nn.kneighbors(Xm / scale, return_distance=False)[:, 1:]
    i = rng.integers(0, len(Xm), need)
    j = neigh[i, rng.integers(0, k, need)]
    new = Xm[i].copy()
    lam = rng.random((need, 1))
    if continuous:
        new[:, continuous] = Xm[i][:, continuous] + lam * (Xm[j][:, continuous] - Xm[i][:, continuous])
    binary = [c for c in range(X.shape[1]) if c not in continuous]
    if binary:   # majority vote over the k neighbours (SMOTE-NC rule for nominal features)
        votes = Xm[neigh[i]][:, :, binary].mean(axis=1)
        new[:, binary] = (votes >= 0.5).astype(float)
    return np.vstack([X, new]), np.concatenate([y, np.full(need, minority)])


class Resampled(BaseEstimator, ClassifierMixin):
    """Wraps a classifier; resamples the training data inside fit()."""

    def __init__(self, estimator=None, method: str = "oversample", continuous: tuple[int, ...] = (), seed: int = 0):
        self.estimator = estimator
        self.method = method
        self.continuous = continuous
        self.seed = seed

    def fit(self, X, y):
        X, y = np.asarray(X, float), np.asarray(y)
        if self.method == "oversample":
            X, y = random_oversample(X, y, self.seed)
        elif self.method == "smote":
            X, y = smote_nc(X, y, list(self.continuous), seed=self.seed)
        else:
            raise ValueError(self.method)
        self.estimator_ = clone(self.estimator).fit(X, y)
        self.classes_ = self.estimator_.classes_
        return self

    def predict_proba(self, X):
        return self.estimator_.predict_proba(np.asarray(X, float))

    def predict(self, X):
        return self.estimator_.predict(np.asarray(X, float))


def build(name: str, features: list[str], imbalance: str = "none", params: dict | None = None, seed: int = config.SEED):
    """Model `name` with imbalance strategy `imbalance`."""
    params = params or {}
    cw = "balanced" if imbalance == "class_weight" else None
    if name == "age_only":
        base = age_only(seed, cw)
    elif name == "logistic":
        base = logistic(features, seed=seed, class_weight=cw, **params)
    elif name == "hgb":
        base = hgb(seed=seed, class_weight=cw, **params)
    else:
        raise ValueError(name)
    if imbalance in ("oversample", "smote"):
        cont, _ = _spline_columns(features)
        return Resampled(base, imbalance, tuple(cont), seed)
    return base
