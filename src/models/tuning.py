from __future__ import annotations

import warnings

import numpy as np
from sklearn.base import clone
from sklearn.model_selection import GridSearchCV, GroupKFold


def fit_with_tuning(est, grid, X, y, groups, inner_splits: int = 3):
    """Tune with GroupKFold BY BATTERY (never mixes a battery across inner folds), refit on all
    training rows.  Falls back to default hyper-parameters if there are too few batteries."""
    n_groups = len(np.unique(groups))
    est = clone(est)
    if not grid or n_groups < 3:
        return est.fit(X, y), {}
    gs = GridSearchCV(est, grid, cv=GroupKFold(n_splits=min(inner_splits, n_groups)),
                      scoring="neg_root_mean_squared_error", n_jobs=1, refit=True)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        gs.fit(X, y, groups=groups)
    return gs.best_estimator_, {k.replace("model__", ""): v for k, v in gs.best_params_.items()}
