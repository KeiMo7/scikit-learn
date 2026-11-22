# Authors: The scikit-learn developers
# SPDX-License-Identifier: BSD-3-Clause

"""Shared utilities for split conformal prediction.

Conformal prediction is a model-agnostic framework that turns point
predictions of any fitted estimator into prediction sets (intervals for
regression, label sets for classification) with a finite-sample marginal
coverage guarantee under exchangeability.

References
----------
Vovk, Gammerman and Shafer, "Algorithmic Learning in a Random World" (2005).

Angelopoulos and Bates, "A Gentle Introduction to Conformal Prediction and
Distribution-Free Uncertainty Quantification" (2023).
"""

import warnings
from numbers import Integral

import numpy as np
from scipy.sparse import issparse

from sklearn.model_selection import GroupKFold, KFold, StratifiedKFold
from sklearn.utils.validation import _num_samples


def _ensure_csr(X):
    """Return ``X`` as a CSR sparse array when sparse, unchanged otherwise.

    Only CSR/CSC sparse formats support row slicing; the estimator checks
    feed many other formats (dok, lil, dia, bsr, coo, ...), so every sparse
    input is normalized to CSR at the fit/predict entry points.
    """
    if issparse(X):
        return X.tocsr()
    return X


def _check_alpha(alpha):
    """Validate a miscoverage level and return it as a float."""
    if isinstance(alpha, bool) or not isinstance(
        alpha, (int, float, np.integer, np.floating)
    ):
        raise TypeError(
            "alpha must be a real number in the open interval (0, 1), got "
            f"{type(alpha).__name__} instead."
        )
    alpha = float(alpha)
    if not 0.0 < alpha < 1.0:
        raise ValueError(
            f"alpha must be strictly between 0 and 1, got {alpha} instead."
        )
    return alpha


def _check_coverage_params(alpha, group_alpha=None):
    """Validate overall and per-group miscoverage levels.

    Per-group (Mondrian) coverage at level ``group_alpha`` is the stronger
    requirement: it implies overall coverage at the same level but not vice
    versa. When both are given and they conflict (``group_alpha > alpha``),
    the per-group level wins and a warning is raised.
    """
    alpha = _check_alpha(alpha)
    if group_alpha is not None:
        group_alpha = _check_alpha(group_alpha)
        if group_alpha > alpha + 1e-12:
            warnings.warn(
                "Per-group coverage (group_alpha="
                f"{group_alpha}) is weaker than overall coverage (alpha="
                f"{alpha}); the two requirements conflict. Per-group "
                "coverage takes precedence: group_alpha is used to build the "
                "intervals for every group, which also guarantees overall "
                "coverage at level group_alpha.",
                stacklevel=3,
            )
            alpha = group_alpha
    return alpha, group_alpha


def _check_min_calibration_size(n_calibration, alpha):
    """Check that the calibration set is large enough for the chosen level.

    Split conformal prediction at miscoverage level ``alpha`` needs the
    finite-sample quantile at rank ``ceil((1 - alpha) * (n + 1))`` to be
    computable, which requires ``n + 1 >= 1 / alpha`` (equivalently
    ``n >= ceil(1 / alpha) - 1``). Raise an informative error otherwise.
    """
    n_min = int(np.ceil(1.0 / alpha)) - 1
    if n_calibration < n_min:
        raise ValueError(
            f"At miscoverage level alpha={alpha}, at least {n_min} "
            "calibration samples are needed so that the finite-sample "
            "quantile at rank ceil((1 - alpha) * (n + 1)) exists, but only "
            f"{n_calibration} calibration samples were provided. Use a "
            "larger calibration set or a larger value of alpha."
        )


def _conformal_quantile(scores, alpha, sample_weight=None):
    """Finite-sample (weighted) conformal quantile of ``scores``.

    For the unweighted case this is the empirical quantile at level

        ceil((n + 1) * (1 - alpha)) / n

    which yields the standard split-conformal guarantee
    ``P(Y in C(X)) >= 1 - alpha`` under exchangeability.

    For weighted scores the calibration distribution is taken as the
    weighted empirical distribution (weights normalized to sum 1) and the
    same rank-based correction is applied on top of the effective sample
    size. Weighted conformal prediction gives an exchangeability guarantee
    only when weights define a likelihood ratio between exchangeable
    distributions; otherwise coverage holds approximately.

    Parameters
    ----------
    scores : ndarray of shape (n_samples,)
        Nonconformity scores computed on calibration data.

    alpha : float
        Miscoverage level, strictly between 0 and 1.

    sample_weight : ndarray of shape (n_samples,), default=None
        Non-negative calibration weights.

    Returns
    -------
    quantile : float
        The conformal quantile.
    """
    scores = np.asarray(scores, dtype=np.float64)
    n = scores.shape[0]
    if sample_weight is None:
        level = np.ceil((n + 1) * (1.0 - alpha)) / n
        return float(np.quantile(scores, level, method="higher"))

    weights = np.asarray(sample_weight, dtype=np.float64)
    if np.any(weights < 0):
        raise ValueError("sample_weight must be non-negative.")
    total = weights.sum()
    if total <= 0:
        raise ValueError(
            "The sum of calibration sample_weight is zero; cannot estimate "
            "a weighted conformal quantile."
        )
    order = np.argsort(scores, kind="mergesort")
    sorted_scores = scores[order]
    sorted_weights = weights[order] / total
    cumulative = np.cumsum(sorted_weights)
    # Effective sample size, used as the "n" of the finite sample correction.
    n_eff = float((weights.sum() ** 2) / np.sum(weights**2))
    target = ((n_eff + 1.0) / n_eff) * (1.0 - alpha)
    target = min(target, 1.0)
    idx = int(np.searchsorted(cumulative, target, side="left"))
    idx = min(idx, n - 1)
    return float(sorted_scores[idx])


def _make_cv_splitter(cv, groups, stratified):
    """Build the K-fold splitter used by cross-conformal prediction.

    Group-aware when ``groups`` is not None, stratified for classifiers when
    possible.
    """
    if groups is not None:
        return GroupKFold(n_splits=cv)
    if stratified:
        return StratifiedKFold(n_splits=cv, shuffle=False)
    return KFold(n_splits=cv, shuffle=False)


def _iter_splits(splitter, X, y, groups):
    """Yield (train_idx, calib_idx) index arrays from a splitter or iterable."""
    if hasattr(splitter, "split"):
        if groups is not None:
            return list(splitter.split(X, y, groups=groups))
        return list(splitter.split(X, y))
    # Already an iterable of (train, test) index pairs.
    return list(splitter)


def _check_prefit(estimator):
    """Validate the estimator argument used in ``cv="prefit"`` mode."""
    from sklearn.utils.validation import check_is_fitted

    check_is_fitted(estimator)
    return estimator


def _resolve_group_alpha(group_alpha, n_calib_per_group):
    """Validate per-group calibration sizes for Mondrian conformal prediction.

    Parameters
    ----------
    group_alpha : float
        Per-group miscoverage level.

    n_calib_per_group : dict
        Mapping from group label to its calibration sample count.
    """
    n_min = int(np.ceil(1.0 / group_alpha)) - 1
    too_small = {
        group: n for group, n in n_calib_per_group.items() if n < n_min
    }
    if too_small:
        details = ", ".join(
            f"group {group!r}: {n} sample(s)" for group, n in too_small.items()
        )
        raise ValueError(
            f"At per-group miscoverage level group_alpha={group_alpha}, "
            f"every group needs at least {n_min} calibration samples. The "
            f"following groups are too small ({details}). Increase the "
            "calibration set, merge small groups, or use a larger "
            "group_alpha."
        )


def _clip_sigma(sigma):
    """Clip an internal local-scale prediction to a strictly positive floor.

    The scale regressor is trained on floored absolute residuals but its
    predictions can still be negative on shifted inputs; clipping keeps the
    normalized nonconformity score well-defined instead of rejecting the
    whole fit (which would make the estimator fail on ordinary negative
    feature values).
    """
    sigma = np.asarray(sigma, dtype=np.float64)
    return np.maximum(sigma, 1e-6)


def _check_n_samples(X, min_n=2):
    n = _num_samples(X)
    if n < min_n:
        raise ValueError(
            f"Found array with {n} sample(s) while a minimum of {min_n} is "
            "required to split the data into a fit and a calibration set."
        )
    return n


def _check_cv_argument(cv):
    """Validate the ``cv`` parameter shared by both conformal meta-estimators.

    Accepted values:

    - ``"prefit"``: the wrapped estimator is already fitted and all data
      passed to ``fit`` is used as calibration data;
    - an integer >= 2: K-fold (cross-)conformal prediction;
    - a splitter object exposing ``split``: used directly. With an integer
      number of folds every fold is taken as calibration once; with a
      single-split splitter the first split is used.
    """
    if isinstance(cv, str):
        if cv != "prefit":
            raise ValueError(
                f"The only accepted string for cv is 'prefit', got {cv!r}."
            )
        return cv
    if isinstance(cv, Integral) and not isinstance(cv, bool):
        if cv < 2:
            raise ValueError(
                "cv must be at least 2 for cross-conformal prediction, got "
                f"{int(cv)}."
            )
        return int(cv)
    if hasattr(cv, "split"):
        return cv
    # An iterable of (train_idx, test_idx) index arrays, as produced by a
    # splitter or accepted by cross_val_predict.
    if not isinstance(cv, (str, bytes)) and hasattr(cv, "__iter__"):
        return cv
    raise TypeError(
        "cv must be 'prefit', an integer >= 2, a cross-validation splitter "
        "with a split method, or an iterable of (train, test) index pairs, "
        f"got {type(cv).__name__} instead."
    )
