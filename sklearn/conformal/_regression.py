# Authors: The scikit-learn developers
# SPDX-License-Identifier: BSD-3-Clause

"""Split conformal prediction for regression tasks."""

import warnings
from numbers import Real

import numpy as np

from sklearn.base import BaseEstimator, MetaEstimatorMixin, RegressorMixin, clone
from sklearn.ensemble import RandomForestRegressor
from sklearn.exceptions import DataConversionWarning
from sklearn.utils._param_validation import (
    HasMethods,
    Interval,
    StrOptions,
)
from sklearn.utils.validation import (
    _check_sample_weight,
    check_array,
    check_is_fitted,
    validate_data,
)

from ._base import (
    _check_alpha,
    _check_cv_argument,
    _check_coverage_params,
    _check_min_calibration_size,
    _check_n_samples,
    _check_prefit,
    _clip_sigma,
    _conformal_quantile,
    _ensure_csr,
    _iter_splits,
    _make_cv_splitter,
    _resolve_group_alpha,
)


class ConformalRegressor(MetaEstimatorMixin, RegressorMixin, BaseEstimator):
    """Model-agnostic conformal prediction interval for regression.

    Wraps any fitted or fit-able regression estimator and turns its point
    predictions into prediction intervals with a finite-sample marginal
    coverage guarantee under exchangeability of the calibration samples.

    Only callers that explicitly request intervals through
    :meth:`predict_interval` are affected; :meth:`predict` keeps returning the
    wrapped estimator's point predictions unchanged.

    The calibration data is kept separate from the data used to fit the
    underlying estimator:

    - with ``cv="prefit"``, the wrapped estimator is assumed already
      fitted and every sample passed to :meth:`fit` is used for calibration;
    - with an integer or splitter ``cv`` (default: 5-fold), the data passed
      to :meth:`fit` is split internally. An integer gives K-fold
      cross-conformal prediction, which fits and calibrates on the same data
      set at the price of losing the exact finite-sample guarantee (the
      guarantee becomes marginal, see :ref:`cross_conformal_note`).

    Parameters
    ----------
    estimator : regressor, default=None
        Any regressor implementing ``fit`` and ``predict``. If ``None``,
        :class:`~sklearn.ensemble.RandomForestRegressor` with 10 trees is
        used.

    alpha : float, default=0.1
        Target overall miscoverage level. Intervals are built to cover at
        least ``1 - alpha`` of the population. Must be strictly between 0
        and 1.

    method : {"absolute", "normalized"}, default="absolute"
        Nonconformity score used to build the intervals:

        - ``"absolute"``: ``|y - y_pred|``. The resulting interval has
          constant half-width, equal to one conformal quantile.
        - ``"normalized"``: ``|y - y_pred| / sigma`` where ``sigma`` is a
          local estimate of the target scale, learned with a second
          regressor on the absolute residuals. The resulting interval
          width adapts sample by sample.

    cv : "prefit", int or splitter, default=5
        Determines how fit and calibration data are separated:

        - integer >= 2 (default): K-fold cross-conformal prediction. The same
          data is used for fitting and calibration across folds; this loses
          the exact finite-sample guarantee (which becomes marginal) and
          reuses data, see :ref:`cross_conformal_note`.
        - ``"prefit"``: ``estimator`` must already be fitted; all data given
          to :meth:`fit` is calibration data. This gives the exact
          finite-sample guarantee.
        - splitter object (e.g.
          :class:`~sklearn.model_selection.ShuffleSplit` or
          :class:`~sklearn.model_selection.GroupShuffleSplit`): its first
          split defines the fit and calibration subsets.

    group_alpha : float, default=None
        Per-group (Mondrian) miscoverage level. If set, a separate conformal
        quantile is computed for every value of ``groups`` passed to
        :meth:`fit`, guaranteeing conditional coverage within each group.
        When this requirement conflicts with ``alpha`` (i.e.
        ``group_alpha > alpha``), per-group coverage takes precedence and a
        warning is raised.

    random_state : int, RandomState instance or None, default=None
        Controls the randomness of the default splitter and of the default
    estimator.

    Attributes
    ----------
    estimator_ : regressor
        The fitted estimator used for point predictions.

    quantile_ : float or dict
        Conformal quantile of the calibration nonconformity scores. A
        dictionary mapping group labels to quantiles when ``groups`` are
        passed to :meth:`fit`.

    n_calibration_ : int
        Number of calibration samples actually used.

    groups_ : ndarray or None
        Sorted unique group labels seen during calibration, or ``None``.

    sigma_estimator_ : regressor or None
        Fitted scale estimator when ``method="normalized"``.

    n_features_in_ : int
        Number of features seen during :meth:`fit`.

    feature_names_in_ : ndarray of shape (`n_features_in_`,)
        Names of features seen during :meth:`fit`. Defined only when `X`
        has feature names that are all strings.

    See Also
    --------
    ConformalClassifier : Conformal prediction sets for classification.

    References
    ----------
    Vovk, Gammerman and Shafer, "Algorithmic Learning in a Random World"
    (2005).

    Lei et al., "Distribution-Free Predictive Inference for Regression"
    (2018).

    Examples
    --------
    >>> from sklearn.datasets import make_regression
    >>> from sklearn.ensemble import RandomForestRegressor
    >>> from sklearn.conformal import ConformalRegressor
    >>> X, y = make_regression(n_samples=200, noise=1.0, random_state=0)
    >>> base = RandomForestRegressor(n_estimators=10, random_state=0)
    >>> conformal = ConformalRegressor(base, cv=5)
    >>> conformal.fit(X, y)
    ConformalRegressor(estimator=RandomForestRegressor(n_estimators=10,
                                                       random_state=0))
    >>> intervals = conformal.predict_interval(X[:5])
    >>> intervals.shape
    (5, 2)
    """

    _parameter_constraints = {
        "estimator": [HasMethods(["fit", "predict"]), None],
        "alpha": [Interval(Real, 0.0, 1.0, closed="neither")],
        "method": [StrOptions({"absolute", "normalized"})],
        "cv": [StrOptions({"prefit"}), "cv_object"],
        "group_alpha": [Interval(Real, 0.0, 1.0, closed="neither"), None],
        "random_state": ["random_state"],
    }

    def __init__(
        self,
        estimator=None,
        alpha=0.1,
        *,
        method="absolute",
        cv=5,
        group_alpha=None,
        random_state=None,
    ):
        self.estimator = estimator
        self.alpha = alpha
        self.method = method
        self.cv = cv
        self.group_alpha = group_alpha
        self.random_state = random_state

    def _get_estimator(self):
        if self.estimator is None:
            return RandomForestRegressor(
                n_estimators=10, random_state=self.random_state
            )
        return self.estimator

    def __sklearn_tags__(self):
        tags = super().__sklearn_tags__()
        estimator_tags = self._get_estimator().__sklearn_tags__()
        tags.input_tags.allow_nan = estimator_tags.input_tags.allow_nan
        tags.input_tags.sparse = estimator_tags.input_tags.sparse
        return tags

    @property
    def _accept_sparse(self):
        """Whether sparse inputs are accepted, mirroring the base estimator."""
        return self._get_estimator().__sklearn_tags__().input_tags.sparse

    @property
    def _allow_nan(self):
        return self._get_estimator().__sklearn_tags__().input_tags.allow_nan

    def _score(self, y_true, y_pred, sigma=None):
        residual = np.abs(y_true - y_pred)
        if self.method == "normalized":
            return residual / sigma
        return residual

    def _calibration_quantiles(self, scores, groups, sample_weight):
        """Compute overall or per-group conformal quantile(s)."""
        if groups is None:
            if self.group_alpha_ is not None:
                raise ValueError(
                    "group_alpha was set but no groups were passed to fit."
                )
            return _conformal_quantile(scores, self.alpha_, sample_weight)

        unique_groups = np.unique(groups)
        n_per_group = {
            group: int(np.sum(groups == group)) for group in unique_groups
        }
        level = (
            self.group_alpha_ if self.group_alpha_ is not None else self.alpha_
        )
        # Validate per-group sizes before computing any quantile so the user
        # gets an informative error instead of a numpy range error.
        _resolve_group_alpha(level, n_per_group)
        quantiles = {}
        for group in unique_groups:
            mask = groups == group
            w = sample_weight[mask] if sample_weight is not None else None
            quantiles[group] = _conformal_quantile(scores[mask], level, w)
        return quantiles

    def fit(self, X, y, *, sample_weight=None, groups=None):
        """Fit the conformal regressor.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training/calibration data. With ``cv="prefit"`` this is purely
            calibration data. Otherwise it is split into fit and
            calibration subsets.

        y : array-like of shape (n_samples,)
            Target values.

        sample_weight : array-like of shape (n_samples,), default=None
            Sample weights. Weights of calibration samples are used in the
            weighted conformal quantile; weights of fit samples are forwarded
            to the wrapped estimator when it accepts them.

        groups : array-like of shape (n_samples,), default=None
            Group labels. When provided, the default splitter keeps every
            group entirely on one side of the fit/calibration split and a
            separate quantile is fitted per group (Mondrian conformal
            prediction).

        Returns
        -------
        self : object
            Fitted estimator.
        """
        self._validate_params()
        cv = _check_cv_argument(self.cv)
        self.alpha_, self.group_alpha_ = _check_coverage_params(
            _check_alpha(self.alpha), self.group_alpha
        )
        n_samples = _check_n_samples(X)
        X, y = validate_data(
            self,
            X,
            y=y,
            accept_sparse=self._accept_sparse,
            ensure_all_finite=not self._allow_nan,
        )
        X = _ensure_csr(X)
        y = check_array(y, ensure_2d=False, dtype="numeric", input_name="y")
        if y.ndim != 1:
            raise ValueError(
                "ConformalRegressor supports single-output regression only; "
                f"got y with {y.ndim} dimensions."
            )
        if y.shape[0] != n_samples:
            raise ValueError(
                f"X and y must have the same number of samples; got {n_samples}"
                f" and {y.shape[0]}."
            )
        if sample_weight is not None:
            sample_weight = _check_sample_weight(sample_weight, X)
        if groups is not None:
            groups = np.asarray(groups)
            if groups.shape[0] != n_samples:
                raise ValueError(
                    "groups must have the same number of samples as X; got "
                    f"{groups.shape[0]} and {n_samples}."
                )
        self._fit_groups = groups

        if cv == "prefit":
            self.estimator_ = _check_prefit(self._get_estimator())
            scores, calib_groups, calib_weights, sigma_est = self._build_scores(
                X,
                y,
                fit_idx=None,
                calib_idx=np.arange(n_samples),
                sample_weight=sample_weight,
            )
        elif isinstance(cv, int):
            splitter = _make_cv_splitter(cv, groups, stratified=False)
            splits = _iter_splits(splitter, X, y, groups)
            scores, calib_groups, calib_weights, sigma_est = (
                self._cross_conformal(X, y, splits, sample_weight)
            )
        else:
            # Custom splitter or iterable of index pairs. A single split is
            # used directly; several splits pool scores cross-conformally.
            splits = _iter_splits(cv, X, y, groups)
            if len(splits) == 1:
                fit_idx, calib_idx = splits[0]
                scores, calib_groups, calib_weights, sigma_est = (
                    self._build_scores(
                        X,
                        y,
                        fit_idx=fit_idx,
                        calib_idx=calib_idx,
                        sample_weight=sample_weight,
                    )
                )
            else:
                scores, calib_groups, calib_weights, sigma_est = (
                    self._cross_conformal(X, y, splits, sample_weight)
                )

        self.n_calibration_ = scores.shape[0]
        if calib_groups is None:
            _check_min_calibration_size(self.n_calibration_, self.alpha_)
        self.quantile_ = self._calibration_quantiles(
            scores, calib_groups, calib_weights
        )
        self.groups_ = None if calib_groups is None else np.unique(calib_groups)
        self.sigma_estimator_ = sigma_est
        self._calibration_scores_ = scores
        self._calibration_groups_ = calib_groups
        self._calibration_weights_ = calib_weights
        return self

    def _fit_sigma(self, X_train, y_train, y_pred_train, fit_weights):
        """Fit the local scale regressor on absolute residuals."""
        sigma_estimator = clone(self._get_estimator())
        residuals = np.abs(y_train - y_pred_train)
        # A strictly positive floor avoids zero-scale scores on perfectly
        # fitted samples.
        target = np.maximum(residuals, 1e-6)
        fit_kwargs = {}
        if fit_weights is not None and self._accepts_sample_weight(
            sigma_estimator
        ):
            fit_kwargs["sample_weight"] = fit_weights
        sigma_estimator.fit(X_train, target, **fit_kwargs)
        return sigma_estimator

    @staticmethod
    def _accepts_sample_weight(estimator):
        from sklearn.utils.validation import has_fit_parameter

        return has_fit_parameter(estimator, "sample_weight")

    def _fit_estimator(self, estimator, X_train, y_train, fit_weights):
        fit_kwargs = {}
        if fit_weights is not None and self._accepts_sample_weight(estimator):
            fit_kwargs["sample_weight"] = fit_weights
        estimator.fit(X_train, y_train, **fit_kwargs)
        return estimator

    def _build_scores(
        self,
        X,
        y,
        fit_idx,
        calib_idx,
        sample_weight,
        fitted_estimator=None,
    ):
        """Compute nonconformity scores on the calibration fold."""
        if fitted_estimator is not None:
            estimator = fitted_estimator
        elif fit_idx is None:
            estimator = _check_prefit(self.estimator_)
        else:
            estimator = clone(self._get_estimator())
            fit_weights = (
                sample_weight[fit_idx] if sample_weight is not None else None
            )
            self._fit_estimator(
                estimator, X[fit_idx], y[fit_idx], fit_weights
            )

        X_calib = X[calib_idx]
        y_calib = y[calib_idx]
        y_pred_calib = estimator.predict(X_calib)

        sigma_est = None
        sigma_calib = None
        if self.method == "normalized":
            if fit_idx is None:
                # prefit mode: fit the scale estimator on calibration
                # residuals directly (no independent fit data available).
                sigma_est = clone(self._get_estimator())
                target = np.maximum(np.abs(y_calib - y_pred_calib), 1e-6)
                w = sample_weight[calib_idx] if sample_weight is not None else None
                kw = {}
                if w is not None and self._accepts_sample_weight(sigma_est):
                    kw["sample_weight"] = w
                sigma_est.fit(X_calib, target, **kw)
            else:
                y_pred_train = estimator.predict(X[fit_idx])
                fit_weights = (
                    sample_weight[fit_idx]
                    if sample_weight is not None
                    else None
                )
                sigma_est = self._fit_sigma(
                    X[fit_idx], y[fit_idx], y_pred_train, fit_weights
                )
            sigma_calib = _clip_sigma(sigma_est.predict(X_calib))

        scores = self._score(y_calib, y_pred_calib, sigma_calib)
        calib_groups = None
        if self._fit_groups is not None:
            calib_groups = self._fit_groups[calib_idx]
        calib_weights = (
            sample_weight[calib_idx] if sample_weight is not None else None
        )
        return scores, calib_groups, calib_weights, sigma_est

    def _cross_conformal(self, X, y, splits, sample_weight):
        """K-fold cross-conformal: pool out-of-fold calibration scores."""
        all_scores = []
        all_groups = []
        all_weights = []
        for fit_idx, calib_idx in splits:
            scores, calib_groups, calib_weights, _ = self._build_scores(
                X,
                y,
                fit_idx=fit_idx,
                calib_idx=calib_idx,
                sample_weight=sample_weight,
            )
            all_scores.append(scores)
            if calib_groups is not None:
                all_groups.append(calib_groups)
            if calib_weights is not None:
                all_weights.append(calib_weights)

        # Refit the point estimator on all data for prediction time.
        estimator = clone(self._get_estimator())
        fit_kwargs = {}
        if sample_weight is not None and self._accepts_sample_weight(estimator):
            fit_kwargs["sample_weight"] = sample_weight
        estimator.fit(X, y, **fit_kwargs)
        self.estimator_ = estimator

        sigma_est = None
        if self.method == "normalized":
            # One scale estimator on all data, using in-sample residuals of
            # the full-data estimator.
            y_pred = estimator.predict(X)
            sigma_est = self._fit_sigma(X, y, y_pred, sample_weight)

        scores = np.concatenate(all_scores)
        calib_groups = np.concatenate(all_groups) if all_groups else None
        calib_weights = np.concatenate(all_weights) if all_weights else None
        return scores, calib_groups, calib_weights, sigma_est

    def _quantile_for(self, groups):
        if self.groups_ is None:
            return np.full(groups.shape[0], self.quantile_)
        quantiles = np.empty(groups.shape[0], dtype=np.float64)
        for i, group in enumerate(groups):
            if group not in self.quantile_:
                # Unseen group at prediction time: fall back to the overall
                # quantile rather than failing.
                if self.groups_ is None:
                    quantiles[i] = self.quantile_
                else:
                    quantiles[i] = float(
                        np.mean(list(self.quantile_.values()))
                    )
            else:
                quantiles[i] = self.quantile_[group]
        return quantiles

    def _validate_predict_X(self, X):
        """Validate prediction-time X, including feature-name consistency."""
        X = validate_data(
            self,
            X,
            reset=False,
            accept_sparse=self._accept_sparse,
            ensure_all_finite=not self._allow_nan,
        )
        return _ensure_csr(X)

    def predict_interval(self, X, *, groups=None, side="both"):
        """Compute conformal prediction intervals.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Samples to predict intervals for.

        groups : array-like of shape (n_samples,), default=None
            Group labels used to select the per-group quantile. Required if
            ``groups`` were passed to :meth:`fit`.

        side : {"both", "lower", "upper"}, default="both"
            Return two-sided intervals, or one-sided lower / upper bounds.
            The open side of a one-sided interval is set to ``-inf`` /
            ``+inf``.

        Returns
        -------
        intervals : ndarray of shape (n_samples, 2)
            Columns are the lower and upper bounds. Bounds are ordered even
            when the target distribution is heavily skewed: if numerical
            computation would swap them, they are sorted per sample.
        """
        check_is_fitted(self, "estimator_")
        if side not in ("both", "lower", "upper"):
            raise ValueError(
                f"side must be 'both', 'lower' or 'upper', got {side!r}."
            )
        X = self._validate_predict_X(X)
        y_pred = self.estimator_.predict(X)

        if self.groups_ is not None:
            if groups is None:
                raise ValueError(
                    "groups were used at fit time and must be provided to "
                    "predict_interval as well."
                )
            groups = np.asarray(groups)
            if groups.shape[0] != X.shape[0]:
                raise ValueError(
                    "groups must have one label per sample; got "
                    f"{groups.shape[0]} labels for {X.shape[0]} samples."
                )
            quantiles = self._quantile_for(groups)
        elif groups is not None:
            warnings.warn(
                "groups passed to predict_interval were not used at fit "
                "time; they are ignored.",
                DataConversionWarning,
                stacklevel=2,
            )
            quantiles = np.full(X.shape[0], self.quantile_)
        else:
            quantiles = np.full(X.shape[0], self.quantile_)

        if self.method == "normalized":
            sigma = _clip_sigma(self.sigma_estimator_.predict(X))
            half_width = quantiles * sigma
        else:
            half_width = quantiles

        lower = y_pred - half_width
        upper = y_pred + half_width
        if side == "lower":
            upper = np.full_like(upper, np.inf)
        elif side == "upper":
            lower = np.full_like(lower, -np.inf)

        # Guard against swapped bounds on extremely skewed targets.
        swapped = lower > upper
        if np.any(swapped):
            lower[swapped], upper[swapped] = upper[swapped], lower[swapped]
        return np.column_stack([lower, upper])

    def predict(self, X, **params):
        """Point predictions of the wrapped estimator (unchanged behavior)."""
        check_is_fitted(self, "estimator_")
        X = self._validate_predict_X(X)
        return self.estimator_.predict(X, **params)
