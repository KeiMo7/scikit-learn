# Authors: The scikit-learn developers
# SPDX-License-Identifier: BSD-3-Clause

"""Split conformal prediction for classification tasks."""

import warnings
from numbers import Real

import numpy as np

from sklearn.base import (
    BaseEstimator,
    ClassifierMixin,
    MetaEstimatorMixin,
    clone,
)
from sklearn.exceptions import DataConversionWarning
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.utils._param_validation import (
    HasMethods,
    Interval,
    StrOptions,
)
from sklearn.utils.multiclass import check_classification_targets
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
    _conformal_quantile,
    _ensure_csr,
    _iter_splits,
    _make_cv_splitter,
    _resolve_group_alpha,
)

_RESPONSE_METHODS = ("predict_proba", "decision_function")


class ConformalClassifier(MetaEstimatorMixin, ClassifierMixin, BaseEstimator):
    """Model-agnostic conformal prediction sets for classification.

    Wraps any classifier and turns its probability (or decision function)
    outputs into per-sample candidate label sets with a finite-sample
    marginal coverage guarantee under exchangeability of the calibration
    samples: the true label belongs to the predicted set with probability
    at least ``1 - alpha``.

    Two nonconformity scores are supported:

    - ``"lac"`` (Least Ambiguous set-valued Classifier, also called adaptive
      prediction sets): the nonconformity score of a sample is one minus the
      probability assigned to its true label. Sets are built by keeping
      labels whose probability exceeds the conformal threshold.
    - ``"aps"`` (Adaptive Prediction Sets, Romano et al. 2020): labels are
      sorted by decreasing probability and added until their cumulative
      probability reaches the conformal threshold. APS tends to produce
      smaller sets than LAC while keeping the same coverage guarantee.

    When a set would be empty, it deterministically falls back to the single
    most likely label instead of raising an error. When no label is likely
    enough (very low probabilities), the full set of classes is returned.

    Parameters
    ----------
    estimator : classifier, default=None
        Any classifier implementing ``fit`` and either ``predict_proba`` or
        ``decision_function``. If ``None``, :class:`~LogisticRegression` is
        used.

    alpha : float, default=0.1
        Target overall miscoverage level. Must be strictly between 0 and 1.

    method : {"lac", "aps"}, default="lac"
        Nonconformity score used to build the prediction sets.

    cv : "prefit", int or splitter, default=5
        Determines how fit and calibration data are separated:

        - integer >= 2 (default): K-fold cross-conformal prediction. The same
          data is used for fitting and calibration across folds, giving a
          marginal rather than exact finite-sample guarantee.
        - ``"prefit"``: ``estimator`` must already be fitted; all data given
          to :meth:`fit` is calibration data. This gives the exact
          finite-sample guarantee.
        - splitter object (e.g.
          :class:`~sklearn.model_selection.StratifiedKFold`): its folds are
          used as calibration folds.

    group_alpha : float, default=None
        Per-group (Mondrian) miscoverage level. If set, a separate conformal
        quantile is computed for every value of ``groups`` passed to
        :meth:`fit`. When this requirement conflicts with ``alpha``,
        per-group coverage takes precedence and a warning is raised.

    random_state : int, RandomState instance or None, default=None
        Controls the randomness of the default estimator.

    Attributes
    ----------
    estimator_ : classifier
        The fitted classifier used for point predictions.

    quantile_ : float or dict
        Conformal quantile of the calibration nonconformity scores. A
        dictionary mapping group labels to quantiles when ``groups`` are
        passed to :meth:`fit`.

    n_calibration_ : int
        Number of calibration samples actually used.

    classes_ : ndarray of shape (n_classes,)
        The class labels known to the classifier.

    groups_ : ndarray or None
        Sorted unique group labels seen during calibration, or ``None``.

    n_features_in_ : int
        Number of features seen during :meth:`fit`.

    feature_names_in_ : ndarray of shape (`n_features_in_`,)
        Names of features seen during :meth:`fit`. Defined only when `X`
        has feature names that are all strings.

    See Also
    --------
    ConformalRegressor : Conformal prediction intervals for regression.

    References
    ----------
    Angelopoulos and Bates, "A Gentle Introduction to Conformal Prediction
    and Distribution-Free Uncertainty Quantification" (2023).

    Romano, Sesia and Candès, "Classification with Valid and Adaptive
    Coverage" (2020).

    Examples
    --------
    >>> from sklearn.datasets import make_classification
    >>> from sklearn.linear_model import LogisticRegression
    >>> from sklearn.conformal import ConformalClassifier
    >>> X, y = make_classification(n_samples=300, n_classes=3,
    ...                            n_informative=5, random_state=0)
    >>> clf = LogisticRegression(max_iter=1000)
    >>> conformal = ConformalClassifier(clf, cv=5)
    >>> conformal.fit(X, y)
    ConformalClassifier(estimator=LogisticRegression(max_iter=1000))
    >>> sets = conformal.predict_set(X[:5])
    >>> len(sets)
    5
    """

    _parameter_constraints = {
        "estimator": [
            HasMethods(["fit", "predict"]),
            None,
        ],
        "alpha": [Interval(Real, 0.0, 1.0, closed="neither")],
        "method": [StrOptions({"lac", "aps"})],
        "cv": [StrOptions({"prefit"}), "cv_object"],
        "group_alpha": [Interval(Real, 0.0, 1.0, closed="neither"), None],
        "random_state": ["random_state"],
    }

    def __init__(
        self,
        estimator=None,
        alpha=0.1,
        *,
        method="lac",
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
            return LogisticRegression(max_iter=1000)
        return self.estimator

    def __sklearn_tags__(self):
        tags = super().__sklearn_tags__()
        estimator_tags = self._get_estimator().__sklearn_tags__()
        tags.input_tags.allow_nan = estimator_tags.input_tags.allow_nan
        tags.input_tags.sparse = estimator_tags.input_tags.sparse
        tags.classifier_tags.multi_class = (
            estimator_tags.classifier_tags.multi_class
        )
        return tags

    @property
    def _accept_sparse(self):
        """Whether sparse inputs are accepted, mirroring the base estimator."""
        return self._get_estimator().__sklearn_tags__().input_tags.sparse

    @property
    def _allow_nan(self):
        return self._get_estimator().__sklearn_tags__().input_tags.allow_nan

    def _check_response_method(self, estimator):
        for name in _RESPONSE_METHODS:
            if hasattr(estimator, name):
                return name
        raise AttributeError(
            f"{estimator.__class__.__name__} must implement predict_proba or "
            "decision_function to be wrapped by ConformalClassifier."
        )

    def _get_scores(self, estimator, X, y_encoded, response_method,
                    random_state=None):
        """Nonconformity scores of labeled calibration samples."""
        proba = self._get_proba(estimator, X, response_method)
        n_samples = proba.shape[0]
        if self.method == "lac":
            true_proba = proba[np.arange(n_samples), y_encoded]
            return 1.0 - true_proba
        # Randomized APS (Romano et al. 2020). Labels are sorted by
        # decreasing probability and the score of a sample is the cumulative
        # probability up to the true label, with a uniform random fraction of
        # the true label's own probability subtracted from the running total.
        # The randomization breaks the score discreteness that otherwise makes
        # non-randomized APS systematically conservative. Randomness is drawn
        # from ``random_state`` so results are reproducible.
        order = np.argsort(-proba, axis=1, kind="mergesort")
        sorted_proba = np.take_along_axis(proba, order, axis=1)
        cumulative = np.cumsum(sorted_proba, axis=1)
        ranks = np.argmax(order == y_encoded[:, None], axis=1)
        true_proba = proba[np.arange(n_samples), y_encoded]
        before = cumulative[np.arange(n_samples), ranks] - true_proba
        u = random_state.uniform(size=n_samples)
        scores = before + u * true_proba
        # Clip numerical overshoot so scores stay in [0, 1].
        return np.clip(scores, 0.0, 1.0)

    def _get_proba(self, estimator, X, response_method):
        """Return a probability-like matrix aligned with estimator.classes_."""
        raw = getattr(estimator, response_method)(X)
        if response_method == "predict_proba":
            proba = np.asarray(raw, dtype=np.float64)
        else:
            # decision_function: 1 column in the binary case, n_classes in
            # the multiclass case. Turn it into a softmax-normalized matrix.
            raw = np.asarray(raw, dtype=np.float64)
            if raw.ndim == 1:
                # Binary case: map to two columns via a logistic transform.
                from scipy.special import expit

                p1 = expit(raw)
                proba = np.column_stack([1.0 - p1, p1])
            else:
                shifted = raw - raw.max(axis=1, keepdims=True)
                exp = np.exp(shifted)
                proba = exp / exp.sum(axis=1, keepdims=True)
        return proba

    def fit(self, X, y, *, sample_weight=None, groups=None):
        """Fit the conformal classifier.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training/calibration data.

        y : array-like of shape (n_samples,)
            Class labels.

        sample_weight : array-like of shape (n_samples,), default=None
            Sample weights. Calibration weights enter the weighted conformal
            quantile; fit weights are forwarded to the wrapped estimator.

        groups : array-like of shape (n_samples,), default=None
            Group labels used both to split the data (no group crosses the
            fit/calibration boundary) and for per-group (Mondrian)
            quantiles.

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
        # Reject continuous targets up front with the standard library
        # message before LabelEncoder would silently treat them as classes.
        check_classification_targets(y)
        X, y = validate_data(
            self,
            X,
            y=y,
            accept_sparse=self._accept_sparse,
            ensure_all_finite=not self._allow_nan,
        )
        X = _ensure_csr(X)
        y = check_array(y, ensure_2d=False, input_name="y", dtype=None)
        if y.shape[0] != n_samples:
            raise ValueError(
                f"X and y must have the same number of samples; got {n_samples}"
                f" and {y.shape[0]}."
            )
        label_encoder = LabelEncoder()
        y_encoded = label_encoder.fit_transform(y)
        fitted_classes = label_encoder.classes_
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
        from sklearn.utils import check_random_state

        rng = check_random_state(self.random_state)

        if cv == "prefit":
            self.estimator_ = _check_prefit(self._get_estimator())
            self._response_method = self._check_response_method(self.estimator_)
            # Adopt the fitted estimator's class ordering so probability
            # columns and encoded labels stay aligned, even when it differs
            # from the sorted order LabelEncoder would produce.
            self._check_classes(self.estimator_, fitted_classes)
            self.classes_ = np.asarray(self.estimator_.classes_)
            # Re-encode the calibration labels using that ordering.
            class_to_index = {c: i for i, c in enumerate(self.classes_)}
            y_encoded = np.array(
                [class_to_index[label] for label in y], dtype=np.int64
            )
            scores, calib_groups, calib_weights = self._build_scores(
                X, y_encoded, fit_idx=None, calib_idx=np.arange(n_samples),
                sample_weight=sample_weight, rng=rng,
            )
        elif isinstance(cv, int):
            self.classes_ = fitted_classes
            splitter = _make_cv_splitter(cv, groups, stratified=True)
            splits = _iter_splits(splitter, X, y_encoded, groups)
            scores, calib_groups, calib_weights = self._cross_conformal(
                X, y_encoded, splits, sample_weight, rng
            )
        else:
            self.classes_ = fitted_classes
            # Custom splitter or iterable of index pairs.
            splits = _iter_splits(cv, X, y_encoded, groups)
            if len(splits) == 1:
                fit_idx, calib_idx = splits[0]
                self.estimator_ = clone(self._get_estimator())
                self._fit_estimator(
                    self.estimator_, X[fit_idx], y_encoded[fit_idx],
                    sample_weight[fit_idx] if sample_weight is not None else None,
                )
                self._response_method = self._check_response_method(
                    self.estimator_
                )
                scores, calib_groups, calib_weights = self._build_scores(
                    X, y_encoded, fit_idx=fit_idx, calib_idx=calib_idx,
                    sample_weight=sample_weight, rng=rng,
                )
            else:
                scores, calib_groups, calib_weights = self._cross_conformal(
                    X, y_encoded, splits, sample_weight, rng
                )

        self.n_calibration_ = scores.shape[0]
        if calib_groups is None:
            _check_min_calibration_size(self.n_calibration_, self.alpha_)
        self.quantile_ = self._calibration_quantiles(
            scores, calib_groups, calib_weights
        )
        self.groups_ = None if calib_groups is None else np.unique(calib_groups)
        self._calibration_scores_ = scores
        self._calibration_groups_ = calib_groups
        self._calibration_weights_ = calib_weights
        return self

    def _check_classes(self, estimator, fitted_classes):
        # Same label set is required; ordering may differ and is handled by
        # re-encoding the calibration labels.
        if set(estimator.classes_) != set(fitted_classes):
            raise ValueError(
                "The classes learned by the prefit estimator "
                f"({list(estimator.classes_)!r}) do not match the labels "
                f"passed to fit ({list(fitted_classes)!r})."
            )

    @staticmethod
    def _accepts_sample_weight(estimator):
        from sklearn.utils.validation import has_fit_parameter

        return has_fit_parameter(estimator, "sample_weight")

    def _fit_estimator(self, estimator, X_train, y_train, fit_weights):
        kwargs = {}
        if fit_weights is not None and self._accepts_sample_weight(estimator):
            kwargs["sample_weight"] = fit_weights
        estimator.fit(X_train, y_train, **kwargs)

    def _build_scores(self, X, y_encoded, fit_idx, calib_idx, sample_weight,
                      rng):
        if fit_idx is None:
            estimator = self.estimator_
        else:
            estimator = clone(self._get_estimator())
            fit_weights = (
                sample_weight[fit_idx] if sample_weight is not None else None
            )
            self._fit_estimator(
                estimator, X[fit_idx], y_encoded[fit_idx], fit_weights
            )
            self.estimator_ = estimator
            self._response_method = self._check_response_method(estimator)
        scores = self._get_scores(
            estimator, X[calib_idx], y_encoded[calib_idx],
            self._response_method, random_state=rng,
        )
        calib_groups = (
            self._fit_groups[calib_idx]
            if self._fit_groups is not None
            else None
        )
        calib_weights = (
            sample_weight[calib_idx] if sample_weight is not None else None
        )
        return scores, calib_groups, calib_weights

    def _cross_conformal(self, X, y_encoded, splits, sample_weight, rng):
        all_scores, all_groups, all_weights = [], [], []
        for fit_idx, calib_idx in splits:
            scores, calib_groups, calib_weights = self._build_scores(
                X, y_encoded, fit_idx, calib_idx, sample_weight, rng
            )
            all_scores.append(scores)
            if calib_groups is not None:
                all_groups.append(calib_groups)
            if calib_weights is not None:
                all_weights.append(calib_weights)

        # Refit on all data for prediction time.
        estimator = clone(self._get_estimator())
        kwargs = {}
        if sample_weight is not None and self._accepts_sample_weight(estimator):
            kwargs["sample_weight"] = sample_weight
        estimator.fit(X, y_encoded, **kwargs)
        self.estimator_ = estimator
        self._response_method = self._check_response_method(estimator)

        scores = np.concatenate(all_scores)
        calib_groups = np.concatenate(all_groups) if all_groups else None
        calib_weights = np.concatenate(all_weights) if all_weights else None
        return scores, calib_groups, calib_weights

    def _calibration_quantiles(self, scores, groups, sample_weight):
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

    def _quantile_vector(self, groups, n):
        if groups is None:
            return np.full(n, self.quantile_, dtype=np.float64)
        out = np.empty(n, dtype=np.float64)
        for i, group in enumerate(groups):
            if group in self.quantile_:
                out[i] = self.quantile_[group]
            else:
                # Unseen group at prediction time: use the mean of stored
                # per-group quantiles, never fail.
                out[i] = float(np.mean(list(self.quantile_.values())))
        return out

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

    def predict_set(self, X, *, groups=None):
        """Compute conformal candidate label sets.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Samples to build prediction sets for.

        groups : array-like of shape (n_samples,), default=None
            Group labels used to select the per-group quantile. Required if
            ``groups`` were passed to :meth:`fit`.

        Returns
        -------
        prediction_sets : ndarray of shape (n_samples,)
            Object array. Each element is a 1d ndarray of labels from
            :attr:`classes_`. Sets always contain at least one label: an
            empty set degenerates to the single most likely label.
        """
        check_is_fitted(self, "estimator_")
        X = self._validate_predict_X(X)
        if self.groups_ is not None:
            if groups is None:
                raise ValueError(
                    "groups were used at fit time and must be provided to "
                    "predict_set as well."
                )
            groups = np.asarray(groups)
            if groups.shape[0] != X.shape[0]:
                raise ValueError(
                    "groups must have one label per sample; got "
                    f"{groups.shape[0]} labels for {X.shape[0]} samples."
                )
        elif groups is not None:
            warnings.warn(
                "groups passed to predict_set were not used at fit time; "
                "they are ignored.",
                DataConversionWarning,
                stacklevel=2,
            )
            groups = None

        proba = self._get_proba(
            self.estimator_, X, self._response_method
        )
        quantiles = self._quantile_vector(groups, X.shape[0])
        n_samples, n_classes = proba.shape
        sets = np.empty(n_samples, dtype=object)

        if self.method == "lac":
            for i in range(n_samples):
                keep = np.flatnonzero(1.0 - proba[i] <= quantiles[i])
                if keep.size == 0:
                    # Degenerate case: fall back to the most likely label.
                    keep = np.array([int(np.argmax(proba[i]))])
                sets[i] = self.classes_[keep]
        else:
            order = np.argsort(-proba, axis=1, kind="mergesort")
            sorted_proba = np.take_along_axis(proba, order, axis=1)
            # Randomized APS inclusion rule, mirroring the calibration score
            # ``L_k = cumulative_before_k + U * p_k`` (Romano et al. 2020).
            # The set keeps labels up to and including the first rank whose
            # randomized cumulative value reaches the quantile.
            from sklearn.utils import check_random_state

            u = check_random_state(self.random_state).uniform(size=n_samples)
            cumulative_before = np.cumsum(sorted_proba, axis=1) - sorted_proba
            randomized = cumulative_before + u[:, None] * sorted_proba
            for i in range(n_samples):
                hits = np.flatnonzero(randomized[i] >= quantiles[i])
                n_keep = int(hits[0]) + 1 if hits.size else n_classes
                n_keep = max(1, min(n_keep, n_classes))
                idx = order[i, :n_keep]
                sets[i] = self.classes_[idx]
        return sets

    def predict(self, X):
        """Point predictions of the wrapped classifier (unchanged behavior)."""
        check_is_fitted(self, "estimator_")
        X = self._validate_predict_X(X)
        return self.estimator_.predict(X)

    def predict_proba(self, X):
        """Probability estimates of the wrapped classifier (unchanged)."""
        check_is_fitted(self, "estimator_")
        X = self._validate_predict_X(X)
        return self.estimator_.predict_proba(X)
