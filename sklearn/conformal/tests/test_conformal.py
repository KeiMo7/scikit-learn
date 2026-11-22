# Authors: The scikit-learn developers
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for the split conformal prediction meta-estimators."""

import pickle

import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal

from sklearn.base import clone
from sklearn.conformal import ConformalClassifier, ConformalRegressor
from sklearn.datasets import make_classification, make_regression
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.model_selection import GroupKFold, GroupShuffleSplit, ShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.utils.estimator_checks import check_estimator
from sklearn.utils.validation import _num_samples

# Tolerance below the nominal level allowed in *empirical* realizations.
# Split conformal gives the exact guarantee in expectation; a single finite
# test realization can dip slightly below, so tests allow a small margin.
COVERAGE_TOL = 0.05


@pytest.fixture
def regression_data():
    X, y = make_regression(n_samples=3000, n_features=15, noise=2.0, random_state=0)
    return X, y


@pytest.fixture
def classification_data():
    X, y = make_classification(
        n_samples=3000, n_classes=3, n_informative=6, random_state=1
    )
    return X, y


# --------------------------------------------------------------------- #
# Estimator consistency checks
# --------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "estimator",
    [
        ConformalRegressor(Ridge(), cv=3),
        ConformalRegressor(
            Ridge(), method="normalized", cv=3, random_state=0
        ),
        ConformalClassifier(LogisticRegression(max_iter=500), cv=3),
        ConformalClassifier(
            LogisticRegression(max_iter=500), method="aps", cv=3,
            random_state=0,
        ),
    ],
)
def test_estimator_checks(estimator):
    """The meta-estimators pass the library's estimator consistency suite."""
    check_estimator(estimator)


def test_clone_get_set_params():
    cr = ConformalRegressor(Ridge(alpha=2.0), alpha=0.2, cv=4)
    cr_clone = clone(cr)
    assert cr_clone.alpha == 0.2
    assert cr_clone.cv == 4
    assert cr_clone.estimator.alpha == 2.0
    cr_clone.set_params(alpha=0.05)
    assert cr_clone.get_params()["alpha"] == 0.05

    cc = ConformalClassifier(LogisticRegression(C=0.5), method="aps")
    cc_clone = clone(cc)
    assert cc_clone.method == "aps"
    assert cc_clone.estimator.C == 0.5


# --------------------------------------------------------------------- #
# Regression: coverage guarantee
# --------------------------------------------------------------------- #
def test_regression_marginal_coverage_prefit(regression_data):
    X, y = regression_data
    base = Ridge().fit(X[:1500], y[:1500])
    cr = ConformalRegressor(base, alpha=0.1).fit(X[1500:2500], y[1500:2500])
    intervals = cr.predict_interval(X[2500:])
    coverage = np.mean(
        (y[2500:] >= intervals[:, 0]) & (y[2500:] <= intervals[:, 1])
    )
    assert coverage >= 0.9 - COVERAGE_TOL
    assert intervals.shape == (500, 2)


def test_regression_coverage_several_levels(regression_data):
    X, y = regression_data
    base = Ridge().fit(X[:1000], y[:1000])
    for alpha in (0.05, 0.1, 0.2):
        cr = ConformalRegressor(base, alpha=alpha).fit(
            X[1000:2000], y[1000:2000]
        )
        intervals = cr.predict_interval(X[2000:])
        coverage = np.mean(
            (y[2000:] >= intervals[:, 0]) & (y[2000:] <= intervals[:, 1])
        )
        assert coverage >= 1 - alpha - COVERAGE_TOL


def test_one_sided_intervals(regression_data):
    X, y = regression_data
    base = Ridge().fit(X[:1500], y[:1500])
    cr = ConformalRegressor(base, alpha=0.1).fit(X[1500:2500], y[1500:2500])
    lower = cr.predict_interval(X[2500:], side="lower")
    upper = cr.predict_interval(X[2500:], side="upper")
    assert np.isinf(lower[:, 1]).all()
    assert np.isinf(upper[:, 0]).all()
    assert np.mean(y[2500:] >= lower[:, 0]) >= 0.9 - COVERAGE_TOL
    assert np.mean(y[2500:] <= upper[:, 1]) >= 0.9 - COVERAGE_TOL


def test_normalized_intervals_have_adaptive_width(regression_data):
    X, y = regression_data
    cr = ConformalRegressor(
        Ridge(), method="normalized", cv=5, random_state=0
    ).fit(X[:2000], y[:2000])
    intervals = cr.predict_interval(X[2000:])
    widths = intervals[:, 1] - intervals[:, 0]
    # Adaptive intervals must not collapse to a constant width.
    assert np.std(widths) > 0.0


def test_point_prediction_unchanged(regression_data):
    X, y = regression_data
    base = Ridge().fit(X[:1500], y[:1500])
    cr = ConformalRegressor(base, alpha=0.1, cv="prefit").fit(
        X[1500:], y[1500:]
    )
    assert_allclose(cr.predict(X[2500:]), base.predict(X[2500:]))


def test_cross_conformal_fit(regression_data):
    X, y = regression_data
    # Same data used for fitting and calibration through K folds.
    cr = ConformalRegressor(Ridge(), alpha=0.1, cv=5).fit(X[:2000], y[:2000])
    assert cr.n_calibration_ == 2000
    intervals = cr.predict_interval(X[2000:])
    coverage = np.mean(
        (y[2000:] >= intervals[:, 0]) & (y[2000:] <= intervals[:, 1])
    )
    assert coverage >= 0.9 - COVERAGE_TOL


def test_splitter_cv_respects_single_split(regression_data):
    X, y = regression_data
    splitter = ShuffleSplit(n_splits=1, test_size=0.3, random_state=0)
    cr = ConformalRegressor(Ridge(), alpha=0.1, cv=splitter).fit(X, y)
    assert cr.n_calibration_ == int(0.3 * X.shape[0])


# --------------------------------------------------------------------- #
# Grouped data
# --------------------------------------------------------------------- #
def _make_grouped_data(n_entities=40, n_per=25, seed=0):
    rng = np.random.RandomState(seed)
    X = rng.normal(size=(n_entities * n_per, 5))
    groups = np.repeat(np.arange(n_entities), n_per)
    y = X[:, 0] * 2 + rng.normal(scale=0.5, size=X.shape[0]) + groups * 0.1
    return X, y, groups


def test_group_split_no_entity_on_both_sides():
    X, y, groups = _make_grouped_data()
    splitter = GroupShuffleSplit(n_splits=1, test_size=0.4, random_state=0)
    fit_idx, calib_idx = next(splitter.split(X, y, groups=groups))
    # The same entity must never appear on both sides.
    assert not set(groups[fit_idx]) & set(groups[calib_idx])

    base = Ridge().fit(X[fit_idx], y[fit_idx])
    cr = ConformalRegressor(base, alpha=0.1, group_alpha=0.1).fit(
        X[calib_idx], y[calib_idx], groups=groups[calib_idx]
    )
    assert isinstance(cr.quantile_, dict)
    assert set(cr.quantile_) == set(np.unique(groups[calib_idx]))
    intervals = cr.predict_interval(X[calib_idx], groups=groups[calib_idx])
    assert intervals.shape == (calib_idx.size, 2)


def test_group_kfold_keeps_entities_whole():
    X, y, groups = _make_grouped_data()
    for train_idx, test_idx in GroupKFold(n_splits=4).split(X, y, groups):
        assert not set(groups[train_idx]) & set(groups[test_idx])
    cr = ConformalRegressor(Ridge(), alpha=0.1, cv=GroupKFold(n_splits=4))
    cr.fit(X, y, groups=groups)
    assert isinstance(cr.quantile_, dict)


def test_group_alpha_conflict_warns():
    X, y, groups = _make_grouped_data()
    with pytest.warns(UserWarning, match="takes precedence"):
        ConformalRegressor(Ridge(), alpha=0.05, group_alpha=0.2, cv=3).fit(
            X, y, groups=groups
        )


def test_group_alpha_without_groups_raises(regression_data):
    X, y = regression_data
    with pytest.raises(ValueError, match="group_alpha"):
        ConformalRegressor(Ridge(), alpha=0.1, group_alpha=0.1, cv=3).fit(
            X, y
        )


def test_predict_requires_groups_when_fitted_with_them():
    X, y, groups = _make_grouped_data()
    cr = ConformalRegressor(Ridge(), alpha=0.1, cv=GroupKFold(4))
    cr.fit(X, y, groups=groups)
    with pytest.raises(ValueError, match="groups"):
        cr.predict_interval(X[:10])


def test_small_group_raises_even_without_group_alpha():
    """A group with too few calibration samples is rejected at the overall
    alpha level because per-group quantiles use that level."""
    rng = np.random.RandomState(0)
    # Two groups: one well populated, one with only 2 calibration samples.
    X = rng.normal(size=(60, 4))
    groups = np.array(["big"] * 50 + ["tiny"] * 10)
    y = X[:, 0] + rng.normal(size=60)
    splitter = ShuffleSplit(n_splits=1, test_size=0.5, random_state=0)
    fit_idx, calib_idx = next(splitter.split(X, y))
    # Force the tiny group to be under-represented on the calibration side.
    base = Ridge().fit(X[fit_idx], y[fit_idx])
    with pytest.raises(ValueError, match="at least"):
        ConformalRegressor(base, alpha=0.1, cv="prefit").fit(
            X[calib_idx], y[calib_idx], groups=groups[calib_idx]
        )


# --------------------------------------------------------------------- #
# Sample weights
# --------------------------------------------------------------------- #
def test_weighted_coverage(regression_data):
    X, y = regression_data
    rng = np.random.RandomState(0)
    base = Ridge().fit(X[:1000], y[:1000])
    weights = rng.uniform(0.5, 3.0, size=1000)
    cr = ConformalRegressor(base, alpha=0.1).fit(
        X[1000:2000], y[1000:2000], sample_weight=weights
    )
    intervals = cr.predict_interval(X[1000:2000])
    mask = (y[1000:2000] >= intervals[:, 0]) & (
        y[1000:2000] <= intervals[:, 1]
    )
    weighted_coverage = np.sum(weights[mask]) / np.sum(weights)
    assert weighted_coverage >= 0.9 - COVERAGE_TOL


def test_negative_weight_raises(regression_data):
    X, y = regression_data
    base = Ridge().fit(X[:1000], y[:1000])
    bad_weights = np.ones(1000)
    bad_weights[0] = -1.0
    with pytest.raises(ValueError, match="non-negative"):
        ConformalRegressor(base, alpha=0.1, cv="prefit").fit(
            X[1000:2000], y[1000:2000], sample_weight=bad_weights
        )


# --------------------------------------------------------------------- #
# Classification: prediction sets
# --------------------------------------------------------------------- #
def test_classifier_lac_coverage(classification_data):
    X, y = classification_data
    base = LogisticRegression(max_iter=1000).fit(X[:1500], y[:1500])
    cc = ConformalClassifier(base, alpha=0.1).fit(X[1500:2250], y[1500:2250])
    sets = cc.predict_set(X[2250:])
    coverage = np.mean([y[2250 + i] in sets[i] for i in range(750)])
    assert coverage >= 0.9 - COVERAGE_TOL
    assert all(len(s) >= 1 for s in sets)
    assert all(np.isin(s, cc.classes_).all() for s in sets)


def test_classifier_aps_coverage_lower_bound(classification_data):
    X, y = classification_data
    base = LogisticRegression(max_iter=1000).fit(X[:1500], y[:1500])
    cc = ConformalClassifier(
        base, alpha=0.1, method="aps", random_state=0
    ).fit(X[1500:2250], y[1500:2250])
    sets = cc.predict_set(X[2250:])
    coverage = np.mean([y[2250 + i] in sets[i] for i in range(750)])
    # APS guarantees the lower bound even when discreteness makes it
    # conservative.
    assert coverage >= 0.9 - COVERAGE_TOL


def test_binary_classification_sets():
    X, y = make_classification(n_samples=2000, random_state=2)
    base = LogisticRegression().fit(X[:1000], y[:1000])
    cc = ConformalClassifier(base, alpha=0.1).fit(X[1000:1500], y[1000:1500])
    sets = cc.predict_set(X[1500:])
    assert all(len(s) >= 1 for s in sets)
    assert all(set(s).issubset({0, 1}) for s in sets)


def test_empty_set_degenerates_to_top_label(classification_data):
    X, y = classification_data
    # An estimator fitted on almost no data produces extreme outputs; the
    # prediction set must never be empty.
    extreme = LogisticRegression().fit(X[:5], y[:5])
    cc = ConformalClassifier(extreme, alpha=0.1).fit(
        X[1500:1600], y[1500:1600]
    )
    sets = cc.predict_set(X[1600:1620])
    assert all(len(s) >= 1 for s in sets)


def test_full_set_when_nobody_looks_likely(classification_data):
    X, y = classification_data
    base = LogisticRegression(max_iter=1000).fit(X[:1500], y[:1500])
    # A very small alpha forces the quantile up to the top of its range, so
    # some samples collect every class instead of an empty set.
    cc = ConformalClassifier(base, alpha=0.01).fit(X[1500:2250], y[1500:2250])
    sets = cc.predict_set(X[2250:2280])
    assert any(len(s) == cc.classes_.size for s in sets)


def test_set_size_adjustable_via_alpha(classification_data):
    X, y = classification_data
    base = LogisticRegression(max_iter=1000).fit(X[:1500], y[:1500])
    small_alpha = ConformalClassifier(base, alpha=0.02).fit(
        X[1500:2250], y[1500:2250]
    )
    large_alpha = ConformalClassifier(base, alpha=0.3).fit(
        X[1500:2250], y[1500:2250]
    )
    size_small = np.mean(
        [len(s) for s in small_alpha.predict_set(X[2250:])]
    )
    size_large = np.mean(
        [len(s) for s in large_alpha.predict_set(X[2250:])]
    )
    assert size_small >= size_large


def test_classifier_point_and_proba_unchanged(classification_data):
    X, y = classification_data
    base = LogisticRegression(max_iter=1000).fit(X[:1500], y[:1500])
    cc = ConformalClassifier(base, alpha=0.1, cv="prefit").fit(
        X[1500:2250], y[1500:2250]
    )
    assert_array_equal(cc.predict(X[2250:]), base.predict(X[2250:]))
    assert_allclose(
        cc.predict_proba(X[2250:]), base.predict_proba(X[2250:])
    )


# --------------------------------------------------------------------- #
# Persistence
# --------------------------------------------------------------------- #
def test_pickle_regression_intervals_bitwise_identical(regression_data):
    X, y = regression_data
    base = Ridge().fit(X[:1500], y[:1500])
    cr = ConformalRegressor(base, alpha=0.1).fit(X[1500:], y[1500:])
    restored = pickle.loads(pickle.dumps(cr))
    assert_array_equal(
        cr.predict_interval(X[2500:]), restored.predict_interval(X[2500:])
    )


def test_pickle_classification_sets_identical(classification_data):
    X, y = classification_data
    base = LogisticRegression(max_iter=1000).fit(X[:1500], y[:1500])
    cc = ConformalClassifier(
        base, alpha=0.1, method="aps", random_state=0
    ).fit(X[1500:2250], y[1500:2250])
    restored = pickle.loads(pickle.dumps(cc))
    before = cc.predict_set(X[2250:])
    after = restored.predict_set(X[2250:])
    assert all(np.array_equal(a, b) for a, b in zip(before, after))


# --------------------------------------------------------------------- #
# Pipelines and arbitrary estimators
# --------------------------------------------------------------------- #
def test_pipeline_regressor(regression_data):
    X, y = regression_data
    pipe = Pipeline([("scaler", StandardScaler()), ("ridge", Ridge())])
    cr = ConformalRegressor(pipe, alpha=0.1, cv=5).fit(X[:2000], y[:2000])
    intervals = cr.predict_interval(X[2000:])
    coverage = np.mean(
        (y[2000:] >= intervals[:, 0]) & (y[2000:] <= intervals[:, 1])
    )
    assert coverage >= 0.9 - COVERAGE_TOL


def test_pipeline_classifier(classification_data):
    X, y = classification_data
    pipe = Pipeline(
        [("scaler", StandardScaler()), ("lr", LogisticRegression(max_iter=1000))]
    )
    cc = ConformalClassifier(pipe, alpha=0.1, cv=5).fit(X[:2000], y[:2000])
    sets = cc.predict_set(X[2000:])
    assert all(len(s) >= 1 for s in sets)


def test_gradient_boosting_regressor(regression_data):
    X, y = regression_data
    cr = ConformalRegressor(
        HistGradientBoostingRegressor(max_iter=50, random_state=0),
        alpha=0.1,
        cv=5,
    ).fit(X[:2000], y[:2000])
    intervals = cr.predict_interval(X[2000:])
    coverage = np.mean(
        (y[2000:] >= intervals[:, 0]) & (y[2000:] <= intervals[:, 1])
    )
    assert coverage >= 0.9 - COVERAGE_TOL


# --------------------------------------------------------------------- #
# Error handling and degenerate inputs
# --------------------------------------------------------------------- #
def test_calibration_set_too_small(regression_data):
    X, y = regression_data
    base = Ridge().fit(X[:100], y[:100])
    # alpha=0.05 needs at least ceil(1/0.05) - 1 = 19 samples.
    with pytest.raises(ValueError, match="at least 19"):
        ConformalRegressor(base, alpha=0.05).fit(X[:5], y[:5])


@pytest.mark.parametrize("bad_alpha", [0.0, 1.0, -0.1, 1.5, "0.1"])
def test_invalid_alpha(regression_data, bad_alpha):
    X, y = regression_data
    with pytest.raises((ValueError, TypeError)):
        ConformalRegressor(Ridge(), alpha=bad_alpha).fit(X[:50], y[:50])


@pytest.mark.parametrize("bad_cv", [1, 0, "not_prefit"])
def test_invalid_cv(regression_data, bad_cv):
    X, y = regression_data
    with pytest.raises((ValueError, TypeError)):
        ConformalRegressor(Ridge(), cv=bad_cv).fit(X[:50], y[:50])


def test_prefit_requires_fitted_estimator(regression_data):
    X, y = regression_data
    with pytest.raises(Exception):
        ConformalRegressor(Ridge(), cv="prefit").fit(X[:50], y[:50])


def test_predict_before_fit_raises(regression_data):
    X, _ = regression_data
    with pytest.raises(Exception):
        ConformalRegressor(Ridge()).predict_interval(X[:5])
    with pytest.raises(Exception):
        ConformalClassifier(LogisticRegression()).predict_set(X[:5])


def test_skewed_target_bounds_stay_ordered():
    rng = np.random.RandomState(0)
    X, _ = make_regression(n_samples=2000, noise=0.0, random_state=5)
    y = np.exp(rng.normal(size=2000) + X[:, 0])
    cr = ConformalRegressor(Ridge(), alpha=0.1, cv=5).fit(X, y)
    intervals = cr.predict_interval(X[:50])
    assert np.all(intervals[:, 0] <= intervals[:, 1])


def test_invalid_side_raises(regression_data):
    X, y = regression_data
    base = Ridge().fit(X[:1500], y[:1500])
    cr = ConformalRegressor(base, alpha=0.1).fit(X[1500:], y[1500:])
    with pytest.raises(ValueError, match="side"):
        cr.predict_interval(X[:5], side="middle")


def test_calibration_classes_must_match_prefit(classification_data):
    X, y = classification_data
    base = LogisticRegression(max_iter=1000).fit(X[:1500], y[:1500])
    # Relabel so the calibration labels differ from the fitted classes.
    y_relabeled = y.copy() + 10
    with pytest.raises(ValueError, match="classes"):
        ConformalClassifier(base, alpha=0.1, cv="prefit").fit(
            X[1500:2250], y_relabeled[1500:2250]
        )


def test_prefit_with_different_class_ordering(classification_data):
    """Probability columns stay aligned when the prefit estimator's class
    order differs from the sorted order of the calibration labels."""
    from sklearn.base import BaseEstimator, ClassifierMixin

    X, y = classification_data

    class ReorderedClassifier(ClassifierMixin, BaseEstimator):
        """LogisticRegression wrapper that exposes classes_ in reverse order."""

        def __init__(self):
            self._inner = LogisticRegression(max_iter=1000)

        def fit(self, X, y):
            self._inner.fit(X, y)
            self.classes_ = self._inner.classes_[::-1]
            return self

        def predict(self, X):
            return self._inner.predict(X)

        def predict_proba(self, X):
            # Reverse columns to match the reversed classes_ ordering.
            return self._inner.predict_proba(X)[:, ::-1]

    base = ReorderedClassifier().fit(X[:1500], y[:1500])
    assert list(base.classes_) == list(np.sort(np.unique(y)))[::-1]
    cc = ConformalClassifier(base, alpha=0.1).fit(
        X[1500:2250], y[1500:2250]
    )
    sets = cc.predict_set(X[2250:])
    coverage = np.mean([y[2250 + i] in sets[i] for i in range(750)])
    assert coverage >= 0.9 - COVERAGE_TOL


def test_decision_function_classifier():
    """A classifier without predict_proba still works via decision_function."""
    from sklearn.svm import LinearSVC

    X, y = make_classification(n_samples=2000, random_state=3)
    base = LinearSVC().fit(X[:1000], y[:1000])
    cc = ConformalClassifier(base, alpha=0.1).fit(X[1000:1500], y[1000:1500])
    sets = cc.predict_set(X[1500:])
    assert all(len(s) >= 1 for s in sets)


def test_multiclass_decision_function_classifier():
    """decision_function with one column per class produces valid sets."""
    from sklearn.svm import LinearSVC

    X, y = make_classification(
        n_samples=3000, n_classes=4, n_informative=8, random_state=5
    )
    base = LinearSVC().fit(X[:1500], y[:1500])
    cc = ConformalClassifier(
        base, alpha=0.1, method="aps", random_state=0
    ).fit(X[1500:2250], y[1500:2250])
    sets = cc.predict_set(X[2250:])
    assert all(len(s) >= 1 for s in sets)
    assert all(np.isin(s, cc.classes_).all() for s in sets)
    coverage = np.mean([y[2250 + i] in sets[i] for i in range(750)])
    assert coverage >= 0.9 - COVERAGE_TOL


def test_estimator_without_response_method_raises(classification_data):
    from sklearn.base import BaseEstimator, ClassifierMixin

    X, y = classification_data

    class PointOnly(ClassifierMixin, BaseEstimator):
        def fit(self, X, y):
            self.classes_ = np.unique(y)
            return self

        def predict(self, X):
            return np.zeros(_num_samples(X), dtype=int)

    with pytest.raises(AttributeError, match="predict_proba or decision_function"):
        ConformalClassifier(PointOnly(), alpha=0.1, cv=3).fit(X[:100], y[:100])
