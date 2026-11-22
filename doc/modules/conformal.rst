.. _conformal_prediction:

===================
Conformal prediction
===================

.. currentmodule:: sklearn.conformal

Conformal prediction turns the point predictions of *any* fitted estimator
into prediction sets with a distribution-free coverage guarantee: for
regression the set is a prediction interval, for classification it is a set
of candidate labels. Unlike quantile regression or quantile loss, the
guarantee does not depend on the underlying model or on distributional
assumptions: it only requires that the calibration samples are exchangeable
with the future samples.

For a user-chosen miscoverage level ``alpha``, split conformal prediction
constructs a set :math:`C(X)` such that

.. math::

    P\big(Y \in C(X)\big) \geq 1 - \alpha

for a fresh test pair :math:`(X, Y)`. The inequality is exact in finite
samples (not just asymptotic) when the calibration data is independent of
the data used to fit the wrapped estimator.

The two meta-estimators, :class:`ConformalRegressor` and
:class:`ConformalClassifier`, wrap an arbitrary estimator without changing
its point predictions: :meth:`ConformalRegressor.predict` and
:meth:`ConformalClassifier.predict` keep delegating to the wrapped
estimator. Intervals and sets are only produced by the dedicated
:meth:`ConformalRegressor.predict_interval` and
:meth:`ConformalClassifier.predict_set` methods.

.. _conformal_split:

Separating fit and calibration data
------------------------------------

The calibration scores must be computed on data that the wrapped estimator
has not been fitted on, otherwise the scores are optimistically biased and
the coverage guarantee no longer holds. Two workflows are supported.

**Prefit workflow.** Fit the base estimator yourself, then pass held-out
calibration data to the conformal estimator with ``cv="prefit"``::

    >>> from sklearn.datasets import make_regression
    >>> from sklearn.linear_model import Ridge
    >>> from sklearn.conformal import ConformalRegressor
    >>> X, y = make_regression(n_samples=500, random_state=0)
    >>> base = Ridge().fit(X[:300], y[:300])
    >>> conformal = ConformalRegressor(base, alpha=0.1, cv="prefit")
    >>> conformal.fit(X[300:], y[300:])  # calibration only
    ConformalRegressor(cv='prefit', estimator=Ridge())
    >>> intervals = conformal.predict_interval(X[:3])
This workflow gives the exact finite-sample guarantee and keeps the
calibration set fully under the caller's control.

**Cross-conformal workflow (default).** With the default integer ``cv=5``
(or any integer of at least 2), the same data is partitioned into K folds:
each fold in turn serves as calibration data for an estimator fitted on the
other folds, and the out-of-fold nonconformity scores are pooled. A final
estimator is then refitted on all the data for prediction time::

    >>> conformal = ConformalRegressor(Ridge(), alpha=0.1)
    >>> conformal.fit(X, y)  # fitting and calibration share the same data

This reuses the data efficiently but loses the exact guarantee: the
coverage becomes marginal rather than exact in finite samples, which is the
source of bias to be aware of when fit and calibration use the same data.

A splitter object can also be passed as ``cv``. With a single split
(e.g. :class:`~sklearn.model_selection.ShuffleSplit`) it behaves like the
prefit workflow while handling the split internally; with several splits it
pools the scores cross-conformally.

.. _conformal_regression:

Regression intervals
--------------------

:class:`ConformalRegressor` supports two nonconformity scores:

- ``method="absolute"`` (default): the score is the absolute residual
  :math:`|y - \hat y|`. The interval is
  :math:`[\hat y - \hat q,\ \hat y + \hat q]` where :math:`\hat q` is the
  conformal quantile of the calibration absolute residuals. The width is
  constant across samples.
- ``method="normalized"``: the score is the residual divided by a local
  estimate of the target scale, itself learned by a second regressor on the
  absolute residuals. The resulting half-width :math:`\hat q\,\hat\sigma(x)`
  adapts sample by sample, widening where the target is more variable.

The conformal quantile uses the finite-sample rank
:math:`\lceil (n+1)(1-\alpha) \rceil / n`, which is what produces the
``1 - alpha`` lower bound. One-sided bounds are available through the
``side`` parameter of :meth:`~ConformalRegressor.predict_interval`; the open
side is set to infinity.

Sample weights
~~~~~~~~~~~~~~

When calibration samples carry weights, they enter a weighted conformal
quantile based on the weighted empirical distribution with an
effective-sample-size finite-sample correction. The weighted guarantee is
exact when the weights define a likelihood ratio between exchangeable
distributions; otherwise it should be read as an approximate guarantee.

.. _conformal_classification:

Classification sets
-------------------

:class:`ConformalClassifier` builds a candidate label set per sample. It
needs a classifier exposing ``predict_proba`` or ``decision_function`` and
supports two scores:

- ``method="lac"`` (default, Least Ambiguous set-valued Classifier): the
  nonconformity score is one minus the probability assigned to the true
  label. At test time, every label whose nonconformity does not exceed the
  conformal quantile is kept. This is the simplest and most stable rule.
- ``method="aps"`` (Adaptive Prediction Sets, Romano et al. 2020): labels
  are ranked by decreasing probability and added until the cumulative
  probability reaches the threshold, with a uniform randomization that keeps
  the rule valid despite the discreteness of cumulative probabilities. The
  randomization is controlled by ``random_state`` for reproducibility. APS
  can produce smaller sets than LAC but, being discrete, is often more
  conservative than the nominal level.

The size of the sets is tuned through ``alpha``: a smaller ``alpha``
(higher requested coverage) yields larger sets. Two degeneracies are handled
by construction:

- a set that would be empty is replaced by the single most likely label, so
  :meth:`~ConformalClassifier.predict_set` never raises and always returns at
  least one label;
- when no label reaches the threshold, the full set of classes is returned,
  explicitly expressing that the sample does not look like any class.

.. _conformal_groups:

Grouped data and conditional coverage
-------------------------------------

Passing ``groups`` to :meth:`fit` has two effects.

1. **Group-aware splitting.** The internal splitter becomes group-aware
   (:class:`~sklearn.model_selection.GroupKFold` for cross-conformal, or the
   group-aware splitter passed as ``cv``), so the same group can never appear
   on both the fit and the calibration sides. This is required when rows of
   the same entity are correlated.
2. **Mondrian (per-group) quantiles.** A separate conformal quantile is
   fitted for every group, giving a coverage guarantee conditional on each
   group rather than only on average. This is requested through
   ``group_alpha``.

Per-group coverage is the stronger requirement. When it conflicts with the
overall level (``group_alpha > alpha``), the per-group level takes
precedence and a warning is raised. Groups that contain too few calibration
samples to compute the requested finite-sample quantile raise an error that
states the minimum required size.

.. _conformal_persistence:

Persistence and arbitrary estimators
------------------------------------

Both meta-estimators are regular scikit-learn estimators: they support
:func:`~sklearn.base.clone`, ``get_params``/``set_params``, pickle
persistence, and pass :func:`~sklearn.utils.estimator_checks.check_estimator`.
Because the calibration output is computed once at :meth:`fit` time and
stored as plain numpy arrays, the intervals and sets produced after a
pickle round-trip are bitwise identical to the ones produced before it.

The wrapped estimator can be any regressor or classifier, including
:class:`~sklearn.ensemble.HistGradientBoostingRegressor`, tree ensembles, or
an entire :class:`~sklearn.pipeline.Pipeline`: the meta-estimator delegates
input validation tags (sparse input, missing values) to the wrapped
estimator.

References
----------

.. [1] Vovk, Gammerman and Shafer, "Algorithmic Learning in a Random
       World", 2005.
.. [2] Angelopoulos and Bates, "A Gentle Introduction to Conformal
       Prediction and Distribution-Free Uncertainty Quantification", 2023.
.. [3] Lei et al., "Distribution-Free Predictive Inference for Regression",
       2018.
.. [4] Romano, Sesia and Candès, "Classification with Valid and Adaptive
       Coverage", 2020.
