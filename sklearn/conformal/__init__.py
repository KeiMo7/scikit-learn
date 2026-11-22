# Authors: The scikit-learn developers
# SPDX-License-Identifier: BSD-3-Clause

"""Model-agnostic conformal prediction with finite-sample coverage guarantees.

This module provides meta-estimators that wrap any regressor or classifier
and turn its point predictions into prediction intervals
(:class:`ConformalRegressor`) or candidate label sets
(:class:`ConformalClassifier`) with a distribution-free coverage guarantee
under exchangeability of the calibration samples.
"""

from ._classification import ConformalClassifier
from ._regression import ConformalRegressor

__all__ = ["ConformalClassifier", "ConformalRegressor"]
