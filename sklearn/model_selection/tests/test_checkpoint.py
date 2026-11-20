"""Tests for the staged, resumable search checkpoint machinery."""

# Authors: The scikit-learn developers
# SPDX-License-Identifier: BSD-3-Clause

import os
import warnings
from unittest.mock import patch

import numpy as np
import pytest

from sklearn.datasets import load_iris
from sklearn.exceptions import (
    CheckpointCorruptedError,
    CheckpointCorruptedWarning,
    CheckpointMismatchError,
)
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.model_selection._checkpoint import (
    _pack_records,
    _unpack_records,
    candidate_key,
)
from sklearn.svm import SVC

X, y = load_iris(return_X_y=True)
PARAM_GRID = {"C": [0.1, 1, 10], "kernel": ["linear", "rbf"]}


class FlakySVC(SVC):
    """SVC that fails for one specific parameter setting."""

    def fit(self, X, y, **kwargs):
        if self.kernel == "linear" and self.C == 1.0:
            raise RuntimeError("boom")
        return super().fit(X, y, **kwargs)


def _assert_results_match(reference, staged):
    """cv_results_ must agree apart from wall-clock timing fields."""
    timing = {
        "mean_fit_time",
        "std_fit_time",
        "mean_score_time",
        "std_score_time",
    }
    for key in reference.cv_results_:
        if key in timing:
            continue
        expected = reference.cv_results_[key]
        result = staged.cv_results_[key]
        if isinstance(expected, np.ndarray) and expected.dtype.kind == "f":
            np.testing.assert_allclose(result, expected, equal_nan=True)
        else:
            assert list(result) == list(expected)
    assert staged.best_params_ == reference.best_params_


def test_checkpoint_param_defaults_unchanged():
    # The feature must be opt-in: defaults keep the historical behaviour.
    search = GridSearchCV(SVC(), PARAM_GRID)
    assert search.stage_size is None
    assert search.checkpoint is None
    assert not hasattr(search, "checkpoint_")


def test_staged_search_matches_one_shot(tmp_path):
    """Normal staged run gives the same answer as a single-shot run."""
    reference = GridSearchCV(SVC(), PARAM_GRID, cv=3).fit(X, y)
    staged = GridSearchCV(
        SVC(), PARAM_GRID, cv=3, stage_size=2, checkpoint=tmp_path / "ckpt"
    ).fit(X, y)

    assert len(staged.stage_summaries_) == 3
    assert all(summary["n_candidates"] == 2 for summary in staged.stage_summaries_)
    assert staged.n_evaluated_ == 6
    _assert_results_match(reference, staged)


def test_resume_after_interruption(tmp_path):
    """A crash mid-search leaves a consistent state; resume reuses results."""
    ckpt = tmp_path / "ckpt"
    from sklearn.model_selection import _search as search_module

    original = search_module._fit_and_score
    n_calls = {"count": 0}

    def crashing(*args, **kwargs):
        # stage_size=2 with 3 splits: the 7th fit belongs to stage 1.
        n_calls["count"] += 1
        if n_calls["count"] == 7:
            raise KeyboardInterrupt("simulated power loss")
        return original(*args, **kwargs)

    with patch.object(search_module, "_fit_and_score", crashing):
        with pytest.raises(KeyboardInterrupt):
            GridSearchCV(
                SVC(), PARAM_GRID, cv=3, stage_size=2, checkpoint=ckpt
            ).fit(X, y)

    # Atomic writes: no temp files, only the completed first stage survives.
    files = os.listdir(ckpt)
    assert not any(name.endswith(".tmp") for name in files)
    assert os.path.exists(ckpt / "segment_000000.pkl")
    assert not os.path.exists(ckpt / "segment_000001.pkl")

    resumed = GridSearchCV(
        SVC(), PARAM_GRID, cv=3, stage_size=2, checkpoint=ckpt
    ).fit(X, y)

    assert resumed.n_evaluated_ == 6
    # The first stage was not re-evaluated: its summary is untouched.
    assert resumed.stage_summaries_[0]["n_candidates"] == 2
    reference = GridSearchCV(SVC(), PARAM_GRID, cv=3).fit(X, y)
    _assert_results_match(reference, resumed)


def test_resume_with_enlarged_candidate_space(tmp_path):
    """New candidates are picked up on resume while old results are reused."""
    ckpt = tmp_path / "ckpt"
    small_grid = {"C": [0.1, 1], "kernel": ["linear", "rbf"]}
    first = GridSearchCV(
        SVC(), small_grid, cv=3, stage_size=2, checkpoint=ckpt
    ).fit(X, y)
    assert first.n_evaluated_ == 4

    resumed = GridSearchCV(
        SVC(), PARAM_GRID, cv=3, stage_size=2, checkpoint=ckpt
    ).fit(X, y)
    assert resumed.n_evaluated_ == 6  # 4 reused + 2 newly evaluated

    reference = GridSearchCV(SVC(), PARAM_GRID, cv=3).fit(X, y)
    _assert_results_match(reference, resumed)


def test_shrinking_candidate_space_is_rejected(tmp_path):
    """Only enlargement is allowed: removing an evaluated candidate is refused."""
    ckpt = tmp_path / "ckpt"
    GridSearchCV(SVC(), PARAM_GRID, cv=3, stage_size=2, checkpoint=ckpt).fit(X, y)

    smaller_grid = {"C": [0.1, 1], "kernel": ["linear", "rbf"]}
    with pytest.raises(CheckpointMismatchError, match="enlarging"):
        GridSearchCV(
            SVC(), smaller_grid, cv=3, stage_size=2, checkpoint=ckpt
        ).fit(X, y)


def test_data_change_is_detected_and_rejected(tmp_path):
    """Changing the training data between runs must abort the resume."""
    ckpt = tmp_path / "ckpt"
    GridSearchCV(
        SVC(), PARAM_GRID, cv=3, stage_size=2, checkpoint=ckpt
    ).fit(X, y)

    X_changed = X.copy()
    X_changed[0, 0] += 5.0
    with pytest.raises(CheckpointMismatchError, match="data_hash"):
        GridSearchCV(
            SVC(), PARAM_GRID, cv=3, stage_size=2, checkpoint=ckpt
        ).fit(X_changed, y)


def test_fit_metadata_change_is_detected_and_rejected(tmp_path):
    """sample_weight and groups are part of the reproducibility contract."""
    ckpt = tmp_path / "ckpt"
    grid = {"C": [0.1, 1], "kernel": ["linear"]}
    groups = np.tile([0, 1, 2, 3, 4], len(y) // 5 + 1)[: len(y)]
    sample_weight = np.ones(len(y))

    GridSearchCV(SVC(), grid, cv=3, stage_size=2, checkpoint=ckpt).fit(
        X, y, sample_weight=sample_weight, groups=groups
    )

    with pytest.raises(CheckpointMismatchError, match="data_hash"):
        GridSearchCV(SVC(), grid, cv=3, stage_size=2, checkpoint=ckpt).fit(
            X, y, sample_weight=sample_weight * 2, groups=groups
        )

    with pytest.raises(CheckpointMismatchError, match="data_hash"):
        GridSearchCV(SVC(), grid, cv=3, stage_size=2, checkpoint=ckpt).fit(
            X, y, sample_weight=sample_weight, groups=groups + 1
        )

    # Identical metadata resumes without re-evaluating.
    resumed = GridSearchCV(
        SVC(), grid, cv=3, stage_size=2, checkpoint=ckpt
    ).fit(X, y, sample_weight=sample_weight, groups=groups)
    assert resumed.n_evaluated_ == 2


def test_random_seed_change_is_detected_and_rejected(tmp_path):
    """The random seed is part of the reproducibility contract."""
    ckpt = tmp_path / "ckpt"
    distributions = {"C": [0.1, 1, 10], "kernel": ["linear", "rbf"]}
    RandomizedSearchCV(
        SVC(),
        distributions,
        n_iter=4,
        random_state=0,
        cv=3,
        stage_size=2,
        checkpoint=ckpt,
    ).fit(X, y)

    with pytest.raises(CheckpointMismatchError, match="random_state_hash"):
        RandomizedSearchCV(
            SVC(),
            distributions,
            n_iter=4,
            random_state=1,
            cv=3,
            stage_size=2,
            checkpoint=ckpt,
        ).fit(X, y)


def test_same_staging_produces_same_candidate_sequence(tmp_path):
    """Two identically staged, seeded runs explore candidates in the same order."""
    distributions = {"C": np.logspace(-2, 2, 8), "kernel": ["linear", "rbf"]}
    common = dict(
        estimator=SVC(),
        param_distributions=distributions,
        n_iter=6,
        random_state=42,
        cv=3,
        stage_size=2,
    )
    first = RandomizedSearchCV(
        **common, checkpoint=tmp_path / "a"
    ).fit(X, y)
    second = RandomizedSearchCV(
        **common, checkpoint=tmp_path / "b"
    ).fit(X, y)

    sequence_first = [
        tuple(sorted(params.items())) for params in first.cv_results_["params"]
    ]
    sequence_second = [
        tuple(sorted(params.items())) for params in second.cv_results_["params"]
    ]
    assert sequence_first == sequence_second


def test_failed_candidate_is_recorded_and_search_continues(tmp_path):
    """A failing candidate becomes NaN scores without aborting the stage."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        search = GridSearchCV(
            FlakySVC(), PARAM_GRID, cv=3, stage_size=2, checkpoint=tmp_path / "ckpt"
        ).fit(X, y)

    assert np.isnan(search.cv_results_["mean_test_score"]).sum() == 1
    assert search.n_evaluated_ == 6

    # The failure is persisted: resume must not re-evaluate anything.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        resumed = GridSearchCV(
            FlakySVC(), PARAM_GRID, cv=3, stage_size=2, checkpoint=tmp_path / "ckpt"
        ).fit(X, y)
    assert resumed.n_evaluated_ == 6
    assert np.isnan(resumed.cv_results_["mean_test_score"]).sum() == 1


def test_corrupted_segment_is_salvaged(tmp_path):
    """A truncated segment loses only its broken trailing record."""
    ckpt = tmp_path / "ckpt"
    GridSearchCV(
        SVC(), PARAM_GRID, cv=3, stage_size=2, checkpoint=ckpt
    ).fit(X, y)

    segment = ckpt / "segment_000002.pkl"
    raw = segment.read_bytes()
    segment.write_bytes(raw[:-10])  # cut the final frame mid-pickle

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        resumed = GridSearchCV(
            SVC(), PARAM_GRID, cv=3, stage_size=2, checkpoint=ckpt
        ).fit(X, y)

    assert any(
        issubclass(warning.category, CheckpointCorruptedWarning) for warning in caught
    )
    report = resumed.checkpoint_.salvage_report
    assert report[0]["stage"] == 2
    assert report[0]["kept_records"] == 1
    assert (ckpt / "quarantine").is_dir()
    # The dropped record was re-evaluated; the final answer is unchanged.
    reference = GridSearchCV(SVC(), PARAM_GRID, cv=3).fit(X, y)
    _assert_results_match(reference, resumed)


def test_corrupted_manifest_raises(tmp_path):
    ckpt = tmp_path / "ckpt"
    GridSearchCV(
        SVC(), PARAM_GRID, cv=3, stage_size=2, checkpoint=ckpt
    ).fit(X, y)
    (ckpt / "manifest.json").write_text("{broken json")

    with pytest.raises(CheckpointCorruptedError):
        GridSearchCV(
            SVC(), PARAM_GRID, cv=3, stage_size=2, checkpoint=ckpt
        ).fit(X, y)


def test_candidate_key_normalisation():
    """Parameter-name order and float rendering must not split one candidate."""
    assert candidate_key({"a": 1, "b": 2}) == candidate_key({"b": 2, "a": 1})
    assert candidate_key({"a": 0.1 + 0.2}) == candidate_key({"a": 0.30000000000000004})
    # int vs float stay distinct candidates.
    assert candidate_key({"a": 1}) != candidate_key({"a": 1.0})


def test_pack_unpack_roundtrip_and_truncation():
    records = [{"key": "a", "out": [1, 2]}, {"key": "b", "out": [3]}]
    packed = _pack_records(records)
    recovered, dropped, error = _unpack_records(packed)
    assert recovered == records and dropped == 0 and error is None

    # Cutting the last frame keeps the intact prefix and flags the loss.
    recovered, dropped, error = _unpack_records(packed[:-5])
    assert recovered == records[:1] and dropped == 1 and error is not None


def test_checkpoint_without_stage_size_runs_single_stage(tmp_path):
    """checkpoint alone (no stage_size) is valid: one stage covering everything."""
    search = GridSearchCV(
        SVC(), PARAM_GRID, cv=3, checkpoint=tmp_path / "ckpt"
    ).fit(X, y)
    assert len(search.stage_summaries_) == 1
    assert search.n_evaluated_ == 6


def test_duplicate_candidates_are_deduplicated(tmp_path):
    """Sampling with replacement can repeat a setting; it is evaluated once."""
    # The all-list sampler samples without replacement, so build a space whose
    # distinct combinations collapse onto the same normalised identity.
    grid = {"C": [1.0, 1.0], "kernel": ["linear", "linear"]}
    search = GridSearchCV(
        SVC(),
        grid,
        cv=3,
        stage_size=1,
        checkpoint=tmp_path / "ckpt",
    ).fit(X, y)
    # ParameterGrid yields 4 slots but they all describe the same setting.
    assert len(search.cv_results_["params"]) == 4
    assert search.n_evaluated_ == 1
    # Every output slot is filled from the single stored evaluation.
    assert not np.isnan(search.cv_results_["mean_test_score"]).any()
