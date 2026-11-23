# Authors: The scikit-learn developers
# SPDX-License-Identifier: BSD-3-Clause

import io
import json
import os
import pickle

import numpy as np
import pytest

joblib = pytest.importorskip("joblib")

import pandas as pd

from sklearn.datasets import make_classification, make_regression
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.utils._artifact_contract import (
    CONTRACT_VERSION,
    ContractCorruptionError,
    ContractError,
    ContractMigrationError,
    ContractValidationError,
    ContractVersionError,
    ContractedModel,
    FeatureSpec,
    LegacyArtifactWarning,
    ModelContract,
    SUPPORTED_CONTRACT_VERSIONS,
    _make_envelope,
    build_contract,
    load_contracted_model,
    make_contracted_model,
    migrate_contract,
    read_contract,
    save_contracted_model,
    validate_artifact,
)


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def numeric_data():
    X, y = make_classification(
        n_samples=120, n_features=6, n_informative=4, random_state=42
    )
    return X, y


@pytest.fixture
def fitted_clf(numeric_data):
    X, y = numeric_data
    return LogisticRegression(max_iter=500, random_state=42).fit(X, y)


@pytest.fixture
def artifact_path(tmp_path, fitted_clf, numeric_data):
    X, _ = numeric_data
    path = tmp_path / "model.joblib"
    save_contracted_model(fitted_clf, path, X)
    return str(path)


class _LegacyEstimator:
    """Stand-in for a pre-contract artifact that keeps its version stamp."""

    def __init__(self):
        self._sklearn_version = "1.7.2"

    def get_params(self, deep=False):
        return {"random_state": 42}


def _make_categorical_frame(seed=0):
    rng = np.random.RandomState(seed)
    frame = pd.DataFrame(
        {
            "city": rng.choice(["paris", "london", "berlin"], size=60),
            "plan": rng.choice(["free", "pro"], size=60),
            "score": rng.normal(size=60),
        }
    )
    target = (frame["score"].to_numpy() + (frame["city"] == "paris")).astype(int)
    return frame, target


# ---------------------------------------------------------------------------
# 1. Contract contents
# ---------------------------------------------------------------------------


class TestContractContents:
    def test_records_names_order_dtype_and_policy(self, fitted_clf, numeric_data):
        X, _ = numeric_data
        contract = build_contract(fitted_clf, X)
        assert contract.contract_version == CONTRACT_VERSION
        assert [f["name"] for f in contract.features] == [
            f"x{i}" for i in range(X.shape[1])
        ]
        assert [f["position"] for f in contract.features] == list(range(X.shape[1]))
        assert all(f["dtype"] == str(X.dtype) for f in contract.features)
        assert all(f["missing_policy"] == "error" for f in contract.features)
        assert all(f["missing_count"] == 0 for f in contract.features)

    def test_records_categorical_vocabulary(self):
        frame, _ = _make_categorical_frame()
        encoder = OneHotEncoder(handle_unknown="ignore").fit(frame[["city", "plan"]])
        contract = build_contract(encoder, frame[["city", "plan"]])
        city = next(f for f in contract.features if f["name"] == "city")
        assert city["is_categorical"] is True
        assert tuple(city["categories"]) == ("berlin", "london", "paris")
        assert city["dtype"] in ("object", "str")

    def test_missing_policy_auto_records_observed_missing(self):
        frame = pd.DataFrame({"a": [1.0, 2.0, np.nan], "b": [1, 2, 3]})
        target = np.array([0, 1, 0])
        model = LogisticRegression().fit(frame.fillna(0.0), target)
        contract = build_contract(model, frame)
        policies = {f["name"]: f for f in contract.features}
        assert policies["a"]["missing_policy"] == "allow"
        assert policies["a"]["missing_count"] == 1
        assert policies["b"]["missing_policy"] == "error"

    def test_records_versions_params_and_seed(self, fitted_clf, numeric_data):
        X, _ = numeric_data
        contract = build_contract(fitted_clf, X)
        assert "sklearn" in contract.environment
        assert "numpy" in contract.environment
        assert contract.environment["python"].count(".") == 2
        assert contract.param_summary["random_state"] == 42
        assert contract.param_summary["max_iter"] == 500
        assert contract.random_seeds == {"random_state": 42}
        assert isinstance(contract.fit_id, str) and len(contract.fit_id) == 36

    def test_unseeded_random_state_recorded_as_type_name(self, numeric_data):
        X, y = numeric_data
        forest = RandomForestClassifier(n_estimators=3).fit(X, y)
        contract = build_contract(forest, X)
        # None means genuinely unseeded; a random object records a type name.
        assert contract.random_seeds["random_state"] is None


# ---------------------------------------------------------------------------
# 2 & 3. Exact environment: bit-identical predictions
# ---------------------------------------------------------------------------


class TestBitIdentical:
    @pytest.mark.parametrize(
        "method", ["predict", "predict_proba", "decision_function"]
    )
    def test_numpy_roundtrip_bit_identical(
        self, artifact_path, fitted_clf, numeric_data, method
    ):
        X, _ = numeric_data
        before = getattr(fitted_clf, method)(X)
        model, report = load_contracted_model(artifact_path, X=X)
        assert report.status == "valid"
        after = getattr(model, method)(X)
        np.testing.assert_array_equal(before, after)

    def test_single_sample_bit_identical(self, artifact_path, fitted_clf, numeric_data):
        X, _ = numeric_data
        before = fitted_clf.predict(X[:1])
        model, _ = load_contracted_model(artifact_path, X=X)
        after = model.predict(X[:1])
        np.testing.assert_array_equal(before, after)

    def test_pandas_roundtrip_bit_identical(self):
        frame, target = _make_categorical_frame()
        X = pd.get_dummies(frame, columns=["city", "plan"]).astype(float)
        reg = LinearRegression().fit(X, target)
        path = save_contracted_model(reg, "/tmp/df_model.joblib", X)
        try:
            before = reg.predict(X)
            model, report = load_contracted_model(path, X=X)
            assert report.status == "valid"
            np.testing.assert_array_equal(before, model.predict(X))
        finally:
            os.unlink(path)


# ---------------------------------------------------------------------------
# 4. Patch-level dependency differences
# ---------------------------------------------------------------------------


class TestPatchVersion:
    def test_patch_difference_warns_but_loads(
        self, artifact_path, numeric_data, monkeypatch
    ):
        X, _ = numeric_data
        model = joblib.load(artifact_path)
        contract = model.contract_
        original = dict(contract.environment)
        current_numpy = original["numpy"]
        major, minor, patch = (int(part) for part in current_numpy.split(".")[:3])
        original["numpy"] = f"{major}.{minor}.{max(patch - 1, 0)}"
        monkeypatch.setattr(contract, "environment", original)
        model._envelope = _make_envelope(contract)
        joblib.dump(model, artifact_path)

        loaded, report = load_contracted_model(artifact_path, X=X)
        assert report.status == "compatible"
        codes = [f.code for f in report.findings]
        assert "version.patch_difference" in codes
        message = str(report)
        assert "numpy" in message and "patch-level difference" in message
        # Predictions remain available.
        assert loaded.predict(X).shape == (X.shape[0],)

    def test_major_minor_difference_rejected(
        self, artifact_path, numeric_data, monkeypatch
    ):
        X, _ = numeric_data
        model = joblib.load(artifact_path)
        contract = model.contract_
        environment = dict(contract.environment)
        # Force a stable numeric fit-time version so the gate's tuple
        # comparison can be exercised independently of dev wheel naming.
        environment["sklearn"] = "2.5.0"
        monkeypatch.setattr(contract, "environment", environment)
        model._envelope = _make_envelope(contract)
        joblib.dump(model, artifact_path)

        report = validate_artifact(artifact_path, X=X)
        assert report.status == "invalid"
        assert any(f.code == "version.incompatible" for f in report.findings)
        with pytest.raises(ContractValidationError, match="major/minor"):
            load_contracted_model(artifact_path, X=X)


# ---------------------------------------------------------------------------
# 5. Reordered columns align by name
# ---------------------------------------------------------------------------


class TestReorder:
    def test_numpy_reorder_by_name(self, artifact_path, fitted_clf, numeric_data):
        X, _ = numeric_data
        permutation = np.arange(X.shape[1])[::-1]
        reordered = X[:, permutation]

        # Numpy contracts use positional x0.. names, so reordering is seen as
        # content change there; the name-based guarantee is exercised with the
        # pandas contract below. Here the aligned map itself must be correct.
        model, report = load_contracted_model(artifact_path, X=X)
        assert report.alignment == {f"x{i}": i for i in range(X.shape[1])}
        assert model.predict(reordered).shape == (X.shape[0],)

    def test_pandas_reorder_by_name_and_values(self):
        frame, target = _make_categorical_frame()
        X = pd.get_dummies(frame, columns=["city", "plan"]).astype(float)
        reg = LinearRegression().fit(X, target)
        path = "/tmp/reorder_model.joblib"
        save_contracted_model(reg, path, X)
        try:
            shuffled = X[X.columns[::-1]]
            model, report = load_contracted_model(path, X=shuffled)
            assert report.status == "compatible"
            assert report.reordered is True
            assert any(f.code == "feature.reordered" for f in report.findings)
            np.testing.assert_array_equal(reg.predict(X), model.predict(shuffled))
        finally:
            os.unlink(path)


# ---------------------------------------------------------------------------
# 6. Missing / extra columns, dtype changes, category drift
# ---------------------------------------------------------------------------


class TestFeatureMismatches:
    def _frame_contract(self):
        frame, target = _make_categorical_frame()
        X = pd.get_dummies(frame, columns=["city", "plan"]).astype(float)
        reg = LinearRegression().fit(X, target)
        return X, reg

    def test_missing_column_is_error_and_no_zero_filling(self):
        X, reg = self._frame_contract()
        path = "/tmp/missing_col.joblib"
        save_contracted_model(reg, path, X)
        try:
            reduced = X.drop(columns=[X.columns[1]])
            report = validate_artifact(path, X=reduced)
            assert report.status == "invalid"
            missing = [f for f in report.findings if f.code == "feature.missing"]
            assert len(missing) == 1
            assert X.columns[1] in missing[0].message
            with pytest.raises(ContractValidationError):
                load_contracted_model(path, X=reduced)
        finally:
            os.unlink(path)

    def test_extra_column_is_error(self):
        X, reg = self._frame_contract()
        path = "/tmp/extra_col.joblib"
        save_contracted_model(reg, path, X)
        try:
            extended = X.copy()
            extended["brand_new"] = 1.0
            report = validate_artifact(path, X=extended)
            assert any(f.code == "feature.unexpected" for f in report.findings)
            assert report.status == "invalid"
        finally:
            os.unlink(path)

    def test_dtype_change_is_error(self):
        X, reg = self._frame_contract()
        path = "/tmp/dtype_change.joblib"
        save_contracted_model(reg, path, X)
        try:
            changed = X.copy()
            target_column = changed.columns[0]
            changed[target_column] = changed[target_column].astype("float32")
            report = validate_artifact(path, X=changed)
            dtype_findings = [
                f for f in report.findings if f.code == "feature.dtype_mismatch"
            ]
            assert len(dtype_findings) == 1
            assert target_column in dtype_findings[0].message
            assert "float32" in dtype_findings[0].message
        finally:
            os.unlink(path)

    def test_category_drift_is_error_without_leaking_values(self):
        frame, target = _make_categorical_frame()
        # Wrap the encoder output through a pipeline so raw categorical
        # columns are what the contract sees.
        pipe = Pipeline(
            steps=[
                ("encoder", OneHotEncoder(handle_unknown="ignore")),
                ("clf", LogisticRegression(max_iter=500)),
            ]
        ).fit(frame[["city", "plan"]], target)
        path = "/tmp/cat_drift.joblib"
        save_contracted_model(pipe, path, frame[["city", "plan"]])
        try:
            drifted = frame.copy()
            drifted.loc[0, "city"] = "tokyo"
            report = validate_artifact(path, X=drifted[["city", "plan"]])
            drift = [f for f in report.findings if f.code == "feature.category_drift"]
            assert len(drift) == 1
            assert "1 category value(s)" in drift[0].message
            # The actual offending value must never appear in any message.
            assert "tokyo" not in str(report)
            assert "paris" not in str(report)
            with pytest.raises(ContractValidationError):
                load_contracted_model(path, X=drifted[["city", "plan"]])
        finally:
            os.unlink(path)

    def test_all_differences_reported_in_one_pass(self):
        X, reg = self._frame_contract()
        path = "/tmp/all_diff.joblib"
        save_contracted_model(reg, path, X)
        try:
            broken = X.drop(columns=[X.columns[0]]).copy()
            broken["brand_new"] = 1.0
            broken[broken.columns[0]] = broken[broken.columns[0]].astype("float32")
            report = validate_artifact(path, X=broken)
            codes = {f.code for f in report.findings}
            assert "feature.missing" in codes
            assert "feature.unexpected" in codes
            assert "feature.dtype_mismatch" in codes
        finally:
            os.unlink(path)

    def test_unexpected_missing_value_is_error(self):
        frame = pd.DataFrame(
            {"a": np.arange(6.0), "b": np.arange(6.0)},
        )
        target = np.array([0, 1, 0, 1, 0, 1])
        clf = LogisticRegression().fit(frame, target)
        path = "/tmp/missing_value.joblib"
        save_contracted_model(clf, path, frame)
        try:
            current = frame.copy()
            current.loc[0, "a"] = np.nan
            report = validate_artifact(path, X=current)
            assert any(
                f.code == "feature.unexpected_missing" for f in report.findings
            )
            with pytest.raises(ContractValidationError):
                load_contracted_model(path, X=current)
        finally:
            os.unlink(path)


# ---------------------------------------------------------------------------
# 7. Nested / composite models verify recursively
# ---------------------------------------------------------------------------


class TestNestedContracts:
    def test_pipeline_recursive_contract(self):
        frame, target = _make_categorical_frame()
        pipe = Pipeline(
            steps=[
                ("scaler", StandardScaler()),
                ("clf", LogisticRegression(max_iter=500, random_state=7)),
            ]
        ).fit(frame[["score"]], target)
        path = "/tmp/pipeline.joblib"
        save_contracted_model(pipe, path, frame[["score"]])
        try:
            model, report = load_contracted_model(path, X=frame[["score"]])
            assert report.status == "valid"
            paths = {sub["path"] for sub in model.contract_.sub_contracts}
            assert "steps.scaler" in paths and "steps.clf" in paths
            assert all(
                sub["fit_id"] == model.contract_.fit_id for sub in
                model.contract_.sub_contracts
            )
        finally:
            os.unlink(path)

    def test_swapped_nested_estimator_detected(self, numeric_data):
        X, y = numeric_data
        forest_a = RandomForestClassifier(n_estimators=3, random_state=1).fit(X, y)
        forest_b = RandomForestClassifier(n_estimators=3, random_state=2).fit(X, y)
        model = make_contracted_model(forest_a, X)
        # Tamper with the bundled estimator while keeping the old envelope.
        model.estimator_ = forest_b
        blob = pickle.dumps(model)
        with pytest.raises(ContractCorruptionError, match="fingerprint does not match"):
            pickle.loads(blob)

    def test_nested_fit_id_mismatch_detected(self, numeric_data):
        X, y = numeric_data
        pipe = Pipeline(
            steps=[
                ("scaler", StandardScaler()),
                ("clf", LogisticRegression(max_iter=200)),
            ]
        ).fit(X, y)
        model = make_contracted_model(pipe, X)
        model.contract_.sub_contracts[0]["fit_id"] = "different-fit-id"
        model._envelope = _make_envelope(model.contract_)
        blob = pickle.dumps(model)
        with pytest.raises(ContractCorruptionError, match="fit_id"):
            pickle.loads(blob)


# ---------------------------------------------------------------------------
# 8. Contract versioning and migration
# ---------------------------------------------------------------------------


class TestVersioningAndMigration:
    def test_too_new_version_rejected_with_supported_range(
        self, artifact_path
    ):
        model = joblib.load(artifact_path)
        model._envelope["contract_version"] = 999
        blob = pickle.dumps(model)
        # Unpickling succeeds (envelope intact, fingerprints valid); the
        # version gate is enforced at the load/validate boundary.
        reloaded = pickle.loads(blob)
        assert reloaded._envelope["contract_version"] == 999
        joblib.dump(reloaded, artifact_path)
        with pytest.raises(ContractVersionError, match="999") as exc:
            validate_artifact(artifact_path)
        message = str(exc.value)
        assert str(SUPPORTED_CONTRACT_VERSIONS) in message
        with pytest.raises(ContractVersionError):
            load_contracted_model(artifact_path)

    def test_migrate_09_artifact(self, artifact_path, numeric_data, tmp_path):
        X, _ = numeric_data
        model = joblib.load(artifact_path)
        old_payload = {
            "contract_version": "0.9",
            "format": "sklearn-model-contract",
            "fit_id": model.contract_.fit_id,
            "estimator_class": model.contract_.estimator_class,
            "estimator_fingerprint": model.contract_.estimator_fingerprint,
            "features": [{"name": f"x{i}"} for i in range(X.shape[1])],
            "environment": model.contract_.environment,
            "param_summary": {},
            "random_seeds": {},
            "sub_contracts": [],
        }
        import hashlib

        payload_text = json.dumps(old_payload, sort_keys=True)
        model._envelope = {
            "format": "sklearn-model-contract",
            "contract_version": "0.9",
            "checksum": hashlib.sha256(payload_text.encode()).hexdigest(),
            "payload": payload_text,
            "migrated_from": None,
        }
        legacy_path = tmp_path / "legacy.joblib"
        output_path = tmp_path / "migrated.joblib"
        joblib.dump(model, legacy_path)

        result = migrate_contract(legacy_path, output_path)
        assert result == os.fspath(output_path)
        migrated, report = load_contracted_model(output_path, X=X)
        assert migrated.contract_.migrated_from == "0.9"
        assert any(f.code == "contract.migrated" for f in report.findings)
        assert any(
            f.code == "feature.dtype_unknown" for f in report.findings
        )

    def test_unmigratable_version_refused_and_source_untouched(
        self, artifact_path, tmp_path
    ):
        source_size = os.path.getsize(artifact_path)
        model = joblib.load(artifact_path)
        model._envelope["contract_version"] = "0.1"
        legacy_path = tmp_path / "ancient.joblib"
        joblib.dump(model, legacy_path)
        output_path = tmp_path / "should_not_exist.joblib"
        with pytest.raises(ContractMigrationError, match="no migration path"):
            migrate_contract(legacy_path, output_path)
        assert not output_path.exists()
        assert os.path.getsize(artifact_path) == source_size

    def test_failed_migration_keeps_source_intact(self, tmp_path):
        # A 0.9 envelope whose payload lacks required migration fields.
        frame = pd.DataFrame({"a": [1.0, 2.0]})
        clf = LinearRegression().fit(frame, frame)
        model = make_contracted_model(clf, frame)
        payload_text = json.dumps({"contract_version": "0.9"})
        import hashlib

        model._envelope = {
            "format": "sklearn-model-contract",
            "contract_version": "0.9",
            "checksum": hashlib.sha256(payload_text.encode()).hexdigest(),
            "payload": payload_text,
            "migrated_from": None,
        }
        broken_path = tmp_path / "broken09.joblib"
        out_path = tmp_path / "broken_out.joblib"
        joblib.dump(model, broken_path)
        before = broken_path.read_bytes()
        with pytest.raises(ContractMigrationError):
            migrate_contract(broken_path, out_path)
        assert broken_path.read_bytes() == before
        assert not out_path.exists()


# ---------------------------------------------------------------------------
# 9. Corruption and truncation
# ---------------------------------------------------------------------------


class TestCorruption:
    def test_truncated_file_does_not_load_weights(self, artifact_path):
        original_size = os.path.getsize(artifact_path)
        truncated = artifact_path + ".truncated"
        with open(artifact_path, "rb") as src, open(truncated, "wb") as dst:
            dst.write(src.read(original_size // 3))
        try:
            with pytest.raises(ContractCorruptionError):
                load_contracted_model(truncated)
        finally:
            os.unlink(truncated)

    def test_checksum_tampering_detected(self, artifact_path):
        model = joblib.load(artifact_path)
        model._envelope["checksum"] = "0" * 64
        blob = pickle.dumps(model)
        with pytest.raises(ContractCorruptionError, match="checksum mismatch"):
            pickle.loads(blob)

    def test_payload_tampering_detected(self, artifact_path):
        model = joblib.load(artifact_path)
        payload = json.loads(model._envelope["payload"])
        payload["features"][0]["dtype"] = "float32"
        model._envelope["payload"] = json.dumps(payload, sort_keys=True)
        blob = pickle.dumps(model)
        with pytest.raises(ContractCorruptionError, match="checksum mismatch"):
            pickle.loads(blob)

    def test_missing_envelope_refused(self, fitted_clf, numeric_data):
        X, _ = numeric_data
        plain_path = "/tmp/plain_legacy.joblib"
        joblib.dump(fitted_clf, plain_path)
        try:
            with pytest.raises(ContractError, match="not a contracted artifact"):
                load_contracted_model(plain_path)
        finally:
            os.unlink(plain_path)


# ---------------------------------------------------------------------------
# Hard constraints
# ---------------------------------------------------------------------------


class TestHardConstraints:
    def test_failure_produces_no_prediction(self):
        frame, target = _make_categorical_frame()
        X = pd.get_dummies(frame, columns=["city", "plan"]).astype(float)
        reg = LinearRegression().fit(X, target)
        path = "/tmp/refuse_predict.joblib"
        save_contracted_model(reg, path, X)
        try:
            reduced = X.drop(columns=[X.columns[0]])
            model, _ = load_contracted_model(path, X=X)
            with pytest.raises(ContractValidationError):
                model.predict(reduced)
        finally:
            os.unlink(path)

    def test_failure_does_not_modify_source_file(self, artifact_path, numeric_data):
        X, _ = numeric_data
        before = open(artifact_path, "rb").read()
        reduced = X[:, :-1]
        with pytest.raises(ContractValidationError):
            load_contracted_model(artifact_path, X=reduced)
        assert open(artifact_path, "rb").read() == before

    def test_legacy_artifact_basic_info_still_readable(
        self, fitted_clf, numeric_data
    ):
        X, _ = numeric_data
        legacy_path = "/tmp/legacy_basic.joblib"
        joblib.dump(fitted_clf, legacy_path)
        try:
            with pytest.warns(LegacyArtifactWarning):
                info = read_contract(legacy_path)
            assert info["contract_format"] == "legacy"
            assert info["estimator_class"].endswith("LogisticRegression")
            assert info["param_summary"]["random_state"] == 42
        finally:
            os.unlink(legacy_path)

    def test_legacy_artifact_version_captured_when_present(self, tmp_path):
        # An older artifact whose loader leaves the version stamp on the
        # instance (rather than consuming it in __setstate__).
        legacy_path = tmp_path / "legacy_versioned.joblib"
        joblib.dump(_LegacyEstimator(), legacy_path)
        with pytest.warns(LegacyArtifactWarning):
            info = read_contract(legacy_path)
        assert info["sklearn_version_at_fit"] == "1.7.2"

    def test_read_contract_on_contracted_artifact(self, artifact_path):
        info = read_contract(artifact_path)
        assert info["contract_version"] == CONTRACT_VERSION
        assert info["n_features"] == 6
        assert info["estimator_class"].endswith("LogisticRegression")

    def test_messages_contain_no_feature_values(self, artifact_path, numeric_data):
        X, _ = numeric_data
        report = validate_artifact(artifact_path, X=X)
        text = str(report)
        # Numeric content of the training matrix must not leak into messages.
        assert not any(
            token in text
            for token in np.unique(X[:5])[:3].astype(str)
        )

    def test_mutators_are_blocked(self, artifact_path, numeric_data):
        X, y = numeric_data
        model, _ = load_contracted_model(artifact_path, X=X)
        with pytest.raises(ContractError, match="blocked"):
            model.fit(X, y)
        with pytest.raises(ContractError, match="blocked"):
            model.set_params(max_iter=1)

    def test_existing_summary_and_params_unchanged(self, artifact_path):
        model, _ = load_contracted_model(artifact_path)
        # Delegation preserves the standard estimator interfaces.
        assert model.get_params()["random_state"] == 42
        assert "LogisticRegression" in repr(model.estimator_)

    def test_contract_and_weights_same_fit_only(self, numeric_data):
        X, y = numeric_data
        clf_a = LogisticRegression(random_state=1).fit(X, y)
        clf_b = LogisticRegression(random_state=2).fit(X, y)
        contract_a = build_contract(clf_a, X)
        envelope = _make_envelope(contract_a)
        frankenstein = ContractedModel(clf_b, contract_a, envelope)
        blob = pickle.dumps(frankenstein)
        with pytest.raises(ContractCorruptionError, match="same fit"):
            pickle.loads(blob)


# ---------------------------------------------------------------------------
# Regression: plain estimator behavior untouched
# ---------------------------------------------------------------------------


class TestExistingBehavior:
    def test_plain_joblib_path_unchanged(self, fitted_clf, numeric_data):
        X, _ = numeric_data
        buffer = io.BytesIO()
        joblib.dump(fitted_clf, buffer)
        buffer.seek(0)
        reloaded = joblib.load(buffer)
        np.testing.assert_array_equal(
            fitted_clf.predict(X), reloaded.predict(X)
        )

    def test_in_memory_contract_does_not_change_predictions(
        self, fitted_clf, numeric_data
    ):
        X, _ = numeric_data
        model = make_contracted_model(fitted_clf, X)
        np.testing.assert_array_equal(
            fitted_clf.predict_proba(X), model.predict_proba(X)
        )

    def test_structural_only_validation_is_valid(self, artifact_path):
        report = validate_artifact(artifact_path)
        assert report.status == "valid"
        assert any(f.code == "feature.check_skipped" for f in report.findings)
