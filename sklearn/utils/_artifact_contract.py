"""Verifiable contracts for serialized model artifacts.

A model artifact produced by :func:`save_contracted_model` bundles the fitted
estimator together with a :class:`ModelContract` that records the provenance of
the fit: feature names and order, dtypes, categorical vocabularies, missing
value policies, dependency versions, parameter summary and random seeds.

The contract is serialized as a checksummed JSON envelope. Loading an artifact
verifies the envelope checksum and an estimator fingerprint before anything is
returned, so a truncated or tampered contract cannot silently degrade into a
weights-only load. Feature validation either succeeds (possibly with an
alignment map) or reports *all* differences at once; it never fills missing
columns with zeros.

The module only relies on serialization capabilities already used by
scikit-learn (``joblib`` and the standard library).
"""

# Authors: The scikit-learn developers
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import hashlib
import json
import os
import re
import tempfile
import uuid
import warnings
from dataclasses import asdict, dataclass, field

import joblib

from sklearn.base import BaseEstimator

__all__ = [
    "CONTRACT_VERSION",
    "SUPPORTED_CONTRACT_VERSIONS",
    "ModelContract",
    "FeatureSpec",
    "Finding",
    "ValidationReport",
    "ContractedModel",
    "ContractError",
    "ContractCorruptionError",
    "ContractValidationError",
    "ContractVersionError",
    "ContractMigrationError",
    "LegacyArtifactWarning",
    "build_contract",
    "make_contracted_model",
    "save_contracted_model",
    "load_contracted_model",
    "validate_artifact",
    "read_contract",
    "migrate_contract",
]

#: Contract format version written by this release.
CONTRACT_VERSION = 1
#: Contract versions that can be verified directly (without migration).
SUPPORTED_CONTRACT_VERSIONS = (1,)
#: Oldest contract version for which a migration path is implemented.
OLDEST_MIGRATABLE_VERSION = "0.9"

_FORMAT = "sklearn-model-contract"
_PREDICT_METHODS = frozenset(
    [
        "predict",
        "predict_proba",
        "predict_log_proba",
        "decision_function",
        "transform",
        "score",
        "score_samples",
        "kneighbors",
        "radius_neighbors",
        "apply",
        "staged_predict",
    ]
)
#: Mutating methods that would invalidate the stored estimator fingerprint.
_BLOCKED_MUTATORS = frozenset(
    ["fit", "fit_transform", "fit_predict", "partial_fit", "set_params"]
)
_DEPENDENCIES = ("sklearn", "numpy", "scipy", "joblib")


# ---------------------------------------------------------------------------
# Errors and warnings
# ---------------------------------------------------------------------------


class ContractError(Exception):
    """Base class for artifact contract failures."""


class ContractCorruptionError(ContractError):
    """The contract envelope is missing, truncated, tampered with, or its
    contents do not match the bundled estimator."""


class ContractValidationError(ContractError):
    """Raised when an otherwise intact contract does not validate against the
    current environment or input data.

    The full report, including every difference found in a single pass, is
    available on the ``report`` attribute.
    """

    def __init__(self, report):
        self.report = report
        super().__init__(str(report))


class ContractVersionError(ContractError):
    """The contract version is newer than this installation, or otherwise
    outside the supported range."""


class ContractMigrationError(ContractError):
    """A legacy contract cannot be migrated to the current version."""


class LegacyArtifactWarning(UserWarning):
    """Warning emitted when a legacy (contract-free) artifact is read."""


# ---------------------------------------------------------------------------
# Plain data containers
# ---------------------------------------------------------------------------


@dataclass
class FeatureSpec:
    """Contract describing a single input feature.

    Parameters
    ----------
    name : str
        Feature name as known at fit time.
    position : int
        Zero-based position of the feature in the training matrix.
    dtype : str
        NumPy dtype name (for example ``"float64"``) or ``"object"`` for
        categorical features held in object/string/category containers.
    is_categorical : bool
        Whether the feature was treated as categorical.
    categories : tuple or None
        Sorted vocabulary observed at fit time. ``None`` for numeric features.
    missing_policy : str
        ``"error"`` if missing values were forbidden at fit time, ``"allow"``
        if they were observed (or explicitly permitted), ``"unknown"`` for
        contracts migrated from a version without this field.
    missing_count : int
        Number of missing values observed in the feature at fit time.
    """

    name: str
    position: int
    dtype: str
    is_categorical: bool
    categories: tuple | None
    missing_policy: str
    missing_count: int

    def migrated_dtype_unknown(self):
        """Whether this spec was produced by migration without dtype info."""
        return self.dtype == "unknown"


@dataclass
class Finding:
    """A single contract verification finding.

    Messages only ever mention column names, dtype names and counts; feature
    values and sample contents are never included.

    Parameters
    ----------
    severity : str
        One of ``"error"``, ``"warning"`` or ``"info"``.
    code : str
        Stable machine-readable code, for example ``"feature.missing"``.
    message : str
        Human-readable, value-free description of the difference.
    """

    severity: str
    code: str
    message: str

    @property
    def is_error(self):
        return self.severity == "error"


@dataclass
class ValidationReport:
    """Result of verifying an artifact contract.

    Parameters
    ----------
    status : str
        ``"valid"`` (exact match), ``"compatible"`` (loadable with caveats) or
        ``"invalid"`` (must not be used for prediction).
    findings : list of Finding
        Every difference collected in a single verification pass.
    alignment : dict or None
        Mapping of current feature names to fit-time positions when the
        feature sets match, possibly in a different order.
    reordered : bool
        Whether the current input columns are in a different order than the
        fit-time contract.
    artifact_path : str or None
        Path of the verified artifact, when loaded from disk.
    """

    status: str
    findings: list = field(default_factory=list)
    alignment: dict | None = None
    reordered: bool = False
    artifact_path: str | None = None

    @property
    def ok(self):
        """Whether the artifact may be used for prediction."""
        return self.status in ("valid", "compatible")

    def raise_for_errors(self):
        """Raise :class:`ContractValidationError` if any finding is an error."""
        if any(f.is_error for f in self.findings):
            raise ContractValidationError(self)

    def to_dict(self):
        return {
            "status": self.status,
            "ok": self.ok,
            "artifact_path": self.artifact_path,
            "alignment": self.alignment,
            "findings": [asdict(f) for f in self.findings],
        }

    def __str__(self):
        lines = [
            f"artifact contract verification: {self.status}",
            f"artifact: {self.artifact_path}",
        ]
        if self.alignment is not None:
            lines.append(
                "feature alignment: "
                + ", ".join(
                    f"{name}->position{pos}" for name, pos in self.alignment.items()
                )
            )
        if not self.findings:
            lines.append("findings: none")
        for finding in self.findings:
            lines.append(f"  [{finding.severity}] {finding.code}: {finding.message}")
        return "\n".join(lines)


@dataclass
class ModelContract:
    """The verifiable contract accompanying a fitted estimator.

    Parameters
    ----------
    contract_version : int
        Version of the contract schema.
    format : str
        Format identifier of the envelope.
    fit_id : str
        Identifier shared by every estimator (outer and inner) produced by the
        same fit.
    estimator_class : str
        Fully qualified class name of the outer estimator.
    estimator_fingerprint : str
        SHA-256 fingerprint of the fitted estimator content.
    features : list of FeatureSpec
        Ordered feature specifications.
    environment : dict
        Dependency versions captured at fit time.
    param_summary : dict
        JSON-safe summary of the outer estimator parameters.
    random_seeds : dict
        Random-state related parameter paths and their values (integers,
        ``None`` or a type name for unseeded random objects).
    sub_contracts : list of dict
        Recursive contracts for nested estimators.
    migrated_from : str or None
        Original schema version when this contract was produced by migration.
    """

    contract_version: int
    format: str
    fit_id: str
    estimator_class: str
    estimator_fingerprint: str
    features: list
    environment: dict
    param_summary: dict
    random_seeds: dict
    sub_contracts: list = field(default_factory=list)
    migrated_from: str | None = None

    def to_dict(self):
        data = asdict(self)
        return data


# ---------------------------------------------------------------------------
# Version handling
# ---------------------------------------------------------------------------


def _parse_version(version):
    """Return ``(major, minor, patch)`` from a leading ``X.Y.Z`` segment."""
    match = re.match(r"\s*(\d+)\.(\d+)\.(\d+)", str(version))
    if match is None:
        return (0, 0, 0)
    return tuple(int(part) for part in match.groups())


def _capture_environment():
    import sys

    environment = {"python": ".".join(str(part) for part in sys.version_info[:3])}
    for package in _DEPENDENCIES:
        if package == "sklearn":
            from sklearn import __version__ as version

            environment["sklearn"] = version
            continue
        try:
            module = __import__(package)
            environment[package] = getattr(module, "__version__", "unknown")
        except ImportError:
            environment[package] = None
    return environment


def _compare_versions(expected, actual, package, findings):
    """Compare two versions, allowing patch-level differences only."""
    expected_tuple = _parse_version(expected)
    actual_tuple = _parse_version(actual)
    if expected_tuple == actual_tuple:
        return
    if expected_tuple[:2] == actual_tuple[:2]:
        findings.append(
            Finding(
                "warning",
                "version.patch_difference",
                (
                    f"dependency {package!r} was fitted with {expected!r}, "
                    f"loading with {actual!r}; patch-level difference, "
                    "results should be reproduced but are not guaranteed "
                    "bit-identical"
                ),
            )
        )
    else:
        findings.append(
            Finding(
                "error",
                "version.incompatible",
                (
                    f"dependency {package!r} was fitted with {expected!r}, "
                    f"loading with {actual!r}; major/minor versions differ "
                    "and the artifact must not be used in this environment"
                ),
            )
        )


# ---------------------------------------------------------------------------
# Feature inspection (numpy and pandas, no new dependencies)
# ---------------------------------------------------------------------------


def _is_pandas_dataframe(X):
    try:
        import pandas

        return isinstance(X, pandas.DataFrame)
    except ImportError:
        return False


def _categorical_dtype(dtype):
    kind = getattr(dtype, "kind", None)
    name = str(dtype)
    if kind == "O":
        return True
    if name in ("string", "string[python]", "string[pyarrow]"):
        return True
    try:
        import pandas

        if isinstance(dtype, pandas.CategoricalDtype):
            return True
    except ImportError:
        pass
    return False


def _sorted_vocabulary(values):
    # ``values`` is a 1d sequence with missing entries already removed.
    unique = set()
    for value in values:
        if isinstance(value, float) and value != value:  # NaN without numpy
            continue
        unique.add(value)
    try:
        return sorted(unique)
    except TypeError:
        # Mixed, unorderable vocabulary: fall back to repr ordering.
        return sorted(unique, key=repr)


def _json_safe(value):
    """Coerce fit-time values into JSON-safe contract content."""
    if isinstance(value, BaseEstimator):
        return f"<{type(value).__module__}.{type(value).__qualname__}>"
    if hasattr(value, "item"):
        try:
            return value.item()
        except (ValueError, TypeError):
            pass
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return repr(value)


def _extract_feature_specs(X, missing_policy):
    """Extract ordered :class:`FeatureSpec` objects from training data."""
    if missing_policy not in ("auto", "error", "allow") and not isinstance(
        missing_policy, dict
    ):
        raise ValueError(
            "missing_policy must be 'auto', 'error', 'allow' or a dict "
            f"mapping feature names to policies, got {missing_policy!r}"
        )

    specs = []
    if _is_pandas_dataframe(X):
        for position, name in enumerate(X.columns):
            name = str(name)
            series = X[name]
            dtype = series.dtype
            is_cat = _categorical_dtype(dtype)
            missing_count = int(series.isna().sum())
            if is_cat:
                categories = _sorted_vocabulary(series.dropna().tolist())
            else:
                categories = None
            policy = _resolve_policy(name, missing_policy, missing_count)
            specs.append(
                FeatureSpec(
                    name=name,
                    position=position,
                    dtype=str(dtype),
                    is_categorical=is_cat,
                    categories=tuple(categories) if categories is not None else None,
                    missing_policy=policy,
                    missing_count=missing_count,
                )
            )
        return specs

    try:
        import numpy as np
    except ImportError as exc:  # pragma: no cover - numpy is a hard dependency
        raise ContractError("numpy is required to build artifact contracts") from exc

    array = np.asarray(X)
    if array.ndim != 2:
        raise ValueError(
            f"expected a 2d feature matrix, got {array.ndim}d input"
        )
    for position in range(array.shape[1]):
        column = array[:, position]
        name = f"x{position}"
        dtype = column.dtype
        is_cat = _categorical_dtype(dtype)
        if is_cat:
            missing_count = int(
                sum(item is None or (isinstance(item, float) and item != item)
                    for item in column.tolist())
            )
            categories = _sorted_vocabulary(
                [item for item in column.tolist()
                 if not (item is None or (isinstance(item, float) and item != item))]
            )
        else:
            missing_count = (
                int(np.isnan(column).sum())
                if np.issubdtype(dtype, np.floating)
                else 0
            )
            categories = None
        policy = _resolve_policy(name, missing_policy, missing_count)
        specs.append(
            FeatureSpec(
                name=name,
                position=position,
                dtype=str(dtype),
                is_categorical=is_cat,
                categories=tuple(categories) if categories is not None else None,
                missing_policy=policy,
                missing_count=missing_count,
            )
        )
    return specs


def _resolve_policy(name, missing_policy, missing_count):
    if isinstance(missing_policy, dict):
        policy = missing_policy.get(name, "auto")
    else:
        policy = missing_policy
    if policy == "auto":
        return "allow" if missing_count > 0 else "error"
    return policy


def _current_feature_summary(X):
    """Lightweight current-feature summary used for verification.

    Returns ``(names, {name: (dtype, is_categorical, categories, missing)})``.
    Only structural information (no values) leaves this function.
    """
    if _is_pandas_dataframe(X):
        summary = {}
        names = [str(column) for column in X.columns]
        for name in names:
            series = X[name]
            is_cat = _categorical_dtype(series.dtype)
            categories = (
                tuple(_sorted_vocabulary(series.dropna().tolist()))
                if is_cat
                else None
            )
            summary[name] = (
                str(series.dtype),
                is_cat,
                categories,
                int(series.isna().sum()),
            )
        return names, summary

    import numpy as np

    array = np.asarray(X)
    if array.ndim != 2:
        raise ValueError(f"expected a 2d feature matrix, got {array.ndim}d input")
    names = [f"x{position}" for position in range(array.shape[1])]
    summary = {}
    for position, name in enumerate(names):
        column = array[:, position]
        is_cat = _categorical_dtype(column.dtype)
        if is_cat:
            values = column.tolist()
            missing = sum(
                item is None or (isinstance(item, float) and item != item)
                for item in values
            )
            categories = tuple(
                _sorted_vocabulary(
                    [item for item in values
                     if not (item is None or (isinstance(item, float) and item != item))]
                )
            )
        else:
            missing = (
                int(np.isnan(column).sum())
                if np.issubdtype(column.dtype, np.floating)
                else 0
            )
            categories = None
        summary[name] = (str(column.dtype), is_cat, categories, missing)
    return names, summary


# ---------------------------------------------------------------------------
# Estimator fingerprinting and nested estimator traversal
# ---------------------------------------------------------------------------


def _qualified_name(obj):
    return f"{type(obj).__module__}.{type(obj).__qualname__}"


# Memo mapping object ids to stable tokens during a single fingerprint run.
# Cython extension objects (sklearn Trees, nodes, ...) reference each other
# cyclically, so recursion is broken by issuing content-addressed tokens.
_FINGERPRINT_MEMO = None


def _value_fingerprint(value):
    import numpy as np

    memo = _FINGERPRINT_MEMO
    if memo is not None and isinstance(value, (BaseEstimator, dict, list)):
        token = memo.get("id", {}).get(id(value))
        if token is not None:
            return "ref:" + token

    if isinstance(value, BaseEstimator):
        return "estimator:" + _estimator_fingerprint(value)
    if isinstance(value, np.ndarray):
        digest = hashlib.sha256()
        digest.update(str(value.shape).encode("utf-8"))
        digest.update(str(value.dtype).encode("utf-8"))
        if value.dtype.kind == "O":
            digest.update(repr(value.tolist()).encode("utf-8"))
        else:
            digest.update(np.ascontiguousarray(value).tobytes())
        return "ndarray:" + digest.hexdigest()
    # Sparse matrices without a hard scipy import in the signature.
    if hasattr(value, "data") and hasattr(value, "indices") and hasattr(
        value, "indptr"
    ):
        digest = hashlib.sha256()
        digest.update(_qualified_name(value).encode("utf-8"))
        digest.update(_value_fingerprint(value.data).encode("utf-8"))
        digest.update(_value_fingerprint(value.indices).encode("utf-8"))
        digest.update(_value_fingerprint(value.indptr).encode("utf-8"))
        return "sparse:" + digest.hexdigest()
    if isinstance(value, dict):
        token = None
        if memo is not None:
            token = memo["id"].get(id(value))
            if token is None:
                token = f"d{memo['counter']}"
                memo["counter"] += 1
                memo["id"][id(value)] = token
        rendered = "dict:" + hashlib.sha256(
            json.dumps(
                {
                    str(key): _value_fingerprint(item)
                    for key, item in value.items()
                },
                sort_keys=True,
            ).encode("utf-8")
        ).hexdigest()
        return rendered + (f"#{token}" if token else "")
    if isinstance(value, (list, tuple)):
        kind = type(value).__name__
        return f"{kind}:[" + ",".join(
            _value_fingerprint(item) for item in value
        ) + "]"
    if isinstance(value, (set, frozenset)):
        # Sorting the per-item digests removes set-order/hash-seed dependence.
        kind = type(value).__name__
        items = sorted(
            (_value_fingerprint(item) for item in value),
            key=repr,
        )
        return f"{kind}:{{" + "|".join(items) + "}"
    if isinstance(value, float):
        return "float:" + repr(value)
    if isinstance(value, (int, str, bool)) or value is None:
        return f"{type(value).__name__}:{value!r}"
    if hasattr(value, "__dict__"):
        return "object:" + _value_fingerprint(vars(value))
    # Extension types (Cython Tree, Node, ...): pickle is deterministic for
    # sklearn's extension classes and handles their internal cycles; we hash
    # the pickle stream rather than trusting repr(), which embeds addresses.
    try:
        stream = joblib.dump(value)
        if isinstance(stream, list):
            stream = b"".join(
                part if isinstance(part, bytes) else repr(part).encode("utf-8")
                for part in stream
            )
        digest = hashlib.sha256()
        digest.update(("extension:" + _qualified_name(value)).encode("utf-8"))
        digest.update(stream)
        return "extension:" + digest.hexdigest()
    except Exception:
        digest = hashlib.sha256()
        digest.update(("opaque:" + _qualified_name(value)).encode("utf-8"))
        return "opaque:" + digest.hexdigest()


class _fingerprint_scope:
    """Context manager installing the per-run cycle-breaking memo."""

    def __enter__(self):
        global _FINGERPRINT_MEMO
        self._previous = _FINGERPRINT_MEMO
        _FINGERPRINT_MEMO = {"id": {}, "counter": 0}
        return self

    def __exit__(self, *exc):
        global _FINGERPRINT_MEMO
        _FINGERPRINT_MEMO = self._previous
        return False


def _estimator_fingerprint(estimator):
    """SHA-256 over the fitted content of an estimator.

    The fingerprint covers the class name and every fitted attribute,
    including array shapes, dtypes and byte content, as well as nested
    estimators and Cython extension objects. Two unpickled objects produced
    from the same fit match; swapping weights or a nested estimator changes
    the digest. Cyclic references inside extension objects are handled by a
    per-run memo rather than object-address reprs.
    """
    with _fingerprint_scope():
        digest = hashlib.sha256()
        digest.update(_qualified_name(estimator).encode("utf-8"))
        for key in sorted(vars(estimator)):
            digest.update(("\x1f" + key).encode("utf-8"))
            digest.update(
                _value_fingerprint(vars(estimator)[key]).encode("utf-8")
            )
        return digest.hexdigest()


def _iter_nested_estimators(estimator, prefix=""):
    """Yield ``(path, nested_estimator)`` pairs, recursively and deterministically."""
    for key in sorted(vars(estimator)):
        value = vars(estimator)[key]
        path = f"{prefix}{key}" if not prefix else f"{prefix}.{key}"
        for nested_path, nested in _iter_value(value, path):
            yield nested_path, nested


def _iter_value(value, path):
    if isinstance(value, BaseEstimator):
        yield path, value
        yield from _iter_nested_estimators(value, path)
    elif isinstance(value, dict):
        for key in sorted(value, key=repr):
            yield from _iter_value(value[key], f"{path}.{key}")
    elif isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            # Pipeline.steps is a list of (name, estimator) tuples: surface
            # the user-facing step name rather than an opaque tuple index.
            if (
                isinstance(item, tuple)
                and len(item) == 2
                and isinstance(item[1], BaseEstimator)
                and isinstance(item[0], str)
            ):
                yield from _iter_value(item[1], f"{path}.{item[0]}")
            else:
                yield from _iter_value(item, f"{path}[{index}]")


# ---------------------------------------------------------------------------
# Parameter summaries and random seeds
# ---------------------------------------------------------------------------


def _summarize_params(estimator):
    try:
        params = estimator.get_params(deep=False)
    except Exception:
        return {}, {}
    summary = {}
    seeds = {}
    for key, value in params.items():
        summary[key] = _json_safe(value)
        if "random_state" in key or key == "seed":
            if value is None:
                seeds[key] = None
            elif isinstance(value, BaseEstimator) or isinstance(value, (int, str)):
                seeds[key] = _json_safe(value)
            else:
                seeds[key] = f"<{_qualified_name(value)} unseeded>"
    return summary, seeds


# ---------------------------------------------------------------------------
# Contract construction
# ---------------------------------------------------------------------------


def build_contract(estimator, X, *, missing_policy="auto", fit_id=None):
    """Build a :class:`ModelContract` for a freshly fitted estimator.

    The contract must be built from the same fit as ``estimator``: the
    estimator fingerprint stored in the contract is recomputed at load time
    and must match the bundled weights, which prevents attaching an old
    contract to different weights or inferring a contract from new data.

    Parameters
    ----------
    estimator : estimator object
        Fitted estimator (must inherit from :class:`~sklearn.base.BaseEstimator`).
    X : {array-like, dataframe} of shape (n_samples, n_features)
        Training matrix; only structural information (names, dtypes,
        vocabularies, counts) is recorded.
    missing_policy : {'auto', 'error', 'allow'} or dict, default='auto'
        Per-feature missing value policy. ``'auto'`` records ``'allow'`` for
        features that contained missing values at fit time and ``'error'``
        otherwise. A dict maps feature names to one of these policies.
    fit_id : str or None, default=None
        Shared fit identifier; a UUID is generated when omitted. The same
        identifier is propagated to every nested estimator contract.

    Returns
    -------
    contract : ModelContract
    """
    if not isinstance(estimator, BaseEstimator):
        raise TypeError(
            "build_contract expects a scikit-learn estimator inheriting from "
            f"BaseEstimator, got {type(estimator).__name__}"
        )

    fit_id = fit_id or str(uuid.uuid4())
    features = _extract_feature_specs(X, missing_policy)
    environment = _capture_environment()
    param_summary, random_seeds = _summarize_params(estimator)

    sub_contracts = []
    for path, nested in _iter_nested_estimators(estimator):
        nested_params, nested_seeds = _summarize_params(nested)
        sub_contracts.append(
            {
                "path": path,
                "fit_id": fit_id,
                "estimator_class": _qualified_name(nested),
                "estimator_fingerprint": _estimator_fingerprint(nested),
                "param_summary": nested_params,
                "random_seeds": nested_seeds,
            }
        )

    return ModelContract(
        contract_version=CONTRACT_VERSION,
        format=_FORMAT,
        fit_id=fit_id,
        estimator_class=_qualified_name(estimator),
        estimator_fingerprint=_estimator_fingerprint(estimator),
        features=[asdict(spec) for spec in features],
        environment=environment,
        param_summary=param_summary,
        random_seeds=random_seeds,
        sub_contracts=sub_contracts,
    )


# ---------------------------------------------------------------------------
# Checksummed JSON envelope
# ---------------------------------------------------------------------------


def _canonical_json(payload):
    return json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        default=_json_safe,
    ).encode("utf-8")


def _make_envelope(contract):
    payload = _canonical_json(contract.to_dict())
    return {
        "format": _FORMAT,
        "contract_version": contract.contract_version,
        "checksum": hashlib.sha256(payload).hexdigest(),
        "payload": payload.decode("utf-8"),
        "migrated_from": contract.migrated_from,
    }


def _parse_envelope(envelope):
    if not isinstance(envelope, dict) or "payload" not in envelope or (
        "checksum" not in envelope
    ):
        raise ContractCorruptionError(
            "contract envelope is missing required fields 'payload'/'checksum'; "
            "refusing to load weights without an intact contract"
        )
    payload = envelope["payload"]
    if not isinstance(payload, str):
        raise ContractCorruptionError(
            "contract payload is truncated or has an unexpected type; "
            "refusing to load weights without an intact contract"
        )
    actual = hashlib.sha256(payload.encode("utf-8")).hexdigest()
    if actual != envelope["checksum"]:
        raise ContractCorruptionError(
            "contract checksum mismatch: the envelope has been modified or "
            "truncated, so its provenance cannot be established; refusing to "
            "load weights without an intact contract"
        )
    try:
        data = json.loads(payload)
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise ContractCorruptionError(
            f"contract payload cannot be decoded ({exc}); refusing to load "
            "weights without an intact contract"
        ) from exc
    return data


def _contract_from_dict(data):
    # Legacy schemas predate parts of the current ModelContract. Unpickling
    # still has to succeed so migrate_contract can read and reject them;
    # structural integrity is already guaranteed by the envelope checksum.
    version = data.get("contract_version")
    try:
        features = [FeatureSpec(**spec) for spec in data["features"]]
    except (KeyError, TypeError) as exc:
        if version in SUPPORTED_CONTRACT_VERSIONS:
            raise ContractCorruptionError(
                f"a supported contract is missing feature fields ({exc}); "
                "refusing to load weights without an intact contract"
            ) from exc
        features = []
    if version not in SUPPORTED_CONTRACT_VERSIONS:
        return ModelContract(
            contract_version=version,
            format=data.get("format", _FORMAT),
            fit_id=data.get("fit_id", ""),
            estimator_class=data.get("estimator_class", "unknown"),
            estimator_fingerprint=data.get("estimator_fingerprint", ""),
            features=features,
            environment=data.get("environment", {}),
            param_summary=data.get("param_summary", {}),
            random_seeds=data.get("random_seeds", {}),
            sub_contracts=data.get("sub_contracts", []),
            migrated_from=data.get("migrated_from"),
        )
    return ModelContract(
        contract_version=data["contract_version"],
        format=data.get("format", _FORMAT),
        fit_id=data["fit_id"],
        estimator_class=data["estimator_class"],
        estimator_fingerprint=data["estimator_fingerprint"],
        features=features,
        environment=data["environment"],
        param_summary=data.get("param_summary", {}),
        random_seeds=data.get("random_seeds", {}),
        sub_contracts=data.get("sub_contracts", []),
        migrated_from=data.get("migrated_from"),
    )


# ---------------------------------------------------------------------------
# Structural verification (always runs, including on unpickle)
# ---------------------------------------------------------------------------


def _verify_structure(estimator, contract):
    """Verify fingerprints, fit ids and the nested estimator tree."""
    # Legacy (pre-supported) contracts skip structural fingerprint matching:
    # the migrator decides whether their recorded provenance is sufficient.
    if contract.contract_version not in SUPPORTED_CONTRACT_VERSIONS:
        return
    actual_fingerprint = _estimator_fingerprint(estimator)
    if actual_fingerprint != contract.estimator_fingerprint:
        raise ContractCorruptionError(
            "estimator fingerprint does not match the contract: the weights "
            "and the contract do not come from the same fit"
        )
    nested = dict(_iter_nested_estimators(estimator))
    if len(nested) != len(contract.sub_contracts):
        raise ContractCorruptionError(
            "nested estimator tree does not match the contract: "
            f"{len(nested)} fitted estimators found, "
            f"{len(contract.sub_contracts)} recorded"
        )
    for sub in contract.sub_contracts:
        path = sub["path"]
        if path not in nested:
            raise ContractCorruptionError(
                f"nested estimator at {path!r} is recorded in the contract "
                "but missing from the bundled estimator"
            )
        if sub["fit_id"] != contract.fit_id:
            raise ContractCorruptionError(
                f"nested estimator at {path!r} carries fit_id "
                f"{sub['fit_id']!r} which differs from the outer fit_id; "
                "inner and outer contracts do not describe the same fit"
            )
        actual = _estimator_fingerprint(nested[path])
        if actual != sub["estimator_fingerprint"]:
            raise ContractCorruptionError(
                f"nested estimator at {path!r}: fingerprint mismatch between "
                "the sub-contract and the bundled estimator"
            )


def _check_contract_version(version):
    if version in SUPPORTED_CONTRACT_VERSIONS:
        return
    if isinstance(version, str) and version == OLDEST_MIGRATABLE_VERSION:
        raise ContractMigrationError(
            f"contract version {version!r} predates the supported range "
            f"{SUPPORTED_CONTRACT_VERSIONS}; run migrate_contract on the "
            "artifact before loading it"
        )
    raise ContractVersionError(
        f"contract version {version!r} is not supported by this installation; "
        f"supported versions are {SUPPORTED_CONTRACT_VERSIONS} "
        f"(oldest migratable version: {OLDEST_MIGRATABLE_VERSION})"
    )


# ---------------------------------------------------------------------------
# Environment and feature verification
# ---------------------------------------------------------------------------


def _verify_environment(contract, findings):
    current = _capture_environment()
    for package, expected in contract.environment.items():
        actual = current.get(package)
        if expected is None:
            continue
        if actual is None:
            findings.append(
                Finding(
                    "error",
                    "version.missing_dependency",
                    f"dependency {package!r} ({expected!r} at fit time) is not "
                    "installed in the current environment"
                )
            )
            continue
        _compare_versions(expected, actual, package, findings)


def _feature_specs(contract):
    return [
        spec if isinstance(spec, FeatureSpec) else FeatureSpec(**spec)
        for spec in contract.features
    ]


def _verify_features(contract, X, findings):
    """Collect every feature-level difference in a single pass."""
    if X is None:
        findings.append(
            Finding(
                "info",
                "feature.check_skipped",
                "no input data provided; only the stored contract was "
                "structurally verified"
            )
        )
        return None, False

    current_names, current = _current_feature_summary(X)
    expected_specs = _feature_specs(contract)
    expected_names = [spec.name for spec in expected_specs]
    expected = {spec.name: spec for spec in expected_specs}

    missing = [name for name in expected_names if name not in current]
    unexpected = [name for name in current_names if name not in expected]
    for name in missing:
        findings.append(
            Finding(
                "error",
                "feature.missing",
                f"column {name!r} (dtype {expected[name].dtype!r}) is required "
                "by the contract but absent from the current input"
            )
        )
    for name in unexpected:
        findings.append(
            Finding(
                "error",
                "feature.unexpected",
                f"column {name!r} (dtype {current[name][0]!r}) is present in "
                "the current input but unknown to the contract"
            )
        )

    alignment = None
    reordered = False
    if not missing and not unexpected:
        alignment = {name: expected[name].position for name in current_names}
        if current_names != expected_names:
            reordered = True
            findings.append(
                Finding(
                    "info",
                    "feature.reordered",
                    "feature names match but their order differs; columns "
                    "will be aligned by name rather than by position "
                    f"({len(current_names)} columns)"
                )
            )

    for name in current_names:
        if name not in expected:
            continue
        spec = expected[name]
        dtype, is_cat, categories, missing_count = current[name]

        if spec.migrated_dtype_unknown():
            findings.append(
                Finding(
                    "warning",
                    "feature.dtype_unknown",
                    f"column {name!r}: dtype was not recorded in the legacy "
                    f"contract; current dtype is {dtype!r} and cannot be "
                    "cross-checked"
                )
            )
        elif dtype != spec.dtype or is_cat != spec.is_categorical:
            findings.append(
                Finding(
                    "error",
                    "feature.dtype_mismatch",
                    f"column {name!r}: contract records dtype {spec.dtype!r} "
                    f"(categorical={spec.is_categorical}), current input has "
                    f"dtype {dtype!r} (categorical={is_cat})"
                )
            )

        if is_cat and spec.categories is not None:
            current_set = set(categories)
            expected_set = set(spec.categories)
            new_count = len(current_set - expected_set)
            absent_count = len(expected_set - current_set)
            if new_count:
                findings.append(
                    Finding(
                        "error",
                        "feature.category_drift",
                        f"column {name!r}: {new_count} category value(s) are "
                        "not in the fit-time vocabulary and would be encoded "
                        "as missing; prediction is refused"
                    )
                )
            if absent_count:
                findings.append(
                    Finding(
                        "info",
                        "feature.category_partial_coverage",
                        f"column {name!r}: {absent_count} fit-time category "
                        "value(s) do not appear in the current input"
                    )
                )

        if spec.missing_policy == "error" and missing_count > 0:
            findings.append(
                Finding(
                    "error",
                    "feature.unexpected_missing",
                    f"column {name!r}: {missing_count} missing value(s) found "
                    "but the fit-time policy forbids missing values"
                )
            )
        elif spec.missing_policy == "unknown":
            findings.append(
                Finding(
                    "warning",
                    "feature.missing_policy_unknown",
                    f"column {name!r}: missing value policy was not recorded "
                    "in the legacy contract"
                )
            )

    return alignment, reordered


def _build_report(contract, X, artifact_path):
    findings = []
    if contract.migrated_from is not None:
        findings.append(
            Finding(
                "warning",
                "contract.migrated",
                f"contract was migrated from legacy version "
                f"{contract.migrated_from!r}; fields absent from the legacy "
                "schema are marked as unknown rather than inferred"
            )
        )
    _verify_environment(contract, findings)
    alignment, reordered = _verify_features(contract, X, findings)

    if any(f.is_error for f in findings):
        status = "invalid"
    elif any(f.severity == "warning" for f in findings):
        status = "compatible"
    elif any(
        f.severity == "info" and f.code != "feature.check_skipped"
        for f in findings
    ):
        # Informational findings (notably name-based column realignment)
        # describe a loadable-but-not-identical situation, not a clean match.
        # The check-skipped note means verification simply had no data to
        # compare against; structural checks already passed.
        status = "compatible"
    else:
        status = "valid"
    return ValidationReport(
        status=status,
        findings=findings,
        alignment=alignment,
        reordered=reordered,
        artifact_path=artifact_path,
    )


# ---------------------------------------------------------------------------
# Feature alignment for prediction
# ---------------------------------------------------------------------------


def _align_input(X, alignment, reordered):
    """Reorder input columns by name according to the fit-time contract.

    When the columns already arrive in fit-time order the original object is
    returned untouched: an unnecessary copy would change the memory layout and
    make BLAS results differ by an ULP, breaking bit-exact reproduction.
    """
    if alignment is None or not reordered:
        return X
    if _is_pandas_dataframe(X):
        ordered = sorted(alignment, key=lambda name: alignment[name])
        return X[ordered]
    import numpy as np

    indices = np.array(
        [alignment[f"x{position}"] for position in range(X.shape[1])],
        dtype="int64",
    )
    return np.asarray(X)[:, indices]


# ---------------------------------------------------------------------------
# Wrapper exposed to users
# ---------------------------------------------------------------------------


class ContractedModel:
    """Fitted estimator bundled with a verified :class:`ModelContract`.

    Instances are normally created by :func:`make_contracted_model` or
    :func:`load_contracted_model`. Prediction methods first validate the
    current input against the contract (collecting every mismatch) and align
    columns by name; all other attributes delegate to the wrapped estimator,
    so existing summaries and parameter access keep working unchanged.
    """

    def __init__(self, estimator, contract, envelope):
        self.estimator_ = estimator
        self.contract_ = contract
        self._envelope = envelope
        self._last_report = None

    # -- persistence -------------------------------------------------------

    def __getstate__(self):
        return {
            "estimator": self.estimator_,
            "envelope": self._envelope,
        }

    def __setstate__(self, state):
        envelope = state.get("envelope")
        estimator = state.get("estimator")
        if envelope is None or estimator is None:
            raise ContractCorruptionError(
                "artifact is truncated: the contract envelope or the fitted "
                "estimator is missing; refusing to load weights without an "
                "intact contract"
            )
        try:
            # Only envelope integrity and estimator provenance are enforced
            # during unpickling. Contract-version checks are deferred to the
            # load/validate entry points so that migrate_contract can still
            # unpickle legacy (migratable) artifacts.
            data = _parse_envelope(envelope)
            contract = _contract_from_dict(data)
            _verify_structure(estimator, contract)
        except ContractError:
            raise
        except Exception as exc:
            raise ContractCorruptionError(
                f"contract cannot be restored ({type(exc).__name__}: {exc}); "
                "refusing to load weights without an intact contract"
            ) from exc
        self.estimator_ = estimator
        self.contract_ = contract
        self._envelope = envelope
        self._last_report = None

    # -- contract-aware prediction ----------------------------------------

    def validate(self, X=None):
        """Verify the artifact against the current environment and ``X``.

        Parameters
        ----------
        X : {array-like, dataframe} or None, default=None
            Current input data. When ``None``, only structural and
            environment checks run.

        Returns
        -------
        report : ValidationReport
        """
        self._last_report = _build_report(self.contract_, X, None)
        return self._last_report

    def _predict(self, method_name, X, *args, **kwargs):
        report = self.validate(X)
        report.raise_for_errors()
        aligned = _align_input(X, report.alignment, report.reordered)
        return getattr(self.estimator_, method_name)(aligned, *args, **kwargs)

    def __getattr__(self, name):
        if name in _BLOCKED_MUTATORS:
            raise ContractError(
                f"{name!r} is blocked on a ContractedModel: refitting or "
                "mutating it would invalidate the stored contract; build a "
                "new contract on a fresh fit instead"
            )
        if name in _PREDICT_METHODS:
            estimator = self.__dict__["estimator_"]
            attr = getattr(estimator, name)
            if callable(attr):
                def guarded(X, *args, **kwargs):
                    return self._predict(name, X, *args, **kwargs)

                guarded.__name__ = name
                return guarded
            return attr
        # Delegation keeps estimator summaries and parameter reads unchanged.
        return getattr(self.__dict__["estimator_"], name)

    def contract_summary(self):
        """Return a JSON-safe summary of the stored contract."""
        contract = self.contract_
        return {
            "contract_version": contract.contract_version,
            "fit_id": contract.fit_id,
            "estimator_class": contract.estimator_class,
            "n_features": len(contract.features),
            "feature_names": [
                spec["name"] if isinstance(spec, dict) else spec.name
                for spec in contract.features
            ],
            "environment": contract.environment,
            "random_seeds": contract.random_seeds,
            "n_sub_contracts": len(contract.sub_contracts),
            "migrated_from": contract.migrated_from,
        }

    def __repr__(self):
        return (
            f"ContractedModel(estimator={self.contract_.estimator_class}, "
            f"contract_version={self.contract_.contract_version}, "
            f"n_features={len(self.contract_.features)})"
        )


def make_contracted_model(estimator, X, *, missing_policy="auto", fit_id=None):
    """Bundle a fitted estimator with a freshly built contract in memory."""
    contract = build_contract(
        estimator, X, missing_policy=missing_policy, fit_id=fit_id
    )
    envelope = _make_envelope(contract)
    return ContractedModel(estimator, contract, envelope)


def save_contracted_model(estimator, path, X, *, missing_policy="auto",
                          fit_id=None, compress=0):
    """Save a fitted estimator and its verifiable contract to ``path``.

    The contract is built in the same call as the save, from the estimator and
    its training matrix, so weights and contract necessarily describe the same
    fit. The serialization format stays joblib-based; existing joblib tooling
    can still inspect the file, while loading through
    :func:`load_contracted_model` enforces verification.

    Parameters
    ----------
    estimator : estimator object
        Fitted estimator.
    path : str or path-like
        Destination file.
    X : {array-like, dataframe}
        Training matrix used to fit ``estimator``.
    missing_policy : {'auto', 'error', 'allow'} or dict, default='auto'
        See :func:`build_contract`.
    fit_id : str or None, default=None
        Shared fit identifier.
    compress : int, default=0
        Joblib compression level, forwarded unchanged.

    Returns
    -------
    path : str
    """
    model = make_contracted_model(
        estimator, X, missing_policy=missing_policy, fit_id=fit_id
    )
    joblib.dump(model, os.fspath(path), compress=compress)
    return os.fspath(path)


# ---------------------------------------------------------------------------
# Loading and validation
# ---------------------------------------------------------------------------


def _load_raw(path):
    try:
        obj = joblib.load(os.fspath(path))
    except FileNotFoundError:
        raise
    except Exception as exc:
        raise ContractCorruptionError(
            f"artifact cannot be deserialized ({type(exc).__name__}: {exc}); "
            "the file is truncated or otherwise corrupted, and no weights "
            "will be returned"
        ) from exc
    return obj


def load_contracted_model(path, *, X=None):
    """Load a contracted artifact after full verification.

    Structural verification (envelope checksum, contract version, estimator
    fingerprints, nested estimator tree) runs while unpickling and cannot be
    bypassed. Environment and feature verification against ``X`` then run in a
    single pass; on any error :class:`ContractValidationError` is raised with
    the complete report and no predictions can be produced.

    Parameters
    ----------
    path : str or path-like
        Artifact produced by :func:`save_contracted_model` (or migrated by
        :func:`migrate_contract`).
    X : {array-like, dataframe} or None, default=None
        Current input data to check against the feature contract.

    Returns
    -------
    model : ContractedModel
    report : ValidationReport
    """
    obj = _load_raw(path)
    if not isinstance(obj, ContractedModel):
        raise ContractError(
            f"{os.fspath(path)!r} is not a contracted artifact; use "
            "read_contract to inspect legacy artifacts"
        )
    _check_contract_version(obj._envelope.get("contract_version"))
    report = _build_report(obj.contract_, X, os.fspath(path))
    obj._last_report = report
    report.raise_for_errors()
    return obj, report


def validate_artifact(path, *, X=None):
    """Validate an artifact and return the full :class:`ValidationReport`.

    Unlike :func:`load_contracted_model`, feature-level findings do not raise;
    the returned report's ``status`` and ``ok`` attributes carry the verdict.
    Corruption and unsupported contract versions still raise.
    """
    obj = _load_raw(path)
    if not isinstance(obj, ContractedModel):
        raise ContractError(
            f"{os.fspath(path)!r} is not a contracted artifact; use "
            "read_contract to inspect legacy artifacts"
        )
    _check_contract_version(obj._envelope.get("contract_version"))
    return _build_report(obj.contract_, X, os.fspath(path))


def read_contract(path):
    """Read basic contract information without constructing predictions.

    Contracted artifacts return a JSON-safe contract summary. Legacy
    artifacts (plain joblib pickles without a contract) remain readable:
    basic estimator information is returned and a
    :class:`LegacyArtifactWarning` is emitted. Nothing is inferred from
    current data.
    """
    obj = _load_raw(path)
    if isinstance(obj, ContractedModel):
        return obj.contract_summary()

    warnings.warn(
        f"{os.fspath(path)!r} is a legacy artifact without a verifiable "
        "contract; only basic estimator information can be shown and no "
        "load-time guarantees are available",
        LegacyArtifactWarning,
    )
    info = {
        "contract_format": "legacy",
        "estimator_class": _qualified_name(obj),
        "artifact_path": os.fspath(path),
    }
    sklearn_version = vars(obj).get("_sklearn_version")
    if sklearn_version is not None:
        info["sklearn_version_at_fit"] = sklearn_version
    params = None
    try:
        params = obj.get_params(deep=False)
    except Exception:
        pass
    if params:
        info["param_summary"] = {
            key: _json_safe(value) for key, value in params.items()
        }
    return info


# ---------------------------------------------------------------------------
# Migration
# ---------------------------------------------------------------------------


def _migrate_09_to_1(data, envelope):
    """Migrate a schema 0.9 contract (feature names only) to schema 1.

    The old schema recorded feature names, dependencies and a fit id but no
    dtypes, vocabularies or missing policies. Those fields cannot be
    reconstructed without the original training data, and contracts must not
    be inferred at load time; they are therefore marked ``"unknown"`` and
    reported as warnings during verification rather than silently invented.
    """
    if not isinstance(data, dict) or "features" not in data or (
        "fit_id" not in data
    ):
        raise ContractMigrationError(
            "legacy 0.9 contract is missing required fields ('features', "
            "'fit_id'); migration is impossible and the original file is "
            "left untouched"
        )
    migrated_features = []
    for position, feature in enumerate(data["features"]):
        name = feature["name"] if isinstance(feature, dict) else str(feature)
        migrated_features.append(
            asdict(
                FeatureSpec(
                    name=name,
                    position=position,
                    dtype="unknown",
                    is_categorical=False,
                    categories=None,
                    missing_policy="unknown",
                    missing_count=0,
                )
            )
        )
    return ModelContract(
        contract_version=CONTRACT_VERSION,
        format=_FORMAT,
        fit_id=data["fit_id"],
        estimator_class=data.get("estimator_class", "unknown"),
        estimator_fingerprint=data.get("estimator_fingerprint", ""),
        features=migrated_features,
        environment=data.get("environment", {}),
        param_summary=data.get("param_summary", {}),
        random_seeds=data.get("random_seeds", {}),
        sub_contracts=data.get("sub_contracts", []),
        migrated_from="0.9",
    )


_MIGRATORS = {"0.9": _migrate_09_to_1}


def migrate_contract(path, output_path):
    """Migrate a legacy contracted artifact to the current contract version.

    The source file is opened read-only. The migrated artifact is first
    written to a temporary file in the destination directory and only moved
    into place once migration and checksum regeneration fully succeed; a
    failed migration therefore never leaves a half-written file behind and
    never overwrites the source.

    Parameters
    ----------
    path : str or path-like
        Source artifact.
    output_path : str or path-like
        Destination path for the migrated artifact.

    Returns
    -------
    output_path : str
    """
    path = os.fspath(path)
    output_path = os.fspath(output_path)

    obj = _load_raw(path)
    if not isinstance(obj, ContractedModel):
        raise ContractMigrationError(
            f"{path!r} is a plain legacy artifact without a contract envelope; "
            "there is no contract to migrate and the original file is left "
            "untouched"
        )

    envelope = obj._envelope
    version = envelope.get("contract_version")
    # Migration is the one entry point allowed to unpickle contracts outside
    # the supported range; corruption and fingerprint checks still run during
    # unpickling itself.
    if version in SUPPORTED_CONTRACT_VERSIONS:
        raise ContractMigrationError(
            f"contract version {version!r} is already supported; migration "
            "is unnecessary and the original file is left untouched"
        )
    migrator = _MIGRATORS.get(str(version))
    if migrator is None:
        raise ContractMigrationError(
            f"no migration path exists for contract version {version!r}; "
            f"supported versions are {SUPPORTED_CONTRACT_VERSIONS} and the "
            f"oldest migratable version is {OLDEST_MIGRATABLE_VERSION}; the "
            "original file is left untouched"
        )

    try:
        data = _parse_envelope(envelope)
        contract = migrator(data, envelope)
        # Re-verify what can be checked: keep the original estimator paired
        # with the migrated envelope only when a fingerprint was recorded.
        if contract.estimator_fingerprint:
            _verify_structure(obj.estimator_, contract)
        migrated = ContractedModel(obj.estimator_, contract, _make_envelope(contract))
    except ContractError:
        raise
    except Exception as exc:
        raise ContractMigrationError(
            f"migration failed ({type(exc).__name__}: {exc}); the original "
            "file is left untouched"
        ) from exc

    destination_dir = os.path.dirname(os.path.abspath(output_path)) or "."
    os.makedirs(destination_dir, exist_ok=True)
    fd, temporary_path = tempfile.mkstemp(
        prefix=".contract-migration-", suffix=".tmp", dir=destination_dir
    )
    os.close(fd)
    try:
        joblib.dump(migrated, temporary_path)
        # Round-trip the temporary artifact before replacing anything.
        reloaded = joblib.load(temporary_path)
        if not isinstance(reloaded, ContractedModel):
            raise ContractMigrationError(
                "migrated artifact failed its round-trip verification; the "
                "original file is left untouched"
            )
        os.replace(temporary_path, output_path)
    except BaseException:
        if os.path.exists(temporary_path):
            os.unlink(temporary_path)
        raise
    return output_path


# FeatureSpec.migrated_dtype_unknown is defined as a regular method above so
# dataclass (de)serialization stays plain.
