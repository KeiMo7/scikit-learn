"""Export, load, replay and diff the state of a fitted pipeline.

This private module implements the machinery behind
:meth:`sklearn.pipeline.Pipeline.export_chain_state`,
:meth:`sklearn.pipeline.Pipeline.load_chain_state`,
:meth:`sklearn.pipeline.Pipeline.replay_chain_state` and
:meth:`sklearn.pipeline.Pipeline.diff_chain_state`.

The state of a fitted chain is a description of *structure and parameters*:

* for every step, the fully qualified class name and the constructor
  parameters (``get_params(deep=False)``);
* the attributes set by ``fit`` (the contents of ``__dict__`` that are not
  constructor parameters), encoded recursively, so that nested estimators
  (``ColumnTransformer.transformers_``, ensembles, ...) are described too;
* the input dimensionality of the step, its output dimensionality when it can
  be inferred from the fitted attributes, and the step it depends on;
* the randomness it consumed: seeded constructor parameters and, when the
  estimator exposes it, the seed it derived from them.

Raw training data is never written.  Estimators that keep a copy of the
training matrix in a fitted attribute (``KNNImputer``, ``KernelRidge``,
Gaussian processes, ...) are flagged as *non-exclusive*: the copy is left out
of the document and such a step can only be restored by re-fitting it on the
caller's data.  Arrays are stored as exact bytes together with their dtype,
shape and memory layout, so loading the state reproduces the fitted estimator
bit for bit without calling ``fit`` again.

The document is a JSON Lines stream::

    {"segment": "header", ...}
    {"segment": "step", "order": 0, ...}
    ...
    {"segment": "end", "n_steps": N, ...}

Each line carries its own SHA-256 checksum, so a corrupted or truncated file
is reported with the exact segment that is broken.  Steps are written and read
one at a time, which keeps the memory footprint of an export independent of the
number of steps.  Writing goes through a temporary file that is atomically
renamed when the document is complete, so a failed export never leaves a
partial file behind.
"""

# Authors: The scikit-learn developers
# SPDX-License-Identifier: BSD-3-Clause

import base64
import hashlib
import json
import numbers
import os
import tempfile
import types

import numpy as np

from sklearn.exceptions import (
    ChainStateCorruptionError,
    ChainStateDataError,
    ChainStateMigrationError,
    ChainStateVersionError,
)

#: Version of the on-disk layout.  Bump the major number on incompatible
#: changes, the minor number on purely additive ones.
FORMAT_VERSION = (1, 0)
FORMAT_NAME = "sklearn-chain-state"

#: Supported on-disk versions.  Documents written by any of these versions can
#: be read back; anything else raises :class:`ChainStateVersionError`.
SUPPORTED_VERSIONS = {(1, 0)}

#: Default comparison tolerances used by replay and by the diff report.  Both
#: are absolute and relative, mirroring :func:`numpy.allclose`, and are written
#: into the document so that a replay never relies on an implicit convention.
DEFAULT_ATOL = 1e-9
DEFAULT_RTOL = 1e-7

#: Fitted attributes that hold a copy of the training matrix rather than a
#: model parameter.  They are never exported.
_TRAINING_DATA_ATTRIBUTES = frozenset(
    {
        "_fit_X",
        "_mask_fit_X",
        "_X_fit",
        "X_fit_",
        "fit_X_",
        "X_train_",
        "y_train_",
    }
)

#: Constructor parameters that make a step's fitted state depend on a seed.
_SEED_PARAMETERS = ("random_state", "seed", "random_seed")


# ---------------------------------------------------------------------------
# Object references
# ---------------------------------------------------------------------------


def _class_ref(obj):
    cls = type(obj)
    return f"{cls.__module__}.{cls.__qualname__}"


def _resolve_ref(ref):
    module_name, _, qualname = ref.partition(":")
    try:
        module = __import__(module_name, fromlist=["_"])
    except ImportError as exc:
        raise ChainStateMigrationError(
            f"the module {module_name!r} recorded in the state does not exist in "
            "this version of scikit-learn",
            fragment=ref,
            suggestion=(
                "The estimator was moved or removed. Install the version that "
                "exported the state, or replace the step and export again."
            ),
        ) from exc
    obj = module
    for part in qualname.split("."):
        try:
            obj = getattr(obj, part)
        except AttributeError as exc:
            raise ChainStateMigrationError(
                f"{qualname!r} does not exist in {module_name!r} in this version "
                "of scikit-learn",
                fragment=ref,
                suggestion=(
                    "The estimator, or one of its public attributes, was renamed "
                    "or removed. Check the changelog of the release that "
                    "introduced the break, replace the step with its new "
                    "equivalent, or pin the version that exported the state."
                ),
            ) from exc
    return obj


def _ref_of(module, qualname):
    return f"{module}:{qualname}"


# ---------------------------------------------------------------------------
# Arrays and dtypes
# ---------------------------------------------------------------------------


def _dtype_descr(dtype):
    return [
        [name, np.dtype(fmt).str] + ([list(shape)] if shape else [])
        for name, fmt, *rest in dtype.descr
        for shape in (rest[0] if rest else None,)
    ]


def _dtype_spec(dtype):
    dtype = np.dtype(dtype)
    if dtype.fields is not None:
        return {"descr": _dtype_descr(dtype)}
    return {"str": dtype.str}


def _dtype_from_spec(spec):
    if "descr" in spec:
        descr = [
            (entry[0], entry[1], tuple(entry[2]))
            if len(entry) == 3
            else (entry[0], entry[1])
            for entry in spec["descr"]
        ]
        return np.lib.format.descr_to_dtype(descr)
    return np.dtype(spec["str"])


def _encode_array(array):
    array = np.asarray(array)
    if array.dtype.hasobject:
        return {
            "kind": "object_array",
            "dtype": _dtype_spec(array.dtype),
            "shape": list(array.shape),
            "items": [_encode_value(item) for item in array.ravel(order="C").tolist()],
        }
    # Keep the dtype byte order: the raw bytes plus the dtype reproduce the
    # exact values, including for non-native byte orders.
    #
    # The memory layout is recorded too.  It looks irrelevant, but the BLAS
    # backend used by a matrix product takes different code paths for C and
    # Fortran ordered operands, and some scikit-learn estimators store their
    # coefficients in Fortran order on purpose.  Restoring the layout is what
    # makes a replay bit-for-bit identical instead of merely close.
    if array.flags["F_CONTIGUOUS"] and not array.flags["C_CONTIGUOUS"]:
        order = "F"
        contiguous = np.asfortranarray(array)
    else:
        order = "C"
        contiguous = np.ascontiguousarray(array)
    raw = contiguous.tobytes(order=order)
    return {
        "kind": "array",
        "dtype": _dtype_spec(array.dtype),
        "shape": list(array.shape),
        "order": order,
        "data": base64.b64encode(raw).decode("ascii"),
    }


def _decode_array(spec):
    dtype = _dtype_from_spec(spec["dtype"])
    raw = base64.b64decode(spec["data"].encode("ascii"))
    shape = tuple(spec["shape"])
    expected = int(np.prod(shape, dtype=np.int64)) * dtype.itemsize
    if len(raw) != expected:
        raise ValueError(
            f"payload holds {len(raw)} bytes, expected {expected} for shape "
            f"{shape} and dtype {dtype}"
        )
    array = np.frombuffer(raw, dtype=dtype).reshape(shape, order=spec.get("order", "C"))
    return array.copy(order=spec.get("order", "C"))


# ---------------------------------------------------------------------------
# Generic value codec
# ---------------------------------------------------------------------------


def _encode_value(value):
    """Encode an arbitrary Python value into a JSON-able structure."""
    from scipy import sparse

    from sklearn.base import BaseEstimator

    if value is None or isinstance(value, str):
        return value
    if isinstance(value, bytes):
        return {"kind": "bytes", "data": base64.b64encode(value).decode("ascii")}
    if isinstance(value, (bool, np.bool_)):
        return {"kind": "bool", "value": bool(value)}
    if isinstance(value, np.generic):
        return {"kind": "scalar", "dtype": value.dtype.str, "value": value.item()}
    if isinstance(value, numbers.Integral):
        return {"kind": "int", "value": int(value)}
    if isinstance(value, numbers.Real):
        return {"kind": "float", "value": float(value)}
    if isinstance(value, numbers.Complex):
        return {"kind": "complex", "real": value.real, "imag": value.imag}
    if isinstance(value, np.ndarray):
        return _encode_array(value)
    if sparse.issparse(value):
        value = value.tocsr()
        return {
            "kind": "sparse",
            "format": value.format,
            "shape": list(value.shape),
            "dtype": _dtype_spec(value.dtype),
            "data": _encode_array(value.data),
            "indices": _encode_array(value.indices),
            "indptr": _encode_array(value.indptr),
        }
    if isinstance(value, (list, tuple)):
        return {
            "kind": "tuple" if isinstance(value, tuple) else "list",
            "items": [_encode_value(item) for item in value],
        }
    if isinstance(value, (set, frozenset)):
        return {
            "kind": "frozenset" if isinstance(value, frozenset) else "set",
            "items": [_encode_value(item) for item in sorted(value, key=repr)],
        }
    if isinstance(value, dict):
        return {
            "kind": "dict",
            "items": [
                [_encode_value(key), _encode_value(item)] for key, item in value.items()
            ],
        }
    if isinstance(value, slice):
        return {
            "kind": "slice",
            "start": value.start,
            "stop": value.stop,
            "step": value.step,
        }
    if isinstance(value, np.dtype):
        return {"kind": "dtype", "spec": _dtype_spec(value)}
    if isinstance(value, type):
        return {
            "kind": "type",
            "ref": _ref_of(value.__module__, value.__qualname__),
        }
    if isinstance(value, np.ufunc):
        return {"kind": "ufunc", "name": value.__name__}
    if isinstance(value, (types.FunctionType, types.BuiltinFunctionType)):
        return {
            "kind": "function",
            "ref": _ref_of(value.__module__, value.__qualname__),
        }
    if isinstance(value, BaseEstimator):
        return {"kind": "estimator", "state": _encode_estimator(value)}
    if isinstance(value, np.random.Generator):
        return {
            "kind": "generator",
            "bit_generator": type(value.bit_generator).__name__,
            "state": _encode_value(dict(value.bit_generator.state)),
        }
    if isinstance(value, np.random.RandomState):
        return {
            "kind": "random_state",
            "state": _encode_value(
                {
                    "bit_generator": value.get_state()[0],
                    "keys": value.get_state()[1],
                    "pos": int(value.get_state()[2]),
                    "has_gauss": int(value.get_state()[3]),
                    "cached_gaussian": float(value.get_state()[4]),
                }
            ),
        }
    if hasattr(value, "get_params") and hasattr(value, "set_params"):
        # Parameter objects such as `make_column_selector()` instances: there
        # is no fitted state, only constructor parameters.
        return {
            "kind": "param_object",
            "cls": _ref_of(type(value).__module__, type(value).__qualname__),
            "params": _encode_value(dict(value.get_params(deep=False))),
        }
    return _encode_opaque(value)


def _encode_opaque(value):
    """Encode an extension-type object through the pickle reduce protocol.

    Only objects whose reconstruction function is importable are accepted.
    The state itself is encoded with the same recursive codec, so arrays are
    stored explicitly rather than as opaque pickle bytes.
    """
    try:
        reduced = value.__reduce_ex__(4)
    except Exception as exc:  # pragma: no cover - defensive
        raise TypeError(
            f"cannot export value of type {type(value).__name__}: {exc}"
        ) from exc
    if not isinstance(reduced, tuple) or len(reduced) < 2:
        raise TypeError(f"cannot export value of type {type(value).__name__}")
    func, args = reduced[0], reduced[1]
    state = reduced[2] if len(reduced) > 2 else None
    listitems = reduced[3] if len(reduced) > 3 else None
    dictitems = reduced[4] if len(reduced) > 4 else None
    if listitems or dictitems:
        raise TypeError(
            f"cannot export value of type {type(value).__name__}: it is reduced "
            "through item iterators"
        )
    if isinstance(func, type):
        func_spec = {"kind": "type", "ref": _ref_of(func.__module__, func.__qualname__)}
    elif callable(func):
        func_spec = {
            "kind": "function",
            "ref": _ref_of(func.__module__, func.__qualname__),
        }
    else:  # pragma: no cover - defensive
        raise TypeError(
            f"cannot export value of type {type(value).__name__}: its reduction "
            "function is not callable"
        )
    return {
        "kind": "opaque",
        "class": _ref_of(type(value).__module__, type(value).__qualname__),
        "func": func_spec,
        "args": _encode_value(tuple(args)),
        "state": _encode_value(state),
    }


def _decode_value(spec):
    if spec is None or isinstance(spec, str):
        return spec
    if not isinstance(spec, dict) or "kind" not in spec:
        raise TypeError(f"malformed encoded value: {spec!r}")
    kind = spec["kind"]
    if kind == "bytes":
        return base64.b64decode(spec["data"].encode("ascii"))
    if kind == "bool":
        return bool(spec["value"])
    if kind == "scalar":
        return np.dtype(spec["dtype"]).type(spec["value"])
    if kind == "int":
        return int(spec["value"])
    if kind == "float":
        return float(spec["value"])
    if kind == "complex":
        return complex(spec["real"], spec["imag"])
    if kind == "array":
        return _decode_array(spec)
    if kind == "object_array":
        items = [_decode_value(item) for item in spec["items"]]
        array = np.empty(len(items), dtype=object)
        array[:] = items
        return array.reshape(tuple(spec["shape"]))
    if kind == "sparse":
        from scipy import sparse as sp

        matrix = sp.csr_matrix(
            (
                _decode_array(spec["data"]),
                _decode_array(spec["indices"]),
                _decode_array(spec["indptr"]),
            ),
            shape=tuple(spec["shape"]),
        )
        return matrix.asformat(spec["format"])
    if kind in ("list", "tuple"):
        items = [_decode_value(item) for item in spec["items"]]
        return tuple(items) if kind == "tuple" else items
    if kind in ("set", "frozenset"):
        items = [_decode_value(item) for item in spec["items"]]
        return frozenset(items) if kind == "frozenset" else set(items)
    if kind == "dict":
        return {_decode_value(key): _decode_value(item) for key, item in spec["items"]}
    if kind == "slice":
        return slice(spec["start"], spec["stop"], spec["step"])
    if kind == "dtype":
        return _dtype_from_spec(spec["spec"])
    if kind == "type":
        return _resolve_ref(spec["ref"])
    if kind == "ufunc":
        return getattr(np, spec["name"])
    if kind == "function":
        return _resolve_ref(spec["ref"])
    if kind == "estimator":
        return _decode_estimator(spec["state"])
    if kind == "generator":
        bit_generator = getattr(np.random, spec["bit_generator"])()
        generator = np.random.Generator(bit_generator)
        generator.bit_generator.state = _decode_value(spec["state"])
        return generator
    if kind == "random_state":
        state = _decode_value(spec["state"])
        random_state = np.random.RandomState()
        random_state.set_state(
            (
                state["bit_generator"],
                np.asarray(state["keys"], dtype=np.uint32),
                state["pos"],
                state["has_gauss"],
                state["cached_gaussian"],
            )
        )
        return random_state
    if kind == "param_object":
        cls = _resolve_ref(spec["cls"])
        return cls(**_decode_value(spec["params"]))
    if kind == "opaque":
        return _decode_opaque(spec)
    raise TypeError(f"unknown encoded value kind {kind!r}")


def _decode_opaque(spec):
    func = _decode_value(spec["func"])
    args = _decode_value(spec["args"])
    state = _decode_value(spec["state"])
    obj = func(*args)
    if state is not None:
        if hasattr(obj, "__setstate__"):
            obj.__setstate__(state)
        elif isinstance(state, dict):
            obj.__dict__.update(state)
        else:  # pragma: no cover - defensive
            raise TypeError(
                f"cannot restore the state of {spec['class']}: it is neither a "
                "dictionary nor accepted by __setstate__"
            )
    return obj


# ---------------------------------------------------------------------------
# Estimators
# ---------------------------------------------------------------------------


def _estimator_entry_label(entry):
    """Return the estimator held by a ``(name, estimator, ...)`` entry, if any."""
    if not isinstance(entry, dict) or entry.get("kind") != "tuple":
        return None
    items = entry.get("items", [])
    if len(items) < 2:
        return None
    name, candidate = items[0], items[1]
    if not isinstance(name, str) or not isinstance(candidate, dict):
        return None
    if candidate.get("kind") != "estimator":
        return None
    return name, candidate["state"]


def _components(record):
    """List the named sub-estimators of a composite step.

    ``Pipeline``, ``ColumnTransformer`` and ``FeatureUnion`` keep their
    children in a list of ``(name, estimator)`` entries.  Naming them in the
    document lets :func:`diff_states` report *which* child was replaced or
    re-parameterised instead of only flagging the enclosing step.
    """
    cls = record["class"]
    candidates = []
    if cls.endswith("Pipeline"):
        candidates.append(("params", "steps"))
    elif cls.endswith("ColumnTransformer"):
        candidates.append(("fitted", "transformers_"))
    elif cls.endswith("FeatureUnion"):
        candidates.append(("params", "transformer_list"))
    components = []
    for source, key in candidates:
        entries = record[source].get(key)
        if not isinstance(entries, dict) or entries.get("kind") != "list":
            continue
        for position, entry in enumerate(entries.get("items", [])):
            labelled = _estimator_entry_label(entry)
            if labelled is None:
                continue
            name, state = labelled
            child = {
                "name": f"{key}[{position}].{name}",
                "class": state["class"],
                "fingerprint": _fingerprint(
                    {
                        "class": state["class"],
                        "params": state["params"],
                        "fitted": state["fitted"],
                    }
                ),
            }
            child_record = {
                "class": state["class"],
                "params": state["params"],
                "fitted": state["fitted"],
            }
            nested = _components(child_record)
            if nested:
                child["components"] = nested
            components.append(child)
    return components


def _encode_estimator(est):
    """Encode an estimator: class, constructor parameters and fitted state."""
    from sklearn.base import BaseEstimator

    params = est.get_params(deep=False)
    fitted = {}
    missing = []
    if isinstance(est, BaseEstimator):
        for name, value in vars(est).items():
            if name in params:
                continue
            if name in _TRAINING_DATA_ATTRIBUTES:
                missing.append(name)
                continue
            fitted[name] = _encode_value(value)
    return {
        "class": _ref_of(type(est).__module__, type(est).__qualname__),
        "params": {name: _encode_value(value) for name, value in params.items()},
        "fitted": fitted,
        "non_exclusive": missing,
        "randomness": _describe_randomness(est, params),
    }


def _describe_randomness(est, params):
    """Describe the seed inputs and derived seeds of a fitted estimator."""
    seeded = {
        name: _encode_value(value)
        for name, value in params.items()
        if name in _SEED_PARAMETERS
    }
    derived = {}
    for name in ("_random_seed", "random_state_", "_seed"):
        value = getattr(est, name, None)
        if value is not None and not hasattr(value, "fit"):
            try:
                derived[name] = _encode_value(value)
            except TypeError:  # pragma: no cover - defensive
                continue
    generator = getattr(est, "_feature_subsample_rng", None)
    if isinstance(generator, np.random.Generator):
        state = json.dumps(generator.bit_generator.state, sort_keys=True)
        derived["_feature_subsample_rng.state_sha256"] = hashlib.sha256(
            state.encode("utf-8")
        ).hexdigest()
    return {
        "seeded": bool(seeded),
        "seed_params": seeded,
        "derived": derived,
        "propagation": (
            "the seed parameters above are passed unchanged to fit(); derived "
            "values are the seeds the estimator computed from them"
        ),
    }


def _validate_recorded_params(est, class_ref):
    """Raise a migration error if a recorded parameter value is now invalid.

    Constructor signatures catch renamed parameters, but a parameter that
    still exists while no longer accepting an old value (a removed option, a
    narrowed interval, ...) only shows up when ``fit`` is called.  Checking
    the recorded values against the current constraints turns that into an
    actionable error at load time.
    """
    from sklearn.utils._param_validation import InvalidParameterError

    try:
        est._validate_params()
    except InvalidParameterError as exc:
        raise ChainStateMigrationError(
            f"{class_ref} no longer accepts the value recorded in the state: "
            f"{str(exc).strip()}",
            fragment=class_ref,
            suggestion=(
                "The value is no longer a valid parameter in this version of "
                "scikit-learn. Pick the closest supported value and re-export "
                "the state, or pin the version that exported it."
            ),
        ) from exc
    except Exception:  # pragma: no cover - validation is best effort
        return


def _decode_estimator(spec):
    cls = _resolve_ref(spec["class"])
    params = {name: _decode_value(value) for name, value in spec["params"].items()}
    try:
        est = cls(**params)
    except TypeError as exc:
        raise ChainStateMigrationError(
            f"the current version cannot build {spec['class']} with the "
            f"parameters recorded in the state: {exc}",
            fragment=spec["class"],
            suggestion=(
                "The constructor signature changed between the version that "
                "exported the state and the version reading it. Re-export the "
                "state with the current version, or pin the original version."
            ),
        ) from exc
    _validate_recorded_params(est, spec["class"])
    for name, value in spec["fitted"].items():
        try:
            setattr(est, name, _decode_value(value))
        except (AttributeError, TypeError) as exc:
            raise ChainStateMigrationError(
                f"cannot restore the fitted attribute {name!r} of "
                f"{spec['class']}: {exc}",
                fragment=spec["class"],
                suggestion=(
                    f"{name!r} does not exist (or cannot be set) in the current "
                    "version of scikit-learn. Check the changelog for this "
                    "estimator's fitted attributes, or re-export the state."
                ),
            ) from exc
    if spec.get("non_exclusive"):
        est._chain_state_missing_attributes = list(spec["non_exclusive"])
    return est


# ---------------------------------------------------------------------------
# Document construction
# ---------------------------------------------------------------------------


def _scalar_or_none(value):
    """Return a plain int for a fitted ``n_features_in_``-like attribute."""
    if isinstance(value, numbers.Integral) and not isinstance(value, bool):
        return int(value)
    return None


def _infer_output_features(est):
    if est is None or est == "passthrough":
        return None
    for name in ("n_components_", "n_components"):
        value = getattr(est, name, None)
        if isinstance(value, numbers.Integral) and not isinstance(value, bool):
            return int(value)
    components = getattr(est, "components_", None)
    if components is not None and hasattr(components, "shape"):
        return int(components.shape[0])
    indices = getattr(est, "output_indices_", None)
    if isinstance(indices, dict) and indices:
        return int(sum(slice_.stop - slice_.start for slice_ in indices.values()))
    return None


def _step_record(name, index, est):
    record = {
        "segment": "step",
        "name": name,
        "index": index,
        "order": index,
    }
    if est is None or est == "passthrough":
        record.update(
            {
                "class": None,
                "passthrough": True,
                "params": {},
                "fitted": {},
                "non_exclusive": [],
                "randomness": {"seeded": False, "seed_params": {}, "derived": {}},
                "n_features_in": None,
                "n_features_out": None,
            }
        )
        record["fingerprint"] = _fingerprint(record)
        return record
    encoded = _encode_estimator(est)
    record.update(
        {
            "class": encoded["class"],
            "passthrough": False,
            "params": encoded["params"],
            "fitted": encoded["fitted"],
            "non_exclusive": encoded["non_exclusive"],
            "randomness": encoded["randomness"],
            "n_features_in": _scalar_or_none(getattr(est, "n_features_in_", None)),
            "n_features_out": _infer_output_features(est),
        }
    )
    components = _components(record)
    if components:
        record["components"] = components
    record["fingerprint"] = _fingerprint(record)
    return record


def _fingerprint(record):
    payload = {
        "class": record["class"],
        "params": record["params"],
        "fitted": record["fitted"],
    }
    return _sha256(
        json.dumps(payload, sort_keys=True, default=repr, separators=(",", ":"))
    )


def export_header(pipeline, *, pipeline_class):
    """Build the header of the state document of a fitted pipeline."""
    import sklearn

    return {
        "segment": "header",
        "format": FORMAT_NAME,
        "format_version": list(FORMAT_VERSION),
        "sklearn_version": sklearn.__version__,
        "python_version": _python_version(),
        "pipeline_class": _ref_of(
            pipeline_class.__module__, pipeline_class.__qualname__
        ),
        "tolerance": {"atol": DEFAULT_ATOL, "rtol": DEFAULT_RTOL},
        "n_steps": len(pipeline.steps),
        "pipeline_params": {
            name: _encode_value(value)
            for name, value in pipeline.get_params(deep=False).items()
            if name != "steps"
        },
        "input_features": _input_features(pipeline),
        "output": _output_description(pipeline),
    }


def iter_records(pipeline, collected=None):
    """Yield the step records of a fitted pipeline, one at a time.

    Only the current step and the dimensionality of its predecessor are kept
    in memory, so a chain with hundreds of steps can be exported without
    materialising every record first.  When ``collected`` is a list, a
    lightweight summary of each step is appended to it as it is produced, so
    that a summary can be returned without keeping the full records.
    """
    previous = None
    previous_features = None
    for index, (name, est) in enumerate(pipeline.steps):
        record = _step_record(name, index, est)
        record["depends_on"] = [] if previous is None else [previous]
        if record["n_features_in"] is None and record["passthrough"]:
            record["n_features_in"] = previous_features
            record["n_features_out"] = previous_features
        if record["n_features_in"] is not None:
            previous_features = record["n_features_in"]
        previous = name
        if collected is not None:
            collected.append(_summarise_record(record))
        yield record


def export_state(pipeline, *, pipeline_class):
    """Build the state document of a fitted pipeline."""
    return export_header(pipeline, pipeline_class=pipeline_class), list(
        iter_records(pipeline)
    )


def _python_version():
    import platform

    return platform.python_version()


def _input_features(pipeline):
    for _, est in pipeline.steps:
        if est is not None and est != "passthrough":
            return getattr(est, "n_features_in_", None)
    return None


def _output_description(pipeline):
    final = pipeline._final_estimator
    if final is None or final == "passthrough":
        return None
    description = {"class": _ref_of(type(final).__module__, type(final).__qualname__)}
    classes = getattr(final, "classes_", None)
    if classes is not None:
        description["classes"] = _encode_value(classes)
    description["n_features_in"] = getattr(final, "n_features_in_", None)
    return description


# ---------------------------------------------------------------------------
# Serialisation with checksums, atomic and streaming
# ---------------------------------------------------------------------------


def _sha256(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _canonical(payload):
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), default=repr)


def _line(payload):
    return _canonical(payload) + "\n"


def _checked(payload):
    """Attach a checksum of the payload (without the checksum field itself)."""
    body = dict(payload)
    body.pop("sha256", None)
    payload = dict(body)
    payload["sha256"] = _sha256(_canonical(body))
    return payload


def dump_state(header, records, fileobj):
    """Write a complete document to a text file object.

    Each step is written as soon as it is produced, so the memory used by the
    export does not grow with the number of steps.  This helper writes to an
    already opened file object and performs no atomic rename; callers that
    write to a path should use :func:`save_state` instead.
    """
    fileobj.write(_line(_checked(dict(header))))
    body = hashlib.sha256()
    step_digests = {}
    n_steps = 0
    for record in records:
        payload = _checked(dict(record))
        text = _line(payload)
        body.update(text.encode("utf-8"))
        step_digests[str(payload.get("index"))] = _sha256(text.strip())
        fileobj.write(text)
        n_steps += 1
    footer = {
        "segment": "end",
        "n_steps": n_steps,
        # The per-step digests let a reader name the exact step that was
        # modified instead of only reporting a mismatch of the whole body.
        "step_digests": step_digests,
        "body_sha256": body.hexdigest(),
    }
    fileobj.write(_line(_checked(footer)))
    return footer


def save_state(header, records, path):
    """Write a document to ``path`` atomically.

    The document is first written to a temporary file in the same directory,
    then renamed over ``path``.  If writing fails, the temporary file is
    removed and ``path`` is left untouched.

    Steps are written as they are consumed, so ``records`` may be a generator
    and only one step needs to be held in memory at a time.
    """
    directory = os.path.dirname(os.path.abspath(os.fspath(path)))
    os.makedirs(directory, exist_ok=True)
    fd, temporary = tempfile.mkstemp(
        prefix=".chain-state-", suffix=".tmp", dir=directory
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fileobj:
            dump_state(header, records, fileobj)
            fileobj.flush()
            os.fsync(fileobj.fileno())
        os.replace(temporary, os.fspath(path))
    except BaseException:
        try:
            os.unlink(temporary)
        except OSError:
            pass
        raise


def _decode_line(text, *, position):
    try:
        payload = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ChainStateCorruptionError(
            f"invalid JSON at column {exc.colno}: {exc.msg}",
            fragment=f"line {position}",
            suggestion=(
                "The document is structurally broken. Restore it from the "
                "original export; if it was edited by hand, re-export instead."
            ),
        ) from exc
    if not isinstance(payload, dict):
        raise ChainStateCorruptionError(
            "the line does not hold a JSON object",
            fragment=f"line {position}",
            suggestion="Restore the document from the original export.",
        )
    stored = payload.pop("sha256", None)
    if stored is None:
        raise ChainStateCorruptionError(
            "the segment has no checksum",
            fragment=f"line {position}",
            suggestion=(
                "The document was produced by an incompatible writer or edited "
                "by hand. Re-export it."
            ),
        )
    if stored != _sha256(_canonical(payload)):
        raise ChainStateCorruptionError(
            "checksum mismatch: this segment was modified or corrupted",
            fragment=f"line {position}"
            + (
                f", step {payload.get('order')} ({payload.get('name')})"
                if payload.get("segment") == "step"
                else f", segment {payload.get('segment')!r}"
            ),
            suggestion="Restore the document from the original export.",
        )
    return payload


def iter_state(path):
    """Yield the segments of a state document, verifying each as it is read.

    The header is yielded first, then every step in execution order, then the
    footer.  Nothing is accumulated besides one line at a time, so documents
    with hundreds of steps can be read with bounded memory.
    """
    with open(os.fspath(path), encoding="utf-8") as fileobj:
        header_text = fileobj.readline()
        if not header_text.strip():
            raise ChainStateCorruptionError(
                "the document is empty",
                fragment="header",
                suggestion="Export the chain state again.",
            )
        header = _decode_line(header_text, position=1)
        if header.get("segment") != "header":
            raise ChainStateCorruptionError(
                "the first segment is not a header",
                fragment="line 1",
                suggestion="Restore the document from the original export.",
            )
        _check_version(header)
        yield header
        body = hashlib.sha256()
        step_digests = {}
        n_steps = 0
        last_name = None
        for position, text in enumerate(fileobj, start=2):
            if not text.strip():
                continue
            payload = _decode_line(text, position=position)
            if payload.get("segment") == "end":
                _check_footer(payload, header, body, n_steps, step_digests)
                yield payload
                remainder = fileobj.read().strip()
                if remainder:
                    raise ChainStateCorruptionError(
                        "unexpected content after the end segment",
                        fragment=f"line {position + 1}",
                        suggestion=(
                            "Trailing data means the file was appended to or "
                            "merged with another file. Re-export it."
                        ),
                    )
                return
            if payload.get("segment") != "step":
                raise ChainStateCorruptionError(
                    f"unknown segment {payload.get('segment')!r}",
                    fragment=f"line {position}",
                    suggestion="Restore the document from the original export.",
                )
            body.update(text.encode("utf-8"))
            step_digests[str(payload.get("index"))] = _sha256(text.strip())
            n_steps += 1
            last_name = payload.get("name")
            yield payload
        raise ChainStateCorruptionError(
            "the document ends without an end segment",
            fragment=(
                "after step " + repr(last_name) if last_name is not None else "header"
            ),
            suggestion=(
                "The file is truncated, most likely because the export was "
                "interrupted or the copy is incomplete. Export it again."
            ),
        )


def _check_version(header):
    if header.get("format") != FORMAT_NAME:
        raise ChainStateVersionError(
            f"unknown document format {header.get('format')!r}",
            fragment="header",
            suggestion=f"This version only reads {FORMAT_NAME!r} documents.",
        )
    version = header.get("format_version")
    try:
        version = tuple(int(part) for part in version)
    except (TypeError, ValueError):
        raise ChainStateVersionError(
            f"malformed format version {header.get('format_version')!r}",
            fragment="header",
            suggestion="Restore the document from the original export.",
        ) from None
    if version not in SUPPORTED_VERSIONS:
        supported = ", ".join(".".join(map(str, v)) for v in sorted(SUPPORTED_VERSIONS))
        raise ChainStateVersionError(
            f"the document uses format version {'.'.join(map(str, version))}, "
            f"which this version of scikit-learn cannot read (supported: "
            f"{supported})",
            fragment="header",
            suggestion=(
                "Install the scikit-learn version that exported the state, or "
                "re-export it with the current version."
            ),
        )


def _check_footer(footer, header, body, n_steps, step_digests=None):
    if footer.get("n_steps") != n_steps:
        raise ChainStateCorruptionError(
            f"the end segment announces {footer.get('n_steps')} steps but "
            f"{n_steps} were read",
            fragment=f"after step {n_steps - 1}",
            suggestion=(
                "Step lines are missing from the document. Restore it from the "
                "original export."
            ),
        )
    if footer.get("n_steps") != header.get("n_steps"):
        raise ChainStateCorruptionError(
            "the number of steps differs between the header and the end segment",
            fragment="header",
            suggestion="Restore the document from the original export.",
        )
    if footer.get("body_sha256") != body.hexdigest():
        expected = footer.get("step_digests") or {}
        faulty = None
        for index, digest in sorted(expected.items(), key=lambda item: item[0]):
            if step_digests is not None and step_digests.get(index) != digest:
                faulty = index
                break
        if faulty is not None:
            raise ChainStateCorruptionError(
                f"the checksum of step {faulty} does not match: its content was "
                "modified or corrupted",
                fragment=f"step {faulty}",
                suggestion="Restore the document from the original export.",
            )
        raise ChainStateCorruptionError(
            "the steps checksum does not match: step content was modified",
            fragment="steps",
            suggestion="Restore the document from the original export.",
        )


def read_state(path):
    """Read and verify a complete document, returning header and steps."""
    header = None
    records = []
    for payload in iter_state(path):
        if payload.get("segment") == "header":
            header = payload
        elif payload.get("segment") == "step":
            records.append(payload)
    return header, records


# ---------------------------------------------------------------------------
# Rebuilding a fitted pipeline
# ---------------------------------------------------------------------------


def build_pipeline(header, records, *, pipeline_class):
    """Instantiate a fitted pipeline from an exported document."""
    pipeline_params = {
        name: _decode_value(value)
        for name, value in header.get("pipeline_params", {}).items()
    }
    steps = []
    missing = []
    for record in records:
        name = record["name"]
        if record.get("passthrough"):
            steps.append((name, "passthrough"))
            continue
        est = _decode_estimator(
            {
                "class": record["class"],
                "params": record["params"],
                "fitted": record["fitted"],
                "non_exclusive": record.get("non_exclusive", []),
            }
        )
        if getattr(est, "_chain_state_missing_attributes", None):
            missing.append((name, est._chain_state_missing_attributes))
            del est._chain_state_missing_attributes
        steps.append((name, est))
    if missing:
        names = ", ".join(f"{name!r} ({attrs})" for name, attrs in missing)
        raise ChainStateDataError(
            "the state of some steps was not exported because it keeps a copy "
            f"of the training data: {names}",
            fragment="steps",
            suggestion=(
                "Those steps can only be restored by re-fitting them on the "
                "training data. Call replay_chain_state with the training data, "
                "or refit the pipeline."
            ),
        )
    pipeline = pipeline_class(steps=steps, **pipeline_params)
    return pipeline


# ---------------------------------------------------------------------------
# Structural diff
# ---------------------------------------------------------------------------


def _type_name(value):
    if isinstance(value, dict):
        return value.get("kind", "dict")
    if isinstance(value, list):
        return "list"
    return type(value).__name__


def _diff_values(old, new, path, entries, *, max_entries):
    if len(entries) >= max_entries:
        return
    if isinstance(old, dict) and isinstance(new, dict):
        if old.get("kind") == "array" and new.get("kind") == "array":
            if old.get("dtype") != new.get("dtype") or old.get("shape") != new.get(
                "shape"
            ):
                entries.append(
                    {
                        "path": path,
                        "kind": "array_structure_changed",
                        "old": {"dtype": old.get("dtype"), "shape": old.get("shape")},
                        "new": {"dtype": new.get("dtype"), "shape": new.get("shape")},
                    }
                )
            elif old.get("data") != new.get("data"):
                entries.append(
                    {
                        "path": path,
                        "kind": "array_values_changed",
                        "old_shape": old.get("shape"),
                        "new_shape": new.get("shape"),
                        **_array_difference(old, new),
                    }
                )
            return
        old_keys = list(old.keys())
        new_keys = list(new.keys())
        for key in old_keys:
            if key not in new:
                entries.append(
                    {
                        "path": f"{path}.{key}",
                        "kind": "removed",
                        "old": old[key],
                        "new": None,
                    }
                )
        for key in new_keys:
            if key not in old:
                entries.append(
                    {
                        "path": f"{path}.{key}",
                        "kind": "added",
                        "old": None,
                        "new": new[key],
                    }
                )
            else:
                _diff_values(
                    old[key],
                    new[key],
                    f"{path}.{key}",
                    entries,
                    max_entries=max_entries,
                )
        return
    if isinstance(old, list) and isinstance(new, list):
        if len(old) != len(new):
            entries.append(
                {
                    "path": path,
                    "kind": "length_changed",
                    "old": len(old),
                    "new": len(new),
                }
            )
        for index, (old_item, new_item) in enumerate(zip(old, new)):
            _diff_values(
                old_item, new_item, f"{path}[{index}]", entries, max_entries=max_entries
            )
        return
    if _type_name(old) != _type_name(new):
        entries.append(
            {
                "path": path,
                "kind": "type_changed",
                "old": repr(old)[:200],
                "new": repr(new)[:200],
            }
        )
        return
    if old != new:
        entries.append(
            {
                "path": path,
                "kind": "value_changed",
                "old": repr(old)[:200],
                "new": repr(new)[:200],
            }
        )


def _array_difference(old, new):
    old_array = _decode_array(old)
    new_array = _decode_array(new)
    if old_array.size == 0:
        return {"max_abs_diff": 0.0, "n_different": 0}
    if old_array.shape != new_array.shape:
        return {"max_abs_diff": None, "n_different": None}
    try:
        difference = np.abs(old_array.astype(np.float64) - new_array.astype(np.float64))
        finite = difference[np.isfinite(difference)]
        return {
            "max_abs_diff": float(finite.max()) if finite.size else 0.0,
            "n_different": int(np.count_nonzero(difference)),
        }
    except (TypeError, ValueError):
        return {"max_abs_diff": None, "n_different": None}


def diff_states(old_header, old_records, new_header, new_records, *, max_entries=200):
    """Compare two exported documents, step by step."""
    entries = []
    header_entries = []
    for key in (
        "format_version",
        "sklearn_version",
        "pipeline_class",
        "pipeline_params",
    ):
        if old_header.get(key) != new_header.get(key):
            header_entries.append(
                {
                    "path": f"header.{key}",
                    "kind": "value_changed",
                    "old": repr(old_header.get(key))[:200],
                    "new": repr(new_header.get(key))[:200],
                }
            )
    old_by_order = {record["order"]: record for record in old_records}
    new_by_order = {record["order"]: record for record in new_records}
    steps = []
    changed = []
    for order in sorted(set(old_by_order) | set(new_by_order), key=repr):
        old_record = old_by_order.get(order)
        new_record = new_by_order.get(order)
        name = (new_record or old_record)["name"]
        step_entries = []
        if old_record is None:
            step_entries.append(
                {
                    "path": "steps",
                    "kind": "added",
                    "old": None,
                    "new": new_record["class"],
                }
            )
            status = "added"
        elif new_record is None:
            step_entries.append(
                {
                    "path": "steps",
                    "kind": "removed",
                    "old": old_record["class"],
                    "new": None,
                }
            )
            status = "removed"
        else:
            for key in ("class", "params", "fitted", "randomness"):
                _diff_values(
                    old_record.get(key),
                    new_record.get(key),
                    key,
                    step_entries,
                    max_entries=max_entries,
                )
            status = "identical" if not step_entries else "changed"
        component_changes = _diff_components(
            old_record, new_record, step_entries, max_entries=max_entries
        )
        if status != "identical":
            changed.append(name)
        steps.append(
            {
                "name": name,
                "index": (new_record or old_record)["index"],
                "status": status,
                "fields": step_entries,
                "changed_components": component_changes,
                "old_fingerprint": None
                if old_record is None
                else old_record.get("fingerprint"),
                "new_fingerprint": None
                if new_record is None
                else new_record.get("fingerprint"),
            }
        )
        entries.extend(step_entries)
    first_component = None
    if changed:
        first_step = next(step for step in steps if step["name"] == changed[0])
        if first_step["changed_components"]:
            first_component = f"{changed[0]}.{first_step['changed_components'][0]}"
        else:
            first_component = changed[0]
    return {
        "identical": not entries and not header_entries,
        "header": header_entries,
        "changed_steps": changed,
        "first_changed_step": changed[0] if changed else None,
        "first_changed_component": first_component,
        "steps": steps,
        "entries": entries[:max_entries],
    }


def _flatten_components(components):
    for component in components:
        yield component
        yield from _flatten_components(component.get("components", []))


def _diff_components(old_record, new_record, entries, *, max_entries):
    """Compare the sub-estimators of composite steps and name the changed ones."""
    if not (old_record or {}).get("components") and not (new_record or {}).get(
        "components"
    ):
        return []
    old_components = {
        component["name"]: component
        for component in _flatten_components((old_record or {}).get("components", []))
    }
    new_components = {
        component["name"]: component
        for component in _flatten_components((new_record or {}).get("components", []))
    }
    # A document that predates component recording cannot localise a change;
    # compare only the sub-estimators it actually described.
    common = set(old_components) & set(new_components)
    added = set(new_components) - set(old_components) if old_components else set()
    removed = set(old_components) - set(new_components) if new_components else set()
    changed = []
    for name in sorted(common | added | removed, key=repr):
        old_component = old_components.get(name)
        new_component = new_components.get(name)
        if old_component is None:
            changed.append(name)
            entries.append(
                {
                    "path": f"components[{name}]",
                    "kind": "added",
                    "old": None,
                    "new": new_component["class"],
                }
            )
            continue
        if new_component is None:
            changed.append(name)
            entries.append(
                {
                    "path": f"components[{name}]",
                    "kind": "removed",
                    "old": old_component["class"],
                    "new": None,
                }
            )
            continue
        if old_component["fingerprint"] != new_component["fingerprint"]:
            changed.append(name)
            if old_component["class"] != new_component["class"]:
                entries.append(
                    {
                        "path": f"components[{name}]",
                        "kind": "class_changed",
                        "old": old_component["class"],
                        "new": new_component["class"],
                    }
                )
            else:
                entries.append(
                    {
                        "path": f"components[{name}]",
                        "kind": "parameters_or_state_changed",
                        "old": old_component["class"],
                        "new": new_component["class"],
                    }
                )
    return changed


# ---------------------------------------------------------------------------
# Replay
# ---------------------------------------------------------------------------


def compare_arrays(reference, replayed, *, rtol, atol):
    """Compare two arrays, returning a JSON-able report.

    Sparse matrices and pandas objects are compared through their dense
    values, so that a replayed ``ColumnTransformer`` can be checked column by
    column regardless of the container each step returns.
    """
    reference = _as_dense(reference)
    replayed = _as_dense(replayed)
    if reference.shape != replayed.shape:
        return {
            "equal": False,
            "exact": False,
            "max_abs_diff": None,
            "reason": f"shape changed from {reference.shape} to {replayed.shape}",
        }
    if reference.dtype.kind not in "biufc" or replayed.dtype.kind not in "biufc":
        try:
            equal = bool(np.array_equal(reference, replayed))
        except TypeError:
            equal = bool(all(a == b for a, b in zip(reference, replayed)))
        return {
            "equal": equal,
            "exact": equal,
            "max_abs_diff": 0.0 if equal else None,
            "reason": None if equal else "non-numeric values differ",
        }
    reference = (
        reference.astype(np.float64) if reference.dtype.kind != "c" else reference
    )
    replayed = replayed.astype(np.float64) if replayed.dtype.kind != "c" else replayed
    exact = bool(np.array_equal(reference, replayed, equal_nan=True))
    close = bool(np.allclose(reference, replayed, rtol=rtol, atol=atol, equal_nan=True))
    finite = np.isfinite(reference) & np.isfinite(replayed)
    if finite.any():
        difference = np.abs(reference[finite] - replayed[finite])
        max_abs_diff = float(difference.max())
    else:
        max_abs_diff = 0.0
    return {
        "equal": close,
        "exact": exact,
        "max_abs_diff": max_abs_diff,
        "rtol": rtol,
        "atol": atol,
        "reason": None if close else "values differ beyond the given tolerance",
    }


def _as_dense(value):
    """Return the value as a 2D-or-less numpy array of its actual entries."""
    from scipy import sparse

    if sparse.issparse(value):
        return np.asarray(value.toarray())
    if hasattr(value, "to_numpy") and not isinstance(value, np.ndarray):
        try:
            return np.asarray(value.to_numpy())
        except Exception:  # pragma: no cover - defensive
            return np.asarray(value)
    return np.asarray(value)


def replay_steps(pipeline, replayed, X, *, rtol, atol):
    """Compare the intermediate outputs of two fitted pipelines on ``X``."""
    report = {
        "equal": True,
        "steps": [],
        "first_divergent_step": None,
        "rtol": rtol,
        "atol": atol,
    }
    current = X
    replayed_current = X
    reference_steps = list(pipeline._iter(with_final=False))
    replay_steps = list(replayed._iter(with_final=False))
    if len(reference_steps) != len(replay_steps):
        report["equal"] = False
        report["first_divergent_step"] = "<structure>"
        report["steps"].append(
            {
                "name": "<structure>",
                "equal": False,
                "reason": (
                    f"the chain has {len(reference_steps)} transformer steps "
                    f"instead of {len(replay_steps)}"
                ),
            }
        )
        return report
    for (_, name, transformer), (_, replay_name, replayed_transformer) in zip(
        reference_steps, replay_steps
    ):
        try:
            current = transformer.transform(current)
            replayed_current = replayed_transformer.transform(replayed_current)
        except Exception as exc:  # pragma: no cover - defensive
            report["equal"] = False
            report["first_divergent_step"] = name
            report["steps"].append(
                {"name": name, "equal": False, "reason": f"replay raised {exc!r}"}
            )
            return report
        comparison = compare_arrays(current, replayed_current, rtol=rtol, atol=atol)
        comparison["name"] = name
        if not comparison["equal"] and report["first_divergent_step"] is None:
            report["first_divergent_step"] = name
            report["equal"] = False
        report["steps"].append(comparison)
    final = getattr(pipeline, "_final_estimator", None)
    replayed_final = getattr(replayed, "_final_estimator", None)
    if final is not None and final != "passthrough":
        for method in ("predict", "decision_function", "predict_proba", "transform"):
            if hasattr(final, method) and hasattr(replayed_final, method):
                try:
                    reference_output = getattr(final, method)(current)
                    replayed_output = getattr(replayed_final, method)(replayed_current)
                except Exception as exc:  # pragma: no cover - defensive
                    comparison = {
                        "name": "<final>",
                        "equal": False,
                        "exact": False,
                        "max_abs_diff": None,
                        "reason": f"{method} raised {exc!r}",
                    }
                else:
                    comparison = compare_arrays(
                        reference_output, replayed_output, rtol=rtol, atol=atol
                    )
                    comparison["name"] = "<final>"
                    comparison["method"] = method
                if not comparison["equal"] and report["first_divergent_step"] is None:
                    report["first_divergent_step"] = "<final>"
                    report["equal"] = False
                report["steps"].append(comparison)
                break
    return report


def _summarise_record(record):
    """The per-step entry of a summary, without the bulky array payloads."""
    return {
        "name": record["name"],
        "index": record["index"],
        "class": record["class"],
        "n_features_in": record.get("n_features_in"),
        "n_features_out": record.get("n_features_out"),
        "depends_on": record.get("depends_on", []),
        "randomness": {
            "seeded": record["randomness"]["seeded"],
            "seed_params": sorted(record["randomness"]["seed_params"]),
        },
        "non_exclusive": record.get("non_exclusive", []),
        "fingerprint": record["fingerprint"],
    }


def summary(header, steps=None):
    """JSON-able summary of an exported document.

    ``steps`` may be the full step records or the lightweight per-step
    summaries built by :func:`iter_records`.
    """
    if steps is None:
        steps = []
    if steps and "fingerprint" not in steps[0]:
        steps = [_summarise_record(record) for record in steps]
    return {
        "format": header["format"],
        "format_version": ".".join(map(str, header["format_version"])),
        "sklearn_version": header["sklearn_version"],
        "n_steps": header["n_steps"],
        "input_features": header.get("input_features"),
        "steps": [
            {
                key: step[key]
                for key in (
                    "name",
                    "index",
                    "class",
                    "n_features_in",
                    "n_features_out",
                    "depends_on",
                    "randomness",
                    "non_exclusive",
                    "fingerprint",
                )
            }
            for step in steps
        ],
    }


def ensure_no_training_data(document_text, X):
    """Check that the training matrix does not appear in an exported document.

    Used by the tests and by users who want to audit an export: every array
    payload is decoded and compared with ``X``.
    """
    X = np.asarray(X)
    marker = base64.b64encode(np.ascontiguousarray(X).tobytes()).decode("ascii")
    if marker in document_text:
        return False
    return True


__all__ = [
    "DEFAULT_ATOL",
    "DEFAULT_RTOL",
    "FORMAT_NAME",
    "FORMAT_VERSION",
    "SUPPORTED_VERSIONS",
    "build_pipeline",
    "compare_arrays",
    "diff_states",
    "dump_state",
    "export_header",
    "export_state",
    "iter_records",
    "iter_state",
    "read_state",
    "replay_steps",
    "save_state",
    "summary",
]
