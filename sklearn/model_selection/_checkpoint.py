"""On-disk, resumable state for staged hyper-parameter searches.

This module is used by :class:`~sklearn.model_selection.GridSearchCV` and
:class:`~sklearn.model_selection.RandomizedSearchCV` when their ``checkpoint``
parameter is set. It stores the outcome of every evaluated candidate on disk
in append-only *segments*, one segment per search stage, together with a JSON
manifest describing everything that is needed to reproduce the search
(random seed, cross-validation splits, scoring, data hash, ...).

Design notes
------------
* All writes are atomic: a segment or the manifest is written to a temporary
  file, ``fsync``-ed and then moved into place with ``os.replace``. A killed
  process therefore never leaves a half-written file behind; at worst an
  orphaned ``*.tmp`` file remains, which is ignored on resume.
* Evaluation results are never fully materialised in memory. Segments are
  frame-pickle files that are read back lazily, one segment at a time, when
  ``cv_results_`` is assembled.
* Candidates are identified by a normalised key: parameter names are sorted,
  floating point values use their exact ``repr`` and the resulting canonical
  representation is hashed with SHA-256. The same parameter setting therefore
  always maps to the same identity, even across runs in which the candidate
  space was enlarged.
"""

# Authors: The scikit-learn developers
# SPDX-License-Identifier: BSD-3-Clause

import hashlib
import json
import os
import pickle
import struct
import tempfile
import warnings
from datetime import datetime, timezone

import numpy as np
from joblib import hash as joblib_hash

from sklearn.exceptions import (
    CheckpointCorruptedError,
    CheckpointCorruptedWarning,
    CheckpointMismatchError,
)

_MANIFEST_NAME = "manifest.json"
_SEGMENT_PREFIX = "segment_"
_SEGMENT_SUFFIX = ".pkl"
_TMP_SUFFIX = ".tmp"
_QUARANTINE_DIR = "quarantine"

# Bump when the on-disk format changes in an incompatible way.
_STATE_VERSION = 1

# Each record in a segment is stored as an 8-byte little-endian length header
# followed by an independent pickle payload. Framing records individually
# makes prefix salvage reliable: a truncated trailing frame is the only thing
# lost, every complete frame before it stays readable.
_HEADER = struct.Struct("<Q")


def _canonical_param_representation(params):
    """Build a deterministic, hashable representation of a parameter dict.

    Parameter names are sorted and floating point values are rendered with
    their shortest round-trip ``repr`` so that ``1.0`` and ``1`` keep their
    natural type distinction while the same float always renders identically.
    """

    def _render(value):
        if isinstance(value, bool):
            return ("bool", value)
        if isinstance(value, float):
            # ``repr`` gives the shortest string that round-trips the float.
            return ("float", repr(value))
        if isinstance(value, int):
            return ("int", value)
        if isinstance(value, str):
            return ("str", value)
        if value is None:
            return ("none", None)
        if isinstance(value, dict):
            return (
                "dict",
                tuple((k, _render(v)) for k, v in sorted(value.items())),
            )
        if isinstance(value, (list, tuple)):
            kind = "list" if isinstance(value, list) else "tuple"
            return (kind, tuple(_render(v) for v in value))
        if hasattr(value, "get_params"):
            # Estimators and similar objects: fall back to their scikit-learn
            # parameters, rendered recursively.
            return (
                "estimator",
                type(value).__name__,
                _canonical_param_representation(value.get_params(deep=False)),
            )
        # Last resort: the joblib hash captures the exact object content.
        return ("object", type(value).__name__, joblib_hash(value))

    return tuple(sorted((name, _render(value)) for name, value in params.items()))


def candidate_key(params):
    """Return the stable identity string of a candidate parameter dict."""
    canonical = repr(_canonical_param_representation(params))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _stable_hash(obj):
    """Hash arbitrary in-memory objects (including numpy arrays) via joblib."""
    return joblib_hash(obj, coerce_mmap=True)


def _utc_now():
    return datetime.now(timezone.utc).isoformat()


def _pack_records(records):
    chunks = []
    for record in records:
        payload = pickle.dumps(record, protocol=5)
        chunks.append(_HEADER.pack(len(payload)))
        chunks.append(payload)
    return b"".join(chunks)


def _unpack_records(raw):
    """Parse a segment byte string.

    Returns
    -------
    records : list of dict
        Every complete, well-formed record found, in order.
    n_dropped : int
        Number of trailing frames that were truncated or malformed.
    error : str or None
        Description of the corruption, when any frame could not be recovered.
    """
    offset = 0
    records = []
    n_dropped = 0
    error = None
    total = len(raw)
    while offset < total:
        if total - offset < _HEADER.size:
            n_dropped += 1
            error = "truncated frame header"
            break
        (frame_len,) = _HEADER.unpack_from(raw, offset)
        start = offset + _HEADER.size
        end = start + frame_len
        if end > total:
            n_dropped += 1
            error = (
                f"truncated frame: expected {frame_len} bytes, "
                f"only {total - start} available"
            )
            break
        try:
            record = pickle.loads(raw[start:end])
        except (pickle.UnpicklingError, EOFError, AttributeError, ValueError) as exc:
            n_dropped += 1
            error = f"unpickling failed: {exc}"
            break
        if not isinstance(record, dict) or "key" not in record or "out" not in record:
            n_dropped += 1
            error = "malformed record"
            break
        records.append(record)
        offset = end
    return records, n_dropped, error


class SearchCheckpoint:
    """Persistent, resumable state for a staged parameter search.

    Parameters
    ----------
    path : str or os.PathLike
        Directory in which the state is stored. It is created if it does not
        exist. A directory whose manifest is incompatible with the current
        search configuration causes an error on :meth:`initialize`.

    Attributes
    ----------
    path : str
        Resolved checkpoint directory.
    manifest : dict or None
        In-memory copy of the manifest, available after :meth:`initialize`.
    salvage_report : list of dict or None
        Populated when a corrupted segment was detected and its intact prefix
        was salvaged. Each entry describes the stage, the number of records
        kept and where the damaged original was quarantined.
    """

    def __init__(self, path):
        self.path = os.fspath(path)
        self.manifest = None
        self.salvage_report = None

    # ------------------------------------------------------------------
    # Path helpers
    # ------------------------------------------------------------------
    @property
    def manifest_path(self):
        return os.path.join(self.path, _MANIFEST_NAME)

    def _segment_path(self, stage_idx):
        return os.path.join(
            self.path, f"{_SEGMENT_PREFIX}{stage_idx:06d}{_SEGMENT_SUFFIX}"
        )

    def _atomic_write_bytes(self, target_path, payload):
        """Write ``payload`` to ``target_path`` atomically."""
        directory = os.path.dirname(target_path) or "."
        fd, tmp_path = tempfile.mkstemp(suffix=_TMP_SUFFIX, dir=directory)
        try:
            with os.fdopen(fd, "wb") as handle:
                handle.write(payload)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(tmp_path, target_path)
        except BaseException:
            # Clean up the temp file on failure; the target is left untouched.
            try:
                os.unlink(tmp_path)
            except FileNotFoundError:
                pass
            raise

    def _flush_manifest(self):
        self.manifest["updated_at"] = _utc_now()
        self._atomic_write_bytes(
            self.manifest_path,
            json.dumps(self.manifest, indent=2, sort_keys=True).encode("utf-8"),
        )

    # ------------------------------------------------------------------
    # Initialisation / compatibility checks
    # ------------------------------------------------------------------
    def initialize(self, *, expected, cv_splits):
        """Initialise or validate the checkpoint for a new ``fit`` call.

        Parameters
        ----------
        expected : dict
            Manifest entries describing the current search configuration. It
            must contain ``data_hash``, ``config_hash`` and ``state_version``.

        cv_splits : list of (train, test) tuples
            Cross-validation splits used by the search. They are stored in the
            manifest on first creation and checked for equality on resume so
            that the splits cannot silently change between runs.

        Returns
        -------
        resuming : bool
            ``True`` when a compatible checkpoint already exists and the
            search resumes from it, ``False`` for a fresh search.
        """
        os.makedirs(self.path, exist_ok=True)
        self.salvage_report = None

        if os.path.exists(self.manifest_path):
            try:
                with open(self.manifest_path, encoding="utf-8") as handle:
                    self.manifest = json.load(handle)
            except (json.JSONDecodeError, OSError) as exc:
                raise CheckpointCorruptedError(
                    f"The checkpoint manifest at {self.manifest_path!r} could not "
                    f"be read ({exc}). The search cannot resume safely; move the "
                    "directory aside and restart to begin a fresh search."
                ) from exc

            mismatches = self._manifest_mismatches(expected, cv_splits)
            if mismatches:
                raise CheckpointMismatchError(
                    "Refusing to resume: the checkpoint in "
                    f"{self.path!r} is not compatible with the current search. "
                    "Likely causes: the training data, the random seed, the CV "
                    "splitter or the scoring configuration changed. Details:\n- "
                    + "\n- ".join(mismatches)
                )
            self._salvage_segments()
            return True

        # Fresh checkpoint: write the manifest before any evaluation happens.
        # The splits themselves are not stored verbatim: a content hash plus
        # their count is enough to detect any change while keeping the
        # manifest independent of the dataset size.
        self.manifest = {
            **expected,
            "n_splits": len(cv_splits),
            "cv_splits_hash": _stable_hash(cv_splits),
            "stages": [],
            "segments": [],
            "created_at": _utc_now(),
            "updated_at": _utc_now(),
        }
        self._flush_manifest()
        return False

    def _manifest_mismatches(self, expected, cv_splits):
        # Fields that must match exactly: any change here would make old and
        # new evaluations incomparable. The candidate space is handled
        # separately because enlarging it must remain possible.
        exact_fields = (
            "state_version",
            "data_hash",
            "config_hash",
            "random_state_hash",
            "refit_metric",
        )
        mismatches = []
        for key in exact_fields:
            if self.manifest.get(key) != expected[key]:
                mismatches.append(
                    f"{key} differs: stored={self.manifest.get(key)!r} "
                    f"current={expected[key]!r}"
                )
        stored_hash = self.manifest.get("cv_splits_hash")
        if (
            stored_hash != _stable_hash(cv_splits)
            or self.manifest.get("n_splits") != len(cv_splits)
        ):
            mismatches.append("cross-validation splits differ")
        return mismatches

    # ------------------------------------------------------------------
    # Segment handling
    # ------------------------------------------------------------------
    def _segment_stage_index(self, file_name):
        if not (
            file_name.startswith(_SEGMENT_PREFIX)
            and file_name.endswith(_SEGMENT_SUFFIX)
        ):
            return None
        middle = file_name[len(_SEGMENT_PREFIX) : -len(_SEGMENT_SUFFIX)]
        return int(middle) if middle.isdigit() else None

    def _list_segment_files(self):
        if not os.path.isdir(self.path):
            return []
        entries = []
        for name in os.listdir(self.path):
            stage_idx = self._segment_stage_index(name)
            if stage_idx is not None:
                entries.append((stage_idx, os.path.join(self.path, name)))
        entries.sort()
        return entries

    def _salvage_segments(self):
        """Validate every segment, salvaging the intact prefix of a bad one.

        A truncated or corrupted segment is rewritten without its broken
        trailing frames and the original file is moved to ``quarantine/``.
        Records preceding the corruption stay usable; :attr:`salvage_report`
        and a warning describe exactly which evaluations remain available.
        """
        report = []
        valid_segments = []
        quarantine_dir = os.path.join(self.path, _QUARANTINE_DIR)

        for stage_idx, path in self._list_segment_files():
            with open(path, "rb") as handle:
                raw = handle.read()
            records, n_dropped, error = _unpack_records(raw)

            if error is None and n_dropped == 0:
                valid_segments.append(stage_idx)
                continue

            os.makedirs(quarantine_dir, exist_ok=True)
            quarantine_path = os.path.join(
                quarantine_dir, os.path.basename(path)
            )
            try:
                os.replace(path, quarantine_path)
            except OSError:
                quarantine_path = "<could not be moved>"

            item = {
                "stage": stage_idx,
                "file": os.path.basename(path),
                "kept_records": len(records),
                "dropped_records": n_dropped,
                "error": error,
                "quarantine": quarantine_path,
            }
            if records:
                # Rewrite the intact prefix atomically; those evaluations are
                # still usable and will not be re-evaluated... unless a record
                # is itself missing, in which case it is simply re-evaluated.
                self._atomic_write_bytes(path, _pack_records(records))
                valid_segments.append(stage_idx)
            report.append(item)

        if report:
            self.salvage_report = report
            warnings.warn(
                "Corrupted checkpoint segment(s) detected and salvaged:\n"
                + "\n".join(
                    f"- stage {item['stage']} ({item['file']}): kept "
                    f"{item['kept_records']} record(s), dropped "
                    f"{item['dropped_records']} ({item['error']}); original "
                    f"moved to {item['quarantine']}"
                    for item in report
                ),
                CheckpointCorruptedWarning,
                stacklevel=3,
            )
            self.manifest["segments"] = [
                idx
                for idx in self.manifest.get("segments", [])
                if idx in valid_segments
            ]
            self.manifest["stages"] = [
                stage
                for stage in self.manifest.get("stages", [])
                if stage["stage"] in valid_segments
            ]
            self._flush_manifest()

    def append_stage(self, *, stage_idx, records, summary):
        """Persist one stage's newly evaluated records atomically.

        A stage file may already exist after a salvage operation kept the
        stage's intact prefix. In that case the new records are merged with
        the existing ones instead of overwriting them.
        """
        if not records:
            return
        target = self._segment_path(stage_idx)
        existing = []
        if os.path.exists(target):
            with open(target, "rb") as handle:
                existing, _n_dropped, _error = _unpack_records(handle.read())
        merged = existing + records
        self._atomic_write_bytes(target, _pack_records(merged))

        stages = self.manifest.setdefault("stages", [])
        stage_entry = next(
            (stage for stage in stages if stage["stage"] == stage_idx), None
        )
        if stage_entry is None:
            self.manifest.setdefault("segments", []).append(stage_idx)
            stages.append({"stage": stage_idx, "written_at": _utc_now(), **summary})
        else:
            # Merge with the salvaged prefix's accounting.
            stage_entry["n_candidates"] = (
                stage_entry.get("n_candidates", 0) + summary["n_candidates"]
            )
            stage_entry["n_candidates_skipped"] = summary.get(
                "n_candidates_skipped", 0
            )
            stage_entry["best_so_far_score"] = summary["best_so_far_score"]
            stage_entry["written_at"] = _utc_now()
        self._flush_manifest()

    def iter_records(self):
        """Yield stored records in stage order without loading them all.

        Segments are opened and closed one at a time so peak memory stays
        bounded by the largest single stage.
        """
        for _stage_idx, path in self._list_segment_files():
            with open(path, "rb") as handle:
                raw = handle.read()
            records, _n_dropped, error = _unpack_records(raw)
            if error is not None:
                # initialize() salvages segments before any iteration, so a
                # corruption here means something damaged the file mid-run.
                raise CheckpointCorruptedError(
                    f"Segment {path!r} is unreadable: {error}. Re-run the fit "
                    "to trigger salvage and resume."
                )
            yield from records

    def evaluated_keys(self):
        """Return the set of candidate identities already evaluated."""
        return {record["key"] for record in self.iter_records()}

    def n_evaluated(self):
        return sum(
            stage.get("n_candidates", 0)
            for stage in self.manifest.get("stages", [])
        )

    def stage_summaries(self):
        """Return the list of stage summaries recorded so far."""
        return list(self.manifest.get("stages", []))


# ----------------------------------------------------------------------
# Fingerprinting of the search configuration
# ----------------------------------------------------------------------
def make_search_fingerprint(
    *,
    estimator,
    scoring,
    cv,
    random_state,
    error_score,
    return_train_score,
    candidate_space,
    X,
    y,
    fit_params,
):
    """Compute the hashes and reproducibility fields stored in the manifest.

    The data hash covers the training data and every fit-time metadata array
    (``sample_weight``, ``groups`` and similar). The configuration hash covers
    everything that changes how a candidate is evaluated: the estimator, the
    scorer, the CV splitter, error handling and the train-score flag. The
    candidate space is hashed separately so that enlarging it on resume does
    not invalidate previously stored evaluations, while a change to any
    evaluation-affecting component is still detected.
    """
    data_hash = _stable_hash({"X": X, "y": y, "fit_params": fit_params})
    config_hash = _stable_hash(
        {
            "estimator": estimator,
            "scoring": scoring,
            "cv": cv,
            "error_score": error_score,
            "return_train_score": return_train_score,
        }
    )
    candidate_space_hash = _stable_hash(candidate_space)
    random_state_fingerprint = _random_state_fingerprint(random_state)
    return {
        "data_hash": data_hash,
        "config_hash": config_hash,
        "candidate_space_hash": candidate_space_hash,
        # Hex strings are JSON-safe regardless of the random_state type, so
        # the exact-compatibility check can rely on the hash alone.
        "random_state_hash": joblib_hash(random_state_fingerprint),
        # Human-readable form, best effort; informational only.
        "random_state": _json_safe(random_state_fingerprint),
    }


def _json_safe(obj):
    """Best-effort conversion of a fingerprint into JSON-serialisable data."""
    try:
        json.dumps(obj)
        return obj
    except (TypeError, ValueError):
        return repr(obj)


def _random_state_fingerprint(random_state):
    """Render a random_state argument into a hashable description."""
    if random_state is None or isinstance(random_state, (int, float, bool, str)):
        return ("value", random_state)
    if isinstance(random_state, np.random.RandomState):
        return ("RandomState", random_state.get_state())
    if hasattr(random_state, "bit_generator"):
        # numpy Generator: capture the underlying bit generator state.
        return ("Generator", random_state.bit_generator.state)
    return ("object", joblib_hash(random_state))
