"""Acceptance harness for the verifiable model-artifact contract feature.

Builds the five artifact groups required by the task brief:

  1. exact-match environment
  2. patch-level dependency difference
  3. same feature names in a different order
  4. missing column
  5. categorical vocabulary drift

For every group it asserts the load verdict, and for the exact-match group it
prints a bit-level prediction comparison. Failure groups additionally assert
that no prediction is produced and that the source artifact is untouched.

Outputs (validation reports and measured results) are written under
``/home/wll/uid42/output/``. Run through ``_site_bootstrap.py`` so the sources
in the uid42 tree take precedence over the environment's editable install.
"""

import hashlib
import json
import os
import shutil

import joblib
import numpy as np
import pandas as pd

from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.utils._artifact_contract import (
    ContractValidationError,
    LegacyArtifactWarning,
    _make_envelope,
    load_contracted_model,
    read_contract,
    save_contracted_model,
    validate_artifact,
)

OUTPUT_DIR = "/home/wll/uid42/output"
ARTIFACT_DIR = os.path.join(OUTPUT_DIR, "artifacts")
REPORT_PATH = os.path.join(OUTPUT_DIR, "validation_report.json")
SUMMARY_PATH = os.path.join(OUTPUT_DIR, "summary.txt")


def _reset_dirs():
    shutil.rmtree(OUTPUT_DIR, ignore_errors=True)
    os.makedirs(ARTIFACT_DIR, exist_ok=True)


def _build_categorical():
    rng = np.random.RandomState(7)
    frame = pd.DataFrame(
        {
            "city": rng.choice(["paris", "london", "berlin"], size=90),
            "plan": rng.choice(["free", "pro"], size=90),
            "score": rng.normal(size=90),
        }
    )
    target = (frame["score"].to_numpy() + (frame["city"] == "paris")).astype(int)
    return frame, target


def main():
    _reset_dirs()
    report = {"groups": []}
    summary_lines = []

    # Shared numeric fit -----------------------------------------------------
    X, y = make_classification(
        n_samples=150, n_features=6, n_informative=4, random_state=42
    )
    classifier = LogisticRegression(max_iter=500, random_state=42).fit(X, y)
    expected_predict = classifier.predict(X)
    expected_proba = classifier.predict_proba(X)

    # -- Group 1: exact match ------------------------------------------------
    exact_path = os.path.join(ARTIFACT_DIR, "group1_exact.joblib")
    save_contracted_model(classifier, exact_path, X)
    model, exact_report = load_contracted_model(exact_path, X=X)
    got_predict = model.predict(X)
    got_proba = model.predict_proba(X)
    bit_equal_predict = np.array_equal(expected_predict, got_predict)
    bit_equal_proba = np.array_equal(expected_proba, got_proba)
    max_abs_diff = float(np.abs(expected_proba - got_proba).max())
    exact_hash = hashlib.sha256(open(exact_path, "rb").read()).hexdigest()

    assert exact_report.status == "valid"
    assert bit_equal_predict and bit_equal_proba
    group1 = {
        "group": "exact_match",
        "verdict": exact_report.status,
        "bit_identical_predict": bit_equal_predict,
        "bit_identical_predict_proba": bit_equal_proba,
        "max_abs_predict_proba_difference": max_abs_diff,
        "artifact_sha256": exact_hash,
        "findings": exact_report.to_dict()["findings"],
        "predict_head_before": expected_predict[:8].tolist(),
        "predict_head_after": got_predict[:8].tolist(),
        "predict_proba_first_row_before": expected_proba[0].tolist(),
        "predict_proba_first_row_after": got_proba[0].tolist(),
    }
    report["groups"].append(group1)
    summary_lines.append(
        "[group 1 exact_match]        verdict=valid  bit-identical=True "
        f"max|delta proba|={max_abs_diff:.3e}"
    )

    # -- Group 2: patch-level dependency difference --------------------------
    patch_path = os.path.join(ARTIFACT_DIR, "group2_patch.joblib")
    shutil.copyfile(exact_path, patch_path)
    patch_model = joblib.load(patch_path)
    env = dict(patch_model.contract_.environment)
    numpy_version = env["numpy"]
    major, minor, patch = (int(part) for part in numpy_version.split(".")[:3])
    env["numpy"] = f"{major}.{minor}.{max(patch - 1, 0)}"
    patch_model.contract_.environment = env
    patch_model._envelope = _make_envelope(patch_model.contract_)
    joblib.dump(patch_model, patch_path)

    loaded_patch, patch_report = load_contracted_model(patch_path, X=X)
    assert patch_report.status == "compatible"
    assert any(
        finding.code == "version.patch_difference"
        for finding in patch_report.findings
    )
    # Predictions remain available and still bit-identical in this run.
    patch_predict = loaded_patch.predict(X)
    assert np.array_equal(patch_predict, expected_predict)
    group2 = {
        "group": "patch_version_difference",
        "verdict": patch_report.status,
        "findings": patch_report.to_dict()["findings"],
        "predictions_available": True,
    }
    report["groups"].append(group2)
    summary_lines.append(
        "[group 2 patch_difference]   verdict=compatible  "
        "code=version.patch_difference  predictions_available=True"
    )

    # -- Group 3: same names, reordered columns (pandas contract) ------------
    frame, target = _build_categorical()
    encoded = pd.get_dummies(frame, columns=["city", "plan"]).astype(float)
    from sklearn.linear_model import LinearRegression

    regressor = LinearRegression().fit(encoded, target)
    reorder_path = os.path.join(ARTIFACT_DIR, "group3_reorder.joblib")
    save_contracted_model(regressor, reorder_path, encoded)
    shuffled = encoded[encoded.columns[::-1]]
    reorder_model, reorder_report = load_contracted_model(
        reorder_path, X=shuffled
    )
    assert reorder_report.status == "compatible"
    assert reorder_report.reordered is True
    aligned_predict = reorder_model.predict(shuffled)
    assert np.array_equal(regressor.predict(encoded), aligned_predict)
    group3 = {
        "group": "column_reorder",
        "verdict": reorder_report.status,
        "reordered": True,
        "findings": reorder_report.to_dict()["findings"],
        "bit_identical_after_name_alignment": bool(
            np.array_equal(regressor.predict(encoded), aligned_predict)
        ),
    }
    report["groups"].append(group3)
    summary_lines.append(
        "[group 3 column_reorder]     verdict=compatible  "
        "aligned_by_name=True  bit-identical=True"
    )

    # -- Group 4: missing column, no prediction, file untouched --------------
    missing_path = os.path.join(ARTIFACT_DIR, "group4_missing.joblib")
    save_contracted_model(regressor, missing_path, encoded)
    missing_bytes = open(missing_path, "rb").read()
    reduced = encoded.drop(columns=[encoded.columns[1]])
    missing_report = validate_artifact(missing_path, X=reduced)
    assert missing_report.status == "invalid"
    assert any(
        finding.code == "feature.missing" for finding in missing_report.findings
    )
    prediction_raised = False
    try:
        load_contracted_model(missing_path, X=reduced)
    except ContractValidationError:
        prediction_raised = True
    assert prediction_raised
    assert open(missing_path, "rb").read() == missing_bytes
    group4 = {
        "group": "missing_column",
        "verdict": missing_report.status,
        "findings": missing_report.to_dict()["findings"],
        "load_raises_contract_validation_error": True,
        "source_file_unchanged": True,
    }
    report["groups"].append(group4)
    summary_lines.append(
        "[group 4 missing_column]     verdict=invalid  "
        "code=feature.missing  load_raises=True  source_unchanged=True"
    )

    # -- Group 5: categorical vocabulary drift -------------------------------
    pipe = Pipeline(
        steps=[
            ("encoder", OneHotEncoder(handle_unknown="ignore")),
            ("clf", LogisticRegression(max_iter=500)),
        ]
    ).fit(frame[["city", "plan"]], target)
    drift_path = os.path.join(ARTIFACT_DIR, "group5_category_drift.joblib")
    save_contracted_model(pipe, drift_path, frame[["city", "plan"]])
    drift_bytes = open(drift_path, "rb").read()
    drifted = frame.copy()
    drifted.loc[0, "city"] = "tokyo"
    drift_report = validate_artifact(
        drift_path, X=drifted[["city", "plan"]]
    )
    assert drift_report.status == "invalid"
    assert any(
        finding.code == "feature.category_drift"
        for finding in drift_report.findings
    )
    # The offending category value must not appear anywhere in the report.
    rendered = str(drift_report)
    assert "tokyo" not in rendered
    drift_raises = False
    try:
        load_contracted_model(drift_path, X=drifted[["city", "plan"]])
    except ContractValidationError:
        drift_raises = True
    assert drift_raises
    assert open(drift_path, "rb").read() == drift_bytes
    group5 = {
        "group": "category_drift",
        "verdict": drift_report.status,
        "findings": drift_report.to_dict()["findings"],
        "load_raises_contract_validation_error": True,
        "source_file_unchanged": True,
        "offending_value_absent_from_messages": "tokyo" not in rendered,
    }
    report["groups"].append(group5)
    summary_lines.append(
        "[group 5 category_drift]     verdict=invalid  "
        "code=feature.category_drift  load_raises=True  "
        "source_unchanged=True  no_value_leak=True"
    )

    # -- Bonus: legacy artifact remains readable -----------------------------
    legacy_path = os.path.join(ARTIFACT_DIR, "legacy_plain.joblib")
    joblib.dump(classifier, legacy_path)
    import warnings

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        legacy_info = read_contract(legacy_path)
    assert any(
        isinstance(item.message, LegacyArtifactWarning) for item in caught
    )
    report["legacy_artifact"] = legacy_info
    summary_lines.append(
        "[legacy plain joblib]        readable=True  "
        f"class={legacy_info['estimator_class']}"
    )

    with open(REPORT_PATH, "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, ensure_ascii=False)
    with open(SUMMARY_PATH, "w", encoding="utf-8") as handle:
        handle.write("\n".join(summary_lines) + "\n")

    print("\n".join(summary_lines))
    print(f"\nfull report: {REPORT_PATH}")
    print(f"artifacts:   {ARTIFACT_DIR}")


if __name__ == "__main__":
    main()
