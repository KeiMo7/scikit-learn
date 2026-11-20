"""
=====================================================
Exporting and replaying the state of a pipeline
=====================================================

Two engineers can run the same training script on the same data and obtain
evaluation metrics that differ by a fraction of a percent, only to find after
days that one step of the chain was not the same. The code rarely shows it:
``Pipeline`` exposes ``get_params``, but the attributes that ``fit`` learns,
the dimensionality each step receives and the seeds it consumes are not
collected anywhere.

This example exports that state, replays it, and then replaces a single
transformer to show that the difference can be attributed to that step.

"""

# Authors: The scikit-learn developers
# SPDX-License-Identifier: BSD-3-Clause

import tempfile

import numpy as np

from sklearn.compose import ColumnTransformer
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
from sklearn.exceptions import ChainStateCorruptionError
from sklearn.metrics import accuracy_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder, StandardScaler

# %%
# Build and fit a chain
# ---------------------
#
# The chain mixes a numeric branch and a categorical branch in a
# :class:`~sklearn.compose.ColumnTransformer`, followed by a random forest
# whose seed matters.
X, y = make_classification(
    n_samples=300, n_features=6, n_informative=4, random_state=42
)


def make_pipeline(scaler):
    return Pipeline(
        [
            (
                "prep",
                ColumnTransformer(
                    [
                        ("num", scaler, [0, 1, 2]),
                        (
                            "cat",
                            OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                            [3],
                        ),
                    ],
                    remainder="passthrough",
                ),
            ),
            ("clf", RandomForestClassifier(n_estimators=10, random_state=7, n_jobs=2)),
        ]
    )


pipe = make_pipeline(StandardScaler()).fit(X, y)

# %%
# Export the state
# ----------------
#
# The exported document describes every step: its class, its parameters, its
# learned attributes, its input and output dimensionality, the step it depends
# on, and the seeds it consumed. It contains no training data.
with tempfile.TemporaryDirectory() as directory:
    reference = f"{directory}/reference.jsonl"
    summary = pipe.export_chain_state(reference)

    for step in summary["steps"]:
        print(
            f"{step['name']:>5} | {step['class'].split(':')[-1]:<26} | "
            f"in={step['n_features_in']} out={step['n_features_out']} | "
            f"depends on {step['depends_on']} | "
            f"seeded={step['randomness']['seeded']}"
        )

    # %%
    # Replay on another machine
    # -------------------------
    #
    # Loading the document rebuilds the fitted chain without calling ``fit``.
    # Arrays are stored as exact bytes, so the replay is bit-for-bit identical
    # and the report says so explicitly.
    restored = Pipeline.load_chain_state(reference)
    print(
        "loaded pipeline is fitted and reproduces the predictions:",
        np.array_equal(restored.predict(X), pipe.predict(X)),
    )

    report = pipe.replay_chain_state(reference, X)
    print(f"equal={report['equal']} exact={report['exact']}")
    print("first divergent step:", report["first_divergent_step"])

    # %%
    # Replace one step and locate the difference
    # ------------------------------------------
    #
    # The metrics of a chain in which the numeric branch uses ``MinMaxScaler``
    # instead of ``StandardScaler`` are very close, but the document diff names
    # the transformer that was replaced and the step that follows it.
    modified = make_pipeline(MinMaxScaler()).fit(X, y)
    modified_path = f"{directory}/modified.jsonl"
    modified.export_chain_state(modified_path)

    print(f"reference accuracy: {accuracy_score(y, pipe.predict(X)):.3f}")
    print(f"modified accuracy:  {accuracy_score(y, modified.predict(X)):.3f}")

    diff = modified.diff_chain_state(reference)
    print("first changed step:", diff["first_changed_step"])
    print("first changed transformer:", diff["first_changed_component"])
    for entry in diff["entries"]:
        if entry["kind"] == "class_changed":
            print(f"  {entry['old']} -> {entry['new']}")

    # %%
    # A corrupted document is reported precisely
    # ------------------------------------------
    #
    # Each line of the document carries its own checksum. Editing one step is
    # therefore reported with the faulty segment, instead of a generic failure.
    lines = open(reference).read().splitlines()
    lines[2] = lines[2].replace('"data":"', '"data":"AA', 1)
    corrupted = f"{directory}/corrupted.jsonl"
    with open(corrupted, "w") as handle:
        handle.write("\n".join(lines) + "\n")

    try:
        Pipeline.load_chain_state(corrupted)
    except ChainStateCorruptionError as error:
        print("corruption reported at:", error.fragment)
