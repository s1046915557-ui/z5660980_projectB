"""Tests for reproducible sentiment development evaluation."""

import pathlib
import sys

import pandas as pd
import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from scripts import run_part_b


def test_development_labels_merge_one_to_one():
    """Every approved label must map to exactly one locked sample ID."""
    candidates = pd.DataFrame(
        {
            "sample_id": ["SV001", "SV002"],
            "text_raw": ["First headline", "Second headline"],
        }
    )
    labels = pd.DataFrame(
        {
            "sample_id": ["SV002", "SV001"],
            "approved_label": ["positive", "negative"],
            "error_type": ["finance_lexicon_gap", "correct"],
            "review_reason": ["buy signal", "clear loss"],
        }
    )

    merged = run_part_b._merge_sentiment_development_labels(candidates, labels)

    assert merged["sample_id"].tolist() == ["SV001", "SV002"]
    assert merged["approved_label"].tolist() == ["negative", "positive"]

    with pytest.raises(ValueError, match="do not match"):
        run_part_b._merge_sentiment_development_labels(
            candidates,
            labels.assign(sample_id=["SV002", "SV003"]),
        )


def test_holdout_labels_merge_one_to_one():
    """Blind holdout labels must cover the locked sample IDs exactly once."""
    holdout = pd.DataFrame(
        {
            "sample_id": ["SH001", "SH002"],
            "text_raw": ["First headline", "Second headline"],
        }
    )
    labels = pd.DataFrame(
        {
            "sample_id": ["SH002", "SH001"],
            "approved_label": ["positive", "neutral"],
            "review_reason": ["clear upside", "impact unclear"],
        }
    )

    merged = run_part_b._merge_sentiment_holdout_labels(holdout, labels)

    assert merged["sample_id"].tolist() == ["SH001", "SH002"]
    assert merged["approved_label"].tolist() == ["neutral", "positive"]

    with pytest.raises(ValueError, match="do not match"):
        run_part_b._merge_sentiment_holdout_labels(
            holdout,
            labels.assign(sample_id=["SH002", "SH003"]),
        )


def test_frozen_sentiment_rule_hash_matches_source():
    """The model file must remain byte-identical after holdout labels are seen."""
    model_path = run_part_b.PROJECT_ROOT / "src" / "sentiment.py"

    assert (
        run_part_b._file_sha256(model_path)
        == run_part_b.FROZEN_SENTIMENT_RULE_SHA256
    )


def test_sentiment_metrics_reward_corrected_predictions():
    """Metrics improve when finance predictions match every approved label."""
    evaluation = pd.DataFrame(
        {
            "approved_label": ["negative", "neutral", "positive", "positive"],
            "baseline_class": ["negative", "negative", "neutral", "positive"],
            "finance_class": ["negative", "neutral", "positive", "positive"],
        }
    )

    metrics = run_part_b._sentiment_classification_metrics(evaluation).set_index(
        "model_name"
    )
    baseline = metrics.loc["Unmodified NLTK VADER"]
    finance = metrics.loc[
        "NLTK VADER + conservative finance extension"
    ]

    assert baseline["accuracy"] == pytest.approx(0.5)
    assert baseline["balanced_accuracy"] == pytest.approx(0.5)
    assert baseline["macro_f1"] == pytest.approx(4.0 / 9.0)
    assert finance["accuracy"] == pytest.approx(1.0)
    assert finance["balanced_accuracy"] == pytest.approx(1.0)
    assert finance["macro_f1"] == pytest.approx(1.0)
