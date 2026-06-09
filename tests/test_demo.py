"""Tests for the M6 demo dataset and pipeline integrity."""

import json
from pathlib import Path

import pytest

# Repo root is two levels up from this file
REPO_ROOT = Path(__file__).parent.parent
LABELS_PATH = REPO_ROOT / "data" / "demo" / "labels.json"


def _load_labels() -> list[dict]:
    with open(LABELS_PATH, encoding="utf-8") as f:
        return json.load(f)


def test_all_pairs_exist():
    """Every path referenced in labels.json must exist on disk."""
    pairs = _load_labels()
    missing = []
    for p in pairs:
        for key in ("human", "gpt4"):
            path = REPO_ROOT / p[key]
            if not path.exists():
                missing.append(str(path))
    assert not missing, f"Missing demo files: {missing}"


def test_labels_schema():
    """labels.json must have exactly 10 entries, each with required fields."""
    pairs = _load_labels()
    assert len(pairs) == 10, f"Expected 10 pairs, got {len(pairs)}"
    required_keys = {"id", "topic", "human", "gpt4", "expected"}
    for p in pairs:
        assert required_keys.issubset(p.keys()), f"Missing keys in entry {p}"
        assert p["expected"] == "different", f"Unexpected label in entry {p}"


def test_pipeline_runs_on_all_pairs():
    """scorer.fuse() must return an AuthorshipScore with overall in [0, 1] for each pair."""
    import sys
    sys.path.insert(0, str(REPO_ROOT / "src"))
    from authorlens import scorer

    pairs = _load_labels()
    for p in pairs:
        human_text = (REPO_ROOT / p["human"]).read_text(encoding="utf-8")
        gpt4_text = (REPO_ROOT / p["gpt4"]).read_text(encoding="utf-8")
        result = scorer.fuse(human_text, gpt4_text)
        assert hasattr(result, "overall"), "AuthorshipScore missing 'overall' attribute"
        assert 0.0 <= result.overall <= 1.0, (
            f"overall score {result.overall} out of [0, 1] for pair {p['id']}"
        )


def test_human_gpt4_scores_below_threshold():
    """All 10 pairs must score < 0.6 (not 'likely same author')."""
    import sys
    sys.path.insert(0, str(REPO_ROOT / "src"))
    from authorlens import scorer

    pairs = _load_labels()
    above_threshold = []
    for p in pairs:
        human_text = (REPO_ROOT / p["human"]).read_text(encoding="utf-8")
        gpt4_text = (REPO_ROOT / p["gpt4"]).read_text(encoding="utf-8")
        result = scorer.fuse(human_text, gpt4_text)
        if result.overall >= 0.6:
            above_threshold.append((p["id"], p["topic"], result.overall))
    assert not above_threshold, (
        f"Pairs scoring >= 0.6 (unexpected similarity): {above_threshold}"
    )
