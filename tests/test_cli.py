"""CLI tests for M4 (typer + rich)."""

from __future__ import annotations

from unittest.mock import patch

import pytest
from typer.testing import CliRunner

from authorlens.cli import app
from authorlens.scorer import AuthorshipScore

runner = CliRunner()

_MOCK_SCORE = AuthorshipScore(
    overall=0.75,
    embedding_sim=0.80,
    stylometric_sim=0.70,
    feature_deltas={
        "ttr_dist": 0.05,
        "mean_sent_len_dist": 0.30,
        "std_sent_len_dist": 0.10,
        "punct_density_dist": 0.02,
        "yule_k_dist": 0.15,
        "burstiness_dist": 0.08,
        "func_word_cosine_dist": 0.22,
    },
)

_MOCK_SCORE_HIGH = AuthorshipScore(
    overall=0.95,
    embedding_sim=0.95,
    stylometric_sim=0.95,
    feature_deltas={
        "ttr_dist": 0.01,
        "mean_sent_len_dist": 0.02,
        "std_sent_len_dist": 0.01,
        "punct_density_dist": 0.01,
        "yule_k_dist": 0.01,
        "burstiness_dist": 0.01,
        "func_word_cosine_dist": 0.01,
    },
)

_MOCK_SCORE_LOW = AuthorshipScore(
    overall=0.50,
    embedding_sim=0.50,
    stylometric_sim=0.50,
    feature_deltas={
        "ttr_dist": 0.05,
        "mean_sent_len_dist": 0.30,
        "std_sent_len_dist": 0.10,
        "punct_density_dist": 0.02,
        "yule_k_dist": 0.15,
        "burstiness_dist": 0.08,
        "func_word_cosine_dist": 0.22,
    },
)


def test_compare_prints_table(tmp_path):
    """compare command prints output including 'Overall' and exits 0."""
    fa = tmp_path / "a.txt"
    fb = tmp_path / "b.txt"
    fa.write_text("Hello world. This is a test.")
    fb.write_text("Goodbye world. This is another test.")

    with patch("authorlens.cli.scorer.fuse", return_value=_MOCK_SCORE):
        result = runner.invoke(app, ["compare", str(fa), str(fb)])

    assert result.exit_code == 0
    assert "Overall" in result.output


def test_compare_threshold_exit(tmp_path):
    """compare exits 1 when overall score exceeds threshold."""
    fa = tmp_path / "a.txt"
    fb = tmp_path / "b.txt"
    fa.write_text("Hello world.")
    fb.write_text("Hello world.")

    with patch("authorlens.cli.scorer.fuse", return_value=_MOCK_SCORE_HIGH):
        result = runner.invoke(app, ["compare", str(fa), str(fb), "--threshold", "0.9"])

    assert result.exit_code == 1


def test_compare_threshold_pass(tmp_path):
    """compare exits 0 when overall score is below threshold."""
    fa = tmp_path / "a.txt"
    fb = tmp_path / "b.txt"
    fa.write_text("Hello world.")
    fb.write_text("Hello world.")

    with patch("authorlens.cli.scorer.fuse", return_value=_MOCK_SCORE_LOW):
        result = runner.invoke(app, ["compare", str(fa), str(fb), "--threshold", "0.9"])

    assert result.exit_code == 0


def test_scan_finds_pairs(tmp_path):
    """scan command lists both file stems in output."""
    (tmp_path / "alpha.txt").write_text("The quick brown fox.")
    (tmp_path / "beta.txt").write_text("A lazy dog sits still.")

    with patch("authorlens.cli.scorer.fuse", return_value=_MOCK_SCORE):
        result = runner.invoke(app, ["scan", str(tmp_path)])

    assert result.exit_code == 0
    assert "alpha" in result.output
    assert "beta" in result.output


def test_scan_threshold_exit(tmp_path):
    """scan exits 1 when any pair exceeds threshold."""
    (tmp_path / "alpha.txt").write_text("The quick brown fox.")
    (tmp_path / "beta.txt").write_text("A lazy dog sits still.")

    with patch("authorlens.cli.scorer.fuse", return_value=_MOCK_SCORE_HIGH):
        result = runner.invoke(app, ["scan", str(tmp_path), "--threshold", "0.9"])

    assert result.exit_code == 1


def test_compare_missing_file(tmp_path):
    """compare exits 2 (Typer validation) for a non-existent file."""
    fa = tmp_path / "real.txt"
    fa.write_text("Hello.")
    missing = tmp_path / "ghost.txt"

    result = runner.invoke(app, ["compare", str(fa), str(missing)])

    assert result.exit_code == 2
