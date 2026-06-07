"""Tests for authorlens.scorer (M3)."""

import numpy as np
import pytest
from unittest.mock import MagicMock

import authorlens.embedder as embedder_mod
from authorlens.scorer import AuthorshipScore, fuse

TEXT_A = (
    "The quick brown fox jumps over the lazy dog. "
    "It was a bright cold day in April. "
    "All happy families are alike."
)
TEXT_B = (
    "In the beginning God created the heavens and the earth. "
    "A rose is a rose is a rose. "
    "Call me Ishmael."
)

_EXPECTED_KEYS = {
    "ttr_dist",
    "mean_sent_len_dist",
    "std_sent_len_dist",
    "punct_density_dist",
    "yule_k_dist",
    "burstiness_dist",
    "func_word_cosine_dist",
}


@pytest.fixture(autouse=True)
def mock_embedder(monkeypatch):
    """Return deterministic embeddings without loading the real model."""
    mock = MagicMock()
    mock.encode.return_value = np.ones((1, 384), dtype=np.float32)
    monkeypatch.setattr(embedder_mod, "_model", mock)


def test_fuse_returns_authorship_score():
    result = fuse(TEXT_A, TEXT_B)
    assert isinstance(result, AuthorshipScore)


def test_fuse_overall_range():
    result = fuse(TEXT_A, TEXT_B)
    assert 0.0 <= result.overall <= 1.0


def test_fuse_identical_texts_high():
    result = fuse(TEXT_A, TEXT_A)
    assert result.overall > 0.9


def test_fuse_custom_weight():
    result = fuse(TEXT_A, TEXT_B, emb_weight=1.0)
    assert abs(result.overall - result.embedding_sim) < 1e-6


def test_fuse_feature_deltas_keys():
    result = fuse(TEXT_A, TEXT_B)
    assert set(result.feature_deltas.keys()) == _EXPECTED_KEYS
