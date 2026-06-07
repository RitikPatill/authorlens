"""Tests for authorlens.embedder (M3)."""

import numpy as np
import pytest
from unittest.mock import MagicMock

import authorlens.embedder as embedder_mod


@pytest.fixture(autouse=False)
def mock_model(monkeypatch):
    """Patch the module-level _model singleton so no download occurs."""
    mock = MagicMock()
    mock.encode.return_value = np.ones((1, 384), dtype=np.float32)
    monkeypatch.setattr(embedder_mod, "_model", mock)
    return mock


def test_encode_shape(mock_model):
    result = embedder_mod.encode("Hello world")
    assert result.shape == (384,)


def test_encode_dtype(mock_model):
    result = embedder_mod.encode("Hello world")
    assert result.dtype == np.float32


def test_cosine_similarity_self():
    v = np.random.rand(384).astype(np.float32)
    sim = embedder_mod.cosine_similarity(v, v)
    assert abs(sim - 1.0) < 1e-5


def test_cosine_similarity_orthogonal():
    a = np.zeros(384, dtype=np.float32)
    b = np.zeros(384, dtype=np.float32)
    a[0] = 1.0
    b[1] = 1.0
    sim = embedder_mod.cosine_similarity(a, b)
    assert abs(sim) < 1e-5


def test_cosine_similarity_range():
    rng = np.random.default_rng(42)
    a = rng.random(384).astype(np.float32)
    b = rng.random(384).astype(np.float32)
    sim = embedder_mod.cosine_similarity(a, b)
    assert -1.0 <= sim <= 1.0
