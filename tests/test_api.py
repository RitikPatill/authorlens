"""Tests for FastAPI REST API (M5)."""

import pytest
from fastapi.testclient import TestClient

from authorlens.api import app

client = TestClient(app)

SHORT_A = "The cat sat on the mat. It was a sunny day."
SHORT_B = "A dog ran across the field. The weather was warm and bright."

VALID_VERDICTS = {"likely same author", "uncertain", "likely different authors"}


def test_health():
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok"}


def test_root_html():
    resp = client.get("/")
    assert resp.status_code == 200
    assert "text/html" in resp.headers["content-type"]
    body = resp.text
    assert "<textarea" in body or "<form" in body


def test_compare_basic():
    resp = client.post("/compare", json={"texts": [SHORT_A, SHORT_B]})
    assert resp.status_code == 200
    data = resp.json()

    assert set(data.keys()) >= {"overall_score", "embedding_sim", "stylometric_sim", "feature_breakdown", "verdict"}
    assert 0.0 <= data["overall_score"] <= 1.0
    assert 0.0 <= data["stylometric_sim"] <= 1.0
    assert data["verdict"] in VALID_VERDICTS

    fb = data["feature_breakdown"]
    expected_keys = {"ttr_dist", "mean_sent_len_dist", "std_sent_len_dist",
                     "punct_density_dist", "yule_k_dist", "burstiness_dist", "func_word_cosine_dist"}
    assert set(fb.keys()) == expected_keys


def test_compare_custom_weight():
    resp = client.post("/compare", json={"texts": [SHORT_A, SHORT_B], "emb_weight": 0.8})
    assert resp.status_code == 200
    data = resp.json()
    assert 0.0 <= data["overall_score"] <= 1.0
    assert data["verdict"] in VALID_VERDICTS


def test_compare_validation_error():
    resp = client.post("/compare", json={"texts": [SHORT_A]})
    assert resp.status_code == 422


def test_cors_header():
    resp = client.options(
        "/compare",
        headers={"Origin": "http://localhost:3000", "Access-Control-Request-Method": "POST"},
    )
    assert "access-control-allow-origin" in resp.headers
