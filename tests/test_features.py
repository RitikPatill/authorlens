import pytest
import numpy as np
from authorlens.features import extract, compare, FeatureVector

SHORT = "The cat sat on the mat. The dog ran fast."
LONG = "However, the investigation revealed that the primary cause was not immediately obvious to the observers."


def test_extract_returns_feature_vector():
    fv = extract(SHORT)
    assert isinstance(fv, FeatureVector)


def test_ttr_between_zero_and_one():
    fv = extract(SHORT)
    assert 0.0 < fv.ttr <= 1.0


def test_mean_sent_len_positive():
    fv = extract(SHORT)
    assert fv.mean_sent_len > 0


def test_punct_density_range():
    fv = extract(SHORT)
    assert 0.0 <= fv.punct_density <= 1.0


def test_func_word_freq_shape():
    fv = extract(SHORT)
    assert fv.func_word_freq.shape == (50,)


def test_yule_k_positive():
    fv = extract(SHORT)
    assert fv.yule_k >= 0.0


def test_compare_keys():
    a, b = extract(SHORT), extract(LONG)
    d = compare(a, b)
    expected = {
        "ttr_dist", "mean_sent_len_dist", "std_sent_len_dist",
        "punct_density_dist", "yule_k_dist", "burstiness_dist",
        "func_word_cosine_dist",
    }
    assert set(d.keys()) == expected


def test_compare_self_is_zero():
    fv = extract(SHORT)
    d = compare(fv, fv)
    for k, v in d.items():
        assert abs(v) < 1e-6, f"{k} should be 0 for identical texts"


def test_func_word_cosine_dist_range():
    a, b = extract(SHORT), extract(LONG)
    d = compare(a, b)
    assert 0.0 <= d["func_word_cosine_dist"] <= 1.0


def test_empty_text_does_not_crash():
    fv = extract("")
    assert fv.ttr == 0.0
    assert fv.mean_sent_len == 0.0
