"""Stylometric feature extraction (M2)."""

from dataclasses import dataclass, field
import re
import math
import string
from collections import Counter

import numpy as np

FUNCTION_WORDS: list[str] = [
    "the", "be", "to", "of", "and", "a", "in", "that", "have", "it",
    "for", "not", "on", "with", "he", "as", "you", "do", "at", "this",
    "but", "his", "by", "from", "they", "we", "say", "her", "she", "or",
    "an", "will", "my", "one", "all", "would", "there", "their", "what", "so",
    "up", "out", "if", "about", "who", "get", "which", "go", "me", "when",
]


@dataclass
class FeatureVector:
    ttr: float
    mean_sent_len: float
    std_sent_len: float
    punct_density: float
    yule_k: float
    burstiness: float
    func_word_freq: np.ndarray
    token_count: int = field(repr=False)


def _tokenize(text: str) -> list[str]:
    """Lowercase and strip punctuation before tokenizing."""
    text = text.lower()
    text = text.translate(str.maketrans("", "", string.punctuation))
    return text.split()


def _sentences(text: str) -> list[str]:
    """Split text into sentences, keeping punctuation attached."""
    parts = re.split(r"(?<=[.!?])\s+", text.strip())
    return [s for s in parts if s.strip()]


def _yule_k(tokens: list[str]) -> float:
    n = len(tokens)
    if n == 0:
        return 0.0
    freq = Counter(tokens)
    freq_of_freq = Counter(freq.values())
    sigma = sum(v * (m ** 2) for m, v in freq_of_freq.items())
    k = 1e4 * (sigma - n) / (n ** 2)
    return max(0.0, k)


def _burstiness(sent_lengths: list[int]) -> float:
    if len(sent_lengths) <= 1:
        return 0.0
    arr = np.array(sent_lengths, dtype=float)
    mu = arr.mean()
    sigma = arr.std()
    if sigma + mu == 0:
        return 0.0
    return float((sigma - mu) / (sigma + mu))


def _func_word_freq(tokens: list[str]) -> np.ndarray:
    total = len(tokens)
    if total == 0:
        return np.zeros(50)
    counts = Counter(tokens)
    vec = np.array([counts.get(w, 0) / total for w in FUNCTION_WORDS], dtype=float)
    s = vec.sum()
    if s == 0:
        return np.zeros(50)
    return vec / s


def extract(text: str) -> FeatureVector:
    tokens = _tokenize(text)
    n = len(tokens)

    # TTR
    ttr = len(set(tokens)) / n if n > 0 else 0.0

    # Sentence lengths
    sents = _sentences(text)
    sent_lengths = [len(s.split()) for s in sents] if sents else []
    if sent_lengths:
        mean_sent_len = float(np.mean(sent_lengths))
        std_sent_len = float(np.std(sent_lengths))
    else:
        mean_sent_len = 0.0
        std_sent_len = 0.0

    # Punctuation density
    if len(text) > 0:
        punct_density = sum(1 for c in text if c in string.punctuation) / len(text)
    else:
        punct_density = 0.0

    return FeatureVector(
        ttr=ttr,
        mean_sent_len=mean_sent_len,
        std_sent_len=std_sent_len,
        punct_density=punct_density,
        yule_k=_yule_k(tokens),
        burstiness=_burstiness(sent_lengths),
        func_word_freq=_func_word_freq(tokens),
        token_count=n,
    )


def _cosine_sim(u: np.ndarray, v: np.ndarray) -> float:
    return float(np.dot(u, v) / (np.linalg.norm(u) * np.linalg.norm(v) + 1e-9))


def compare(a: FeatureVector, b: FeatureVector) -> dict[str, float]:
    return {
        "ttr_dist": abs(a.ttr - b.ttr),
        "mean_sent_len_dist": abs(a.mean_sent_len - b.mean_sent_len),
        "std_sent_len_dist": abs(a.std_sent_len - b.std_sent_len),
        "punct_density_dist": abs(a.punct_density - b.punct_density),
        "yule_k_dist": abs(a.yule_k - b.yule_k),
        "burstiness_dist": abs(a.burstiness - b.burstiness),
        "func_word_cosine_dist": float(1 - _cosine_sim(a.func_word_freq, b.func_word_freq)),
    }
