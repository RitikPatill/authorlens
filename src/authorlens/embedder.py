"""Sentence-transformer embedding fingerprint (M3)."""

import numpy as np
from sentence_transformers import SentenceTransformer
from rich.progress import Progress, SpinnerColumn, TextColumn

_MODEL_NAME = "all-MiniLM-L6-v2"
_model: SentenceTransformer | None = None


def _get_model() -> SentenceTransformer:
    global _model
    if _model is not None:
        return _model
    with Progress(SpinnerColumn(), TextColumn("[cyan]Loading model…")) as progress:
        progress.add_task("load", total=None)
        _model = SentenceTransformer(_MODEL_NAME)
    return _model


def encode(text: str) -> np.ndarray:
    """Encode text into a 384-dim float32 embedding vector."""
    model = _get_model()
    result = model.encode([text], convert_to_numpy=True, show_progress_bar=False)
    return result[0].astype(np.float32)


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity between two vectors; returns float in [-1, 1]."""
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))
