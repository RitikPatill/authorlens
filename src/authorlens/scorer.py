"""Pairwise similarity scoring — weighted fusion of embeddings + stylometrics (M3)."""

from dataclasses import dataclass

from authorlens import embedder, features

# Per-feature normalisation caps (transparent, auditable)
_CAPS: dict[str, float] = {
    "ttr_dist": 1.0,
    "mean_sent_len_dist": 30.0,
    "std_sent_len_dist": 20.0,
    "punct_density_dist": 0.1,
    "yule_k_dist": 200.0,
    "burstiness_dist": 2.0,
    "func_word_cosine_dist": 1.0,
}


@dataclass
class AuthorshipScore:
    overall: float           # 0 = completely different, 1 = identical
    embedding_sim: float     # raw cosine similarity of embeddings
    stylometric_sim: float   # 1 - normalised aggregate stylometric distance
    feature_deltas: dict[str, float]  # raw output of features.compare()


def fuse(text_a: str, text_b: str, emb_weight: float = 0.5) -> AuthorshipScore:
    """Combine embedding cosine similarity and stylometric distance into one score."""
    fv_a = features.extract(text_a)
    fv_b = features.extract(text_b)
    feature_deltas = features.compare(fv_a, fv_b)

    # Normalise each delta to [0, 1] using per-feature caps, then average
    normalised = [
        min(feature_deltas[key] / cap, 1.0)
        for key, cap in _CAPS.items()
    ]
    stylometric_dist = sum(normalised) / len(normalised)
    stylometric_sim = max(0.0, min(1.0, 1.0 - stylometric_dist))

    emb_a = embedder.encode(text_a)
    emb_b = embedder.encode(text_b)
    embedding_sim = embedder.cosine_similarity(emb_a, emb_b)

    overall = emb_weight * embedding_sim + (1.0 - emb_weight) * stylometric_sim
    overall = max(0.0, min(1.0, overall))  # clamp: embedding_sim in [-1,1] can push overall negative

    return AuthorshipScore(
        overall=overall,
        embedding_sim=embedding_sim,
        stylometric_sim=stylometric_sim,
        feature_deltas=feature_deltas,
    )
