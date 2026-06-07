# AuthorLens

> Fingerprint writing style. Attribute authorship. Explain the diff.

![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue)
![License: MIT](https://img.shields.io/badge/license-MIT-green)
![Status: WIP](https://img.shields.io/badge/status-WIP-yellow)

## Motivation

The explosion of LLM-generated content has made authorship attribution a live, unsolved problem. Researchers, journalists, and educators need lightweight, interpretable tools — not black-box classifiers. AuthorLens combines classical stylometric feature engineering with modern sentence-transformer embeddings to produce an explainable similarity score grounded in the 2025 paper *Explainable Disentangled Representation Learning for Generalizable Authorship Attribution in the Era of Generative AI*.

## Architecture

```
 doc1.txt ──┐                          ┌─── Rich terminal table
            ├──► features.py ──────────┤
 doc2.txt ──┘    (stylometrics)        │
            │                          ├──► CLI  (typer)
            ├──► embedder.py ──────────┤
                 (all-MiniLM-L6-v2)    └──► API  (fastapi /compare)
                        │
                        ▼
                    scorer.py
               (weighted fusion score
                + top-3 feature diff)
```

## Current Status

**M3 — embedding fingerprinting complete.**

| Component | State |
|---|---|
| Repo layout (`src/`, `tests/`, `data/demo/`) | done |
| Package version (`authorlens.__version__ = "0.1.0"`) | done |
| Stub modules (`features`, `embedder`, `scorer`, `cli`, `api`) | done |
| Smoke test (`test_version`, `test_stub_imports`) | done |
| Pinned dependencies (`requirements.txt`) | done |
| `pyproject.toml`, `.gitignore`, MIT license | done |
| `FeatureVector` dataclass + `extract()` + `compare()` | done |
| 8 stylometric features: TTR, mean/std sent len, punct density, Yule K, burstiness, func-word freq | done |
| 10 unit tests in `tests/test_features.py` | done |
| **`embedder.py`: lazy `all-MiniLM-L6-v2` load, `encode()` → 384-dim float32, `cosine_similarity()`** | **done** |
| **`scorer.py`: `AuthorshipScore` dataclass + `fuse()` weighted fusion** | **done** |
| **10 unit tests in `tests/test_embedder.py` + `tests/test_scorer.py`** | **done** |
| Rich CLI table, FastAPI endpoint + HTML UI | M4 – M5 |

## Quickstart

```bash
pip install -r requirements.txt
authorlens compare doc1.txt doc2.txt   # stub — implemented in M4
# or scan a folder
authorlens scan ./folder/              # stub — implemented in M4
```

Start the REST API:

```bash
uvicorn authorlens.api:app --reload
# POST {"texts": ["...", "..."]} to http://localhost:8000/compare  # stub — M5
# Open http://localhost:8000/ for the single-page UI              # stub — M5
```

## Project Layout

```
authorlens/
├── src/
│   └── authorlens/
│       ├── __init__.py      # package version
│       ├── features.py      # stylometric feature extraction (M2)
│       ├── embedder.py      # sentence-transformer fingerprint (M3)
│       ├── scorer.py        # weighted fusion scorer (M3)
│       ├── cli.py           # typer CLI entry point (M4)
│       └── api.py           # fastapi REST + HTML UI (M5)
├── tests/
│   ├── __init__.py
│   ├── test_smoke.py
│   ├── test_features.py   # stylometric feature unit tests (M2)
│   ├── test_embedder.py   # embedding unit tests (M3)
│   └── test_scorer.py     # fusion scorer unit tests (M3)
├── data/
│   └── demo/               # 10 human vs GPT-4 paragraph pairs (M6)
├── requirements.txt
├── requirements-dev.txt
├── pyproject.toml
└── README.md
```

## Roadmap

- [x] **M1** — Repo scaffold, stub modules, smoke tests, pinned dependencies, MIT license
- [x] **M2** — Implement 8+ stylometric features (TTR, sentence length, punctuation density, function-word frequency, Yule's K, burstiness)
- [x] **M3** — Integrate `all-MiniLM-L6-v2` sentence-transformer embeddings (CPU, no API key) + weighted fusion scorer
- [ ] **M4** — Rich CLI table with top-3 diverging feature explanations
- [ ] **M5** — FastAPI `/compare` endpoint + minimal single-page HTML form
- [ ] **M6** — Bundle demo dataset (10 human vs GPT-4 pairs) and end-to-end integration tests

## License

MIT — see [LICENSE](LICENSE).
