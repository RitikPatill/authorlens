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

**M1 — scaffold complete.** The repository layout, package skeleton, and development infrastructure are in place. All source modules exist as stubs; no feature logic has been implemented yet.

| Component | State |
|---|---|
| Repo layout (`src/`, `tests/`, `data/demo/`) | done |
| Package version (`authorlens.__version__ = "0.1.0"`) | done |
| Stub modules (`features`, `embedder`, `scorer`, `cli`, `api`) | done |
| Smoke test (`test_version`, `test_stub_imports`) | done |
| Pinned dependencies (`requirements.txt`) | done |
| `pyproject.toml`, `.gitignore`, MIT license | done |
| Feature extraction, embeddings, scoring, CLI, API | M2 – M5 |

## Quickstart

```bash
pip install -e .
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
│       ├── scorer.py        # pairwise similarity scoring (M4)
│       ├── cli.py           # typer CLI entry point (M4)
│       └── api.py           # fastapi REST + HTML UI (M5)
├── tests/
│   ├── __init__.py
│   └── test_smoke.py
├── data/
│   └── demo/               # 10 human vs GPT-4 paragraph pairs (M6)
├── requirements.txt
├── requirements-dev.txt
├── pyproject.toml
└── README.md
```

## Roadmap

- [x] **M1** — Repo scaffold, stub modules, smoke tests, pinned dependencies, MIT license
- [ ] **M2** — Implement 8+ stylometric features (TTR, sentence length, punctuation density, function-word frequency, Yule's K, burstiness)
- [ ] **M3** — Integrate `all-MiniLM-L6-v2` sentence-transformer embeddings (CPU, no API key)
- [ ] **M4** — Build weighted fusion scorer + Rich CLI table with top-3 diverging feature explanations
- [ ] **M5** — FastAPI `/compare` endpoint + minimal single-page HTML form
- [ ] **M6** — Bundle demo dataset (10 human vs GPT-4 pairs) and end-to-end integration tests

## License

MIT — see [LICENSE](LICENSE).
