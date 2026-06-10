# AuthorLens


> **Video walkthrough:** https://youtu.be/wF3vxBjlhE8
> **60-second overview:** https://youtu.be/74dvp3JjHyU

> Explainable authorship fingerprinting: compare writing styles and detect AI-generated text using local sentence embeddings.

<!-- TODO: replace with a 5-10 second demo gif. Record with ScreenToGif on
     Windows or peek on macOS. Save to docs/demo.gif and update path here. -->
![demo](docs/demo.gif)

## What it is

AuthorLens computes a stylometric profile for each document — type-token ratio, sentence-length distribution, punctuation density, function-word frequency, Yule's K, and burstiness — and fuses those features with a 384-dimensional embedding produced by `all-MiniLM-L6-v2` running entirely on CPU. The result is a single similarity score between 0 and 1, plus a ranked diff of the three stylometric features that diverge most, each annotated with a plain-English explanation.

The tool addresses a concrete gap: LLM-generated content is increasingly common, yet most authorship tools are either black-box classifiers or academic prototypes. AuthorLens is neither. It runs offline, requires no API key, and shows its reasoning at every step. The design is grounded in the 2025 paper *Explainable Disentangled Representation Learning for Generalizable Authorship Attribution in the Era of Generative AI*.

## Quickstart

```bash
git clone https://github.com/RitikPatill/authorlens.git
cd authorlens
pip install -r requirements.txt
pip install -e .

# Compare two documents
authorlens compare doc1.txt doc2.txt

# Scan every pair in a folder
authorlens scan ./data/demo/

# Run the bundled 10-pair human-vs-GPT-4 demo
python demo.py
```

Start the REST API:

```bash
uvicorn authorlens.api:app --reload
```

## Usage

**CLI** — `authorlens compare` prints a Rich terminal table with the combined score, the embedding cosine similarity, the stylometric similarity, a verdict ("likely same author" / "uncertain" / "likely different authors"), and the top-3 diverging features. `authorlens scan` does the same for every `.txt` pair in a directory and exits with code 1 if any pair exceeds the threshold.

Optional flags:

```bash
authorlens compare doc1.txt doc2.txt --emb-weight 0.6 --threshold 0.8
authorlens scan ./folder/ --threshold 0.7
```

**REST API** — `POST /compare` accepts `{"texts": ["...", "..."]}` and returns a JSON object containing `overall`, `embedding_sim`, `stylometric_sim`, `verdict`, and a `feature_deltas` list. A single-page HTML form is served at `/` for browser-based testing. Health check at `GET /health`.

```bash
curl -s -X POST http://localhost:8000/compare \
  -H "Content-Type: application/json" \
  -d '{"texts": ["I walked to the shop, tired.", "The individual proceeded to the retail establishment."]}' \
  | python -m json.tool
```

**Demo dataset** — `python demo.py` runs all 10 bundled human-vs-GPT-4 paragraph pairs and prints an accuracy summary. On the default weights, the majority of pairs resolve to "likely different authors".

## Architecture

```
┌─────────────┐     ┌──────────────────┐
│  Text A/B   │────▶│  features.py     │──────┐
└─────────────┘     │  (8 stylometrics)│      │   ┌────────────────┐
                    └──────────────────┘      ├──▶│  scorer.py     │──▶ score + diff
┌─────────────┐     ┌──────────────────┐      │   │ (weighted avg) │
│  Text A/B   │────▶│  embedder.py     │──────┘   └────────────────┘
└─────────────┘     │  (MiniLM 384-d)  │
                    └──────────────────┘

        ┌──────────┐          ┌──────────────┐
        │  cli.py  │          │   api.py     │
        │  (typer) │          │  (fastapi)   │
        └──────────┘          └──────────────┘
```

## Project structure

```
authorlens/
├── src/authorlens/
│   ├── __init__.py       # package version (0.1.0)
│   ├── features.py       # 8 stylometric features
│   ├── embedder.py       # sentence-transformer encode + cosine similarity
│   ├── scorer.py         # weighted fusion, AuthorshipScore dataclass
│   ├── cli.py            # typer CLI (compare, scan)
│   ├── api.py            # FastAPI REST + CORS + health
│   └── templates/
│       └── index.html    # single-page HTML form
├── tests/                # 30+ unit + integration tests
├── data/demo/            # 10 human-vs-GPT-4 paragraph pairs + labels.json
├── demo.py               # end-to-end demo script
├── pyproject.toml
├── requirements.txt
└── requirements-dev.txt
```

## Roadmap

- [ ] Support multi-document authorship clustering (more than two texts at once)
- [ ] Add a confidence interval to each feature delta using bootstrap resampling
- [ ] Expose a `--format json` flag on the CLI for machine-readable output
- [ ] Package and publish to PyPI for single-command install
- [ ] Extend the demo dataset with non-English pairs to measure feature portability

## License

MIT — see [LICENSE](LICENSE).

---

Built autonomously by [autodev](https://github.com/RitikPatill/autodev),
a multi-agent orchestrator I designed. Each commit in this repo was
authored by me; the implementation work was performed by Sonnet under
the orchestrator's control. Read the orchestrator's README to see how.
