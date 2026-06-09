"""FastAPI REST API with HTML demo page (M5)."""

from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel, Field

from authorlens import scorer

app = FastAPI(title="AuthorLens", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

templates = Jinja2Templates(directory=Path(__file__).parent / "templates")


# ---------- Pydantic models ----------

class CompareRequest(BaseModel):
    texts: list[str] = Field(..., min_length=2, max_length=2)
    emb_weight: float = Field(0.5, ge=0.0, le=1.0)


class FeatureBreakdown(BaseModel):
    ttr_dist: float
    mean_sent_len_dist: float
    std_sent_len_dist: float
    punct_density_dist: float
    yule_k_dist: float
    burstiness_dist: float
    func_word_cosine_dist: float


class CompareResponse(BaseModel):
    overall_score: float
    embedding_sim: float
    stylometric_sim: float
    feature_breakdown: FeatureBreakdown
    verdict: str


# ---------- Endpoints ----------

@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/", response_class=HTMLResponse)
def root(request: Request):
    return templates.TemplateResponse(request, "index.html")


@app.post("/compare", response_model=CompareResponse)
def compare(payload: CompareRequest):
    result = scorer.fuse(payload.texts[0], payload.texts[1], payload.emb_weight)
    return CompareResponse(
        overall_score=result.overall,
        embedding_sim=result.embedding_sim,
        stylometric_sim=result.stylometric_sim,
        feature_breakdown=FeatureBreakdown(**result.feature_deltas),
        verdict=scorer.verdict(result.overall),
    )
