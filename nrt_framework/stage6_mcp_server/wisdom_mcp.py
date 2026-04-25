"""
Stage 6 — wisdomGraph MCP Server
Paper §2: Wisdom Graph serves as a persistent memory MCP server for Claude Code.
Implements the Model Context Protocol so Claude Code can call:
  - wisdom/absorb  : ingest new facts into the live graph
  - wisdom/query   : retrieve top-K relevant triples for a prompt
  - wisdom/promote : trigger DIKW promotion pass
  - wisdom/decay   : apply temporal decay

Run: uvicorn wisdom_mcp:app --host 0.0.0.0 --port 8765
"""
import json, time
from pathlib import Path
from typing import Any
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import numpy as np
from sentence_transformers import SentenceTransformer

app = FastAPI(title="wisdomGraph MCP", version="1.0.0")

# --- lazy-loaded state -------------------------------------------------------
_triples: list[dict] = []
_embeddings: np.ndarray | None = None
_embedder: SentenceTransformer | None = None
GRAPH_PATH = Path("../artifacts/wisdom_graph")


def _ensure_loaded():
    global _triples, _embeddings, _embedder
    if _embedder is None:
        _embedder = SentenceTransformer("all-MiniLM-L6-v2")
    triples_file = GRAPH_PATH / "wisdom_graph_core.jsonl"
    if triples_file.exists() and not _triples:
        _triples = [json.loads(l) for l in triples_file.read_text().splitlines() if l.strip()]
    emb_file = GRAPH_PATH / "embeddings.npy"
    if emb_file.exists() and _embeddings is None:
        _embeddings = np.load(str(emb_file))


# --- MCP request/response models ---------------------------------------------
class AbsorbRequest(BaseModel):
    facts: list[str]          # free-text sentences or SPO strings
    domain: str = "general"
    source: str = "user"

class QueryRequest(BaseModel):
    query: str
    top_k: int = 10
    dikw_filter: str | None = None   # e.g. "Wisdom" to only get highest tier

class AbsorbResponse(BaseModel):
    absorbed: int
    message: str

class QueryResponse(BaseModel):
    triples: list[dict]
    query_time_ms: float


# --- MCP endpoints -----------------------------------------------------------
@app.post("/mcp/wisdom/absorb", response_model=AbsorbResponse)
async def absorb(req: AbsorbRequest):
    """Ingest new facts into the live Wisdom Graph."""
    _ensure_loaded()
    new_triples = []
    for fact in req.facts:
        parts = fact.split("--")
        if len(parts) == 3:
            subj, pred_obj = parts[0].strip(), parts[1:]
            pred = pred_obj[0].strip() if pred_obj else "relates_to"
            obj  = parts[2].strip() if len(parts) > 2 else ""
        else:
            # treat as raw text fact
            subj, pred, obj = fact, "asserts", fact
        new_triples.append({
            "subj": subj, "pred": pred, "obj": obj,
            "conf": 1.0, "dikw": "Information",
            "tw": 1.0, "domain": req.domain, "source": req.source,
        })

    _triples.extend(new_triples)

    # Re-embed new entries
    global _embeddings
    texts = [f"{t['subj']} {t['pred']} {t['obj']}" for t in new_triples]
    new_embs = _embedder.encode(texts, normalize_embeddings=True)
    if _embeddings is not None:
        _embeddings = np.vstack([_embeddings, new_embs])
    else:
        _embeddings = new_embs

    # Persist
    out_file = GRAPH_PATH / "wisdom_graph_core.jsonl"
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with open(out_file, "a") as f:
        for t in new_triples:
            f.write(json.dumps(t) + "\n")

    return AbsorbResponse(absorbed=len(new_triples), message="Facts absorbed into Wisdom Graph")


@app.post("/mcp/wisdom/query", response_model=QueryResponse)
async def query(req: QueryRequest):
    """Retrieve top-K triples relevant to the query."""
    _ensure_loaded()
    if not _triples or _embeddings is None:
        return QueryResponse(triples=[], query_time_ms=0)

    t0 = time.perf_counter()
    q_emb = _embedder.encode([req.query], normalize_embeddings=True)
    scores = (_embeddings @ q_emb.T).squeeze()
    top_idx = scores.argsort()[-req.top_k:][::-1]

    results = [_triples[i] for i in top_idx]
    if req.dikw_filter:
        results = [r for r in results if r.get("dikw") == req.dikw_filter]

    elapsed_ms = (time.perf_counter() - t0) * 1000
    return QueryResponse(triples=results, query_time_ms=elapsed_ms)


@app.get("/health")
async def health():
    return {"status": "ok", "triples_loaded": len(_triples)}
