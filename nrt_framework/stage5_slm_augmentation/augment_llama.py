"""
Stage 5 — SLM Augmentation via Wisdom Graph Retrieval-Augmented Generation
Paper §4.2: NRT-augmented Llama 3.2 1B achieves 92.4% parity on HLE at 24 tok/s.

PROOF TASK:
  1. At inference time, embed the user query.
  2. Retrieve top-K Wisdom Graph triples via cosine search on the graph embeddings.
  3. Prepend the retrieved context (structured as SPO) to the SLM prompt.
  4. Run Llama 3.2 1B (INT4, Core ML / GGUF for mobile) and measure tok/s.

  Target metrics (Table 2):
    - HLE Score: 49.9% (92.4% of Kimi K2.6's 54.0%)
    - Inference speed: 24 tok/s on iPhone 15 Pro (Apple Neural Engine)
    - TTFT: 180 ms
"""
import json, time
from pathlib import Path
from dataclasses import dataclass

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from sentence_transformers import SentenceTransformer
import numpy as np


@dataclass
class AugmentedResponse:
    query: str
    retrieved_triples: list[dict]
    prompt: str
    response: str
    tokens_generated: int
    elapsed_s: float
    tps: float
    ttft_ms: float


def load_wisdom_graph_index(graph_path: Path) -> tuple[list[dict], np.ndarray]:
    """Load wisdom graph triples and pre-computed embeddings for fast retrieval."""
    triples_file = graph_path / "wisdom_graph_core.jsonl"
    embeddings_file = graph_path / "embeddings.npy"

    triples = [json.loads(l) for l in triples_file.read_text().splitlines() if l.strip()]

    if embeddings_file.exists():
        embeddings = np.load(str(embeddings_file))
    else:
        # Build embeddings on first run
        embedder = SentenceTransformer("all-MiniLM-L6-v2")
        texts = [f"{t['subj']} {t['pred']} {t['obj']}" for t in triples]
        embeddings = embedder.encode(texts, batch_size=512, show_progress_bar=True,
                                     normalize_embeddings=True)
        np.save(str(embeddings_file), embeddings)

    return triples, embeddings


def retrieve_top_k(
    query: str,
    triples: list[dict],
    embeddings: np.ndarray,
    embedder: SentenceTransformer,
    k: int = 10,
) -> list[dict]:
    """Cosine-similarity retrieval from Wisdom Graph."""
    q_emb = embedder.encode([query], normalize_embeddings=True)
    scores = (embeddings @ q_emb.T).squeeze()
    top_k_idx = scores.argsort()[-k:][::-1]
    return [triples[i] for i in top_k_idx]


def build_augmented_prompt(query: str, retrieved: list[dict]) -> str:
    """
    Construct RAG prompt: wisdom graph context + query.
    Format mirrors the paper's §2 description of MCP memory injection.
    """
    context_lines = []
    for t in retrieved:
        dikw = t.get("dikw", "Knowledge")
        context_lines.append(
            f"[{dikw}] {t['subj']} --{t['pred']}--> {t['obj']} (conf={t.get('conf',0):.2f})"
        )
    context_block = "\n".join(context_lines)
    return (
        f"<wisdom_graph>\n{context_block}\n</wisdom_graph>\n\n"
        f"Using the above structured knowledge, answer precisely:\n{query}"
    )


def run_augmented_inference(
    query: str,
    model,
    tokenizer,
    triples: list[dict],
    embeddings: np.ndarray,
    embedder: SentenceTransformer,
    max_new_tokens: int = 256,
) -> AugmentedResponse:
    """Full NRT-augmented inference pipeline."""
    retrieved = retrieve_top_k(query, triples, embeddings, embedder)
    prompt = build_augmented_prompt(query, retrieved)

    input_ids = tokenizer(prompt, return_tensors="pt").input_ids

    t0 = time.perf_counter()
    with torch.no_grad():
        out = model.generate(
            input_ids,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
    t1 = time.perf_counter()

    # TTFT approximation: first-token latency (model-dependent hook needed for precise measure)
    ttft_ms = (t1 - t0) * 1000 / max(out.shape[1] - input_ids.shape[1], 1) * 1

    new_tokens = out.shape[1] - input_ids.shape[1]
    elapsed = t1 - t0
    tps = new_tokens / max(elapsed, 1e-9)

    response_text = tokenizer.decode(out[0, input_ids.shape[1]:], skip_special_tokens=True)
    return AugmentedResponse(
        query=query,
        retrieved_triples=retrieved,
        prompt=prompt,
        response=response_text,
        tokens_generated=new_tokens,
        elapsed_s=elapsed,
        tps=tps,
        ttft_ms=ttft_ms,
    )


def run(config_path: str = "../config/pipeline.yaml"):
    import yaml
    cfg = yaml.safe_load(Path(config_path).read_text())
    graph_path = Path("..") / cfg["stage6_mcp"]["graph_path"]

    repo = cfg["stage5_slm"]["base_model"]
    tokenizer = AutoTokenizer.from_pretrained(repo)
    model = AutoModelForCausalLM.from_pretrained(
        repo, torch_dtype=torch.float16, device_map="auto", load_in_4bit=True,
    )
    model.eval()

    embedder = SentenceTransformer("all-MiniLM-L6-v2")
    triples, embeddings = load_wisdom_graph_index(graph_path)

    # Smoke test with a sample HLE-style question
    sample_query = "Prove that for any n > 1, there exists a prime p such that n < p < 2n."
    resp = run_augmented_inference(sample_query, model, tokenizer, triples, embeddings, embedder)
    print(f"Query: {resp.query}")
    print(f"TPS: {resp.tps:.1f} tok/s  |  TTFT: {resp.ttft_ms:.0f} ms")
    print(f"Response:\n{resp.response}")


if __name__ == "__main__":
    import sys
    run(sys.argv[1] if len(sys.argv) > 1 else "../config/pipeline.yaml")
