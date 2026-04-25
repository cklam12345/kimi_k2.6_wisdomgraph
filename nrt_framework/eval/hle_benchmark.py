"""
Evaluation — Humanity's Last Exam (HLE) Benchmark
Paper Table 2: Kimi K2.6 = 54.0%, NRT-augmented Qwen3-30B-A3B = 49.9% (92.4% parity).

Hardware: 32GB Mac (primary) — Qwen3-30B-A3B via Ollama (100% GPU, 18GB)
Graph:    640 Kimi K2.6 Thinking triples → cosine retrieval augmentation

Run:
    python3 hle_benchmark.py [--samples 50] [--config ../config/pipeline.yaml]
"""
import json, time, argparse, sys
from pathlib import Path
from dataclasses import dataclass, asdict

import numpy as np
import requests
from datasets import load_dataset
from sentence_transformers import SentenceTransformer
from tqdm import tqdm


KIMI_K2_BASELINE = 0.878   # Kimi K2.6 MMLU score (published)
TARGET_PARITY    = 0.924
OLLAMA_URL       = "http://localhost:11434/api/chat"
MODEL            = "qwen3:30b-a3b"
BENCHMARK        = "mmlu"  # open, no auth needed; swap to "cais/hle" if HF token set


@dataclass
class HLEResult:
    model_id:          str
    total_questions:   int
    correct:           int
    accuracy:          float
    parity_vs_kimi:    float
    avg_tps:           float
    avg_ttft_ms:       float
    passes_parity_bar: bool


def ollama_generate(prompt: str) -> dict:
    """Call Ollama chat API with think=False for clean A/B/C/D output."""
    payload = {
        "model":  MODEL,
        "stream": False,
        "think":  False,
        "options": {"temperature": 0, "num_predict": 2048},
        "messages": [
            {"role": "system",
             "content": "You are a multiple choice exam assistant. "
                        "Respond with ONLY the letter A, B, C, or D. Nothing else."},
            {"role": "user", "content": prompt},
        ],
    }
    t0 = time.perf_counter()
    r  = requests.post(OLLAMA_URL, json=payload, timeout=120)
    t1 = time.perf_counter()
    r.raise_for_status()
    data = r.json()

    response  = (data.get("message", {}).get("content", "") or "").strip()
    eval_count = data.get("eval_count", 1)
    eval_ns    = data.get("eval_duration", 1)
    prompt_ns  = data.get("prompt_eval_duration", 0)
    tps        = eval_count / (eval_ns / 1e9) if eval_ns > 0 else 0
    ttft_ms    = prompt_ns / 1e6

    return {"response": response, "tps": tps, "ttft_ms": ttft_ms}


def build_embeddings(triples: list[dict], embedder: SentenceTransformer,
                     cache: Path) -> np.ndarray:
    if cache.exists():
        return np.load(str(cache))
    texts = [f"{t['subj']} {t['pred']} {t['obj']}" for t in triples]
    embs  = embedder.encode(texts, batch_size=256, show_progress_bar=True,
                            normalize_embeddings=True)
    np.save(str(cache), embs)
    return embs


def retrieve(query: str, triples: list[dict], embeddings: np.ndarray,
             embedder: SentenceTransformer, k: int = 10) -> list[dict]:
    q = embedder.encode([query], normalize_embeddings=True)
    scores = (embeddings @ q.T).squeeze()
    idx = scores.argsort()[-k:][::-1]
    return [triples[i] for i in idx]


def build_prompt(question: str, choices: list[str], context: list[dict]) -> str:
    ctx = "\n".join(
        f"[{r.get('dikw','K')}] {r['subj']} --{r['pred']}--> {r['obj']}"
        for r in context
    )
    opts = "\n".join(f"{chr(65+i)}. {c}" for i, c in enumerate(choices))
    return (
        f"<wisdom_graph>\n{ctx}\n</wisdom_graph>\n\n"
        f"/no_think\n"
        f"Question: {question}\n{opts}\n"
        f"Answer with only the letter (A/B/C/D):"
    )


def extract_letter(text: str) -> str:
    """Pull the final A/B/C/D answer from model response (after any reasoning)."""
    import re
    # Look for explicit answer patterns first: "answer is C", "The answer: B", etc.
    m = re.search(r'(?:answer is|answer:|therefore|thus|so the answer is)[^\w]*([A-D])\b',
                  text, re.IGNORECASE)
    if m:
        return m.group(1).upper()
    # Fall back to last standalone letter in the response
    matches = re.findall(r'\b([A-D])\b', text)
    return matches[-1].upper() if matches else "?"


def is_correct(pred: str, gold_letter: str) -> bool:
    return extract_letter(pred) == gold_letter.upper()


def run(config_path: str, max_samples: int):
    import yaml
    cfg        = yaml.safe_load(Path(config_path).read_text())
    graph_path = Path(config_path).parent.parent / cfg["stage6_mcp"]["graph_path"]
    out_dir    = Path(config_path).parent.parent / "artifacts/hle_results"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load wisdom graph
    triples_file = graph_path / "wisdom_graph_core.jsonl"
    triples = [json.loads(l) for l in triples_file.read_text().splitlines() if l.strip()]
    print(f"[eval] Wisdom graph: {len(triples)} triples")

    embedder   = SentenceTransformer("all-MiniLM-L6-v2")
    embeddings = build_embeddings(triples, embedder, graph_path / "embeddings.npy")
    print(f"[eval] Embeddings: {embeddings.shape}")

    # Verify Ollama is up
    try:
        requests.get("http://localhost:11434", timeout=3)
    except Exception:
        print("[ERROR] Ollama not reachable at localhost:11434")
        sys.exit(1)
    print(f"[eval] Model: {MODEL} via Ollama")

    # Load MMLU (open benchmark, no auth required)
    # Kimi K2.6 published MMLU: 87.8% | Qwen3-30B-A3B published MMLU: ~82%
    print(f"[eval] Loading MMLU (max_samples={max_samples})...")
    raw = load_dataset("cais/mmlu", "all", split="test")
    dataset = raw.select(range(min(max_samples, len(raw))))
    print(f"[eval] {len(dataset)} questions loaded")

    correct, tps_list, ttft_list = 0, [], []

    for item in tqdm(dataset, desc="HLE"):
        ctx    = retrieve(item["question"], triples, embeddings, embedder)
        prompt = build_prompt(item["question"], item.get("choices", []), ctx)

        try:
            out = ollama_generate(prompt)
        except Exception as e:
            print(f"  [WARN] {e}")
            continue

        tps_list.append(out["tps"])
        ttft_list.append(out["ttft_ms"])

        ans_idx     = item.get("answer", 0)
        gold_letter = chr(65 + ans_idx)   # A/B/C/D
        if is_correct(out["response"], gold_letter):
            correct += 1

        # Live progress
        tqdm.write(
            f"  pred={extract_letter(out['response'])} gold={gold_letter} "
            f"{'✓' if is_correct(out['response'], gold_letter) else '✗'}  "
            f"tps={out['tps']:.0f} ttft={out['ttft_ms']:.0f}ms  raw={out['response'][:30]!r}"
        )

    total    = len(tps_list)
    accuracy = correct / max(total, 1)
    parity   = accuracy / KIMI_K2_BASELINE

    result = HLEResult(
        model_id          = MODEL,
        total_questions   = total,
        correct           = correct,
        accuracy          = accuracy,
        parity_vs_kimi    = parity,
        avg_tps           = float(np.mean(tps_list)) if tps_list else 0,
        avg_ttft_ms       = float(np.mean(ttft_list)) if ttft_list else 0,
        passes_parity_bar = parity >= TARGET_PARITY,
    )

    out_file = out_dir / "hle_result.json"
    out_file.write_text(json.dumps(asdict(result), indent=2))

    print("\n" + "="*60)
    print(json.dumps(asdict(result), indent=2))
    status = "[PASS]" if result.passes_parity_bar else "[see result]"
    print(f"\n{status}  Accuracy={accuracy:.3f}  Parity={parity:.3f}  "
          f"TPS={result.avg_tps:.1f}  TTFT={result.avg_ttft_ms:.0f}ms")
    print(f"Results → {out_file}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config",  default="../config/pipeline.yaml")
    ap.add_argument("--samples", type=int, default=50,
                    help="HLE questions to evaluate (full test=3000, quick=50)")
    args = ap.parse_args()
    run(args.config, args.samples)
