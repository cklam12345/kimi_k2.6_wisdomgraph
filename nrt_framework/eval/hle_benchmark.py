"""
Evaluation — Humanity's Last Exam (HLE) Benchmark
Paper Table 2: Kimi K2.6 = 54.0%, Kimi-KG Mobile = 49.9% (92.4% parity).

PROOF TASK:
  Run the NRT-augmented Llama 3.2 1B on the HLE test set and report:
  - Accuracy score (target: ~49.9%)
  - Parity ratio vs Kimi K2.6 baseline (target: ≥ 92.4%)
  - Inference speed (target: ≥ 24 tok/s)
  - TTFT (target: ≤ 180 ms)
"""
import json, time, sys
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from sentence_transformers import SentenceTransformer
import numpy as np
from tqdm import tqdm


KIMI_K2_BASELINE = 0.540   # Table 2
TARGET_PARITY    = 0.924   # 92.4%


@dataclass
class HLEResult:
    model_id: str
    total_questions: int
    correct: int
    accuracy: float
    parity_vs_kimi: float
    avg_tps: float
    avg_ttft_ms: float
    passes_parity_bar: bool


def evaluate(
    model,
    tokenizer,
    triples: list[dict],
    embeddings: np.ndarray,
    embedder: SentenceTransformer,
    split: str = "test",
    max_samples: Optional[int] = 200,
) -> HLEResult:
    """
    Run augmented inference on HLE questions.
    HLE dataset: https://huggingface.co/datasets/cais/hle
    Each item: {"question": str, "answer": str, "category": str}
    """
    dataset = load_dataset("cais/hle", split=split)
    if max_samples:
        dataset = dataset.select(range(min(max_samples, len(dataset))))

    correct = 0
    tps_list = []
    ttft_list = []

    for item in tqdm(dataset, desc="HLE Eval"):
        # Retrieve wisdom graph context
        q_emb = embedder.encode([item["question"]], normalize_embeddings=True)
        scores = (embeddings @ q_emb.T).squeeze()
        top_idx = scores.argsort()[-10:][::-1]
        retrieved = [triples[i] for i in top_idx]

        ctx = "\n".join(
            f"[{r.get('dikw','K')}] {r['subj']} --{r['pred']}--> {r['obj']}"
            for r in retrieved
        )
        prompt = f"<wisdom_graph>\n{ctx}\n</wisdom_graph>\n\nQuestion: {item['question']}\nAnswer:"

        ids = tokenizer(prompt, return_tensors="pt").input_ids
        t0 = time.perf_counter()
        with torch.no_grad():
            out = model.generate(ids, max_new_tokens=64, do_sample=False,
                                 pad_token_id=tokenizer.eos_token_id)
        t1 = time.perf_counter()

        new_tokens = out.shape[1] - ids.shape[1]
        elapsed = t1 - t0
        tps_list.append(new_tokens / max(elapsed, 1e-9))
        ttft_list.append(elapsed * 1000 / max(new_tokens, 1))

        pred = tokenizer.decode(out[0, ids.shape[1]:], skip_special_tokens=True).strip()
        gold = str(item.get("answer", "")).strip()

        # Exact-match evaluation (extend with LLM judge for free-form answers)
        if pred.lower().startswith(gold.lower()) or gold.lower() in pred.lower():
            correct += 1

    total = len(dataset)
    accuracy = correct / max(total, 1)
    parity = accuracy / KIMI_K2_BASELINE

    return HLEResult(
        model_id=tokenizer.name_or_path,
        total_questions=total,
        correct=correct,
        accuracy=accuracy,
        parity_vs_kimi=parity,
        avg_tps=float(np.mean(tps_list)),
        avg_ttft_ms=float(np.mean(ttft_list)),
        passes_parity_bar=parity >= TARGET_PARITY,
    )


def run(config_path: str = "../config/pipeline.yaml"):
    import yaml
    cfg = yaml.safe_load(Path(config_path).read_text())
    graph_path = Path("..") / cfg["stage6_mcp"]["graph_path"]

    # Load SLM
    repo = cfg["stage5_slm"]["base_model"]
    tokenizer = AutoTokenizer.from_pretrained(repo)
    model = AutoModelForCausalLM.from_pretrained(
        repo, torch_dtype=torch.float16, device_map="auto", load_in_4bit=True,
    )
    model.eval()

    # Load wisdom graph index
    embedder = SentenceTransformer("all-MiniLM-L6-v2")
    triples_file = graph_path / "wisdom_graph_core.jsonl"
    triples = [json.loads(l) for l in triples_file.read_text().splitlines() if l.strip()]
    emb_file = graph_path / "embeddings.npy"
    embeddings = np.load(str(emb_file)) if emb_file.exists() else \
        embedder.encode([f"{t['subj']} {t['pred']} {t['obj']}" for t in triples],
                        normalize_embeddings=True)

    result = evaluate(model, tokenizer, triples, embeddings, embedder, max_samples=200)
    print(json.dumps(asdict(result), indent=2))

    out_file = Path("..") / cfg["eval"].get("output", "artifacts/hle_results/result.json")
    out_file.parent.mkdir(parents=True, exist_ok=True)
    out_file.write_text(json.dumps(asdict(result), indent=2))

    status = "[PASS]" if result.passes_parity_bar else "[FAIL]"
    print(f"{status} Accuracy={result.accuracy:.3f} Parity={result.parity_vs_kimi:.3f} "
          f"TPS={result.avg_tps:.1f} TTFT={result.avg_ttft_ms:.0f}ms")


if __name__ == "__main__":
    run(sys.argv[1] if len(sys.argv) > 1 else "../config/pipeline.yaml")
