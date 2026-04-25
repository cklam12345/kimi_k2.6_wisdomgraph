"""
Stage 2 — Subject-Predicate-Object Triple Extraction
Paper §3.1: Map 32B active parameters per token directly into SPO triples.
Target: 450 Billion triples at 14.5 GB (Table 1).

PROOF TASK:
  For each ExpertConvergenceNode, derive a symbolic SPO representation of
  the implicit knowledge encoded in that expert's weight subspace.
  Technique: activation-guided token perturbation + relation head probing.
"""
import json, pickle
from pathlib import Path
from dataclasses import dataclass
from typing import Iterator
import torch
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm


@dataclass
class SPOTriple:
    subject: str
    predicate: str
    object: str
    confidence: float
    source_layer: int
    source_expert: int
    domain: str


def relation_head_probe(
    model,
    tokenizer,
    subject_tokens: list[str],
    candidate_relations: list[str],
    layer_idx: int,
) -> list[SPOTriple]:
    """
    Probing protocol (Tenney et al. 2019 style):
      1. Embed subject in context.
      2. Extract hidden state at `layer_idx`.
      3. Classify into candidate relations via a lightweight linear probe.
      4. Generate object by greedy decoding conditioned on (subject, relation).

    PROOF: Train linear probe on LAMA/T-REx for each relation type,
    then evaluate on K2.6 hidden states. Confidence = probe softmax max.
    """
    triples: list[SPOTriple] = []
    # --- stub: replace with actual probe weights loaded from disk -----------
    for subj in subject_tokens:
        for rel in candidate_relations:
            prompt = f"{subj} {rel}"
            ids = tokenizer(prompt, return_tensors="pt").input_ids
            with torch.no_grad():
                out = model(ids, output_hidden_states=True)
            # take hidden state at target layer
            hidden = out.hidden_states[layer_idx][:, -1, :]   # (1, D)
            # TODO: apply trained probe to `hidden` → confidence, object_token_ids
            # Placeholder:
            confidence = float(torch.sigmoid(hidden.norm()).item())
            obj_ids = model.generate(ids, max_new_tokens=8, do_sample=False)
            obj_text = tokenizer.decode(obj_ids[0, ids.shape[1]:], skip_special_tokens=True)
            if confidence > 0.5:
                triples.append(SPOTriple(
                    subject=subj,
                    predicate=rel,
                    object=obj_text.strip(),
                    confidence=confidence,
                    source_layer=layer_idx,
                    source_expert=-1,
                    domain="unknown",
                ))
    return triples


def stream_triples(
    model, tokenizer,
    manifold_map_path: Path,
    batch_size: int = 512,
    confidence_threshold: float = 0.72,
) -> Iterator[list[SPOTriple]]:
    """Yield batches of SPO triples derived from the manifold map."""
    manifold = json.loads(manifold_map_path.read_text())
    candidate_relations = [
        "is_a", "part_of", "calls", "returns", "depends_on",
        "equivalent_to", "causes", "prevents", "defined_in",
    ]
    batch: list[SPOTriple] = []
    for node in tqdm(manifold["convergence_nodes"], desc="Extracting triples"):
        subjects = [node["task_domain"].replace("_", " ")]
        triples = relation_head_probe(
            model, tokenizer, subjects, candidate_relations, node["layer_idx"]
        )
        for t in triples:
            if t.confidence >= confidence_threshold:
                t.source_expert = node["expert_id"]
                t.domain = node["task_domain"]
                batch.append(t)
        if len(batch) >= batch_size:
            yield batch
            batch = []
    if batch:
        yield batch


def run(config_path: str = "../config/pipeline.yaml"):
    import yaml
    cfg = yaml.safe_load(Path(config_path).read_text())
    out_dir = Path("..") / cfg["stage2_spo"]["output_dir"]
    out_dir.mkdir(parents=True, exist_ok=True)

    from transformers import AutoModelForCausalLM, AutoTokenizer
    repo = cfg["model"]["kimi_k2"]["repo"]
    tokenizer = AutoTokenizer.from_pretrained(repo, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        repo, torch_dtype=torch.float16, device_map="auto",
        load_in_4bit=True, trust_remote_code=True,
    )
    model.eval()

    manifold_path = Path("..") / cfg["stage1_probe"]["output"]
    shard = 0
    total = 0
    for batch in stream_triples(
        model, tokenizer, manifold_path,
        batch_size=cfg["stage2_spo"]["batch_size"],
        confidence_threshold=cfg["stage2_spo"]["confidence_threshold"],
    ):
        shard_path = out_dir / f"shard_{shard:06d}.pkl"
        with open(shard_path, "wb") as f:
            pickle.dump(batch, f)
        total += len(batch)
        shard += 1

    print(f"Stage 2 complete. {total} triples across {shard} shards → {out_dir}")


if __name__ == "__main__":
    import sys
    run(sys.argv[1] if len(sys.argv) > 1 else "../config/pipeline.yaml")
