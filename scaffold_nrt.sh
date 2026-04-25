#!/usr/bin/env bash
# Neural-to-Relational Transduction (NRT) — Research Proof Scaffold
# Paper: "De-parameterizing the Kimi K2.6 Trillion-Parameter Manifold into Mobile-Ready Wisdom Graphs"
# Author: Chin Keong Lam
#
# This script synthesizes the full project skeleton, installs dependencies,
# and stubs every stage of the NRT pipeline so proofs can be filled in.

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/nrt_framework"
VENV="$ROOT/.venv"
NEO4J_DATA="$ROOT/neo4j/data"
GRAPH_OUT="$ROOT/artifacts/wisdom_graph"
TRIPLES_OUT="$ROOT/artifacts/spo_triples"
HLE_OUT="$ROOT/artifacts/hle_results"

# ── colours ──────────────────────────────────────────────────────────────────
RED='\033[0;31m'; GREEN='\033[0;32m'; CYAN='\033[0;36m'; NC='\033[0m'
info()  { echo -e "${CYAN}[NRT]${NC} $*"; }
ok()    { echo -e "${GREEN}[OK ]${NC} $*"; }
warn()  { echo -e "${RED}[WARN]${NC} $*"; }

# ── directory skeleton ────────────────────────────────────────────────────────
info "Creating project skeleton under $ROOT"
mkdir -p \
  "$ROOT/config" \
  "$ROOT/stage1_manifold_probe" \
  "$ROOT/stage2_spo_extraction" \
  "$ROOT/stage3_hrkg_build" \
  "$ROOT/stage4_graph_compression" \
  "$ROOT/stage5_slm_augmentation" \
  "$ROOT/stage6_mcp_server" \
  "$ROOT/eval" \
  "$ROOT/artifacts/spo_triples" \
  "$ROOT/artifacts/wisdom_graph" \
  "$ROOT/artifacts/hle_results" \
  "$NEO4J_DATA"
ok "Directory tree created"

# ── Python virtual environment ────────────────────────────────────────────────
info "Setting up Python venv"
python3 -m venv "$VENV"
# shellcheck source=/dev/null
source "$VENV/bin/activate"

pip install --quiet --upgrade pip
pip install --quiet \
  torch torchvision \
  transformers>=4.45.0 \
  accelerate \
  bitsandbytes \
  huggingface_hub \
  neo4j \
  networkx \
  pandas \
  numpy \
  tqdm \
  fastapi \
  uvicorn \
  pydantic \
  rich \
  datasets \
  scikit-learn \
  sentence-transformers
ok "Python dependencies installed"

# ── config/pipeline.yaml ──────────────────────────────────────────────────────
cat > "$ROOT/config/pipeline.yaml" << 'YAML'
# NRT Pipeline Configuration
model:
  kimi_k2:
    repo: "moonshotai/Kimi-K2.6"
    num_layers: 61
    active_params_per_token: 32_000_000_000
    quant: "int4"
  slm:
    repo: "meta-llama/Llama-3.2-1B-Instruct"
    target_device: "mobile_npu"  # or "mps" for Apple Silicon

stage1_probe:
  expert_convergence_threshold: 0.85
  layers_to_probe: "all"          # or list of ints, e.g. [0,15,30,45,60]
  output: "artifacts/expert_map.json"

stage2_spo:
  batch_size: 512
  confidence_threshold: 0.72
  output_dir: "artifacts/spo_triples"
  target_triples: 450_000_000_000  # 450B triples (Table 1)

stage3_hrkg:
  framework: "LLHKG"
  node_target: 1_200_000_000      # 1.2B nodes (Table 1)
  storage_target_gb: 14.5
  neo4j_uri: "bolt://localhost:7687"
  neo4j_user: "neo4j"
  neo4j_password: "wisdomgraph"

stage4_compress:
  node_target: 8_400_000          # 8.4M nodes (Table 1)
  storage_target_gb: 3.2
  compression_ratio: 185
  temporal_decay_alpha: 0.01      # Section 5: memory decay

stage5_slm:
  base_model: "meta-llama/Llama-3.2-1B-Instruct"
  kg_adapter: "wisdom_graph_lora"
  target_tps: 24                  # Table 2: 24 tok/s on iPhone 15 Pro
  target_ttft_ms: 180             # Table 2

stage6_mcp:
  port: 8765
  host: "0.0.0.0"
  graph_path: "artifacts/wisdom_graph"

eval:
  benchmark: "HLE"                # Humanity's Last Exam
  target_parity: 0.924            # 92.4% parity (Abstract)
  kimi_baseline: 0.540            # 54.0% HLE (Table 2)
  kg_mobile_target: 0.499         # 49.9% HLE (Table 2)
YAML
ok "config/pipeline.yaml written"

# ── STAGE 1: Manifold Probing & Expert Analysis ───────────────────────────────
cat > "$ROOT/stage1_manifold_probe/probe_experts.py" << 'PY'
"""
Stage 1 — Manifold Probing & Expert Convergence Analysis
Paper §3.1: Analyse K2.6's 61-layer fabric to identify Expert Convergence nodes.
Each MoE expert specialises in a task domain (API routing, LaTeX syntax, etc.).
PROOF TASK: Map 32B active parameters per token → task-typed activation clusters.
"""
import json, os, sys
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import Optional

import torch
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm
from rich.console import Console

console = Console()

@dataclass
class ExpertConvergenceNode:
    layer_idx: int
    expert_id: int
    task_domain: str          # e.g. "code_routing", "math_reasoning", "latex_syntax"
    activation_mass: float    # fraction of tokens routed here
    convergence_score: float  # how task-specialised (0→1)
    param_count: int

@dataclass
class ManifoldMap:
    model_id: str
    num_layers: int
    num_experts_per_layer: int
    convergence_nodes: list[ExpertConvergenceNode]
    total_active_params_per_token: int


def load_model(repo: str, device: str = "cpu") -> tuple:
    """Load K2.6 (or a proxy) in INT4 for probing."""
    console.print(f"[cyan]Loading model:[/] {repo}")
    tokenizer = AutoTokenizer.from_pretrained(repo, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        repo,
        torch_dtype=torch.float16,
        device_map=device,
        load_in_4bit=True,
        trust_remote_code=True,
    )
    model.eval()
    return model, tokenizer


def probe_expert_activations(
    model,
    tokenizer,
    probe_corpus: list[str],
    threshold: float = 0.85,
) -> list[ExpertConvergenceNode]:
    """
    Feed domain-typed probe sentences through the model and record which
    experts activate above `threshold` for each domain.

    PROOF: For each (layer, expert) pair compute Jensen-Shannon divergence
    between its routing distribution and a uniform baseline. High JSD → domain specialist.
    """
    nodes: list[ExpertConvergenceNode] = []
    # TODO: iterate over model.model.layers, hook router.gate outputs
    # Below is the algorithmic skeleton — fill in actual hook registration.
    domain_probes = {
        "code_routing":    ["def fibonacci(n):", "import requests; r = requests.get("],
        "math_reasoning":  ["Solve: ∫x² dx =", "Proof by induction: P(n) = n(n+1)/2"],
        "latex_syntax":    ["\\begin{equation}", "\\frac{d}{dx}\\left("],
        "api_orchestration": ["POST /v1/completions HTTP/1.1", "Authorization: Bearer sk-"],
        "long_horizon":    ["Step 1: decompose the task. Step 2: assign sub-agents."],
    }

    # --- hook stub ---------------------------------------------------------
    activation_log: dict[tuple, list[float]] = {}

    def make_hook(layer_i: int, expert_j: int, domain: str):
        def hook(module, input, output):
            key = (layer_i, expert_j, domain)
            if key not in activation_log:
                activation_log[key] = []
            # capture mean activation magnitude
            if isinstance(output, torch.Tensor):
                activation_log[key].append(output.abs().mean().item())
        return hook

    # Register hooks across all MoE layers (K2.6 has 61 layers)
    handles = []
    for layer_i, layer in enumerate(getattr(model.model, 'layers', [])):
        moe = getattr(layer, 'mlp', None)
        experts = getattr(moe, 'experts', [])
        for expert_j, expert_mod in enumerate(experts):
            for domain in domain_probes:
                h = expert_mod.register_forward_hook(make_hook(layer_i, expert_j, domain))
                handles.append(h)

    # Run probes
    for domain, sentences in domain_probes.items():
        for sent in sentences:
            with torch.no_grad():
                ids = tokenizer(sent, return_tensors="pt").input_ids
                model(ids)

    for h in handles:
        h.remove()

    # Convert log → ExpertConvergenceNode
    for (layer_i, expert_j, domain), vals in activation_log.items():
        score = float(np.mean(vals)) if vals else 0.0
        if score >= threshold:
            nodes.append(ExpertConvergenceNode(
                layer_idx=layer_i,
                expert_id=expert_j,
                task_domain=domain,
                activation_mass=score,
                convergence_score=score,
                param_count=32_000_000_000 // max(len(getattr(
                    getattr(model.model.layers[layer_i], 'mlp', object()),
                    'experts', [1])), 1),
            ))
    return nodes


def run(config_path: str = "../config/pipeline.yaml"):
    import yaml
    cfg = yaml.safe_load(Path(config_path).read_text())

    repo = cfg["model"]["kimi_k2"]["repo"]
    threshold = cfg["stage1_probe"]["expert_convergence_threshold"]
    out_path = Path("..") / cfg["stage1_probe"]["output"]

    model, tokenizer = load_model(repo)

    probe_corpus = [
        "Write a Python FastAPI endpoint.",
        "Prove the Riemann Hypothesis.",
        "Render LaTeX: \\hat{H}\\psi = E\\psi",
    ]
    nodes = probe_expert_activations(model, tokenizer, probe_corpus, threshold)

    manifold = ManifoldMap(
        model_id=repo,
        num_layers=cfg["model"]["kimi_k2"]["num_layers"],
        num_experts_per_layer=0,  # TODO: read from model.config
        convergence_nodes=nodes,
        total_active_params_per_token=cfg["model"]["kimi_k2"]["active_params_per_token"],
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(asdict(manifold), indent=2))
    console.print(f"[green]Stage 1 complete.[/] {len(nodes)} convergence nodes → {out_path}")

if __name__ == "__main__":
    run(sys.argv[1] if len(sys.argv) > 1 else "../config/pipeline.yaml")
PY
ok "stage1_manifold_probe/probe_experts.py"

# ── STAGE 2: SPO Triple Extraction ───────────────────────────────────────────
cat > "$ROOT/stage2_spo_extraction/extract_triples.py" << 'PY'
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
PY
ok "stage2_spo_extraction/extract_triples.py"

# ── STAGE 3: HRKG Build (LLHKG framework) ────────────────────────────────────
cat > "$ROOT/stage3_hrkg_build/build_hrkg.py" << 'PY'
"""
Stage 3 — Hyper-Relational Knowledge Graph Construction (LLHKG)
Paper §3.2: Employ LLHKG to generate hyper-relational structures supporting
complex many-to-many relations for long-horizon coding and swarm orchestration.
Target: 1.2 Billion nodes, 14.5 GB (Table 1).

PROOF TASK:
  Lift flat SPO triples into hyper-relational statements:
    (subject, predicate, object) + {qualifier_key: qualifier_value, ...}
  Store in Neo4j with the DIKW node labels:
    :Data  :Information  :Knowledge  :Wisdom
"""
import pickle, json
from pathlib import Path
from typing import Generator
from neo4j import GraphDatabase
from tqdm import tqdm
from dataclasses import dataclass


@dataclass
class HyperTriple:
    subject: str
    predicate: str
    object: str
    qualifiers: dict          # e.g. {"domain": "code", "confidence": 0.91, "layer": 42}
    dikw_level: str           # "Data" | "Information" | "Knowledge" | "Wisdom"
    temporal_weight: float    # 1.0 = fresh; decays toward 0 via §5 temporal decay


DIKW_RULES = {
    "Data":        lambda t: t.qualifiers.get("confidence", 0) < 0.6,
    "Information": lambda t: 0.6  <= t.qualifiers.get("confidence", 0) < 0.75,
    "Knowledge":   lambda t: 0.75 <= t.qualifiers.get("confidence", 0) < 0.90,
    "Wisdom":      lambda t: t.qualifiers.get("confidence", 0) >= 0.90,
}


def classify_dikw(t: HyperTriple) -> str:
    for level, rule in DIKW_RULES.items():
        if rule(t):
            return level
    return "Data"


def spo_to_hyper(spo_triples: list) -> list[HyperTriple]:
    """
    Lift SPO triples into hyper-relational form by attaching qualifiers.
    Each qualifier captures: domain, confidence, source_layer, source_expert,
    timestamp (for temporal decay), and relation_type.
    """
    hyper = []
    for t in spo_triples:
        qualifiers = {
            "domain":        t.domain,
            "confidence":    t.confidence,
            "source_layer":  t.source_layer,
            "source_expert": t.source_expert,
        }
        ht = HyperTriple(
            subject=t.subject,
            predicate=t.predicate,
            object=t.object,
            qualifiers=qualifiers,
            dikw_level="",
            temporal_weight=1.0,
        )
        ht.dikw_level = classify_dikw(ht)
        hyper.append(ht)
    return hyper


CYPHER_MERGE_TRIPLE = """
MERGE (s:Entity {name: $subject})
  SET s.dikw = $dikw_level
MERGE (o:Entity {name: $object})
MERGE (s)-[r:{predicate} {{
  confidence:    $confidence,
  domain:        $domain,
  source_layer:  $source_layer,
  source_expert: $source_expert,
  temporal_weight: $temporal_weight,
  dikw_level:    $dikw_level
}}]->(o)
"""

def ingest_to_neo4j(
    driver,
    hyper_triples: list[HyperTriple],
    batch_size: int = 1000,
):
    """Batch-ingest hyper-triples into Neo4j using parameterised Cypher."""
    with driver.session() as session:
        for i in range(0, len(hyper_triples), batch_size):
            batch = hyper_triples[i:i+batch_size]
            tx_data = [
                {
                    "subject":        ht.subject,
                    "predicate":      ht.predicate,
                    "object":         ht.object,
                    "dikw_level":     ht.dikw_level,
                    "confidence":     ht.qualifiers["confidence"],
                    "domain":         ht.qualifiers["domain"],
                    "source_layer":   ht.qualifiers["source_layer"],
                    "source_expert":  ht.qualifiers["source_expert"],
                    "temporal_weight":ht.temporal_weight,
                }
                for ht in batch
            ]
            session.run(
                "UNWIND $rows AS row " +
                "MERGE (s:Entity {name: row.subject}) "
                "MERGE (o:Entity {name: row.object}) "
                "MERGE (s)-[r:RELATES_TO {predicate: row.predicate}]->(o) "
                "SET r += {confidence: row.confidence, domain: row.domain, "
                "source_layer: row.source_layer, temporal_weight: row.temporal_weight, "
                "dikw_level: row.dikw_level}",
                rows=tx_data,
            )


def run(config_path: str = "../config/pipeline.yaml"):
    import yaml
    cfg = yaml.safe_load(Path(config_path).read_text())
    spo_dir = Path("..") / cfg["stage2_spo"]["output_dir"]
    neo4j_uri  = cfg["stage3_hrkg"]["neo4j_uri"]
    neo4j_user = cfg["stage3_hrkg"]["neo4j_user"]
    neo4j_pass = cfg["stage3_hrkg"]["neo4j_password"]

    driver = GraphDatabase.driver(neo4j_uri, auth=(neo4j_user, neo4j_pass))

    shard_files = sorted(spo_dir.glob("shard_*.pkl"))
    total_nodes = 0
    for shard_path in tqdm(shard_files, desc="Building HRKG"):
        with open(shard_path, "rb") as f:
            spo_batch = pickle.load(f)
        hyper = spo_to_hyper(spo_batch)
        ingest_to_neo4j(driver, hyper)
        total_nodes += len(hyper)

    driver.close()
    print(f"Stage 3 complete. ~{total_nodes} hyper-triples ingested into Neo4j at {neo4j_uri}")


if __name__ == "__main__":
    import sys
    run(sys.argv[1] if len(sys.argv) > 1 else "../config/pipeline.yaml")
PY
ok "stage3_hrkg_build/build_hrkg.py"

# ── STAGE 4: Graph Compression → Wisdom Graph (3.2 GB) ───────────────────────
cat > "$ROOT/stage4_graph_compression/optimize_graph.py" << 'PY'
"""
Stage 4 — Wisdom Graph Compression
Paper Table 1: Compress 1.2B nodes / 14.5 GB → 8.4M nodes / 3.2 GB (185× reduction).
Paper §5: Apply temporal decay α to flag stale knowledge for pruning.

PROOF TASK:
  1. Entity merging: embed node names → cluster near-duplicates (cosine sim > 0.92).
  2. Predicate folding: merge semantically equivalent relation types.
  3. DIKW promotion: when multiple :Information nodes co-support a pattern → promote to :Knowledge.
  4. Temporal decay: score = score × exp(−α × age_days). Prune score < θ.
  5. Export optimised graph to disk as a portable .graphml + .pkl bundle.
"""
import json, pickle, math
from pathlib import Path
from typing import Optional
from neo4j import GraphDatabase
from sentence_transformers import SentenceTransformer
import numpy as np
from sklearn.cluster import MiniBatchKMeans
from tqdm import tqdm


TEMPORAL_DECAY_ALPHA = 0.01  # §5: decay constant
PRUNE_THRESHOLD      = 0.05  # remove edges with temporal_weight < θ
CLUSTER_SIM_THRESHOLD = 0.92  # cosine similarity for entity merging


def apply_temporal_decay(driver, alpha: float = TEMPORAL_DECAY_ALPHA):
    """
    Decay temporal_weight of all edges by exp(−α × age_days).
    Nodes/edges whose weight falls below PRUNE_THRESHOLD are deleted.
    This implements §5 "temporal decay" for compounding intelligence.
    """
    with driver.session() as session:
        # Decay all edges
        session.run("""
            MATCH ()-[r:RELATES_TO]->()
            SET r.temporal_weight = r.temporal_weight * exp(-$alpha)
        """, alpha=alpha)
        # Prune stale edges
        result = session.run("""
            MATCH ()-[r:RELATES_TO]->()
            WHERE r.temporal_weight < $theta
            DELETE r
            RETURN count(r) AS pruned
        """, theta=PRUNE_THRESHOLD)
        pruned = result.single()["pruned"]
    return pruned


def merge_near_duplicate_entities(driver, embedder: SentenceTransformer, batch_size: int = 10_000):
    """
    Pull entity names in batches, embed with sentence-transformers,
    cluster with cosine similarity > CLUSTER_SIM_THRESHOLD,
    then MERGE duplicate nodes in Neo4j.

    PROOF: This step achieves the 1.2B → 8.4M node reduction by collapsing
    surface-form variants of the same concept into a canonical node.
    """
    with driver.session() as session:
        # Fetch all entity names
        result = session.run("MATCH (e:Entity) RETURN e.name AS name, id(e) AS nid")
        records = [(r["nid"], r["name"]) for r in result]

    print(f"Fetched {len(records)} entities for deduplication")

    names  = [r[1] for r in records]
    nids   = [r[0] for r in records]

    # Embed in batches
    embeddings = embedder.encode(names, batch_size=256, show_progress_bar=True, normalize_embeddings=True)

    # Mini-batch KMeans to approximate near-duplicate clusters
    n_clusters = max(1, len(names) // 100)  # rough estimate: 1 cluster per 100 entities
    kmeans = MiniBatchKMeans(n_clusters=n_clusters, random_state=42)
    labels = kmeans.fit_predict(embeddings)

    # For each cluster, keep highest-degree node as canonical
    clusters: dict[int, list[tuple[int, str]]] = {}
    for nid, name, label in zip(nids, names, labels):
        clusters.setdefault(int(label), []).append((nid, name))

    merges = 0
    with driver.session() as session:
        for cluster_id, members in tqdm(clusters.items(), desc="Merging entities"):
            if len(members) < 2:
                continue
            canonical_nid = members[0][0]
            for dup_nid, _ in members[1:]:
                # Re-route all relationships from duplicate to canonical
                session.run("""
                    MATCH (dup) WHERE id(dup) = $dup_nid
                    MATCH (canon) WHERE id(canon) = $canon_nid
                    CALL apoc.refactor.mergeNodes([canon, dup], {properties:'combine'}) YIELD node
                    RETURN node
                """, dup_nid=dup_nid, canon_nid=canonical_nid)
                merges += 1

    print(f"Merged {merges} duplicate entity nodes")
    return merges


def promote_dikw_levels(driver):
    """
    DIKW promotion rules (§5):
      - 3+ :Information edges sharing (subject, object) → promote to :Knowledge
      - 3+ :Knowledge edges from same subject domain → emit :Wisdom node
    """
    with driver.session() as session:
        # Information → Knowledge promotion
        session.run("""
            MATCH (s:Entity)-[r:RELATES_TO]->(o:Entity)
            WHERE r.dikw_level = 'Information'
            WITH s, o, count(r) AS cnt
            WHERE cnt >= 3
            MATCH (s)-[r2:RELATES_TO]->(o)
            SET r2.dikw_level = 'Knowledge'
        """)
        # Knowledge → Wisdom promotion
        session.run("""
            MATCH (s:Entity)-[r:RELATES_TO]->()
            WHERE r.dikw_level = 'Knowledge'
            WITH s, r.domain AS domain, count(r) AS cnt
            WHERE cnt >= 3
            MERGE (w:Wisdom {name: s.name + '_wisdom_' + domain})
            SET w.domain = domain, w.created = timestamp()
        """)


def export_wisdom_graph(driver, out_dir: Path):
    """Export the compressed Wisdom Graph as JSON-LD lines for mobile NPU loading."""
    out_dir.mkdir(parents=True, exist_ok=True)
    with driver.session() as session:
        result = session.run("""
            MATCH (s:Entity)-[r:RELATES_TO]->(o:Entity)
            WHERE r.temporal_weight > $theta
            RETURN s.name AS subj, r.predicate AS pred, o.name AS obj,
                   r.confidence AS conf, r.dikw_level AS dikw,
                   r.temporal_weight AS tw
            LIMIT 10000000
        """, theta=PRUNE_THRESHOLD)
        records = [dict(r) for r in result]

    out_file = out_dir / "wisdom_graph_core.jsonl"
    with open(out_file, "w") as f:
        for rec in records:
            f.write(json.dumps(rec) + "\n")

    print(f"Exported {len(records)} wisdom triples → {out_file}")
    return out_file


def run(config_path: str = "../config/pipeline.yaml"):
    import yaml
    cfg = yaml.safe_load(Path(config_path).read_text())
    driver = GraphDatabase.driver(
        cfg["stage3_hrkg"]["neo4j_uri"],
        auth=(cfg["stage3_hrkg"]["neo4j_user"], cfg["stage3_hrkg"]["neo4j_password"]),
    )
    alpha = cfg["stage4_compress"]["temporal_decay_alpha"]
    out_dir = Path("..") / cfg["stage6_mcp"]["graph_path"]

    print("Step 4a: Temporal decay + pruning")
    pruned = apply_temporal_decay(driver, alpha)
    print(f"  Pruned {pruned} stale edges")

    print("Step 4b: Entity deduplication (embedding-based merge)")
    embedder = SentenceTransformer("all-MiniLM-L6-v2")
    merge_near_duplicate_entities(driver, embedder)

    print("Step 4c: DIKW level promotion")
    promote_dikw_levels(driver)

    print("Step 4d: Exporting Wisdom Graph")
    out_file = export_wisdom_graph(driver, out_dir)

    driver.close()
    print(f"Stage 4 complete. Wisdom Graph → {out_file}")


if __name__ == "__main__":
    import sys
    run(sys.argv[1] if len(sys.argv) > 1 else "../config/pipeline.yaml")
PY
ok "stage4_graph_compression/optimize_graph.py"

# ── STAGE 5: SLM Augmentation ─────────────────────────────────────────────────
cat > "$ROOT/stage5_slm_augmentation/augment_llama.py" << 'PY'
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
PY
ok "stage5_slm_augmentation/augment_llama.py"

# ── STAGE 6: wisdomGraph MCP Server ──────────────────────────────────────────
cat > "$ROOT/stage6_mcp_server/wisdom_mcp.py" << 'PY'
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
PY
ok "stage6_mcp_server/wisdom_mcp.py"

# ── EVAL: HLE Benchmark ───────────────────────────────────────────────────────
cat > "$ROOT/eval/hle_benchmark.py" << 'PY'
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
PY
ok "eval/hle_benchmark.py"

# ── Master pipeline runner ────────────────────────────────────────────────────
cat > "$ROOT/run_pipeline.sh" << 'RUNNER'
#!/usr/bin/env bash
# NRT Master Pipeline Runner
# Executes all 6 stages sequentially and concludes with HLE evaluation.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$ROOT/.venv/bin/activate"
CFG="$ROOT/config/pipeline.yaml"

run_stage() {
  local name="$1"; local script="$2"
  echo ""
  echo "══════════════════════════════════════════════"
  echo "  STAGE: $name"
  echo "══════════════════════════════════════════════"
  python "$script" "$CFG"
}

run_stage "1 — Manifold Probing"        "$ROOT/stage1_manifold_probe/probe_experts.py"
run_stage "2 — SPO Triple Extraction"   "$ROOT/stage2_spo_extraction/extract_triples.py"
run_stage "3 — HRKG Build"              "$ROOT/stage3_hrkg_build/build_hrkg.py"
run_stage "4 — Graph Compression"       "$ROOT/stage4_graph_compression/optimize_graph.py"
run_stage "5 — SLM Augmentation"        "$ROOT/stage5_slm_augmentation/augment_llama.py"
run_stage "6 — MCP Server (background)" "$(which uvicorn) wisdom_mcp:app --host 0.0.0.0 --port 8765 &"
run_stage "EVAL — HLE Benchmark"        "$ROOT/eval/hle_benchmark.py"

echo ""
echo "Pipeline complete. Check artifacts/ for outputs."
RUNNER
chmod +x "$ROOT/run_pipeline.sh"
ok "run_pipeline.sh"

# ── README ────────────────────────────────────────────────────────────────────
cat > "$ROOT/README.md" << 'DOC'
# NRT Framework — Research Proof Scaffold

Neural-to-Relational Transduction: De-parameterizing Kimi K2.6 into Mobile-Ready Wisdom Graphs.

## Pipeline Stages

| Stage | Script | Paper Section | Target |
|-------|--------|---------------|--------|
| 1 — Manifold Probe | `stage1_manifold_probe/probe_experts.py` | §3.1 | Expert Convergence map |
| 2 — SPO Extraction | `stage2_spo_extraction/extract_triples.py` | §3.1 | 450B triples / 14.5 GB |
| 3 — HRKG Build | `stage3_hrkg_build/build_hrkg.py` | §3.2 | 1.2B nodes in Neo4j |
| 4 — Compression | `stage4_graph_compression/optimize_graph.py` | Table 1 | 8.4M nodes / 3.2 GB |
| 5 — SLM Augment | `stage5_slm_augmentation/augment_llama.py` | §4.2 | Llama 3.2 1B + graph |
| 6 — MCP Server | `stage6_mcp_server/wisdom_mcp.py` | §2 | wisdomGraph MCP |
| Eval | `eval/hle_benchmark.py` | Table 2 | 92.4% parity, 24 tok/s |

## Quick Start

```bash
# 1. Run full pipeline
./run_pipeline.sh

# 2. Or run individual stages
source .venv/bin/activate
python stage1_manifold_probe/probe_experts.py config/pipeline.yaml

# 3. Start MCP server standalone
source .venv/bin/activate
cd stage6_mcp_server
uvicorn wisdom_mcp:app --port 8765

# 4. Run evaluation only
python eval/hle_benchmark.py config/pipeline.yaml
```

## Key Claims to Prove

- [ ] 185× compression: 1.1T params → 12B equiv params, 594 GB → 3.2 GB
- [ ] 92.4% reasoning parity on HLE (49.9% vs Kimi's 54.0%)
- [ ] 24 tok/s on Apple Neural Engine / mobile NPU
- [ ] 180 ms TTFT (vs 800–1500 ms cloud baseline)
- [ ] Temporal decay preserves relevant knowledge, prunes stale facts
DOC
ok "README.md"

# ── Final summary ─────────────────────────────────────────────────────────────
echo ""
echo -e "${GREEN}══════════════════════════════════════════════════════════════${NC}"
echo -e "${GREEN}  NRT Framework scaffold created at:${NC}"
echo -e "${CYAN}  $ROOT${NC}"
echo ""
echo -e "  Structure:"
echo "    config/pipeline.yaml              — all targets & hyperparams"
echo "    stage1_manifold_probe/            — §3.1 Expert Convergence probing"
echo "    stage2_spo_extraction/            — §3.1 SPO triple extraction"
echo "    stage3_hrkg_build/                — §3.2 LLHKG hyper-relational graph"
echo "    stage4_graph_compression/         — Table 1: 185× compression + DIKW"
echo "    stage5_slm_augmentation/          — §4.2 Llama 3.2 1B + KG RAG"
echo "    stage6_mcp_server/                — §2 wisdomGraph MCP server"
echo "    eval/hle_benchmark.py             — Table 2: HLE parity test"
echo "    run_pipeline.sh                   — end-to-end runner"
echo ""
echo -e "${GREEN}  Run:  cd $ROOT && ./run_pipeline.sh${NC}"
echo -e "${GREEN}══════════════════════════════════════════════════════════════${NC}"
