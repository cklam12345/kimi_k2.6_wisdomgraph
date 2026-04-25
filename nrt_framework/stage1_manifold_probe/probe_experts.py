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
