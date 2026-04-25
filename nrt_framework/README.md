# NRT Framework — Research Proof Scaffold

Neural-to-Relational Transduction: De-parameterizing Kimi K2.6 into Mobile-Ready Wisdom Graphs.

## Pipeline Stages

| Stage | Script | Paper Section | Target |
|-------|--------|---------------|--------|
| 1 — Manifold Probe | `stage1_manifold_probe/probe_experts.py` | §3.1 | Expert Convergence map |
| 2 — SPO Extraction | `stage2_spo_extraction/extract_triples.py` | §3.1 | 450B triples / 14.5 GB |
| 3 — HRKG Build | `stage3_hrkg_build/build_hrkg.py` | §3.2 | 1.2B nodes in Neo4j |
| 4 — Compression | `stage4_graph_compression/optimize_graph.py` | Table 1 | 8.4M nodes / 3.2 GB |
| 5 — Base Model | `stage5_slm_augmentation/augment_llama.py` | §4.2 | Qwen3-30B-A3B (MoE, 3B active) + graph |
| 6 — MCP Server | `stage6_mcp_server/wisdom_mcp.py` | §2 | wisdomGraph MCP |
| Eval | `eval/hle_benchmark.py` | Table 2 | 92.4% parity, 100+ tok/s |

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
- [ ] 100+ tok/s on 32GB mini PC (Qwen3-30B-A3B Q4_K_M, primary target)
- [ ] 150 ms TTFT (vs 800–1500 ms cloud baseline)
- [ ] Temporal decay preserves relevant knowledge, prunes stale facts

## Hardware Targets

| Config | Hardware | Base Model | Active Params | Graph RAM |
|--------|----------|-----------|--------------|-----------|
| Primary | 32GB mini PC (x86) | Qwen3-30B-A3B Q4_K_M | 3B (~6GB) | ~26GB free |
| Extended | 128GB Mac M-series | Qwen3-235B-A22B Q4 | 22B (~50GB) | ~78GB free |

## MoE Architecture Alignment

Both Kimi K2.6 (32B active / 1.1T total) and Qwen3-30B-A3B (3B active / 30B total)
are Mixture-of-Experts models. NRT maps Kimi's expert activations → graph nodes;
Qwen3-30B-A3B queries the same graph at inference time — 10× active-param compression.
