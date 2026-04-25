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
