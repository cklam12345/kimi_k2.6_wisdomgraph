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
