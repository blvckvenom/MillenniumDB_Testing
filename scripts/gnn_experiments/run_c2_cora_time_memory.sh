#!/usr/bin/env bash
set -euo pipefail

mkdir -p logs/C2_scalability/time_memory

run_measured() {
  local name="$1"
  local script="$2"
  local log="logs/C2_scalability/time_memory/${name}.log"

  echo
  echo "================================================================"
  echo "C2 measured experiment: ${name}"
  echo "Script: ${script}"
  echo "Log: ${log}"
  echo "================================================================"

  /usr/bin/time -v "${script}" 2>&1 | tee "${log}"
}

run_measured "cora_original" "scripts/gnn_experiments/run_cora_original.sh"
run_measured "b5_structural_cora" "scripts/gnn_experiments/run_b5_structural_cora.sh"
run_measured "b6_query_rich_cora" "scripts/gnn_experiments/run_b6_query_rich_cora.sh"
run_measured "b5_combined_cora" "scripts/gnn_experiments/run_b5_combined_cora.sh"
run_measured "b6_query_combined_cora" "scripts/gnn_experiments/run_b6_query_combined_cora.sh"

echo
echo "C2 time/memory logs written to logs/C2_scalability/time_memory/"
