#!/usr/bin/env bash
set -euo pipefail

SCRIPTS_DIR="scripts/gnn_experiments"

run_experiment() {
  local name="$1"
  local script="$2"
  local log="/tmp/c1_${name}.out"

  echo
  echo "================================================================"
  echo "C1 experiment: ${name}"
  echo "Script: ${script}"
  echo "Log: ${log}"
  echo "================================================================"

  "${script}" 2>&1 | tee "$log"
}

run_experiment "cora_original" "${SCRIPTS_DIR}/run_cora_original.sh"
run_experiment "b5_structural_cora" "${SCRIPTS_DIR}/run_b5_structural_cora.sh"
run_experiment "b6_query_rich_cora" "${SCRIPTS_DIR}/run_b6_query_rich_cora.sh"
run_experiment "b5_combined_cora" "${SCRIPTS_DIR}/run_b5_combined_cora.sh"
run_experiment "b6_query_combined_cora" "${SCRIPTS_DIR}/run_b6_query_combined_cora.sh"

echo
echo "C1 logs:"
echo "/tmp/c1_cora_original.out"
echo "/tmp/c1_b5_structural_cora.out"
echo "/tmp/c1_b6_query_rich_cora.out"
echo "/tmp/c1_b5_combined_cora.out"
echo "/tmp/c1_b6_query_combined_cora.out"
