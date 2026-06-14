#!/usr/bin/env bash
set -euo pipefail

MDB="build/GNN/bin/mdb"
CORA_GQL="data/example/gql/cora/cora.gql"
DB_DIR="/tmp/mdb_cora_b5_structural"
LOG="/tmp/mdb_cora_b5_structural.log"
PORT=19901

FEATURE_NAME="cora_structural_b5_zscore"
GRAPH_NAME="cora_b5_structural_graph"
SAMPLE_NAME="cora_b5_structural_graph_s"
OUTPUT_DIR="cora_b5_structural_train"
SERVER_PID=""

fail() {
  echo "ERROR: $*" >&2
  exit 1
}

cleanup() {
  if [[ -n "${SERVER_PID:-}" ]] && kill -0 "$SERVER_PID" 2>/dev/null; then
    kill "$SERVER_PID" 2>/dev/null || true
    wait "$SERVER_PID" 2>/dev/null || true
  fi
}
trap cleanup EXIT

check_prereqs() {
  [[ -x "$MDB" ]] || fail "Expected executable $MDB. Build it before running this script."
  [[ -f "$CORA_GQL" ]] || fail "Missing Cora import file: $CORA_GQL"
}

wait_for_server() {
  echo "Waiting for MDB server on port ${PORT}..."
  for _ in $(seq 1 60); do
    if curl -sS --max-time 2 -X POST "http://localhost:${PORT}/gql" \
        -H "Accept:text/csv" \
        -H "Content-Type: application/gql" \
        --data-binary "RETURN 1" 2>/dev/null | grep -q "1"; then
      echo "Server is ready."
      return 0
    fi
    if ! kill -0 "$SERVER_PID" 2>/dev/null; then
      echo "Server log follows:"
      tail -n 100 "$LOG" 2>/dev/null || true
      fail "MDB server exited during startup."
    fi
    sleep 1
  done
  echo "Server log follows:"
  tail -n 100 "$LOG" 2>/dev/null || true
  fail "MDB server was not ready after 60 seconds."
}

run_gql() {
  local label="$1"
  local query="$2"
  local body

  echo
  echo "=== ${label} ==="
  body=$(curl -sS --max-time 7200 -X POST "http://localhost:${PORT}/gql" \
    -H "Accept:text/csv" \
    -H "Content-Type: application/gql" \
    --data-binary "$query")
  printf '%s\n' "$body"

  if printf '%s\n' "$body" | grep -qiE "error|exception|failed|not found|bad query|unexpected|out of range"; then
    fail "${label} returned an error response."
  fi
}

check_prereqs

echo "Removing previous database directory: ${DB_DIR}"
rm -rf "$DB_DIR"

echo "Importing Cora without tensors into ${DB_DIR}"
"$MDB" import "$CORA_GQL" "$DB_DIR"

echo "Starting MDB server on port ${PORT}; logs: ${LOG}"
"$MDB" server "$DB_DIR" -p "$PORT" -t 7200 --browser false >"$LOG" 2>&1 &
SERVER_PID=$!
wait_for_server

run_gql "Create native structural features" \
"CALL gnn_create_structural_features('${FEATURE_NAME}', 'Paper', 'CITES', {normalize:'zscore'})
YIELD featureName,nodeCount,featureDim,fmatPath,rmapPath,normalized
RETURN *"

run_gql "Project Cora graph" \
"CALL graph_project('${GRAPH_NAME}', 'Paper', 'CITES', {orientation:'UNDIRECTED',includeFeatures:'${FEATURE_NAME}',labelProperty:'label',splitProperty:'split'})
YIELD graphName,nodeCount,relationshipCount,featureDim,numClasses
RETURN *"

run_gql "Offline sample" \
"CALL gnn_offline_sample('${GRAPH_NAME}', '${SAMPLE_NAME}', [10,5], {batchSize:64,randomSeed:42,usePredefinedSplits:true,orientation:'UNDIRECTED'})
YIELD sampleName,projectionName,totalBatches,trainBatches,validationBatches,testBatches,uniqueNodes,computeMillis
RETURN *"

run_gql "Materialize batches" \
"CALL gnn_materialize_batches('${SAMPLE_NAME}', '${FEATURE_NAME}', {reorder:1,numHashes:2,force:1})
YIELD sampleName,featureName,totalBatches,reordered,reorderTimeMs,packTimeMs,totalTimeMs,packedDir
RETURN *"

run_gql "Build feature store" \
"CALL gnn_build_feature_store('${SAMPLE_NAME}', '${FEATURE_NAME}', {gpu_budget_mb:0,cpu_budget_mb:100,reorder:1,force:1})
YIELD sampleName,featureName,l1Nodes,l2Nodes,l3Nodes,l4Nodes,gpuAvailable,buildTimeMs
RETURN *"

run_gql "Train GraphSAGE" \
"CALL gnn_train('${SAMPLE_NAME}', '${FEATURE_NAME}', {model:'graphsage',hiddenDim:128,epochs:30,lr:0.01,dropout:0.5,patience:10,randomSeed:42,exportEmbeddings:true,outputDir:'${OUTPUT_DIR}'})
YIELD modelName,ranEpochs,didConverge,bestValAccuracy,testAccuracy,trainSeconds,bestCheckpointPath,finalCheckpointPath
RETURN *"

echo
echo "Done. Database: ${DB_DIR}"
echo "Server log: ${LOG}"
