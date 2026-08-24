#!/usr/bin/env bash
set -euo pipefail

MDB=${MDB:-build/GNN/bin/mdb}
DB=${DB:-/tmp/mdb_c3_cora_declarative_reductions}
PORT=${PORT:-7899}

CORA_GQL=${CORA_GQL:-data/example/gql/cora/cora.gql}
CORA_NPY=${CORA_NPY:-data/example/gql/cora/cora_features.npy}
FEATURE_NAME=${FEATURE_NAME:-node_features}

LOG_DIR=${LOG_DIR:-logs/C3_declarative_projection/cora_reductions}
mkdir -p "$LOG_DIR"

SERVER_LOG="$LOG_DIR/server.log"
SUMMARY_CSV="$LOG_DIR/summary.csv"
SUMMARY_TXT="$LOG_DIR/summary.txt"

query() {
  curl -s -X POST "http://localhost:$PORT/gql" \
    -H "Content-Type: text/plain" \
    --data-binary "$1"
}

cleanup() {
  if [[ -n "${SRV:-}" ]]; then
    kill "$SRV" 2>/dev/null || true
    wait "$SRV" 2>/dev/null || true
  fi
}
trap cleanup EXIT

extract_second_line() {
  tail -n +2 | head -n 1
}

echo "==== C3 Cora Declarative Reductions ====" | tee "$SUMMARY_TXT"
echo "DB=$DB" | tee -a "$SUMMARY_TXT"
echo "FEATURE_NAME=$FEATURE_NAME" | tee -a "$SUMMARY_TXT"
echo | tee -a "$SUMMARY_TXT"

rm -rf "$DB"

echo "==== Import Cora ====" | tee -a "$SUMMARY_TXT"
"$MDB" import "$CORA_GQL" "$DB" --with-tensors "$CORA_NPY" 2>&1 | tee "$LOG_DIR/01_import.log"

echo "==== Start server ====" | tee -a "$SUMMARY_TXT"
"$MDB" server "$DB" -p "$PORT" --browser false > "$SERVER_LOG" 2>&1 &
SRV=$!

for i in $(seq 1 60); do
  if query "RETURN 1" | grep -q 1; then
    break
  fi
  sleep 1
done

echo "variant,edges,projectedNodes,totalBatches,trainBatches,validationBatches,testBatches,uniqueNodes,bestValAccuracy,testAccuracy,trainSeconds,dbSize" > "$SUMMARY_CSV"

run_variant() {
  local variant="$1"
  local graph_name="$2"
  local sample_name="$3"
  local where_clause="$4"

  echo | tee -a "$SUMMARY_TXT"
  echo "==== Variant: $variant ====" | tee -a "$SUMMARY_TXT"

  local count_query
  local project_query

  if [[ -z "$where_clause" ]]; then
    count_query="MATCH (n:Paper)-[r:CITES]->(m:Paper) RETURN COUNT(r) AS edgeCount"
    project_query="MATCH (n:Paper)-[r:CITES]->(m:Paper) RETURN PROJECT('$graph_name' INCLUDE LABELS INCLUDE PROPERTIES)"
  else
    count_query="MATCH (n:Paper)-[r:CITES]->(m:Paper) WHERE $where_clause RETURN COUNT(r) AS edgeCount"
    project_query="MATCH (n:Paper)-[r:CITES]->(m:Paper) WHERE $where_clause RETURN PROJECT('$graph_name' INCLUDE LABELS INCLUDE PROPERTIES)"
  fi

  local edges_result
  edges_result=$(query "$count_query")
  echo "$edges_result" | tee "$LOG_DIR/${variant}_01_edges.log" | tee -a "$SUMMARY_TXT"
  local edges
  edges=$(echo "$edges_result" | tail -n +2 | head -n 1)

  local project_result
  project_result=$(query "$project_query")
  echo "$project_result" | tee "$LOG_DIR/${variant}_02_project.log" | tee -a "$SUMMARY_TXT"

  local nodes_result
  nodes_result=$(query "USE $graph_name MATCH (n:Paper) RETURN COUNT(n) AS projectedNodes")
  echo "$nodes_result" | tee "$LOG_DIR/${variant}_03_nodes.log" | tee -a "$SUMMARY_TXT"
  local projected_nodes
  projected_nodes=$(echo "$nodes_result" | tail -n +2 | head -n 1)

  local prepare_result
  prepare_result=$(query "CALL gnn_prepare_projection('$graph_name', {includeFeatures:'$FEATURE_NAME',labelProperty:'label',splitProperty:'split'}) YIELD projectionName,featureName,nodeCount,featureDim,numClasses,hasLabels,hasSplits RETURN *")
  echo "$prepare_result" | tee "$LOG_DIR/${variant}_04_prepare.log" | tee -a "$SUMMARY_TXT"

  local sample_result
  sample_result=$(query "CALL gnn_offline_sample('$graph_name','$sample_name',[10,5],{batchSize:64,randomSeed:42,usePredefinedSplits:true,orientation:'UNDIRECTED'}) YIELD totalBatches,trainBatches,validationBatches,testBatches,uniqueNodes,numWorkersUsed RETURN *")
  echo "$sample_result" | tee "$LOG_DIR/${variant}_05_sample.log" | tee -a "$SUMMARY_TXT"

  local sample_values
  sample_values=$(echo "$sample_result" | tail -n +2 | head -n 1)
  IFS=',' read -r total_batches train_batches val_batches test_batches unique_nodes workers <<< "$sample_values"

  local store_result
  store_result=$(query "CALL gnn_build_feature_store('$sample_name','$FEATURE_NAME',{force:true}) YIELD sampleName,featureName,totalNodes RETURN *")
  echo "$store_result" | tee "$LOG_DIR/${variant}_06_store.log" | tee -a "$SUMMARY_TXT"

  local train_result
  train_result=$(query "CALL gnn_train('$sample_name','$FEATURE_NAME',{model:'graphsage',epochs:30,hiddenDim:64,learningRate:0.01,patience:10,randomSeed:42}) YIELD bestValAccuracy,testAccuracy,epochsRun,trainSeconds RETURN *")
  echo "$train_result" | tee "$LOG_DIR/${variant}_07_train.log" | tee -a "$SUMMARY_TXT"

  local train_values
  train_values=$(echo "$train_result" | tail -n +2 | head -n 1)
  IFS=',' read -r best_val test_acc epochs_run train_seconds <<< "$train_values"

  local db_size
  db_size=$(du -sh "$DB" | awk '{print $1}')

  echo "$variant,$edges,$projected_nodes,$total_batches,$train_batches,$val_batches,$test_batches,$unique_nodes,$best_val,$test_acc,$train_seconds,$db_size" >> "$SUMMARY_CSV"
}

run_variant \
  "full_declarative" \
  "cora_c3_full_declarative" \
  "cora_c3_full_declarative_s" \
  ""

run_variant \
  "train_touch" \
  "cora_c3_train_touch" \
  "cora_c3_train_touch_s" \
  'n.split = "train" OR m.split = "train"'

run_variant \
  "train_val_touch" \
  "cora_c3_train_val_touch" \
  "cora_c3_train_val_touch_s" \
  'n.split = "train" OR n.split = "val" OR n.split = "valid" OR m.split = "train" OR m.split = "val" OR m.split = "valid"'

echo | tee -a "$SUMMARY_TXT"
echo "==== Summary CSV ====" | tee -a "$SUMMARY_TXT"
cat "$SUMMARY_CSV" | tee -a "$SUMMARY_TXT"

echo "==== PASS C3 Cora Declarative Reductions ====" | tee -a "$SUMMARY_TXT"
