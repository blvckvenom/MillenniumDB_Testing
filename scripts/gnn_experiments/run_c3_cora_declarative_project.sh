#!/usr/bin/env bash
set -euo pipefail

MDB=${MDB:-build/GNN/bin/mdb}
DB=${DB:-/tmp/mdb_c3_cora_declarative_project}
PORT=${PORT:-7897}

CORA_GQL=${CORA_GQL:-data/example/gql/cora/cora.gql}
CORA_NPY=${CORA_NPY:-data/example/gql/cora/cora_features.npy}

GRAPH_NAME=${GRAPH_NAME:-cora_c3_declarative_project}
SAMPLE_NAME=${SAMPLE_NAME:-cora_c3_declarative_project_s}
FEATURE_NAME=${FEATURE_NAME:-node_features}

LOG_DIR=${LOG_DIR:-logs/C3_declarative_projection/cora}
mkdir -p "$LOG_DIR"

SERVER_LOG="$LOG_DIR/server.log"
SUMMARY_LOG="$LOG_DIR/summary.log"

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

echo "==== C3 Cora Declarative Projection Experiment ====" | tee "$SUMMARY_LOG"
echo "DB=$DB" | tee -a "$SUMMARY_LOG"
echo "GRAPH_NAME=$GRAPH_NAME" | tee -a "$SUMMARY_LOG"
echo "SAMPLE_NAME=$SAMPLE_NAME" | tee -a "$SUMMARY_LOG"
echo "FEATURE_NAME=$FEATURE_NAME" | tee -a "$SUMMARY_LOG"
echo | tee -a "$SUMMARY_LOG"

echo "==== Import Cora ====" | tee -a "$SUMMARY_LOG"
rm -rf "$DB"
"$MDB" import "$CORA_GQL" "$DB" --with-tensors "$CORA_NPY" 2>&1 | tee "$LOG_DIR/01_import.log"

echo "==== Start server ====" | tee -a "$SUMMARY_LOG"
"$MDB" server "$DB" -p "$PORT" --browser false > "$SERVER_LOG" 2>&1 &
SRV=$!

for i in $(seq 1 60); do
  if query "RETURN 1" | grep -q 1; then
    break
  fi
  sleep 1
done

echo "==== Declarative PROJECT() ====" | tee -a "$SUMMARY_LOG"
PROJECT_RESULT=$(query "MATCH (n:Paper)-[r:CITES]->(m:Paper) RETURN PROJECT('$GRAPH_NAME' INCLUDE LABELS INCLUDE PROPERTIES)")
echo "$PROJECT_RESULT" | tee "$LOG_DIR/02_project.log" | tee -a "$SUMMARY_LOG"

echo "==== gnn_prepare_projection ====" | tee -a "$SUMMARY_LOG"
PREPARE_RESULT=$(query "CALL gnn_prepare_projection('$GRAPH_NAME', {includeFeatures:'$FEATURE_NAME',labelProperty:'label',splitProperty:'split'}) YIELD projectionName,featureName,nodeCount,featureDim,numClasses,hasLabels,hasSplits RETURN *")
echo "$PREPARE_RESULT" | tee "$LOG_DIR/03_prepare_projection.log" | tee -a "$SUMMARY_LOG"

echo "==== gnn_offline_sample ====" | tee -a "$SUMMARY_LOG"
SAMPLE_RESULT=$(query "CALL gnn_offline_sample('$GRAPH_NAME','$SAMPLE_NAME',[10,5],{batchSize:64,randomSeed:42,usePredefinedSplits:true,orientation:'UNDIRECTED'}) YIELD totalBatches,trainBatches,validationBatches,testBatches,uniqueNodes,numWorkersUsed RETURN *")
echo "$SAMPLE_RESULT" | tee "$LOG_DIR/04_sample.log" | tee -a "$SUMMARY_LOG"

echo "==== gnn_build_feature_store ====" | tee -a "$SUMMARY_LOG"
STORE_RESULT=$(query "CALL gnn_build_feature_store('$SAMPLE_NAME','$FEATURE_NAME',{force:true}) YIELD sampleName,featureName,totalNodes RETURN *")
echo "$STORE_RESULT" | tee "$LOG_DIR/05_build_feature_store.log" | tee -a "$SUMMARY_LOG"

echo "==== gnn_train ====" | tee -a "$SUMMARY_LOG"
TRAIN_RESULT=$(query "CALL gnn_train('$SAMPLE_NAME','$FEATURE_NAME',{model:'graphsage',epochs:30,hiddenDim:64,learningRate:0.01,patience:10,randomSeed:42}) YIELD bestValAccuracy,testAccuracy,epochsRun,trainSeconds RETURN *")
echo "$TRAIN_RESULT" | tee "$LOG_DIR/06_train.log" | tee -a "$SUMMARY_LOG"

echo "==== Sidecars ====" | tee -a "$SUMMARY_LOG"
ls -lh "$DB/projections/$GRAPH_NAME" | grep -E "gnn_meta|labels|splits" | tee "$LOG_DIR/07_sidecars.log" | tee -a "$SUMMARY_LOG"

echo "==== DB size ====" | tee -a "$SUMMARY_LOG"
du -sh "$DB" | tee "$LOG_DIR/08_db_size.log" | tee -a "$SUMMARY_LOG"

echo "==== PASS C3 Cora Declarative Projection ====" | tee -a "$SUMMARY_LOG"
