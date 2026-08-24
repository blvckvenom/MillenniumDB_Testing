#!/usr/bin/env bash
set -euo pipefail

MDB=${MDB:-build/GNN/bin/mdb}
DB=${DB:-/tmp/mdb_c4_cora_2hop_reductions_multiseed}
PORT=${PORT:-7906}

CORA_GQL=${CORA_GQL:-data/example/gql/cora/cora.gql}
CORA_NPY=${CORA_NPY:-data/example/gql/cora/cora_features.npy}
FEATURE_NAME=${FEATURE_NAME:-node_features}

SEEDS=${SEEDS:-"1 2 3 4 5"}

LOG_DIR=${LOG_DIR:-logs/C4_query_algebra/cora_2hop_reductions_multiseed}
mkdir -p "$LOG_DIR"

SERVER_LOG="$LOG_DIR/server.log"
RESULTS_CSV="$LOG_DIR/results_by_seed.csv"
SUMMARY_CSV="$LOG_DIR/summary_by_variant.csv"
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

echo "==== C4 Cora 2-hop Reductions Multi-Seed ====" | tee "$SUMMARY_TXT"
echo "DB=$DB" | tee -a "$SUMMARY_TXT"
echo "FEATURE_NAME=$FEATURE_NAME" | tee -a "$SUMMARY_TXT"
echo "SEEDS=$SEEDS" | tee -a "$SUMMARY_TXT"
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

echo "variant,seed,edges,projectedNodes,totalBatches,trainBatches,validationBatches,testBatches,uniqueNodes,bestValAccuracy,testAccuracy,trainSeconds,dbSize" > "$RESULTS_CSV"

project_and_prepare_variant() {
  local variant="$1"
  local graph_name="$2"
  local match_clause="$3"
  local where_clause="$4"
  local edge_var="$5"

  echo | tee -a "$SUMMARY_TXT"
  echo "==== Project variant: $variant ====" | tee -a "$SUMMARY_TXT"

  local count_query
  local project_query

  if [[ -z "$where_clause" ]]; then
    count_query="$match_clause RETURN COUNT($edge_var) AS edgeCount"
    project_query="$match_clause RETURN PROJECT('$graph_name' INCLUDE LABELS INCLUDE PROPERTIES)"
  else
    count_query="$match_clause WHERE $where_clause RETURN COUNT($edge_var) AS edgeCount"
    project_query="$match_clause WHERE $where_clause RETURN PROJECT('$graph_name' INCLUDE LABELS INCLUDE PROPERTIES)"
  fi

  local edges_result
  edges_result=$(query "$count_query")
  echo "$edges_result" | tee "$LOG_DIR/${variant}_01_edges.log" | tee -a "$SUMMARY_TXT"

  local project_result
  project_result=$(query "$project_query")
  echo "$project_result" | tee "$LOG_DIR/${variant}_02_project.log" | tee -a "$SUMMARY_TXT"

  local nodes_result
  nodes_result=$(query "USE $graph_name MATCH (n:Paper) RETURN COUNT(n) AS projectedNodes")
  echo "$nodes_result" | tee "$LOG_DIR/${variant}_03_nodes.log" | tee -a "$SUMMARY_TXT"

  local projected_edges_result
  projected_edges_result=$(query "USE $graph_name MATCH (n:Paper)-[r:CITES]->(m:Paper) RETURN COUNT(r) AS projectedEdges")
  echo "$projected_edges_result" | tee "$LOG_DIR/${variant}_04_projected_edges.log" | tee -a "$SUMMARY_TXT"

  local prepare_result
  prepare_result=$(query "CALL gnn_prepare_projection('$graph_name', {includeFeatures:'$FEATURE_NAME',labelProperty:'label',splitProperty:'split'}) YIELD projectionName,featureName,nodeCount,featureDim,numClasses,hasLabels,hasSplits RETURN *")
  echo "$prepare_result" | tee "$LOG_DIR/${variant}_05_prepare.log" | tee -a "$SUMMARY_TXT"
}

run_variant_seed() {
  local variant="$1"
  local graph_name="$2"
  local seed="$3"

  local sample_name="${graph_name}_seed_${seed}_s"

  echo | tee -a "$SUMMARY_TXT"
  echo "==== Run variant=$variant seed=$seed ====" | tee -a "$SUMMARY_TXT"

  local edges
  edges=$(tail -n +2 "$LOG_DIR/${variant}_04_projected_edges.log" | head -n 1)

  local projected_nodes
  projected_nodes=$(tail -n +2 "$LOG_DIR/${variant}_03_nodes.log" | head -n 1)

  local sample_result
  sample_result=$(query "CALL gnn_offline_sample('$graph_name','$sample_name',[10,5],{batchSize:64,randomSeed:$seed,usePredefinedSplits:true,orientation:'UNDIRECTED',force:true}) YIELD totalBatches,trainBatches,validationBatches,testBatches,uniqueNodes,numWorkersUsed RETURN *")
  echo "$sample_result" | tee "$LOG_DIR/${variant}_seed_${seed}_01_sample.log" | tee -a "$SUMMARY_TXT"

  local sample_values
  sample_values=$(echo "$sample_result" | tail -n +2 | head -n 1)
  IFS=',' read -r total_batches train_batches val_batches test_batches unique_nodes workers <<< "$sample_values"

  local store_result
  store_result=$(query "CALL gnn_build_feature_store('$sample_name','$FEATURE_NAME',{force:true}) YIELD sampleName,featureName,totalNodes RETURN *")
  echo "$store_result" | tee "$LOG_DIR/${variant}_seed_${seed}_02_store.log" | tee -a "$SUMMARY_TXT"

  local train_result
  train_result=$(query "CALL gnn_train('$sample_name','$FEATURE_NAME',{model:'graphsage',epochs:30,hiddenDim:64,learningRate:0.01,patience:10,randomSeed:$seed}) YIELD bestValAccuracy,testAccuracy,epochsRun,trainSeconds RETURN *")
  echo "$train_result" | tee "$LOG_DIR/${variant}_seed_${seed}_03_train.log" | tee -a "$SUMMARY_TXT"

  local train_values
  train_values=$(echo "$train_result" | tail -n +2 | head -n 1)
  IFS=',' read -r best_val test_acc epochs_run train_seconds <<< "$train_values"

  local db_size
  db_size=$(du -sh "$DB" | awk '{print $1}')

  echo "$variant,$seed,$edges,$projected_nodes,$total_batches,$train_batches,$val_batches,$test_batches,$unique_nodes,$best_val,$test_acc,$train_seconds,$db_size" >> "$RESULTS_CSV"
}

MATCH_1HOP='MATCH (n:Paper)-[r:CITES]->(m:Paper)'
MATCH_2HOP='MATCH (a:Paper)-[r1:CITES]->(b:Paper)-[r2:CITES]->(c:Paper)'

TRAIN_VAL_1HOP='n.split = "train" OR n.split = "val" OR n.split = "valid" OR m.split = "train" OR m.split = "val" OR m.split = "valid"'
MIDDLE_TRAIN_VAL='b.split = "train" OR b.split = "val" OR b.split = "valid"'
ANY_TRAIN_VAL='a.split = "train" OR a.split = "val" OR a.split = "valid" OR b.split = "train" OR b.split = "val" OR b.split = "valid" OR c.split = "train" OR c.split = "val" OR c.split = "valid"'

project_and_prepare_variant \
  "full_declarative" \
  "cora_c4_full_ms" \
  "$MATCH_1HOP" \
  "" \
  "r"

project_and_prepare_variant \
  "train_val_touch" \
  "cora_c4_train_val_touch_ms" \
  "$MATCH_1HOP" \
  "$TRAIN_VAL_1HOP" \
  "r"

project_and_prepare_variant \
  "2hop_middle_train_val" \
  "cora_c4_2hop_middle_train_val_ms" \
  "$MATCH_2HOP" \
  "$MIDDLE_TRAIN_VAL" \
  "r2"

project_and_prepare_variant \
  "2hop_any_train_val" \
  "cora_c4_2hop_any_train_val_ms" \
  "$MATCH_2HOP" \
  "$ANY_TRAIN_VAL" \
  "r2"

for seed in $SEEDS; do
  run_variant_seed "full_declarative" "cora_c4_full_ms" "$seed"
  run_variant_seed "train_val_touch" "cora_c4_train_val_touch_ms" "$seed"
  run_variant_seed "2hop_middle_train_val" "cora_c4_2hop_middle_train_val_ms" "$seed"
  run_variant_seed "2hop_any_train_val" "cora_c4_2hop_any_train_val_ms" "$seed"
done

echo | tee -a "$SUMMARY_TXT"
echo "==== Results by seed ====" | tee -a "$SUMMARY_TXT"
cat "$RESULTS_CSV" | tee -a "$SUMMARY_TXT"

python3 - "$RESULTS_CSV" "$SUMMARY_CSV" <<'PY'
import csv
import math
import sys
from collections import defaultdict

in_path, out_path = sys.argv[1], sys.argv[2]

rows = []
with open(in_path, newline="") as f:
    reader = csv.DictReader(f)
    for row in reader:
        rows.append(row)

groups = defaultdict(list)
for row in rows:
    groups[row["variant"]].append(row)

def mean(xs):
    return sum(xs) / len(xs) if xs else float("nan")

def std(xs):
    if len(xs) <= 1:
        return 0.0
    m = mean(xs)
    return math.sqrt(sum((x - m) ** 2 for x in xs) / (len(xs) - 1))

fields = [
    "variant",
    "runs",
    "edges",
    "projectedNodes",
    "totalBatches_mean",
    "totalBatches_std",
    "uniqueNodes_mean",
    "uniqueNodes_std",
    "bestValAccuracy_mean",
    "bestValAccuracy_std",
    "testAccuracy_mean",
    "testAccuracy_std",
    "trainSeconds_mean",
    "trainSeconds_std",
]

with open(out_path, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=fields)
    writer.writeheader()

    for variant in sorted(groups.keys()):
        rs = groups[variant]

        total_batches = [float(r["totalBatches"]) for r in rs]
        unique_nodes = [float(r["uniqueNodes"]) for r in rs]
        best_val = [float(r["bestValAccuracy"]) for r in rs]
        test_acc = [float(r["testAccuracy"]) for r in rs]
        train_seconds = [float(r["trainSeconds"]) for r in rs]

        writer.writerow({
            "variant": variant,
            "runs": len(rs),
            "edges": rs[0]["edges"],
            "projectedNodes": rs[0]["projectedNodes"],
            "totalBatches_mean": f"{mean(total_batches):.6g}",
            "totalBatches_std": f"{std(total_batches):.6g}",
            "uniqueNodes_mean": f"{mean(unique_nodes):.6g}",
            "uniqueNodes_std": f"{std(unique_nodes):.6g}",
            "bestValAccuracy_mean": f"{mean(best_val):.6g}",
            "bestValAccuracy_std": f"{std(best_val):.6g}",
            "testAccuracy_mean": f"{mean(test_acc):.6g}",
            "testAccuracy_std": f"{std(test_acc):.6g}",
            "trainSeconds_mean": f"{mean(train_seconds):.6g}",
            "trainSeconds_std": f"{std(train_seconds):.6g}",
        })
PY

echo | tee -a "$SUMMARY_TXT"
echo "==== Summary by variant ====" | tee -a "$SUMMARY_TXT"
cat "$SUMMARY_CSV" | tee -a "$SUMMARY_TXT"

echo "==== PASS C4 Cora 2-hop Reductions Multi-Seed ====" | tee -a "$SUMMARY_TXT"
