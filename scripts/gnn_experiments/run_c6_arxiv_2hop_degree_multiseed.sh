#!/usr/bin/env bash
set -euo pipefail

MDB=${MDB:-build/GNN/bin/mdb}
DB=${DB:-/tmp/mdb_c6_arxiv_2hop_degree_multiseed}
PORT=${PORT:-7912}

ARXIV_GQL=${ARXIV_GQL:-data/example/gql/ogbn-arxiv/ogbn-arxiv/ogbn_arxiv.gql}
ARXIV_NPY=${ARXIV_NPY:-data/example/gql/ogbn-arxiv/ogbn-arxiv/ogbn_arxiv_features.npy}
FEATURE_NAME=${FEATURE_NAME:-node_features}

SEEDS=${SEEDS:-"1 2 3"}

LOG_DIR=${LOG_DIR:-logs/C6_arxiv_query_algebra/2hop_degree_multiseed}
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

echo "==== C6 Arxiv 2-hop Degree Reductions Multi-Seed ====" | tee "$SUMMARY_TXT"
echo "DB=$DB" | tee -a "$SUMMARY_TXT"
echo "FEATURE_NAME=$FEATURE_NAME" | tee -a "$SUMMARY_TXT"
echo "SEEDS=$SEEDS" | tee -a "$SUMMARY_TXT"
echo | tee -a "$SUMMARY_TXT"

rm -rf "$DB"

echo "==== Import Arxiv ====" | tee -a "$SUMMARY_TXT"
"$MDB" import "$ARXIV_GQL" "$DB" --with-tensors "$ARXIV_NPY" 2>&1 | tee "$LOG_DIR/01_import.log"

echo "==== Start server ====" | tee -a "$SUMMARY_TXT"
"$MDB" server "$DB" -p "$PORT" --browser false > "$SERVER_LOG" 2>&1 &
SRV=$!

for i in $(seq 1 120); do
  if query "RETURN 1" | grep -q 1; then
    break
  fi
  sleep 1
done

echo "variant,seed,edges,projectedNodes,totalBatches,trainBatches,validationBatches,testBatches,uniqueNodes,bestValAccuracy,testAccuracy,trainSeconds,dbSize" > "$RESULTS_CSV"

project_and_prepare_variant() {
  local variant="$1"
  local graph_name="$2"
  local where_clause="$3"

  echo | tee -a "$SUMMARY_TXT"
  echo "==== Project variant: $variant ====" | tee -a "$SUMMARY_TXT"

  local project_query

  if [[ -z "$where_clause" ]]; then
    project_query="MATCH (n:Node)-[r:CONNECTS]->(m:Node) RETURN PROJECT('$graph_name' INCLUDE LABELS INCLUDE PROPERTIES)"
  else
    project_query="MATCH (a:Node)-[r1:CONNECTS]->(b:Node)-[r2:CONNECTS]->(c:Node) WHERE $where_clause RETURN PROJECT('$graph_name' INCLUDE LABELS INCLUDE PROPERTIES)"
  fi

  local project_result
  project_result=$(query "$project_query")
  echo "$project_result" | tee "$LOG_DIR/${variant}_01_project.log" | tee -a "$SUMMARY_TXT"

  local nodes_result
  nodes_result=$(query "USE $graph_name MATCH (n:Node) RETURN COUNT(n) AS projectedNodes")
  echo "$nodes_result" | tee "$LOG_DIR/${variant}_02_nodes.log" | tee -a "$SUMMARY_TXT"

  local edges_result
  edges_result=$(query "USE $graph_name MATCH (n:Node)-[r:CONNECTS]->(m:Node) RETURN COUNT(r) AS projectedEdges")
  echo "$edges_result" | tee "$LOG_DIR/${variant}_03_edges.log" | tee -a "$SUMMARY_TXT"

  local prepare_result
  prepare_result=$(query "CALL gnn_prepare_projection('$graph_name', {includeFeatures:'$FEATURE_NAME',labelProperty:'label',splitProperty:'split'}) YIELD projectionName,featureName,nodeCount,featureDim,numClasses,hasLabels,hasSplits RETURN *")
  echo "$prepare_result" | tee "$LOG_DIR/${variant}_04_prepare.log" | tee -a "$SUMMARY_TXT"
}

run_variant_seed() {
  local variant="$1"
  local graph_name="$2"
  local seed="$3"

  local sample_name="${graph_name}_seed_${seed}_s"

  echo | tee -a "$SUMMARY_TXT"
  echo "==== Run variant=$variant seed=$seed ====" | tee -a "$SUMMARY_TXT"

  local edges
  edges=$(tail -n +2 "$LOG_DIR/${variant}_03_edges.log" | head -n 1)

  local projected_nodes
  projected_nodes=$(tail -n +2 "$LOG_DIR/${variant}_02_nodes.log" | head -n 1)

  local sample_result
  sample_result=$(query "CALL gnn_offline_sample('$graph_name','$sample_name',[10,5],{batchSize:1024,randomSeed:$seed,usePredefinedSplits:true,orientation:'UNDIRECTED',force:true}) YIELD totalBatches,trainBatches,validationBatches,testBatches,uniqueNodes,numWorkersUsed RETURN *")
  echo "$sample_result" | tee "$LOG_DIR/${variant}_seed_${seed}_01_sample.log" | tee -a "$SUMMARY_TXT"

  local sample_values
  sample_values=$(echo "$sample_result" | tail -n +2 | head -n 1)
  IFS=',' read -r total_batches train_batches val_batches test_batches unique_nodes workers <<< "$sample_values"

  local store_result
  store_result=$(query "CALL gnn_build_feature_store('$sample_name','$FEATURE_NAME',{force:true}) YIELD sampleName,featureName,totalNodes RETURN *")
  echo "$store_result" | tee "$LOG_DIR/${variant}_seed_${seed}_02_store.log" | tee -a "$SUMMARY_TXT"

  local train_result
  train_result=$(query "CALL gnn_train('$sample_name','$FEATURE_NAME',{model:'graphsage',epochs:20,hiddenDim:64,learningRate:0.01,patience:5,randomSeed:$seed}) YIELD bestValAccuracy,testAccuracy,epochsRun,trainSeconds RETURN *")
  echo "$train_result" | tee "$LOG_DIR/${variant}_seed_${seed}_03_train.log" | tee -a "$SUMMARY_TXT"

  local train_values
  train_values=$(echo "$train_result" | tail -n +2 | head -n 1)
  IFS=',' read -r best_val test_acc epochs_run train_seconds <<< "$train_values"

  local db_size
  db_size=$(du -sh "$DB" | awk '{print $1}')

  echo "$variant,$seed,$edges,$projected_nodes,$total_batches,$train_batches,$val_batches,$test_batches,$unique_nodes,$best_val,$test_acc,$train_seconds,$db_size" >> "$RESULTS_CSV"
}

TV_MIDDLE_DEG_20='(b.split = "train" OR b.split = "val" OR b.split = "valid") AND degree(b) <= 20'
TV_MIDDLE_DEG_50='(b.split = "train" OR b.split = "val" OR b.split = "valid") AND degree(b) <= 50'
TV_MIDDLE_DEG_100='(b.split = "train" OR b.split = "val" OR b.split = "valid") AND degree(b) <= 100'

project_and_prepare_variant \
  "full_declarative" \
  "arxiv_c6_full_ms" \
  ""

project_and_prepare_variant \
  "2hop_middle_deg_20" \
  "arxiv_c6_2hop_middle_deg_20_ms" \
  "$TV_MIDDLE_DEG_20"

project_and_prepare_variant \
  "2hop_middle_deg_50" \
  "arxiv_c6_2hop_middle_deg_50_ms" \
  "$TV_MIDDLE_DEG_50"

project_and_prepare_variant \
  "2hop_middle_deg_100" \
  "arxiv_c6_2hop_middle_deg_100_ms" \
  "$TV_MIDDLE_DEG_100"

for seed in $SEEDS; do
  run_variant_seed "full_declarative" "arxiv_c6_full_ms" "$seed"
  run_variant_seed "2hop_middle_deg_20" "arxiv_c6_2hop_middle_deg_20_ms" "$seed"
  run_variant_seed "2hop_middle_deg_50" "arxiv_c6_2hop_middle_deg_50_ms" "$seed"
  run_variant_seed "2hop_middle_deg_100" "arxiv_c6_2hop_middle_deg_100_ms" "$seed"
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

echo "==== PASS C6 Arxiv 2-hop Degree Reductions Multi-Seed ====" | tee -a "$SUMMARY_TXT"
