#!/usr/bin/env bash
# T0 MICRO: papers100M smoke test with fanout=[5,5] batchSize=256.
# Goal: minimal scale debug — does the stack run end-to-end without crashes
# and does loss decrease? Total wall-clock target ~25 min.
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DB="$REPO_DIR/data/dbs/gql/papers100M"
MDB="$REPO_DIR/build/Release/bin/mdb"
PORT="${PORT:-29960}"
OUT_DIR="${OUT_DIR:-$HOME/Desktop/spec13_papers100m_e2e/post_pop_os/35_t0_micro_f5_b256}"
mkdir -p "$OUT_DIR"

SAMPLE_NAME="papers100M_t0_f5_b256"

pkill -f "bin/mdb server.*--port $PORT" 2>/dev/null || true
sleep 3

LOG="$OUT_DIR/server.log"
echo "=== Launching server (port $PORT) ==="
env MDB_GNN_PIPELINE_OVERLAP=1 \
    MDB_GNN_REORDER_STRATEGY=external_sort \
    nohup "$MDB" server "$DB" --port "$PORT" --timeout 86400 > "$LOG" 2>&1 &
SERVER_PID=$!
echo "server pid=$SERVER_PID"
sleep 8

# ------------------------------------------------------------------
# Phase 1: gnn_offline_sample (smallest config — fanout [5,5] batch 256)
# ------------------------------------------------------------------
cat > "$OUT_DIR/sample_query.gql" <<EOF
CALL gnn_offline_sample(
    "papers100M_e2e_opt",
    "$SAMPLE_NAME",
    [5, 5],
    {
        batchSize: 256,
        randomSeed: 42,
        orientation: 'UNDIRECTED',
        usePredefinedSplits: true,
        numWorkers: 10,
        useFourLevelTopologyStore: false,
        useAdjacencyCache: false
    }
) YIELD totalBatches, trainBatches, valBatches, testBatches,
        uniqueNodes, computeMillis, numWorkersUsed
RETURN *
EOF

echo
echo "=== Phase 1: gnn_offline_sample (fanout [5,5] batch 256) ==="
P1_START=$(date +%s)
curl -s --max-time 1200 -X POST -H 'Content-Type: application/gql' \
    --data-binary @"$OUT_DIR/sample_query.gql" \
    -o "$OUT_DIR/sample.csv" \
    -w "P1 HTTP=%{http_code} time=%{time_total}s\n" \
    "http://localhost:$PORT/gql" | tee "$OUT_DIR/sample.curl"
P1_END=$(date +%s)
P1_ELAPSED=$((P1_END - P1_START))
echo "Phase 1 sample build: ${P1_ELAPSED}s"
cat "$OUT_DIR/sample.csv"

# ------------------------------------------------------------------
# Phase 2: gnn_build_feature_store — reuse reordered.fmat (force_reorder=false)
# ------------------------------------------------------------------
cat > "$OUT_DIR/feature_query.gql" <<EOF
CALL gnn_build_feature_store(
    "$SAMPLE_NAME",
    "node_features",
    {
        gpu_budget_mb:      4000,
        cpu_budget_mb:      1500,
        force:              true,
        force_reorder:      false,
        force_caches:       true,
        force_packed_slim:  true
    }
) YIELD * RETURN *
EOF

echo
echo "=== Phase 2: gnn_build_feature_store (reuse reordered.fmat) ==="
P2_START=$(date +%s)
curl -s --max-time 3600 -X POST -H 'Content-Type: application/gql' \
    --data-binary @"$OUT_DIR/feature_query.gql" \
    -o "$OUT_DIR/feature.csv" \
    -w "P2 HTTP=%{http_code} time=%{time_total}s\n" \
    "http://localhost:$PORT/gql" | tee "$OUT_DIR/feature.curl"
P2_END=$(date +%s)
P2_ELAPSED=$((P2_END - P2_START))
echo "Phase 2 feature_store build: ${P2_ELAPSED}s"

# ------------------------------------------------------------------
# Phase 3: gnn_train 5 epochs
# ------------------------------------------------------------------
cat > "$OUT_DIR/train_query.gql" <<EOF
CALL gnn_train(
    "$SAMPLE_NAME",
    "node_features",
    {
        epochs: 5,
        hiddenDim: 256,
        dropout: 0.2,
        batchSize: 256,
        lr: 0.003,
        randomSeed: 42,
        patience: 999,
        useAsyncPrefetcher: true,
        prefetchQueueSize: 2,
        saveOnBestVal: false,
        saveFinal: false,
        outputDir: "t0_micro_5ep"
    }
) YIELD modelName, ranEpochs, bestValAccuracy, testAccuracy,
        trainSeconds, assembleSeconds, forwardSeconds, backwardSeconds,
        l1HitRatio, l2HitRatio, l3Reads, l4Reads
RETURN *
EOF

echo
echo "=== Phase 3: gnn_train 5 epochs fanout [5,5] batch 256 ==="
P3_START=$(date +%s)
curl -s --max-time 7200 -X POST -H 'Content-Type: application/gql' \
    --data-binary @"$OUT_DIR/train_query.gql" \
    -o "$OUT_DIR/train.csv" \
    -w "P3 HTTP=%{http_code} time=%{time_total}s\n" \
    "http://localhost:$PORT/gql" | tee "$OUT_DIR/train.curl"
P3_END=$(date +%s)
P3_ELAPSED=$((P3_END - P3_START))
echo "Phase 3 train: ${P3_ELAPSED}s"

grep -E '\[TrainingLoop\] epoch=' "$LOG" > "$OUT_DIR/epochs.log" || true

kill -15 "$SERVER_PID" 2>/dev/null || true
sleep 5
kill -9 "$SERVER_PID" 2>/dev/null || true

echo
echo "=== T0 MICRO Summary ==="
echo "Phase 1 sample build  : ${P1_ELAPSED}s"
echo "Phase 2 feature_store : ${P2_ELAPSED}s"
echo "Phase 3 train (5ep)   : ${P3_ELAPSED}s"
echo "Total                 : $((P1_ELAPSED + P2_ELAPSED + P3_ELAPSED))s"
echo
echo "Per-epoch:"
cat "$OUT_DIR/epochs.log"
echo
echo "Train CSV:"; cat "$OUT_DIR/train.csv"
echo
echo "Gate: loss must decrease across epochs."
echo "      If val_acc moves above 0.0776 baseline → stack works at micro scale."
