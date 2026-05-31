#!/usr/bin/env bash
# Fast config bench: fanout [10,10,10] + batchSize=512 + lr=0.003
# Rebuilds sample + feature_store + runs 50-epoch training
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DB="$REPO_DIR/data/dbs/gql/papers100M"
MDB="$REPO_DIR/build/Release/bin/mdb"
PORT="${PORT:-29954}"
OUT_DIR="${OUT_DIR:-$HOME/Desktop/spec13_papers100m_e2e/post_pop_os/33_fast_config_5ep}"
mkdir -p "$OUT_DIR"

SAMPLE_NAME="papers100M_f10_b512"

pkill -f "bin/mdb server.*--port $PORT" 2>/dev/null || true
sleep 3

LOG="$OUT_DIR/server.log"
env MDB_GNN_PIPELINE_OVERLAP=1 MDB_GNN_REORDER_STRATEGY=external_sort \
    nohup "$MDB" server "$DB" --port "$PORT" --timeout 86400 > "$LOG" 2>&1 &
SERVER_PID=$!
echo "server pid=$SERVER_PID"
sleep 8

# ------------------------------------------------------------------
# T0: gnn_offline_sample with fanout [10,10,10] + batchSize=512
# ------------------------------------------------------------------
cat > "$OUT_DIR/sample_query.gql" <<EOF
CALL gnn_offline_sample(
    "papers100M_e2e_opt",
    "$SAMPLE_NAME",
    [10, 10, 10],
    {
        batchSize: 512,
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

echo "=== T0: gnn_offline_sample fanout [10,10,10] batchSize=512 ==="
T0_START=$(date +%s)
curl -s --max-time 7200 -X POST -H 'Content-Type: application/gql' \
    --data-binary @"$OUT_DIR/sample_query.gql" \
    -o "$OUT_DIR/sample.csv" \
    -w "T0 HTTP=%{http_code} time=%{time_total}s\n" \
    "http://localhost:$PORT/gql" | tee "$OUT_DIR/sample.curl"
T0_END=$(date +%s)
T0_ELAPSED=$((T0_END - T0_START))
echo "T0 sample: ${T0_ELAPSED}s"
cat "$OUT_DIR/sample.csv"

# ------------------------------------------------------------------
# T1: gnn_build_feature_store (caches preserved, packed_slim regen)
# ------------------------------------------------------------------
cat > "$OUT_DIR/feature_query.gql" <<EOF
CALL gnn_build_feature_store(
    "$SAMPLE_NAME",
    "node_features",
    {
        gpu_budget_mb:      10000,
        cpu_budget_mb:      5000,
        force:              true,
        force_reorder:      false,
        force_caches:       false,
        force_packed_slim:  true
    }
) YIELD * RETURN *
EOF

echo
echo "=== T1: feature_store rebuild (packed_slim only) ==="
T1_START=$(date +%s)
curl -s --max-time 7200 -X POST -H 'Content-Type: application/gql' \
    --data-binary @"$OUT_DIR/feature_query.gql" \
    -o "$OUT_DIR/feature.csv" \
    -w "T1 HTTP=%{http_code} time=%{time_total}s\n" \
    "http://localhost:$PORT/gql" | tee "$OUT_DIR/feature.curl"
T1_END=$(date +%s)
T1_ELAPSED=$((T1_END - T1_START))
echo "T1 feature_store: ${T1_ELAPSED}s"

# ------------------------------------------------------------------
# T2: gnn_train 5 epochs (quick validate before 50)
# ------------------------------------------------------------------
cat > "$OUT_DIR/train_query.gql" <<EOF
CALL gnn_train(
    "$SAMPLE_NAME",
    "node_features",
    {
        epochs: 5,
        hiddenDim: 256,
        dropout: 0.2,
        batchSize: 512,
        learningRate: 0.003,
        randomSeed: 42,
        patience: 999,
        useAsyncPrefetcher: true,
        prefetchQueueSize: 2,
        saveOnBestVal: false,
        saveFinal: false,
        outputDir: "fast_config_5ep"
    }
) YIELD modelName, ranEpochs, bestValAccuracy, testAccuracy,
        trainSeconds, assembleSeconds, forwardSeconds, backwardSeconds,
        l1HitRatio, l2HitRatio, l3Reads, l4Reads
RETURN *
EOF

echo
echo "=== T2: gnn_train 5 epochs fast config ==="
T2_START=$(date +%s)
curl -s --max-time 14400 -X POST -H 'Content-Type: application/gql' \
    --data-binary @"$OUT_DIR/train_query.gql" \
    -o "$OUT_DIR/train.csv" \
    -w "T2 HTTP=%{http_code} time=%{time_total}s\n" \
    "http://localhost:$PORT/gql" | tee "$OUT_DIR/train.curl"
T2_END=$(date +%s)
T2_ELAPSED=$((T2_END - T2_START))
echo "T2 train: ${T2_ELAPSED}s"

grep -E '\[TrainingLoop\] epoch=' "$LOG" > "$OUT_DIR/epochs.log" || true

kill -15 "$SERVER_PID" 2>/dev/null || true
sleep 5
kill -9 "$SERVER_PID" 2>/dev/null || true

echo
echo "=== Summary ==="
echo "T0 sample build      : ${T0_ELAPSED}s"
echo "T1 feature store    : ${T1_ELAPSED}s"
echo "T2 5-epoch train     : ${T2_ELAPSED}s"
echo
cat "$OUT_DIR/epochs.log"
echo
echo "Train CSV:"; cat "$OUT_DIR/train.csv"
