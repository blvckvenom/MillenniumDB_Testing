#!/usr/bin/env bash
# scripts/bench_e2e_diskgnn.sh — DiskGNN E2E pipeline replication on papers100M
#
# Replicates DiskGNN SIGMOD'25 Table 4 row 1 (SAGE/Papers100M = 65.91% / 1.09 hr)
# with paper-canonical hyperparameters.
#
# Phases timed:
#   T1: gnn_train (50 epochs, SAGE 3-layer hidden=256, fanout already [10,15,20]
#       in sample, batchSize=1024, dropout=0.2, lr=1e-3, seed=42)
#   T2: gnn_predict from best checkpoint
#
# Sample (papers100M_paper_und) and feature_store ASSUMED EXIST.
# Server runs with D-config env vars (best from bench 27):
#   MDB_GNN_REORDER_STRATEGY=external_sort MDB_GNN_PIPELINE_OVERLAP=1
# (Fix #22 fadvise stays default-on; Fix #14 idempotency saves re-build.)

set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DB="${DB:-$REPO_DIR/data/dbs/gql/papers100M}"
MDB="${MDB:-$REPO_DIR/build/Release/bin/mdb}"
PORT="${PORT:-29951}"
OUT_DIR="${OUT_DIR:-$HOME/Desktop/spec13_papers100m_e2e/post_pop_os/28_e2e_diskgnn}"
mkdir -p "$OUT_DIR"

if [[ ! -x "$MDB" ]]; then
    echo "FATAL: mdb binary not found at $MDB"; exit 1
fi
if [[ ! -d "$DB/samples/papers100M_paper_und" ]]; then
    echo "FATAL: sample not found"; exit 1
fi
if [[ ! -f "$DB/gnn_features/node_features_reordered.fmat" ]]; then
    echo "FATAL: feature_store reordered.fmat not found — build it first"; exit 1
fi

# Kill any prior server on PORT
pkill -f "bin/mdb server.*--port $PORT" 2>/dev/null || true
sleep 3

# Launch server with D-config env vars
LOG="$OUT_DIR/server.log"
echo "=== Launching server (port $PORT) ==="
env MDB_GNN_PIPELINE_OVERLAP=1 \
    MDB_GNN_REORDER_STRATEGY=external_sort \
    nohup "$MDB" server "$DB" --port "$PORT" --timeout 86400 \
    > "$LOG" 2>&1 &
SERVER_PID=$!
echo "server pid=$SERVER_PID"
sleep 8

# ------------------------------------------------------------------
# T1: gnn_train (paper config)
# ------------------------------------------------------------------
cat > "$OUT_DIR/train_query.gql" <<'EOF'
CALL gnn_train(
    "papers100M_paper_und",
    "node_features",
    {
        epochs:           50,
        hiddenDim:        256,
        dropout:          0.2,
        batchSize:        1024,
        learningRate:     0.001,
        randomSeed:       42,
        patience:         999,
        useAsyncPrefetcher: true,
        prefetchQueueSize: 2,
        saveOnBestVal:    true,
        saveFinal:        true,
        outputDir:        "diskgnn_e2e_50ep"
    }
) YIELD modelName, ranEpochs, didConverge, bestValAccuracy, testAccuracy,
        trainSeconds, assembleSeconds, forwardSeconds, backwardSeconds,
        l1HitRatio, l2HitRatio, l3Reads, l4Reads, totalBytesDisk,
        bestCheckpointPath, finalCheckpointPath
RETURN *
EOF

echo
echo "=== T1: gnn_train (50 epochs, SAGE-3L-h256, paper config) ==="
T1_START=$(date +%s)
curl -s --max-time 86400 -X POST -H 'Content-Type: application/gql' \
    --data-binary @"$OUT_DIR/train_query.gql" \
    -o "$OUT_DIR/train.csv" \
    -w "T1 HTTP=%{http_code} time=%{time_total}s\n" \
    "http://localhost:$PORT/gql" | tee "$OUT_DIR/train.curl"
T1_END=$(date +%s)
T1_ELAPSED=$((T1_END - T1_START))
echo "T1 wall-clock: ${T1_ELAPSED}s"

# Capture per-epoch lines from server log for the epoch timeline
grep -E '\[gnn_train\]|epoch.*loss|epoch.*acc|Epoch [0-9]' "$LOG" | tail -200 \
    > "$OUT_DIR/train_epochs.log" || true

# ------------------------------------------------------------------
# T2: gnn_predict from best checkpoint
# ------------------------------------------------------------------
cat > "$OUT_DIR/predict_query.gql" <<'EOF'
CALL gnn_predict(
    "papers100M_paper_und",
    "node_features",
    "diskgnn_e2e_50ep/checkpoints/best_model"
) YIELD checkpointPath, checkpointEpoch, checkpointValAccuracy,
        numBatches, numSeedNodes, embeddingDim,
        nodesWritten, nodesInferred, inferenceMillis,
        l1HitRatio, l2HitRatio, l3Reads, l4Reads
RETURN *
EOF

echo
echo "=== T2: gnn_predict (best_model checkpoint) ==="
T2_START=$(date +%s)
curl -s --max-time 7200 -X POST -H 'Content-Type: application/gql' \
    --data-binary @"$OUT_DIR/predict_query.gql" \
    -o "$OUT_DIR/predict.csv" \
    -w "T2 HTTP=%{http_code} time=%{time_total}s\n" \
    "http://localhost:$PORT/gql" | tee "$OUT_DIR/predict.curl"
T2_END=$(date +%s)
T2_ELAPSED=$((T2_END - T2_START))
echo "T2 wall-clock: ${T2_ELAPSED}s"

# Shutdown
kill -15 "$SERVER_PID" 2>/dev/null || true
sleep 5
kill -9 "$SERVER_PID" 2>/dev/null || true

# ------------------------------------------------------------------
# Summary
# ------------------------------------------------------------------
echo
echo "=== E2E Summary ==="
echo "T1 gnn_train   : ${T1_ELAPSED}s ($((T1_ELAPSED / 60))m $((T1_ELAPSED % 60))s)"
echo "T2 gnn_predict : ${T2_ELAPSED}s"
echo "Total E2E      : $((T1_ELAPSED + T2_ELAPSED))s"
echo
echo "Train CSV:";   cat "$OUT_DIR/train.csv"
echo
echo "Predict CSV:"; cat "$OUT_DIR/predict.csv"
echo
echo "Paper target: 65.91% test accuracy / 1.09 hr (DiskGNN Table 4)"
echo "Outputs in:   $OUT_DIR"
