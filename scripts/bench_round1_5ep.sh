#!/usr/bin/env bash
# Quick 5-epoch validation of Round 1 optimizations.
# Reuses existing feature store + sample. No rebuild.
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DB="$REPO_DIR/data/dbs/gql/papers100M"
MDB="$REPO_DIR/build/Release/bin/mdb"
PORT="${PORT:-29953}"
OUT_DIR="${OUT_DIR:-$HOME/Desktop/spec13_papers100m_e2e/post_pop_os/31_round1_5ep}"
mkdir -p "$OUT_DIR"

pkill -f "bin/mdb server.*--port $PORT" 2>/dev/null || true
sleep 3

LOG="$OUT_DIR/server.log"
env MDB_GNN_PIPELINE_OVERLAP=1 MDB_GNN_REORDER_STRATEGY=external_sort \
    nohup "$MDB" server "$DB" --port "$PORT" --timeout 86400 > "$LOG" 2>&1 &
SERVER_PID=$!
echo "server pid=$SERVER_PID"
sleep 8

cat > "$OUT_DIR/train_query.gql" <<'EOF'
CALL gnn_train(
    "papers100M_paper_und",
    "node_features",
    {
        epochs: 5,
        hiddenDim: 256,
        dropout: 0.2,
        batchSize: 1024,
        learningRate: 0.001,
        randomSeed: 42,
        patience: 999,
        useAsyncPrefetcher: true,
        prefetchQueueSize: 2,
        saveOnBestVal: false,
        saveFinal: false,
        outputDir: "round1_5ep"
    }
) YIELD modelName, ranEpochs, bestValAccuracy, trainSeconds,
        assembleSeconds, forwardSeconds, backwardSeconds,
        l1HitRatio, l2HitRatio, l3Reads, l4Reads
RETURN *
EOF

echo "=== Round 1 5-epoch bench ==="
T0=$(date +%s)
curl -s --max-time 7200 -X POST -H 'Content-Type: application/gql' \
    --data-binary @"$OUT_DIR/train_query.gql" \
    -o "$OUT_DIR/train.csv" \
    -w "HTTP=%{http_code} time=%{time_total}s\n" \
    "http://localhost:$PORT/gql" | tee "$OUT_DIR/train.curl"
T1=$(date +%s)
ELAPSED=$((T1 - T0))

grep -E '\[TrainingLoop\] epoch=' "$LOG" > "$OUT_DIR/epochs.log" || true

kill -15 "$SERVER_PID" 2>/dev/null || true
sleep 5
kill -9 "$SERVER_PID" 2>/dev/null || true

echo
echo "=== Summary ==="
echo "Total: ${ELAPSED}s"
echo
echo "Train CSV:"; cat "$OUT_DIR/train.csv"
echo
echo "Epochs:"; cat "$OUT_DIR/epochs.log"
echo
echo "Baseline (no Round 1): epoch_t=1335s per epoch"
echo "Target: epoch_t ≤ 500s (3× speedup minimum)"
