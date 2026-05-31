#!/usr/bin/env bash
# Test lr=0.003 (DGL default) on existing paper_und sample.
# Hypothesis: lr=0.001 was too conservative, val_acc stuck at 0.0776.
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DB="$REPO_DIR/data/dbs/gql/papers100M"
MDB="$REPO_DIR/build/Release/bin/mdb"
PORT="${PORT:-29955}"
OUT_DIR="${OUT_DIR:-$HOME/Desktop/spec13_papers100m_e2e/post_pop_os/34_lr003_5ep}"
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
        lr: 0.003,
        randomSeed: 42,
        patience: 999,
        useAsyncPrefetcher: true,
        prefetchQueueSize: 2,
        saveOnBestVal: false,
        saveFinal: false,
        outputDir: "lr003_5ep"
    }
) YIELD modelName, ranEpochs, bestValAccuracy, testAccuracy,
        trainSeconds, assembleSeconds, forwardSeconds, backwardSeconds,
        l1HitRatio, l2HitRatio
RETURN *
EOF

echo "=== Train 5 epochs lr=0.003 on existing paper_und sample ==="
T0=$(date +%s)
curl -s --max-time 14400 -X POST -H 'Content-Type: application/gql' \
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
echo "Wall-clock: ${ELAPSED}s"
echo
cat "$OUT_DIR/epochs.log"
echo
echo "Train CSV:"; cat "$OUT_DIR/train.csv"
echo
echo "Compare to lr=0.001 baseline:"
echo "  epoch 1: 1067.8s loss=4.3579 val_acc=0.0776"
echo "  epoch 5: 992.2s  loss=4.2683 val_acc=0.0776"
echo "Target: val_acc > 0.1 by epoch 5 → confirms lr=0.003 helps"
