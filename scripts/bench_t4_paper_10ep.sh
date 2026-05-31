#!/usr/bin/env bash
# T4 FINAL: paper_und 10 epochs paper config.
# Gated on T2 showing positive val_acc trend. Total wall-clock ~2:45 hr.
# Reuses existing sample + feature_store. Diagnostic per paper §7.1 fig 4(c):
# val_acc should reach ~0.65 by epoch 10.
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DB="$REPO_DIR/data/dbs/gql/papers100M"
MDB="$REPO_DIR/build/Release/bin/mdb"
PORT="${PORT:-29964}"
OUT_DIR="${OUT_DIR:-$HOME/Desktop/spec13_papers100m_e2e/post_pop_os/38_t4_paper_und_10ep}"
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
        epochs: 10,
        hiddenDim: 256,
        dropout: 0.2,
        batchSize: 1024,
        lr: 0.003,
        randomSeed: 42,
        patience: 999,
        useAsyncPrefetcher: true,
        prefetchQueueSize: 2,
        saveOnBestVal: true,
        saveFinal: true,
        outputDir: "paper_10ep"
    }
) YIELD modelName, ranEpochs, didConverge, bestValAccuracy, testAccuracy,
        trainSeconds, assembleSeconds, forwardSeconds, backwardSeconds,
        l1HitRatio, l2HitRatio, l3Reads, l4Reads,
        bestCheckpointPath, finalCheckpointPath
RETURN *
EOF

echo "=== T4 FINAL: gnn_train 10 epochs paper_und lr=0.003 ==="
T0=$(date +%s)
curl -s --max-time 18000 -X POST -H 'Content-Type: application/gql' \
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
echo "=== T4 FINAL Summary ==="
echo "Wall-clock: ${ELAPSED}s = $((ELAPSED / 60))m $((ELAPSED % 60))s"
echo
cat "$OUT_DIR/epochs.log"
echo
echo "Train CSV:"; cat "$OUT_DIR/train.csv"
echo
echo "Paper §7.1 fig 4(c) reference: val_acc reaches ~0.65 by epoch 10"
echo "Paper §7.1 final val_acc / test_acc: 65.91% at epoch 50"
