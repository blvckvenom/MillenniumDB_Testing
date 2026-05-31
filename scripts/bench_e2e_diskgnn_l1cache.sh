#!/usr/bin/env bash
# bench_e2e_diskgnn_l1cache.sh — Rebuild feature store with bigger GPU+CPU
# caches, then run E2E DiskGNN replication.
#
# Strategy: our active-set refactor freed ~13 GB VRAM that the paper's old
# model couldn't use. Push gpu_budget_mb to 10000 (vs paper's 0 / our prior 2)
# to absorb more L4 disk reads. Push cpu_budget_mb to 10000 (vs paper's 5290)
# since we have 30 GB host RAM.
#
# Phases:
#   T0: feature_store rebuild (force_caches=true; preserve reordered.fmat)
#   T1: gnn_train 50 epochs paper config
#   T2: gnn_predict from best_model checkpoint

set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DB="${DB:-$REPO_DIR/data/dbs/gql/papers100M}"
MDB="${MDB:-$REPO_DIR/build/Release/bin/mdb}"
PORT="${PORT:-29952}"
OUT_DIR="${OUT_DIR:-$HOME/Desktop/spec13_papers100m_e2e/post_pop_os/30_e2e_l1cache10g}"
mkdir -p "$OUT_DIR"

# Kill any prior server on PORT
pkill -f "bin/mdb server.*--port $PORT" 2>/dev/null || true
sleep 3

# Launch server with D-config env (best Fix #21+#22)
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
# T0: gnn_build_feature_store — force rebuild caches with new budgets
# ------------------------------------------------------------------
cat > "$OUT_DIR/rebuild_query.gql" <<'EOF'
CALL gnn_build_feature_store(
    "papers100M_paper_und",
    "node_features",
    {
        gpu_budget_mb:      10000,
        cpu_budget_mb:      5000,
        force:              true,
        force_reorder:      false,
        force_packed_slim:  false
    }
) YIELD * RETURN *
EOF

echo
echo "=== T0: feature_store rebuild (gpu=10GB, cpu=10GB, preserve reordered+packed_slim) ==="
T0_START=$(date +%s)
curl -s --max-time 86400 -X POST -H 'Content-Type: application/gql' \
    --data-binary @"$OUT_DIR/rebuild_query.gql" \
    -o "$OUT_DIR/rebuild.csv" \
    -w "T0 HTTP=%{http_code} time=%{time_total}s\n" \
    "http://localhost:$PORT/gql" | tee "$OUT_DIR/rebuild.curl"
T0_END=$(date +%s)
T0_ELAPSED=$((T0_END - T0_START))
echo "T0 wall-clock: ${T0_ELAPSED}s"

# ------------------------------------------------------------------
# T1: gnn_train 50 epochs paper config
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
        outputDir:        "diskgnn_e2e_l1cache10g"
    }
) YIELD modelName, ranEpochs, didConverge, bestValAccuracy, testAccuracy,
        trainSeconds, assembleSeconds, forwardSeconds, backwardSeconds,
        l1HitRatio, l2HitRatio, l3Reads, l4Reads, totalBytesDisk,
        bestCheckpointPath, finalCheckpointPath
RETURN *
EOF

echo
echo "=== T1: gnn_train (50 epochs, SAGE-3L-h256, paper config, L1=10GB) ==="
T1_START=$(date +%s)
curl -s --max-time 86400 -X POST -H 'Content-Type: application/gql' \
    --data-binary @"$OUT_DIR/train_query.gql" \
    -o "$OUT_DIR/train.csv" \
    -w "T1 HTTP=%{http_code} time=%{time_total}s\n" \
    "http://localhost:$PORT/gql" | tee "$OUT_DIR/train.curl"
T1_END=$(date +%s)
T1_ELAPSED=$((T1_END - T1_START))
echo "T1 wall-clock: ${T1_ELAPSED}s"

# Capture per-epoch summaries
grep -E '\[TrainingLoop\] epoch=' "$LOG" | tail -100 > "$OUT_DIR/train_epochs.log" || true

# ------------------------------------------------------------------
# T2: gnn_predict from best_model checkpoint
# ------------------------------------------------------------------
cat > "$OUT_DIR/predict_query.gql" <<'EOF'
CALL gnn_predict(
    "papers100M_paper_und",
    "node_features",
    "diskgnn_e2e_l1cache10g/checkpoints/best_model"
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
echo "=== E2E (L1=10GB CPU=10GB) Summary ==="
echo "T0 rebuild_cache : ${T0_ELAPSED}s ($((T0_ELAPSED / 60))m $((T0_ELAPSED % 60))s)"
echo "T1 gnn_train     : ${T1_ELAPSED}s ($((T1_ELAPSED / 60))m $((T1_ELAPSED % 60))s)"
echo "T2 gnn_predict   : ${T2_ELAPSED}s"
echo "Total            : $((T0_ELAPSED + T1_ELAPSED + T2_ELAPSED))s"
echo
echo "Train CSV:";   cat "$OUT_DIR/train.csv"
echo
echo "Predict CSV:"; cat "$OUT_DIR/predict.csv"
echo
echo "Paper target: 65.91% test accuracy / 1.09 hr"
echo "Outputs:      $OUT_DIR"
