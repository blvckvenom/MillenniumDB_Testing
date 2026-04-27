#!/usr/bin/env bash
# papers100m_resample_train.sh
#
# Full re-sample → re-materialize → re-feature_store → train pipeline
# para papers100M. Útil para experimentar con fanouts diferentes (e.g., [2,2]
# DiskGNN-style, o [15,10,5] OGB-style).
#
# Reusa la projection existente (papers100M_e2e_opt). NO toca Phase 1 (graph_project).
#
# Usage:
#   ./scripts/papers100m_resample_train.sh [fanouts] [hidden_dim] [epochs] [lr] [dropout] [patience]
#
# Defaults (replicates last successful run):
#   fanouts:    [10, 5]     (2 layers)
#   hidden_dim: 256
#   epochs:     100
#   lr:         0.001
#   dropout:    0.3
#   patience:   20
#
# Examples:
#   # DiskGNN-style fanouts [2,2]:
#   ./scripts/papers100m_resample_train.sh "[2, 2]"
#
#   # OGB-style 3-layer [15,10,5]:
#   ./scripts/papers100m_resample_train.sh "[15, 10, 5]" 256 100 0.001 0.3 20
#
#   # Bigger model:
#   ./scripts/papers100m_resample_train.sh "[10, 5]" 512 200 0.001 0.3 30
#
# Estimated wall clocks (papers100M on celebi 32 GB):
#   sample [2,2]:           ~5-10 min
#   sample [10,5]:          ~16 min
#   sample [15,10,5]:       ~25-40 min
#   materialize (any):      ~70 min
#   feature_store:          ~10-15 min
#   train (≤100 epochs):    ~10-20 min (early-stop common)

set -uo pipefail

# === Parameters ===
FANOUTS="${1:-[10, 5]}"
HIDDEN_DIM="${2:-256}"
EPOCHS="${3:-100}"
LR="${4:-0.001}"
DROPOUT="${5:-0.3}"
PATIENCE="${6:-20}"

# Set USE_EXISTING_SERVER=1 to skip server start/stop:
#   The script will NOT kill prior mdb processes, NOT start a new server,
#   NOT clean gnn_features files (assumed handled by you), NOT run RSS sampler.
#   It only verifies the port answers and runs phases 2-5.
#   Recommended: start the server yourself with tee to also log it:
#     ./build/Release/bin/mdb server data/dbs/gql/papers100M --port 29950 \
#         --threads 16 --browser false --timeout 86400 2>&1 \
#         | tee ~/Desktop/spec13_papers100m_e2e/server_manual_$(date +%Y%m%d_%H%M%S).log
USE_EXISTING_SERVER="${USE_EXISTING_SERVER:-0}"

# === Constants ===
cd "$(dirname "$0")/.."
MDB=./build/Release/bin/mdb
DBPATH=data/dbs/gql/papers100M
PROJ_NAME=papers100M_e2e_opt
PORT=29950

# Sample name encodes the fanouts so different runs don't conflict
FANOUTS_TAG=$(echo "$FANOUTS" | tr -d ' []' | tr ',' '-')
SAMPLE_NAME="papers100M_f${FANOUTS_TAG}_sample"
TS=$(date +%Y%m%d_%H%M%S)
RUN_NAME="papers100m_f${FANOUTS_TAG}_h${HIDDEN_DIM}_${TS}"

LOG_DIR=~/Desktop/spec13_papers100m_e2e
mkdir -p "$LOG_DIR"
LOG="$LOG_DIR/server_${RUN_NAME}.log"
RESULTS="$LOG_DIR/results_${RUN_NAME}.csv"

# === Pre-flight ===
echo "========================================================"
echo "papers100M re-sample → train pipeline"
echo "========================================================"
echo "  fanouts:      $FANOUTS"
echo "  hidden_dim:   $HIDDEN_DIM"
echo "  epochs:       $EPOCHS"
echo "  lr:           $LR"
echo "  dropout:      $DROPOUT"
echo "  patience:     $PATIENCE"
echo "  sample_name:  $SAMPLE_NAME"
echo "  run_name:     $RUN_NAME"
echo "  use_existing_server: $USE_EXISTING_SERVER"
echo "========================================================"

# Verify projection exists
if [ ! -d "$DBPATH/projections/$PROJ_NAME" ]; then
    echo "ERROR: Projection $PROJ_NAME not found at $DBPATH/projections/"
    echo "       Run graph_project first."
    exit 1
fi

# Drop old sample with same name if exists (to allow re-runs)
if [ -d "$DBPATH/samples/$SAMPLE_NAME" ]; then
    echo ""
    echo "Sample $SAMPLE_NAME already exists. Dropping..."
    rm -rf "$DBPATH/samples/$SAMPLE_NAME"
fi

if [ "$USE_EXISTING_SERVER" = "1" ]; then
    # === External server mode ===
    # User started the server manually — we only verify it answers and dispatch
    # phases. No file cleanup (server might have them mmap'd), no kill, no RSS
    # sampler, no exit trap.
    echo ""
    echo "=== USE_EXISTING_SERVER=1: skipping server start ==="
    echo "Probing http://127.0.0.1:$PORT/ ..."
    if ! curl -sSf -o /dev/null --max-time 10 --data-binary "RETURN 1" \
            -H "Accept: text/csv" "http://127.0.0.1:$PORT/"; then
        echo "ERROR: no MillenniumDB server reachable on port $PORT."
        echo ""
        echo "Open the server in another terminal, e.g.:"
        echo "  ./build/Release/bin/mdb server $DBPATH --port $PORT \\"
        echo "      --threads 16 --browser false --timeout 86400 2>&1 \\"
        echo "      | tee $LOG_DIR/server_manual_${RUN_NAME}.log"
        echo ""
        echo "Then re-run with USE_EXISTING_SERVER=1."
        exit 1
    fi
    echo "Server reachable. Phase output (CSV results) is captured in $RESULTS."
    echo "Server-side prints (sample/materialize/feature_store/train) appear in"
    echo "the terminal where you started mdb — pipe through 'tee' to also save them."
else
    # Drop any partial reordered.fmat from previous runs
    rm -f "$DBPATH/gnn_features/node_features_reordered.fmat" 2>/dev/null
    rm -f "$DBPATH/gnn_features/node_features_reordered.rmap" 2>/dev/null
    rm -f "$DBPATH/gnn_features/node_features_gpu_cache.bin" 2>/dev/null
    rm -f "$DBPATH/gnn_features/node_features_cpu_cache.bin" 2>/dev/null
    rm -f "$DBPATH/gnn_features/node_features_store.meta" 2>/dev/null

    # Kill any prior server
    pkill -9 -f "mdb server.*papers100M" 2>/dev/null || true
    sleep 3

    # === Server ===
    export MDB_PROJECTION_SORTER=radix
    export MDB_SORT_BUFFER_MB=4096

    echo ""
    echo "=== $(date +%H:%M:%S) Starting mdb server (port $PORT) ==="
    "$MDB" server "$DBPATH" --port "$PORT" --threads 16 --browser false --timeout 86400 \
        > "$LOG" 2>&1 &
    SRV_PID=$!
    echo "Server PID: $SRV_PID  →  log: $LOG"

    # Wait for ready
    for i in $(seq 1 120); do
        if curl -sSf -o /dev/null --data-binary "RETURN 1" -H "Accept: text/csv" \
            "http://127.0.0.1:$PORT/" 2>/dev/null; then
            echo "Server ready after ${i}s"
            break
        fi
        sleep 1
    done

    # Background RSS sampler (every 30s)
    (
        echo "ts,etime_s,vmrss_mb,vmhwm_mb,threads" > "$LOG_DIR/rss_${RUN_NAME}.csv"
        T0=$(date +%s)
        while kill -0 $SRV_PID 2>/dev/null; do
            T1=$(date +%s)
            ETIME=$((T1 - T0))
            VMRSS=$(awk '/^VmRSS:/ {print int($2/1024)}' /proc/$SRV_PID/status 2>/dev/null || echo 0)
            VMHWM=$(awk '/^VmHWM:/ {print int($2/1024)}' /proc/$SRV_PID/status 2>/dev/null || echo 0)
            TH=$(awk '/^Threads:/ {print $2}' /proc/$SRV_PID/status 2>/dev/null || echo 0)
            echo "$T1,$ETIME,$VMRSS,$VMHWM,$TH" >> "$LOG_DIR/rss_${RUN_NAME}.csv"
            sleep 30
        done
    ) &
    SAMPLER_PID=$!

    # Cleanup on exit
    trap 'kill -TERM $SAMPLER_PID 2>/dev/null; kill -TERM $SRV_PID 2>/dev/null; wait $SRV_PID 2>/dev/null' EXIT
fi

echo "phase,wall_sec,result,details" > "$RESULTS"

query() {
    curl -sS --max-time 86400 -H "Accept: text/csv" --data-binary "$1" \
        "http://127.0.0.1:$PORT/"
}

run_phase() {
    local phase="$1" sql="$2"
    local t0=$(date +%s)
    echo ""
    echo "================================================================"
    echo "=== $(date +%H:%M:%S) Phase: $phase ==="
    echo "================================================================"
    local result
    result=$(query "$sql" 2>&1)
    local t1=$(date +%s)
    local wall=$((t1 - t0))
    local last_line
    last_line=$(echo "$result" | tail -1)
    echo "$result"
    echo ""
    echo "  Phase wall: ${wall}s ($((wall / 60)) min $((wall % 60)) sec)"
    if echo "$result" | grep -qiE "error|exception|empty reply"; then
        echo "  STATUS: FAILED"
        echo "$phase,$wall,FAILED,$last_line" >> "$RESULTS"
        return 1
    fi
    echo "$phase,$wall,OK,$last_line" >> "$RESULTS"
    return 0
}

# ============================================================
# Phase 2: gnn_offline_sample (Camino-D mode — sin L1/L2 caches)
# ============================================================
PHASE2_SQL="CALL gnn_offline_sample('$PROJ_NAME', '$SAMPLE_NAME', $FANOUTS, {
    batchSize: 1024,
    randomSeed: 42,
    usePredefinedSplits: true,
    orientation: 'UNDIRECTED',
    useFourLevelTopologyStore: false,
    useAdjacencyCache: false
}) YIELD sampleName, totalBatches, trainBatches, validationBatches, testBatches, uniqueNodes
RETURN *"

run_phase "gnn_offline_sample" "$PHASE2_SQL" || { echo "STOP: sample failed"; exit 1; }

# ============================================================
# Phase 3: gnn_materialize_batches (REORDER: 0 — skip 56 GB feature copy)
# ============================================================
PHASE3_SQL="CALL gnn_materialize_batches('$SAMPLE_NAME', 'node_features', {reorder: 0, force: 1})
YIELD sampleName, totalBatches, reordered RETURN *"

run_phase "gnn_materialize_batches" "$PHASE3_SQL" || { echo "STOP: materialize failed"; exit 1; }

# ============================================================
# Phase 4: gnn_build_feature_store
# ============================================================
PHASE4_SQL="CALL gnn_build_feature_store('$SAMPLE_NAME', 'node_features', {
    gpu_budget_mb: 12000,
    cpu_budget_mb: 12000,
    reorder: 0,
    force: 1
}) YIELD sampleName, l1Nodes, l2Nodes, l3Nodes, l4Nodes, buildTimeMs
RETURN *"

run_phase "gnn_build_feature_store" "$PHASE4_SQL" || { echo "STOP: feature_store failed"; exit 1; }

# ============================================================
# Phase 5: gnn_train (with provided hyperparameters)
# ============================================================
PHASE5_SQL="CALL gnn_train('$SAMPLE_NAME', 'node_features', {
    model: 'graphsage',
    hiddenDim: $HIDDEN_DIM,
    epochs: $EPOCHS,
    lr: $LR,
    dropout: $DROPOUT,
    patience: $PATIENCE,
    randomSeed: 42,
    exportEmbeddings: true,
    outputDir: '$RUN_NAME',
    inferenceBatchSize: 4096,
    saveOnBestVal: true,
    saveFinal: true
}) YIELD modelName, ranEpochs, didConverge, bestValAccuracy, testAccuracy, trainSeconds, bestCheckpointPath, finalCheckpointPath
RETURN *"

run_phase "gnn_train" "$PHASE5_SQL" || { echo "STOP: train failed"; exit 1; }

# ============================================================
# Done
# ============================================================
echo ""
echo "================================================================"
echo "=== $(date +%H:%M:%S) PIPELINE COMPLETE ==="
echo "================================================================"
column -t -s, "$RESULTS"
echo ""
if [ "$USE_EXISTING_SERVER" = "1" ]; then
    echo "(USE_EXISTING_SERVER=1: peak RSS not tracked here — see your server terminal.)"
else
    echo "Server peak RSS: $(awk -F',' 'NR>1 && $4>max {max=$4} END {print max}' "$LOG_DIR/rss_${RUN_NAME}.csv") MB"
fi
echo ""
echo "Train artifacts:"
ls -la "$DBPATH/projections/$PROJ_NAME/gnn_output/$RUN_NAME/" 2>/dev/null
echo ""
if [ "$USE_EXISTING_SERVER" != "1" ]; then
    echo "Server log: $LOG"
fi
echo "Results CSV: $RESULTS"
echo ""
echo "DONE"
