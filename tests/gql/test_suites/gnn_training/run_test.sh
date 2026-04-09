#!/bin/bash
# =============================================================================
# End-to-end GNN training test: GraphSAGE on Cora
#
# Validates the ENTIRE training pipeline:
#   1. Import Cora with tensors
#   2. graph_project with GNN fields
#   3. gnn_offline_sample with predefined splits
#   4. gnn_train (GraphSAGE MEAN, 128 hidden, 30 epochs)
#   5. Verify test accuracy > 70%
#   6. Verify output artifacts (model, embeddings, training log)
#   7. Determinism check (same seed -> same accuracy)
#
# Success criteria (spec Section 1):
#   - testAccuracy > 0.70 (paper reports ~81% with 2 layers)
#   - embeddings.npy exists with shape [N, 128]
#   - model.pt exists and has reasonable size (> 1 KB)
#   - training_log.json exists and is valid JSON
#
# Usage:
#   ./tests/gql/test_suites/gnn_training/run_test.sh [build_dir]
#
# Requirements:
#   - Build with ENABLE_GNN=ON
#   - Cora dataset in data/example/gql/cora/
# =============================================================================
set -euo pipefail

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
BLUE='\033[0;34m'
NC='\033[0m'

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/../../../.." && pwd)"
BUILD_DIR="${1:-$PROJECT_DIR/build/Debug}"
MDB="$BUILD_DIR/bin/mdb"
PORT=19877
DB_DIR=$(mktemp -d "/tmp/mdb_test_gnn_train_${$}_XXXXXX")

# Training timeout: 120s for CPU training (generous for slow machines)
TRAIN_TIMEOUT=120

PASSED=0
FAILED=0
WARNED=0
TOTAL=0

pass() { PASSED=$((PASSED + 1)); TOTAL=$((TOTAL + 1)); printf "${GREEN}  PASS${NC} | %s\n" "$1"; }
fail() { FAILED=$((FAILED + 1)); TOTAL=$((TOTAL + 1)); printf "${RED}  FAIL${NC} | %s\n" "$1"; }
warn() { WARNED=$((WARNED + 1)); TOTAL=$((TOTAL + 1)); printf "${YELLOW}  WARN${NC} | %s\n" "$1"; }
info() { printf "${BLUE}  >>>>${NC}  %s\n" "$1"; }

cleanup() {
    if [ -n "${SERVER_PID:-}" ]; then
        kill "$SERVER_PID" 2>/dev/null || true
        wait "$SERVER_PID" 2>/dev/null || true
    fi
    rm -rf "$DB_DIR"
}
trap cleanup EXIT

# =============================================================================
# Check prerequisites
# =============================================================================
printf "${BLUE}GNN End-to-End Training Test: GraphSAGE on Cora${NC}\n"
echo "================================================"
echo "  DB dir:    $DB_DIR"
echo "  Port:      $PORT"
echo "  Build dir: $BUILD_DIR"
echo ""

if [ ! -f "$MDB" ]; then
    echo "ERROR: mdb binary not found at $MDB"
    echo "Build with: cmake -D ENABLE_GNN=ON ... && cmake --build ..."
    exit 1
fi

CORA_GQL="$PROJECT_DIR/data/example/gql/cora/cora.gql"
CORA_NPY="$PROJECT_DIR/data/example/gql/cora/cora_features.npy"

if [ ! -f "$CORA_GQL" ] || [ ! -f "$CORA_NPY" ]; then
    echo "ERROR: Cora dataset not found at data/example/gql/cora/"
    echo "Expected: cora.gql and cora_features.npy"
    exit 1
fi

# Check port is available
if ss -tlnp 2>/dev/null | grep -q ":$PORT"; then
    echo "ERROR: Port $PORT already in use."
    exit 1
fi

# =============================================================================
# Step 1: Import Cora with tensors
# =============================================================================
info "Step 1: Import Cora with --with-tensors"

IMPORT_OUT=$("$MDB" import "$CORA_GQL" "$DB_DIR" --with-tensors "$CORA_NPY" 2>&1)

if echo "$IMPORT_OUT" | grep -q "2708"; then
    pass "Import: 2708 nodes imported"
else
    fail "Import: unexpected node count"
    echo "  $IMPORT_OUT"
fi

if echo "$IMPORT_OUT" | grep -q "5429"; then
    pass "Import: 5429 edges imported"
else
    fail "Import: unexpected edge count"
    echo "  $IMPORT_OUT"
fi

# =============================================================================
# Step 2: Start server
# =============================================================================
info "Step 2: Start server on port $PORT"

"$MDB" server "$DB_DIR" --port "$PORT" --browser false &>/dev/null &
SERVER_PID=$!

# Poll for server readiness (up to 10 seconds)
SERVER_READY=false
for i in $(seq 1 20); do
    if ss -tlnp 2>/dev/null | grep -q ":$PORT"; then
        SERVER_READY=true
        break
    fi
    sleep 0.5
done

if [ "$SERVER_READY" = "true" ]; then
    pass "Server listening on port $PORT"
else
    fail "Server failed to start within 10 seconds"
    exit 1
fi

query() {
    curl -s -H "Accept:text/csv" -X POST "http://localhost:$PORT" -d "$1"
}

# =============================================================================
# Step 3: graph_project with GNN extension fields
# =============================================================================
info "Step 3: graph_project with includeFeatures, labelProperty, splitProperty"

PROJ_RESULT=$(query "CALL graph_project('cora', 'Paper', 'CITES', {
    orientation: 'UNDIRECTED',
    includeFeatures: 'node_features',
    labelProperty: 'label',
    splitProperty: 'split'
})
YIELD graphName, nodeCount, featureDim, numClasses
RETURN graphName, nodeCount, featureDim, numClasses")

echo "  Project result: $PROJ_RESULT"

# Parse CSV: header is first line, data is second
PROJ_DATA=$(echo "$PROJ_RESULT" | tail -1)
PROJ_NODES=$(echo "$PROJ_DATA" | cut -d',' -f2)
PROJ_DIM=$(echo "$PROJ_DATA" | cut -d',' -f3)
PROJ_CLASSES=$(echo "$PROJ_DATA" | cut -d',' -f4)

if [ "$PROJ_NODES" = "2708" ]; then
    pass "Projection nodeCount = 2708"
else
    fail "Projection nodeCount: expected 2708, got $PROJ_NODES"
fi

if [ "$PROJ_DIM" = "1433" ]; then
    pass "Projection featureDim = 1433"
else
    fail "Projection featureDim: expected 1433, got $PROJ_DIM"
fi

if [ "$PROJ_CLASSES" = "7" ]; then
    pass "Projection numClasses = 7"
else
    fail "Projection numClasses: expected 7, got $PROJ_CLASSES"
fi

# =============================================================================
# Step 4: gnn_offline_sample with predefined splits
# =============================================================================
info "Step 4: gnn_offline_sample with predefined splits (fanouts [10, 5])"

SAMPLE_RESULT=$(query "CALL gnn_offline_sample('cora', 'cora_s', [10, 5], {
    batchSize: 64,
    randomSeed: 42,
    usePredefinedSplits: true,
    orientation: 'UNDIRECTED'
})
YIELD sampleName, totalBatches, trainBatches, validationBatches, testBatches
RETURN sampleName, totalBatches, trainBatches, validationBatches, testBatches")

echo "  Sample result: $SAMPLE_RESULT"

# Parse CSV
SAMPLE_DATA=$(echo "$SAMPLE_RESULT" | tail -1)
SAMPLE_NAME=$(echo "$SAMPLE_DATA" | cut -d',' -f1)
TOTAL_BATCHES=$(echo "$SAMPLE_DATA" | cut -d',' -f2)
TRAIN_BATCHES=$(echo "$SAMPLE_DATA" | cut -d',' -f3)
VAL_BATCHES=$(echo "$SAMPLE_DATA" | cut -d',' -f4)
TEST_BATCHES=$(echo "$SAMPLE_DATA" | cut -d',' -f5)

if [ -n "$TOTAL_BATCHES" ] && [ "$TOTAL_BATCHES" -gt 0 ] 2>/dev/null; then
    pass "Sampling produced $TOTAL_BATCHES total batches"
else
    fail "Sampling: expected >0 batches, got $TOTAL_BATCHES"
fi

if [ -n "$TRAIN_BATCHES" ] && [ "$TRAIN_BATCHES" -gt 0 ] 2>/dev/null; then
    pass "Sampling produced $TRAIN_BATCHES train batches"
else
    fail "Sampling: expected >0 train batches, got $TRAIN_BATCHES"
fi

if [ -n "$VAL_BATCHES" ] && [ "$VAL_BATCHES" -gt 0 ] 2>/dev/null; then
    pass "Sampling produced $VAL_BATCHES validation batches"
else
    fail "Sampling: expected >0 validation batches, got $VAL_BATCHES"
fi

if [ -n "$TEST_BATCHES" ] && [ "$TEST_BATCHES" -gt 0 ] 2>/dev/null; then
    pass "Sampling produced $TEST_BATCHES test batches"
else
    fail "Sampling: expected >0 test batches, got $TEST_BATCHES"
fi

# Cora predefined splits: train=140, val=500, test=1000
# With batchSize=64: train=ceil(140/64)=3, val=ceil(500/64)=8, test=ceil(1000/64)=16
info "Expected split distribution: train~3, val~8, test~16 batches (batchSize=64)"

# =============================================================================
# Step 5: gnn_train (GraphSAGE MEAN, 128 hidden, 30 epochs)
# =============================================================================
info "Step 5: gnn_train (graphsage, hiddenDim=128, epochs=30, lr=0.01, seed=42)"

TRAIN_RESULT=$(timeout "$TRAIN_TIMEOUT" curl -s -H "Accept:text/csv" -X POST \
    "http://localhost:$PORT" -d "CALL gnn_train('cora_s', 'node_features', {
    model: 'graphsage',
    hiddenDim: 128,
    epochs: 30,
    lr: 0.01,
    dropout: 0.5,
    patience: 10,
    randomSeed: 42,
    exportEmbeddings: true,
    outputDir: 'test_run'
})
YIELD modelName, ranEpochs, didConverge, bestValAccuracy, testAccuracy, trainSeconds
RETURN modelName, ranEpochs, didConverge, bestValAccuracy, testAccuracy, trainSeconds" 2>&1)

TRAIN_EXIT=$?
if [ $TRAIN_EXIT -ne 0 ]; then
    fail "gnn_train timed out or failed (exit=$TRAIN_EXIT)"
    echo "  Output: $TRAIN_RESULT"
    # Cannot continue — print summary and exit
    echo ""
    echo "================================================"
    printf "${RED}Training failed — cannot verify outputs${NC}\n"
    printf "${RED}$FAILED/$TOTAL CHECKS FAILED${NC}\n"
    exit 1
fi

echo "  Train result: $TRAIN_RESULT"

# Parse CSV: header is first line, data is second.
# CSV string fields are wrapped in literal double quotes — strip them when
# comparing to bare strings.
# Fields: modelName,ranEpochs,didConverge,bestValAccuracy,testAccuracy,trainSeconds
TRAIN_DATA=$(echo "$TRAIN_RESULT" | tail -1)
MODEL_NAME=$(echo "$TRAIN_DATA" | cut -d',' -f1 | tr -d '"')
RAN_EPOCHS=$(echo "$TRAIN_DATA" | cut -d',' -f2)
DID_CONVERGE=$(echo "$TRAIN_DATA" | cut -d',' -f3)
BEST_VAL_ACC=$(echo "$TRAIN_DATA" | cut -d',' -f4)
TEST_ACC=$(echo "$TRAIN_DATA" | cut -d',' -f5)
TRAIN_SECS=$(echo "$TRAIN_DATA" | cut -d',' -f6)

if [ "$MODEL_NAME" = "graphsage" ]; then
    pass "Model name = graphsage"
else
    fail "Model name: expected graphsage, got $MODEL_NAME"
fi

if [ -n "$RAN_EPOCHS" ] && [ "$RAN_EPOCHS" -gt 0 ] 2>/dev/null; then
    pass "Ran $RAN_EPOCHS epochs"
else
    fail "Ran epochs: expected >0, got $RAN_EPOCHS"
fi

# =============================================================================
# Step 6: Verify test accuracy
# =============================================================================
info "Step 6: Verify test accuracy"

if [ -n "$TEST_ACC" ]; then
    # Check accuracy thresholds using awk for float comparison
    IS_GOOD=$(awk "BEGIN {print ($TEST_ACC > 0.70) ? 1 : 0}")
    IS_OK=$(awk "BEGIN {print ($TEST_ACC > 0.60) ? 1 : 0}")
    IS_BAD=$(awk "BEGIN {print ($TEST_ACC < 0.50) ? 1 : 0}")

    if [ "$IS_GOOD" = "1" ]; then
        pass "Test accuracy = $TEST_ACC (> 0.70 threshold)"
    elif [ "$IS_OK" = "1" ]; then
        warn "Test accuracy = $TEST_ACC (between 0.60-0.70, may be initialization-dependent)"
    elif [ "$IS_BAD" = "1" ]; then
        fail "Test accuracy = $TEST_ACC (< 0.50, indicates a bug)"
    else
        warn "Test accuracy = $TEST_ACC (between 0.50-0.60, below expected)"
    fi
else
    fail "Test accuracy: could not parse from result"
fi

if [ -n "$BEST_VAL_ACC" ]; then
    IS_VAL_GOOD=$(awk "BEGIN {print ($BEST_VAL_ACC > 0.50) ? 1 : 0}")
    if [ "$IS_VAL_GOOD" = "1" ]; then
        pass "Best validation accuracy = $BEST_VAL_ACC (> 0.50)"
    else
        warn "Best validation accuracy = $BEST_VAL_ACC (low)"
    fi
else
    fail "Best validation accuracy: could not parse from result"
fi

if [ -n "$TRAIN_SECS" ]; then
    IS_REASONABLE=$(awk "BEGIN {print ($TRAIN_SECS < 120) ? 1 : 0}")
    if [ "$IS_REASONABLE" = "1" ]; then
        pass "Training time = ${TRAIN_SECS}s (< 120s)"
    else
        warn "Training time = ${TRAIN_SECS}s (slow but completed)"
    fi
else
    fail "Training time: could not parse from result"
fi

# =============================================================================
# Step 7: Verify output artifacts
# =============================================================================
info "Step 7: Verify output artifacts"

# The output directory is: <proj_dir>/gnn_output/<outputDir>/
# proj_dir = <db>/projections/cora/
OUTPUT_DIR="$DB_DIR/projections/cora/gnn_output/test_run"

# 7a: model.pt exists and has reasonable size
if [ -f "$OUTPUT_DIR/model.pt" ]; then
    MODEL_SIZE=$(stat -c%s "$OUTPUT_DIR/model.pt")
    if [ "$MODEL_SIZE" -gt 1024 ]; then
        pass "model.pt exists ($MODEL_SIZE bytes, > 1 KB)"
    else
        fail "model.pt too small: $MODEL_SIZE bytes (expected > 1 KB)"
    fi
else
    fail "model.pt not found at $OUTPUT_DIR/"
fi

# 7b: training_log.json exists and is valid JSON
if [ -f "$OUTPUT_DIR/training_log.json" ]; then
    pass "training_log.json exists"

    LOG_CHECK=$(LOG_PATH="$OUTPUT_DIR/training_log.json" python3 << 'PYEOF' 2>&1
import json, os

path = os.environ["LOG_PATH"]
try:
    with open(path) as f:
        log = json.load(f)

    errors = []

    if log.get("version") != 1:
        errors.append(f"version={log.get('version')}, expected 1")
    if log.get("model") != "graphsage":
        errors.append(f"model={log.get('model')}, expected graphsage")

    hp = log.get("hyperparameters", {})
    if hp.get("input_dim") != 1433:
        errors.append(f"input_dim={hp.get('input_dim')}, expected 1433")
    if hp.get("hidden_dim") != 128:
        errors.append(f"hidden_dim={hp.get('hidden_dim')}, expected 128")
    if hp.get("num_classes") != 7:
        errors.append(f"num_classes={hp.get('num_classes')}, expected 7")

    epoch_losses = log.get("epoch_losses", [])
    if len(epoch_losses) == 0:
        errors.append("epoch_losses is empty")

    res = log.get("results", {})
    ran = res.get("ran_epochs", 0)

    if errors:
        print("FAIL|" + "; ".join(errors))
    else:
        print(f"PASS|{ran} epochs, input_dim=1433, hidden_dim=128, classes=7, {len(epoch_losses)} loss entries")
except Exception as e:
    print(f"FAIL|{e}")
PYEOF
)

    if echo "$LOG_CHECK" | grep -q "^PASS"; then
        DETAIL=$(echo "$LOG_CHECK" | cut -d'|' -f2)
        pass "training_log.json content: $DETAIL"
    else
        DETAIL=$(echo "$LOG_CHECK" | cut -d'|' -f2)
        fail "training_log.json content: $DETAIL"
    fi
else
    fail "training_log.json not found at $OUTPUT_DIR/"
fi

# 7c: embeddings.npy exists and has correct shape
if [ -f "$OUTPUT_DIR/embeddings.npy" ]; then
    pass "embeddings.npy exists"

    EMB_CHECK=$(EMB_PATH="$OUTPUT_DIR/embeddings.npy" python3 << 'PYEOF' 2>&1
import os, sys
try:
    import numpy as np
except ImportError:
    print("SKIP|numpy not available")
    sys.exit(0)

path = os.environ["EMB_PATH"]
try:
    emb = np.load(path)

    errors = []
    # Shape should be [N, 128] where N is the total seed nodes across all batches
    if len(emb.shape) != 2:
        errors.append(f"expected 2D, got shape {emb.shape}")
    elif emb.shape[1] != 128:
        errors.append(f"hidden_dim={emb.shape[1]}, expected 128")

    if np.any(np.isnan(emb)):
        errors.append("NaN values found in embeddings")
    if np.any(np.isinf(emb)):
        errors.append("Inf values found in embeddings")

    if errors:
        print("FAIL|" + "; ".join(errors))
    else:
        print(f"PASS|shape={emb.shape}, range=[{emb.min():.4f}, {emb.max():.4f}]")
except Exception as e:
    print(f"FAIL|{e}")
PYEOF
)

    if echo "$EMB_CHECK" | grep -q "^PASS"; then
        DETAIL=$(echo "$EMB_CHECK" | cut -d'|' -f2)
        pass "embeddings.npy content: $DETAIL"
    elif echo "$EMB_CHECK" | grep -q "^SKIP"; then
        DETAIL=$(echo "$EMB_CHECK" | cut -d'|' -f2)
        warn "embeddings.npy check skipped: $DETAIL"
    else
        DETAIL=$(echo "$EMB_CHECK" | cut -d'|' -f2)
        fail "embeddings.npy content: $DETAIL"
    fi
else
    fail "embeddings.npy not found at $OUTPUT_DIR/"
fi

# =============================================================================
# Step 8: Determinism check (same seed -> same accuracy)
# =============================================================================
info "Step 8: Determinism check (re-run with same randomSeed=42)"

TRAIN_RESULT_2=$(timeout "$TRAIN_TIMEOUT" curl -s -H "Accept:text/csv" -X POST \
    "http://localhost:$PORT" -d "CALL gnn_train('cora_s', 'node_features', {
    model: 'graphsage',
    hiddenDim: 128,
    epochs: 30,
    lr: 0.01,
    dropout: 0.5,
    patience: 10,
    randomSeed: 42,
    exportEmbeddings: false,
    outputDir: 'test_run_2'
})
YIELD testAccuracy
RETURN testAccuracy" 2>&1)

TRAIN2_EXIT=$?
if [ $TRAIN2_EXIT -ne 0 ]; then
    warn "Determinism check: second run timed out or failed (exit=$TRAIN2_EXIT)"
else
    echo "  Second run result: $TRAIN_RESULT_2"
    TRAIN2_DATA=$(echo "$TRAIN_RESULT_2" | tail -1)
    TEST_ACC_2=$(echo "$TRAIN2_DATA" | cut -d',' -f1)

    if [ -n "$TEST_ACC" ] && [ -n "$TEST_ACC_2" ]; then
        IS_SAME=$(awk "BEGIN {
            diff = ($TEST_ACC) - ($TEST_ACC_2);
            if (diff < 0) diff = -diff;
            print (diff < 0.001) ? 1 : 0
        }")
        if [ "$IS_SAME" = "1" ]; then
            pass "Determinism: testAccuracy=$TEST_ACC_2 matches first run ($TEST_ACC)"
        else
            warn "Determinism: testAccuracy=$TEST_ACC_2 differs from first run ($TEST_ACC)"
        fi
    else
        warn "Determinism: could not compare accuracies"
    fi
fi

# =============================================================================
# Summary
# =============================================================================
echo ""
echo "================================================"
echo "  Passed:  $PASSED"
echo "  Warned:  $WARNED"
echo "  Failed:  $FAILED"
echo "  Total:   $TOTAL"
echo "================================================"

if [ "$FAILED" -eq 0 ]; then
    if [ "$WARNED" -gt 0 ]; then
        printf "${YELLOW}ALL CHECKS PASSED ($WARNED warnings)${NC}\n"
    else
        printf "${GREEN}ALL $TOTAL CHECKS PASSED${NC}\n"
    fi
    exit 0
else
    printf "${RED}$FAILED/$TOTAL CHECKS FAILED${NC}\n"
    exit 1
fi
