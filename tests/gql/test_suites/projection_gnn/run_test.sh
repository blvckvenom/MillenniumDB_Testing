#!/bin/bash
# =============================================================================
# Integration tests for graph_project GNN extension
#
# Tests that graph_project produces gnn_meta.bin, labels.bin, and splits.bin
# when includeFeatures, labelProperty, and splitProperty are provided.
#
# Uses the Cora dataset (2708 nodes, 1433 dims, 7 classes, 4 split types).
#
# Usage:
#   ./tests/gql/test_suites/projection_gnn/run_test.sh [build_dir]
#
# Requirements:
#   - Build with ENABLE_GNN=ON
#   - Cora dataset in data/example/gql/cora/
# =============================================================================
set -euo pipefail

RED='\033[0;31m'
GREEN='\033[0;32m'
BLUE='\033[0;34m'
NC='\033[0m'

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/../../../.." && pwd)"
BUILD_DIR="${1:-$PROJECT_DIR/build/Debug}"
MDB="$BUILD_DIR/bin/mdb"
PORT=19876
DB_DIR=$(mktemp -d /tmp/mdb_test_proj_gnn_XXXXXX)

PASSED=0
FAILED=0
TOTAL=0

pass() { PASSED=$((PASSED + 1)); TOTAL=$((TOTAL + 1)); printf "${GREEN}  PASS${NC} | %s\n" "$1"; }
fail() { FAILED=$((FAILED + 1)); TOTAL=$((TOTAL + 1)); printf "${RED}  FAIL${NC} | %s\n" "$1"; }
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
printf "${BLUE}graph_project GNN Extension Integration Tests${NC}\n"
echo "================================================"

if [ ! -f "$MDB" ]; then
    echo "ERROR: mdb binary not found at $MDB"
    echo "Build with: cmake -D ENABLE_GNN=ON ... && cmake --build ..."
    exit 1
fi

CORA_GQL="$PROJECT_DIR/data/example/gql/cora/cora.gql"
CORA_NPY="$PROJECT_DIR/data/example/gql/cora/cora_features.npy"

if [ ! -f "$CORA_GQL" ] || [ ! -f "$CORA_NPY" ]; then
    echo "ERROR: Cora dataset not found at data/example/gql/cora/"
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
    echo "$IMPORT_OUT"
fi

if echo "$IMPORT_OUT" | grep -q "5429"; then
    pass "Import: 5429 edges imported"
else
    fail "Import: unexpected edge count"
    echo "$IMPORT_OUT"
fi

# =============================================================================
# Step 2: Start server
# =============================================================================
info "Step 2: Start server on port $PORT"

"$MDB" server "$DB_DIR" --port "$PORT" --browser false &>/dev/null &
SERVER_PID=$!
sleep 2

if ss -tlnp 2>/dev/null | grep -q ":$PORT"; then
    pass "Server listening on port $PORT"
else
    fail "Server failed to start"
    exit 1
fi

query() {
    curl -s -H "Accept:text/csv" -X POST "http://localhost:$PORT" -d "$1"
}

# =============================================================================
# Test 1: graph_project with GNN extension fields
# =============================================================================
info "Test 1: graph_project with includeFeatures, labelProperty, splitProperty"

PROJ_RESULT=$(query "CALL graph_project('cora_gnn', 'Paper', 'CITES', {orientation: 'UNDIRECTED', includeFeatures: 'node_features', labelProperty: 'label', splitProperty: 'split'})
YIELD graphName, nodeCount, relationshipCount, featureDim, numClasses
RETURN graphName, nodeCount, relationshipCount, featureDim, numClasses")

echo "  Project result: $PROJ_RESULT"

# Parse the CSV (second line is data)
DATA_LINE=$(echo "$PROJ_RESULT" | tail -1)

# Extract fields (CSV: graphName,nodeCount,relationshipCount,featureDim,numClasses)
NODE_COUNT=$(echo "$DATA_LINE" | cut -d',' -f2)
REL_COUNT=$(echo "$DATA_LINE" | cut -d',' -f3)
FEATURE_DIM=$(echo "$DATA_LINE" | cut -d',' -f4)
NUM_CLASSES=$(echo "$DATA_LINE" | cut -d',' -f5)

if [ "$NODE_COUNT" = "2708" ]; then
    pass "YIELD nodeCount = 2708"
else
    fail "YIELD nodeCount: expected 2708, got $NODE_COUNT"
fi

if [ "$FEATURE_DIM" = "1433" ]; then
    pass "YIELD featureDim = 1433"
else
    fail "YIELD featureDim: expected 1433, got $FEATURE_DIM"
fi

if [ "$NUM_CLASSES" = "7" ]; then
    pass "YIELD numClasses = 7"
else
    fail "YIELD numClasses: expected 7, got $NUM_CLASSES"
fi

# UNDIRECTED doubles the stored edge count
if [ -n "$REL_COUNT" ] && [ "$REL_COUNT" -gt 5000 ] 2>/dev/null; then
    pass "YIELD relationshipCount = $REL_COUNT (UNDIRECTED)"
else
    fail "YIELD relationshipCount: expected >5000, got $REL_COUNT"
fi

# =============================================================================
# Test 2: Verify gnn_meta.bin exists
# =============================================================================
info "Test 2: Verify GNN output files exist"

PROJ_DIR="$DB_DIR/projections/cora_gnn"

if [ -f "$PROJ_DIR/gnn_meta.bin" ]; then
    pass "gnn_meta.bin exists"
else
    fail "gnn_meta.bin missing at $PROJ_DIR/"
fi

if [ -f "$PROJ_DIR/labels.bin" ]; then
    pass "labels.bin exists"
else
    fail "labels.bin missing at $PROJ_DIR/"
fi

if [ -f "$PROJ_DIR/splits.bin" ]; then
    pass "splits.bin exists"
else
    fail "splits.bin missing at $PROJ_DIR/"
fi

# =============================================================================
# Test 3: Verify labels.bin file format and size
# =============================================================================
info "Test 3: Verify labels.bin format"

# Format: magic(8) + version(4) + reserved(4) + num_nodes(8) + num_classes(8) + int64[N]
# Header = 8 + 4 + 4 + 8 + 8 = 32 bytes
# Data   = 2708 * 8 = 21664 bytes
# Total  = 32 + 21664 = 21696 bytes
LABELS_SIZE=$(stat -c%s "$PROJ_DIR/labels.bin")
EXPECTED_LABELS=$((32 + 2708 * 8))

if [ "$LABELS_SIZE" -eq "$EXPECTED_LABELS" ]; then
    pass "labels.bin size = $EXPECTED_LABELS bytes"
else
    fail "labels.bin size: expected $EXPECTED_LABELS, got $LABELS_SIZE"
fi

# Check magic bytes (GNNL)
LABELS_MAGIC=$(xxd -p -l 4 "$PROJ_DIR/labels.bin")
if [ "$LABELS_MAGIC" = "474e4e4c" ]; then
    pass "labels.bin magic = GNNL"
else
    fail "labels.bin magic: expected 474e4e4c (GNNL), got $LABELS_MAGIC"
fi

# =============================================================================
# Test 4: Verify splits.bin file format and size
# =============================================================================
info "Test 4: Verify splits.bin format"

# Format: magic(8) + version(4) + reserved(4) + num_nodes(8) + uint8[N]
# Header = 8 + 4 + 4 + 8 = 24 bytes
# Data   = 2708 * 1 = 2708 bytes
# Total  = 24 + 2708 = 2732 bytes
SPLITS_SIZE=$(stat -c%s "$PROJ_DIR/splits.bin")
EXPECTED_SPLITS=$((24 + 2708))

if [ "$SPLITS_SIZE" -eq "$EXPECTED_SPLITS" ]; then
    pass "splits.bin size = $EXPECTED_SPLITS bytes"
else
    fail "splits.bin size: expected $EXPECTED_SPLITS, got $SPLITS_SIZE"
fi

# Check magic bytes (GNNS)
SPLITS_MAGIC=$(xxd -p -l 4 "$PROJ_DIR/splits.bin")
if [ "$SPLITS_MAGIC" = "474e4e53" ]; then
    pass "splits.bin magic = GNNS"
else
    fail "splits.bin magic: expected 474e4e53 (GNNS), got $SPLITS_MAGIC"
fi

# =============================================================================
# Test 5: Verify labels.bin content via Python
# =============================================================================
info "Test 5: Verify labels.bin content (all values in [0..6])"

LABELS_CHECK=$(python3 << PYEOF
import struct

path = "$PROJ_DIR/labels.bin"
with open(path, "rb") as f:
    magic = f.read(8)
    version = struct.unpack("<I", f.read(4))[0]
    _reserved = f.read(4)
    num_nodes = struct.unpack("<Q", f.read(8))[0]
    num_classes = struct.unpack("<Q", f.read(8))[0]
    labels = struct.unpack(f"<{num_nodes}q", f.read(num_nodes * 8))

errors = []
if num_nodes != 2708:
    errors.append(f"num_nodes={num_nodes}, expected 2708")
if num_classes != 7:
    errors.append(f"num_classes={num_classes}, expected 7")

min_label = min(labels)
max_label = max(labels)
unique_labels = len(set(labels))
if min_label < 0 or max_label > 6:
    errors.append(f"label range [{min_label}, {max_label}], expected [0, 6]")
if unique_labels != 7:
    errors.append(f"unique_labels={unique_labels}, expected 7")

if errors:
    print("FAIL|" + "; ".join(errors))
else:
    print(f"PASS|{num_nodes} labels, {unique_labels} classes, range [{min_label}, {max_label}]")
PYEOF
)

if echo "$LABELS_CHECK" | grep -q "^PASS"; then
    DETAIL=$(echo "$LABELS_CHECK" | cut -d'|' -f2)
    pass "labels.bin content: $DETAIL"
else
    DETAIL=$(echo "$LABELS_CHECK" | cut -d'|' -f2)
    fail "labels.bin content: $DETAIL"
fi

# =============================================================================
# Test 6: Verify splits.bin content via Python
# =============================================================================
info "Test 6: Verify splits.bin content (train/val/test/unlabeled)"

SPLITS_CHECK=$(python3 << PYEOF
import struct
from collections import Counter

path = "$PROJ_DIR/splits.bin"
with open(path, "rb") as f:
    magic = f.read(8)
    version = struct.unpack("<I", f.read(4))[0]
    _reserved = f.read(4)
    num_nodes = struct.unpack("<Q", f.read(8))[0]
    splits = struct.unpack(f"<{num_nodes}B", f.read(num_nodes))

errors = []
if num_nodes != 2708:
    errors.append(f"num_nodes={num_nodes}, expected 2708")

counts = Counter(splits)
# Cora splits: train=140, val=500, test=1000, unlabeled=1068
# Split encoding: 0=train, 1=validation, 2=test, 3=unlabeled (or similar)
# We just verify all values are in a small range and sum to 2708
unique_splits = sorted(counts.keys())
total = sum(counts.values())

if total != 2708:
    errors.append(f"total={total}, expected 2708")
if max(unique_splits) > 10:
    errors.append(f"max split value={max(unique_splits)}, expected small")

# Build distribution string
dist = ", ".join(f"split_{k}={v}" for k, v in sorted(counts.items()))

if errors:
    print("FAIL|" + "; ".join(errors))
else:
    print(f"PASS|{num_nodes} nodes, {len(unique_splits)} split types: {dist}")
PYEOF
)

if echo "$SPLITS_CHECK" | grep -q "^PASS"; then
    DETAIL=$(echo "$SPLITS_CHECK" | cut -d'|' -f2)
    pass "splits.bin content: $DETAIL"
else
    DETAIL=$(echo "$SPLITS_CHECK" | cut -d'|' -f2)
    fail "splits.bin content: $DETAIL"
fi

# =============================================================================
# Test 7: Verify gnn_meta.bin content via Python
# =============================================================================
info "Test 7: Verify gnn_meta.bin content"

META_CHECK=$(python3 << PYEOF
import struct

path = "$PROJ_DIR/gnn_meta.bin"
with open(path, "rb") as f:
    data = f.read()

# GnnMeta format: magic(8) + version(4) + feature_dim(4) + num_nodes(8) +
#   num_classes(8) + has_labels(1) + has_splits(1) + reserved(2) +
#   feature_name_len(4) + feature_name(variable)
# Minimum size: 8 + 4 + 4 + 8 + 8 + 1 + 1 + 2 + 4 = 40 bytes
errors = []

if len(data) < 40:
    print(f"FAIL|file too small: {len(data)} bytes, minimum 40")
    sys.exit(0)

magic = data[0:8]
version = struct.unpack_from("<I", data, 8)[0]
feature_dim = struct.unpack_from("<I", data, 12)[0]
num_nodes = struct.unpack_from("<Q", data, 16)[0]
num_classes = struct.unpack_from("<Q", data, 24)[0]
has_labels = data[32]
has_splits = data[33]
# reserved at offset 34 (2 bytes)
name_len = struct.unpack_from("<I", data, 36)[0]
feature_name = data[40:40+name_len].decode("utf-8", errors="replace") if name_len > 0 else ""

if feature_dim != 1433:
    errors.append(f"feature_dim={feature_dim}, expected 1433")
if num_nodes != 2708:
    errors.append(f"num_nodes={num_nodes}, expected 2708")
if num_classes != 7:
    errors.append(f"num_classes={num_classes}, expected 7")
if has_labels != 1:
    errors.append(f"has_labels={has_labels}, expected 1")
if has_splits != 1:
    errors.append(f"has_splits={has_splits}, expected 1")
if feature_name != "node_features":
    errors.append(f"feature_name='{feature_name}', expected 'node_features'")

if errors:
    print("FAIL|" + "; ".join(errors))
else:
    print(f"PASS|dim={feature_dim}, nodes={num_nodes}, classes={num_classes}, labels={has_labels}, splits={has_splits}, name='{feature_name}'")
PYEOF
)

if echo "$META_CHECK" | grep -q "^PASS"; then
    DETAIL=$(echo "$META_CHECK" | cut -d'|' -f2)
    pass "gnn_meta.bin content: $DETAIL"
else
    DETAIL=$(echo "$META_CHECK" | cut -d'|' -f2)
    fail "gnn_meta.bin content: $DETAIL"
fi

# =============================================================================
# Test 8: Verify projection topology with USE query
# =============================================================================
info "Test 8: Verify projection topology via USE query"

TOPO_RESULT=$(query "USE cora_gnn
MATCH (n)-[e]->(m)
RETURN count(*) AS edge_count")

echo "  Topology result: $TOPO_RESULT"

EDGE_COUNT=$(echo "$TOPO_RESULT" | tail -1)
# UNDIRECTED projection stores both directions, so edges >= 5429*2
if [ -n "$EDGE_COUNT" ] && [ "$EDGE_COUNT" -ge 10858 ] 2>/dev/null; then
    pass "Topology query: $EDGE_COUNT directed edges (UNDIRECTED doubles)"
else
    fail "Topology query: expected >= 10858, got $EDGE_COUNT"
fi

# =============================================================================
# Test 9: Verify node count via USE query
# =============================================================================
info "Test 9: Verify node count via USE query"

NODE_RESULT=$(query "USE cora_gnn
MATCH (n)
RETURN count(*) AS node_count")

echo "  Node result: $NODE_RESULT"

RESULT_NODES=$(echo "$NODE_RESULT" | tail -1)
if [ "$RESULT_NODES" = "2708" ]; then
    pass "Topology node count: 2708"
else
    fail "Topology node count: expected 2708, got $RESULT_NODES"
fi

# =============================================================================
# Test 10: graph_project without GNN fields (baseline)
# =============================================================================
info "Test 10: graph_project without GNN fields (featureDim=0, numClasses=0)"

BASELINE_RESULT=$(query "CALL graph_project('cora_baseline', 'Paper', 'CITES')
YIELD graphName, nodeCount, featureDim, numClasses
RETURN graphName, nodeCount, featureDim, numClasses")

echo "  Baseline result: $BASELINE_RESULT"

BASELINE_DATA=$(echo "$BASELINE_RESULT" | tail -1)
BASELINE_DIM=$(echo "$BASELINE_DATA" | cut -d',' -f3)
BASELINE_CLS=$(echo "$BASELINE_DATA" | cut -d',' -f4)

if [ "$BASELINE_DIM" = "0" ]; then
    pass "Baseline featureDim = 0 (no GNN config)"
else
    fail "Baseline featureDim: expected 0, got $BASELINE_DIM"
fi

if [ "$BASELINE_CLS" = "0" ]; then
    pass "Baseline numClasses = 0 (no GNN config)"
else
    fail "Baseline numClasses: expected 0, got $BASELINE_CLS"
fi

# Verify no GNN files in baseline projection
BASELINE_DIR="$DB_DIR/projections/cora_baseline"
if [ ! -f "$BASELINE_DIR/gnn_meta.bin" ]; then
    pass "Baseline: no gnn_meta.bin (expected)"
else
    fail "Baseline: gnn_meta.bin should not exist without GNN config"
fi

# =============================================================================
# Test 11: graph_project with only includeFeatures (no labels/splits)
# =============================================================================
info "Test 11: graph_project with includeFeatures only"

FEAT_ONLY_RESULT=$(query "CALL graph_project('cora_feat_only', 'Paper', 'CITES', {includeFeatures: 'node_features'})
YIELD graphName, nodeCount, featureDim, numClasses
RETURN graphName, nodeCount, featureDim, numClasses")

echo "  Features-only result: $FEAT_ONLY_RESULT"

FEAT_DATA=$(echo "$FEAT_ONLY_RESULT" | tail -1)
FEAT_DIM=$(echo "$FEAT_DATA" | cut -d',' -f3)
FEAT_CLS=$(echo "$FEAT_DATA" | cut -d',' -f4)

if [ "$FEAT_DIM" = "1433" ]; then
    pass "Features-only featureDim = 1433"
else
    fail "Features-only featureDim: expected 1433, got $FEAT_DIM"
fi

if [ "$FEAT_CLS" = "0" ]; then
    pass "Features-only numClasses = 0 (no labelProperty)"
else
    fail "Features-only numClasses: expected 0, got $FEAT_CLS"
fi

FEAT_ONLY_DIR="$DB_DIR/projections/cora_feat_only"
if [ -f "$FEAT_ONLY_DIR/gnn_meta.bin" ]; then
    pass "Features-only: gnn_meta.bin exists"
else
    fail "Features-only: gnn_meta.bin missing"
fi

if [ ! -f "$FEAT_ONLY_DIR/labels.bin" ]; then
    pass "Features-only: no labels.bin (expected, no labelProperty)"
else
    fail "Features-only: labels.bin should not exist without labelProperty"
fi

if [ ! -f "$FEAT_ONLY_DIR/splits.bin" ]; then
    pass "Features-only: no splits.bin (expected, no splitProperty)"
else
    fail "Features-only: splits.bin should not exist without splitProperty"
fi

# =============================================================================
# Test 12: Error path -- invalid feature name
# =============================================================================
info "Test 12: Error path -- invalid feature name"

ERR_RESULT=$(query "CALL graph_project('cora_bad_feat', 'Paper', 'CITES', {includeFeatures: 'nonexistent_feature'})
YIELD graphName
RETURN graphName" 2>&1)

if echo "$ERR_RESULT" | grep -qi "not found"; then
    pass "Error path: invalid feature name rejected"
else
    fail "Error path: expected 'not found' error, got: $ERR_RESULT"
fi

# =============================================================================
# Summary
# =============================================================================
echo ""
echo "================================================"
if [ "$FAILED" -eq 0 ]; then
    printf "${GREEN}ALL $TOTAL CHECKS PASSED${NC}\n"
    exit 0
else
    printf "${RED}$FAILED/$TOTAL CHECKS FAILED${NC}\n"
    exit 1
fi
