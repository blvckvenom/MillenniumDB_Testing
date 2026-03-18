#!/bin/bash
# =============================================================================
# GNN Pipeline E2E Test
#
# Tests the complete pipeline: import → project → sample → materialize → verify
# Uses the Cora dataset (2708 nodes, 1433 dims) with real embeddings.
#
# Usage:
#   ./tests/gnn_e2e_test.sh [build_dir]
#
# Requirements:
#   - Build with ENABLE_GNN=ON
#   - Cora dataset in data/example/gql/cora/
#   - Python3 with numpy
# =============================================================================
set -euo pipefail

RED='\033[0;31m'
GREEN='\033[0;32m'
BLUE='\033[0;34m'
NC='\033[0m'

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
BUILD_DIR="${1:-$PROJECT_DIR/build/Release}"
MDB="$BUILD_DIR/bin/mdb"
PORT=11234  # Use high port to avoid conflicts
DB_DIR=$(mktemp -d /tmp/gnn_e2e_test_XXXXXX)

PASSED=0
FAILED=0
TOTAL=0

pass() { PASSED=$((PASSED + 1)); TOTAL=$((TOTAL + 1)); printf "${GREEN}  PASS${NC} | %s\n" "$1"; }
fail() { FAILED=$((FAILED + 1)); TOTAL=$((TOTAL + 1)); printf "${RED}  FAIL${NC} | %s\n" "$1"; }
info() { printf "${BLUE}  >>>>${NC}  %s\n" "$1"; }

cleanup() {
    # Kill server if running
    if [ -n "${SERVER_PID:-}" ]; then
        kill "$SERVER_PID" 2>/dev/null || true
        wait "$SERVER_PID" 2>/dev/null || true
    fi
    # Remove temp database
    rm -rf "$DB_DIR"
}
trap cleanup EXIT

# =============================================================================
# Check prerequisites
# =============================================================================
printf "${BLUE}GNN Pipeline E2E Test${NC}\n"
echo "=============================="

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
    echo "ERROR: Port $PORT already in use. Kill the process or change PORT."
    exit 1
fi

# =============================================================================
# Step 1: Import Cora with embeddings
# =============================================================================
info "Step 1: Import Cora with --with-tensors"

IMPORT_OUT=$("$MDB" import "$CORA_GQL" "$DB_DIR" --with-tensors "$CORA_NPY" 2>&1)

if echo "$IMPORT_OUT" | grep -q "GNN Features:.*1 registered"; then
    pass "Import: GNN features registered"
else
    fail "Import: GNN features not registered"
    echo "$IMPORT_OUT"
    exit 1
fi

if echo "$IMPORT_OUT" | grep -q "Nodes:.*2708"; then
    pass "Import: 2708 nodes"
else
    fail "Import: unexpected node count"
fi

if echo "$IMPORT_OUT" | grep -q "Edges:.*5429"; then
    pass "Import: 5429 edges"
else
    fail "Import: unexpected edge count"
fi

# Verify files exist
if [ -f "$DB_DIR/gnn_features/node_features.fmat" ] && [ -f "$DB_DIR/gnn_features/node_features.rmap" ]; then
    pass "Import: .fmat and .rmap files created"
else
    fail "Import: missing .fmat or .rmap"
    exit 1
fi

# =============================================================================
# Step 2: Start server
# =============================================================================
info "Step 2: Start server on port $PORT"

"$MDB" server "$DB_DIR" --port "$PORT" --browser false &>/dev/null &
SERVER_PID=$!
sleep 2

if ss -tlnp | grep -q ":$PORT"; then
    pass "Server listening on port $PORT"
else
    fail "Server failed to start"
    exit 1
fi

query() {
    curl -s -X POST "http://localhost:$PORT" -d "$1"
}

# =============================================================================
# Step 3: Create projection
# =============================================================================
info "Step 3: Create projection"

PROJ_OUT=$(query "CALL graph_project('e2e_proj', 'Paper', 'CITES') YIELD graphName, nodeCount, relationshipCount RETURN graphName, nodeCount, relationshipCount")

if echo "$PROJ_OUT" | grep -q "2708"; then
    pass "Projection: 2708 nodes"
else
    fail "Projection: unexpected output: $PROJ_OUT"
fi

# =============================================================================
# Step 4: Offline sampling
# =============================================================================
info "Step 4: Offline sampling"

SAMPLE_OUT=$(query "CALL gnn_offline_sample('e2e_proj', 'e2e_sample', [10, 5], {batchSize: 256, randomSeed: 42}) YIELD sampleName, totalBatches, uniqueNodes RETURN sampleName, totalBatches, uniqueNodes")

TOTAL_BATCHES=$(echo "$SAMPLE_OUT" | tail -1 | cut -d',' -f2)
UNIQUE_NODES=$(echo "$SAMPLE_OUT" | tail -1 | cut -d',' -f3)

if [ "$UNIQUE_NODES" = "2708" ]; then
    pass "Sampling: all 2708 nodes covered"
else
    fail "Sampling: expected 2708 unique nodes, got $UNIQUE_NODES"
fi

if [ -n "$TOTAL_BATCHES" ] && [ "$TOTAL_BATCHES" -gt 0 ] 2>/dev/null; then
    pass "Sampling: $TOTAL_BATCHES batches created"
else
    fail "Sampling: invalid batch count: $TOTAL_BATCHES"
fi

# =============================================================================
# Step 5: Materialize batches (L3 + L4)
# =============================================================================
info "Step 5: Materialize batches (reorder + pack)"

MAT_OUT=$(query "CALL gnn_materialize_batches('e2e_sample', 'node_features', {reorder: 1, numHashes: 2}) YIELD sampleName, totalBatches, reordered, reorderTimeMs, packTimeMs, totalTimeMs, packedDir RETURN sampleName, totalBatches, reordered, reorderTimeMs, packTimeMs, totalTimeMs, packedDir")

MAT_BATCHES=$(echo "$MAT_OUT" | tail -1 | cut -d',' -f2)
MAT_REORDERED=$(echo "$MAT_OUT" | tail -1 | cut -d',' -f3)

if [ "$MAT_BATCHES" = "$TOTAL_BATCHES" ]; then
    pass "Materialize: $MAT_BATCHES batches match sampling"
else
    fail "Materialize: batch count mismatch (sampling=$TOTAL_BATCHES, materialize=$MAT_BATCHES)"
fi

if [ "$MAT_REORDERED" = "true" ]; then
    pass "Materialize: L3 reordering performed"
else
    fail "Materialize: reordering not performed"
fi

# Check output files
if [ -f "$DB_DIR/gnn_features/node_features_reordered.fmat" ]; then
    pass "Materialize: reordered .fmat created"
else
    fail "Materialize: missing reordered .fmat"
fi

if [ -f "$DB_DIR/gnn_features/node_features_reordered.rmap" ]; then
    pass "Materialize: reordered .rmap created"
else
    fail "Materialize: missing reordered .rmap"
fi

PACKED_DIR="$DB_DIR/samples/e2e_sample/packed"
PACKED_COUNT=$(ls "$PACKED_DIR"/batch_*.bin 2>/dev/null | wc -l)
if [ "$PACKED_COUNT" = "$TOTAL_BATCHES" ]; then
    pass "Materialize: $PACKED_COUNT packed batch files"
else
    fail "Materialize: expected $TOTAL_BATCHES packed files, got $PACKED_COUNT"
fi

# =============================================================================
# Step 5b: Materialize without reorder (L4 only)
# =============================================================================
info "Step 5b: Materialize without reorder (separate sample)"

SAMPLE_NOREORD=$(query "CALL gnn_offline_sample('e2e_proj', 'e2e_noreord', [10, 5], {batchSize: 256, randomSeed: 42}) YIELD sampleName, totalBatches RETURN sampleName, totalBatches")
NR_BATCHES=$(echo "$SAMPLE_NOREORD" | tail -1 | cut -d',' -f2)

MAT_NR=$(query "CALL gnn_materialize_batches('e2e_noreord', 'node_features', {reorder: 0}) YIELD totalBatches, reordered RETURN totalBatches, reordered")
NR_REORDERED=$(echo "$MAT_NR" | tail -1 | cut -d',' -f2)

if [ "$NR_REORDERED" = "false" ]; then
    pass "No-reorder: reordered=false"
else
    fail "No-reorder: expected reordered=false, got $NR_REORDERED"
fi

# Verify NO reordered files were created for this feature (they exist from Step 5,
# but the no-reorder path should NOT create new ones — it reuses existing)
NR_PACKED="$DB_DIR/samples/e2e_noreord/packed"
NR_COUNT=$(ls "$NR_PACKED"/batch_*.bin 2>/dev/null | wc -l)
if [ "$NR_COUNT" = "$NR_BATCHES" ]; then
    pass "No-reorder: $NR_COUNT packed batch files"
else
    fail "No-reorder: expected $NR_BATCHES packed files, got $NR_COUNT"
fi

# =============================================================================
# Step 5c: Error path — already materialized
# =============================================================================
info "Step 5c: Error path — already materialized"

ERR_OUT=$(query "CALL gnn_materialize_batches('e2e_sample', 'node_features') YIELD totalBatches RETURN totalBatches" 2>&1)
if echo "$ERR_OUT" | grep -qi "already exist\|force"; then
    pass "Error path: already materialized mentions force"
else
    fail "Error path: unexpected output: $ERR_OUT"
fi

# =============================================================================
# Step 5d: Error path — non-existent sample
# =============================================================================
info "Step 5d: Error path — non-existent sample"

ERR_OUT2=$(query "CALL gnn_materialize_batches('nonexistent_sample', 'node_features') YIELD totalBatches RETURN totalBatches" 2>&1)
if echo "$ERR_OUT2" | grep -qi "not found"; then
    pass "Error path: non-existent sample reports not found"
else
    fail "Error path: unexpected output: $ERR_OUT2"
fi

# =============================================================================
# Step 6: Verify features (S4 Critical Property)
# =============================================================================
info "Step 6: Verify packed features match originals (S4 property)"

# Stop server — Python verification reads files directly
kill "$SERVER_PID" 2>/dev/null || true
wait "$SERVER_PID" 2>/dev/null || true
unset SERVER_PID

VERIFY_OUT=$(python3 << PYEOF
import struct, os, sys, numpy as np

DB = "$DB_DIR"
FMAT_H = 64
RMAP_H = 16

# Load original features
orig = np.load("$CORA_NPY")
N, D = orig.shape

# Load original RowMapping
with open(f"{DB}/gnn_features/node_features.rmap", "rb") as f:
    f.seek(RMAP_H)
    orig_oids = np.frombuffer(f.read(N * 8), dtype=np.uint64)
oid_to_row = {oid: i for i, oid in enumerate(orig_oids)}

# Load reordered RowMapping
with open(f"{DB}/gnn_features/node_features_reordered.rmap", "rb") as f:
    f.seek(RMAP_H)
    reord_oids = np.frombuffer(f.read(N * 8), dtype=np.uint64)

# Load reordered FeatureMatrix
with open(f"{DB}/gnn_features/node_features_reordered.fmat", "rb") as f:
    f.seek(FMAT_H)
    reord_fm = np.frombuffer(f.read(N * D * 4), dtype=np.float32).reshape(N, D)

errors = 0

# CHECK 1: RowMapping bijection
orig_set = set(orig_oids.tolist())
reord_set = set(reord_oids.tolist())
if orig_set != reord_set or len(reord_set) != N:
    print("FAIL|bijection|OID sets differ")
    errors += 1
else:
    print(f"PASS|bijection|{N} unique OIDs match")

# CHECK 2: RowMapping coherence (rm[i] ↔ fm[i])
coherence_err = 0
for i in range(N):
    oid = reord_oids[i]
    orig_row = oid_to_row[oid]
    if not np.array_equal(reord_fm[i], orig[orig_row]):
        coherence_err += 1
if coherence_err == 0:
    print(f"PASS|coherence|{N} rows verified")
else:
    print(f"FAIL|coherence|{coherence_err} mismatches")
    errors += 1

# CHECK 3: Non-trivial reorder
if np.array_equal(reord_oids, orig_oids):
    print("FAIL|reorder|identity permutation (nothing changed)")
    errors += 1
else:
    changed = int(np.sum(reord_oids != orig_oids))
    print(f"PASS|reorder|{changed}/{N} rows changed position")

# CHECK 4: S4 property — packed features match originals
sample_dir = f"{DB}/samples/e2e_sample"
packed_dir = f"{sample_dir}/packed"

with open(f"{sample_dir}/batches.idx", "rb") as f:
    f.read(4+4)
    num_batches = struct.unpack("<Q", f.read(8))[0]
    entries = [(struct.unpack("<Q", f.read(8))[0], struct.unpack("<Q", f.read(8))[0])
               for _ in range(num_batches)]

total_verified = 0
s4_errors = 0

with open(f"{sample_dir}/batches.dat", "rb") as bf:
    for bid in range(num_batches):
        bf.seek(entries[bid][0])
        bf.read(4+4+8+1)  # magic,version,batch_id,split
        nl = struct.unpack("<Q", bf.read(8))[0]
        for _ in range(nl):
            ls = struct.unpack("<Q", bf.read(8))[0]
            bf.read(ls * 8)
        nel = struct.unpack("<Q", bf.read(8))[0]
        for _ in range(nel):
            ns = struct.unpack("<Q", bf.read(8))[0]; bf.read(ns*4)
            nd = struct.unpack("<Q", bf.read(8))[0]; bf.read(nd*4)
            ne = struct.unpack("<Q", bf.read(8))[0]; bf.read(ne*8)
        nu = struct.unpack("<Q", bf.read(8))[0]
        batch_oids = [struct.unpack("<Q", bf.read(8))[0] for _ in range(nu)]

        with open(f"{packed_dir}/batch_{bid:06d}.bin", "rb") as pf:
            pf.read(4+4)
            pn = struct.unpack("<Q", pf.read(8))[0]
            pd = struct.unpack("<Q", pf.read(8))[0]
            pf.read(1+7)
            packed = np.frombuffer(pf.read(pn*pd*4), dtype=np.float32).reshape(pn, pd)

        for i, oid in enumerate(batch_oids):
            orow = oid_to_row.get(oid)
            if orow is not None and i < pn:
                if not np.array_equal(packed[i], orig[orow]):
                    s4_errors += 1
                total_verified += 1

if s4_errors == 0:
    print(f"PASS|s4_property|{total_verified} node-features across {num_batches} batches")
else:
    print(f"FAIL|s4_property|{s4_errors} mismatches in {total_verified} checks")
    errors += 1

# CHECK 5: File sizes
orig_fmat_size = os.path.getsize(f"{DB}/gnn_features/node_features.fmat")
reord_fmat_size = os.path.getsize(f"{DB}/gnn_features/node_features_reordered.fmat")
if orig_fmat_size == reord_fmat_size:
    print(f"PASS|file_sizes|{orig_fmat_size} bytes (original == reordered)")
else:
    print(f"FAIL|file_sizes|original={orig_fmat_size} != reordered={reord_fmat_size}")
    errors += 1

# CHECK 6: Spread reduction (locality improvement)
reord_oid_to_row = {oid: i for i, oid in enumerate(reord_oids)}
orig_spreads = []
reord_spreads = []
with open(f"{sample_dir}/batches.dat", "rb") as bf:
    for bid in range(num_batches):
        bf.seek(entries[bid][0])
        bf.read(4+4+8+1)
        nl = struct.unpack("<Q", bf.read(8))[0]
        for _ in range(nl):
            ls = struct.unpack("<Q", bf.read(8))[0]; bf.read(ls*8)
        nel = struct.unpack("<Q", bf.read(8))[0]
        for _ in range(nel):
            ns = struct.unpack("<Q", bf.read(8))[0]; bf.read(ns*4)
            nd = struct.unpack("<Q", bf.read(8))[0]; bf.read(nd*4)
            ne = struct.unpack("<Q", bf.read(8))[0]; bf.read(ne*8)
        nu = struct.unpack("<Q", bf.read(8))[0]
        boids = [struct.unpack("<Q", bf.read(8))[0] for _ in range(nu)]
        orows = sorted([oid_to_row[o] for o in boids if o in oid_to_row])
        rrows = sorted([reord_oid_to_row[o] for o in boids if o in reord_oid_to_row])
        if orows: orig_spreads.append(orows[-1]-orows[0])
        if rrows: reord_spreads.append(rrows[-1]-rrows[0])

avg_orig = np.mean(orig_spreads)
avg_reord = np.mean(reord_spreads)
if avg_reord < avg_orig:
    reduction = (1 - avg_reord/avg_orig)*100
    print(f"PASS|locality|{reduction:.1f}% spread reduction ({avg_orig:.0f} -> {avg_reord:.0f})")
else:
    print(f"FAIL|locality|no improvement ({avg_orig:.0f} -> {avg_reord:.0f})")
    errors += 1

sys.exit(1 if errors > 0 else 0)
PYEOF
)

# Parse Python output
while IFS='|' read -r status check detail; do
    if [ "$status" = "PASS" ]; then
        pass "Verify $check: $detail"
    else
        fail "Verify $check: $detail"
    fi
done <<< "$VERIFY_OUT"

# =============================================================================
# Summary
# =============================================================================
echo ""
echo "=============================="
if [ "$FAILED" -eq 0 ]; then
    printf "${GREEN}ALL $TOTAL CHECKS PASSED${NC}\n"
    exit 0
else
    printf "${RED}$FAILED/$TOTAL CHECKS FAILED${NC}\n"
    exit 1
fi
