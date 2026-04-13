#!/bin/bash
# =============================================================================
# GNN Pipeline E2E Test
#
# Tests the complete pipeline: import → project → sample → materialize → feature store → verify
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
# Step 3: Create projection (with GNN extension)
#
# Passes includeFeatures/labelProperty/splitProperty so graph_project also
# writes gnn_meta.bin, labels.bin, splits.bin into the projection directory.
# These files are required by gnn_offline_sample(usePredefinedSplits=true)
# and by gnn_train.
# =============================================================================
info "Step 3: Create projection (with GNN config: features+labels+splits)"

PROJ_OUT=$(query "CALL graph_project('e2e_proj', 'Paper', 'CITES', {includeFeatures: 'node_features', labelProperty: 'label', splitProperty: 'split'}) YIELD graphName, nodeCount, relationshipCount, featureDim, numClasses RETURN graphName, nodeCount, relationshipCount, featureDim, numClasses")

if echo "$PROJ_OUT" | grep -q "2708"; then
    pass "Projection: 2708 nodes"
else
    fail "Projection: unexpected output: $PROJ_OUT"
fi

if echo "$PROJ_OUT" | grep -q ",1433,"; then
    pass "Projection: featureDim = 1433"
else
    fail "Projection: featureDim != 1433: $PROJ_OUT"
fi

if echo "$PROJ_OUT" | grep -qE ",7$"; then
    pass "Projection: numClasses = 7"
else
    fail "Projection: numClasses != 7: $PROJ_OUT"
fi

# Verify GNN binaries written by graph_project
PROJ_DIR="$DB_DIR/projections/e2e_proj"
for f in gnn_meta.bin labels.bin splits.bin; do
    if [ -f "$PROJ_DIR/$f" ]; then
        pass "Projection: $f generated"
    else
        fail "Projection: missing $f at $PROJ_DIR"
    fi
done

# =============================================================================
# Step 4: Offline sampling (Planetoid splits + UNDIRECTED)
#
# usePredefinedSplits=true makes the engine read splits.bin and produce
# train/val/test batches that match the Planetoid 140/500/1000 partition.
# orientation=UNDIRECTED is the standard convention for Cora-style citation
# networks (PyG, DGL, OGB) — matches paper Shchur 2018 et al. setup.
# =============================================================================
info "Step 4: Offline sampling (Planetoid splits, UNDIRECTED)"

SAMPLE_OUT=$(query "CALL gnn_offline_sample('e2e_proj', 'e2e_sample', [10, 5], {batchSize: 256, randomSeed: 42, usePredefinedSplits: true, orientation: 'UNDIRECTED'}) YIELD sampleName, totalBatches, trainBatches, validationBatches, testBatches, uniqueNodes RETURN sampleName, totalBatches, trainBatches, validationBatches, testBatches, uniqueNodes")

TOTAL_BATCHES=$(echo "$SAMPLE_OUT" | tail -1 | cut -d',' -f2)
TRAIN_BATCHES=$(echo "$SAMPLE_OUT" | tail -1 | cut -d',' -f3)
VAL_BATCHES=$(echo "$SAMPLE_OUT" | tail -1 | cut -d',' -f4)
TEST_BATCHES=$(echo "$SAMPLE_OUT" | tail -1 | cut -d',' -f5)
UNIQUE_NODES=$(echo "$SAMPLE_OUT" | tail -1 | cut -d',' -f6)

# With Planetoid splits (140 train / 500 val / 1000 test) and batchSize=256:
#   - 1 train batch  (140 < 256)
#   - 2 val batches  (500 ≈ 256+244)
#   - 4 test batches (1000 ≈ 4*256)
#   - 7 total
if [ "$TOTAL_BATCHES" = "7" ]; then
    pass "Sampling: 7 total batches (Planetoid)"
else
    fail "Sampling: expected 7 total, got $TOTAL_BATCHES"
fi
if [ "$TRAIN_BATCHES" = "1" ]; then
    pass "Sampling: 1 train batch (140 nodes)"
else
    fail "Sampling: expected 1 train batch, got $TRAIN_BATCHES"
fi
if [ "$VAL_BATCHES" = "2" ]; then
    pass "Sampling: 2 validation batches (500 nodes)"
else
    fail "Sampling: expected 2 val batches, got $VAL_BATCHES"
fi
if [ "$TEST_BATCHES" = "4" ]; then
    pass "Sampling: 4 test batches (1000 nodes)"
else
    fail "Sampling: expected 4 test batches, got $TEST_BATCHES"
fi
if [ -n "$UNIQUE_NODES" ] && [ "$UNIQUE_NODES" -gt 2000 ] 2>/dev/null; then
    pass "Sampling: $UNIQUE_NODES unique nodes covered (2-hop expansion)"
else
    fail "Sampling: unexpected unique node count: $UNIQUE_NODES"
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
# Step 5e: Build feature store (FourLevelStore)
# =============================================================================
info "Step 5e: Build feature store (FourLevelStore)"

# Build with no GPU (gpu_budget_mb=0) and 100MB CPU budget
FS_OUT=$(query "CALL gnn_build_feature_store('e2e_sample', 'node_features', {gpu_budget_mb: 0, cpu_budget_mb: 100, reorder: 1, force: 0}) YIELD sampleName, featureName, l1Nodes, l2Nodes, l3Nodes, l4Nodes, gpuAvailable, buildTimeMs RETURN sampleName, featureName, l1Nodes, l2Nodes, l3Nodes, l4Nodes, gpuAvailable, buildTimeMs")

FS_L1=$(echo "$FS_OUT" | tail -1 | cut -d',' -f3)
FS_L2=$(echo "$FS_OUT" | tail -1 | cut -d',' -f4)
FS_L3=$(echo "$FS_OUT" | tail -1 | cut -d',' -f5)
FS_L4=$(echo "$FS_OUT" | tail -1 | cut -d',' -f6)

# gpu_budget_mb=0 => L1 should be 0
if [ "$FS_L1" = "0" ]; then
    pass "FourLevelStore: L1 nodes = 0 (no GPU budget)"
else
    fail "FourLevelStore: expected L1=0, got $FS_L1"
fi

# L2 should be > 0 (100MB budget, 1433 dims * 4 bytes = 5732 bytes/row)
if [ -n "$FS_L2" ] && [ "$FS_L2" -gt 0 ] 2>/dev/null; then
    pass "FourLevelStore: L2 has $FS_L2 nodes"
else
    fail "FourLevelStore: L2 should have nodes, got $FS_L2"
fi

# Total classified nodes should be sensible
FS_TOTAL=$((FS_L1 + FS_L2 + FS_L3 + FS_L4))
if [ "$FS_TOTAL" -gt 0 ] 2>/dev/null; then
    pass "FourLevelStore: $FS_TOTAL total classified nodes (L1=$FS_L1 L2=$FS_L2 L3=$FS_L3 L4=$FS_L4)"
else
    fail "FourLevelStore: no nodes classified"
fi

# Check output files exist
if [ -f "$DB_DIR/gnn_features/node_features_cpu_cache.bin" ]; then
    pass "FourLevelStore: cpu_cache.bin exists"
else
    fail "FourLevelStore: missing cpu_cache.bin"
fi

if [ -f "$DB_DIR/gnn_features/node_features_store.meta" ]; then
    pass "FourLevelStore: store.meta exists"
else
    fail "FourLevelStore: missing store.meta"
fi

# Check GNNC magic on cpu_cache.bin (first 4 bytes)
CPU_MAGIC=$(xxd -p -l 4 "$DB_DIR/gnn_features/node_features_cpu_cache.bin")
if [ "$CPU_MAGIC" = "434e4e47" ] || [ "$CPU_MAGIC" = "474e4e43" ]; then
    pass "FourLevelStore: cpu_cache.bin has GNNC magic ($CPU_MAGIC)"
else
    fail "FourLevelStore: cpu_cache.bin bad magic (expected GNNC, got $CPU_MAGIC)"
fi

# Check GFLS magic on store.meta
META_MAGIC=$(xxd -p -l 4 "$DB_DIR/gnn_features/node_features_store.meta")
if [ "$META_MAGIC" = "534c4647" ] || [ "$META_MAGIC" = "47464c53" ]; then
    pass "FourLevelStore: store.meta has GFLS magic ($META_MAGIC)"
else
    fail "FourLevelStore: store.meta bad magic (expected GFLS, got $META_MAGIC)"
fi

# Check packed_slim directory exists and has batch files
SLIM_DIR="$DB_DIR/samples/e2e_sample/packed_slim"
if [ -d "$SLIM_DIR" ]; then
    SLIM_COUNT=$(ls "$SLIM_DIR"/batch_*.bin 2>/dev/null | wc -l)
    if [ "$SLIM_COUNT" -gt 0 ]; then
        pass "FourLevelStore: packed_slim has $SLIM_COUNT batch files"
    else
        fail "FourLevelStore: packed_slim directory is empty"
    fi
else
    # packed_slim might not exist if all nodes fit in L2
    pass "FourLevelStore: packed_slim not created (all nodes may fit in L2 cache)"
fi

# =============================================================================
# Step 5f: Error path — force=0 re-run should fail
# =============================================================================
info "Step 5f: Error path — force=0 re-run"

FS_ERR=$(query "CALL gnn_build_feature_store('e2e_sample', 'node_features', {gpu_budget_mb: 0, cpu_budget_mb: 100}) YIELD sampleName RETURN sampleName" 2>&1)
if echo "$FS_ERR" | grep -qi "already exist\|force"; then
    pass "FourLevelStore: force=0 prevents overwrite"
else
    fail "FourLevelStore: force=0 should fail, got: $FS_ERR"
fi

# =============================================================================
# Step 5g: Force overwrite
# =============================================================================
info "Step 5g: Force overwrite (force=1)"

FS_FORCE=$(query "CALL gnn_build_feature_store('e2e_sample', 'node_features', {gpu_budget_mb: 0, cpu_budget_mb: 100, force: 1}) YIELD buildTimeMs RETURN buildTimeMs")
if echo "$FS_FORCE" | grep -qE '[0-9]+'; then
    pass "FourLevelStore: force=1 overwrite succeeded"
else
    fail "FourLevelStore: force=1 overwrite failed: $FS_FORCE"
fi

# =============================================================================
# Step 6: Verify features (S4 Critical Property)
# =============================================================================
info "Step 6: Verify packed features match originals (S4 property)"

# NOTE: server stays alive for Step 7 (gnn_train). The Python verification
# below reads files directly — that is safe because mdb flushes all writes
# to disk after each procedure call. The cleanup() trap will kill the server
# at the end of the script.

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
# Step 7.0: Verify gnn_train requires a pre-built FourLevelStore
#
# gnn_train MUST fail with a clear error message when the FourLevelStore
# metadata (.meta) is missing. This contract is the reason the training
# procedure is DiskGNN-faithful: there is no silent fallback to a plain
# mmap FeatureMatrix that would bypass the L1/L2/L3/L4 cache hierarchy.
# We temporarily move the .meta file out of the way, call gnn_train,
# assert the error is explicit, and restore the file for Step 7.
# =============================================================================
info "Step 7.0: Verify gnn_train requires FourLevelStore (error path)"

META_FILE="$DB_DIR/gnn_features/node_features_store.meta"
if [ ! -f "$META_FILE" ]; then
    fail "Step 7.0 precondition: $META_FILE not found (gnn_build_feature_store did not run)"
else
    mv "$META_FILE" "${META_FILE}.bak"

    NOMETA_OUT=$(query "CALL gnn_train('e2e_sample', 'node_features', {model: 'graphsage', epochs: 1, randomSeed: 42}) YIELD testAccuracy RETURN testAccuracy" 2>&1)

    # Restore .meta BEFORE asserting, so Step 7 can run even if assertion fails.
    mv "${META_FILE}.bak" "$META_FILE"

    if echo "$NOMETA_OUT" | grep -qiE "gnn_build_feature_store|store\.meta|FourLevelStore|feature store"; then
        pass "gnn_train: reports clear error when store.meta is missing"
    else
        fail "gnn_train: expected error referencing FourLevelStore, got: $(echo "$NOMETA_OUT" | head -3 | tr '\n' ' ')"
    fi
fi

# =============================================================================
# Step 7: Train GraphSAGE and verify accuracy
#
# Two configurations are supported via the GNN_TRAIN_CONFIG env var:
#
#   GNN_TRAIN_CONFIG=quick   (default)  ~2-3s training, threshold testAcc>=0.72
#                            hidden=64, lr=0.01, dropout=0.5, wd=5e-4,
#                            epochs=200, patience=20
#
#   GNN_TRAIN_CONFIG=paper              ~7s training, threshold testAcc>=0.75
#                            hidden=128, lr=0.005, dropout=0.5, wd=5e-4,
#                            epochs=500, patience=50
#
# Both configs train on the SAME sample (e2e_sample) with Planetoid splits
# (140/500/1000) and UNDIRECTED orientation. Reference accuracy from Shchur
# et al. 2018 ("Pitfalls of GNN Evaluation"): GS-mean on Cora reaches a
# median of ~80% test accuracy with range 75-85% across 100 random splits
# and 20 inits per split.
#
# The accuracy gap between quick and paper modes is the result of more
# training time (more epochs, larger hidden dim) — not different splits or
# implementation. Both modes are valid; quick is for CI, paper for thesis.
# =============================================================================
GNN_TRAIN_CONFIG="${GNN_TRAIN_CONFIG:-quick}"
info "Step 7: Train GraphSAGE (config=$GNN_TRAIN_CONFIG)"

if [ "$GNN_TRAIN_CONFIG" = "paper" ]; then
    HIDDEN=128
    LR=0.005
    DROPOUT=0.5
    WD=0.0005
    EPOCHS=500
    PATIENCE=50
    MIN_TEST_ACC="0.75"
    MIN_VAL_ACC="0.75"
else
    HIDDEN=64
    LR=0.01
    DROPOUT=0.5
    WD=0.0005
    EPOCHS=200
    PATIENCE=20
    MIN_TEST_ACC="0.72"
    MIN_VAL_ACC="0.72"
fi

TRAIN_OUT=$(query "CALL gnn_train('e2e_sample', 'node_features', {model: 'graphsage', hiddenDim: $HIDDEN, dropout: $DROPOUT, epochs: $EPOCHS, lr: $LR, weightDecay: $WD, patience: $PATIENCE, randomSeed: 42}) YIELD modelName, ranEpochs, didConverge, bestValAccuracy, testAccuracy, trainSeconds, l1HitRatio, l2HitRatio, l3Reads, l4Reads RETURN modelName, ranEpochs, didConverge, bestValAccuracy, testAccuracy, trainSeconds, l1HitRatio, l2HitRatio, l3Reads, l4Reads")

# Parse YIELDs (skip header)
TRAIN_LINE=$(echo "$TRAIN_OUT" | tail -1)
T_MODEL=$(echo "$TRAIN_LINE"  | cut -d',' -f1 | tr -d '"')
T_EPOCHS=$(echo "$TRAIN_LINE" | cut -d',' -f2)
T_VAL=$(echo "$TRAIN_LINE"    | cut -d',' -f4)
T_TEST=$(echo "$TRAIN_LINE"   | cut -d',' -f5)
T_SEC=$(echo "$TRAIN_LINE"    | cut -d',' -f6)
T_L1=$(echo "$TRAIN_LINE"     | cut -d',' -f7)
T_L2=$(echo "$TRAIN_LINE"     | cut -d',' -f8)
T_L3=$(echo "$TRAIN_LINE"     | cut -d',' -f9)
T_L4=$(echo "$TRAIN_LINE"     | cut -d',' -f10)

# Level 1: YIELD presence + values
if [ "$T_MODEL" = "graphsage" ]; then
    pass "Train: modelName=graphsage"
else
    fail "Train: modelName='$T_MODEL'"
fi

if [ -n "$T_EPOCHS" ] && [ "$T_EPOCHS" -gt 0 ] 2>/dev/null; then
    pass "Train: ranEpochs=$T_EPOCHS (>0)"
else
    fail "Train: invalid ranEpochs '$T_EPOCHS'"
fi

# Compare floats with awk
val_pass=$(awk -v v="$T_VAL" -v t="$MIN_VAL_ACC" 'BEGIN {print (v+0 >= t+0) ? "1":"0"}' 2>/dev/null)
if [ "$val_pass" = "1" ]; then
    pass "Train: bestValAccuracy=$T_VAL >= $MIN_VAL_ACC"
else
    fail "Train: bestValAccuracy=$T_VAL < $MIN_VAL_ACC (config=$GNN_TRAIN_CONFIG)"
fi

test_pass=$(awk -v v="$T_TEST" -v t="$MIN_TEST_ACC" 'BEGIN {print (v+0 >= t+0) ? "1":"0"}' 2>/dev/null)
if [ "$test_pass" = "1" ]; then
    pass "Train: testAccuracy=$T_TEST >= $MIN_TEST_ACC"
else
    fail "Train: testAccuracy=$T_TEST < $MIN_TEST_ACC (config=$GNN_TRAIN_CONFIG)"
fi

# Sanity bound: training shouldn't take more than 60s on CPU for Cora
sec_pass=$(awk -v s="$T_SEC" 'BEGIN {print (s+0 < 60.0) ? "1":"0"}' 2>/dev/null)
if [ "$sec_pass" = "1" ]; then
    pass "Train: trainSeconds=$T_SEC < 60s"
else
    fail "Train: trainSeconds=$T_SEC too high (>60s)"
fi

# Level 1b: FourLevelStore cache stats (full mode only, should always be present
# because gnn_train is mandated to use FourLevelStore). A missing YIELD field
# returns the literal string "NULL" in the CSV output, so we must reject it
# explicitly before running the numeric check (awk silently treats "NULL"+0 as 0).
for name in l1HitRatio l2HitRatio l3Reads l4Reads; do
    case $name in
        l1HitRatio) v=$T_L1 ;;
        l2HitRatio) v=$T_L2 ;;
        l3Reads)    v=$T_L3 ;;
        l4Reads)    v=$T_L4 ;;
    esac
    if [ -z "$v" ] || [ "$v" = "NULL" ]; then
        fail "Train cache: $name missing from YIELD (got '$v')"
    elif awk -v x="$v" 'BEGIN{exit !(x ~ /^[0-9.eE+-]+$/ && x+0 >= 0)}' 2>/dev/null; then
        pass "Train cache: $name=$v (present, numeric, non-negative)"
    else
        fail "Train cache: $name non-numeric or negative: '$v'"
    fi
done

# Level 2: artifact existence
OUT_DIR="$DB_DIR/projections/e2e_proj/gnn_output/default"
for art in model.pt embeddings.npy training_log.json; do
    if [ -f "$OUT_DIR/$art" ] && [ -s "$OUT_DIR/$art" ]; then
        SZ=$(stat -c%s "$OUT_DIR/$art" 2>/dev/null || stat -f%z "$OUT_DIR/$art")
        pass "Train artifact: $art ($SZ bytes)"
    else
        fail "Train artifact missing or empty: $art"
    fi
done

# Level 3: cross-validation between YIELDs and training_log.json
TRAIN_LOG_VERIFY=$(python3 << PYEOF
import json, numpy as np, sys

LOG = "$OUT_DIR/training_log.json"
EMB = "$OUT_DIR/embeddings.npy"
META = "$DB_DIR/projections/e2e_proj/gnn_meta.bin"
LBL = "$DB_DIR/projections/e2e_proj/labels.bin"
SPL = "$DB_DIR/projections/e2e_proj/splits.bin"

errors = 0

# 1. training_log.json is parseable JSON
try:
    with open(LOG) as f:
        log = json.load(f)
    print("PASS|json_valid|training_log.json parseable")
except Exception as e:
    print(f"FAIL|json_valid|{e}")
    sys.exit(1)

# 2. hyperparameters match what we asked for
hp = log["hyperparameters"]
if hp["input_dim"] == 1433:
    print(f"PASS|hp_input_dim|input_dim=1433")
else:
    print(f"FAIL|hp_input_dim|expected 1433 got {hp['input_dim']}")
    errors += 1
if hp["num_classes"] == 7:
    print(f"PASS|hp_num_classes|num_classes=7")
else:
    print(f"FAIL|hp_num_classes|expected 7 got {hp['num_classes']}")
    errors += 1
if hp["hidden_dim"] == $HIDDEN:
    print(f"PASS|hp_hidden_dim|hidden_dim=$HIDDEN")
else:
    print(f"FAIL|hp_hidden_dim|expected $HIDDEN got {hp['hidden_dim']}")
    errors += 1
if hp["num_layers"] == 2:
    print(f"PASS|hp_num_layers|num_layers=2 (matches fanouts [10,5])")
else:
    print(f"FAIL|hp_num_layers|expected 2 got {hp['num_layers']}")
    errors += 1

# 3. losses are decreasing overall (start ~log(7)=1.95, end significantly lower)
losses = log["epoch_losses"]
import math
expected_init = math.log(7)
if 1.5 <= losses[0] <= 2.5:
    print(f"PASS|init_loss|losses[0]={losses[0]:.3f} near log(7)={expected_init:.3f}")
else:
    print(f"FAIL|init_loss|losses[0]={losses[0]:.3f} far from log(7)={expected_init:.3f}")
    errors += 1
if losses[-1] < losses[0] * 0.7:
    reduction = (1 - losses[-1]/losses[0])*100
    print(f"PASS|loss_decrease|{reduction:.1f}% reduction ({losses[0]:.3f} -> {losses[-1]:.3f})")
else:
    print(f"FAIL|loss_decrease|insufficient ({losses[0]:.3f} -> {losses[-1]:.3f})")
    errors += 1

# 4. results in training_log match YIELD values
log_test = log["results"]["test_accuracy"]
yield_test = $T_TEST
if abs(log_test - yield_test) < 1e-4:
    print(f"PASS|yield_match|training_log test={log_test:.4f} == YIELD test={yield_test:.4f}")
else:
    print(f"FAIL|yield_match|log={log_test} yield={yield_test}")
    errors += 1

# 5. embeddings.npy shape and sanity
# The exporter writes one row per seed across ALL batches:
#   1 train batch (140) + 2 val batches (500) + 4 test batches (1000) = 1640 seeds
emb = np.load(EMB)
expected_seeds = 140 + 500 + 1000
expected_shape = (expected_seeds, $HIDDEN)
if emb.shape == expected_shape:
    print(f"PASS|emb_shape|shape={emb.shape}")
else:
    print(f"FAIL|emb_shape|expected {expected_shape} got {emb.shape}")
    errors += 1
if np.isnan(emb).any() or np.isinf(emb).any():
    print(f"FAIL|emb_finite|NaN or Inf detected")
    errors += 1
else:
    print(f"PASS|emb_finite|no NaN/Inf, mean={emb.mean():.4f} std={emb.std():.4f}")

# 6. labels.bin and splits.bin distribution sanity
import struct
with open(LBL,'rb') as f:
    raw = f.read()
labels_arr = np.frombuffer(raw[32:], dtype=np.int64)  # header is 32 bytes
if len(labels_arr) == 2708:
    classes = sorted(set(labels_arr.tolist()))
    if classes == list(range(7)):
        print(f"PASS|labels_classes|all 7 classes present in labels.bin")
    else:
        print(f"FAIL|labels_classes|got {classes}")
        errors += 1
else:
    print(f"FAIL|labels_count|expected 2708 got {len(labels_arr)}")
    errors += 1

with open(SPL,'rb') as f:
    raw = f.read()
splits_arr = np.frombuffer(raw[24:], dtype=np.uint8)  # header is 24 bytes
train_n  = int((splits_arr == 0).sum())
val_n    = int((splits_arr == 1).sum())
test_n   = int((splits_arr == 2).sum())
unlab_n  = int((splits_arr == 255).sum())
if (train_n, val_n, test_n) == (140, 500, 1000):
    print(f"PASS|planetoid_splits|140/500/1000 train/val/test exact (Planetoid)")
else:
    print(f"FAIL|planetoid_splits|got {train_n}/{val_n}/{test_n}")
    errors += 1
if unlab_n == 1068:
    print(f"PASS|unlabeled_count|1068 unlabeled (2708 - 1640)")
else:
    print(f"FAIL|unlabeled_count|got {unlab_n}")
    errors += 1

# 7. FourLevelStore cache stats in training_log.json
# These must be present because gnn_train mandates full mode.
if "cache_stats" in log:
    cs = log["cache_stats"]
    required = ["l1_hits", "l2_hits", "l3_reads", "l4_reads", "total_requests"]
    missing = [k for k in required if k not in cs]
    if not missing:
        print(f"PASS|cache_stats_present|keys={sorted(cs.keys())}")
        total = int(cs["total_requests"])
        if total > 0:
            print(f"PASS|cache_stats_active|total_requests={total} > 0")
        else:
            print(f"FAIL|cache_stats_active|total_requests=0 (store never queried)")
            errors += 1
    else:
        print(f"FAIL|cache_stats_present|missing keys: {missing}")
        errors += 1
else:
    print(f"FAIL|cache_stats_present|'cache_stats' section not in training_log.json")
    errors += 1

# Do NOT sys.exit() — bash counts PASS/FAIL via the loop below.
# Exiting here would propagate to set -e and abort the script before
# the output can be parsed.
PYEOF
)

while IFS='|' read -r status check detail; do
    if [ "$status" = "PASS" ]; then
        pass "Train $check: $detail"
    elif [ "$status" = "FAIL" ]; then
        fail "Train $check: $detail"
    fi
done <<< "$TRAIN_LOG_VERIFY"

# =============================================================================
# Step 8: CUDA FeatureAssembler path (gpu_budget > 0)
#
# Rebuilds the FourLevelStore with GPU budget so hot features land in L1
# (GPU VRAM). This forces gnn_train to use assemble_cuda() — the DiskGNN
# CUDA kernel that reads L1 from HBM and L2 via UVA in a single pass.
#
# For Cora (2708 nodes × 1433 dims × 4 bytes ≈ 15 MB), 100 MB GPU budget
# fits the entire dataset in L1. Accuracy must match the CPU-only run.
# =============================================================================
info "Step 8: CUDA FeatureAssembler path (gpu_budget_mb=100)"

# 8a: Rebuild feature store with GPU budget
FS_GPU=$(query "CALL gnn_build_feature_store('e2e_sample', 'node_features', {gpu_budget_mb: 100, cpu_budget_mb: 100, reorder: 1, force: 1}) YIELD l1Nodes, l2Nodes, l3Nodes, l4Nodes, gpuAvailable RETURN l1Nodes, l2Nodes, l3Nodes, l4Nodes, gpuAvailable")

GPU_L1=$(echo "$FS_GPU" | tail -1 | cut -d',' -f1)
GPU_L2=$(echo "$FS_GPU" | tail -1 | cut -d',' -f2)
GPU_AVAIL=$(echo "$FS_GPU" | tail -1 | cut -d',' -f5)

SKIP_CUDA_CHECKS=0
if [ -n "$GPU_L1" ] && [ "$GPU_L1" -gt 0 ] 2>/dev/null; then
    pass "CUDA store: L1 has $GPU_L1 nodes on GPU (L2=$GPU_L2)"
else
    if [ "$GPU_AVAIL" = "0" ] || [ "$GPU_AVAIL" = "false" ]; then
        pass "CUDA store: GPU not available, skipping CUDA path checks (L1=$GPU_L1)"
        SKIP_CUDA_CHECKS=1
    else
        fail "CUDA store: expected L1>0 with gpu_budget_mb=100, got L1=$GPU_L1 (gpuAvail=$GPU_AVAIL)"
    fi
fi

if [ "$SKIP_CUDA_CHECKS" = "0" ]; then

# 8b: Train with GPU-backed feature store
CUDA_TRAIN=$(query "CALL gnn_train('e2e_sample', 'node_features', {model: 'graphsage', hiddenDim: $HIDDEN, dropout: $DROPOUT, epochs: $EPOCHS, lr: $LR, weightDecay: $WD, patience: $PATIENCE, randomSeed: 42}) YIELD testAccuracy, l1HitRatio, l2HitRatio, l3Reads, l4Reads RETURN testAccuracy, l1HitRatio, l2HitRatio, l3Reads, l4Reads")

CT_ACC=$(echo "$CUDA_TRAIN" | tail -1 | cut -d',' -f1)
CT_L1R=$(echo "$CUDA_TRAIN" | tail -1 | cut -d',' -f2)
CT_L2R=$(echo "$CUDA_TRAIN" | tail -1 | cut -d',' -f3)
CT_L3=$(echo "$CUDA_TRAIN"  | tail -1 | cut -d',' -f4)
CT_L4=$(echo "$CUDA_TRAIN"  | tail -1 | cut -d',' -f5)

# 8c: Verify L1 was actually used (l1HitRatio > 0)
l1_used=$(awk -v r="$CT_L1R" 'BEGIN {print (r+0 > 0) ? "1":"0"}' 2>/dev/null)
if [ "$l1_used" = "1" ]; then
    pass "CUDA train: l1HitRatio=$CT_L1R (CUDA kernel exercised)"
else
    fail "CUDA train: l1HitRatio=$CT_L1R — expected >0 with GPU budget"
fi

# 8d: Accuracy must meet the same threshold as CPU-only run
cuda_acc_pass=$(awk -v a="$CT_ACC" -v t="$MIN_TEST_ACC" 'BEGIN {print (a+0 >= t+0) ? "1":"0"}' 2>/dev/null)
if [ "$cuda_acc_pass" = "1" ]; then
    pass "CUDA train: testAccuracy=$CT_ACC >= $MIN_TEST_ACC"
else
    fail "CUDA train: testAccuracy=$CT_ACC < $MIN_TEST_ACC"
fi

# 8e: Accuracy should be close to CPU-only run (±5pp tolerance for seed variation)
acc_diff=$(awk -v a="$CT_ACC" -v b="$T_TEST" 'BEGIN {d=a-b; print (d<0?-d:d)}' 2>/dev/null)
acc_close=$(awk -v d="$acc_diff" 'BEGIN {print (d+0 <= 0.05) ? "1":"0"}' 2>/dev/null)
if [ "$acc_close" = "1" ]; then
    pass "CUDA train: accuracy delta=${acc_diff} vs CPU-only (within 5pp)"
else
    fail "CUDA train: accuracy delta=${acc_diff} vs CPU-only (>5pp divergence)"
fi

# 8f: Log cache distribution
info "  CUDA cache: l1Hit=$CT_L1R l2Hit=$CT_L2R l3=$CT_L3 l4=$CT_L4"

fi  # SKIP_CUDA_CHECKS

# =============================================================================
# Step 9: Write-back embeddings to projection + GQL query verification
#
# Trains with writeProperty='embedding', which triggers the EmbeddingWriter
# to persist per-node embeddings as tensor properties in the projection.
# After write-back, GQL queries verify that:
#   (a) all 2708 nodes received an embedding
#   (b) embeddings display as tensor arrays
#   (c) cosineDistance between distinct nodes is in [0, 2]
#   (d) cosineDistance of a node with itself is ~0
# =============================================================================
info "Step 9: EmbeddingWriter — write-back + GQL queries"

# 9a: Rebuild feature store with CPU-only budget (Step 8 set GPU mode)
FS_RESET=$(query "CALL gnn_build_feature_store('e2e_sample', 'node_features', {gpu_budget_mb: 0, cpu_budget_mb: 100, force: 1}) YIELD buildTimeMs RETURN buildTimeMs")

WB_OUT=$(query "CALL gnn_train('e2e_sample', 'node_features', {
    model: 'graphsage', hiddenDim: $HIDDEN, dropout: $DROPOUT,
    epochs: $EPOCHS, lr: $LR, weightDecay: $WD, patience: $PATIENCE,
    randomSeed: 42, writeProperty: 'embedding'
}) YIELD testAccuracy, nodesWritten, nodesInferred, inferenceMillis, writeMillis
RETURN testAccuracy, nodesWritten, nodesInferred, inferenceMillis, writeMillis")

# Parse output
WB_LINE=$(echo "$WB_OUT" | tail -1)
WB_ACC=$(echo "$WB_LINE" | cut -d',' -f1)
WB_WRITTEN=$(echo "$WB_LINE" | cut -d',' -f2)
WB_INFERRED=$(echo "$WB_LINE" | cut -d',' -f3)
WB_INF_MS=$(echo "$WB_LINE" | cut -d',' -f4)
WB_WRITE_MS=$(echo "$WB_LINE" | cut -d',' -f5)

# Debug: show raw output on unexpected results
if [ -z "$WB_WRITTEN" ] || [ "$WB_WRITTEN" = "NULL" ]; then
    info "  DEBUG raw WB_OUT: $(echo "$WB_OUT" | head -5 | tr '\n' ' ')"
fi

# Check nodesWritten
if [ "$WB_WRITTEN" = "2708" ]; then
    pass "WriteBack: nodesWritten=$WB_WRITTEN (all Cora nodes)"
else
    fail "WriteBack: expected nodesWritten=2708, got $WB_WRITTEN"
fi

# Check nodesInferred > 0
if [ -n "$WB_INFERRED" ] && [ "$WB_INFERRED" -gt 0 ] 2>/dev/null; then
    pass "WriteBack: nodesInferred=$WB_INFERRED"
else
    fail "WriteBack: expected nodesInferred>0, got $WB_INFERRED"
fi

# Check writeMillis > 0
wms_ok=$(awk -v w="$WB_WRITE_MS" 'BEGIN {print (w+0 > 0) ? "1":"0"}')
if [ "$wms_ok" = "1" ]; then
    pass "WriteBack: writeMillis=$WB_WRITE_MS"
else
    fail "WriteBack: writeMillis=$WB_WRITE_MS (expected >0)"
fi

# 9b: Query — all nodes have embedding property
EMB_COUNT=$(query "USE e2e_proj MATCH (n) WHERE n.embedding IS NOT NULL RETURN count(n)" | tail -1)
if [ "$EMB_COUNT" = "2708" ]; then
    pass "GQL: all 2708 nodes have embedding"
else
    fail "GQL: expected 2708 embeddings, got $EMB_COUNT"
fi

# 9c: Embedding is a tensor (check it's non-null and non-empty)
EMB_SAMPLE=$(query "USE e2e_proj MATCH (n) RETURN n.embedding LIMIT 1" | tail -1)
if [ -n "$EMB_SAMPLE" ] && [ "$EMB_SAMPLE" != "NULL" ]; then
    pass "GQL: embedding present ($EMB_SAMPLE)"
else
    fail "GQL: embedding missing or NULL"
fi

# 9d: cosineDistance between two different nodes
DIST=$(query "USE e2e_proj MATCH (a:Paper), (b:Paper) WHERE a <> b RETURN cosineDistance(a.embedding, b.embedding) LIMIT 1" | tail -1)
dist_ok=$(awk -v d="$DIST" 'BEGIN {print (d+0 >= 0 && d+0 <= 2.0) ? "1":"0"}')
if [ "$dist_ok" = "1" ]; then
    pass "GQL: cosineDistance=$DIST (in [0, 2])"
else
    fail "GQL: cosineDistance=$DIST (expected [0, 2])"
fi

# 9e: Self-similarity = 0
SELF=$(query "USE e2e_proj MATCH (n) RETURN cosineDistance(n.embedding, n.embedding) LIMIT 1" | tail -1)
self_ok=$(awk -v s="$SELF" 'BEGIN {print (s+0 < 1e-4) ? "1":"0"}')
if [ "$self_ok" = "1" ]; then
    pass "GQL: self-cosineDistance=$SELF (~0)"
else
    fail "GQL: self-cosineDistance=$SELF (expected ~0)"
fi

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
