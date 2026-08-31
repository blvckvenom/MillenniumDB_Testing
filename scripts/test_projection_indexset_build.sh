#!/bin/bash
# Spec #3 T3.7 + T3.8 — IndexSet build-phase gating integration test.
#
# Verifies that graph_project honors the `indexSet` config key across BOTH
# build pipelines (CLASSIC in projection_storage.cc::build_all_indexes_bulk()
# and SERIAL via native_projection_builder.cc::finalize_serialized_()).
#
# Presets:
#   ALL                -> 10 topology+label .leaf files (baseline: current behavior)
#   GNN_MINIMAL        ->  5 .leaf files
#                         (nodes, node_label, label_node, from_to_edge, to_from_edge)
#   READONLY_TRAVERSAL ->  7 .leaf files (GNN_MINIMAL + edge_label + label_edge)
#
# Property indexes (node_key_value, etc.) are NOT controlled by IndexSet
# (Spec #3 §3.4) and therefore are not counted in these checks — cora_gnn's
# "Paper" projection here has no declared node/edge property config, so no
# property .leaf files are built.
#
# Also tests byte-identical output for ALL mode across classic and serial
# paths, confirming IndexSet=ALL is a no-op preserving pre-Spec-#3 behavior.
#
# Usage: ./scripts/test_projection_indexset_build.sh
# Exit:  0 on all checks pass, 1 on any failure, 2 on setup/server error.

set -euo pipefail

# Resolved from this script rather than from the caller's directory, so the
# benchmark runs from anywhere. MDB_HOME still overrides.
MDB_HOME="${MDB_HOME:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
MDB=${MDB:-$MDB_HOME/build/Release/bin/mdb}
DB=${DB:-data/dbs/gql/cora_gnn}
PORT=${PORT:-19881}
NODE_LABEL=${NODE_LABEL:-Paper}
EDGE_TYPE=${EDGE_TYPE:-CITES}

# 6 test projections: 3 presets × 2 scan paths
PROJS=(
    t3_build_all_classic
    t3_build_gnnmin_classic
    t3_build_ro_classic
    t3_build_all_serial
    t3_build_gnnmin_serial
    t3_build_ro_serial
)
# SERIAL_SCAN env value per projection
SCAN_ENV=(0 0 0 1 1 1)
# IndexSet preset per projection
PRESETS=(ALL GNN_MINIMAL READONLY_TRAVERSAL ALL GNN_MINIMAL READONLY_TRAVERSAL)

# Expected .leaf file names per preset. Property .leaf names excluded since
# they are NOT gated by IndexSet (Spec #3 §3.4).
EXPECTED_ALL=(
    nodes.leaf
    node_label.leaf label_node.leaf
    from_to_edge.leaf to_from_edge.leaf
    edge_direction.leaf edge_from_to.leaf edge_n1_n2.leaf
    edge_label.leaf label_edge.leaf
)
EXPECTED_GNN_MINIMAL=(
    nodes.leaf
    node_label.leaf label_node.leaf
    from_to_edge.leaf to_from_edge.leaf
)
EXPECTED_READONLY=(
    nodes.leaf
    node_label.leaf label_node.leaf
    from_to_edge.leaf to_from_edge.leaf
    edge_label.leaf label_edge.leaf
)
# Names that MUST NOT appear under GNN_MINIMAL.
FORBIDDEN_GNN_MINIMAL=(
    edge_direction.leaf edge_from_to.leaf edge_n1_n2.leaf
    edge_label.leaf label_edge.leaf
)
# Names that MUST NOT appear under READONLY_TRAVERSAL.
FORBIDDEN_READONLY=(
    edge_direction.leaf edge_from_to.leaf edge_n1_n2.leaf
)

cleanup() {
    if [[ -n "${SRV_PID:-}" ]] && kill -0 "$SRV_PID" 2>/dev/null; then
        kill "$SRV_PID" 2>/dev/null || true
        wait "$SRV_PID" 2>/dev/null || true
    fi
    for p in "${PROJS[@]}"; do
        "$MDB" drop-projection "$DB" "$p" 2>/dev/null || true
    done
}
trap cleanup EXIT

SRV_PID=

start_server() {
    local scan_env="$1"
    local logfile="$2"
    MDB_PROJECTION_SERIAL_SCAN="$scan_env" \
        "$MDB" server "$DB" --port "$PORT" --timeout 600 \
        > "$logfile" 2>&1 &
    SRV_PID=$!

    for i in $(seq 1 20); do
        if curl -sSf -o /dev/null "http://127.0.0.1:$PORT/" \
                --data-binary "RETURN 1" -H "Accept: text/csv" 2>/dev/null; then
            return 0
        fi
        sleep 0.5
    done
    echo "ERROR: server on port $PORT did not come up within 10 s." >&2
    cat "$logfile" >&2 || true
    return 2
}

stop_server() {
    if [[ -n "${SRV_PID:-}" ]] && kill -0 "$SRV_PID" 2>/dev/null; then
        kill "$SRV_PID" 2>/dev/null || true
        wait "$SRV_PID" 2>/dev/null || true
    fi
    SRV_PID=
}

run_projection() {
    local preset="$1"
    local proj="$2"
    local query
    query="CALL graph_project('$proj', '$NODE_LABEL', '$EDGE_TYPE', {orientation: 'NATURAL', indexSet: '$preset'}) YIELD graphName RETURN graphName"
    local response
    response=$(curl -sS --data-binary "$query" -H "Accept: text/csv" \
        "http://127.0.0.1:$PORT/" || true)
    echo "    response: $(echo "$response" | tr '\n' ' ')"
}

# --- Run all 6 (path, preset) combinations ---
echo "=== T3.7/T3.8: IndexSet build-phase gating on $(basename "$DB") ==="
echo

for i in 0 1 2 3 4 5; do
    scan="${SCAN_ENV[$i]}"
    preset="${PRESETS[$i]}"
    proj="${PROJS[$i]}"
    echo ">>> [scan=$scan preset=$preset] Starting server..."
    start_server "$scan" "/tmp/mdb_indexset_${i}.log"
    echo ">>> [scan=$scan preset=$preset] Creating projection '$proj'..."
    run_projection "$preset" "$proj"
    stop_server
    echo
done

# --- Verify file presence / absence per preset ---
FAIL=0

check_present() {
    local proj="$1"; shift
    local expected=("$@")
    local dir="$DB/projections/$proj"
    if [[ ! -d "$dir" ]]; then
        echo "FAIL: projection directory missing: $dir"
        FAIL=$((FAIL + 1))
        return
    fi
    for name in "${expected[@]}"; do
        if [[ -f "$dir/$name" ]]; then
            echo "  OK    present: $proj/$name"
        else
            echo "  FAIL  missing: $proj/$name"
            FAIL=$((FAIL + 1))
        fi
    done
}

check_absent() {
    local proj="$1"; shift
    local forbidden=("$@")
    local dir="$DB/projections/$proj"
    for name in "${forbidden[@]}"; do
        if [[ -f "$dir/$name" ]]; then
            echo "  FAIL  unexpected: $proj/$name"
            FAIL=$((FAIL + 1))
        else
            echo "  OK    absent:     $proj/$name"
        fi
    done
}

check_leaf_count() {
    local proj="$1"
    local expected="$2"
    local dir="$DB/projections/$proj"
    local actual
    actual=$(find "$dir" -maxdepth 1 -name "*.leaf" | wc -l)
    if [[ "$actual" -eq "$expected" ]]; then
        echo "  OK    leaf count: $proj -> $actual"
    else
        echo "  FAIL  leaf count: $proj expected $expected got $actual"
        FAIL=$((FAIL + 1))
    fi
}

# Test 1 & 4: ALL preset creates all 10 expected .leaf files (both paths)
echo "--- Test 1: BuildCoraAllModeCreatesAllExpectedIndexes (CLASSIC) ---"
check_present t3_build_all_classic "${EXPECTED_ALL[@]}"
check_leaf_count t3_build_all_classic 10
echo
echo "--- Test 1b: BuildCoraAllModeCreatesAllExpectedIndexes (SERIAL) ---"
check_present t3_build_all_serial "${EXPECTED_ALL[@]}"
check_leaf_count t3_build_all_serial 10
echo

# Test 2 & 5: GNN_MINIMAL creates exactly 5 .leaf files (both paths)
echo "--- Test 2: BuildCoraGnnMinimalCreatesExactly5Indexes (CLASSIC) ---"
check_present t3_build_gnnmin_classic "${EXPECTED_GNN_MINIMAL[@]}"
check_absent  t3_build_gnnmin_classic "${FORBIDDEN_GNN_MINIMAL[@]}"
check_leaf_count t3_build_gnnmin_classic 5
echo
echo "--- Test 5: BuildCoraGnnMinimalSerialPath ---"
check_present t3_build_gnnmin_serial "${EXPECTED_GNN_MINIMAL[@]}"
check_absent  t3_build_gnnmin_serial "${FORBIDDEN_GNN_MINIMAL[@]}"
check_leaf_count t3_build_gnnmin_serial 5
echo

# Test 3 & 6: READONLY creates exactly 7 .leaf files (both paths)
echo "--- Test 3: BuildCoraReadonlyCreatesExactly7Indexes (CLASSIC) ---"
check_present t3_build_ro_classic "${EXPECTED_READONLY[@]}"
check_absent  t3_build_ro_classic "${FORBIDDEN_READONLY[@]}"
check_leaf_count t3_build_ro_classic 7
echo
echo "--- Test 3b: BuildCoraReadonlyCreatesExactly7Indexes (SERIAL) ---"
check_present t3_build_ro_serial "${EXPECTED_READONLY[@]}"
check_absent  t3_build_ro_serial "${FORBIDDEN_READONLY[@]}"
check_leaf_count t3_build_ro_serial 7
echo

# Test 4: BuildCoraAllModePreservesByteIdenticalToPreSpec3
# ALL preset across classic and serial must produce byte-identical B+Tree output
# (invariant I4 from Spec #2 / ADR 004, still holds under IndexSet).
echo "--- Test 4: BuildCoraAllModePreservesByteIdenticalToPreSpec3 ---"
DIR_CC="$DB/projections/t3_build_all_classic"
DIR_SS="$DB/projections/t3_build_all_serial"
TOTAL=0
MISMATCH=0
for f in "$DIR_CC"/*.leaf "$DIR_CC"/*.dir; do
    [[ -e "$f" ]] || continue
    name=$(basename "$f")
    f2="$DIR_SS/$name"
    TOTAL=$((TOTAL + 1))
    if [[ ! -f "$f2" ]]; then
        echo "  FAIL  missing in serial: $name"
        MISMATCH=$((MISMATCH + 1))
    elif ! cmp -s "$f" "$f2"; then
        sz1=$(stat -c %s "$f"); sz2=$(stat -c %s "$f2")
        echo "  FAIL  mismatch: $name (classic=${sz1}B serial=${sz2}B)"
        MISMATCH=$((MISMATCH + 1))
    else
        sz=$(stat -c %s "$f")
        echo "  OK    byte-match: $name (${sz}B)"
    fi
done
if [[ $MISMATCH -gt 0 ]]; then
    FAIL=$((FAIL + MISMATCH))
fi
echo "  (compared $TOTAL files; $MISMATCH mismatches)"
echo

# --- Summary ---
echo "================================================================"
if [[ $FAIL -eq 0 ]]; then
    echo "ALL CHECKS PASSED"
    exit 0
else
    echo "FAILED: $FAIL checks"
    exit 1
fi
