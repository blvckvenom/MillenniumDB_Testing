#!/usr/bin/env bash
#
# Spec #8 T8.10 — 4-mode golden compare for graphStorage x leafFormat.
#
# Builds the cora_gnn projection four times under the matrix:
#
#   Run 1: leafFormat=BITSET        graphStorage=BTREE       indexSet=GNN_MINIMAL
#   Run 2: leafFormat=DELTA_VARINT  graphStorage=BTREE       indexSet=GNN_MINIMAL
#   Run 3: leafFormat=BITSET        graphStorage=CSR_HYBRID  indexSet=GNN_MINIMAL
#   Run 4: leafFormat=DELTA_VARINT  graphStorage=CSR_HYBRID  indexSet=GNN_MINIMAL
#
# Asserts:
#   (Phase 1) .leaf presence for {nodes, node_label, label_node,
#       from_to_edge, to_from_edge} and absence for the 5 edge indexes
#       not in GNN_MINIMAL.
#   (Phase 2) First-byte format dispatch per run:
#       - Run 1: edge leaves byte 0 != 0x03 (BITSET v1 value_count).
#       - Run 2: edge leaves byte 0 == 0x02 (DELTA_VARINT v2 magic).
#       - Run 3: edge leaves byte 0 == 0x03 (CSR v3), non-edge leaves
#                byte 0 != 0x03 (BITSET v1).
#       - Run 4: edge leaves byte 0 == 0x03, non-edge leaves byte 0 == 0x02.
#   (Phase 3) topology_fwd.csr / topology_rev.csr absent under
#       graphStorage=CSR_HYBRID (design §3.8 D8 — CSR supersedes sidecar).
#   (Phase 4) `USE <proj> MATCH (n) RETURN count(n)` returns 2708 for all
#       four runs.
#   (Phase 5) `gnn_offline_sample(proj, name, [3, 5])` returns identical
#       uniqueNodes + totalBatches across all four runs.  The adjacency
#       semantic (src -> {dsts}) is invariant; this is the GNN workload
#       invariant Spec #8 G-series targets.
#   (Phase 6) Drop all 4 test projections.
#
# Note: full-triple semantic equality (src, dst, edge_id) is NOT asserted.
# BPTLeafCSRWriter currently emits flags=0 with no edge_id stream and the
# reader returns edge_id=0 for all tuples; that is a documented deferred
# item for Spec #8-B.
#
# Usage: ./scripts/test_projection_csr_hybrid.sh
# Exit:  0 on full pass, 1 on any assertion mismatch, 2 on setup/server error.
#
# Spec reference:
#   docs/superpowers/specs/2026-04-25-csr-hybrid-design.md §3.4 / §3.8
#   docs/superpowers/plans/2026-04-25-csr-hybrid-plan.md §T8.10
set -euo pipefail

MDB=${MDB:-./build/Release/bin/mdb}
DB=${DB:-data/dbs/gql/cora_gnn}
PORT=${PORT:-19850}
NODE_LABEL=${NODE_LABEL:-Paper}
EDGE_TYPE=${EDGE_TYPE:-CITES}
EXPECTED_NODE_COUNT=${EXPECTED_NODE_COUNT:-2708}

# --- prerequisites ---
if [[ ! -x "$MDB" ]]; then
    echo "ERROR: mdb binary not found at $MDB (run 'cmake --build build/Release' first)" >&2
    exit 2
fi
if [[ ! -d "$DB" ]]; then
    echo "ERROR: database directory not found at $DB (expected cora_gnn to be pre-imported)" >&2
    exit 2
fi

# --- 4 test projections: {BITSET, DELTA_VARINT} x {BTREE, CSR_HYBRID} ---
PROJS=(
    csrhyb_bitset_btree
    csrhyb_delta_btree
    csrhyb_bitset_csr
    csrhyb_delta_csr
)
FORMATS=(
    BITSET
    DELTA_VARINT
    BITSET
    DELTA_VARINT
)
STORAGES=(
    BTREE
    BTREE
    CSR_HYBRID
    CSR_HYBRID
)

# GNN_MINIMAL preset materializes exactly these 5 indexes.
EXPECTED_GNN_MINIMAL_NAMES=(
    nodes node_label label_node
    from_to_edge to_from_edge
)
# Edge indexes outside the GNN_MINIMAL preset — must be absent.
FORBIDDEN_GNN_MINIMAL_NAMES=(
    edge_direction edge_from_to edge_n1_n2
    edge_label label_edge
)
# Edge indexes that receive CSR v3 leaves under graphStorage=CSR_HYBRID.
# (For GNN_MINIMAL this is exactly from_to_edge + to_from_edge; the other
# edge indexes are absent.)
EDGE_INDEX_NAMES=(
    from_to_edge to_from_edge
)
# Non-edge indexes — always BITSET v1 or DELTA_VARINT v2 regardless of
# graphStorage (CSR_HYBRID only targets the edge indexes).
NON_EDGE_INDEX_NAMES=(
    nodes node_label label_node
)

# --- server lifecycle -------------------------------------------------------
SRV_PID=

cleanup() {
    if [[ -n "${SRV_PID:-}" ]] && kill -0 "$SRV_PID" 2>/dev/null; then
        kill "$SRV_PID" 2>/dev/null || true
        wait "$SRV_PID" 2>/dev/null || true
    fi
    for p in "${PROJS[@]}"; do
        "$MDB" drop-projection "$DB" "$p" >/dev/null 2>&1 || true
    done
    # Clean up phase 5 sample directories left under data/dbs/<db>/samples/.
    for i in 0 1 2 3; do
        rm -rf "$DB/samples/phase5_sample_${i}" 2>/dev/null || true
    done
}
trap cleanup EXIT

# Drop leftovers from a previous run so the script is idempotent.
for p in "${PROJS[@]}"; do
    "$MDB" drop-projection "$DB" "$p" >/dev/null 2>&1 || true
done

start_server() {
    local logfile="$1"
    "$MDB" server "$DB" --port "$PORT" --timeout 600 > "$logfile" 2>&1 &
    SRV_PID=$!
    local i
    for i in $(seq 1 20); do
        if curl -sSf -o /dev/null "http://127.0.0.1:$PORT/" \
                --data-binary "RETURN 1" -H "Accept: text/csv" 2>/dev/null; then
            return 0
        fi
        sleep 0.5
    done
    echo "ERROR: server on port $PORT did not come up within 10s." >&2
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

send_query() {
    local query="$1"
    curl -sS --max-time 600 --data-binary "$query" -H "Accept: text/csv" \
        "http://127.0.0.1:$PORT/" || true
}

run_projection() {
    local proj="$1"
    local format="$2"
    local storage="$3"
    local query
    query="CALL graph_project('$proj', '$NODE_LABEL', '$EDGE_TYPE', "
    query+="{orientation: 'NATURAL', indexSet: 'GNN_MINIMAL', "
    query+="leafFormat: '$format', graphStorage: '$storage'}) "
    query+="YIELD graphName, nodeCount, relCount RETURN graphName, nodeCount, relCount"
    local response
    response=$(send_query "$query")
    echo "    response: $(echo "$response" | tr '\n' ' ' | head -c 200)"
    if ! grep -q "$proj" <<<"$response"; then
        echo "ERROR: projection '$proj' not in response" >&2
        return 1
    fi
}

# --- phase 1: build all 4 projections ---------------------------------------
echo "=== T8.10: 4-mode golden compare on $(basename "$DB") ==="
echo "    graphStorage x leafFormat matrix (indexSet=GNN_MINIMAL):"
for i in 0 1 2 3; do
    echo "      [$i] ${PROJS[$i]} = ${FORMATS[$i]} x ${STORAGES[$i]}"
done
echo

LOG_DIR="/tmp/mdb_csrhyb_logs"
mkdir -p "$LOG_DIR"

START_TS=$(date +%s)

start_server "$LOG_DIR/server.log"
for i in 0 1 2 3; do
    proj="${PROJS[$i]}"
    format="${FORMATS[$i]}"
    storage="${STORAGES[$i]}"
    echo ">>> Building projection $proj (format=$format storage=$storage)..."
    run_projection "$proj" "$format" "$storage"
done
stop_server

# --- phase 1 continued: file presence per preset ----------------------------
FAIL=0

echo
echo "=== Phase 1: .leaf presence/absence per GNN_MINIMAL preset ==="

check_present_list() {
    local proj="$1"; shift
    for name in "$@"; do
        local f="$DB/projections/$proj/${name}.leaf"
        if [[ -f "$f" ]]; then
            echo "  OK    present: $proj/${name}.leaf"
        else
            echo "  FAIL  missing: $proj/${name}.leaf"
            FAIL=$((FAIL + 1))
        fi
    done
}

check_absent_list() {
    local proj="$1"; shift
    for name in "$@"; do
        local f="$DB/projections/$proj/${name}.leaf"
        if [[ -f "$f" ]]; then
            echo "  FAIL  unexpected: $proj/${name}.leaf"
            FAIL=$((FAIL + 1))
        else
            echo "  OK    absent:     $proj/${name}.leaf"
        fi
    done
}

for i in 0 1 2 3; do
    proj="${PROJS[$i]}"
    echo "--- $proj ---"
    check_present_list "$proj" "${EXPECTED_GNN_MINIMAL_NAMES[@]}"
    check_absent_list  "$proj" "${FORBIDDEN_GNN_MINIMAL_NAMES[@]}"
done

# --- phase 2: first-byte format dispatch per run ----------------------------
echo
echo "=== Phase 2: first-byte format dispatch per run ==="

hex_byte0() {
    local f="$1"
    if [[ ! -f "$f" ]]; then
        echo "--"
        return
    fi
    od -An -tx1 -N1 "$f" | tr -d ' \n'
}

check_byte0_eq() {
    local f="$1"
    local expected="$2"
    local tag="$3"
    local b
    b=$(hex_byte0 "$f")
    if [[ "$b" == "$expected" ]]; then
        echo "  OK    byte0=0x$b  $(basename "$(dirname "$f")")/$(basename "$f") ($tag)"
    else
        echo "  FAIL  byte0=0x$b (expected 0x$expected)  $(basename "$(dirname "$f")")/$(basename "$f") ($tag)"
        FAIL=$((FAIL + 1))
    fi
}

check_byte0_ne() {
    local f="$1"
    local forbidden="$2"
    local tag="$3"
    local b
    b=$(hex_byte0 "$f")
    if [[ "$b" != "$forbidden" ]]; then
        echo "  OK    byte0=0x$b (!= 0x$forbidden)  $(basename "$(dirname "$f")")/$(basename "$f") ($tag)"
    else
        echo "  FAIL  byte0=0x$b unexpectedly equals 0x$forbidden  $(basename "$(dirname "$f")")/$(basename "$f") ($tag)"
        FAIL=$((FAIL + 1))
    fi
}

for i in 0 1 2 3; do
    proj="${PROJS[$i]}"
    format="${FORMATS[$i]}"
    storage="${STORAGES[$i]}"
    echo "--- Run $((i+1)): $proj ($format x $storage) ---"

    # Edge leaves (from_to_edge, to_from_edge).
    for idx in "${EDGE_INDEX_NAMES[@]}"; do
        f="$DB/projections/$proj/${idx}.leaf"
        if [[ ! -f "$f" ]]; then
            echo "  SKIP  missing: $proj/${idx}.leaf"
            continue
        fi
        if [[ "$storage" == "CSR_HYBRID" ]]; then
            # Under CSR_HYBRID, edge leaves MUST be v3 (byte 0 == 0x03).
            check_byte0_eq "$f" "03" "EDGE/$storage"
        else
            # Under BTREE, edge leaves follow leafFormat.
            if [[ "$format" == "DELTA_VARINT" ]]; then
                check_byte0_eq "$f" "02" "EDGE/$format"
            else
                # BITSET — byte 0 is LSB of value_count (!= 0x03 in practice).
                check_byte0_ne "$f" "03" "EDGE/$format"
            fi
        fi
    done

    # Non-edge leaves — always follow leafFormat, never CSR regardless of
    # graphStorage. The test asserts explicitly that CSR_HYBRID does NOT
    # leak byte 0x03 into non-edge leaves.
    for idx in "${NON_EDGE_INDEX_NAMES[@]}"; do
        f="$DB/projections/$proj/${idx}.leaf"
        if [[ ! -f "$f" ]]; then
            echo "  SKIP  missing: $proj/${idx}.leaf"
            continue
        fi
        if [[ "$format" == "DELTA_VARINT" ]]; then
            check_byte0_eq "$f" "02" "NONEDGE/$format"
        else
            check_byte0_ne "$f" "03" "NONEDGE/$format"
        fi
    done
done

# --- phase 3: sidecar absence under CSR_HYBRID ------------------------------
echo
echo "=== Phase 3: topology_*.csr absence under graphStorage=CSR_HYBRID ==="

for i in 0 1 2 3; do
    proj="${PROJS[$i]}"
    storage="${STORAGES[$i]}"
    dir="$DB/projections/$proj"
    fwd="$dir/topology_fwd.csr"
    rev="$dir/topology_rev.csr"
    if [[ "$storage" == "CSR_HYBRID" ]]; then
        for f in "$fwd" "$rev"; do
            if [[ -f "$f" ]]; then
                echo "  FAIL  unexpected sidecar under CSR_HYBRID: $f"
                FAIL=$((FAIL + 1))
            else
                echo "  OK    absent: $proj/$(basename "$f")"
            fi
        done
    else
        # Under BTREE the sidecar is only written when
        # buildTopologySnapshot=true (off by default), so we don't assert
        # presence — just log.
        for f in "$fwd" "$rev"; do
            if [[ -f "$f" ]]; then
                echo "  INFO  present (BTREE, opt-in sidecar): $proj/$(basename "$f")"
            else
                echo "  INFO  absent (BTREE, default off):    $proj/$(basename "$f")"
            fi
        done
    fi
done

# --- phase 4: USE + count(n) per run ----------------------------------------
echo
echo "=== Phase 4: USE <proj> MATCH (n) RETURN count(n) ==="
start_server "$LOG_DIR/server_phase4.log"
for p in "${PROJS[@]}"; do
    q="USE $p MATCH (n) RETURN count(n)"
    r=$(send_query "$q")
    if grep -q "${EXPECTED_NODE_COUNT}" <<<"$r"; then
        echo "  OK    $p: count(n)=${EXPECTED_NODE_COUNT}"
    else
        echo "  FAIL  $p: expected count=${EXPECTED_NODE_COUNT}, got response: $(echo "$r" | tr '\n' ' ' | head -c 200)"
        FAIL=$((FAIL + 1))
    fi
done

# --- phase 5: GNN adjacency-semantic equality via gnn_offline_sample --------
# Fanouts [5, 3] + fixed randomSeed => deterministic seed set.  The
# uniqueNodes + totalBatches counts are adjacency-derived invariants: they
# depend only on the src -> {dsts} relation, not on the concrete ordering
# in which neighbors are listed.  Across the 4 storage+format combos the
# adjacency relation is identical, so these two scalars MUST match.
echo
echo "=== Phase 5: gnn_offline_sample adjacency-semantic equality ==="
declare -A SAMPLE_UNIQUE
declare -A SAMPLE_BATCHES
SAMPLE_CSV_FMT="sampleName, totalBatches, trainBatches, validationBatches, testBatches, uniqueNodes, computeMillis"
for i in 0 1 2 3; do
    proj="${PROJS[$i]}"
    sname="phase5_sample_${i}"
    # gnn_sample_drop to clear any leftover sample (samples are global per-DB,
    # not per-projection; use a distinct name per run).
    drop_q="CALL gnn_sample_drop('$sname')"
    send_query "$drop_q" >/dev/null 2>&1 || true

    sample_query="CALL gnn_offline_sample('$proj', '$sname', [3, 5], "
    sample_query+="{batchSize: 256, randomSeed: 42}) "
    sample_query+="YIELD $SAMPLE_CSV_FMT RETURN *"
    resp=$(send_query "$sample_query")
    # Parse CSV: header line + data line.
    data_line=$(echo "$resp" | sed -n '2p' | tr -d '"')
    if [[ -z "$data_line" ]]; then
        echo "  FAIL  $proj: gnn_offline_sample returned empty response"
        echo "        raw: $(echo "$resp" | tr '\n' ' ' | head -c 300)"
        FAIL=$((FAIL + 1))
        continue
    fi
    # CSV columns: sampleName, totalBatches, trainBatches, validationBatches,
    #              testBatches, uniqueNodes, computeMillis
    total_batches=$(echo "$data_line" | awk -F',' '{print $2}' | tr -d ' ')
    unique_nodes=$(echo "$data_line" | awk -F',' '{print $6}' | tr -d ' ')
    if [[ -z "$total_batches" || -z "$unique_nodes" ]]; then
        echo "  FAIL  $proj: could not parse totalBatches/uniqueNodes from: $data_line"
        FAIL=$((FAIL + 1))
        continue
    fi
    SAMPLE_BATCHES[$proj]="$total_batches"
    SAMPLE_UNIQUE[$proj]="$unique_nodes"
    echo "  INFO  $proj: totalBatches=$total_batches uniqueNodes=$unique_nodes"
done
stop_server

# Compare all 4 runs against run 0 as the baseline.
BASELINE="${PROJS[0]}"
BASE_BATCHES="${SAMPLE_BATCHES[$BASELINE]:-}"
BASE_UNIQUE="${SAMPLE_UNIQUE[$BASELINE]:-}"
if [[ -z "$BASE_BATCHES" || -z "$BASE_UNIQUE" ]]; then
    echo "  FAIL  baseline run $BASELINE missing sample output"
    FAIL=$((FAIL + 1))
else
    echo "  baseline ($BASELINE): totalBatches=$BASE_BATCHES uniqueNodes=$BASE_UNIQUE"
    for i in 1 2 3; do
        p="${PROJS[$i]}"
        tb="${SAMPLE_BATCHES[$p]:-}"
        un="${SAMPLE_UNIQUE[$p]:-}"
        if [[ "$tb" == "$BASE_BATCHES" && "$un" == "$BASE_UNIQUE" ]]; then
            echo "  OK    $p: totalBatches=$tb uniqueNodes=$un (matches baseline)"
        else
            echo "  FAIL  $p: totalBatches=$tb uniqueNodes=$un"
            echo "        baseline was: totalBatches=$BASE_BATCHES uniqueNodes=$BASE_UNIQUE"
            FAIL=$((FAIL + 1))
        fi
    done
fi

# --- phase 6: drop all 4 projections ----------------------------------------
echo
echo "=== Phase 6: drop all 4 test projections ==="
for p in "${PROJS[@]}"; do
    if "$MDB" drop-projection "$DB" "$p" >/dev/null 2>&1; then
        echo "  OK    dropped $p"
    else
        echo "  WARN  drop-projection failed for $p (may already be gone)"
    fi
done

# --- summary ----------------------------------------------------------------
END_TS=$(date +%s)
ELAPSED=$((END_TS - START_TS))
echo
echo "================================================================"
echo "  Elapsed: ${ELAPSED}s"
if [[ $FAIL -eq 0 ]]; then
    echo "ALL CHECKS PASSED"
    exit 0
else
    echo "FAILED: $FAIL checks"
    exit 1
fi
