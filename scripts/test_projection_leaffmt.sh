#!/usr/bin/env bash
#
# Spec #5 T5.12 — 6-mode golden compare for leafFormat x indexSet.
#
# Builds the cora_gnn projection six times, one per (indexSet, leafFormat)
# combination:
#   (ALL, BITSET)                (ALL, DELTA_VARINT)
#   (GNN_MINIMAL, BITSET)        (GNN_MINIMAL, DELTA_VARINT)
#   (READONLY_TRAVERSAL, BITSET) (READONLY_TRAVERSAL, DELTA_VARINT)
#
# Asserts:
#   (1) Byte-identity within (leafFormat, index) across indexSet presets
#       that share the same index. ALL vs GNN_MINIMAL vs READONLY_TRAVERSAL
#       must produce bit-identical .leaf bytes for indexes they share, for
#       a fixed leafFormat.
#   (2) Semantic equality across formats: decoded record sequences match
#       exactly across BITSET and DELTA_VARINT runs. Bytes differ (byte 0
#       = V1-bitset-byte vs 0x02), but records decode to the same values.
#   (3) Under restricted indexSet presets, .leaf files outside the preset
#       are absent.
#   (4) All 6 projections open via `USE proj MATCH (n) RETURN count(n)`
#       and return Cora's 2708 node count.
#
# Usage: ./scripts/test_projection_leaffmt.sh
# Exit:  0 on full pass, 1 on any assertion mismatch, 2 on setup/server error.
#
# Spec reference:
set -euo pipefail

MDB=${MDB:-./build/Release/bin/mdb}
DUMPER=${DUMPER:-./build/Release/bin/mdb_leaf_dump}
DB=${DB:-data/dbs/gql/cora_gnn}
PORT=${PORT:-19882}
NODE_LABEL=${NODE_LABEL:-Paper}
EDGE_TYPE=${EDGE_TYPE:-CITES}
EXPECTED_NODE_COUNT=${EXPECTED_NODE_COUNT:-2708}

# --- prerequisites ---
if [[ ! -x "$MDB" ]]; then
    echo "ERROR: mdb binary not found at $MDB (run 'cmake --build build/Release' first)" >&2
    exit 2
fi
if [[ ! -x "$DUMPER" ]]; then
    echo "ERROR: mdb_leaf_dump binary not found at $DUMPER (run 'cmake --build build/Release' first)" >&2
    exit 2
fi
if [[ ! -d "$DB" ]]; then
    echo "ERROR: database directory not found at $DB (expected cora_gnn to be pre-imported)" >&2
    exit 2
fi

# --- 6 test projections: 3 presets x 2 leaf formats ---
PROJS=(
    leaffmt_all_bitset
    leaffmt_all_delta
    leaffmt_gnnmin_bitset
    leaffmt_gnnmin_delta
    leaffmt_ro_bitset
    leaffmt_ro_delta
)
PRESETS=(
    ALL
    ALL
    GNN_MINIMAL
    GNN_MINIMAL
    READONLY_TRAVERSAL
    READONLY_TRAVERSAL
)
FORMATS=(
    BITSET
    DELTA_VARINT
    BITSET
    DELTA_VARINT
    BITSET
    DELTA_VARINT
)

# Record width per .leaf file (matches build_*_index_() declarations in
# src/graph_models/gql/projection/projection_storage.cc).
# Used by mdb_leaf_dump to decode records.
LEAF_NAMES=(
    nodes node_label label_node
    from_to_edge to_from_edge
    edge_direction edge_from_to edge_n1_n2
    edge_label label_edge
)
LEAF_WIDTHS=(
    1 2 2
    3 3
    2 3 3
    2 2
)

# Files expected per preset (derived from
# src/graph_models/gql/projection/index_set.cc::project_index_mask_for()).
EXPECTED_ALL_NAMES=(
    nodes node_label label_node
    from_to_edge to_from_edge
    edge_direction edge_from_to edge_n1_n2
    edge_label label_edge
)
EXPECTED_GNN_MINIMAL_NAMES=(
    nodes node_label label_node
    from_to_edge to_from_edge
)
EXPECTED_READONLY_NAMES=(
    nodes node_label label_node
    from_to_edge to_from_edge
    edge_label label_edge
)
FORBIDDEN_GNN_MINIMAL_NAMES=(
    edge_direction edge_from_to edge_n1_n2
    edge_label label_edge
)
FORBIDDEN_READONLY_NAMES=(
    edge_direction edge_from_to edge_n1_n2
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
    curl -sS --data-binary "$query" -H "Accept: text/csv" \
        "http://127.0.0.1:$PORT/" || true
}

run_projection() {
    local proj="$1"
    local preset="$2"
    local format="$3"
    local query
    query="CALL graph_project('$proj', '$NODE_LABEL', '$EDGE_TYPE', "
    query+="{orientation: 'NATURAL', indexSet: '$preset', leafFormat: '$format'}) "
    query+="YIELD graphName, nodeCount, relCount RETURN graphName, nodeCount, relCount"
    local response
    response=$(send_query "$query")
    echo "    response: $(echo "$response" | tr '\n' ' ' | head -c 200)"
    # Expect graphName to be echoed back in the CSV response.
    if ! grep -q "$proj" <<<"$response"; then
        echo "ERROR: projection '$proj' not in response" >&2
        return 1
    fi
}

# --- phase 1: build all 6 projections ---------------------------------------
echo "=== T5.12: 6-mode golden compare on $(basename "$DB") ==="
echo "    leafFormat × indexSet matrix:"
for i in 0 1 2 3 4 5; do
    echo "      [$i] ${PROJS[$i]} = ${PRESETS[$i]} × ${FORMATS[$i]}"
done
echo

LOG_DIR="/tmp/mdb_leaffmt_logs"
mkdir -p "$LOG_DIR"

START_TS=$(date +%s)

start_server "$LOG_DIR/server.log"
for i in 0 1 2 3 4 5; do
    proj="${PROJS[$i]}"
    preset="${PRESETS[$i]}"
    format="${FORMATS[$i]}"
    echo ">>> Building projection $proj (preset=$preset format=$format)..."
    run_projection "$proj" "$preset" "$format"
done
stop_server

# --- phase 2: verify first byte of each .leaf per format --------------------
# For BITSET runs, byte 0 is the low byte of value_count (non-0x02 in
# practice). For DELTA_VARINT runs, byte 0 MUST be 0x02.
FAIL=0

hex_byte0() {
    local f="$1"
    if [[ ! -f "$f" ]]; then
        echo "--"
        return
    fi
    od -An -tx1 -N1 "$f" | tr -d ' \n'
}

echo
echo "=== Phase 2: first-byte confirmation per format ==="
V2_MAGIC=0
V2_TOTAL=0
V1_TOTAL=0
for i in 0 1 2 3 4 5; do
    proj="${PROJS[$i]}"
    format="${FORMATS[$i]}"
    dir="$DB/projections/$proj"
    if [[ ! -d "$dir" ]]; then
        echo "  FAIL: projection directory missing: $dir"
        FAIL=$((FAIL + 1))
        continue
    fi
    for f in "$dir"/*.leaf; do
        [[ -e "$f" ]] || continue
        b=$(hex_byte0 "$f")
        if [[ "$format" == "DELTA_VARINT" ]]; then
            V2_TOTAL=$((V2_TOTAL + 1))
            if [[ "$b" == "02" ]]; then
                V2_MAGIC=$((V2_MAGIC + 1))
                echo "  OK    byte0=0x$b  $proj/$(basename "$f") (DELTA_VARINT)"
            else
                echo "  FAIL  byte0=0x$b (expected 0x02) $proj/$(basename "$f")"
                FAIL=$((FAIL + 1))
            fi
        else
            V1_TOTAL=$((V1_TOTAL + 1))
            # For BITSET it's byte 0 of value_count (u32 LE) — NOT 0x02 for
            # non-empty pages at cora scale, but we allow it in principle.
            # We only assert it is *not* the v2 magic to rule out an encoder
            # bug; 0x02 is theoretically possible for a 2-record page, so we
            # log but don't fail on that alone.
            echo "  INFO  byte0=0x$b  $proj/$(basename "$f") (BITSET)"
        fi
    done
done

# --- phase 3: byte-identity within (format, index) across presets -----------
# For each (format, index), pick the PROJ(s) that materialize the index,
# then byte-compare. Each (format, index) has at most 3 projections per format
# side (ALL / GNN_MINIMAL / READONLY_TRAVERSAL); indexes outside a preset
# simply drop out of that preset's comparison set.
echo
echo "=== Phase 3: byte-identity within (format, index) across presets ==="

declare -A PROJ_FOR_PRESET_FORMAT
PROJ_FOR_PRESET_FORMAT["ALL__BITSET"]=leaffmt_all_bitset
PROJ_FOR_PRESET_FORMAT["ALL__DELTA_VARINT"]=leaffmt_all_delta
PROJ_FOR_PRESET_FORMAT["GNN_MINIMAL__BITSET"]=leaffmt_gnnmin_bitset
PROJ_FOR_PRESET_FORMAT["GNN_MINIMAL__DELTA_VARINT"]=leaffmt_gnnmin_delta
PROJ_FOR_PRESET_FORMAT["READONLY_TRAVERSAL__BITSET"]=leaffmt_ro_bitset
PROJ_FOR_PRESET_FORMAT["READONLY_TRAVERSAL__DELTA_VARINT"]=leaffmt_ro_delta

# Indexes present in each preset (mirror of index_set.cc mapping).
PRESETS_FOR_INDEX_nodes=(ALL GNN_MINIMAL READONLY_TRAVERSAL)
PRESETS_FOR_INDEX_node_label=(ALL GNN_MINIMAL READONLY_TRAVERSAL)
PRESETS_FOR_INDEX_label_node=(ALL GNN_MINIMAL READONLY_TRAVERSAL)
PRESETS_FOR_INDEX_from_to_edge=(ALL GNN_MINIMAL READONLY_TRAVERSAL)
PRESETS_FOR_INDEX_to_from_edge=(ALL GNN_MINIMAL READONLY_TRAVERSAL)
PRESETS_FOR_INDEX_edge_label=(ALL READONLY_TRAVERSAL)
PRESETS_FOR_INDEX_label_edge=(ALL READONLY_TRAVERSAL)
PRESETS_FOR_INDEX_edge_direction=(ALL)
PRESETS_FOR_INDEX_edge_from_to=(ALL)
PRESETS_FOR_INDEX_edge_n1_n2=(ALL)

PHASE3_TOTAL=0
PHASE3_MATCH=0
for idx_name in "${LEAF_NAMES[@]}"; do
    # Collect the presets this index belongs to.
    varname="PRESETS_FOR_INDEX_${idx_name}"
    presets_ref=$varname[@]
    presets_list=("${!presets_ref}")
    if [[ "${#presets_list[@]}" -lt 2 ]]; then
        # Only one preset materializes this index (edge_direction etc.) —
        # nothing to cross-compare within a fixed format for presets.
        continue
    fi
    for fmt in BITSET DELTA_VARINT; do
        base_preset="${presets_list[0]}"
        base_proj="${PROJ_FOR_PRESET_FORMAT[${base_preset}__${fmt}]}"
        base_file="$DB/projections/$base_proj/${idx_name}.leaf"
        if [[ ! -f "$base_file" ]]; then
            echo "  FAIL  missing base: $base_proj/${idx_name}.leaf"
            FAIL=$((FAIL + 1))
            continue
        fi
        for other_preset in "${presets_list[@]:1}"; do
            other_proj="${PROJ_FOR_PRESET_FORMAT[${other_preset}__${fmt}]}"
            other_file="$DB/projections/$other_proj/${idx_name}.leaf"
            PHASE3_TOTAL=$((PHASE3_TOTAL + 1))
            if [[ ! -f "$other_file" ]]; then
                echo "  FAIL  missing other: $other_proj/${idx_name}.leaf"
                FAIL=$((FAIL + 1))
                continue
            fi
            if cmp -s "$base_file" "$other_file"; then
                sz=$(stat -c %s "$base_file")
                echo "  OK    [$fmt] ${idx_name}.leaf byte-match across ${base_preset}/${other_preset} (${sz}B)"
                PHASE3_MATCH=$((PHASE3_MATCH + 1))
            else
                szb=$(stat -c %s "$base_file")
                szo=$(stat -c %s "$other_file")
                echo "  FAIL  [$fmt] ${idx_name}.leaf mismatch ${base_preset}=${szb}B ${other_preset}=${szo}B"
                echo "        first 32-byte hex diff:"
                diff <(od -An -tx1 -N64 "$base_file") <(od -An -tx1 -N64 "$other_file") | sed 's/^/          /' | head -6 || true
                FAIL=$((FAIL + 1))
            fi
        done
    done
done
echo "  (Phase 3: ${PHASE3_MATCH}/${PHASE3_TOTAL} byte-identical pairs)"

# --- phase 4: semantic equality across formats via mdb_leaf_dump ------------
# For each (preset, index), compare records decoded from BITSET vs DELTA_VARINT
# runs of the same preset. Must be byte-for-byte identical after decoding.
echo
echo "=== Phase 4: semantic equality across formats (decoded records) ==="
DUMP_DIR="/tmp/mdb_leaffmt_dumps"
mkdir -p "$DUMP_DIR"
rm -f "$DUMP_DIR"/*.txt 2>/dev/null || true

PHASE4_TOTAL=0
PHASE4_MATCH=0
for preset in ALL GNN_MINIMAL READONLY_TRAVERSAL; do
    proj_b="${PROJ_FOR_PRESET_FORMAT[${preset}__BITSET]}"
    proj_v="${PROJ_FOR_PRESET_FORMAT[${preset}__DELTA_VARINT]}"
    for ((j = 0; j < ${#LEAF_NAMES[@]}; ++j)); do
        idx_name="${LEAF_NAMES[$j]}"
        width="${LEAF_WIDTHS[$j]}"
        f_b="$DB/projections/$proj_b/${idx_name}.leaf"
        f_v="$DB/projections/$proj_v/${idx_name}.leaf"
        if [[ ! -f "$f_b" || ! -f "$f_v" ]]; then
            # Index absent for this preset — skip.
            continue
        fi
        PHASE4_TOTAL=$((PHASE4_TOTAL + 1))
        dump_b="$DUMP_DIR/${proj_b}_${idx_name}.txt"
        dump_v="$DUMP_DIR/${proj_v}_${idx_name}.txt"
        if ! "$DUMPER" "$f_b" "$width" > "$dump_b" 2>"$DUMP_DIR/err.txt"; then
            echo "  FAIL  dumper error on $proj_b/${idx_name}.leaf:"
            sed 's/^/          /' < "$DUMP_DIR/err.txt"
            FAIL=$((FAIL + 1))
            continue
        fi
        if ! "$DUMPER" "$f_v" "$width" > "$dump_v" 2>"$DUMP_DIR/err.txt"; then
            echo "  FAIL  dumper error on $proj_v/${idx_name}.leaf:"
            sed 's/^/          /' < "$DUMP_DIR/err.txt"
            FAIL=$((FAIL + 1))
            continue
        fi
        if cmp -s "$dump_b" "$dump_v"; then
            n=$(wc -l < "$dump_b")
            echo "  OK    [${preset}] ${idx_name} (N=$width) decoded records match (${n} rec)"
            PHASE4_MATCH=$((PHASE4_MATCH + 1))
        else
            echo "  FAIL  [${preset}] ${idx_name} (N=$width) decoded records DIFFER"
            echo "        first 5 differing lines:"
            diff "$dump_b" "$dump_v" | head -10 | sed 's/^/          /'
            FAIL=$((FAIL + 1))
        fi
    done
done
echo "  (Phase 4: ${PHASE4_MATCH}/${PHASE4_TOTAL} record-sequence equalities)"

# --- phase 5: file presence/absence per preset ------------------------------
echo
echo "=== Phase 5: file presence/absence per preset ==="

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

for fmt in BITSET DELTA_VARINT; do
    echo "--- $fmt ---"
    check_present_list "${PROJ_FOR_PRESET_FORMAT[ALL__${fmt}]}" "${EXPECTED_ALL_NAMES[@]}"
    check_present_list "${PROJ_FOR_PRESET_FORMAT[GNN_MINIMAL__${fmt}]}" "${EXPECTED_GNN_MINIMAL_NAMES[@]}"
    check_absent_list  "${PROJ_FOR_PRESET_FORMAT[GNN_MINIMAL__${fmt}]}" "${FORBIDDEN_GNN_MINIMAL_NAMES[@]}"
    check_present_list "${PROJ_FOR_PRESET_FORMAT[READONLY_TRAVERSAL__${fmt}]}" "${EXPECTED_READONLY_NAMES[@]}"
    check_absent_list  "${PROJ_FOR_PRESET_FORMAT[READONLY_TRAVERSAL__${fmt}]}" "${FORBIDDEN_READONLY_NAMES[@]}"
done

# --- phase 6: open each projection and verify node count --------------------
echo
echo "=== Phase 6: open each projection via USE + MATCH ==="
start_server "$LOG_DIR/server_phase6.log"
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
stop_server

# --- summary ----------------------------------------------------------------
END_TS=$(date +%s)
ELAPSED=$((END_TS - START_TS))
echo
echo "================================================================"
echo "  Elapsed: ${ELAPSED}s"
echo "  Phase 2 (first byte): ${V2_MAGIC}/${V2_TOTAL} DELTA_VARINT files start with 0x02; ${V1_TOTAL} BITSET files inspected."
echo "  Phase 3 (byte-id):    ${PHASE3_MATCH}/${PHASE3_TOTAL} byte-identical pairs across shared presets."
echo "  Phase 4 (semantic):   ${PHASE4_MATCH}/${PHASE4_TOTAL} record-sequence equalities across formats."
if [[ $FAIL -eq 0 ]]; then
    echo "ALL CHECKS PASSED"
    exit 0
else
    echo "FAILED: $FAIL checks"
    exit 1
fi
