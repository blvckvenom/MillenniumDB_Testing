#!/bin/bash
# Golden compare: all 4 (scan × sort) mode combinations on cora_gnn.
#
# Runs graph_project 4 times, one per (MDB_PROJECTION_SERIAL_SCAN, MDB_PROJECTION_SORTER)
# combination: classic-classic (cc), classic-radix (cr), serial-classic (sc), serial-radix (sr).
# Every .leaf / .dir B+Tree file from the baseline (cc) is byte-compared against the same
# file from the other 3 modes (~20 files × 3 pairs = ~60 cmp operations total).
#
# A clean run proves invariant I4: bit-identical B+Tree output across all mode combinations,
# confirming both backends are pure-refactor alternatives to the classic path.
#
# Usage:  ./scripts/test_projection_radix.sh
# Exit:   0 on ALL MATCH, 1 on mismatch, 2 on setup/server error.
#
# Spec reference: docs/superpowers/specs/2026-04-21-radix-partition-sort-design.md §6 (I4), §8 (T1)
set -euo pipefail

MDB=${MDB:-./build/Release/bin/mdb}
DB=${DB:-data/dbs/gql/cora_gnn}
PORT=${PORT:-19880}

# Cora schema. Override via NODE_LABEL=... EDGE_TYPE=... envs if needed.
NODE_LABEL=${NODE_LABEL:-Paper}
EDGE_TYPE=${EDGE_TYPE:-CITES}

# 4 projections: (scan=classic|serial) × (sort=classic|radix)
PROJS=(test_golden_cc test_golden_cr test_golden_sc test_golden_sr)
SCAN_MODES=(0 0 1 1)
SORT_MODES=(classic radix classic radix)

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

run_projection() {
    local scan="$1"
    local sort="$2"
    local proj="$3"

    echo ">>> [scan=$scan sort=$sort] Starting server on port $PORT..."
    MDB_PROJECTION_SERIAL_SCAN="$scan" \
    MDB_PROJECTION_SORTER="$sort" \
        "$MDB" server "$DB" --port "$PORT" --timeout 600 \
        > "/tmp/mdb_golden_${scan}_${sort}.log" 2>&1 &
    SRV_PID=$!

    # Wait for server to accept connections (max 10 s).
    for i in $(seq 1 20); do
        if curl -sSf -o /dev/null "http://127.0.0.1:$PORT/" \
                --data-binary "RETURN 1" -H "Accept: text/csv" 2>/dev/null; then
            break
        fi
        sleep 0.5
    done

    echo ">>> [scan=$scan sort=$sort] Creating projection '$proj'..."
    local query="CALL graph_project('$proj', '$NODE_LABEL', '$EDGE_TYPE') YIELD graphName, nodeCount, relCount RETURN *"
    local response
    response=$(curl -sS --data-binary "$query" -H "Accept: text/csv" "http://127.0.0.1:$PORT/" || true)
    echo "$response" | head -5

    kill "$SRV_PID" 2>/dev/null || true
    wait "$SRV_PID" 2>/dev/null || true
    SRV_PID=
    echo ">>> [scan=$scan sort=$sort] Server stopped; projection written."
    echo
}

echo "=== Task 13: Golden compare — 4 scan×sort mode combinations on $(basename "$DB") ==="
echo "    Modes: cc (classic/classic)  cr (classic/radix)"
echo "           sc (serial/classic)   sr (serial/radix)"
echo

for i in 0 1 2 3; do
    run_projection "${SCAN_MODES[$i]}" "${SORT_MODES[$i]}" "${PROJS[$i]}"
done

# Verify all projection directories exist before comparing.
for p in "${PROJS[@]}"; do
    pdir="$DB/projections/$p"
    if [[ ! -d "$pdir" ]]; then
        echo "ERROR: projection directory missing: $pdir" >&2
        exit 2
    fi
done

BASE="${PROJS[0]}"
BASE_DIR="$DB/projections/$BASE"

echo "=== Byte-comparing B+Tree files (baseline: $BASE) ==="
MISMATCH=0
TOTAL=0

for other in "${PROJS[@]:1}"; do
    OTHER_DIR="$DB/projections/$other"
    echo "--- $BASE vs $other ---"
    for f in "$BASE_DIR"/*.leaf "$BASE_DIR"/*.dir; do
        [[ -e "$f" ]] || continue
        name=$(basename "$f")
        f2="$OTHER_DIR/$name"
        TOTAL=$((TOTAL + 1))

        if [[ ! -f "$f2" ]]; then
            echo "MISSING in $other: $name"
            MISMATCH=$((MISMATCH + 1))
        elif ! cmp -s "$f" "$f2"; then
            sz_b=$(stat -c %s "$f")
            sz_o=$(stat -c %s "$f2")
            echo "MISMATCH: $name (${BASE}=${sz_b}B, ${other}=${sz_o}B)"
            MISMATCH=$((MISMATCH + 1))
        else
            sz=$(stat -c %s "$f")
            echo "MATCH:    $name (${sz}B)"
        fi
    done
    echo
done

if [[ $MISMATCH -eq 0 && $TOTAL -gt 0 ]]; then
    echo "ALL 4 MODES MATCH — golden compare PASSED on $TOTAL files x 3 pairs"
    exit 0
else
    echo "FAILED — $MISMATCH mismatches across $TOTAL files x 3 pairs"
    exit 1
fi
