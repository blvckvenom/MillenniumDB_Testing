#!/bin/bash
# Golden compare: CLASSIC vs RADIX projection backends on cora_gnn.
#
# Runs the same projection twice (different MDB_PROJECTION_SORTER env values),
# then byte-compares every .leaf / .dir B+Tree file. A match proves functional
# equivalence at the on-disk B+Tree level — stronger than query-result
# equivalence alone (which the 347/347 GQL test suite already validates).
#
# Usage:  ./scripts/test_projection_radix.sh
# Exit:   0 on ALL MATCH, 1 on mismatch, 2 on setup/server error.
#
# Spec reference: docs/superpowers/specs/2026-04-21-radix-partition-sort-design.md §8.4
set -euo pipefail

MDB=${MDB:-./build/Release/bin/mdb}
DB=${DB:-data/dbs/gql/cora_gnn}
PORT=${PORT:-19880}
CLASSIC_PROJ=test_golden_classic
RADIX_PROJ=test_golden_radix

# Cora schema. If your cora_gnn import used different labels/types, override via
# NODE_LABEL=... EDGE_TYPE=... envs.
NODE_LABEL=${NODE_LABEL:-Paper}
EDGE_TYPE=${EDGE_TYPE:-CITES}

cleanup() {
    if [[ -n "${SRV_PID:-}" ]] && kill -0 "$SRV_PID" 2>/dev/null; then
        kill "$SRV_PID" 2>/dev/null || true
        wait "$SRV_PID" 2>/dev/null || true
    fi
    "$MDB" drop-projection "$DB" "$CLASSIC_PROJ" 2>/dev/null || true
    "$MDB" drop-projection "$DB" "$RADIX_PROJ" 2>/dev/null || true
}
trap cleanup EXIT

run_projection() {
    local backend="$1"
    local proj_name="$2"

    echo ">>> [$backend] Starting server on port $PORT..."
    MDB_PROJECTION_SORTER="$backend" "$MDB" server "$DB" --port "$PORT" --timeout 600 \
        > "/tmp/mdb_golden_${backend}.log" 2>&1 &
    SRV_PID=$!

    # Wait for server to accept connections (max 10 s).
    for i in $(seq 1 20); do
        if curl -sSf -o /dev/null "http://127.0.0.1:$PORT/" \
                --data-binary "RETURN 1" -H "Accept: text/csv" 2>/dev/null; then
            break
        fi
        sleep 0.5
    done

    echo ">>> [$backend] Creating projection '$proj_name'..."
    local query="CALL graph_project('$proj_name', '$NODE_LABEL', '$EDGE_TYPE') YIELD graphName, nodeCount, relCount RETURN *"
    local response
    response=$(curl -sS --data-binary "$query" -H "Accept: text/csv" "http://127.0.0.1:$PORT/" || true)
    echo "$response" | head -5

    kill "$SRV_PID" 2>/dev/null || true
    wait "$SRV_PID" 2>/dev/null || true
    SRV_PID=
    echo ">>> [$backend] Server stopped; projection written."
}

echo "=== Task 13: Golden compare RADIX vs CLASSIC on $(basename $DB) ==="
run_projection classic "$CLASSIC_PROJ"
run_projection radix   "$RADIX_PROJ"

CLASSIC_DIR="$DB/projections/$CLASSIC_PROJ"
RADIX_DIR="$DB/projections/$RADIX_PROJ"

if [[ ! -d "$CLASSIC_DIR" || ! -d "$RADIX_DIR" ]]; then
    echo "ERROR: one or both projection directories missing." >&2
    echo "  CLASSIC: $CLASSIC_DIR"
    echo "  RADIX:   $RADIX_DIR"
    exit 2
fi

echo
echo "=== Byte-comparing B+Tree files ==="
MISMATCH=0
TOTAL=0

for f_classic in "$CLASSIC_DIR"/*.leaf "$CLASSIC_DIR"/*.dir; do
    [[ -e "$f_classic" ]] || continue
    rel="${f_classic#$CLASSIC_DIR/}"
    f_radix="$RADIX_DIR/$rel"
    TOTAL=$((TOTAL + 1))

    if [[ ! -f "$f_radix" ]]; then
        echo "MISSING in radix: $rel"
        MISMATCH=$((MISMATCH + 1))
        continue
    fi
    if ! cmp -s "$f_classic" "$f_radix"; then
        sz_c=$(stat -c %s "$f_classic")
        sz_r=$(stat -c %s "$f_radix")
        echo "MISMATCH: $rel (classic=${sz_c}B, radix=${sz_r}B)"
        MISMATCH=$((MISMATCH + 1))
    else
        sz=$(stat -c %s "$f_classic")
        echo "MATCH:    $rel (${sz}B)"
    fi
done

echo
if [[ $MISMATCH -eq 0 && $TOTAL -gt 0 ]]; then
    echo "ALL MATCH — golden test PASSED on $TOTAL files."
    exit 0
else
    echo "FAILED — $MISMATCH of $TOTAL files differ."
    exit 1
fi
