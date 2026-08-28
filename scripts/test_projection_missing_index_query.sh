#!/bin/bash
# Spec #3 T3.9 — Query-layer missing-index diagnostic integration test.
#
# Verifies that GQL queries on projections built under a restricted IndexSet
# preset raise a friendly QueryException — not segfault, not generic
# null-deref — when they hit an index that was not materialized.
#
# Scenarios:
#   1. GNN_MINIMAL projection + node-only traversal  -> succeeds
#   2. GNN_MINIMAL projection + edge-label filter     -> friendly error
#   3. GNN_MINIMAL projection + property access       -> friendly error
#   4. READONLY_TRAVERSAL + edge-first edge_from_to   -> friendly error
#   5. ALL projection + every query shape             -> succeeds
#
# Each expected error must contain:
#   - the literal string "Cannot execute query - index '"
#   - the projection name
#   - the current indexSet preset in the form "indexSet='<NAME>'"
#   - a rebuild suggestion in the form "indexSet='<NAME>'"
#
# Usage: ./scripts/test_projection_missing_index_query.sh
# Exit:  0 on all checks pass, 1 on any failure, 2 on setup/server error.

set -euo pipefail

MDB=${MDB:-./build/Release/bin/mdb}
DB=${DB:-data/dbs/gql/cora_gnn}
PORT=${PORT:-19884}
NODE_LABEL=${NODE_LABEL:-Paper}
EDGE_TYPE=${EDGE_TYPE:-CITES}

PROJS=(t3_9_gnn_proj t3_9_ro_proj t3_9_all_proj)

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
    "$MDB" server "$DB" --port "$PORT" --timeout 300 \
        > /tmp/mdb_t3_9_server.log 2>&1 &
    SRV_PID=$!
    for i in $(seq 1 20); do
        if curl -sSf -o /dev/null "http://127.0.0.1:$PORT/" \
                --data-binary "RETURN 1" -H "Accept: text/csv" 2>/dev/null; then
            return 0
        fi
        sleep 0.5
    done
    echo "ERROR: server on port $PORT did not come up within 10 s." >&2
    cat /tmp/mdb_t3_9_server.log >&2 || true
    return 2
}

create_projection() {
    local proj="$1" preset="$2"
    local q="CALL graph_project('$proj', '$NODE_LABEL', '$EDGE_TYPE', {orientation: 'NATURAL', indexSet: '$preset'}) YIELD graphName RETURN graphName"
    curl -sS --data-binary "$q" -H "Accept: text/csv" \
        "http://127.0.0.1:$PORT/" > /dev/null
}

# Run a query and echo the raw response (trimmed).
run_query() {
    local q="$1"
    curl -sS --data-binary "$q" -H "Accept: text/csv" \
        "http://127.0.0.1:$PORT/" 2>/dev/null
}

FAIL=0
check_contains() {
    local label="$1" response="$2" needle="$3"
    if echo "$response" | grep -Fq "$needle"; then
        echo "  OK    $label contains: $needle"
    else
        echo "  FAIL  $label missing:  $needle"
        echo "        response: $(echo "$response" | tr '\n' ' ')"
        FAIL=$((FAIL + 1))
    fi
}
check_not_contains() {
    local label="$1" response="$2" needle="$3"
    if echo "$response" | grep -Fq "$needle"; then
        echo "  FAIL  $label unexpectedly contains: $needle"
        echo "        response: $(echo "$response" | tr '\n' ' ')"
        FAIL=$((FAIL + 1))
    else
        echo "  OK    $label does not contain: $needle"
    fi
}

echo "=== T3.9: Query-layer missing-index diagnostic on $(basename "$DB") ==="
echo
start_server

# --- Build projections under 3 presets ---
echo ">>> Creating t3_9_gnn_proj (GNN_MINIMAL)..."
create_projection t3_9_gnn_proj GNN_MINIMAL
echo ">>> Creating t3_9_ro_proj (READONLY_TRAVERSAL)..."
create_projection t3_9_ro_proj READONLY_TRAVERSAL
echo ">>> Creating t3_9_all_proj (ALL)..."
create_projection t3_9_all_proj ALL
echo

# --- Test 1: GNN_MINIMAL legitimate query succeeds ---
echo "--- Test 1: GnnMinimalLegitimateQuerySucceeds ---"
resp=$(run_query "USE t3_9_gnn_proj MATCH (a)-[r]->(b) RETURN count(*)")
check_contains "response"   "$resp" "5429"
check_not_contains "response" "$resp" "Cannot execute query"
echo

# --- Test 2: GNN_MINIMAL edge-label-filter query raises ---
echo "--- Test 2: GnnMinimalEdgeLabelFilterRaises ---"
resp=$(run_query "USE t3_9_gnn_proj MATCH (a)-[r:CITES]->(b) RETURN count(*)")
check_contains "response" "$resp" "Cannot execute query - index '"
check_contains "response" "$resp" "'t3_9_gnn_proj'"
check_contains "response" "$resp" "indexSet='GNN_MINIMAL'"
check_contains "response" "$resp" "rebuild the projection with indexSet='ALL'"
echo

# --- Test 3: GNN_MINIMAL property access raises ---
echo "--- Test 3: GnnMinimalPropertyAccessRaises ---"
resp=$(run_query "USE t3_9_gnn_proj MATCH (a) RETURN a.label LIMIT 3")
check_contains "response" "$resp" "Cannot execute query - index '"
check_contains "response" "$resp" "node_key_value"
check_contains "response" "$resp" "indexSet='GNN_MINIMAL'"
echo

# --- Test 4: READONLY_TRAVERSAL edge-first (edge_from_to) raises ---
echo "--- Test 4: ReadonlyTraversalEdgeFromToAccessRaises ---"
resp=$(run_query "USE t3_9_ro_proj MATCH (a)-[r:CITES]->(b) RETURN count(*)")
check_contains "response" "$resp" "Cannot execute query - index '"
check_contains "response" "$resp" "edge_from_to"
check_contains "response" "$resp" "indexSet='READONLY_TRAVERSAL'"
check_contains "response" "$resp" "rebuild the projection with indexSet='ALL'"
echo

# --- Test 5: ALL mode topology queries all succeed ---
# Note: STRING-syntax graph_project ('Paper','CITES',...) doesn't include
# property indexes even under IndexSet=ALL, because property indexes are
# gated by includeProperties config (Spec #3 §3.4), not by IndexSet. So
# this test covers the label / edge-label / traversal queries only.
echo "--- Test 5: AllModeTopologyQueriesSucceed ---"
resp=$(run_query "USE t3_9_all_proj MATCH (a)-[r]->(b) RETURN count(*)")
check_contains "response"   "$resp" "5429"
check_not_contains "response" "$resp" "Cannot execute query"

resp=$(run_query "USE t3_9_all_proj MATCH (a)-[r:CITES]->(b) RETURN count(*)")
check_contains "response"   "$resp" "5429"
check_not_contains "response" "$resp" "Cannot execute query"
echo

# --- Test 6: error message includes projection name ---
echo "--- Test 6: ErrorMessageContainsProjectionName ---"
resp=$(run_query "USE t3_9_gnn_proj MATCH (a)-[r:CITES]->(b) RETURN count(*)")
check_contains "response" "$resp" "t3_9_gnn_proj"
echo

# --- Test 7: error message includes active indexSet name ---
echo "--- Test 7: ErrorMessageContainsIndexSetName ---"
resp=$(run_query "USE t3_9_ro_proj MATCH (a)-[r:CITES]->(b) RETURN count(*)")
check_contains "response" "$resp" "indexSet='READONLY_TRAVERSAL'"
echo

# --- Test 8: error message suggests rebuild preset ---
echo "--- Test 8: ErrorMessageSuggestsRebuild ---"
resp=$(run_query "USE t3_9_gnn_proj MATCH (a)-[r:CITES]->(b) RETURN count(*)")
check_contains "response" "$resp" "rebuild the projection with indexSet='"
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
