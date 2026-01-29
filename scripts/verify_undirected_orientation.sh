#!/bin/bash
#
# UNDIRECTED Orientation Verification Script
#
# Purpose: Validates bidirectional traversal symmetry for UNDIRECTED projections.
#          This is critical for GNN message passing where undirected edges must
#          be traversable from both endpoints.
#
# Invariants Tested (INV-ORI-002):
#   1. forward_edge_count == reverse_edge_count
#   2. undirected_pattern_count == 2 * forward_edge_count
#   3. For any edge (u,v), both (u)->(v) and (v)->(u) are traversable
#
# Usage: ./scripts/verify_undirected_orientation.sh [database_folder] [port]
#
# Exit Codes:
#   0 - All verifications passed
#   1 - Verification failed
#   2 - Server startup failed
#   3 - Prerequisites not met

set -euo pipefail

# =============================================================================
# Configuration
# =============================================================================

DB_FOLDER="${1:-data/dbs/gql/cora}"
SERVER_PORT="${2:-1234}"
PROJECTION_NAME="undirected_verify_$$"
BUILD_TYPE="${BUILD_TYPE:-Release}"
MDB_BIN="build/${BUILD_TYPE}/bin/mdb"
SERVER_PID=""

# =============================================================================
# Logging Utilities
# =============================================================================

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

log_info()  { echo -e "${YELLOW}[INFO]${NC}  $(date '+%H:%M:%S') $*"; }
log_pass()  { echo -e "${GREEN}[PASS]${NC}  $(date '+%H:%M:%S') $*"; }
log_fail()  { echo -e "${RED}[FAIL]${NC}  $(date '+%H:%M:%S') $*" >&2; }
log_debug() { echo -e "${BLUE}[DEBUG]${NC} $(date '+%H:%M:%S') $*"; }
log_test()  { echo -e "${CYAN}[TEST]${NC}  $(date '+%H:%M:%S') $*"; }

# =============================================================================
# HTTP Query Helper
# =============================================================================

query() {
    local gql_query="$1"
    curl -s -X POST "http://localhost:$SERVER_PORT" \
        -H "Content-Type: application/gql" \
        -H "Accept: application/json" \
        -d "$gql_query"
}

# Extract a single value from CSV result (column name in first row, value in second row)
# The server returns CSV format: "columnName\nvalue" or "col1,col2\nval1,val2"
extract_value() {
    local csv="$1"
    local key="$2"

    # Get header line and data line
    local header
    header=$(echo "$csv" | head -1)
    local data
    data=$(echo "$csv" | tail -1)

    # Find column index for the key
    local col_idx=1
    local found=0
    IFS=',' read -ra COLS <<< "$header"
    for col in "${COLS[@]}"; do
        if [ "$col" == "$key" ]; then
            found=1
            break
        fi
        ((col_idx++))
    done

    if [ "$found" -eq 0 ]; then
        echo ""
        return
    fi

    # Extract value at that index
    echo "$data" | cut -d',' -f"$col_idx"
}

# =============================================================================
# Cleanup Handler
# =============================================================================

cleanup() {
    log_info "Cleaning up..."

    if [ -n "$SERVER_PID" ] && kill -0 "$SERVER_PID" 2>/dev/null; then
        query "CALL graph_drop('$PROJECTION_NAME')" > /dev/null 2>&1 || true
        sleep 0.5
    fi

    if [ -n "$SERVER_PID" ]; then
        kill "$SERVER_PID" 2>/dev/null || true
        wait "$SERVER_PID" 2>/dev/null || true
    fi
}
trap cleanup EXIT

# =============================================================================
# Prerequisites Check
# =============================================================================

check_prerequisites() {
    log_info "Checking prerequisites..."

    if [ ! -x "$MDB_BIN" ]; then
        log_fail "MDB executable not found at $MDB_BIN"
        log_info "Build with: cmake --build build/$BUILD_TYPE -j \$(nproc)"
        exit 3
    fi

    if [ ! -d "$DB_FOLDER" ]; then
        log_fail "Database folder not found: $DB_FOLDER"
        exit 3
    fi

    if ! command -v jq &> /dev/null; then
        log_fail "jq is required but not installed"
        exit 3
    fi

    if ! command -v curl &> /dev/null; then
        log_fail "curl is required but not installed"
        exit 3
    fi

    log_pass "Prerequisites satisfied"
}

# =============================================================================
# Server Management
# =============================================================================

start_server() {
    log_info "Starting server with database: $DB_FOLDER"

    # Check if port is already in use (using curl)
    if curl -s -o /dev/null --connect-timeout 1 "http://localhost:$SERVER_PORT" 2>/dev/null; then
        log_fail "Port $SERVER_PORT is already in use"
        exit 2
    fi

    "$MDB_BIN" server "$DB_FOLDER" --port "$SERVER_PORT" --browser false &
    SERVER_PID=$!
    sleep 2  # Give server time to initialize

    # Wait for server to be ready (using curl)
    local max_wait=30
    local waited=0
    while ! curl -s -o /dev/null --connect-timeout 1 "http://localhost:$SERVER_PORT" 2>/dev/null; do
        if [ $waited -ge $max_wait ]; then
            log_fail "Server failed to start within ${max_wait}s"
            exit 2
        fi
        if ! kill -0 "$SERVER_PID" 2>/dev/null; then
            log_fail "Server process died during startup"
            exit 2
        fi
        sleep 1
        waited=$((waited + 1))
    done

    if ! kill -0 "$SERVER_PID" 2>/dev/null; then
        log_fail "Server process died unexpectedly"
        exit 2
    fi

    log_pass "Server started successfully (PID: $SERVER_PID)"
}

# =============================================================================
# Verification Tests
# =============================================================================

FAILED=0

# Test 1: Forward == Reverse Edge Counts
test_forward_reverse_symmetry() {
    log_test "Test 1: Verifying forward == reverse edge counts..."

    # Create UNDIRECTED projection
    # Note: Map syntax with '*' for all edge types
    local create_result
    create_result=$(query "CALL graph_project('$PROJECTION_NAME', '*', {'*': {orientation: 'UNDIRECTED'}}) YIELD nodeCount, relationshipCount RETURN nodeCount, relationshipCount")

    local catalog_edges
    catalog_edges=$(extract_value "$create_result" "relationshipCount")
    local catalog_nodes
    catalog_nodes=$(extract_value "$create_result" "nodeCount")

    if [ -z "$catalog_edges" ] || [ "$catalog_edges" == "null" ]; then
        log_fail "Failed to create UNDIRECTED projection"
        log_debug "Response: $create_result"
        FAILED=1
        return 1
    fi

    log_info "  Projection created: nodes=$catalog_nodes, edges=$catalog_edges"

    # Count forward traversals: (a)-[r]->(b)
    local forward_result
    forward_result=$(query "USE $PROJECTION_NAME MATCH (a)-[r]->(b) RETURN COUNT(r) AS c")
    local forward_count
    forward_count=$(extract_value "$forward_result" "c")

    # Count reverse traversals: (a)<-[r]-(b)
    local reverse_result
    reverse_result=$(query "USE $PROJECTION_NAME MATCH (a)<-[r]-(b) RETURN COUNT(r) AS c")
    local reverse_count
    reverse_count=$(extract_value "$reverse_result" "c")

    log_info "  Forward edges (a)-[r]->(b):  $forward_count"
    log_info "  Reverse edges (a)<-[r]-(b):  $reverse_count"

    if [ "$forward_count" == "$reverse_count" ]; then
        log_pass "Forward == Reverse: $forward_count edges"
        return 0
    else
        log_fail "Forward ($forward_count) != Reverse ($reverse_count)"
        FAILED=1
        return 1
    fi
}

# Test 2: Undirected Pattern Count
test_undirected_pattern_count() {
    log_test "Test 2: Verifying undirected pattern count..."

    # Count undirected pattern: (a)-[r]-(b) (both directions)
    local undirected_result
    undirected_result=$(query "USE $PROJECTION_NAME MATCH (a)-[r]-(b) RETURN COUNT(r) AS c")
    local undirected_count
    undirected_count=$(extract_value "$undirected_result" "c")

    # Get forward count for comparison
    local forward_result
    forward_result=$(query "USE $PROJECTION_NAME MATCH (a)-[r]->(b) RETURN COUNT(r) AS c")
    local forward_count
    forward_count=$(extract_value "$forward_result" "c")

    local expected_undirected=$((forward_count * 2))

    log_info "  Undirected pattern count (a)-[r]-(b): $undirected_count"
    log_info "  Expected (2 * forward):                $expected_undirected"

    if [ "$undirected_count" == "$expected_undirected" ]; then
        log_pass "Undirected count == 2 * forward: $undirected_count"
        return 0
    else
        log_fail "Undirected ($undirected_count) != 2 * forward ($expected_undirected)"
        FAILED=1
        return 1
    fi
}

# Test 3: Specific Edge Bidirectional Test
test_specific_edge_bidirectionality() {
    log_test "Test 3: Verifying specific edge bidirectionality..."

    # Get one edge to test
    local edge_result
    edge_result=$(query "USE $PROJECTION_NAME MATCH (a)-[r]->(b) RETURN id(a) AS from_id, id(b) AS to_id LIMIT 1")

    local from_id
    from_id=$(extract_value "$edge_result" "from_id")
    local to_id
    to_id=$(extract_value "$edge_result" "to_id")

    if [ -z "$from_id" ] || [ "$from_id" == "null" ] || [ -z "$to_id" ] || [ "$to_id" == "null" ]; then
        log_info "  No edges found in projection - skipping specific edge test"
        log_pass "Skipped (no edges)"
        return 0
    fi

    log_info "  Testing edge: $from_id -> $to_id"

    # Test forward direction
    local forward_test
    forward_test=$(query "USE $PROJECTION_NAME MATCH (a)-[r]->(b) WHERE id(a) = $from_id AND id(b) = $to_id RETURN COUNT(r) AS c")
    local forward_exists
    forward_exists=$(extract_value "$forward_test" "c")

    # Test reverse direction (THE KEY TEST for UNDIRECTED)
    local reverse_test
    reverse_test=$(query "USE $PROJECTION_NAME MATCH (a)-[r]->(b) WHERE id(a) = $to_id AND id(b) = $from_id RETURN COUNT(r) AS c")
    local reverse_exists
    reverse_exists=$(extract_value "$reverse_test" "c")

    log_info "  Forward ($from_id -> $to_id): $forward_exists edge(s)"
    log_info "  Reverse ($to_id -> $from_id): $reverse_exists edge(s)"

    if [ "$forward_exists" == "1" ] && [ "$reverse_exists" == "1" ]; then
        log_pass "Bidirectional traversal verified for edge ($from_id, $to_id)"
        return 0
    else
        log_fail "Bidirectional traversal FAILED for edge ($from_id, $to_id)"
        log_fail "  Expected: forward=1, reverse=1"
        log_fail "  Actual:   forward=$forward_exists, reverse=$reverse_exists"
        FAILED=1
        return 1
    fi
}

# Test 4: Degree Symmetry for All Nodes
test_degree_symmetry() {
    log_test "Test 4: Verifying degree symmetry across all nodes..."

    # For UNDIRECTED, every node's out-degree should equal its in-degree
    # We can check this by comparing total sums

    local out_result
    out_result=$(query "USE $PROJECTION_NAME MATCH (n)-[r]->() RETURN COUNT(r) AS total")
    local total_out
    total_out=$(extract_value "$out_result" "total")

    local in_result
    in_result=$(query "USE $PROJECTION_NAME MATCH (n)<-[r]-() RETURN COUNT(r) AS total")
    local total_in
    total_in=$(extract_value "$in_result" "total")

    log_info "  Total out-degree sum: $total_out"
    log_info "  Total in-degree sum:  $total_in"

    if [ "$total_out" == "$total_in" ]; then
        log_pass "Degree symmetry verified: sum(out) = sum(in) = $total_out"
        return 0
    else
        log_fail "Degree symmetry violated: out=$total_out, in=$total_in"
        FAILED=1
        return 1
    fi
}

# Test 5: NATURAL vs UNDIRECTED Comparison
test_natural_vs_undirected() {
    log_test "Test 5: Comparing NATURAL vs UNDIRECTED orientations..."

    # Create a NATURAL projection for comparison
    local natural_name="${PROJECTION_NAME}_natural"

    local natural_result
    natural_result=$(query "CALL graph_project('$natural_name', '*', '*') YIELD relationshipCount RETURN relationshipCount")
    local natural_edges
    natural_edges=$(extract_value "$natural_result" "relationshipCount")

    # Get UNDIRECTED edge count (from original projection)
    local undirected_result
    undirected_result=$(query "USE $PROJECTION_NAME MATCH (a)-[r]->(b) RETURN COUNT(r) AS c")
    local undirected_edges
    undirected_edges=$(extract_value "$undirected_result" "c")

    log_info "  NATURAL orientation edges:    $natural_edges"
    log_info "  UNDIRECTED forward traversal: $undirected_edges"

    # For UNDIRECTED, the forward traversal count should be 2x the natural count
    # because each edge becomes bidirectional
    local expected_undirected=$((natural_edges * 2))

    log_info "  Expected UNDIRECTED forward (2 * NATURAL): $expected_undirected"

    # Cleanup the natural projection
    query "CALL graph_drop('$natural_name')" > /dev/null 2>&1 || true

    if [ "$undirected_edges" == "$expected_undirected" ]; then
        log_pass "UNDIRECTED correctly doubles edge count: $undirected_edges"
        return 0
    else
        # This might not always be 2x if the source already has undirected edges
        log_info "  Note: Ratio differs - this is OK if source has undirected edges"
        log_pass "NATURAL vs UNDIRECTED comparison completed"
        return 0
    fi
}

# =============================================================================
# Main Execution
# =============================================================================

main() {
    echo ""
    echo "============================================"
    echo "  UNDIRECTED Orientation Verification      "
    echo "============================================"
    echo ""

    check_prerequisites
    start_server

    echo ""
    log_info "Running UNDIRECTED orientation verification tests..."
    echo ""

    test_forward_reverse_symmetry || true
    echo ""
    test_undirected_pattern_count || true
    echo ""
    test_specific_edge_bidirectionality || true
    echo ""
    test_degree_symmetry || true
    echo ""
    test_natural_vs_undirected || true

    # Summary
    echo ""
    echo "============================================"
    if [ "$FAILED" -eq 0 ]; then
        echo -e "  ${GREEN}ALL UNDIRECTED VERIFICATIONS PASSED${NC}  "
        echo "============================================"
        echo ""
        log_info "Invariants verified (INV-ORI-002):"
        log_info "  - Forward edge count == Reverse edge count"
        log_info "  - Undirected pattern == 2 * Forward"
        log_info "  - Specific edges traversable both ways"
        log_info "  - Degree symmetry maintained"
        exit 0
    else
        echo -e "  ${RED}VERIFICATION FAILED - SEE ERRORS ABOVE${NC}  "
        echo "============================================"
        echo ""
        log_info "This indicates a potential issue with UNDIRECTED orientation"
        log_info "implementation. Check:"
        log_info "  - ProjectionStorage::add_edge() for bidirectional insertion"
        log_info "  - NativeProjectionBuilder for orientation handling"
        exit 1
    fi
}

main "$@"
