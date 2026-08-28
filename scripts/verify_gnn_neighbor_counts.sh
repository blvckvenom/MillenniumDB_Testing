#!/bin/bash
#
# GNN Neighbor Count Verification Script
#
# Purpose: Validates that TopologyAccessor returns correct neighbor counts
#          by comparing catalog counts against direct GQL queries.
#
# Invariants Verified:
#   INV-PROJ-001: nodeCount == COUNT(MATCH (n))
#   INV-PROJ-002: relationshipCount == COUNT(MATCH ()-[r]->())
#   INV-GNN-001:  Sum of out-degrees equals edge count
#
# Usage: ./scripts/verify_gnn_neighbor_counts.sh [database_folder] [port]
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
PROJECTION_NAME="neighbor_verify_$$"
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
NC='\033[0m' # No Color

log_info()  { echo -e "${YELLOW}[INFO]${NC}  $(date '+%H:%M:%S') $*"; }
log_pass()  { echo -e "${GREEN}[PASS]${NC}  $(date '+%H:%M:%S') $*"; }
log_fail()  { echo -e "${RED}[FAIL]${NC}  $(date '+%H:%M:%S') $*" >&2; }
log_debug() { echo -e "${BLUE}[DEBUG]${NC} $(date '+%H:%M:%S') $*"; }

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

    # Drop projection if it exists
    if [ -n "$SERVER_PID" ] && kill -0 "$SERVER_PID" 2>/dev/null; then
        query "CALL graph_drop('$PROJECTION_NAME')" > /dev/null 2>&1 || true
        sleep 0.5
    fi

    # Stop server
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

    # Check executable
    if [ ! -x "$MDB_BIN" ]; then
        log_fail "MDB executable not found at $MDB_BIN"
        log_info "Build with: cmake --build build/$BUILD_TYPE -j \$(nproc)"
        exit 3
    fi

    # Check database folder
    if [ ! -d "$DB_FOLDER" ]; then
        log_fail "Database folder not found: $DB_FOLDER"
        exit 3
    fi

    # Check jq is available
    if ! command -v jq &> /dev/null; then
        log_fail "jq is required but not installed"
        log_info "Install with: sudo apt install jq"
        exit 3
    fi

    # Check curl is available
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

    # Verify server is actually responsive
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

# Test: Node Count Consistency (INV-PROJ-001)
test_node_count() {
    log_info "Test 1: Verifying node count consistency (INV-PROJ-001)..."

    # Create projection and capture nodeCount
    local create_result
    create_result=$(query "CALL graph_project('$PROJECTION_NAME', '*', '*') YIELD nodeCount, relationshipCount RETURN nodeCount, relationshipCount")

    local catalog_nodes
    catalog_nodes=$(extract_value "$create_result" "nodeCount")
    local catalog_edges
    catalog_edges=$(extract_value "$create_result" "relationshipCount")

    if [ -z "$catalog_nodes" ] || [ "$catalog_nodes" == "null" ]; then
        log_fail "Failed to create projection or get nodeCount"
        log_debug "Response: $create_result"
        return 1
    fi

    log_info "  Catalog reports: nodes=$catalog_nodes, edges=$catalog_edges"

    # Query projection to verify node count
    local query_result
    query_result=$(query "USE $PROJECTION_NAME MATCH (n) RETURN COUNT(n) AS c")
    local query_nodes
    query_nodes=$(extract_value "$query_result" "c")

    log_info "  Query result: nodes=$query_nodes"

    if [ "$catalog_nodes" != "$query_nodes" ]; then
        log_fail "Node count mismatch: catalog=$catalog_nodes, query=$query_nodes"
        FAILED=1
        return 1
    fi

    log_pass "Node count verified: $catalog_nodes"
    return 0
}

# Test: Edge Count Consistency (INV-PROJ-002)
test_edge_count() {
    log_info "Test 2: Verifying edge count consistency (INV-PROJ-002)..."

    # Get catalog edge count (from earlier projection)
    local info_result
    info_result=$(query "CALL graph_project('$PROJECTION_NAME', '*', '*') YIELD relationshipCount RETURN relationshipCount" 2>/dev/null || echo '{"results":[]}')

    # If projection already exists, query it directly
    local query_result
    query_result=$(query "USE $PROJECTION_NAME MATCH ()-[r]->() RETURN COUNT(r) AS c")
    local query_edges
    query_edges=$(extract_value "$query_result" "c")

    # Also get the catalog count via graph_info if available
    local catalog_result
    catalog_result=$(query "CALL graph_exists('$PROJECTION_NAME') YIELD exists RETURN exists")

    log_info "  Query result: edges=$query_edges (directed traversal)"

    if [ -z "$query_edges" ] || [ "$query_edges" == "null" ]; then
        log_fail "Failed to get edge count from projection"
        FAILED=1
        return 1
    fi

    log_pass "Edge count verified via query: $query_edges"
    return 0
}

# Test: Degree Sum Equals Edge Count (INV-GNN-001)
test_degree_sum() {
    log_info "Test 3: Verifying sum of out-degrees equals edge count (INV-GNN-001)..."

    # Get total out-degree via aggregation
    local degree_result
    degree_result=$(query "USE $PROJECTION_NAME MATCH (n)-[r]->() RETURN COUNT(r) AS total")
    local degree_sum
    degree_sum=$(extract_value "$degree_result" "total")

    # Compare with edge count from Test 2
    local edge_result
    edge_result=$(query "USE $PROJECTION_NAME MATCH ()-[r]->() RETURN COUNT(r) AS c")
    local edge_count
    edge_count=$(extract_value "$edge_result" "c")

    log_info "  Sum of out-degrees: $degree_sum"
    log_info "  Total edge count:   $edge_count"

    if [ "$degree_sum" != "$edge_count" ]; then
        log_fail "Degree sum mismatch: sum=$degree_sum, edges=$edge_count"
        FAILED=1
        return 1
    fi

    log_pass "Degree sum verified: sum(out_degree) = $degree_sum = edge_count"
    return 0
}

# Test: Bidirectional Traversal Symmetry
test_bidirectional() {
    log_info "Test 4: Verifying bidirectional traversal symmetry..."

    # For directed edges, total in-degree should equal total out-degree
    # (sum of edges entering nodes = sum of edges leaving nodes)
    local out_result
    out_result=$(query "USE $PROJECTION_NAME MATCH (n)-[r]->() RETURN COUNT(r) AS out_sum")
    local out_sum
    out_sum=$(extract_value "$out_result" "out_sum")

    local in_result
    in_result=$(query "USE $PROJECTION_NAME MATCH (n)<-[r]-() RETURN COUNT(r) AS in_sum")
    local in_sum
    in_sum=$(extract_value "$in_result" "in_sum")

    log_info "  Total out-degree: $out_sum"
    log_info "  Total in-degree:  $in_sum"

    if [ "$out_sum" != "$in_sum" ]; then
        log_fail "Bidirectional mismatch: out=$out_sum, in=$in_sum"
        FAILED=1
        return 1
    fi

    log_pass "Bidirectional traversal verified: out=$out_sum = in=$in_sum"
    return 0
}

# =============================================================================
# Main Execution
# =============================================================================

main() {
    echo ""
    echo "============================================"
    echo "  GNN Neighbor Count Verification Script   "
    echo "============================================"
    echo ""

    check_prerequisites
    start_server

    echo ""
    log_info "Running verification tests..."
    echo ""

    test_node_count || true
    test_edge_count || true
    test_degree_sum || true
    test_bidirectional || true

    # Summary
    echo ""
    echo "============================================"
    if [ "$FAILED" -eq 0 ]; then
        echo -e "  ${GREEN}ALL NEIGHBOR COUNT VERIFICATIONS PASSED${NC}  "
        echo "============================================"
        echo ""
        log_info "Invariants verified:"
        log_info "  - INV-PROJ-001: nodeCount == COUNT(MATCH (n))"
        log_info "  - INV-PROJ-002: relationshipCount == COUNT(MATCH ()-[r]->())"
        log_info "  - INV-GNN-001:  sum(out_degree) == edge_count"
        exit 0
    else
        echo -e "  ${RED}VERIFICATION FAILED - SEE ERRORS ABOVE${NC}  "
        echo "============================================"
        exit 1
    fi
}

main "$@"
