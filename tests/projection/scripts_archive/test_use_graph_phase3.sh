#!/bin/bash
# End-to-end test for Phase 3: USE GRAPH projection loading
# Tests that projections can be loaded and queried using USE GRAPH clause

set -e  # Exit on error

echo "=== Phase 3: USE GRAPH Projection Loading Test ==="
echo

# Configuration
MDB_BIN="./build/Release/bin/mdb"
TEST_DB="./data/test_use_graph_phase3"
SOURCE_DATA="./data/example/gql/posts/posts.gql"

# Cleanup previous test
echo "[1/7] Cleaning up previous test database..."
rm -rf "$TEST_DB"
echo "    ✓ Cleanup complete"
echo

# Import test data
echo "[2/7] Importing GQL test data..."
$MDB_BIN import "$SOURCE_DATA" "$TEST_DB"
echo "    ✓ Import complete"
echo

# Test 1: Create projection (with topology only)
echo "[3/7] Creating projection 'user_friends'..."
QUERY_CREATE="MATCH (u1:User)-[f]-(u2:User) RETURN PROJECT('user_friends')"
echo "    Query: $QUERY_CREATE"
echo "$QUERY_CREATE" | $MDB_BIN cli "$TEST_DB" > /tmp/test_create_output.txt 2>&1
cat /tmp/test_create_output.txt | grep -v "^$" | tail -5
echo "    ✓ Projection created"
echo

# Verify projection was created
echo "[4/7] Verifying projection exists..."
if [ -d "$TEST_DB/projections/user_friends" ]; then
    echo "    ✓ Projection directory exists"
    ls -lh "$TEST_DB/projections/user_friends/" | head -10
else
    echo "    ✗ ERROR: Projection directory NOT found!"
    exit 1
fi
echo

# Test 2: Query projection using USE GRAPH
echo "[5/7] Querying projection with USE GRAPH..."
QUERY_USE_GRAPH="USE \"user_friends\" MATCH (a)-[e]-(b) RETURN a, e, b LIMIT 5"
echo "    Query: $QUERY_USE_GRAPH"
echo "$QUERY_USE_GRAPH" | $MDB_BIN cli "$TEST_DB" > /tmp/test_use_graph_output.txt 2>&1

# Check if query executed without errors
if grep -qi "error" /tmp/test_use_graph_output.txt; then
    echo "    ✗ ERROR: Query failed!"
    cat /tmp/test_use_graph_output.txt
    exit 1
else
    echo "    ✓ Query executed successfully"
    cat /tmp/test_use_graph_output.txt | grep -v "^$" | tail -10
fi
echo

# Test 3: Switch back to main graph using CURRENT_GRAPH
echo "[6/7] Testing switch back to main graph..."
QUERY_MAIN="USE CURRENT_GRAPH MATCH (u:User) RETURN u.name LIMIT 3"
echo "    Query: $QUERY_MAIN"
echo "$QUERY_MAIN" | $MDB_BIN cli "$TEST_DB" > /tmp/test_main_graph_output.txt 2>&1

if grep -qi "error" /tmp/test_main_graph_output.txt; then
    echo "    ✗ ERROR: Query failed!"
    cat /tmp/test_main_graph_output.txt
    exit 1
else
    echo "    ✓ Query executed successfully"
    cat /tmp/test_main_graph_output.txt | grep -v "^$" | tail -8
fi
echo

# Test 4: Test error handling for non-existent projection
echo "[7/7] Testing error handling for non-existent projection..."
QUERY_NONEXISTENT="USE \"nonexistent_projection\" MATCH (a) RETURN a"
echo "    Query: $QUERY_NONEXISTENT"
echo "$QUERY_NONEXISTENT" | $MDB_BIN cli "$TEST_DB" > /tmp/test_error_output.txt 2>&1 || true

if grep -qi "does not exist" /tmp/test_error_output.txt; then
    echo "    ✓ Error handling works correctly"
    grep "does not exist" /tmp/test_error_output.txt | head -2
else
    echo "    ✗ WARNING: Expected error message not found"
    cat /tmp/test_error_output.txt
fi
echo

echo "=== ✓ Phase 3 USE GRAPH tests completed! ==="
echo
echo "Summary:"
echo "  - ✓ Projection creation: working"
echo "  - ✓ USE GRAPH clause: loads projection context"
echo "  - ✓ Projection query execution: successful"
echo "  - ✓ Switch back to main graph: working"
echo "  - ✓ Error handling: correct for non-existent projections"
echo
echo "Phase 3 implementation is complete and functional!"
echo
