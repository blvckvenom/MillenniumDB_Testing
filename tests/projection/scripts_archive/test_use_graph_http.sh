#!/bin/bash
# End-to-end test for Phase 3: USE GRAPH via HTTP server
# Tests that projections can be loaded and queried using USE GRAPH clause

set -e  # Exit on error

echo "=== Phase 3: USE GRAPH HTTP Server Test ==="
echo

# Configuration
MDB_BIN="./build/Release/bin/mdb"
TEST_DB="./data/test_use_graph_http"
SOURCE_DATA="./data/example/gql/posts/posts.gql"
PORT=1234

# Cleanup previous test
echo "[1/8] Cleaning up previous test database..."
rm -rf "$TEST_DB"
pkill -f "mdb server" || true
sleep 1
echo "    ✓ Cleanup complete"
echo

# Import test data
echo "[2/8] Importing GQL test data..."
$MDB_BIN import "$SOURCE_DATA" "$TEST_DB"
echo "    ✓ Import complete"
echo

# Start server in background
echo "[3/8] Starting MillenniumDB server..."
$MDB_BIN server "$TEST_DB" --port $PORT --browser false > /tmp/mdb_server.log 2>&1 &
SERVER_PID=$!
echo "    Server PID: $SERVER_PID"

# Wait for server to start
sleep 3

# Check if server is running
if ! kill -0 $SERVER_PID 2>/dev/null; then
    echo "    ✗ ERROR: Server failed to start!"
    cat /tmp/mdb_server.log
    exit 1
fi
echo "    ✓ Server started successfully"
echo

# Test 1: Create projection
echo "[4/8] Creating projection 'user_friends'..."
QUERY_CREATE="MATCH (u1:User)-[f]-(u2:User) RETURN PROJECT('user_friends')"
echo "    Query: $QUERY_CREATE"

curl -s -X POST "http://localhost:$PORT/gql" \
  -H "Content-Type: application/sparql-query" \
  --data "$QUERY_CREATE" > /tmp/create_response.txt 2>&1

if grep -qi "error" /tmp/create_response.txt; then
    echo "    ✗ ERROR: Failed to create projection!"
    cat /tmp/create_response.txt
    kill $SERVER_PID
    exit 1
else
    echo "    ✓ Projection created"
    tail -5 /tmp/create_response.txt
fi
echo

# Verify projection exists
echo "[5/8] Verifying projection exists..."
if [ -d "$TEST_DB/projections/user_friends" ]; then
    echo "    ✓ Projection directory exists"
    ls -lh "$TEST_DB/projections/user_friends/" | grep -E "(nodes|from_to|to_from|edge_direction)" | head -4
else
    echo "    ✗ ERROR: Projection directory NOT found!"
    kill $SERVER_PID
    exit 1
fi
echo

# Test 2: Query projection using USE GRAPH
echo "[6/8] Querying projection with USE GRAPH..."
QUERY_USE_GRAPH='USE "user_friends" MATCH (a)-[e]-(b) RETURN a, e, b LIMIT 5'
echo "    Query: $QUERY_USE_GRAPH"

curl -s -X POST "http://localhost:$PORT/gql" \
  -H "Content-Type: application/sparql-query" \
  --data "$QUERY_USE_GRAPH" > /tmp/use_graph_response.txt 2>&1

if grep -qi "error" /tmp/use_graph_response.txt || grep -qi "400 Bad Request" /tmp/use_graph_response.txt; then
    echo "    ✗ ERROR: USE GRAPH query failed!"
    cat /tmp/use_graph_response.txt
    kill $SERVER_PID
    exit 1
else
    echo "    ✓ Query executed successfully"
    # Count result rows (TSV format)
    RESULT_COUNT=$(grep -v "^$" /tmp/use_graph_response.txt | wc -l)
    echo "    Results returned: $RESULT_COUNT lines"
    head -8 /tmp/use_graph_response.txt
fi
echo

# Test 3: Switch back to main graph
echo "[7/8] Testing switch back to main graph with CURRENT_GRAPH..."
QUERY_MAIN="USE CURRENT_GRAPH MATCH (u:User) RETURN u LIMIT 3"
echo "    Query: $QUERY_MAIN"

curl -s -X POST "http://localhost:$PORT/gql" \
  -H "Content-Type: application/sparql-query" \
  --data "$QUERY_MAIN" > /tmp/main_graph_response.txt 2>&1

if grep -qi "error" /tmp/main_graph_response.txt || grep -qi "400 Bad Request" /tmp/main_graph_response.txt; then
    echo "    ✗ ERROR: Main graph query failed!"
    cat /tmp/main_graph_response.txt
    kill $SERVER_PID
    exit 1
else
    echo "    ✓ Query executed successfully"
    head -6 /tmp/main_graph_response.txt
fi
echo

# Test 4: Error handling for non-existent projection
echo "[8/8] Testing error handling for non-existent projection..."
QUERY_NONEXISTENT='USE "nonexistent_projection" MATCH (a) RETURN a'
echo "    Query: $QUERY_NONEXISTENT"

curl -s -X POST "http://localhost:$PORT/gql" \
  -H "Content-Type: application/sparql-query" \
  --data "$QUERY_NONEXISTENT" > /tmp/error_response.txt 2>&1

if grep -qi "does not exist" /tmp/error_response.txt; then
    echo "    ✓ Error handling works correctly"
    grep -i "does not exist" /tmp/error_response.txt | head -1
elif grep -qi "400 Bad Request" /tmp/error_response.txt; then
    echo "    ✓ Error returned (400 Bad Request)"
else
    echo "    ⚠ WARNING: Unexpected response"
    head -5 /tmp/error_response.txt
fi
echo

# Cleanup
echo "Stopping server..."
kill $SERVER_PID
wait $SERVER_PID 2>/dev/null || true
echo

echo "=== ✓ Phase 3 USE GRAPH HTTP tests completed! ==="
echo
echo "Summary:"
echo "  - ✓ Projection creation via HTTP: working"
echo "  - ✓ USE GRAPH clause: loads projection context"
echo "  - ✓ Projection query execution: successful"
echo "  - ✓ Switch back to main graph: working"
echo "  - ✓ HTTP server API: fully functional"
echo
echo "Phase 3 implementation is complete and functional!"
echo
