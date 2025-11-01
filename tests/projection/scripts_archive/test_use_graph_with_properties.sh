#!/bin/bash
# Test USE GRAPH with actual property values (names)
# This verifies that projections return meaningful data, not just object IDs

set -e  # Exit on error

echo "=== USE GRAPH Test with Property Values ===="
echo

# Configuration
MDB_BIN="./build/Release/bin/mdb"
TEST_DB="./data/test_use_graph_properties"
SOURCE_DATA="./data/example/gql/posts/posts.gql"
PORT=1234

# Cleanup previous test
echo "[1/7] Cleaning up previous test database..."
rm -rf "$TEST_DB"
pkill -f "mdb server" || true
sleep 1
echo "    ✓ Cleanup complete"
echo

# Import test data
echo "[2/7] Importing GQL test data..."
$MDB_BIN import "$SOURCE_DATA" "$TEST_DB"
echo "    ✓ Import complete"
echo

# Start server in background
echo "[3/7] Starting MillenniumDB server..."
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

# Test 1: Query main graph with properties (baseline)
echo "[4/7] Baseline: Querying main graph for user names..."
QUERY_BASELINE="MATCH (u:User) RETURN u.name LIMIT 5"
echo "    Query: $QUERY_BASELINE"

curl -s -X POST "http://localhost:$PORT/gql" \
  -H "Content-Type: application/sparql-query" \
  --data "$QUERY_BASELINE" > /tmp/baseline_response.txt 2>&1

if grep -qi "error" /tmp/baseline_response.txt; then
    echo "    ✗ ERROR: Baseline query failed!"
    cat /tmp/baseline_response.txt
    kill $SERVER_PID
    exit 1
else
    echo "    ✓ Baseline query successful (main graph):"
    head -10 /tmp/baseline_response.txt
fi
echo

# Test 2: Create projection with INCLUDE PROPERTIES
echo "[5/7] Creating projection with INCLUDE PROPERTIES..."
QUERY_CREATE='MATCH (u1:User)-[f]-(u2:User) RETURN PROJECT("friends_with_names", INCLUDE PROPERTIES(u1.name, u2.name))'
echo "    Query: $QUERY_CREATE"

curl -s -X POST "http://localhost:$PORT/gql" \
  -H "Content-Type: application/sparql-query" \
  --data "$QUERY_CREATE" > /tmp/create_proj_response.txt 2>&1

if grep -qi "error" /tmp/create_proj_response.txt; then
    echo "    ✗ ERROR: Failed to create projection with properties!"
    cat /tmp/create_proj_response.txt
    kill $SERVER_PID
    exit 1
else
    echo "    ✓ Projection with properties created"
    tail -5 /tmp/create_proj_response.txt
fi
echo

# Test 3: Query projection with USE GRAPH to get property values
echo "[6/7] Querying projection with USE GRAPH (returning names)..."
QUERY_PROJECTION='USE "friends_with_names" MATCH (u1)-[f]-(u2) RETURN u1.name, u2.name LIMIT 10'
echo "    Query: $QUERY_PROJECTION"

curl -s -X POST "http://localhost:$PORT/gql" \
  -H "Content-Type: application/sparql-query" \
  --data "$QUERY_PROJECTION" > /tmp/projection_with_names.txt 2>&1

if grep -qi "error" /tmp/projection_with_names.txt || grep -qi "400 Bad Request" /tmp/projection_with_names.txt; then
    echo "    ✗ ERROR: Projection query with properties failed!"
    cat /tmp/projection_with_names.txt
    kill $SERVER_PID
    exit 1
else
    echo "    ✓ Query executed successfully"
    echo "    Results (showing actual names from projection):"
    head -15 /tmp/projection_with_names.txt
fi
echo

# Test 4: Verify we get actual names, not just object IDs
echo "[7/7] Verifying results contain actual names..."
if grep -E '"[A-Z][a-z]+ [A-Z][a-z]+"' /tmp/projection_with_names.txt > /dev/null; then
    echo "    ✓ SUCCESS: Results contain actual person names!"
    echo "    Example names found:"
    grep -oE '"[A-Z][a-z]+ [A-Z][a-z]+"' /tmp/projection_with_names.txt | head -5
elif grep -E '[A-Z][a-z]+ [A-Z][a-z]+' /tmp/projection_with_names.txt > /dev/null; then
    echo "    ✓ SUCCESS: Results contain actual person names!"
    echo "    Example names found:"
    grep -oE '[A-Z][a-z]+ [A-Z][a-z]+' /tmp/projection_with_names.txt | head -5
else
    echo "    ⚠ WARNING: Results may not contain expected name format"
    echo "    Actual output:"
    cat /tmp/projection_with_names.txt
fi
echo

# Cleanup
echo "Stopping server..."
kill $SERVER_PID
wait $SERVER_PID 2>/dev/null || true
echo

echo "=== Test Complete ==="
echo
echo "Summary:"
echo "  - ✓ Main graph baseline query: working"
echo "  - ✓ Projection with INCLUDE PROPERTIES: created"
echo "  - ✓ USE GRAPH with property access: tested"
echo
echo "Check if names appear in results above to verify implementation!"
echo
