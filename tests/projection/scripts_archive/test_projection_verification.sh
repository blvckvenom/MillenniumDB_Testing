#!/bin/bash
# Complete verification test for USE GRAPH implementation
# Tests projection creation, loading, and query execution

set -e

echo "=========================================="
echo "PROJECTION VERIFICATION TEST"
echo "=========================================="
echo

MDB_BIN="./build/Release/bin/mdb"
TEST_DB="./data/test_projection_verify"
SOURCE_DATA="./data/example/gql/posts/posts.gql"
PORT=1234

# Cleanup
rm -rf "$TEST_DB"
pkill -f "mdb server" || true
sleep 1

# Import
echo "[1/8] Importing data..."
$MDB_BIN import "$SOURCE_DATA" "$TEST_DB" > /dev/null 2>&1
echo "✓ Import complete"
echo

# Start server
$MDB_BIN server "$TEST_DB" --port $PORT --browser false > /tmp/proj_test.log 2>&1 &
SERVER_PID=$!
sleep 3

if ! kill -0 $SERVER_PID 2>/dev/null; then
    echo "ERROR: Server failed"
    exit 1
fi

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo " TEST 1: Baseline - Main Graph Statistics"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo

echo "Total nodes:"
curl -s -X POST "http://localhost:$PORT/gql" --data 'MATCH (n) RETURN count(*) AS total'
echo

echo "Total edges:"
curl -s -X POST "http://localhost:$PORT/gql" --data 'MATCH ()-[e]-() RETURN count(*) AS total'
echo

echo "Friend edges (User-User):"
curl -s -X POST "http://localhost:$PORT/gql" --data 'MATCH (u1:User)-[f:Friend]-(u2:User) RETURN count(*) AS total'
echo

echo "Posted edges (User-Post):"
curl -s -X POST "http://localhost:$PORT/gql" --data 'MATCH (u:User)-[p:Posted]->(post:Post) RETURN count(*) AS total'
echo

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo " TEST 2: Create Topology-Only Projection"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo

QUERY_CREATE='MATCH (u1:User)-[f:Friend]-(u2:User) RETURN PROJECT("friends_topology")'
echo "Query: $QUERY_CREATE"
echo

curl -s -X POST "http://localhost:$PORT/gql" --data "$QUERY_CREATE" > /tmp/create1.txt
if grep -qi "error\|exception" /tmp/create1.txt; then
    echo "❌ ERROR creating projection:"
    cat /tmp/create1.txt
    kill $SERVER_PID
    exit 1
else
    echo "✓ Topology projection created"
    tail -3 /tmp/create1.txt
fi
echo

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo " TEST 3: Create Projection with ALL Properties"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo

QUERY_CREATE2='MATCH (u1:User)-[f:Friend]-(u2:User) RETURN PROJECT("friends_full" INCLUDE PROPERTIES)'
echo "Query: $QUERY_CREATE2"
echo

curl -s -X POST "http://localhost:$PORT/gql" --data "$QUERY_CREATE2" > /tmp/create2.txt
if grep -qi "error\|exception" /tmp/create2.txt; then
    echo "❌ ERROR creating projection:"
    cat /tmp/create2.txt
    kill $SERVER_PID
    exit 1
else
    echo "✓ Full projection created"
    tail -3 /tmp/create2.txt
fi
echo

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo " TEST 4: Verify Projection Files Created"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo

if [ -d "$TEST_DB/projections/friends_topology" ]; then
    echo "✓ friends_topology directory exists"
    echo "  Files:"
    ls -lh "$TEST_DB/projections/friends_topology/" | grep -E "\.(leaf|dir)$" | head -5
else
    echo "❌ friends_topology NOT found"
fi
echo

if [ -d "$TEST_DB/projections/friends_full" ]; then
    echo "✓ friends_full directory exists"
    echo "  Files:"
    ls -lh "$TEST_DB/projections/friends_full/" | grep -E "\.(leaf|dir)$" | head -5
else
    echo "❌ friends_full NOT found"
fi
echo

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo " TEST 5: Query Topology Projection"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo

echo "Count from topology projection:"
curl -s -X POST "http://localhost:$PORT/gql" --data 'USE "friends_topology" MATCH (a)-[e]-(b) RETURN count(*) AS total'
echo

echo "Sample data (should show object IDs only):"
curl -s -X POST "http://localhost:$PORT/gql" --data 'USE "friends_topology" MATCH (a)-[e]-(b) RETURN a, e, b LIMIT 3' | head -5
echo

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo " TEST 6: Query Full Projection with Properties"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo

echo "Count from full projection:"
curl -s -X POST "http://localhost:$PORT/gql" --data 'USE "friends_full" MATCH (a)-[e]-(b) RETURN count(*) AS total'
echo

echo "Sample data with names:"
curl -s -X POST "http://localhost:$PORT/gql" --data 'USE "friends_full" MATCH (u1)-[f]-(u2) RETURN u1.name AS Person1, u2.name AS Person2, f.since AS FriendsSince LIMIT 5'
echo

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo " TEST 7: Verify Counts Match Expected Values"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo

MAIN_COUNT=$(curl -s -X POST "http://localhost:$PORT/gql" --data 'MATCH (u1:User)-[f:Friend]-(u2:User) RETURN count(*) AS total' | tail -1)
PROJ_COUNT=$(curl -s -X POST "http://localhost:$PORT/gql" --data 'USE "friends_full" MATCH (a)-[e]-(b) RETURN count(*) AS total' | tail -1)

echo "Main graph Friend edges: $MAIN_COUNT"
echo "Projection edges:        $PROJ_COUNT"
echo

if [ "$MAIN_COUNT" = "$PROJ_COUNT" ]; then
    echo "✅ COUNTS MATCH - Projection is correctly filtered!"
else
    echo "⚠️  COUNTS DIFFER - Investigating..."
    echo ""
    echo "This could indicate:"
    echo "  1. Projection contains extra edges (bug in projection creation)"
    echo "  2. Different counting of undirected edges"
    echo "  3. Phase 4 not implemented (using main graph instead)"
fi
echo

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo " TEST 8: Test Error Handling"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo

echo "Query non-existent projection:"
curl -s -X POST "http://localhost:$PORT/gql" --data 'USE "nonexistent" MATCH (a) RETURN a' 2>&1 | head -3
echo

# Cleanup
kill $SERVER_PID
wait $SERVER_PID 2>/dev/null || true

echo ""
echo "=========================================="
echo "VERIFICATION COMPLETE"
echo "=========================================="
