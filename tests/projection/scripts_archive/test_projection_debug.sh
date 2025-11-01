#!/bin/bash
# Debug projection creation with detailed output

set -e

MDB_BIN="./build/Release/bin/mdb"
TEST_DB="./data/test_proj_debug"
SOURCE_DATA="./data/example/gql/posts/posts.gql"
PORT=1234

# Cleanup
rm -rf "$TEST_DB"
pkill -f "mdb server" || true
sleep 1

# Import
echo "Importing data..."
$MDB_BIN import "$SOURCE_DATA" "$TEST_DB" > /dev/null 2>&1

# Start server
$MDB_BIN server "$TEST_DB" --port $PORT --browser false > /dev/null 2>&1 &
SERVER_PID=$!
sleep 3

echo "=========================================="
echo "PROJECTION CREATION DEBUG"
echo "=========================================="
echo

echo "Step 1: Show what MATCH returns (first 10 rows)"
echo "------------------------------------------------"
curl -s -X POST "http://localhost:$PORT/gql" \
  --data 'MATCH (u1:User)-[f:Friend]-(u2:User) RETURN u1, f, u2 LIMIT 10'
echo

echo "Step 2: Count unique edges in MATCH"
echo "------------------------------------"
curl -s -X POST "http://localhost:$PORT/gql" \
  --data 'MATCH (u1:User)-[f:Friend]-(u2:User) RETURN count(DISTINCT f) AS unique_friend_edges'
echo

echo "Step 3: Create projection"
echo "-------------------------"
curl -s -X POST "http://localhost:$PORT/gql" \
  --data 'MATCH (u1:User)-[f:Friend]-(u2:User) RETURN PROJECT("debug_friends")'
echo

sleep 1

echo ""
echo "Step 4: Count edges in projection"
echo "----------------------------------"
curl -s -X POST "http://localhost:$PORT/gql" \
  --data 'USE "debug_friends" MATCH ()-[e]-() RETURN count(DISTINCT e) AS edges_in_projection'
echo

echo "Step 5: Sample edges from projection"
echo "-------------------------------------"
curl -s -X POST "http://localhost:$PORT/gql" \
  --data 'USE "debug_friends" MATCH ()-[e]-() RETURN e LIMIT 10'
echo

# Check if projection has directed edges (which shouldn't be there)
echo "Step 6: Check edge types in projection"
echo "---------------------------------------"
echo "Note: Friend edges are undirected, Posted edges are directed"
echo ""

curl -s -X POST "http://localhost:$PORT/gql" \
  --data 'USE "debug_friends" MATCH ()-[e]-() RETURN e LIMIT 20' > /tmp/proj_edges.txt

# Count how many start with _u (undirected) vs _e (directed)
UNDIRECTED=$(grep -o "_u[0-9]*" /tmp/proj_edges.txt | sort -u | wc -l)
DIRECTED=$(grep -o "_e[0-9]*" /tmp/proj_edges.txt | sort -u | wc -l)

echo "Undirected edges (_u*): $UNDIRECTED"
echo "Directed edges (_e*): $DIRECTED"
echo

if [ "$DIRECTED" -gt "0" ]; then
    echo "⚠️  WARNING: Projection contains directed edges!"
    echo "   This suggests Posted edges are being included"
    echo "   when they shouldn't be."
fi

# Cleanup
kill $SERVER_PID
wait $SERVER_PID 2>/dev/null || true

echo ""
echo "=========================================="
echo "ANALYSIS"
echo "=========================================="
echo "Expected: 50 unique Friend edges (all undirected)"
echo "Check the counts above to see if they match."
echo "=========================================="
