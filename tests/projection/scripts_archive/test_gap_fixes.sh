#!/bin/bash
# Diagnose the 100 vs 250 edge count discrepancy

set -e

MDB_BIN="./build/Release/bin/mdb"
TEST_DB="./data/test_gap_debug"
SOURCE_DATA="./data/example/gql/posts/posts.gql"
PORT=1234

# Cleanup
rm -rf "$TEST_DB"
pkill -f "mdb server" || true
sleep 1

# Import
$MDB_BIN import "$SOURCE_DATA" "$TEST_DB" > /dev/null 2>&1

# Start server
$MDB_BIN server "$TEST_DB" --port $PORT --browser false > /tmp/gap_test.log 2>&1 &
SERVER_PID=$!
sleep 3

echo "=========================================="
echo " DIAGNOSTIC: Edge Count Mismatch"
echo "=========================================="
echo

echo "Main graph statistics:"
echo "----------------------"
curl -s -X POST "http://localhost:$PORT/gql" --data 'MATCH (n) RETURN count(DISTINCT n) AS nodes'
curl -s -X POST "http://localhost:$PORT/gql" --data 'MATCH ()-[e]-() RETURN count(DISTINCT e) AS unique_edges'
curl -s -X POST "http://localhost:$PORT/gql" --data 'MATCH ()-[e]-() RETURN count(*) AS edge_pattern_matches'
curl -s -X POST "http://localhost:$PORT/gql" --data 'MATCH (u1:User)-[f:Friend]-(u2:User) RETURN count(DISTINCT f) AS unique_friend_edges'
curl -s -X POST "http://localhost:$PORT/gql" --data 'MATCH (u1:User)-[f:Friend]-(u2:User) RETURN count(*) AS friend_pattern_matches'
echo

echo "Edge types breakdown:"
echo "--------------------"
curl -s -X POST "http://localhost:$PORT/gql" --data 'MATCH ()-[e:Friend]-() RETURN count(DISTINCT e) AS friend_edges'
curl -s -X POST "http://localhost:$PORT/gql" --data 'MATCH ()-[e:Posted]-() RETURN count(DISTINCT e) AS posted_edges'
echo

echo "Creating minimal projection:"
echo "---------------------------"
curl -s -X POST "http://localhost:$PORT/gql" --data 'MATCH (u1:User)-[f:Friend]-(u2:User) RETURN PROJECT("debug_proj")'
echo

sleep 1

echo "Query projection:"
echo "-----------------"
curl -s -X POST "http://localhost:$PORT/gql" --data 'USE "debug_proj" MATCH ()-[e]-() RETURN count(DISTINCT e) AS unique_edges_in_projection'
curl -s -X POST "http://localhost:$PORT/gql" --data 'USE "debug_proj" MATCH ()-[e]-() RETURN count(*) AS edge_pattern_matches_in_projection'
echo

echo "Checking projection directory:"
echo "------------------------------"
if [ -d "$TEST_DB/projections/debug_proj" ]; then
    ls -lh "$TEST_DB/projections/debug_proj"
fi
echo

# Cleanup
kill $SERVER_PID
wait $SERVER_PID 2>/dev/null || true

echo ""
echo "=========================================="
echo "ANALYSIS:"
echo "=========================================="
echo "The discrepancy is likely because:"
echo "  1. Undirected edges appear twice in MATCH results"
echo "  2. Each occurrence gets stored separately"
echo "  3. Projection ends up with 2x or more edges"
echo "=========================================="
