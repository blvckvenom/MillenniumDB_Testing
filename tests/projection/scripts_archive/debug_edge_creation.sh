#!/bin/bash

# Focused debug script to trace edge creation in projection

set -e

rm -rf ./data/debug_edge_trace
mkdir -p ./data/debug_edge_trace

pkill -f "mdb server" || true
sleep 2

echo "Importing data..."
./build/Release/bin/mdb import ./data/example/gql/posts/posts.gql ./data/debug_edge_trace >/dev/null 2>&1

echo "Starting server..."
./build/Release/bin/mdb server ./data/debug_edge_trace --port 1234 --browser false >/tmp/debug_server.log 2>&1 &
SRV_PID=$!
sleep 4

if ! kill -0 $SRV_PID 2>/dev/null; then
    echo "ERROR: Server failed to start"
    cat /tmp/debug_server.log
    exit 1
fi

echo "Creating projection with Friend edges only..."
echo "=========================================="
curl -s -X POST "http://localhost:1234/gql" --data 'MATCH (u1:User)-[f:Friend]-(u2:User) RETURN PROJECT("debug_friends")' 2>&1 | head -5
echo ""
echo "=========================================="

sleep 2

echo ""
echo "Checking projection contents..."
curl -s -X POST "http://localhost:1234/gql" --data 'USE "debug_friends" MATCH ()-[e:Friend]-() RETURN count(DISTINCT e)'
curl -s -X POST "http://localhost:1234/gql" --data 'USE "debug_friends" MATCH ()-[e:Posted]-() RETURN count(DISTINCT e)'
curl -s -X POST "http://localhost:1234/gql" --data 'USE "debug_friends" MATCH ()-[e]-() RETURN count(DISTINCT e)'

kill $SRV_PID 2>/dev/null
wait $SRV_PID 2>/dev/null || true

echo ""
echo "Server log (showing edge additions):"
grep -E "\[AggProject\]|\[ProjectionStorage\]" /tmp/debug_server.log | head -200
