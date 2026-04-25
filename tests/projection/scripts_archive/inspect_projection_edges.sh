#!/bin/bash

# Script to inspect what edges are actually in the projection

set -e

pkill -f "mdb server" || true
sleep 2

echo "Starting server..."
./build/Release/bin/mdb server ./data/test_normalization_fix --port 1234 --browser false >/tmp/server_inspect.log 2>&1 &
SRV_PID=$!
sleep 4

if ! kill -0 $SRV_PID 2>/dev/null; then
    echo "ERROR: Server failed to start"
    cat /tmp/server_inspect.log
    exit 1
fi

echo "=========================================="
echo "Inspecting Projection Edge Details"
echo "=========================================="
echo ""

echo "1. Unique edge IDs in projection (first 100):"
curl -s -X POST "http://localhost:1234/gql" --data 'USE "test_friends" MATCH ()-[e]->() RETURN DISTINCT e LIMIT 100' > /tmp/proj_edges.txt
wc -l /tmp/proj_edges.txt
echo ""

echo "2. Count edges by label in projection:"
echo "   Friend edges:"
curl -s -X POST "http://localhost:1234/gql" --data 'USE "test_friends" MATCH ()-[e:Friend]-() RETURN count(DISTINCT e)'
echo ""

echo "   Posted edges:"
curl -s -X POST "http://localhost:1234/gql" --data 'USE "test_friends" MATCH ()-[e:Posted]-() RETURN count(DISTINCT e)'
echo ""

echo "   All edges (no label filter):"
curl -s -X POST "http://localhost:1234/gql" --data 'USE "test_friends" MATCH ()-[e]-() RETURN count(DISTINCT e)'
echo ""

kill $SRV_PID 2>/dev/null
wait $SRV_PID 2>/dev/null || true

echo "Inspection complete!"
