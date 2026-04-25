#!/bin/bash

pkill -f "mdb server" || true
sleep 2

./build/Release/bin/mdb server ./data/debug_edge_trace --port 1234 --browser false >/tmp/label_query_test.log 2>&1 &
SRV_PID=$!
sleep 4

if ! kill -0 $SRV_PID 2>/dev/null; then
    echo "ERROR: Server failed to start"
    exit 1
fi

echo "1. Query without label (should work):"
curl -s -X POST "http://localhost:1234/gql" --data 'USE "debug_friends" MATCH ()-[e]-() RETURN count(DISTINCT e)'
echo ""
echo ""

echo "2. Query with Friend label (should fail or return wrong count):"
curl -s -X POST "http://localhost:1234/gql" --data 'USE "debug_friends" MATCH ()-[e:Friend]-() RETURN count(DISTINCT e)'
echo ""
echo ""

echo "3. Query with Posted label (should fail or return 0):"
curl -s -X POST "http://localhost:1234/gql" --data 'USE "debug_friends" MATCH ()-[e:Posted]-() RETURN count(DISTINCT e)'
echo ""

kill $SRV_PID 2>/dev/null
wait $SRV_PID 2>/dev/null || true
