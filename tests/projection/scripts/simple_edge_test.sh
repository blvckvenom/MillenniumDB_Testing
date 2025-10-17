#!/bin/bash
pkill -f "mdb server" || true
sleep 2

./build/Release/bin/mdb server ./data/test_proj_debug --port 1234 --browser false >/dev/null 2>&1 &
SV=$!
sleep 4

echo "Total edges in main graph:"
curl -s -X POST "http://localhost:1234/gql" --data 'MATCH ()-[e]-() RETURN count(DISTINCT e)'

echo ""
echo "Friend edges:"
curl -s -X POST "http://localhost:1234/gql" --data 'MATCH ()-[e:Friend]-() RETURN count(DISTINCT e)'

echo ""
echo "Posted edges:"
curl -s -X POST "http://localhost:1234/gql" --data 'MATCH ()-[e:Posted]-() RETURN count(DISTINCT e)'

echo ""
echo "Projection edges:"
curl -s -X POST "http://localhost:1234/gql" --data 'USE "debug_friends" MATCH ()-[e]-() RETURN count(DISTINCT e)'

kill $SV 2>/dev/null
wait $SV 2>/dev/null || true
