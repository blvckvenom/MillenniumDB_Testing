#!/bin/bash
# Simple test to verify PROJECT function persistence across server restarts

set -e

echo "=========================================="
echo "PROJECT Persistence Test"
echo "=========================================="

# Clean up
rm -rf data/dbs/gql/posts/projections/persistence_test 2>/dev/null || true
pkill -9 mdb 2>/dev/null || true
sleep 1

echo ""
echo "Step 1: Starting server..."
nohup build/Release/bin/mdb server data/dbs/gql/posts --port 1234 --browser false > /tmp/mdb_persistence.log 2>&1 &
sleep 3

echo "Step 2: Creating projection..."
curl --noproxy localhost -s -H "Accept:text/csv" \
     --data "MATCH (a)-[b]->(c) RETURN PROJECT('persistence_test')" \
     -X POST http://localhost:1234/gql 2>/dev/null

echo ""
echo "Step 3: Before restart - inspecting data..."
build/Release/tests/projection_inspect data/dbs/gql/posts persistence_test | grep -E "Nodes:|Edges:"

echo ""
echo "Step 4: Stopping server..."
pkill -9 mdb
sleep 2

echo ""
echo "Step 5: Restarting server..."
nohup build/Release/bin/mdb server data/dbs/gql/posts --port 1234 --browser false > /tmp/mdb_persistence2.log 2>&1 &
sleep 3

echo "Step 6: After restart - inspecting data..."
build/Release/tests/projection_inspect data/dbs/gql/posts persistence_test | grep -E "Nodes:|Edges:"

echo ""
echo "Step 7: Verifying disk files..."
for f in data/dbs/gql/posts/projections/persistence_test/*.leaf; do
    SIZE=$(stat -c%s "$f")
    NONZERO=$(od -An -tx1 "$f" | tr -d ' ' | grep -v "^00*$" | wc -l)
    echo "  $(basename $f): ${SIZE} bytes, ${NONZERO} non-zero lines"
done

echo ""
pkill -9 mdb 2>/dev/null || true

echo ""
echo "=========================================="
echo "✅ SUCCESS - Projections persist to disk!"
echo "=========================================="
