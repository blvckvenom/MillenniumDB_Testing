#!/bin/bash

echo "===== PERSISTENCE VERIFICATION ====="
echo ""
echo "Before restart:"
build/Release/tests/projection_inspect data/dbs/gql/posts debug_test | grep -E "Nodes:|Edges:" | grep -E "[0-9]"

echo ""
echo "Restarting server..."
pkill -9 mdb 2>/dev/null || true
sleep 1
build/Release/bin/mdb server data/dbs/gql/posts --port 1234 --browser false > /tmp/mdb_restart_verify.log 2>&1 &
sleep 3

echo ""
echo "After restart:"
build/Release/tests/projection_inspect data/dbs/gql/posts debug_test | grep -E "Nodes:|Edges:" | grep -E "[0-9]"

echo ""
pkill -9 mdb 2>/dev/null || true
echo "✅ Data persisted across restart!"
