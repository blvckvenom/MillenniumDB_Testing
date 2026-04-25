#!/bin/bash

# Test script for PROJECT function after validation fix
# Tests the two queries that were failing

echo "=========================================="
echo "Testing PROJECT function - Validation Fix"
echo "=========================================="
echo ""

# Make sure we're using the Release build
MDB_BIN="build/Release/bin/mdb"

if [ ! -f "$MDB_BIN" ]; then
    echo "ERROR: $MDB_BIN not found. Please build first."
    exit 1
fi

# Database path
DB_PATH="data/dbs/gql/posts"

if [ ! -d "$DB_PATH" ]; then
    echo "WARNING: Database at $DB_PATH not found."
    echo "Creating database from example data..."
    $MDB_BIN import data/example/gql/posts/posts.gql $DB_PATH
fi

echo "Starting server in background..."
$MDB_BIN server $DB_PATH --port 1234 &
SERVER_PID=$!

# Wait for server to start
echo "Waiting for server to initialize..."
sleep 3

# Check if server is running
if ! ps -p $SERVER_PID > /dev/null; then
    echo "ERROR: Server failed to start"
    exit 1
fi

echo "Server started with PID $SERVER_PID"
echo ""

# Test Query 1
echo "=========================================="
echo "Test 1: MATCH (p:Paper) RETURN PROJECT('all_papers')"
echo "=========================================="
curl --noproxy localhost -H "Accept:text/csv" \
     --data "MATCH (p:Paper) RETURN PROJECT('all_papers')" \
     -X POST http://localhost:1234
echo -e "\n"

# Test Query 2
echo "=========================================="
echo "Test 2: MATCH (p) RETURN PROJECT('test')"
echo "=========================================="
curl --noproxy localhost -H "Accept:text/csv" \
     --data "MATCH (p) RETURN PROJECT('test')" \
     -X POST http://localhost:1234
echo -e "\n"

# Clean up
echo "=========================================="
echo "Stopping server..."
kill $SERVER_PID 2>/dev/null
wait $SERVER_PID 2>/dev/null

echo "Test complete!"
echo "=========================================="
