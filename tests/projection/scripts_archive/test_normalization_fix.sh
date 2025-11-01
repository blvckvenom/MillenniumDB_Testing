#!/bin/bash

# Test script to verify undirected edge normalization fix
# Expected behavior: Projection should contain exactly 50 Friend edges (not 125)

set -e

echo "=========================================="
echo "Testing Undirected Edge Normalization Fix"
echo "=========================================="
echo ""

# Clean up previous test database
rm -rf ./data/test_normalization_fix
mkdir -p ./data/test_normalization_fix

# Kill any existing server
pkill -f "mdb server" || true
sleep 2

# Import the posts dataset
echo "1. Importing posts dataset..."
./build/Release/bin/mdb import ./data/example/gql/posts/posts.gql ./data/test_normalization_fix
echo ""

# Start server
echo "2. Starting server..."
./build/Release/bin/mdb server ./data/test_normalization_fix --port 1234 --browser false >/tmp/server_normalization.log 2>&1 &
SRV_PID=$!
sleep 4

# Verify server is running
if ! kill -0 $SRV_PID 2>/dev/null; then
    echo "ERROR: Server failed to start"
    cat /tmp/server_normalization.log
    exit 1
fi

echo "3. Verifying main graph edge counts..."
echo ""

# Count total edges in main graph
echo "   a) Total edges in main graph:"
TOTAL_EDGES=$(curl -s -X POST "http://localhost:1234/gql" --data 'MATCH ()-[e]-() RETURN count(DISTINCT e)' | grep -o '[0-9]\+' | head -1)
echo "      Result: $TOTAL_EDGES edges"
echo ""

# Count Friend edges (undirected)
echo "   b) Friend edges (undirected) in main graph:"
FRIEND_EDGES=$(curl -s -X POST "http://localhost:1234/gql" --data 'MATCH ()-[e:Friend]-() RETURN count(DISTINCT e)' | grep -o '[0-9]\+' | head -1)
echo "      Result: $FRIEND_EDGES edges"
echo ""

# Count Posted edges (directed)
echo "   c) Posted edges (directed) in main graph:"
POSTED_EDGES=$(curl -s -X POST "http://localhost:1234/gql" --data 'MATCH ()-[e:Posted]-() RETURN count(DISTINCT e)' | grep -o '[0-9]\+' | head -1)
echo "      Result: $POSTED_EDGES edges"
echo ""

echo "4. Creating projection with only Friend edges..."
PROJ_RESULT=$(curl -s -X POST "http://localhost:1234/gql" --data 'MATCH (u1:User)-[f:Friend]-(u2:User) RETURN PROJECT("test_friends")')
echo "   Projection created: $PROJ_RESULT"
echo ""

# Small delay to ensure projection is flushed
sleep 2

echo "5. Querying projection to verify edge count..."
echo ""

# Count edges in projection
echo "   a) Total edges in projection:"
PROJ_EDGES=$(curl -s -X POST "http://localhost:1234/gql" --data 'USE "test_friends" MATCH ()-[e]-() RETURN count(DISTINCT e)' | grep -o '[0-9]\+' | head -1)
echo "      Result: $PROJ_EDGES edges"
echo ""

# Verify the count
echo "6. Verification Results:"
echo "=========================================="
if [ "$PROJ_EDGES" -eq "$FRIEND_EDGES" ]; then
    echo "✓ SUCCESS: Projection contains correct number of edges!"
    echo "  Expected: $FRIEND_EDGES Friend edges"
    echo "  Got:      $PROJ_EDGES edges in projection"
    echo ""
    echo "The undirected edge normalization fix is working correctly."
    RESULT=0
else
    echo "✗ FAILURE: Edge count mismatch!"
    echo "  Expected: $FRIEND_EDGES Friend edges"
    echo "  Got:      $PROJ_EDGES edges in projection"
    echo ""
    echo "The bug may not be fully fixed."
    RESULT=1
fi
echo "=========================================="
echo ""

# Additional verification: Query both main graph and projection to compare results
echo "7. Additional Verification - Comparing query results:"
echo ""

echo "   a) Main graph Friend query (first 10 edges):"
curl -s -X POST "http://localhost:1234/gql" --data 'MATCH (u1:User)-[f:Friend]-(u2:User) RETURN u1, f, u2 LIMIT 10' | head -20
echo ""

echo "   b) Projection query (first 10 edges):"
curl -s -X POST "http://localhost:1234/gql" --data 'USE "test_friends" MATCH (u1)-[f]-(u2) RETURN u1, f, u2 LIMIT 10' | head -20
echo ""

# Cleanup
kill $SRV_PID 2>/dev/null
wait $SRV_PID 2>/dev/null || true

echo "Test complete!"
exit $RESULT
