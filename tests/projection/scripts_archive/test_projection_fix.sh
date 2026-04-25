#!/usr/bin/env bash
set -e

echo "Testing PROJECT() fix..."
echo

# Kill any existing server
pkill -f "mdb server" 2>/dev/null || true
sleep 1

# Start server in background
echo "Starting server..."
build/Release/bin/mdb server data/dbs/gql/posts --port 1234 --browser false > /dev/null 2>&1 &
SERVER_PID=$!
sleep 3

# Test query
echo "Running test query..."
cat > /tmp/test_query.gql << 'EOF'
MATCH (u:User)-[f:FRIEND]->(v:User)
WHERE u.age > 18
RETURN PROJECT('adult_friendships')
EOF

# Execute query
RESULT=$(curl -s -X POST http://localhost:1234/gql \
    -H "Content-Type: application/sparql-query" \
    --data-binary @/tmp/test_query.gql)

echo "Query result:"
echo "${RESULT}"
echo

# Stop server
kill ${SERVER_PID} 2>/dev/null || true
wait ${SERVER_PID} 2>/dev/null || true

# Check projection was created
if [ -d "data/dbs/gql/posts/projections/adult_friendships" ]; then
    echo "✓ SUCCESS: Projection was created!"
    ls -la data/dbs/gql/posts/projections/adult_friendships/
else
    echo "✗ FAILED: Projection directory not found"
    exit 1
fi

echo
echo "Test completed successfully!"
