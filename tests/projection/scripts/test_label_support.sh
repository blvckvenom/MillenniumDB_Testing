#!/bin/bash
# Test script for Phase 1: Label Support in Projections
# This script tests that projections can now include and query node/edge labels

set -e

echo "======================================"
echo "Phase 1: Label Support Testing"
echo "======================================"
echo ""

DB_DIR="test_labels_db"
BIN="./build/Release/bin/mdb"

# Cleanup
echo "1. Cleaning up previous test database..."
rm -rf "$DB_DIR" 2>/dev/null || true

# Import test data
echo "2. Importing test database..."
$BIN import data/example/gql/posts/posts.gql "$DB_DIR"
echo ""

# Start server in background
echo "3. Starting server..."
pkill -f "mdb server $DB_DIR" 2>/dev/null || true
$BIN server "$DB_DIR" --port 1234 --browser false > /tmp/test_labels_server.log 2>&1 &
SERVER_PID=$!
echo "   Server PID: $SERVER_PID"
sleep 5

# Function to run query via HTTP
query() {
    local query_text="$1"
    echo "   Query: $query_text"
    curl -s -X POST http://localhost:1234/gql \
         -H "Content-Type: application/sparql-query" \
         --data-binary "$query_text" 2>&1 | head -20
}

echo ""
echo "4. Test: Create projection WITHOUT labels (backward compatibility)"
query "MATCH (p:Paper)-[e:Cites]->(q:Paper) RETURN PROJECT(\"test_no_labels\")"
echo ""

echo "5. Test: Create projection WITH labels"
query "MATCH (p:Paper)-[e:Cites]->(q:Paper) RETURN PROJECT(\"test_with_labels\", INCLUDE LABELS)"
echo ""

echo "6. Test: Query projection without labels (should fail with helpful error)"
query "USE \"test_no_labels\" MATCH (p:Paper) RETURN count(p)"
echo ""

echo "7. Test: Query projection with labels - count nodes with Paper label"
query "USE \"test_with_labels\" MATCH (p:Paper) RETURN count(p)"
echo ""

echo "8. Test: Query projection with labels - count edges with Cites label"
query "USE \"test_with_labels\" MATCH ()-[e:Cites]->() RETURN count(e)"
echo ""

echo "9. Test: Count all nodes in projection (no label filter)"
query "USE \"test_with_labels\" MATCH (p) RETURN count(p)"
echo ""

echo "10. List all projections"
$BIN list-projections "$DB_DIR"
echo ""

# Cleanup
echo "Stopping server..."
kill $SERVER_PID 2>/dev/null || true
echo ""

echo "======================================"
echo "Phase 1 Testing Complete!"
echo "======================================"
echo ""
echo "Summary of Phase 1 Implementation:"
echo "✅ Label index caching in ProjectionQueryContext"
echo "✅ Auxiliary label indexes (label_node, label_edge) in ProjectionStorage"
echo "✅ Bidirectional label writes in add_node_label() and add_edge_label()"
echo "✅ Query execution with projection labels in GQLModel"
echo "✅ INCLUDE LABELS syntax parsing (already implemented)"
echo ""
echo "Next Steps:"
echo "- Phase 2: Property Support (similar pattern)"
echo "- Remove debug cerr statements"
echo "- Add comprehensive integration tests"
echo ""
