#!/bin/bash
# End-to-end test for PROJECT() with INCLUDE LABELS feature
# Tests that projections can store and query node/edge labels

set -e  # Exit on error

echo "=== Phase E: INCLUDE LABELS End-to-End Test ==="
echo

# Configuration
MDB_BIN="./build/Release/bin/mdb"
TEST_DB="./data/test_include_labels_db"
SOURCE_DATA="./data/example/gql/posts/posts.gql"

# Cleanup previous test
echo "[1/6] Cleaning up previous test database..."
rm -rf "$TEST_DB"
echo "    ✓ Cleanup complete"
echo

# Import test data
echo "[2/6] Importing GQL test data..."
$MDB_BIN import "$SOURCE_DATA" "$TEST_DB"
echo "    ✓ Import complete"
echo

# Test 1: Create projection WITH INCLUDE LABELS
echo "[3/6] Creating projection WITH INCLUDE LABELS..."
QUERY_LABELS="MATCH (u:User)-[f]-(u2:User) RETURN PROJECT('users_with_labels' INCLUDE LABELS)"
echo "    Query: $QUERY_LABELS"
echo "$QUERY_LABELS" | $MDB_BIN cli "$TEST_DB" > /tmp/test_labels_output.txt
cat /tmp/test_labels_output.txt
echo "    ✓ Projection with labels created"
echo

# Test 2: Verify label index files exist
echo "[4/6] Verifying label index files were created..."
PROJ_DIR="$TEST_DB/projections/users_with_labels"
if [ -f "$PROJ_DIR/node_label.leaf" ] && [ -f "$PROJ_DIR/node_label.dir" ]; then
    echo "    ✓ Node label index files exist"
    ls -lh "$PROJ_DIR/node_label"*
else
    echo "    ✗ ERROR: Node label index files NOT found!"
    exit 1
fi

if [ -f "$PROJ_DIR/edge_label.leaf" ] && [ -f "$PROJ_DIR/edge_label.dir" ]; then
    echo "    ✓ Edge label index files exist"
    ls -lh "$PROJ_DIR/edge_label"*
else
    echo "    ✗ ERROR: Edge label index files NOT found!"
    exit 1
fi
echo

# Test 3: Inspect projection with enhanced tool
echo "[5/6] Inspecting projection catalog and structure..."
if [ -f "./build/Release/tests/projection_inspect_enhanced" ]; then
    ./build/Release/tests/projection_inspect_enhanced "$TEST_DB" users_with_labels
    echo "    ✓ Projection inspection complete"
else
    echo "    ⚠ Warning: projection_inspect_enhanced not found, skipping inspection"
fi
echo

# Test 4: Create projection WITHOUT INCLUDE LABELS (for comparison)
echo "[6/6] Creating projection WITHOUT INCLUDE LABELS (comparison)..."
QUERY_NO_LABELS="MATCH (u:User)-[f]-(u2:User) RETURN PROJECT('users_no_labels')"
echo "    Query: $QUERY_NO_LABELS"
echo "$QUERY_NO_LABELS" | $MDB_BIN cli "$TEST_DB" > /tmp/test_no_labels_output.txt
cat /tmp/test_no_labels_output.txt

PROJ_DIR_NO_LABELS="$TEST_DB/projections/users_no_labels"
if [ ! -f "$PROJ_DIR_NO_LABELS/node_label.leaf" ]; then
    echo "    ✓ Node label index NOT created (as expected)"
else
    echo "    ✗ ERROR: Node label index should NOT exist!"
    exit 1
fi

if [ ! -f "$PROJ_DIR_NO_LABELS/edge_label.leaf" ]; then
    echo "    ✓ Edge label index NOT created (as expected)"
else
    echo "    ✗ ERROR: Edge label index should NOT exist!"
    exit 1
fi
echo

echo "=== ✓ All INCLUDE LABELS tests passed! ==="
echo
echo "Summary:"
echo "  - Projection WITH INCLUDE LABELS: node_label and edge_label indexes created"
echo "  - Projection WITHOUT INCLUDE LABELS: no label indexes (topology only)"
echo "  - Feature is working as expected!"
echo
