#!/bin/bash

# Final verification test for USE GRAPH bug fix
# This test verifies that the critical bug is fixed

set -e

echo "=========================================="
echo "Final Verification Test: USE GRAPH Fix"
echo "=========================================="
echo ""

# Setup
rm -rf ./data/final_test
mkdir -p ./data/final_test

echo "1. Importing test data..."
./build/Release/bin/mdb import ./data/example/gql/posts/posts.gql ./data/final_test >/dev/null 2>&1

echo "2. Starting server..."
pkill -f "mdb server" 2>/dev/null || true
sleep 2
./build/Release/bin/mdb server ./data/final_test --port 1234 --browser false >/tmp/final_test.log 2>&1 &
SRV_PID=$!
sleep 5

if ! kill -0 $SRV_PID 2>/dev/null; then
    echo "ERROR: Server failed to start"
    cat /tmp/final_test.log
    exit 1
fi

echo "3. Creating projection with Friend edges only..."
curl -s -X POST "http://localhost:1234/gql" \
  --data 'MATCH (u1:User)-[f:Friend]-(u2:User) RETURN PROJECT("test_friends")' >/dev/null

sleep 2

echo "4. Running verification queries..."
echo ""

# Test 1: Main graph Friend edges
echo "   Test 1: Main graph Friend edges"
MAIN_FRIENDS=$(curl -s -X POST "http://localhost:1234/gql" \
  --data 'MATCH ()-[e:Friend]-() RETURN count(DISTINCT e)' | grep -o '[0-9]\+' | head -1)
echo "      Result: $MAIN_FRIENDS edges"

# Test 2: Main graph total edges
echo "   Test 2: Main graph total edges"
MAIN_TOTAL=$(curl -s -X POST "http://localhost:1234/gql" \
  --data 'MATCH ()-[e]-() RETURN count(DISTINCT e)' | grep -o '[0-9]\+' | head -1)
echo "      Result: $MAIN_TOTAL edges"

# Test 3: Projection total edges (CRITICAL TEST)
echo "   Test 3: Projection total edges (CRITICAL)"
PROJ_TOTAL=$(curl -s -X POST "http://localhost:1234/gql" \
  --data 'USE "test_friends" MATCH ()-[e]-() RETURN count(DISTINCT e)' | grep -o '[0-9]\+' | head -1)
echo "      Result: $PROJ_TOTAL edges"

echo ""
echo "=========================================="
echo "VERIFICATION RESULTS"
echo "=========================================="

PASSED=0
FAILED=0

# Check 1: Main graph should have 50 Friend edges
if [ "$MAIN_FRIENDS" -eq 50 ]; then
    echo "✓ PASS: Main graph has 50 Friend edges"
    PASSED=$((PASSED + 1))
else
    echo "✗ FAIL: Main graph Friend count incorrect (expected 50, got $MAIN_FRIENDS)"
    FAILED=$((FAILED + 1))
fi

# Check 2: Main graph should have 125 total edges
if [ "$MAIN_TOTAL" -eq 125 ]; then
    echo "✓ PASS: Main graph has 125 total edges"
    PASSED=$((PASSED + 1))
else
    echo "✗ FAIL: Main graph total incorrect (expected 125, got $MAIN_TOTAL)"
    FAILED=$((FAILED + 1))
fi

# Check 3: CRITICAL - Projection should have 50 edges (not 125)
if [ "$PROJ_TOTAL" -eq 50 ]; then
    echo "✓ PASS: ★★★ Projection has 50 edges (BUG FIXED!) ★★★"
    PASSED=$((PASSED + 1))
elif [ "$PROJ_TOTAL" -eq 125 ]; then
    echo "✗ FAIL: ★★★ Projection has 125 edges (BUG STILL PRESENT) ★★★"
    echo "         This means USE GRAPH is still using main graph indexes"
    FAILED=$((FAILED + 1))
else
    echo "✗ FAIL: Projection has unexpected count: $PROJ_TOTAL"
    FAILED=$((FAILED + 1))
fi

echo "=========================================="
echo "Summary: $PASSED passed, $FAILED failed"
echo "=========================================="
echo ""

# Additional diagnostic info
if [ "$PROJ_TOTAL" -eq 125 ]; then
    echo "DIAGNOSTIC INFO:"
    echo "The projection still returns 125 edges, which indicates:"
    echo "  1. load_projection() is not being called, OR"
    echo "  2. projection_ctx is not being used by queries, OR"
    echo "  3. get_from_to_edge() is not checking projection context"
    echo ""
    echo "Check server log for debug messages:"
    grep -E "\[QueryContext\]|\[GQLModel\]" /tmp/final_test.log | tail -20
fi

# Cleanup
kill $SRV_PID 2>/dev/null
wait $SRV_PID 2>/dev/null || true

if [ "$FAILED" -gt 0 ]; then
    exit 1
else
    exit 0
fi
