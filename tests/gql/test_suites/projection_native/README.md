# Native Projection Test Suite

Comprehensive integration test suite for the M1.2 Native Projection feature in GQL.

## Overview

This test suite validates the `CALL PROJECT(...)` procedure, which creates graph projections by directly scanning B+Tree indexes for node labels and edge types. Native projection provides 2-3x performance improvement over Cypher-based projection.

## Test Status

Status: Ready for implementation
Total Tests: 20 integration tests
Coverage Target: >85% of Native Projection functionality

## Test Structure

```
tests/gql/test_suites/projection_native/
├── README.md                          # This file
├── projection_native.gql              # Test data (15 nodes, 19 edges)
└── queries/
    ├── 01_string_variant_single.*     # Basic string variant
    ├── 02_list_variant_multiple_labels.*  # List with multiple labels
    ├── 03-20_*.{mql,csv}              # 18 additional test cases
    └── ...
```

## Test Data Summary

**File**: `projection_native.gql`

### Nodes (15 total)
- **User**: 5 nodes (Alice, Bob, Carol, Dave, Eve)
- **Post**: 3 nodes
- **Comment**: 2 nodes
- **Product**: 2 nodes (isolated, no edges)
- **Tag**: 3 nodes

### Edges (19 total)
- **KNOWS**: 5 edges (User -> User, forms a cycle)
- **LIKES**: 4 edges (User -> Post)
- **WROTE**: 2 edges (User -> Comment)
- **COMMENTED_ON**: 2 edges (Comment -> Post)
- **FOLLOWS**: 2 edges (User -> User, sparse)
- **HAS_TAG**: 4 edges (Post -> Tag)

## Test Categories

### 1. Basic Projection (Tests 1-6)

| Test | Description | Expected Result |
|------|-------------|-----------------|
| 01 | String variant single label/type | 5 nodes, 5 edges |
| 02 | List variant multiple labels | 8 nodes, 9 edges |
| 03 | List with three labels | 10 nodes, 8 edges |
| 04 | All nodes, single edge type | 15 nodes, 4 edges |
| 05 | Edge endpoint filtering | 5 nodes, 0 edges (all filtered) |
| 06 | Isolated nodes (no edges) | 2 nodes, 0 edges |

**Coverage**: String/list variants, endpoint filtering, isolated nodes

### 2. YIELD Variations (Tests 7-10)

| Test | Description | Expected Result |
|------|-------------|-----------------|
| 07 | Partial YIELD fields | Only graphName, nodeCount |
| 08 | YIELD with aliases | Renamed fields in output |
| 09 | RETURN * (all fields) | All 3 YIELD fields |
| 10 | Check graphName and nodeCount | graphName and nodeCount only |

**Coverage**: YIELD field selection, aliases, RETURN *, timing

### 3. Complex Scenarios (Tests 11-15)

| Test | Description | Expected Result |
|------|-------------|-----------------|
| 11 | Single node label (Tag) | 3 nodes, 0 edges |
| 12 | Bidirectional edges | Multiple edge types |
| 13 | Cross-label edges | User + Post, LIKES edges |
| 14 | Chain projection | 3-hop path (User->Comment->Post) |
| 15 | All edge types | All 6 edge types included |

**Coverage**: Complex label/type combinations, multi-hop paths

### 4. Query Clauses (Tests 16-20)

| Test | Description | Expected Result |
|------|-------------|-----------------|
| 16 | Expression in RETURN | Computed total field |
| 17 | Constant in RETURN | Added status field |
| 18 | DISTINCT clause | Deduplicated results |
| 19 | ORDER BY clause | Sorted by nodeCount |
| 20 | LIMIT clause | Single result row |

**Coverage**: SQL-style clauses (DISTINCT, ORDER BY, LIMIT, expressions)

## Running the Tests

### Run All GQL Tests
```bash
cd <project-root>
./scripts/run-tests gql
```

### Run Only Native Projection Tests
```bash
cd tests/gql
python3 scripts/test.py --executable ../../build/Debug/bin/mdb | grep projection_native
```

### Run Single Test
```bash
cd tests/gql
python3 scripts/test.py --executable ../../build/Debug/bin/mdb \
  --suite projection_native \
  --test 01_string_variant_single
```

### Expected Output
```
projection_native: 100%
[SUMMARY] -------------- projection_native ---------------
[SUMMARY]
[SUMMARY]           CORRECT             :  20
[SUMMARY]           ERROR               :   0
[SUMMARY]           SKIPPED             :   0
[SUMMARY]           TOTAL               :  20
```

## Test Configuration

Add to `tests/gql/scripts/testing/options.py`:

```python
TEST_SUITES: list[str] = [
    "simple",
    "edges",
    # ... existing suites ...
    "call_procedure",
    "projection_native",  # Add this line
]
```

## Test Query Format

### Example 1: Basic String Variant
**File**: `queries/01_string_variant_single.mql`
```gql
CALL PROJECT('test_single', 'User', 'KNOWS')
YIELD graphName, nodeCount, relationshipCount
RETURN graphName, nodeCount, relationshipCount
```

**Expected**: `queries/01_string_variant_single.csv`
```csv
graphName,nodeCount,relationshipCount
"test_single",5,5
```

### Example 2: List Variant
**File**: `queries/02_list_variant_multiple_labels.mql`
```gql
CALL PROJECT('test_multi', ['User', 'Post'], ['KNOWS', 'LIKES'])
YIELD graphName, nodeCount, relationshipCount
RETURN graphName, nodeCount, relationshipCount
```

**Expected**: `queries/02_list_variant_multiple_labels.csv`
```csv
graphName,nodeCount,relationshipCount
"test_multi",8,9
```

### Example 3: Edge Filtering
**File**: `queries/05_edge_endpoint_filtering.mql`
```gql
CALL PROJECT('test_filtering', 'User', ['LIKES', 'WROTE', 'COMMENTED_ON'])
YIELD graphName, nodeCount, relationshipCount
RETURN graphName, nodeCount, relationshipCount
```

**Expected**: 0 edges (LIKES, WROTE, COMMENTED_ON all filtered out because endpoints are not all User)
```csv
graphName,nodeCount,relationshipCount
"test_filtering",5,0
```

## Adding New Tests

### Step 1: Create Query File
```bash
echo "CALL PROJECT('my_test', 'User', 'KNOWS')
YIELD graphName, nodeCount
RETURN graphName, nodeCount" > queries/21_my_test.mql
```

### Step 2: Create Expected Output
```bash
echo "graphName,nodeCount
\"my_test\",5" > queries/21_my_test.csv
```

### Step 3: Run Tests
```bash
./scripts/run-tests gql
```

### Step 4: Verify
Check that `21_my_test` appears in test output and passes.

## Debugging Failed Tests

### Check Server Logs
```bash
cat tests/gql/tmp/server-logs/projection_native.log
```

### Run Query Manually
```bash
# Start server
build/Debug/bin/mdb server tests/gql/tmp/dbs/projection_native --port 8080 &

# Send query
curl -H "Content-Type:application/sparql-query" \
     --data-binary @tests/gql/test_suites/projection_native/queries/01_string_variant_single.mql \
     -X POST http://localhost:8080/sparql
```

### Compare Output
```bash
diff queries/01_string_variant_single.csv <(curl ...)
```

### Debug with GDB
```bash
gdb build/Debug/bin/mdb
(gdb) run server tests/gql/tmp/dbs/projection_native
# In another terminal, send query
# (gdb) break ProjectProcedure::execute
```

## Test Coverage Goals

| Component | Target Coverage | Test Count |
|-----------|----------------|------------|
| String variant (single label/type) | 100% | 5 tests |
| List variant (multiple labels/types) | 100% | 8 tests |
| Edge endpoint filtering | 100% | 3 tests |
| YIELD variations | 100% | 4 tests |
| Query clauses | 80% | 5 tests |
| Error handling | 0% | 0 tests (future) |

**Overall Coverage**: ~85% of implemented functionality

## Future Test Additions

### Error Cases (Not Yet Implemented)
```gql
-- Invalid label name
CALL PROJECT('test', 'InvalidLabel', 'KNOWS')
-- Expected: Error message with available labels

-- Empty list
CALL PROJECT('test', [], ['KNOWS'])
-- Expected: Error "nodeProjection list cannot be empty"

-- Wrong parameter count
CALL PROJECT('test', 'User')
-- Expected: Error "PROJECT requires 3-4 parameters, got 2"

-- Wrong parameter type
CALL PROJECT('test', 123, 'KNOWS')
-- Expected: Error "nodeProjection must be STRING or LIST<STRING>"

-- Duplicate projection name
CALL PROJECT('test', 'User', 'KNOWS')
CALL PROJECT('test', 'Post', 'LIKES')  -- Second call
-- Expected: Error "Projection 'test' already exists"
```

### Advanced Features (Future)
```gql
-- WHERE clause after CALL
CALL PROJECT('test', 'User', 'KNOWS')
YIELD graphName, nodeCount
WHERE nodeCount > 0
RETURN graphName

-- MATCH + CALL combination
MATCH (u:User)
CALL PROJECT('test', 'User', 'KNOWS')
YIELD graphName
RETURN u.name, graphName

-- Configuration parameter (M1.3+)
CALL PROJECT('test', 'User', 'KNOWS', {concurrency: 4})
YIELD graphName, nodeCount
RETURN graphName, nodeCount
```

## Performance Benchmarks

Expected performance vs Cypher projection:
- Small graphs (1K nodes): 2.5x faster
- Medium graphs (100K nodes): 2.8x faster
- Large graphs (1M nodes): 2.8x faster

See `tests/gql/benchmarks/projection_performance.sh` for benchmark scripts.

## Related Documentation

- **Architecture**: `investigacion_project_2025_11_01/M1.2_Native_Projection/ARCHITECTURE_DESIGN.md`
- **Implementation**: Section 10 - Testing Strategy
- **CALL/YIELD Tests**: `tests/gql/test_suites/call_procedure/README.md`
- **Procedure API**: `src/query/procedure/procedure.h`

## Maintenance

**Created**: 2025-11-11
**Status**: Ready for implementation
**Owner**: Benito Fuentes
**Review**: After M1.2 implementation complete

---

## Quick Reference

### Procedure Signature
```gql
CALL PROJECT(
    graphName: STRING,
    nodeProjection: STRING | LIST<STRING>,
    relationshipProjection: STRING | LIST<STRING>,
    configuration: MAP? -- Optional, future
)
YIELD graphName, nodeCount, relationshipCount, projectMillis
```

### Test Naming Convention
```
<number>_<feature>_<variant>.{mql,csv}

Examples:
01_string_variant_single.mql
02_list_variant_multiple_labels.mql
05_edge_endpoint_filtering.mql
```

### CSV Format
```csv
field1,field2,field3
"value1",value2,value3
```

Note: Use `*` for wildcard matching (e.g., projectMillis timing value)
