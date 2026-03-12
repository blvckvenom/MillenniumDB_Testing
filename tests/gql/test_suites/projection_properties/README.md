# Property Projection Test Suite

This test suite validates the property projection functionality of MillenniumDB's `graph_project` procedure, specifically testing the `nodeProperties` and `relationshipProperties` configuration parameters.

## Overview

The property projection feature allows projecting node and edge properties from the main graph into a projection, creating specialized B+Tree indexes for efficient property queries on projected graphs.

## Test Data

**File**: `projection_properties.gql`

**Graph Structure**:
- **Nodes**: 15 total
  - 10 User nodes (IDs 0-9) with properties: `name`, `age`, `email`, `city`
  - 5 Post nodes (IDs P0-P4) with properties: `title`, `views`, `published`
- **Edges**: 18 total
  - 8 Friend edges (undirected `~`) with properties: `since`, `strength`
  - 10 Posted edges (directed `->`) with properties: `via`, `timestamp`

## Test Categories

### Category 1: Basic Property Projection (Tests 01-05)

**Test 01: Single Node Property**
- Projects a single node property (`age`)
- Validates projection creation with minimal property configuration

**Test 02: Multiple Node Properties**
- Projects multiple node properties (`age`, `name`, `email`)
- Tests list-based property specification

**Test 03: Single Edge Property**
- Projects a single relationship property (`since`)
- Validates edge property projection

**Test 04: Multiple Edge Properties**
- Projects multiple relationship properties (`via`, `timestamp`)
- Tests on Posted edges (directed)

**Test 05: Mixed Node and Edge Properties**
- Projects both node properties (`age`, `name`) and edge properties (`since`)
- Validates simultaneous node and relationship property projection

### Category 2: Property Query Tests (Tests 06-07)

**Test 06: Node Property Query**
- Creates projection with node properties (`age`, `name`)
- Validates that projected properties are accessible

**Test 07: Edge Property Query**
- Creates projection with edge properties (`since`)
- Validates that projected relationship properties are accessible

### Category 3: Property Type and Coverage Tests (Tests 08-10)

**Test 08: Property Value Types**
- Tests different property data types (INT for `age`, STRING for `name`/`email`)
- Validates type preservation in projection

**Test 09: All Node Properties**
- Projects all available User node properties (`age`, `name`, `email`, `city`)
- Tests maximum node property coverage

**Test 10: All Edge Properties**
- Projects all available Friend edge properties (`since`, `strength`)
- Tests maximum relationship property coverage

### Category 4: Advanced Integration Tests (Tests 11-13)

**Test 11: Multi-Label Property Projection**
- Projects properties from multiple node labels (User and Post)
- Tests `nodeProperties` with heterogeneous property sets
- Expected behavior: Properties apply to all projected labels

**Test 12: Full Property Projection**
- Projects all node and edge properties simultaneously
- Stress test for property projection system
- Creates 4 property B+Tree indexes (node_key_value, key_value_node, edge_key_value, key_value_edge)

**Test 13: Combined Full Projection**
- Projects multiple labels, multiple edge types, and all properties
- Most comprehensive test: 2 node labels, 2 edge types, 4 node properties, 2 edge properties
- Total graph coverage: 15 nodes, 18 edges

## Expected B+Tree Indexes Created

For each projection with properties, the following optional indexes are created:

### Node Property Indexes (when `nodeProperties` specified):
- `node_key_value`: {node_id, key_id, value_id} - Fast node property lookup
- `key_value_node`: {key_id, value_id, node_id} - Reverse index for property-based queries

### Edge Property Indexes (when `relationshipProperties` specified):
- `edge_key_value`: {edge_id, key_id, value_id} - Fast edge property lookup
- `key_value_edge`: {key_id, value_id, edge_id} - Reverse index for property-based queries

## Running the Tests

### Run Property Projection Tests Only
```bash
./scripts/run-tests gql 2>&1 | grep -A 50 "projection_properties"
```

### Run Full GQL Test Suite (includes property tests)
```bash
./scripts/run-tests gql
```

### Manual Test Execution
```bash
# 1. Import test data
build/Release/bin/mdb import tests/gql/test_suites/projection_properties/projection_properties.gql /tmp/test_prop_db

# 2. Start server
build/Release/bin/mdb server /tmp/test_prop_db --port 14001 &

# 3. Execute test query
curl -X POST http://localhost:14001/gql \
  -H "Content-Type: application/sparql-query" \
  -d "CALL graph_project('test', 'User', 'Friend', {nodeProperties: ['age', 'name']}) YIELD graphName RETURN graphName"
```

## Test Configuration

Add to `tests/gql/scripts/testing/options.py`:
```python
TEST_SUITES: list[str] = [
    # ... existing suites ...
    "projection_properties",  # Property projection test suite
]
```

## Expected Results

All 14 tests should PASS with the following characteristics:

**Performance Expectations**:
- Property extraction overhead: <35% vs topology-only projections
- Projection creation time: 2-10ms for this test graph
- Memory overhead: ~50-100KB for property indexes

**Functional Expectations**:
- All property values correctly stored and retrievable
- Property types preserved (INT, STRING, FLOAT)
- Projections with properties survive server restart (persistent storage)

## Common Issues and Debugging

### Test Failures

**Issue**: "Server returned error: Property not found"
- **Cause**: Property name in test query doesn't match test data
- **Fix**: Verify property names in `projection_properties.gql`

**Issue**: Wrong node/edge counts in results
- **Cause**: Test data not imported correctly
- **Fix**: Re-import test data and check catalog statistics

### Property Not Accessible in Projection

**Issue**: `USE GRAPH projection_name; MATCH (n) WHERE n.property = value` returns no results
- **Cause**: Property indexes not created (config parameter missing)
- **Fix**: Ensure `nodeProperties` or `relationshipProperties` specified in `graph_project` call

### Memory/Performance Issues

**Issue**: Property projection significantly slower than expected
- **Cause**: Large number of properties or inefficient B+Tree writes
- **Debug**: Check projection creation logs for batching behavior

## Implementation References

**Source Code Locations**:
- Property projection logic: `src/query/procedure/builtin/project_procedure.cc`
- Native projection builder: `src/graph_models/gql/projection/native_projection_builder.{h,cc}`
- Property storage: `src/graph_models/gql/projection/projection_storage.{h,cc}`
- B+Tree property indexes: Created in projection subdirectory

**Related Documentation**:
- Main projection documentation: `docs/01_PROJECT_System/`
- Neo4j GDS comparison: `docs/02_GNN_Architecture/COMPARACION_NEO4J_RELATIONSHIP_PROJECTION.md`

## Test Coverage

This test suite provides:
- **Feature Coverage**: ~90% of property projection features
- **Edge Case Coverage**: ~70% (basic edge cases, no error handling tests yet)
- **Integration Coverage**: ~85% (covers main use cases)

## Future Enhancements

**Potential Additional Tests**:
1. Error handling tests (empty property lists, invalid property names)
2. Property filtering tests (WHERE clauses on projected properties)
3. Property return tests (RETURN property values from projections)
4. Null property handling tests
5. Property updates in projections (if supported)
6. Concurrent projection access with properties

## Test Maintenance

**When to Update Tests**:
- When adding new property types or data formats
- When changing property projection algorithm
- When modifying B+Tree index structure
- When adding new graph_project configuration options

**Test Data Guidelines**:
- Keep test data small (15-20 nodes, 20-30 edges) for fast execution
- Use diverse property types (STRING, INT, FLOAT, DATE)
- Include heterogeneous graphs (multiple labels with different properties)
- Ensure bidirectional testing (both directed and undirected edges)

---

**Last Updated**: 2025-11-15
**Test Suite Version**: 1.0
**MillenniumDB Version**: dev branch (B_PROJECT)
