# Projection Documentation

This directory contains documentation related to the **Projection** and **USE GRAPH** features for MillenniumDB GQL.

## Files

### Implementation Documentation
- **FINAL_IMPLEMENTATION_SUMMARY.md** - Final summary of the projection implementation
- **IMPLEMENTATION_SUMMARY.md** - Detailed implementation notes and status
- **PROJECT_IMPLEMENTATION_COMPLETE.md** - Complete implementation details
- **FILE_ORGANIZATION.md** - Current file organization and structure

### User Guides
- **PROJECTION_QUERY_GUIDE.md** - Guide for querying projections with USE GRAPH
- **PROJECT_EXAMPLES.md** - Examples of creating and using projections
- **PROJECT_FUNCTION_COMPLETE_GUIDE.md** - Complete guide to the PROJECT() function

### Debugging & Development
- **PROJECT_DEBUGGING_SUMMARY.md** - Debugging notes and issues
- **GDB_DEBUGGING_GUIDE.md** - GDB debugging instructions

## Related Resources

- **Test Scripts**: `../../tests/projection/scripts/`
- **Test Queries**: `../../tests/projection/queries/`
- **Debug Files**: `../../debug/projection/`
- **Implementation Plan**: `../../USE_GRAPH_IMPLEMENTATION_PLAN.md` (root directory)

## Feature Overview

Projections allow you to:
1. **Create subgraphs** from pattern matches using `PROJECT("name")`
2. **Query subgraphs** efficiently using `USE "projection_name"`
3. **Store topology** with **optional labels and properties** (v1.1 - ✅ implemented)

### Storage Features (Catalog v1.1)

Projections now support selective data storage:
- **Topology only** (default): Nodes, edges, and directionality
- **Optional labels**: Store node/edge labels with `INCLUDE LABELS`
- **Optional properties**: Store node/edge properties with `INCLUDE PROPERTIES`

This provides significant **storage savings** when you only need graph structure!

## Quick Start

```gql
-- Create a topology-only projection (minimal storage)
MATCH (u1:User)-[f:Friend]-(u2:User)
RETURN PROJECT("friends_network")

-- Create a projection with labels
MATCH (p:Paper)-[c:Cites]->(q:Paper)
RETURN PROJECT("citations" INCLUDE LABELS)

-- Create a projection with labels and properties
MATCH (a:Author)-[w:Wrote]->(p:Paper)
RETURN PROJECT("authorship" INCLUDE LABELS INCLUDE PROPERTIES)

-- Query the projection
USE "friends_network"
MATCH (a)-[e]-(b)
RETURN a, e, b LIMIT 10

-- Switch back to main graph
USE CURRENT_GRAPH
MATCH (u:User)
RETURN u.name LIMIT 10
```

## Implementation Status

### ✅ Phase A: Storage Layer (Completed)
- v1.1 catalog with feature flags
- Optional B+tree indexes for labels/properties
- Backward compatibility with v1.0 projections
- Comprehensive test suite (`projection_features_test`)

### ✅ Phase B: Parser & Visitor (Completed)
- `INCLUDE LABELS` and `INCLUDE PROPERTIES` syntax
- GQL grammar extended with projection options
- Query visitor extracts options from parse tree
- Options passed to execution layer

### ✅ Phase C: Label Extraction (Completed)
- AggProject queries main graph labels via `gql_model.node_label` and `edge_label`
- Labels stored in projection's optional indexes
- Full data flow: Query → Parser → Visitor → ExprAggProject → AggProject → ProjectionStorage

### ✅ Phase D: Validation (Deferred)
- Graceful behavior: projections without labels simply have no label data
- Explicit validation deferred to future enhancement

### ✅ Phase E: Testing & Documentation (Completed)
- C++ unit tests (`projection_features_test.cc`) validate all features
- End-to-end test scripts created (`test_include_labels.sh`)
- Documentation updated with complete examples
