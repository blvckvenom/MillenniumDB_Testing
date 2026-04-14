# Projection Architecture Summary

**Date**: 2025-10-17

This document provides a visual overview of the projection architecture and what needs to be implemented.

---

## Current Architecture (After USE GRAPH Bug Fix)

```
┌─────────────────────────────────────────────────────────────────┐
│                         USER QUERY                               │
│  USE "projection_name" MATCH ()-[e]-() RETURN count(DISTINCT e) │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│                    QUERY PARSER (query_visitor.cc)               │
│  ✅ visitUseGraphClause() - NOW CALLED                           │
│  ✅ visitGraphExpression() - HANDLES "quoted_name"               │
│  ✅ Calls: get_query_ctx().load_projection(name)                │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│               QUERY CONTEXT (query_context.cc)                   │
│  ✅ load_projection(name):                                       │
│     - Sets active_projection = name                             │
│     - Creates ProjectionQueryContext                            │
│  ✅ prepare() - NO LONGER CLEARS PROJECTION                      │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│      PROJECTION QUERY CONTEXT (projection_query_context.h)       │
│  ✅ Opens projection storage                                     │
│  ✅ Caches REQUIRED indexes:                                     │
│     - nodes_index (BPlusTree<1>)                                │
│     - from_to_edge_index (BPlusTree<3>)                         │
│     - to_from_edge_index (BPlusTree<3>)                         │
│     - edge_direction_index (BPlusTree<2>)                       │
│                                                                  │
│  ❌ OPTIONAL indexes NOT cached (NEEDS IMPLEMENTATION):          │
│     - node_label_index, label_node_index                        │
│     - edge_label_index, label_edge_index                        │
│     - node_key_value_index, key_value_node_index                │
│     - edge_key_value_index, key_value_edge_index                │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│           QUERY EXECUTION (gql_model.cc)                         │
│  ✅ get_from_to_edge() - Returns projection index                │
│  ✅ get_to_from_edge() - Returns projection index                │
│  ✅ get_n1_n2_edge() - Returns projection index (fallback)       │
│  ✅ get_edge_n1_n2() - Returns projection index (fallback)       │
│                                                                  │
│  ❌ get_node_label() - THROWS ERROR (NEEDS FIX)                  │
│  ❌ get_edge_label() - THROWS ERROR (NEEDS FIX)                  │
│  ❌ get_node_key_value() - THROWS ERROR (NEEDS FIX)              │
│  ❌ get_edge_key_value() - THROWS ERROR (NEEDS FIX)              │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│         PROJECTION STORAGE (projection_storage.h/cc)             │
│  ✅ REQUIRED indexes (always created):                           │
│     - nodes_index: {node_id}                                    │
│     - from_to_edge_index: {from, to, edge}                      │
│     - to_from_edge_index: {to, from, edge}                      │
│     - edge_direction_index: {edge, is_directed}                 │
│                                                                  │
│  ⚠️  OPTIONAL label indexes (EXIST but not fully wired):         │
│     - node_label_index: {node_id, label_id}                     │
│     - edge_label_index: {edge_id, label_id}                     │
│     ❌ label_node_index: {label_id, node_id} - MISSING!          │
│     ❌ label_edge_index: {label_id, edge_id} - MISSING!          │
│                                                                  │
│  ⚠️  OPTIONAL property indexes (EXIST but not fully wired):      │
│     - node_key_value_index: {node, key, value}                  │
│     - edge_key_value_index: {edge, key, value}                  │
│     ❌ key_value_node_index: {key, value, node} - MISSING!       │
│     ❌ key_value_edge_index: {key, value, edge} - MISSING!       │
└─────────────────────────────────────────────────────────────────┘
```

---

## Projection Creation Flow (Current State)

```
┌─────────────────────────────────────────────────────────────────┐
│                    USER CREATE PROJECTION                        │
│  MATCH (u:User)-[f:Friend]-(v:User)                             │
│  RETURN PROJECT("friends")                                      │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│                PARSER (query_visitor.cc)                         │
│  ✅ visitGqlProjectFunction()                                    │
│  ⚠️  ProjectionOptions (exist but syntax not parsed):            │
│     - include_labels = false (hardcoded)                        │
│     - include_properties = false (hardcoded)                    │
│  ❌ INCLUDE LABELS clause - NOT PARSED                           │
│  ❌ INCLUDE PROPERTIES clause - NOT PARSED                       │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│            AGGREGATION (agg_project.h)                           │
│  ✅ Collects nodes from MATCH results                            │
│  ✅ Collects edges from MATCH results                            │
│  ✅ Detects property variables (n.name, e.since)                 │
│                                                                  │
│  ⚠️  Label extraction code (EXISTS but only runs if include_labels): │
│     - Scans main graph node_label index                         │
│     - Scans main graph edge_label index                         │
│     - Calls projection_storage->add_node_label()                │
│     - Calls projection_storage->add_edge_label()                │
│                                                                  │
│  ⚠️  Property extraction (PARTIALLY IMPLEMENTED):                 │
│     - Collects properties in node.properties map               │
│     ❌ But doesn't write to node_key_value_index!                │
│     - Collects properties in edge.properties map               │
│     ❌ But doesn't write to edge_key_value_index!                │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│         PROJECTION STORAGE (projection_storage.cc)               │
│  ✅ add_node(node) - Writes to nodes_index                       │
│  ✅ add_edge(edge) - Writes to edge indexes                      │
│                                                                  │
│  ⚠️  add_node_label(node, label) - EXISTS but:                   │
│     ✅ Writes to node_label_index                                │
│     ❌ Doesn't write to label_node_index (doesn't exist!)        │
│                                                                  │
│  ⚠️  add_edge_label(edge, label) - EXISTS but:                   │
│     ✅ Writes to edge_label_index                                │
│     ❌ Doesn't write to label_edge_index (doesn't exist!)        │
│                                                                  │
│  ❌ add_node_property() - DOESN'T EXIST!                         │
│  ❌ add_edge_property() - DOESN'T EXIST!                         │
└─────────────────────────────────────────────────────────────────┘
```

---

## Target Architecture (After Full Implementation)

```
┌─────────────────────────────────────────────────────────────────┐
│                    USER QUERY WITH LABELS                        │
│  USE "friends" MATCH (u:User) RETURN count(u)                   │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│              QUERY EXECUTION (gql_model.cc)                      │
│  ✅ get_label_node() called by query optimizer                   │
│  ✅ Checks: ctx.is_using_projection()                            │
│  ✅ Returns: ctx.projection_ctx->label_node_index                │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│      PROJECTION QUERY CONTEXT (UPDATED)                          │
│  ✅ Caches ALL indexes (required + optional):                    │
│     - Required: nodes, from_to_edge, to_from_edge, direction    │
│     - Optional labels: node_label, label_node, edge_label, label_edge │
│     - Optional properties: node_key_value, key_value_node, etc. │
│  ✅ Sets to nullptr if not available                             │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│         PROJECTION STORAGE (COMPLETE)                            │
│  ✅ REQUIRED indexes                                             │
│  ✅ OPTIONAL label indexes (all 4):                              │
│     - node_label: {node → label} (for given node, find labels)  │
│     - label_node: {label → node} (for given label, find nodes)  │
│     - edge_label: {edge → label}                                │
│     - label_edge: {label → edge}                                │
│  ✅ OPTIONAL property indexes (all 4):                           │
│     - node_key_value: {node, key → value}                       │
│     - key_value_node: {key, value → node}                       │
│     - edge_key_value: {edge, key → value}                       │
│     - key_value_edge: {key, value → edge}                       │
└─────────────────────────────────────────────────────────────────┘
```

---

## Index Pairing Explanation

### Why Do We Need Both Directions?

Main graph has bidirectional indexes for efficient querying in both directions:

```
Example: Finding User nodes

Query: MATCH (u:User) RETURN count(u)

Execution Plan (Optimal):
1. Lookup "User" label_id in string manager
2. Scan label_node index: {User_id, *} → Get all node_ids with User label
3. Return count

Alternative (Slow):
1. Scan ALL nodes in nodes_index
2. For each node, lookup node_label index: {node_id, *}
3. Filter nodes where label = User
4. Return count

With 1M nodes and 1K User nodes:
- Optimal: Scan 1K entries (label_node index)
- Slow: Scan 1M entries (all nodes) + 1M lookups
```

### Projection Index Pairs Needed

#### Label Indexes
1. **node_label** (primary): {node_id, label_id}
   - Use: Given node, check if it has label X
   - Query: `MATCH (u:User) WHERE id(u) = 123`

2. **label_node** (auxiliary): {label_id, node_id}
   - Use: Find all nodes with label X
   - Query: `MATCH (u:User) RETURN count(u)`
   - **Currently MISSING**

3. **edge_label** (primary): {edge_id, label_id}
   - Use: Given edge, check if it has label X

4. **label_edge** (auxiliary): {label_id, edge_id}
   - Use: Find all edges with label X
   - Query: `MATCH ()-[e:Friend]-() RETURN count(e)`
   - **Currently MISSING**

#### Property Indexes
1. **node_key_value** (primary): {node_id, key_id, value_id}
   - Use: Given node, get property value
   - Query: `MATCH (u) WHERE id(u) = 123 RETURN u.name`

2. **key_value_node** (auxiliary): {key_id, value_id, node_id}
   - Use: Find nodes with property = value
   - Query: `MATCH (u {age: 25}) RETURN count(u)`
   - **Currently MISSING**

3. **edge_key_value** (primary): {edge_id, key_id, value_id}
   - Use: Given edge, get property value

4. **key_value_edge** (auxiliary): {key_id, value_id, edge_id}
   - Use: Find edges with property = value
   - **Currently MISSING**

---

## Implementation Gaps Summary

### Gap 1: Missing Auxiliary Label Indexes ⚠️  HIGH PRIORITY
**Impact**: Without these, label queries will be VERY slow

**What's missing**:
- `label_node_index` in `projection_storage.h/cc`
- `label_edge_index` in `projection_storage.h/cc`

**What needs to change**:
- Add B+tree creation in `init()`
- Write to both indexes in `add_node_label()` and `add_edge_label()`
- Cache in `ProjectionQueryContext`
- Return from `gql_model.cc` get methods

**Estimated time**: 1 hour

---

### Gap 2: Missing Auxiliary Property Indexes ⚠️  MEDIUM PRIORITY
**Impact**: Without these, property-based filtering will be slow

**What's missing**:
- `key_value_node_index` in `projection_storage.h/cc`
- `key_value_edge_index` in `projection_storage.h/cc`

**What needs to change**:
- Add B+tree creation in `init()`
- Implement `add_node_property()` and `add_edge_property()` methods
- Write to both indexes
- Cache in `ProjectionQueryContext`
- Return from `gql_model.cc` get methods

**Estimated time**: 1.5 hours

---

### Gap 3: Property Values Not Written to Indexes ❌ HIGH PRIORITY
**Impact**: Properties are collected but never persisted to indexes

**Current state**:
- `agg_project.h` collects properties in maps
- `projection_storage->add_node()` is called with properties
- But `add_node()` doesn't write properties to B+tree

**What needs to change**:
- `projection_storage.cc` `add_node()`: Loop through properties and write to indexes
- `projection_storage.cc` `add_edge()`: Loop through properties and write to indexes
- Handle property key name → ObjectId conversion

**Estimated time**: 1 hour

---

### Gap 4: Syntax Parsing for INCLUDE Clauses ⚠️  MEDIUM PRIORITY
**Impact**: Can't specify features via query syntax (hardcoded to false)

**Current state**:
- `ProjectionOptions` struct exists
- `agg_project.h` receives options
- But parser doesn't parse INCLUDE clauses

**What needs to change**:
- `GQLParser.g4`: Add grammar rules for INCLUDE clauses (if needed)
- `query_visitor.cc`: Parse INCLUDE LABELS and INCLUDE PROPERTIES
- Create `ProjectionOptions` with correct flags

**Estimated time**: 1 hour

---

### Gap 5: Query Execution Error Handling ❌ HIGH PRIORITY
**Impact**: Queries throw exceptions instead of returning projection indexes

**Current state**:
- `gql_model.cc` get_node_label(): throws exception
- `gql_model.cc` get_edge_label(): throws exception
- `gql_model.cc` get_node_key_value(): throws exception
- `gql_model.cc` get_edge_key_value(): throws exception

**What needs to change**:
- Check if projection context has the index
- Return projection index if available
- Throw helpful error if not available (with hint to use INCLUDE clauses)

**Estimated time**: 30 minutes

---

## Testing Strategy

### Unit Tests (NEW)
```cpp
// tests/projection/test_projection_labels.cc
TEST(ProjectionStorage, CreateWithLabels) {
    // Create projection with INCLUDE LABELS
    // Verify label_node and label_edge indexes created
    // Verify can query by label
}

// tests/projection/test_projection_properties.cc
TEST(ProjectionStorage, CreateWithProperties) {
    // Create projection with INCLUDE PROPERTIES
    // Verify property indexes created
    // Verify can filter by property value
}
```

### Integration Tests (NEW)
```bash
# tests/projection/scripts/test_labels_end_to_end.sh
# Full workflow: import → create projection with labels → query by label

# tests/projection/scripts/test_properties_end_to_end.sh
# Full workflow: import → create projection with properties → filter by property
```

### Regression Tests (EXISTING)
```bash
# Ensure existing tests still pass
./tests/projection/scripts/test_use_graph.sh
./tests/projection/scripts/simple_edge_test.sh
```

---

## File Reference Quick Guide

### Core Files (MUST MODIFY)
1. **projection_query_context.h** - Add label/property index caching
2. **projection_storage.h** - Add auxiliary indexes
3. **projection_storage.cc** - Implement auxiliary index writes
4. **gql_model.cc** - Enable label/property queries
5. **query_visitor.cc** - Parse INCLUDE clauses
6. **agg_project.h** - Fix property value writing

### Supporting Files (MAY NEED TO MODIFY)
7. **GQLParser.g4** - Add grammar rules (if not present)
8. **expr_agg_project.h** - Update ProjectionOptions (if needed)

### Test Files (NEW)
9. **test_projection_labels.cc** - Unit tests for labels
10. **test_projection_properties.cc** - Unit tests for properties
11. **test_labels_end_to_end.sh** - Integration test for labels
12. **test_properties_end_to_end.sh** - Integration test for properties

### Documentation Files (UPDATE)
13. **PROJECTION_QUERY_GUIDE.md** - Add label/property examples
14. **COMPREHENSIVE_PROJECTION_PLAN.md** - This plan document
15. **IMPLEMENTATION_CHECKLIST.md** - Step-by-step checklist

---

## Summary: What Works vs. What Doesn't

### ✅ What Works Now (After USE GRAPH Bug Fix)
- Basic projection creation: `MATCH ... RETURN PROJECT("name")`
- Topology queries: `USE "proj" MATCH ()-[e]-() RETURN count(e)`
- Node counting: `USE "proj" MATCH (n) RETURN count(n)`
- Edge traversal: `USE "proj" MATCH (n)-[e]-(m) RETURN n, e, m`

### ❌ What Doesn't Work (Throws Errors)
- Node label queries: `USE "proj" MATCH (u:User) RETURN count(u)`
- Edge label queries: `USE "proj" MATCH ()-[e:Friend]-() RETURN count(e)`
- Node property access: `USE "proj" MATCH (n) RETURN n.name`
- Edge property access: `USE "proj" MATCH ()-[e]-() RETURN e.since`
- Property filtering: `USE "proj" MATCH (n {age: 25}) RETURN n`

### ⚠️  What's Partially Implemented
- Label infrastructure exists (needs wiring)
- Property infrastructure exists (needs implementation)
- Options syntax exists (needs parsing)

---

**Next Step**: Start with Gap 1 - Add auxiliary label indexes (highest ROI)
