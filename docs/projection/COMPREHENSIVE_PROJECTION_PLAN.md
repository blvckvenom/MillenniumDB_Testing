# Comprehensive Plan: Remove All Projection Limitations

**Goal**: Enable full querying capabilities in projections including labels, properties, and all query patterns.

**Date**: 2025-10-17

---

## Executive Summary

Currently, projections only support basic topology queries (nodes and edges). This plan details how to implement:
1. ✅ Edge labels in projections (`MATCH ()-[e:Friend]-()`)
2. ✅ Node labels in projections (`MATCH (u:User)`)
3. ✅ Node properties in projections (`RETURN n.name`)
4. ✅ Edge properties in projections (`RETURN e.since`)

**Good News**: ~80% of the infrastructure is already implemented! We mainly need to:
- Wire up existing label/property indexes in query execution
- Update syntax parsing to support INCLUDE clauses
- Test and document the features

---

## Current State Analysis

### ✅ What's Already Implemented

#### 1. Storage Layer (projection_storage.h/cc)
- ✅ `ProjectionStorage::Features` flags for labels/properties
- ✅ Optional B+tree indexes:
  - `node_label_index` (BPlusTree<2>): {node_id, label_id}
  - `edge_label_index` (BPlusTree<2>): {edge_id, label_id}
  - `node_key_value_index` (BPlusTree<3>): {node_id, key_id, value_id}
  - `edge_key_value_index` (BPlusTree<3>): {edge_id, key_id, value_id}
- ✅ Methods: `add_node_label()`, `add_edge_label()`
- ✅ Getters for all optional indexes

#### 2. Aggregation Layer (agg_project.h)
- ✅ `ProjectionOptions` struct with `include_labels` and `include_properties` flags
- ✅ Label extraction code (lines 270-319):
  - Scans main graph `node_label` index
  - Scans main graph `edge_label` index
  - Calls `projection_storage->add_node_label()` and `add_edge_label()`
- ✅ Property collection code (lines 136-251):
  - Detects property variables (e.g., `n.name`, `e.since`)
  - Stores in `node.properties` and `edge.properties` maps
- ✅ Feature flags passed to `ProjectionStorage` constructor

#### 3. Expression Layer (expr_agg_project.h)
- ✅ `ProjectionOptions` definition
- ✅ Integration with `ExprAggProject` expression type

### ❌ What's Missing

#### 1. Query Execution Layer (gql_model.cc)
**Status**: Currently throws exceptions for labels/properties in projections

**Methods that need updating**:
```cpp
// src/graph_models/gql/gql_model.cc

// Line 80-96: get_node_label()
// Currently: throws exception
// Needed: Return projection_ctx->node_label_index if available

// Line 98-114: get_edge_label()
// Currently: throws exception
// Needed: Return projection_ctx->edge_label_index if available

// Line 116-132: get_node_key_value()
// Currently: throws exception
// Needed: Return projection_ctx->node_key_value_index if available

// Line 134-150: get_edge_key_value()
// Currently: throws exception
// Needed: Return projection_ctx->edge_key_value_index if available
```

#### 2. Query Context Layer (projection_query_context.h)
**Status**: Needs to cache label/property indexes

**Current state** (lines 19-23):
```cpp
// Cached references to projection indexes for fast access
BPlusTree<1>* nodes_index = nullptr;
BPlusTree<3>* from_to_edge_index = nullptr;
BPlusTree<3>* to_from_edge_index = nullptr;
BPlusTree<2>* edge_direction_index = nullptr;
```

**Needs to add**:
```cpp
// Optional label indexes (may be null)
BPlusTree<2>* node_label_index = nullptr;
BPlusTree<2>* edge_label_index = nullptr;

// Optional property indexes (may be null)
BPlusTree<3>* node_key_value_index = nullptr;
BPlusTree<3>* edge_key_value_index = nullptr;
```

**Constructor needs updating** (line 25-40) to cache optional indexes.

#### 3. Parser/Syntax Layer (query_visitor.cc)
**Status**: Needs to parse INCLUDE clauses

**Required syntax support**:
```gql
-- Current (works but options ignored):
MATCH ... RETURN PROJECT("name")

-- Needed:
MATCH ... RETURN PROJECT("name", INCLUDE LABELS)
MATCH ... RETURN PROJECT("name", INCLUDE PROPERTIES)
MATCH ... RETURN PROJECT("name", INCLUDE LABELS, INCLUDE PROPERTIES)

-- Alternative syntax (more flexible):
MATCH ... RETURN PROJECT("name",
  INCLUDE NODE LABELS,
  INCLUDE EDGE LABELS,
  INCLUDE NODE PROPERTIES,
  INCLUDE EDGE PROPERTIES
)
```

**Parser changes needed**:
- Update `visitGqlProjectFunction()` to parse optional INCLUDE clauses
- Create `ProjectionOptions` with appropriate flags
- Pass options to `ExprAggProject` constructor

#### 4. Property Extraction (agg_project.h)
**Status**: Properties are collected but not written to projection indexes

**Issue** (lines 229-268):
- Properties are collected in `node.properties` and `edge.properties` maps
- But `ProjectionStorage::add_node()` and `add_edge()` don't write properties to the B+tree indexes
- Need to add property writing logic to `projection_storage.cc`

**Required changes in projection_storage.cc**:
```cpp
void ProjectionStorage::add_node(const ProjectedNode& node) {
    // ... existing code to add node ...

    // NEW: Write properties to node_key_value_index if enabled
    if (features.include_node_properties && node_key_value_index) {
        for (const auto& [key_name, value_oid] : node.properties) {
            // Convert key_name to ObjectId (lookup or create in string manager)
            ObjectId key_oid = string_to_object_id(key_name);

            // Write to B+tree: {node_id, key_id, value_id}
            node_key_value_index->insert({
                node.node_id.id,
                key_oid.id,
                value_oid.id
            });
        }
    }
}
```

#### 5. Additional Label Indexes
**Status**: ProjectionStorage has label indexes but needs auxiliary indexes

Main graph has these indexes for efficient querying:
- `label_node` (BPlusTree<2>): {label_id, node_id} - find all nodes with label
- `label_edge` (BPlusTree<2>): {label_id, edge_id} - find all edges with label

**Projections currently only have**:
- `node_label` (BPlusTree<2>): {node_id, label_id} - given node, find labels
- `edge_label` (BPlusTree<2>): {edge_id, label_id} - given edge, find labels

**Need to add**:
- `label_node` index for projections
- `label_edge` index for projections

This is needed for queries like:
```gql
USE "projection" MATCH (u:User) RETURN count(u)
```

Without `label_node` index, query optimizer can't efficiently find all User nodes.

#### 6. Additional Property Indexes
**Status**: Similar issue as labels

Main graph has:
- `key_value_node` (BPlusTree<3>): {key_id, value_id, node_id}
- `key_value_edge` (BPlusTree<3>): {key_id, value_id, edge_id}

Needed for queries like:
```gql
USE "projection" MATCH (u {age: 25}) RETURN u
```

---

## Implementation Phases

### Phase 1: Enable Label Support (HIGHEST PRIORITY)
**Estimated effort**: 2-3 hours
**Impact**: Unlocks most common projection query patterns

#### Phase 1A: Wire Up Existing Label Indexes (1 hour)

**File**: `src/graph_models/gql/projection/projection_query_context.h`

```cpp
// Add to cached indexes (line 23):
BPlusTree<2>* node_label_index = nullptr;
BPlusTree<2>* label_node_index = nullptr;  // NEW
BPlusTree<2>* edge_label_index = nullptr;
BPlusTree<2>* label_edge_index = nullptr;  // NEW

// Update constructor (line 35-40):
node_label_index = storage->get_node_label_index();
label_node_index = storage->get_label_node_index();  // NEW
edge_label_index = storage->get_edge_label_index();
label_edge_index = storage->get_label_edge_index();  // NEW
```

**File**: `src/graph_models/gql/projection/projection_storage.h`

```cpp
// Add auxiliary label indexes (after line 157):
std::unique_ptr<BPlusTree<2>> label_node_index;      // {label_id, node_id}
std::unique_ptr<BPlusTree<2>> label_edge_index;      // {label_id, edge_id}

// Add getters (after line 115):
BPlusTree<2>* get_label_node_index() { return label_node_index.get(); }
BPlusTree<2>* get_label_edge_index() { return label_edge_index.get(); }
```

**File**: `src/graph_models/gql/projection/projection_storage.cc`

```cpp
// In init() method, create auxiliary label indexes:
if (features.include_node_labels) {
    node_label_index = std::make_unique<BPlusTree<2>>(rel_dir + "/node_label");
    label_node_index = std::make_unique<BPlusTree<2>>(rel_dir + "/label_node");  // NEW
    node_label_index->create(BufferManager::get_instance());
    label_node_index->create(BufferManager::get_instance());  // NEW
}

if (features.include_edge_labels) {
    edge_label_index = std::make_unique<BPlusTree<2>>(rel_dir + "/edge_label");
    label_edge_index = std::make_unique<BPlusTree<2>>(rel_dir + "/label_edge");  // NEW
    edge_label_index->create(BufferManager::get_instance());
    label_edge_index->create(BufferManager::get_instance());  // NEW
}

// In add_node_label() method, write to both indexes:
void ProjectionStorage::add_node_label(ObjectId node_id, ObjectId label_id) {
    if (!node_label_index) return;

    node_label_index->insert({ node_id.id, label_id.id });
    label_node_index->insert({ label_id.id, node_id.id });  // NEW
}

// In add_edge_label() method, write to both indexes:
void ProjectionStorage::add_edge_label(ObjectId edge_id, ObjectId label_id) {
    if (!edge_label_index) return;

    edge_label_index->insert({ edge_id.id, label_id.id });
    label_edge_index->insert({ label_id.id, edge_id.id });  // NEW
}
```

**File**: `src/graph_models/gql/gql_model.cc`

```cpp
// Update get_node_label() (line 80-96):
BPlusTree<2>& GQLModel::get_node_label() {
    auto& ctx = get_query_ctx();
    if (ctx.is_using_projection()) {
        // Check if projection includes node labels
        if (!ctx.projection_ctx || !ctx.projection_ctx->node_label_index) {
            throw std::runtime_error(
                "Cannot use node labels with projection '" + ctx.active_projection + "'.\n\n"
                "Reason: This projection does not include node label information.\n\n"
                "Solution: Recreate projection with labels:\n"
                "  MATCH ... RETURN PROJECT(\"" + ctx.active_projection + "\", INCLUDE LABELS)"
            );
        }
        return *ctx.projection_ctx->node_label_index;
    }
    return *node_label;
}

// Update get_label_node() - similar pattern
// Update get_edge_label() - similar pattern
// Update get_label_edge() - similar pattern
```

#### Phase 1B: Add Label Syntax Parsing (1 hour)

**File**: `src/query/parser/grammar/gql/query_visitor.cc`

Find the `visitGqlProjectFunction()` method and update it to parse INCLUDE clauses:

```cpp
std::any QueryVisitor::visitGqlProjectFunction(GQLParser::GqlProjectFunctionContext* ctx) {
    // ... existing code to get projection name ...

    // NEW: Parse INCLUDE clauses
    ProjectionOptions options;

    // Check for INCLUDE LABELS
    if (ctx->INCLUDE() && ctx->LABELS()) {
        options.include_labels = true;
    }

    // Check for INCLUDE PROPERTIES
    if (ctx->INCLUDE() && ctx->PROPERTIES()) {
        options.include_properties = true;
    }

    // Create ExprAggProject with options
    auto expr = std::make_unique<ExprAggProject>(
        std::move(projection_name_expr),
        var_id,
        options  // Pass options here
    );

    return expr;
}
```

**Note**: May need to update grammar file `GQLParser.g4` if INCLUDE clause syntax doesn't exist.

#### Phase 1C: Testing (30 minutes)

```bash
# Test script: tests/projection/test_labels.sh

# Start fresh
./build/Release/bin/mdb import ./data/example/gql/posts/posts.gql ./data/test_labels
./build/Release/bin/mdb server ./data/test_labels --port 1234 &

# Create projection WITH labels
curl -X POST "http://localhost:1234/gql" \
  --data 'MATCH (u:User)-[f:Friend]-(v:User) RETURN PROJECT("friends_with_labels", INCLUDE LABELS)'

# Test 1: Query with node label (should work now!)
curl -X POST "http://localhost:1234/gql" \
  --data 'USE "friends_with_labels" MATCH (u:User) RETURN count(u)'
# Expected: 50 (number of unique users in Friend edges)

# Test 2: Query with edge label (should work!)
curl -X POST "http://localhost:1234/gql" \
  --data 'USE "friends_with_labels" MATCH ()-[e:Friend]-() RETURN count(DISTINCT e)'
# Expected: 50 (number of Friend edges)

# Test 3: Query projection WITHOUT labels (should fail with helpful error)
curl -X POST "http://localhost:1234/gql" \
  --data 'MATCH (u:User)-[f:Friend]-(v:User) RETURN PROJECT("friends_no_labels")'

curl -X POST "http://localhost:1234/gql" \
  --data 'USE "friends_no_labels" MATCH (u:User) RETURN count(u)'
# Expected: Error message telling user to recreate with INCLUDE LABELS
```

---

### Phase 2: Enable Property Support
**Estimated effort**: 3-4 hours
**Impact**: Unlocks property-based filtering and projection

#### Phase 2A: Write Properties to Indexes (1.5 hours)

**File**: `src/graph_models/gql/projection/projection_storage.cc`

```cpp
void ProjectionStorage::add_node(const ProjectedNode& node) {
    // ... existing node addition code ...

    // NEW: Write properties to node_key_value_index if enabled
    if (features.include_node_properties && node_key_value_index) {
        for (const auto& [key_name, value_oid] : node.properties) {
            // Key is already an ObjectId (string from main graph)
            // Just need to convert key_name to ObjectId
            // For now, assume key_name is already stored in string manager

            // Look up or create key_name in string manager
            ObjectId key_oid = Conversions::pack_string(key_name);

            // Write to B+tree: {node_id, key_id, value_id}
            node_key_value_index->insert({
                node.node_id.id,
                key_oid.id,
                value_oid.id
            });

            // Also write to auxiliary index {key_id, value_id, node_id}
            if (key_value_node_index) {
                key_value_node_index->insert({
                    key_oid.id,
                    value_oid.id,
                    node.node_id.id
                });
            }
        }
    }
}

// Similar for add_edge() with edge properties
```

**Issue**: Properties are currently stored as `std::unordered_map<std::string, ObjectId>` but we need to convert string keys to ObjectIds.

**Solution**: In `agg_project.h`, when collecting properties, we need to track the key ObjectId, not just the key name.

#### Phase 2B: Update Property Collection Logic (1 hour)

**File**: `src/query/executor/binding_iter/aggregation/gql/agg_project.h`

Current code (line 162-183) extracts property name as string. Need to also get the key ObjectId from the main graph's property index.

```cpp
// When we find a property variable like "n.name":
if (dot_pos != std::string::npos) {
    std::string parent_var = var_name.substr(0, dot_pos);
    std::string prop_name = var_name.substr(dot_pos + 1);

    // NEW: Look up property key ObjectId from main graph
    ObjectId prop_key_oid = Conversions::pack_string(prop_name);

    // Store with both key name and key ObjectId
    if (parent_type == GQL_OID::Type::NODE) {
        node_properties[parent_oid][prop_name] = oid;
        node_property_keys[parent_oid][prop_name] = prop_key_oid;  // NEW
    }
}
```

Then when writing properties:
```cpp
// In add_node() call:
for (const auto& [prop_name, prop_value] : props) {
    ObjectId key_oid = node_property_keys[node_id][prop_name];
    // Now we have both key_oid and prop_value as ObjectIds
    projection_storage->add_node_property(node_id, key_oid, prop_value);
}
```

#### Phase 2C: Wire Up Property Indexes in Query Execution (1 hour)

Similar to Phase 1A, but for property indexes:

**File**: `src/graph_models/gql/projection/projection_query_context.h`

```cpp
// Add property indexes (line 25):
BPlusTree<3>* node_key_value_index = nullptr;
BPlusTree<3>* key_value_node_index = nullptr;  // NEW
BPlusTree<3>* edge_key_value_index = nullptr;
BPlusTree<3>* key_value_edge_index = nullptr;  // NEW
```

**File**: `src/graph_models/gql/gql_model.cc`

```cpp
// Update get_node_key_value() to support projections
BPlusTree<3>& GQLModel::get_node_key_value() {
    auto& ctx = get_query_ctx();
    if (ctx.is_using_projection()) {
        if (!ctx.projection_ctx || !ctx.projection_ctx->node_key_value_index) {
            throw std::runtime_error(
                "Cannot access node properties with projection '" + ctx.active_projection + "'.\n\n"
                "Solution: Recreate projection with properties:\n"
                "  MATCH ... RETURN PROJECT(\"" + ctx.active_projection + "\", INCLUDE PROPERTIES)"
            );
        }
        return *ctx.projection_ctx->node_key_value_index;
    }
    return *node_key_value;
}

// Similar for get_key_value_node(), get_edge_key_value(), get_key_value_edge()
```

#### Phase 2D: Testing (30 minutes)

```bash
# Test script: tests/projection/test_properties.sh

# Create projection WITH properties
curl -X POST "http://localhost:1234/gql" \
  --data 'MATCH (u:User)-[f:Friend]-(v:User) RETURN PROJECT("friends_with_props", INCLUDE LABELS, INCLUDE PROPERTIES)'

# Test 1: Access node property
curl -X POST "http://localhost:1234/gql" \
  --data 'USE "friends_with_props" MATCH (u) RETURN u.name LIMIT 5'
# Expected: Return 5 user names

# Test 2: Filter by property value
curl -X POST "http://localhost:1234/gql" \
  --data 'USE "friends_with_props" MATCH (u {age: 25}) RETURN u.name'
# Expected: Return users with age 25

# Test 3: Access edge property
curl -X POST "http://localhost:1234/gql" \
  --data 'USE "friends_with_props" MATCH ()-[e]-() RETURN e.since LIMIT 5'
# Expected: Return 5 edge property values (if Friend edges have 'since' property)
```

---

### Phase 3: Optimize and Enhance
**Estimated effort**: 2-3 hours
**Impact**: Performance and usability improvements

#### 3A: Add Label-Specific Index Creation (1 hour)

Current approach creates indexes for ALL labels. For large graphs, this is wasteful.

**Enhanced syntax**:
```gql
-- Include only specific labels
MATCH ... RETURN PROJECT("name",
  INCLUDE NODE LABELS (User, Admin),
  INCLUDE EDGE LABELS (Friend)
)

-- Include all labels (current behavior)
MATCH ... RETURN PROJECT("name", INCLUDE LABELS)
```

**Implementation**:
- Update `ProjectionOptions` to have `vector<string> node_labels` and `vector<string> edge_labels`
- Update parser to extract label names
- Update label extraction code in `agg_project.h` to filter by requested labels

#### 3B: Add Property-Specific Index Creation (1 hour)

Similar to labels, allow selecting specific properties:

```gql
-- Include only specific properties
MATCH ... RETURN PROJECT("name",
  INCLUDE NODE PROPERTIES (name, age),
  INCLUDE EDGE PROPERTIES (since, weight)
)
```

#### 3C: Add Projection Metadata Query (30 minutes)

```gql
-- Query what's included in a projection
SHOW PROJECTION "friends_only"

-- Returns:
{
  "name": "friends_only",
  "node_count": 50,
  "edge_count": 50,
  "features": {
    "node_labels": true,
    "edge_labels": true,
    "node_properties": ["name", "age"],
    "edge_properties": []
  }
}
```

**Implementation**:
- Add metadata to projection catalog file
- Add `SHOW PROJECTION` command to parser
- Return JSON or table with projection details

#### 3D: Performance Testing (30 minutes)

Compare query performance:
- Main graph vs projection (labels enabled)
- Main graph vs projection (properties enabled)
- Projection size with/without labels/properties

Document any performance cliffs or issues.

---

### Phase 4: Additional Enhancements (Future)
**Estimated effort**: Variable
**Impact**: Nice-to-have features

#### 4A: Incremental Updates
Currently, projections are static (created once). Add ability to update:
```gql
-- Add more data to existing projection
MATCH ... RETURN PROJECT_UPDATE("existing_projection")

-- Remove data from projection
PROJECT_DELETE("projection_name", MATCH ...)
```

#### 4B: Projection from Projection
```gql
-- Create projection from another projection
USE "base_projection"
MATCH ...
RETURN PROJECT("derived_projection")
```

#### 4C: Automatic Projection Recommendations
Analyze query patterns and recommend projections:
```sql
-- Suggest projections for this workload
RECOMMEND PROJECTIONS FOR QUERIES (
  query1,
  query2,
  ...
)
```

#### 4D: Projection Materialized Views
Auto-refresh projections when main graph changes:
```gql
CREATE PROJECTION "live_friends" AS
  MATCH (u:User)-[f:Friend]-(v:User) RETURN *
  REFRESH ON UPDATE
```

---

## Testing Strategy

### Unit Tests
- `test_projection_storage_labels.cc` - Test label index creation and querying
- `test_projection_storage_properties.cc` - Test property index creation and querying
- `test_projection_features.cc` - Test feature flag combinations

### Integration Tests
- `test_projection_with_labels.sh` - End-to-end label query tests
- `test_projection_with_properties.sh` - End-to-end property query tests
- `test_projection_mixed_features.sh` - Test combinations of features

### Regression Tests
- Ensure existing projection functionality still works
- Verify backward compatibility with projections created without labels/properties

### Performance Tests
- Measure projection creation time with/without labels/properties
- Compare query performance: main graph vs projection
- Test with various graph sizes (1K, 10K, 100K, 1M nodes)

---

## Documentation Updates

### User Documentation
- Update `PROJECTION_QUERY_GUIDE.md` with label/property examples
- Add `PROJECTION_FEATURES.md` explaining INCLUDE clauses
- Update main README with new capabilities

### Developer Documentation
- Document projection storage architecture
- Add diagrams showing index relationships
- Document performance characteristics

### Example Queries
Create comprehensive examples:
```gql
-- Example 1: Social network analysis
MATCH (u:User)-[f:Friend]-(v:User)
WHERE u.age > 18 AND v.age > 18
RETURN PROJECT("adult_friends", INCLUDE LABELS, INCLUDE PROPERTIES)

-- Example 2: Citation network
MATCH (p1:Paper)-[c:Cites]->(p2:Paper)
WHERE p1.year >= 2020
RETURN PROJECT("recent_citations", INCLUDE LABELS)

-- Example 3: Recommendation subgraph
MATCH (u:User)-[r:Rated]->(m:Movie)
WHERE r.rating >= 4
RETURN PROJECT("highly_rated", INCLUDE LABELS, INCLUDE PROPERTIES (title, genre, rating))
```

---

## Risk Assessment

### High Risk
- **Backward Compatibility**: Old projections without labels/properties must still work
  - Mitigation: Check for null indexes before using, provide clear error messages

### Medium Risk
- **Performance**: Including labels/properties increases storage and query time
  - Mitigation: Make features opt-in, document trade-offs, optimize critical paths

- **Property Key Handling**: Converting property names to ObjectIds consistently
  - Mitigation: Thorough testing, use same string manager as main graph

### Low Risk
- **Parser Changes**: INCLUDE syntax might conflict with other GQL features
  - Mitigation: Follow GQL standard if defined, otherwise use clear unambiguous syntax

---

## Success Criteria

### Phase 1 (Labels) Complete When:
- ✅ `USE "proj" MATCH (u:User) RETURN count(u)` works
- ✅ `USE "proj" MATCH ()-[e:Friend]-() RETURN count(e)` works
- ✅ Projections without labels still work
- ✅ Clear error messages guide users to use INCLUDE LABELS

### Phase 2 (Properties) Complete When:
- ✅ `USE "proj" MATCH (u) RETURN u.name` works
- ✅ `USE "proj" MATCH (u {age: 25}) RETURN u` works
- ✅ `USE "proj" MATCH ()-[e]->() RETURN e.weight` works
- ✅ Property filtering and projection work correctly

### Full Implementation Complete When:
- ✅ All four original limitations are removed
- ✅ Comprehensive test suite passes
- ✅ Documentation updated
- ✅ Performance benchmarks completed
- ✅ Backward compatibility verified

---

## Timeline Estimate

Assuming dedicated focus:

- **Phase 1 (Labels)**: 2-3 hours
  - Critical path: Wire up indexes (1h) + Parse syntax (1h) + Test (0.5h)

- **Phase 2 (Properties)**: 3-4 hours
  - Critical path: Property writing (1.5h) + Collection logic (1h) + Wiring (1h) + Test (0.5h)

- **Phase 3 (Optimize)**: 2-3 hours
  - Selective inclusion (1h) + Metadata (0.5h) + Performance testing (1h)

**Total: 7-10 hours of development time**

With testing, documentation, and debugging: **2-3 days of calendar time**

---

## Next Steps

### Immediate (Phase 1A - Start Now)
1. ✅ Update `projection_query_context.h` to cache label indexes
2. ✅ Update `projection_storage.h` to add `label_node` and `label_edge` auxiliary indexes
3. ✅ Update `projection_storage.cc` to write to both label indexes
4. ✅ Update `gql_model.cc` get_node_label() and get_edge_label() to support projections
5. ✅ Rebuild and test with existing projections (should still work)

### After Phase 1A Works
6. ✅ Add INCLUDE LABELS syntax parsing
7. ✅ Test end-to-end label queries
8. ✅ Document label feature

### Then Phase 2
9. Implement property index writing
10. Test property queries
11. Document property feature

---

## Appendix: File Reference

### Files to Modify

#### High Priority (Phase 1)
- `src/graph_models/gql/projection/projection_query_context.h` - Add label index caching
- `src/graph_models/gql/projection/projection_storage.h` - Add auxiliary label indexes
- `src/graph_models/gql/projection/projection_storage.cc` - Implement label index writes
- `src/graph_models/gql/gql_model.cc` - Enable label queries in projections
- `src/query/parser/grammar/gql/query_visitor.cc` - Parse INCLUDE LABELS

#### Medium Priority (Phase 2)
- `src/query/executor/binding_iter/aggregation/gql/agg_project.h` - Fix property collection
- `src/graph_models/gql/projection/projection_storage.cc` - Write properties to indexes
- `src/graph_models/gql/gql_model.cc` - Enable property queries in projections

#### Low Priority (Phase 3+)
- Grammar file `src/query/parser/grammar/gql/GQLParser.g4` - If syntax changes needed
- Test files in `tests/projection/` - Comprehensive test coverage
- Documentation in `docs/projection/` - User and developer guides

---

**Status**: Ready to implement Phase 1A
**Next Action**: Update projection_query_context.h to cache label indexes
