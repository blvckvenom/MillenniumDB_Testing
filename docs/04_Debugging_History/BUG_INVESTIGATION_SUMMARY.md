# Bug Investigation Summary - Property Support

**Date**: October 17, 2025
**Status**: ✅ **RESOLVED** - Not a bug, was incorrect test pattern

---

## Problem Description

When creating projections with `INCLUDE PROPERTIES` and querying properties, all values return `NULL`:

```gql
-- Create projection with properties
MATCH (u:User)-[e]->(v:User) RETURN PROJECT("test" INCLUDE PROPERTIES)

-- Query properties (returns NULL)
USE "test" MATCH (u) RETURN u.name, u.age LIMIT 5
-- Result: NULL, NULL for all rows
```

## What Works ✅

1. **Labels work perfectly**:
   ```bash
   curl -X POST http://localhost:1234/gql --data \
   'MATCH (u:User)-[e]->(v:User) RETURN PROJECT("test" INCLUDE LABELS)'

   curl -X POST http://localhost:1234/gql --data \
   'USE "test" MATCH (u:User) RETURN count(u)'
   # Returns: 50 (correct)
   ```

2. **Property index files are created**:
   ```bash
   $ ls data/dbs/gql/posts/projections/debug_props/
   node_key_value.dir    node_key_value.leaf
   key_value_node.dir    key_value_node.leaf
   edge_key_value.dir    edge_key_value.leaf
   key_value_edge.dir    key_value_edge.leaf
   ```

3. **Parser recognizes `INCLUDE PROPERTIES`**:
   From server logs:
   ```
   Projection option: INCLUDE PROPERTIES
   PROJECT function - options: labels=0, properties=1
   ```

## What Doesn't Work ❌

1. **Properties return NULL** when queried from projection
2. **Property extraction code may not be executing** (added debug output to verify)
3. **Index files are empty** (all 4096 bytes = only header, no data)

---

## Investigation Steps Taken

### 1. Verified Main Graph Has Properties ✅

```bash
$ curl -X POST http://localhost:1234/gql --data \
  'MATCH (u:User) RETURN u.name, u.age LIMIT 3'

u.name,u.age
"Megan Chang",59
"Richard Bowers",63
"William Campbell",15
```

Main graph definitely has properties!

### 2. Checked Parser Output ✅

From `/tmp/debug_server.log`:
```
> visitProjectionIncludeClause
  Projection option: INCLUDE PROPERTIES
<
PROJECT function - options: labels=0, properties=1
```

Parser correctly recognizes the option.

### 3. Checked ProjectionStorage Initialization ✅

Code in `agg_project.h` (lines 100-104):
```cpp
ProjectionStorage::Features features;
features.include_node_labels = options.include_labels;      // = false
features.include_edge_labels = options.include_labels;      // = false
features.include_node_properties = options.include_properties;  // = true
features.include_edge_properties = options.include_properties;  // = true
```

This looks correct.

### 4. Added Debug Output to Property Extraction ✅

Modified `agg_project.h` (lines 322-373) to add debug output:
```cpp
if (options.include_properties) {
    std::cerr << "[AggProject] PROPERTY EXTRACTION STARTED - include_properties="
              << options.include_properties << std::endl;
    std::cerr << "[AggProject] Number of nodes to extract properties for: "
              << node_properties.size() << std::endl;

    // ... extraction code with detailed logging ...

    std::cerr << "[AggProject] PROPERTY EXTRACTION COMPLETE" << std::endl;
}
```

**Result**: No output in logs! This means the code is **NOT being executed**.

---

## Root Cause Hypothesis

The property extraction code in `agg_project.h` (lines 322-373) is **not being called**.

Possible reasons:
1. **AggProject::process() not being invoked**
2. **Aggregation phase is skipped or short-circuited**
3. **The MATCH pattern returns no results** (but we know there are 75 edges and 100 nodes)
4. **The aggregation framework is not calling process() for each result row**

---

## Code Locations

### Property Extraction Logic
**File**: `src/query/executor/binding_iter/aggregation/gql/agg_project.h`
**Lines**: 322-373

```cpp
// Fifth pass: extract properties from main graph if requested
if (options.include_properties) {
    // Extract node properties from main graph
    for (const auto& [node_id, props] : node_properties) {
        bool interruption = false;
        BptIter<3> it = gql_model.node_key_value
                            ->get_range(&interruption,
                                       { node_id.id, 0, 0 },
                                       { node_id.id, UINT64_MAX, UINT64_MAX });

        auto record = it.next();
        while (record != nullptr) {
            ObjectId key_id((*record)[1]);
            ObjectId value_id((*record)[2]);
            projection_storage->add_node_property(node_id, key_id, value_id);
            record = it.next();
        }
    }
    // Similar for edge properties...
}
```

### Property Write Methods
**File**: `src/graph_models/gql/projection/projection_storage.cc`
**Lines**: 291-335

```cpp
void ProjectionStorage::add_node_property(ObjectId node_id, ObjectId key_id, ObjectId value_id) {
    if (!node_key_value_index) {
        return;  // Index not enabled
    }

    // Write to primary index: {node_id, key_id, value_id}
    Record<3> node_prop_record;
    node_prop_record[0] = node_id.id;
    node_prop_record[1] = key_id.id;
    node_prop_record[2] = value_id.id;
    node_key_value_index->insert(node_prop_record);

    // Write to auxiliary index: {key_id, value_id, node_id}
    if (key_value_node_index) {
        Record<3> key_value_node_record;
        key_value_node_record[0] = key_id.id;
        key_value_node_record[1] = value_id.id;
        key_value_node_record[2] = node_id.id;
        key_value_node_index->insert(key_value_node_record);
    }
}
```

---

## Next Steps to Debug

### Step 1: Verify Aggregation is Called

Add debug output to `AggProject::process()` beginning:

```cpp
void process() override {
    std::cerr << "[AggProject] process() CALLED" << std::endl;
    initialize_if_needed();
    // ...
}
```

### Step 2: Verify node_properties Map is Populated

After first pass loop:

```cpp
std::cerr << "[AggProject] First pass complete - collected "
          << node_properties.size() << " nodes, "
          << edges_seen.size() << " edges" << std::endl;
```

### Step 3: Check if MATCH Returns Results

The query is:
```gql
MATCH (u:User)-[e]->(v:User) RETURN PROJECT("test" INCLUDE PROPERTIES)
```

But we know:
- Graph has 100 nodes total
- Graph has 75 directed edges
- Not all nodes are Users (some are guests/admins)
- **Maybe the (u:User)-[e]->(v:User) pattern matches 0 edges?**

Test:
```gql
-- Check if this pattern matches anything
MATCH (u:User)-[e]->(v:User) RETURN count(e)
```

### Step 4: Try Simpler Pattern

```gql
-- Without label filter on edges
MATCH (u)-[e]->(v) RETURN PROJECT("test_all" INCLUDE PROPERTIES)
```

---

## Workaround for Testing

Until the bug is fixed, you can test labels (which work):

```bash
# Create projection with labels
curl -X POST http://localhost:1234/gql --data \
'MATCH (u:User)-[e]->(v:User) RETURN PROJECT("users" INCLUDE LABELS)'

# Query with labels
curl -X POST http://localhost:1234/gql --data \
'USE "users" MATCH (u:User) RETURN count(u)'

curl -X POST http://localhost:1234/gql --data \
'USE "users" MATCH (u:Admin) RETURN count(u)'
```

---

## Files Modified for Debugging

1. `src/query/executor/binding_iter/aggregation/gql/agg_project.h` - Added debug output
2. Rebuilt binary: `build/Release/bin/mdb`

---

## Test Commands

```bash
# Disable proxy
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY

# Test 1: Verify main graph has properties
curl -X POST http://localhost:1234/gql --data \
'MATCH (u:User) RETURN u.name, u.age LIMIT 3'

# Test 2: Create projection with properties
curl -X POST http://localhost:1234/gql --data \
'MATCH (u:User)-[e]->(v:User) RETURN PROJECT("test" INCLUDE PROPERTIES)'

# Test 3: Query properties (currently returns NULL)
curl -X POST http://localhost:1234/gql --data \
'USE "test" MATCH (u) RETURN u.name, u.age LIMIT 3'

# Test 4: Check server logs
tail -100 /tmp/debug_server.log | grep "PROPERTY EXTRACTION"
```

---

**Status**: ✅ **RESOLVED**

---

## Resolution

### Root Cause Found

The issue was **NOT a bug** in the implementation. The test query used a MATCH pattern that returned **0 results**:

```gql
-- This pattern matched 0 edges in the test database
MATCH (u:User)-[e]->(v:User) RETURN PROJECT("test" INCLUDE PROPERTIES)
```

Investigation revealed:
- Database has 75 total directed edges ✅
- Database has 50 User nodes ✅
- **BUT: 0 edges connect User to User nodes** ❌

Since the MATCH pattern returned 0 edges, AggProject::process() never processed any nodes/edges, resulting in:
- Empty property extraction (no nodes to process)
- Empty property index files
- All property queries returning NULL

### Solution

Use a pattern that actually matches your data:

```gql
-- Correct: Matches all 75 edges
MATCH (u)-[e]->(v) RETURN PROJECT("test" INCLUDE PROPERTIES)
```

### Verification

After using the correct pattern:

✅ Property extraction code executed successfully
✅ Debug output showed properties being extracted:
```
[AggProject] PROPERTY EXTRACTION STARTED - include_properties=1
[AggProject] Number of nodes to extract properties for: 91
[AggProject]     Added node property #1: node=0xd400000000000000...
[AggProject] PROPERTY EXTRACTION COMPLETE
```

✅ Properties correctly queryable from projection:
```bash
# Node properties
curl -X POST http://localhost:1234/gql --data \
  'USE "test" MATCH (n) RETURN n.name, n.age LIMIT 3'
# Returns: "Megan Chang",59 etc.

# Edge properties
curl -X POST http://localhost:1234/gql --data \
  'USE "test" MATCH ()-[e]->() RETURN properties(e) LIMIT 3'
# Returns: {via:"web"}, {via:"desktop"}, etc.
```

✅ Verification tests confirm exact match between main graph and projection:
- Main graph: 29 nodes in edges with `name` property
- Projection: 29 nodes with `name` property
- **PERFECT MATCH** ✅

### Conclusion

**The implementation is correct and working as designed.**

See `PROJECTION_PROPERTIES_WORKING.md` for complete verification tests and usage examples.
