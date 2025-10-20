# USE GRAPH Implementation Plan

## Overview
This document outlines the complete implementation plan for adding `USE GRAPH <projection_name>` support to MillenniumDB's GQL query engine, enabling users to query projected graphs.

## Current State

### What Works ✅
- ✅ `PROJECT('name')` function creates projections
- ✅ **NEW:** `PROJECT('name' INCLUDE LABELS)` - Store node/edge labels (Phase B1)
- ✅ **NEW:** `PROJECT('name' INCLUDE PROPERTIES)` - Store properties (Phase B1)
- ✅ Projections persist to disk in `<db>/projections/<name>/`
- ✅ **NEW:** Catalog v1.1 with feature flags (Phase A1)
- ✅ **NEW:** Optional B+tree indexes for labels/properties (Phase A2)
- ✅ **NEW:** Backward compatibility with v1.0 catalogs (Phase A3)
- ✅ Projection storage uses B+tree indexes:
  - **Required (always present):**
    - `nodes` - Node IDs
    - `from_to_edge` - Edge connectivity (from, to, edge_id)
    - `to_from_edge` - Reverse edge connectivity
    - `edge_direction` - Edge directionality (directed/undirected)
  - **Optional (only if INCLUDE specified):**
    - `node_label` - Node labels (2-column: node_id, label_id)
    - `edge_label` - Edge labels (2-column: edge_id, label_id)
    - `node_key_value` - Node properties (3-column: node_id, key_id, value_id)
    - `edge_key_value` - Edge properties (3-column: edge_id, key_id, value_id)

### What's Missing ⏳
- ⏳ `USE GRAPH` clause parser implementation (Phase 2)
- ⏳ Projection-aware query execution (Phases 3-4)
- ⏳ Query validation for missing labels/properties (Phase D - partially deferred)

## Architecture Analysis

### How Current GQL Queries Work

1. **Query Parsing** (`src/query/parser/grammar/gql/query_visitor.cc`)
   - ANTLR4 parses GQL query into parse tree
   - `QueryVisitor` converts parse tree to logical plan (`Op` tree)
   - Logical plan uses `Op` classes: `OpReturn`, `OpMatchStatement`, `OpBasicGraphPattern`, etc.

2. **Query Execution** (`src/query/executor/`)
   - Physical plan builder converts logical `Op` tree to `BindingIter` tree
   - `BindingIter` uses iterator pattern to produce result bindings
   - `IndexScan<N>` iterates over B+tree indexes from `gql_model` global object

3. **Data Access** (`src/graph_models/gql/gql_model.cc`)
   - Global `gql_model` object holds all B+tree indexes:
     ```cpp
     extern GQLModel& gql_model;

     struct GQLModel {
         std::unique_ptr<BPlusTree<2>> node_label;
         std::unique_ptr<BPlusTree<2>> label_node;
         std::unique_ptr<BPlusTree<3>> from_to_edge;
         std::unique_ptr<BPlusTree<3>> to_from_edge;
         std::unique_ptr<BPlusTree<3>> node_key_value;
         // ... more indexes
     };
     ```
   - `IndexScan` directly references these trees: `IndexScan<3>(gql_model.from_to_edge, ranges)`

### Problem
The current architecture is **hardcoded to use `gql_model`**. There's no mechanism to:
- Switch data source to a projection
- Use projection's simpler index structure
- Track which graph is active during query execution

## Implementation Plan

### Phase 1: Foundation - Query Context Enhancement

**File:** `src/query/query_context.h`

**Changes:**
```cpp
class QueryContext {
public:
    // Existing fields...
    std::string active_projection;  // ✅ Already added!

    // Add methods for projection management
    bool is_using_projection() const {
        return !active_projection.empty();
    }

    void set_active_projection(const std::string& projection_name) {
        active_projection = projection_name;
    }

    void clear_active_projection() {
        active_projection.clear();
    }
};
```

**Status:** ✅ Completed

---

### Phase 2: Parser - USE GRAPH Clause Support

**File:** `src/query/parser/grammar/gql/query_visitor.h`

**Add method:**
```cpp
class QueryVisitor : public GQLParserBaseVisitor {
public:
    // Add visitor for USE GRAPH clause
    std::any visitUseGraphClause(GQLParser::UseGraphClauseContext* ctx) override;
    std::any visitGraphReference(GQLParser::GraphReferenceContext* ctx) override;
};
```

**File:** `src/query/parser/grammar/gql/query_visitor.cc`

**Implementation:**
```cpp
std::any QueryVisitor::visitGraphExpression(GQLParser::GraphExpressionContext* ctx) {
    // Currently throws NotSupportedException
    // Change to handle graphReference case

    if (ctx->graphReference()) {
        return visitGraphReference(ctx->graphReference());
    } else if (ctx->currentGraph()) {
        // Clear any active projection - use main graph
        get_query_ctx().clear_active_projection();
        return 0;
    } else {
        // Other cases still not supported
        throw NotSupportedException("Graph expression (only graphReference supported)");
    }
}

std::any QueryVisitor::visitGraphReference(GQLParser::GraphReferenceContext* ctx) {
    std::string projection_name;

    if (ctx->delimitedGraphName()) {
        // Handle delimited identifier (e.g., "my_projection")
        auto text = ctx->delimitedGraphName()->getText();
        // Remove surrounding quotes
        projection_name = text.substr(1, text.length() - 2);
    } else if (ctx->graphName()) {
        // Handle regular identifier
        projection_name = ctx->graphName()->getText();
    } else {
        throw std::runtime_error("Invalid graph reference syntax");
    }

    // Verify projection exists
    auto& proj_manager = GQL::ProjectionManager::get_instance();
    auto projections = proj_manager.list_projections();

    bool found = false;
    for (const auto& proj : projections) {
        if (proj == projection_name) {
            found = true;
            break;
        }
    }

    if (!found) {
        throw std::runtime_error("Projection '" + projection_name + "' does not exist");
    }

    // Set active projection in QueryContext
    get_query_ctx().set_active_projection(projection_name);

    return 0;
}

std::any QueryVisitor::visitUseGraphClause(GQLParser::UseGraphClauseContext* ctx) {
    if (ctx->graphExpression()) {
        visitGraphExpression(ctx->graphExpression());
    }
    return 0;
}
```

**File:** `src/query/parser/grammar/gql/query_visitor.cc` (existing methods)

**Update statement visitors to handle USE GRAPH:**
```cpp
std::any QueryVisitor::visitPrimitiveQueryStatement(GQLParser::PrimitiveQueryStatementContext* ctx) {
    // Check for USE GRAPH clause first
    if (ctx->useGraphClause()) {
        visitUseGraphClause(ctx->useGraphClause());
    }

    // Continue with existing logic...
    visitChildren(ctx);
    // ...
}
```

**Testing:**
```bash
# Should parse without error
echo "USE GRAPH my_projection MATCH (n) RETURN n" | build/Release/bin/mdb cli data/dbs/gql/posts
```

---

### Phase 3: Projection Storage Loader

**New File:** `src/graph_models/gql/projection/projection_query_context.h`

**Purpose:** Manage projection storage during query execution

```cpp
#pragma once

#include <memory>
#include <string>
#include "graph_models/gql/projection/projection_storage.h"
#include "storage/index/bplus_tree/bplus_tree.h"

namespace GQL {

// Holds projection indexes for query execution
class ProjectionQueryContext {
public:
    std::string projection_name;
    std::unique_ptr<ProjectionStorage> storage;

    // Cached references to projection indexes
    BPlusTree<1>* nodes_index = nullptr;
    BPlusTree<3>* from_to_edge_index = nullptr;
    BPlusTree<3>* to_from_edge_index = nullptr;
    BPlusTree<2>* edge_direction_index = nullptr;

    explicit ProjectionQueryContext(const std::string& proj_name)
        : projection_name(proj_name)
    {
        auto& manager = ProjectionManager::get_instance();
        std::string proj_dir = manager.get_projection_dir(proj_name);
        std::string db_folder = manager.get_db_folder();

        storage = std::make_unique<ProjectionStorage>(proj_dir, db_folder);
        storage->open();  // Open existing projection

        // Cache index pointers for fast access
        nodes_index = storage->get_nodes_index();
        from_to_edge_index = storage->get_from_to_edge_index();
        to_from_edge_index = storage->get_to_from_edge_index();
        edge_direction_index = storage->get_edge_direction_index();
    }

    ~ProjectionQueryContext() = default;
};

} // namespace GQL
```

**File:** `src/graph_models/gql/projection/projection_storage.h`

**Add public getters:**
```cpp
class ProjectionStorage {
public:
    // ... existing methods ...

    // Getters for query execution
    BPlusTree<1>* get_nodes_index() { return nodes_index.get(); }
    BPlusTree<3>* get_from_to_edge_index() { return from_to_edge_index.get(); }
    BPlusTree<3>* get_to_from_edge_index() { return to_from_edge_index.get(); }
    BPlusTree<2>* get_edge_direction_index() { return edge_direction_index.get(); }
};
```

**File:** `src/query/query_context.h`

**Add projection query context:**
```cpp
#include "graph_models/gql/projection/projection_query_context.h"

class QueryContext {
public:
    // Existing fields...
    std::string active_projection;
    std::unique_ptr<GQL::ProjectionQueryContext> projection_ctx;

    // Initialize projection context when needed
    void load_projection(const std::string& proj_name) {
        active_projection = proj_name;
        projection_ctx = std::make_unique<GQL::ProjectionQueryContext>(proj_name);
    }

    void unload_projection() {
        active_projection.clear();
        projection_ctx.reset();
    }
};
```

**Update prepare() method:**
```cpp
void prepare(BufferManager::VersionScope& version_scope, std::chrono::seconds timeout) {
    // ... existing cleanup ...

    // Clean up projection context
    projection_ctx.reset();
    active_projection.clear();

    // ... rest of method ...
}
```

---

### Phase 4: Physical Plan Builder - Projection-Aware Index Selection

**File:** `src/query/executor/gql/physical_plan_builder.cc` (or equivalent)

**Strategy:** Modify index selection to use projection indexes when active

**Current code pattern:**
```cpp
// Current: Always uses gql_model
auto index_scan = std::make_unique<IndexScan<3>>(
    gql_model.from_to_edge,  // ← Hardcoded to main graph
    std::move(ranges)
);
```

**New code pattern:**
```cpp
BPlusTree<3>& get_from_to_edge_index() {
    auto& ctx = get_query_ctx();
    if (ctx.is_using_projection()) {
        if (!ctx.projection_ctx) {
            throw std::runtime_error("Projection context not loaded");
        }
        return *ctx.projection_ctx->from_to_edge_index;
    }
    return *gql_model.from_to_edge;
}

// Usage in physical plan builder
auto index_scan = std::make_unique<IndexScan<3>>(
    get_from_to_edge_index(),  // ← Dynamic selection
    std::move(ranges)
);
```

**Files to modify:**
1. Find all uses of `gql_model.from_to_edge`
2. Find all uses of `gql_model.to_from_edge`
3. Find all uses of `gql_model.node_label` → Skip for projections (no labels)
4. Find all uses of `gql_model.edge_label` → Skip for projections (no labels)

**Search command:**
```bash
grep -r "gql_model\." src/query/executor/binding_iter/gql/
```

**Key files likely needing updates:**
- `src/query/executor/binding_iter/gql/path_pattern.cc`
- `src/query/executor/binding_iter/gql/linear_pattern_path.cc`
- Any file building `IndexScan` operators

---

### Phase 5: Query Feature Limitations for Projections

**Projections have a simpler structure than main graphs:**
- ❌ No node labels → `node_label` index doesn't exist
- ❌ No edge labels → `edge_label` index doesn't exist
- ⚠️ Limited properties → Only if explicitly stored during projection
- ✅ Node connectivity → `nodes` index
- ✅ Edge connectivity → `from_to_edge`, `to_from_edge` indexes
- ✅ Edge direction → `edge_direction` index

**Add validation in query visitor:**

**File:** `src/query/parser/grammar/gql/query_visitor.cc`

```cpp
void validate_projection_query_features() {
    auto& ctx = get_query_ctx();
    if (!ctx.is_using_projection()) {
        return;  // Not using projection, no restrictions
    }

    // Collect restrictions from current query
    std::vector<std::string> unsupported_features;

    // These checks would be added throughout the visitor
    // For example, in visitLabelExpression:
    if (has_label_expression && ctx.is_using_projection()) {
        unsupported_features.push_back("Label filtering (:Label syntax)");
    }

    if (!unsupported_features.empty()) {
        std::stringstream ss;
        ss << "Projection queries do not support: ";
        for (size_t i = 0; i < unsupported_features.size(); i++) {
            if (i > 0) ss << ", ";
            ss << unsupported_features[i];
        }
        throw std::runtime_error(ss.str());
    }
}
```

**Supported projection query patterns:**
```sql
-- ✅ Basic pattern matching
USE GRAPH my_projection
MATCH (a)-[e]->(b)
RETURN a, e, b

-- ✅ Direction-aware queries
USE GRAPH my_projection
MATCH (a)<-[e]-(b)
RETURN a, e, b

-- ✅ Filtering by connectivity
USE GRAPH my_projection
MATCH (a)-[e1]->(b)-[e2]->(c)
RETURN a, b, c

-- ✅ Undirected edges (if projection has them)
USE GRAPH my_projection
MATCH (a)-[e]-(b)
RETURN a, b

-- ❌ Label filtering (not supported)
USE GRAPH my_projection
MATCH (a:Person)-[e]->(b)  -- ERROR: Labels not available in projections
RETURN a, b

-- ❌ Property filtering (unless implemented)
USE GRAPH my_projection
MATCH (a)-[e]->(b)
WHERE a.age > 30  -- ERROR: Properties may not be available
RETURN a, b
```

---

### Phase 6: Session Management & Graph Switching

**Requirement:** Allow switching between main graph and projections within a session

**Implementation approach:**

**Option A: Query-scoped (Recommended)**
```sql
-- Each query explicitly declares graph
USE GRAPH my_projection MATCH (n) RETURN n;
USE GRAPH my_other_projection MATCH (n) RETURN n;
MATCH (n) RETURN n;  -- Back to main graph (no USE clause)
```

**Option B: Session-scoped**
```sql
-- Set for entire session until changed
USE GRAPH my_projection;
MATCH (n) RETURN n;  -- Uses my_projection
MATCH (a)-[e]->(b) RETURN a, e, b;  -- Still uses my_projection

USE CURRENT_GRAPH;  -- Switch back to main graph
MATCH (n) RETURN n;  -- Uses main graph
```

**Recommendation:** Start with **Option A** (query-scoped) because:
- Simpler to implement
- More explicit (easier to debug)
- No session state to manage
- Matches `PROJECT()` function behavior (per-query)

**For Option A:**
- `USE GRAPH` only affects the query it's attached to
- `QueryContext.active_projection` is set during parsing, cleared in `prepare()`
- No persistent session state needed

---

### Phase 7: Error Handling & Edge Cases

**Scenarios to handle:**

1. **Projection doesn't exist:**
   ```sql
   USE GRAPH nonexistent_proj MATCH (n) RETURN n
   ```
   **Error:** `Projection 'nonexistent_proj' does not exist. Available projections: [...]`

2. **Empty projection:**
   ```sql
   USE GRAPH empty_proj MATCH (n) RETURN n
   ```
   **Result:** Empty result set (not an error)

3. **Projection deleted mid-query:**
   - Projection opened at query start
   - Held in `QueryContext.projection_ctx` during execution
   - If deleted externally, files remain open → Query completes
   - Next query will fail at parse time

4. **Concurrent projection creation/deletion:**
   - Handled by filesystem & `ProjectionManager`
   - List projections at parse time (snapshot)
   - Opening projection locks files (OS-level)

5. **Nested USE GRAPH:**
   ```sql
   USE GRAPH proj1
   USE GRAPH proj2 MATCH (n) RETURN n  -- Syntax error or override?
   ```
   **Decision:** Syntax error - only one USE clause per query

6. **USE GRAPH with PROJECT():**
   ```sql
   USE GRAPH proj1
   MATCH (n)-[e]->(m)
   RETURN PROJECT('proj2')  -- Create projection FROM a projection
   ```
   **Decision:** Allow it - proj2 is created from proj1's data

---

### Phase 8: Testing Strategy

**Test 1: Basic Projection Query**
```bash
# Setup
build/Release/bin/mdb server data/dbs/gql/posts &
curl -X POST http://localhost:1234/gql \
  --data "MATCH (a)-[b]->(c) LIMIT 10 RETURN PROJECT('test_proj')"

# Query projection
curl -X POST http://localhost:1234/gql \
  --data "USE GRAPH test_proj MATCH (n)-[e]->(m) RETURN n, e, m"

# Expected: 10 edges returned
```

**Test 2: Projection vs Main Graph**
```bash
# Query main graph
curl -X POST http://localhost:1234/gql \
  --data "MATCH (a)-[b]->(c) RETURN count(*)"
# Expected: 125 (all edges)

# Query projection (subset)
curl -X POST http://localhost:1234/gql \
  --data "USE GRAPH test_proj MATCH (a)-[b]->(c) RETURN count(*)"
# Expected: 10 (filtered edges)
```

**Test 3: Direction Handling**
```bash
# Create directed-only projection
curl -X POST http://localhost:1234/gql \
  --data "MATCH (a)-[b:DIRECTED]->(c) RETURN PROJECT('directed_only')"

# Query with different directions
curl -X POST http://localhost:1234/gql \
  --data "USE GRAPH directed_only MATCH (a)-[e]->(b) RETURN count(*)"  # Forward
curl -X POST http://localhost:1234/gql \
  --data "USE GRAPH directed_only MATCH (a)<-[e]-(b) RETURN count(*)"  # Backward
curl -X POST http://localhost:1234/gql \
  --data "USE GRAPH directed_only MATCH (a)-[e]-(b) RETURN count(*)"  # Any
```

**Test 4: Error Cases**
```bash
# Nonexistent projection
curl -X POST http://localhost:1234/gql \
  --data "USE GRAPH fake_proj MATCH (n) RETURN n"
# Expected: Error message

# Label filter on projection (should fail)
curl -X POST http://localhost:1234/gql \
  --data "USE GRAPH test_proj MATCH (n:Person) RETURN n"
# Expected: Error - labels not supported in projections
```

**Test 5: Switching Graphs**
```bash
# Query 1: Main graph
curl -X POST http://localhost:1234/gql \
  --data "MATCH (n) RETURN count(*)"

# Query 2: Projection
curl -X POST http://localhost:1234/gql \
  --data "USE GRAPH test_proj MATCH (n) RETURN count(*)"

# Query 3: Back to main graph
curl -X POST http://localhost:1234/gql \
  --data "MATCH (n) RETURN count(*)"
```

---

### Phase 9: Integration Tests

**New test file:** `tests/gql/projection_queries.test`

```yaml
# Test projection querying functionality

# Setup: Create test projection
- query: |
    MATCH (a:Paper)-[e:CITES]->(b:Paper)
    WHERE a.year > 2020
    RETURN PROJECT('recent_citations')
  expected_rows: 1

# Test 1: Query all nodes in projection
- query: |
    USE GRAPH recent_citations
    MATCH (n)
    RETURN count(n) AS node_count
  expected_rows: 1
  expected_columns: [node_count]

# Test 2: Query edges in projection
- query: |
    USE GRAPH recent_citations
    MATCH (a)-[e]->(b)
    RETURN count(e) AS edge_count
  expected_rows: 1
  expected_columns: [edge_count]

# Test 3: Pattern matching in projection
- query: |
    USE GRAPH recent_citations
    MATCH (a)-[e1]->(b)-[e2]->(c)
    RETURN a, b, c, e1, e2
    LIMIT 10
  max_rows: 10
  expected_columns: [a, b, c, e1, e2]

# Test 4: Compare projection to main graph
- query: |
    MATCH (a:Paper)-[e:CITES]->(b:Paper)
    WHERE a.year > 2020
    RETURN count(*) AS main_count
  save_result_as: main_count

- query: |
    USE GRAPH recent_citations
    MATCH (a)-[e]->(b)
    RETURN count(*) AS proj_count
  save_result_as: proj_count
  assert: main_count == proj_count
```

---

### Phase 10: Documentation

**Update:** `MillenniumDB.wiki/GQL-Projections.md`

```markdown
## Querying Projections

Once a projection has been created using the `PROJECT()` function, you can query it using the `USE GRAPH` clause:

### Syntax

    USE GRAPH <projection_name>
    MATCH <pattern>
    RETURN <expressions>

### Examples

1. **Create a projection:**

       MATCH (author:Author)-[:WROTE]->(paper:Paper)
       WHERE paper.year = 2023
       RETURN PROJECT('papers_2023')

2. **Query the projection:**

       USE GRAPH papers_2023
       MATCH (a)-[e]->(p)
       RETURN a, e, p
       LIMIT 10

3. **Multi-hop pattern:**

       USE GRAPH papers_2023
       MATCH (a)-[:WROTE]->(p1)-[:CITES]->(p2)
       RETURN a, p1, p2

### Limitations

Projections are simplified subgraphs with the following restrictions:

- **No label filtering:** Labels are not stored in projections
  - ❌ `MATCH (n:Label)` - Not supported
  - ✅ `MATCH (n)` - Supported

- **Limited properties:** Properties must be explicitly included during projection creation
  - Property filtering may not work if properties weren't projected

- **Supported features:**
  - ✅ Node and edge patterns
  - ✅ Directed/undirected edges
  - ✅ Multi-hop patterns
  - ✅ Aggregations (COUNT, SUM, etc.)
  - ✅ LIMIT, OFFSET, ORDER BY

### Performance

Querying projections is typically **faster** than the main graph for targeted subsets:
- Smaller index sizes
- Fewer edges to scan
- Optimized for specific access patterns

Use projections when:
- Repeatedly querying the same subgraph
- Working with filtered datasets
- Need fast access to specific connectivity patterns
```

---

## Implementation Checklist

### Phase 1: Foundation ✅ COMPLETED
- [x] Add `active_projection` field to `QueryContext`
- [x] Add `is_using_projection()` helper method
- [x] Reset projection context in `prepare()`

### Phase A: Optional Labels & Properties ✅ COMPLETED
- [x] **A1:** Extended ProjectionCatalog to v1.1 with feature flags
  - [x] Added `includes_node_labels`, `includes_edge_labels`, `includes_node_properties`, `includes_edge_properties`
  - [x] Backward-compatible load/save with version detection
  - [x] Enhanced print() method
- [x] **A2:** Extended ProjectionStorage with optional B+tree indexes
  - [x] Created Features struct for conditional index creation
  - [x] Added 4 optional indexes: node_label, edge_label, node_key_value, edge_key_value
  - [x] Updated init() for conditional index creation based on features
  - [x] Updated open() for auto-detection of existing optional indexes
  - [x] Added catalog loading in open() to restore statistics
  - [x] Updated save_catalog() to persist feature flags
- [x] **A3:** Comprehensive backward compatibility testing
  - [x] Created `projection_features_test.cc` with 9 test cases
  - [x] Verified v1.0 projections (topology-only) still work
  - [x] Verified v1.1 projections with all features
  - [x] Verified selective feature inclusion
  - [x] All tests passing

### Phase B: GQL Parser Extensions ✅ COMPLETED
- [x] **B1:** Grammar updates ✅ COMPLETED
  - [x] Added INCLUDE keyword to GQLLexer.g4
  - [x] Extended PROJECT function syntax with projectionOptions
  - [x] Added projectionIncludeClause rule
  - [x] Parser regenerated and compiled successfully
- [x] **B2:** Query visitor implementation ✅ COMPLETED
  - [x] Created ProjectionOptions struct in expr_agg_project.h
  - [x] Implemented visitProjectionOptions() and visitProjectionIncludeClause()
  - [x] Updated visitGqlProjectFunction() to extract options
  - [x] Options passed through ExprAggProject to AggProject

### Phase 2: Parser (USE GRAPH) ⏳ PENDING
- [ ] Implement `visitGraphExpression()` to handle `graphReference`
- [ ] Implement `visitGraphReference()` to extract projection name
- [ ] Implement `visitUseGraphClause()` to process USE GRAPH
- [ ] Update `visitPrimitiveQueryStatement()` to handle USE clause
- [ ] Add projection existence validation
- [ ] Test: Parse "USE GRAPH proj_name MATCH ..."

### Phase 3: Projection Storage Loader ⏳ PENDING
- [x] Add getters to `ProjectionStorage` for index access (done in A2)
- [ ] Create `ProjectionQueryContext` class
- [ ] Add `load_projection()` / `unload_projection()` to `QueryContext`
- [ ] Update `prepare()` to clean up projection context
- [ ] Test: Load and access projection indexes

### Phase 4: Physical Plan Builder ⏳ PENDING
- [ ] Create `get_from_to_edge_index()` helper function
- [ ] Create `get_to_from_edge_index()` helper function
- [ ] Create `get_edge_direction_index()` helper function
- [ ] Find all `gql_model.*` index accesses in executors
- [ ] Replace hardcoded accesses with dynamic selection
- [ ] Test: Execute simple pattern on projection

### Phase C: AggProject Execution ✅ COMPLETED
- [x] Updated AggProject to accept ProjectionOptions with default parameter
- [x] Extracted and stored node labels from main graph (gql_model.node_label)
- [x] Extracted and stored edge labels from main graph (gql_model.edge_label)
- [x] Properties already handled via existing property extraction logic
- [x] Optional B+tree indexes populated based on features
- [x] Added add_node_label() and add_edge_label() to ProjectionStorage
- [x] Full data flow: Query text → Parser → Visitor → ExprAggProject → expr_to_binding_expr → AggProject → ProjectionStorage

### Phase 5: Query Limitations (now Phase D) ⏳ PENDING
- [ ] Add validation for label expressions (reject if not included)
- [ ] Add validation for property filters (check availability)
- [ ] Document supported vs unsupported features
- [ ] Test: Verify error messages for unsupported features

### Phase 6: Session Management ⏳ PENDING
- [ ] Implement query-scoped USE GRAPH behavior
- [ ] Test graph switching between queries
- [ ] Test switching back to main graph (no USE clause)

### Phase 7: Error Handling ⏳ PENDING
- [ ] Handle nonexistent projection error
- [ ] Handle empty projection results
- [ ] Handle concurrent projection operations
- [ ] Test all error scenarios

### Phase 8: Testing (now Phase E) ⏳ PENDING
- [x] Create backward compatibility tests (done in A3)
- [ ] Create basic projection query test with INCLUDE
- [ ] Create projection vs main graph comparison test
- [ ] Create direction handling test
- [ ] Create error case tests for missing labels/properties
- [ ] Create graph switching test

### Phase 9: Integration Tests ⏳ PENDING
- [ ] Add `tests/gql/projection_queries.test`
- [ ] Run full test suite
- [ ] Verify no regressions in main graph queries

### Phase 10: Documentation ⏳ PENDING
- [x] Updated projection documentation (docs/projection/)
- [x] Updated FILE_ORGANIZATION.md with Phase A changes
- [ ] Update `GQL-Projections.md` with INCLUDE syntax
- [ ] Add examples to documentation
- [ ] Document limitations and supported features
- [ ] Add performance notes

---

## Estimated Effort

| Phase | Complexity | Estimated Time |
|-------|-----------|----------------|
| 1. Foundation | Low | ✅ Done |
| 2. Parser | Medium | 1-2 hours |
| 3. Storage Loader | Low | 30 min |
| 4. Physical Plan Builder | **High** | 2-3 hours |
| 5. Query Limitations | Medium | 1 hour |
| 6. Session Management | Low | 30 min |
| 7. Error Handling | Medium | 1 hour |
| 8. Testing | Medium | 1-2 hours |
| 9. Integration Tests | Low | 1 hour |
| 10. Documentation | Low | 30 min |
| **Total** | | **~10-13 hours** |

**Critical Path:** Phase 4 (Physical Plan Builder) is the most complex, requiring:
- Finding all index access points
- Ensuring correct index selection
- Handling index structure differences (2-column vs 3-column)
- Testing edge direction handling

---

## Risk Mitigation

### Risk 1: Index Structure Mismatches
**Problem:** Projections use simpler indexes than main graph
**Mitigation:**
- Start with exact matching (3-column `from_to_edge`)
- Use wrapper functions to abstract index access
- Validate index compatibility before query execution

### Risk 2: Performance Regression
**Problem:** Adding dynamic dispatch may slow main graph queries
**Mitigation:**
- Use inline functions for index selection
- Add benchmark tests comparing before/after
- Profile hot paths to ensure no overhead when not using projections

### Risk 3: Incomplete Feature Coverage
**Problem:** Users may expect full GQL features on projections
**Mitigation:**
- Clear error messages explaining limitations
- Comprehensive documentation of supported features
- Consider phased rollout (basic → advanced features)

---

## Future Enhancements

### 1. Projection Properties Support
- Store and query node/edge properties in projections
- Requires extending projection schema
- Enable `WHERE` clauses on projection queries

### 2. Projection Metadata Queries
```sql
-- List all projections
SHOW GRAPHS;

-- Get projection statistics
DESCRIBE GRAPH my_projection;
```

### 3. Projection Materialized Views
- Auto-update projections when base graph changes
- Track dependencies between graph and projections
- Incremental projection maintenance

### 4. Cross-Graph Queries
```sql
-- Join main graph with projection
FROM GRAPH my_projection MATCH (n)-[e]->(m)
MATCH (n)-[:IN_MAIN_GRAPH]->(x)
RETURN n, m, x
```

### 5. Projection Indexes Optimization
- Custom index strategies for projections
- Compressed storage for large projections
- Memory-mapped projections for fast access

---

## References

- GQL Grammar: `src/query/parser/grammar/gql/GQLParser.g4`
- Query Visitor: `src/query/parser/grammar/gql/query_visitor.cc`
- Physical Plan Builder: `src/query/executor/gql/`
- Projection Storage: `src/graph_models/gql/projection/projection_storage.{h,cc}`
- IndexScan: `src/query/executor/binding_iter/index_scan.{h,cc}`
- GQL Model: `src/graph_models/gql/gql_model.{h,cc}`

---

## Questions for Discussion

1. **Property Support:** Should we automatically include all properties when creating projections, or require explicit specification?

2. **Performance:** Should we add projection-specific query optimizations, or rely on generic optimization?

3. **Concurrency:** How should we handle projection updates while queries are running?

4. **Syntax:** Should we support `CURRENT_GRAPH` keyword to explicitly switch back to main graph, or rely on absence of USE clause?

5. **Caching:** Should projection indexes stay loaded after query completes (session cache), or reload per query?

---

**Last Updated:** 2025-10-15
**Author:** Claude Code
**Status:** Planning Complete, Ready for Implementation
