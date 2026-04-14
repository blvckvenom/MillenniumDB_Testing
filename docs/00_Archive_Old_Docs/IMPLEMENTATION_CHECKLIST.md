# Projection Feature Implementation Checklist

**Goal**: Remove all 4 projection limitations (labels and properties support)

**Total Estimated Time**: 7-10 hours of development

---

## Phase 1: Label Support (PRIORITY 1)

### Step 1.1: Add Label Index Caching (15 min)
- [ ] File: `src/graph_models/gql/projection/projection_query_context.h`
  - [ ] Add `BPlusTree<2>* node_label_index = nullptr;` (line ~24)
  - [ ] Add `BPlusTree<2>* label_node_index = nullptr;` (line ~25)
  - [ ] Add `BPlusTree<2>* edge_label_index = nullptr;` (line ~26)
  - [ ] Add `BPlusTree<2>* label_edge_index = nullptr;` (line ~27)
  - [ ] Update constructor to cache these indexes (line ~36-39)

### Step 1.2: Add Auxiliary Label Indexes to Storage (30 min)
- [ ] File: `src/graph_models/gql/projection/projection_storage.h`
  - [ ] Add `std::unique_ptr<BPlusTree<2>> label_node_index;` (after line 157)
  - [ ] Add `std::unique_ptr<BPlusTree<2>> label_edge_index;` (after line 158)
  - [ ] Add getters `get_label_node_index()` and `get_label_edge_index()` (after line 115)
  - [ ] Add const getters (after line 127)

- [ ] File: `src/graph_models/gql/projection/projection_storage.cc`
  - [ ] In `init()`: Create `label_node` and `label_edge` B+trees if labels enabled
  - [ ] In `open()`: Open `label_node` and `label_edge` B+trees if they exist
  - [ ] In `add_node_label()`: Write to both `node_label` and `label_node` indexes
  - [ ] In `add_edge_label()`: Write to both `edge_label` and `label_edge` indexes

### Step 1.3: Enable Label Queries in Projections (30 min)
- [ ] File: `src/graph_models/gql/gql_model.cc`
  - [ ] Update `get_node_label()` (line 80-96): Return projection label index if available
  - [ ] Update `get_label_node()`: Return projection label index if available
  - [ ] Update `get_edge_label()` (line 98-114): Return projection label index if available
  - [ ] Update `get_label_edge()`: Return projection label index if available
  - [ ] Keep helpful error messages if indexes not available

### Step 1.4: Add INCLUDE LABELS Syntax (45 min)
- [ ] File: `src/query/parser/grammar/gql/GQLParser.g4` (if needed)
  - [ ] Check if INCLUDE and LABELS keywords exist
  - [ ] Add grammar rule for PROJECT function with INCLUDE clauses
  - [ ] Example: `PROJECT '(' expression (',' includeClause)* ')'`

- [ ] File: `src/query/parser/grammar/gql/query_visitor.cc`
  - [ ] Find `visitGqlProjectFunction()` method
  - [ ] Parse INCLUDE LABELS clause
  - [ ] Create `ProjectionOptions` with `include_labels = true`
  - [ ] Pass options to `ExprAggProject` constructor

### Step 1.5: Test Label Support (30 min)
- [ ] Rebuild: `cmake --build build/Release --target mdb -j 4`
- [ ] Test backward compatibility: Old projections without labels still work
- [ ] Test label creation:
  ```gql
  MATCH (u:User)-[f:Friend]-(v:User)
  RETURN PROJECT("test_labels", INCLUDE LABELS)
  ```
- [ ] Test node label query:
  ```gql
  USE "test_labels" MATCH (u:User) RETURN count(u)
  ```
- [ ] Test edge label query:
  ```gql
  USE "test_labels" MATCH ()-[e:Friend]-() RETURN count(e)
  ```
- [ ] Test error message for projection without labels

### Step 1.6: Document Label Feature (15 min)
- [ ] Update `PROJECTION_QUERY_GUIDE.md` with label examples
- [ ] Add troubleshooting section for label-related errors

---

## Phase 2: Property Support (PRIORITY 2)

### Step 2.1: Add Property Index Caching (15 min)
- [ ] File: `src/graph_models/gql/projection/projection_query_context.h`
  - [ ] Add `BPlusTree<3>* node_key_value_index = nullptr;` (line ~28)
  - [ ] Add `BPlusTree<3>* key_value_node_index = nullptr;` (line ~29)
  - [ ] Add `BPlusTree<3>* edge_key_value_index = nullptr;` (line ~30)
  - [ ] Add `BPlusTree<3>* key_value_edge_index = nullptr;` (line ~31)
  - [ ] Update constructor to cache these indexes

### Step 2.2: Add Auxiliary Property Indexes to Storage (30 min)
- [ ] File: `src/graph_models/gql/projection/projection_storage.h`
  - [ ] Add `std::unique_ptr<BPlusTree<3>> key_value_node_index;`
  - [ ] Add `std::unique_ptr<BPlusTree<3>> key_value_edge_index;`
  - [ ] Add getters

- [ ] File: `src/graph_models/gql/projection/projection_storage.cc`
  - [ ] In `init()`: Create property B+trees if properties enabled
  - [ ] In `open()`: Open property B+trees if they exist

### Step 2.3: Write Properties to Indexes (60 min)
- [ ] File: `src/graph_models/gql/projection/projection_storage.cc`
  - [ ] Implement `add_node_property(ObjectId node_id, ObjectId key_id, ObjectId value_id)`
  - [ ] Implement `add_edge_property(ObjectId edge_id, ObjectId key_id, ObjectId value_id)`
  - [ ] Write to both primary and auxiliary indexes

- [ ] File: `src/query/executor/binding_iter/aggregation/gql/agg_project.h`
  - [ ] Update property collection logic (line 162-183)
  - [ ] Track property key ObjectIds (not just names)
  - [ ] Call `add_node_property()` and `add_edge_property()` in process()

### Step 2.4: Enable Property Queries in Projections (30 min)
- [ ] File: `src/graph_models/gql/gql_model.cc`
  - [ ] Update `get_node_key_value()`: Return projection property index if available
  - [ ] Update `get_key_value_node()`: Return projection property index if available
  - [ ] Update `get_edge_key_value()`: Return projection property index if available
  - [ ] Update `get_key_value_edge()`: Return projection property index if available

### Step 2.5: Add INCLUDE PROPERTIES Syntax (30 min)
- [ ] File: `src/query/parser/grammar/gql/GQLParser.g4` (if needed)
  - [ ] Add PROPERTIES keyword if missing
  - [ ] Update PROJECT grammar rule

- [ ] File: `src/query/parser/grammar/gql/query_visitor.cc`
  - [ ] Parse INCLUDE PROPERTIES clause
  - [ ] Set `ProjectionOptions.include_properties = true`

### Step 2.6: Test Property Support (45 min)
- [ ] Test property creation:
  ```gql
  MATCH (u:User)-[f:Friend]-(v:User)
  RETURN PROJECT("test_props", INCLUDE LABELS, INCLUDE PROPERTIES)
  ```
- [ ] Test node property access:
  ```gql
  USE "test_props" MATCH (u) RETURN u.name LIMIT 5
  ```
- [ ] Test property filtering:
  ```gql
  USE "test_props" MATCH (u {age: 25}) RETURN u.name
  ```
- [ ] Test edge property access:
  ```gql
  USE "test_props" MATCH ()-[e]-() RETURN e.since LIMIT 5
  ```

### Step 2.7: Document Property Feature (15 min)
- [ ] Update `PROJECTION_QUERY_GUIDE.md` with property examples
- [ ] Document property-related error messages

---

## Phase 3: Testing & Documentation (PRIORITY 3)

### Step 3.1: Comprehensive Testing (90 min)
- [ ] Create `tests/projection/test_labels_comprehensive.sh`
- [ ] Create `tests/projection/test_properties_comprehensive.sh`
- [ ] Create `tests/projection/test_mixed_features.sh`
- [ ] Test all combinations:
  - [ ] No features (backward compatibility)
  - [ ] Labels only
  - [ ] Properties only
  - [ ] Labels + Properties
- [ ] Test error cases:
  - [ ] Query labels without INCLUDE LABELS
  - [ ] Query properties without INCLUDE PROPERTIES
  - [ ] Invalid projection names
  - [ ] Missing projections

### Step 3.2: Performance Testing (60 min)
- [ ] Benchmark projection creation time:
  - [ ] Basic projection (no features)
  - [ ] With labels
  - [ ] With properties
  - [ ] With both
- [ ] Benchmark query performance:
  - [ ] Main graph vs projection (labels)
  - [ ] Main graph vs projection (properties)
- [ ] Measure storage overhead:
  - [ ] Projection size with/without features
- [ ] Document results in `docs/projection/PERFORMANCE.md`

### Step 3.3: Update Documentation (60 min)
- [ ] Update `PROJECTION_QUERY_GUIDE.md`:
  - [ ] Add "Features" section explaining INCLUDE clauses
  - [ ] Add label query examples
  - [ ] Add property query examples
  - [ ] Add mixed feature examples
- [ ] Update `FILE_ORGANIZATION_NEW.md`:
  - [ ] Document new indexes
  - [ ] Update architecture section
- [ ] Create `docs/projection/FEATURE_MATRIX.md`:
  - [ ] Table showing supported features
  - [ ] Comparison with main graph capabilities
  - [ ] Performance characteristics
- [ ] Update main `README.md` or `CLAUDE.md`:
  - [ ] Highlight new projection capabilities
  - [ ] Add quick examples

### Step 3.4: Code Cleanup (30 min)
- [ ] Remove debug logging added during USE GRAPH bug fix:
  - [ ] `src/graph_models/gql/gql_model.cc` (cerr statements)
  - [ ] `src/query/query_context.cc` (cerr statements)
  - [ ] `src/query/parser/grammar/gql/query_visitor.cc` (cerr statements)
  - [ ] `src/query/executor/binding_iter/aggregation/gql/agg_project.h` (cerr statements)
- [ ] Remove unnecessary includes
- [ ] Add final comments to complex logic

---

## Phase 4: Optional Enhancements (FUTURE)

### Step 4.1: Selective Label/Property Inclusion
- [ ] Extend syntax:
  ```gql
  PROJECT("name",
    INCLUDE NODE LABELS (User, Admin),
    INCLUDE EDGE LABELS (Friend),
    INCLUDE NODE PROPERTIES (name, age)
  )
  ```
- [ ] Update `ProjectionOptions` to have label/property lists
- [ ] Filter during extraction

### Step 4.2: Projection Metadata Query
- [ ] Implement `SHOW PROJECTION "name"` command
- [ ] Return JSON with projection details
- [ ] Show feature flags, counts, sizes

### Step 4.3: Projection Management
- [ ] Implement `DROP PROJECTION "name"`
- [ ] Implement `RENAME PROJECTION "old" TO "new"`
- [ ] Implement `COPY PROJECTION "source" TO "dest"`

---

## Quick Start (If Starting Fresh)

### Fastest Path to Working Labels (2 hours)
1. ✅ Phase 1.1: Add label index caching (15 min)
2. ✅ Phase 1.2: Add auxiliary label indexes (30 min)
3. ✅ Phase 1.3: Enable label queries (30 min)
4. ✅ Phase 1.5: Test (skip syntax parsing for now, test manually) (30 min)
5. ✅ Phase 1.4: Add syntax parsing (come back to this) (45 min)

### Fastest Path to Working Properties (3 hours)
1. ✅ Complete label support first (dependencies)
2. ✅ Phase 2.1: Add property index caching (15 min)
3. ✅ Phase 2.2: Add auxiliary property indexes (30 min)
4. ✅ Phase 2.3: Write properties to indexes (60 min)
5. ✅ Phase 2.4: Enable property queries (30 min)
6. ✅ Phase 2.6: Test (skip syntax parsing for now) (45 min)
7. ✅ Phase 2.5: Add syntax parsing (come back to this) (30 min)

---

## Progress Tracking

### Current Status
- ✅ **USE GRAPH bug fixed** (projection context loading works)
- ✅ **Basic topology queries work** (nodes and edges without labels/properties)
- ⏳ **Label support** (infrastructure exists, needs wiring)
- ⏳ **Property support** (infrastructure exists, needs implementation)

### Completion Criteria
- [ ] All 4 original limitations removed
- [ ] Backward compatibility maintained
- [ ] Comprehensive test suite passes
- [ ] Documentation updated
- [ ] Performance benchmarks completed

---

## Need Help?

### If Tests Fail
1. Check server log: `tail -f /tmp/server.log`
2. Look for error messages about missing indexes
3. Verify projection was created with correct features: `./build/Release/bin/mdb list-projections <db_folder>`
4. Use inspection tool: `./build/Release/tests/projection_inspect <db_folder> <projection_name>`

### If Syntax Parsing Fails
1. Check grammar file: `src/query/parser/grammar/gql/GQLParser.g4`
2. Regenerate parser: `./src/query/parser/grammar/gql/generate.sh`
3. Rebuild: `cmake --build build/Release --target mdb -j 4`
4. Check parser visitor logs (if enabled)

### If Query Execution Fails
1. Enable debug logging in `gql_model.cc` (temporary)
2. Check which index method is being called
3. Verify projection context has correct indexes loaded
4. Check for null pointer errors

---

**Ready to Start**: Phase 1.1 - Add Label Index Caching
**Next File**: `src/graph_models/gql/projection/projection_query_context.h`
