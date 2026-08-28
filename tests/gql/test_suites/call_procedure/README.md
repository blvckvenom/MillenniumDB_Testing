# CALL/YIELD Integration Tests

Comprehensive integration test suite for the CALL procedure feature in GQL.

## Test Status

| Test | Description | Status |
|------|-------------|--------|
| `basic_call.mql` | Basic CALL with YIELD and RETURN | PASS |
| `call_with_alias.mql` | CALL with YIELD field alias | PASS |
| `call_return_all.mql` | CALL with RETURN * | PASS |
| `call_with_distinct.mql` | CALL with DISTINCT clause | PASS |
| `call_with_expression.mql` | CALL with expression in RETURN | PASS |

---

## Test Coverage

### Implemented Features

1. **Basic CALL execution**
   - Execute procedures with arguments
   - YIELD single fields
   - Return procedure results

2. **YIELD with aliases**
   - Rename YIELD fields with AS clause
   - Use aliased fields in RETURN

3. **RETURN variations**
   - RETURN specific fields
   - RETURN * (all yielded fields)
   - RETURN with expressions

4. **DISTINCT**
   - Deduplicate procedure results

---

## Running the Tests

### Run All GQL Tests
```bash
./scripts/run-tests gql
```

### Run Only CALL/YIELD Tests
```bash
cd tests/gql
python3 scripts/test.py --executable ../../build/Debug/bin/mdb | grep call_procedure
```

---

## Test Files

### Test Queries

All tests use `graph_exists('nonexistent_projection')` which returns `exists: false` (a boolean).

#### 1. `basic_call.mql`
```gql
CALL graph_exists('nonexistent_projection') YIELD exists
RETURN exists
```
**Expected**: One row with `exists = false`

#### 2. `call_with_alias.mql`
```gql
CALL graph_exists('nonexistent_projection') YIELD exists AS projection_found
RETURN projection_found
```
**Expected**: One row with `projection_found = false`

#### 3. `call_return_all.mql`
```gql
CALL graph_exists('nonexistent_projection') YIELD exists
RETURN *
```
**Expected**: All yielded fields returned (`exists = false`)

#### 4. `call_with_distinct.mql`
```gql
CALL graph_exists('nonexistent_projection') YIELD exists
RETURN DISTINCT exists
```
**Expected**: Deduplicated results (`exists = false`)

#### 5. `call_with_expression.mql`
```gql
CALL graph_exists('nonexistent_projection') YIELD exists
RETURN NOT exists AS does_not_exist
```
**Expected**: Expression result (`does_not_exist = true`)

---

## Test Procedure

### graph_exists(projectionName)
- **Parameters**: `projectionName` (STRING)
- **Yields**: `exists` (BOOL)
- **Returns**: Single row with `true` if projection exists, `false` otherwise

**Implementation**: `src/query/procedure/builtin/graph_exists_procedure.h`

---

## Adding New Tests

1. Create `.mql` file with query in `queries/` directory
2. Create `.csv` file with expected results (same name as `.mql`)
3. Run tests to verify

---

## Integration with CI/CD

These tests are part of the main GQL test suite and run automatically via:
```bash
./scripts/run-tests all
```

---

## Test Configuration

Tests are enabled in `tests/gql/scripts/testing/options.py`:
```python
TEST_SUITES: list[str] = [
    ...
    "call_procedure",
]
```
