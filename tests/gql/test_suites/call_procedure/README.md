# CALL/YIELD Integration Tests

Comprehensive integration test suite for the CALL procedure feature in GQL.

## Test Status

✅ **All 5 tests passing** (100% success rate)

| Test | Description | Status |
|------|-------------|--------|
| `basic_call.mql` | Basic CALL with YIELD and RETURN | ✅ PASS |
| `call_with_alias.mql` | CALL with YIELD field alias | ✅ PASS |
| `call_return_all.mql` | CALL with RETURN * | ✅ PASS |
| `call_with_distinct.mql` | CALL with DISTINCT clause | ✅ PASS |
| `call_with_expression.mql` | CALL with expression in RETURN | ✅ PASS |

---

## Test Coverage

### ✅ Implemented Features

1. **Basic CALL execution**
   - Execute procedures without arguments
   - YIELD single fields
   - Return procedure results

2. **YIELD with aliases**
   - Rename YIELD fields with AS clause
   - Use aliased fields in RETURN

3. **RETURN variations**
   - RETURN specific fields
   - RETURN * (all yielded fields)
   - RETURN with expressions
   - RETURN with constants alongside procedure results

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

### Expected Output
```
call_procedure: 100%
[SUMMARY] -------------- call_procedure ---------------
[SUMMARY]
[SUMMARY]           CORRECT             :   5
[SUMMARY]           ERROR               :   0
[SUMMARY]           SKIPPED             :   0
[SUMMARY]           TOTAL               :   5
```

---

## Test Files

### Test Data
**File**: `call_procedure.gql`

Simple graph with Person and Company nodes:
- 2 Person nodes (Alice, Bob)
- 1 Company node (TechCorp)
- 3 edges (KNOWS, WORKS_AT relationships)

### Test Queries

#### 1. `basic_call.mql`
```gql
CALL test_hello() YIELD message
RETURN message
```
**Expected**: One row with `"Hello, World!"`

#### 2. `call_with_alias.mql`
```gql
CALL test_hello() YIELD message AS greeting
RETURN greeting
```
**Expected**: One row with greeting = `"Hello, World!"`

#### 3. `call_return_all.mql`
```gql
CALL test_hello() YIELD message
RETURN *
```
**Expected**: All yielded fields returned

#### 4. `call_with_distinct.mql`
```gql
CALL test_hello() YIELD message
RETURN DISTINCT message
```
**Expected**: Deduplicated results

#### 5. `call_with_expression.mql`
```gql
CALL test_hello() YIELD message
RETURN message, "Extra text" AS extra
```
**Expected**: Procedure result combined with constant expression

---

## Future Tests

Advanced tests requiring additional implementation are in:
**Location**: `/test_call_yield/future_tests/`

### Pending Features

1. **WHERE clause after CALL** (`call_with_where.mql`)
   ```gql
   CALL test_hello() YIELD message
   WHERE message = "Hello, World!"
   RETURN message
   ```
   **Status**: ❌ Not implemented - requires query rewriter support

2. **MATCH + CALL** (`call_with_match.mql`)
   ```gql
   MATCH (n:Person)
   CALL test_hello() YIELD message
   RETURN n.name, message
   ```
   **Status**: ⚠️ Partially working - cartesian product logic needs fixing

See `test_call_yield/future_tests/README.md` for details.

---

## Test Procedures

### test_hello()
- **Parameters**: None
- **Yields**: `message` (STRING)
- **Returns**: Single row with `"Hello, World!"`

**Implementation**: `src/query/procedure/builtin/test_hello.h`

---

## Adding New Tests

1. Create `.mql` file with query in `queries/` directory
2. Create `.csv` file with expected results (same name as `.mql`)
3. Run tests to verify

**Example**:
```bash
# Create test files
echo "CALL test_hello() YIELD message RETURN message" > queries/my_test.mql
echo 'message\n"Hello, World!"' > queries/my_test.csv

# Run tests
cd ../../../.. && ./scripts/run-tests gql
```

---

## Integration with CI/CD

These tests are part of the main GQL test suite and run automatically via:
```bash
./scripts/run-tests all
```

Total GQL tests: **241 tests**
- call_procedure tests: **5 tests**
- Overall pass rate: **100%**

---

## Test Configuration

Tests are enabled in `tests/gql/scripts/testing/options.py`:
```python
TEST_SUITES: list[str] = [
    ...
    "call_procedure",  # Line 58
]
```

---

## Debugging Failed Tests

If tests fail:

1. **Check server logs**:
   ```bash
   cat tests/gql/tmp/server-logs/call_procedure.log
   ```

2. **Run query manually**:
   ```bash
   build/Debug/bin/mdb server tests/gql/tmp/dbs/call_procedure &
   curl -H "Content-Type:application/sparql-query" \
        --data "CALL test_hello() YIELD message RETURN message" \
        -X POST http://localhost:8080/sparql
   ```

3. **Compare output**:
   ```bash
   diff queries/basic_call.csv <actual_output>
   ```

---

## Related Documentation

- **Bug fixes**: `test_call_yield/BUG_FIX_SUMMARY.md`
- **Manual testing**: `test_call_yield/README.md`
- **Implementation**: `src/query/executor/binding_iter/gql/call_procedure.{h,cc}`
- **Test procedure**: `src/query/procedure/builtin/test_hello.h`

---

## Maintenance

**Last updated**: 2025-11-08
**Status**: ✅ All tests passing
**Coverage**: Basic CALL/YIELD functionality
**Future work**: See `test_call_yield/future_tests/README.md`
