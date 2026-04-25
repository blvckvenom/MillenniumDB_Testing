# Projection Tests

Test suite for the **Projection** and **USE GRAPH** features.

## Directory Structure

```
tests/projection/
├── scripts/          # Shell test scripts (.sh)
├── queries/          # Test query files (.gql)
└── README.md         # This file

src/tests/
└── projection_features_test.cc  # Unit test for optional labels/properties
```

## Unit Tests (C++)

### projection_features_test
**Location**: `src/tests/projection_features_test.cc`
**Executable**: `build/Release/tests/projection_features_test`

**Purpose**: Tests backward compatibility and optional features (Phase A3)

**Test Cases**:
1. ✅ Creating v1.0 projection (topology only)
2. ✅ Verifying v1.0 catalog defaults
3. ✅ Opening v1.0 projection (backward compatibility)
4. ✅ Creating v1.1 projection with all optional features
5. ✅ Verifying v1.1 catalog feature flags
6. ✅ Opening v1.1 projection (auto-detection)
7. ✅ Creating v1.1 projection with selective features
8. ✅ Verifying selective features
9. ✅ Verifying physical file existence

**Run unit test**:
```bash
./build/Release/tests/projection_features_test
```

**Run via ctest**:
```bash
cd build/Release
ctest -R projection_features_test -V
```

## Shell Test Scripts

### End-to-End Tests
- **test_use_graph.sh** - Comprehensive end-to-end test for USE GRAPH functionality
- **test_projection_e2e.sh** - End-to-end projection workflow test
- **test_gap_fixes.sh** - Tests for Gap 2 & 3 fixes (cache refresh, error messages)

### Integration Tests
- **test_project_integration.sh** - Integration tests for PROJECT() function
- **test_projection_query.gql** - Query test cases

### Debugging & Development Tests
- **test_projection_debug.sh** - Debug-focused tests with verbose output
- **test_projection_persistence.sh** - Tests projection persistence to disk
- **test_projection_fix.sh** - Tests for specific bug fixes
- **test_project_fix.sh** - Tests for PROJECT function fixes
- **verify_persistence.sh** - Verification script for projection persistence

## Running Tests

### Run all tests
```bash
# From repository root
./tests/projection/scripts/test_use_graph.sh
```

### Run with server kept alive
```bash
./tests/projection/scripts/test_use_graph.sh --keep-server
```

### Run gap fix tests
```bash
./tests/projection/scripts/test_gap_fixes.sh
```

## Test Data

Most tests use the example data:
- **Location**: `data/example/gql/posts/posts.gql`
- **Contains**: Users, Papers, Friend relationships, Cites relationships

## Expected Behavior

All scripts should:
1. ✅ Clean up previous test databases
2. ✅ Import fresh test data
3. ✅ Start server on port 1234
4. ✅ Run test queries via curl
5. ✅ Report pass/fail status with colors
6. ✅ Clean up server on exit

## Common Issues

**Server already running:**
```bash
pkill -f "mdb server"
```

**Port 1234 in use:**
```bash
# Edit script to use different port
SERVER_PORT=5678
```

**Permission denied:**
```bash
chmod +x tests/projection/scripts/*.sh
```
