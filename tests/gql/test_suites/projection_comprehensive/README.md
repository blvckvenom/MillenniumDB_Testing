# Comprehensive Projection Test Suite

**Created:** 2026-03-24

Tests for basic projection functionality, orientation modes, USE queries, and security validation.

## Test Data

**File**: `projection_comprehensive.gql`

- **Nodes (6)**: 4 Person (Alice, Bob, Carol, Dave) + 2 Item (Laptop, Phone)
- **Edges (10)**: 4 types
  - KNOWS (directed): Person-to-Person
  - FOLLOWS (directed): Person-to-Person
  - BOUGHT (directed): Person-to-Item (cross-label)
  - SIMILAR (undirected): Item-to-Item

## Test Categories

### Basic Projection (Tests 01-04)

| Test | Description |
|------|-------------|
| 01 | String variant (single label, single type) |
| 02 | List variant (multiple labels) |
| 03 | Wildcard node labels |
| 04 | Wildcard relationship types |

### Property Configuration (Tests 05-07)

| Test | Description |
|------|-------------|
| 05 | Node properties only |
| 06 | Edge properties only |
| 07 | Mixed node and edge properties |

### Orientation (Tests 08-09)

| Test | Description |
|------|-------------|
| 08 | UNDIRECTED orientation |
| 09 | REVERSE orientation |

### Edge Cases (Tests 10-12)

| Test | Description |
|------|-------------|
| 10 | Edge endpoint filtering (cross-label edges filtered) |
| 11 | Isolated label (no matching edges) |
| 12 | Deduplication of overlapping results |

### YIELD Variations (Tests 13-15)

| Test | Description |
|------|-------------|
| 13 | Partial YIELD (subset of fields) |
| 14 | YIELD with aliases |
| 15 | YIELD with expression in RETURN |

### Maximum Coverage (Test 16)

| Test | Description |
|------|-------------|
| 16 | All edge types projected simultaneously |

### USE Queries (Tests 17-20)

| Test | Description | Depends On |
|------|-------------|------------|
| 17 | Count edges in projected graph | Test 01 |
| 18 | Verify labels in projected graph | Test 02 |
| 19 | Query undirected edges | Test 04 |
| 20 | Count nodes in projected graph | Test 16 |

**Note**: Tests 17-20 depend on projections created by earlier tests. The test runner sorts queries alphabetically, so numbered prefixes guarantee execution order.

### Bad Queries (Bad 01-06)

| Test | Description | Expected |
|------|-------------|----------|
| bad_01 | Path traversal (`../`) in graph name | Error |
| bad_02 | Path traversal (`..`) in graph name | Error |
| bad_03 | Whitespace in graph name | Error |
| bad_04 | Empty string graph name | Error |
| bad_05 | Empty label in node list | Error |
| bad_06 | Wrong parameter type (integer) | Error |

## Summary

- **Regular tests**: 20
- **Bad queries**: 6
- **Total**: 26 tests

## Running

```bash
./scripts/run-tests gql          # Full GQL suite
```
