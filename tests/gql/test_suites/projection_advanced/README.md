# Advanced Projection Test Suite

**Created:** 2026-03-24

Tests for MAP syntax, aggregation modes, per-type overrides, and bugfix verification.

## Test Data

**File**: `projection_advanced.gql`

- **Nodes (6)**: 4 Person (Alice, Bob, Carol, Dave) + 2 Item (Laptop, Phone)
  - Persons have varying properties: `rating` is absent on Carol and Dave
- **Edges (11)**: Includes 5 parallel TRANSFER edges for aggregation testing
  - 3 TRANSFER edges: node 0 to node 1
  - 2 TRANSFER edges: node 1 to node 2
  - Plus KNOWS, FOLLOWS, BOUGHT, SIMILAR edges

## Test Categories

### MAP Syntax (Tests 01-05)

| Test | Description |
|------|-------------|
| 01 | MAP with node properties |
| 02 | MAP with relationship orientation |
| 03 | MAP with both node and relationship config |
| 04 | MAP with node alias |
| 05 | MAP with type alias |

### Aggregation Modes (Tests 06-08)

| Test | Description |
|------|-------------|
| 06 | SINGLE aggregation (success on non-parallel edges) |
| 07 | COUNT aggregation on parallel edges |
| 08 | SUM aggregation on parallel edges |

### Per-Type Overrides (Tests 09-10)

| Test | Description |
|------|-------------|
| 09 | Per-type orientation override |
| 10 | Per-type aggregation override |

### Direction and Source Tests (Tests 11-12)

| Test | Description |
|------|-------------|
| 11 | REVERSE creation |
| 12 | UNDIRECTED on undirected source (no-op) |

### Wildcard and Rename (Tests 13-14)

| Test | Description |
|------|-------------|
| 13 | Wildcard relationship types |
| 14 | MAP property rename |

### USE Verification (Tests 15-16)

| Test | Description | Depends On |
|------|-------------|------------|
| 15 | Verify reverse direction in projected graph | Test 11 |
| 16 | Verify SUM values in projected graph | Test 08 |

### Bugfix Verification (Tests 17-23)

| Test | Expected Value | Verifies |
|------|---------------|----------|
| 17 | MIN = 50 | MIN aggregation fix |
| 18 | MIN = 25 | MIN aggregation fix |
| 19 | MAX = 200 | MAX aggregation fix |
| 20 | MAX = 75 | MAX aggregation fix |
| 21 | COUNT _count = 3 | COUNT _count column fix |
| 22 | COUNT _count = 2 | COUNT _count column fix |
| 23 | USE after DROP (no state leak) | State leak bugfix |

**Note**: Tests 15-22 depend on projections created by earlier tests. Alphabetical sort of query filenames guarantees execution order.

### Bad Queries (Bad 01-04)

| Test | Description | Expected |
|------|-------------|----------|
| bad_01 | SINGLE on parallel edges | Error |
| bad_02 | MIN without numeric property | Error |
| bad_03 | Too few parameters | Error |
| bad_04 | Too many parameters | Error |

## Summary

- **Regular tests**: 23
- **Bad queries**: 4
- **Total**: 27 tests

## Running

```bash
./scripts/run-tests gql          # Full GQL suite
```
