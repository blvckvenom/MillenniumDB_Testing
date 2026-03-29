# Projections

**Last updated:** 2026-03-24

## Overview

Projections create disk-based subgraphs from the main database for analysis. A projection contains a subset of nodes (by label) and edges (by type), optionally including properties. Once created, a projection can be queried using the `USE` clause.

Projections are stored on disk alongside the main database and persist across server restarts.

## Creating Projections

### CALL graph_project()

```
CALL graph_project(graphName, nodeProjection, relationshipProjection [, configuration])
YIELD graphName, nodeCount, relationshipCount, projectMillis
```

**Parameters:**

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| graphName | STRING | Yes | Name of the projection to create |
| nodeProjection | STRING, LIST, or MAP | Yes | Node label(s) to include |
| relationshipProjection | STRING, LIST, or MAP | Yes | Edge type(s) to include |
| configuration | MAP | No | Global configuration options |

### Node and Relationship Projection Variants

**String variant** -- a single label or type:

```
CALL graph_project('social', 'Person', 'KNOWS')
YIELD graphName, nodeCount, relationshipCount
RETURN graphName, nodeCount, relationshipCount
```

**List variant** -- multiple labels or types:

```
CALL graph_project('social', ['Person', 'Post'], ['KNOWS', 'LIKES'])
YIELD graphName, nodeCount, relationshipCount
RETURN graphName, nodeCount, relationshipCount
```

**Wildcard** -- all labels or all types:

```
CALL graph_project('full', '*', '*')
YIELD graphName, nodeCount, relationshipCount
RETURN graphName, nodeCount, relationshipCount
```

**Map variant** -- per-label or per-type configuration:

```
CALL graph_project('detailed', {
  Person: {label: 'Person', properties: ['age', 'name']},
  Book: {properties: ['title', 'price']}
}, {
  KNOWS: {orientation: 'UNDIRECTED'},
  REVIEWED: {orientation: 'NATURAL', properties: ['rating']}
})
YIELD graphName, nodeCount, relationshipCount
RETURN graphName, nodeCount, relationshipCount
```

### Configuration Options (4th Parameter)

The optional 4th parameter is a MAP that sets global defaults.

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| nodeProperties | STRING or LIST | [] | Node properties to include (`'age'` or `['age', 'name']`) |
| relationshipProperties | STRING or LIST | [] | Edge properties to include |
| orientation | STRING | 'NATURAL' | Edge orientation mode |
| aggregation | STRING | 'SINGLE' | Parallel edge aggregation strategy |
| aggregationProperty | STRING | (auto) | Property to use for MIN/MAX/SUM. If omitted, uses the first entry in `relationshipProperties`. |

Example:

```
CALL graph_project('g', 'Person', 'TRANSFER', {
  orientation: 'UNDIRECTED',
  aggregation: 'SUM',
  relationshipProperties: ['amount']
})
YIELD graphName, nodeCount, relationshipCount
RETURN graphName, nodeCount, relationshipCount
```

### Orientation Modes

Controls edge directionality in the projection.

| Mode | Behavior |
|------|----------|
| NATURAL | Edges stored as-is (from -> to). Default. |
| REVERSE | Edge direction reversed (to -> from). |
| UNDIRECTED | Directed source edges become bidirectional (A->B stored as both A->B and B->A). Undirected source edges are unchanged (already symmetric). |

UNDIRECTED on directed edges produces approximately 1.75x the storage of NATURAL. UNDIRECTED on undirected edges has no overhead.

### Aggregation Modes

Controls how parallel edges (multiple edges between the same node pair) are handled.

| Mode | Behavior |
|------|----------|
| SINGLE | Fail if parallel edges are detected. Default, strict validation. |
| COUNT | Count parallel edges and store as a property. |
| SUM | Sum a numeric property across parallel edges. Requires `relationshipProperties`. |
| MIN | Keep the edge with the minimum property value. Requires `relationshipProperties`. |
| MAX | Keep the edge with the maximum property value. Requires `relationshipProperties`. |

COUNT example -- the count is stored as a synthetic `_count` property:

```
CALL graph_project('counted', 'Person', 'TRANSFER', {aggregation: 'COUNT'})
YIELD graphName, nodeCount, relationshipCount
RETURN graphName, nodeCount, relationshipCount
```

Query the count via `e._count`:

```
USE counted
MATCH (a)-[e]->(b)
RETURN a, e._count, b
```

SUM example (requires specifying the property to aggregate):

```
CALL graph_project('summed', 'Person', 'TRANSFER', {
  aggregation: 'SUM',
  relationshipProperties: ['amount']
})
YIELD graphName, nodeCount, relationshipCount
RETURN graphName, nodeCount, relationshipCount
```

MIN/MAX example with explicit `aggregationProperty`:

```
CALL graph_project('cheapest', 'Person', 'TRANSFER', {
  aggregation: 'MIN',
  aggregationProperty: 'amount',
  relationshipProperties: ['amount']
})
YIELD graphName, nodeCount, relationshipCount
RETURN graphName, nodeCount, relationshipCount
```

### Per-Type Configuration

In the MAP variant for relationship projection, each type can override the global orientation and aggregation:

```
CALL graph_project('mixed', 'Person', {
  KNOWS: {orientation: 'UNDIRECTED'},
  FOLLOWS: {orientation: 'NATURAL', aggregation: 'COUNT'}
})
YIELD graphName, nodeCount, relationshipCount
RETURN graphName, nodeCount, relationshipCount
```

Per-type settings override the global defaults from the 4th parameter.

### Property Configuration

**Simple property list:**

```
CALL graph_project('g', {Person: {properties: ['age', 'name']}}, 'KNOWS')
YIELD graphName, nodeCount, relationshipCount
RETURN graphName, nodeCount, relationshipCount
```

**Renaming** -- read a source property under a different name:

```
CALL graph_project('g', {Person: {properties: {score: {property: 'rating'}}}}, 'KNOWS')
YIELD graphName, nodeCount, relationshipCount
RETURN graphName, nodeCount, relationshipCount
```

This reads the `rating` property from the source graph and stores it as `score` in the projection.

**Default values** -- substitute a value when the property is missing:

```
CALL graph_project('g', {Person: {properties: {age: {defaultValue: 0}}}}, 'KNOWS')
YIELD graphName, nodeCount, relationshipCount
RETURN graphName, nodeCount, relationshipCount
```

**Per-property aggregation** -- aggregate a specific property independently:

```
CALL graph_project('g', 'Person', {
  TRANSFER: {properties: {total: {property: 'amount', aggregation: 'SUM'}}}
})
YIELD graphName, nodeCount, relationshipCount
RETURN graphName, nodeCount, relationshipCount
```

### GNN Data Extraction

When building a projection for GNN training, additional fields in the configuration map instruct `graph_project` to extract classification labels, train/val/test splits, and link to an existing feature matrix.

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `includeFeatures` | STRING | `''` | Name of a registered FeatureMatrix (imported via `--with-tensors`) |
| `labelProperty` | STRING | `''` | Node property containing the classification label (integer) |
| `splitProperty` | STRING | `''` | Node property containing train/val/test assignment (string) |

All three fields are optional. Without them, `graph_project` behaves identically to before.

Example:

```gql
CALL graph_project('gnn_graph', ':Paper', ':CITES', {
    orientation: 'UNDIRECTED',
    includeFeatures: 'node_features',
    labelProperty: 'label',
    splitProperty: 'split'
})
YIELD graphName, nodeCount, relationshipCount, featureDim, numClasses, projectMillis
RETURN *
```

This produces three additional files in the projection directory:
- `gnn_meta.bin` — Metadata linking to the feature matrix
- `labels.bin` — Classification labels per node (int64, -1 for unlabeled)
- `splits.bin` — Split assignments per node (0=train, 1=val, 2=test, 255=unlabeled)

The `splitProperty` accepts values: `"train"`, `"val"`, `"validation"`, `"test"`. Other values map to unlabeled (255).

**Additional YIELD fields:**
| Field | Type | Description |
|-------|------|-------------|
| `featureDim` | INTEGER | Dimension of the feature matrix (0 if `includeFeatures` not set) |
| `numClasses` | INTEGER | Unique classification classes found (0 if `labelProperty` not set) |

### YIELD Fields

| Field | Type | Description |
|-------|------|-------------|
| graphName | STRING | Name of the created projection |
| nodeCount | INTEGER | Total number of nodes in the projection |
| relationshipCount | INTEGER | Total number of relationships in the projection |
| projectMillis | INTEGER | Time to create the projection in milliseconds |
| featureDim | INTEGER | Dimension of the linked feature matrix (0 if `includeFeatures` not set) |
| numClasses | INTEGER | Unique classification classes found (0 if `labelProperty` not set) |

All fields can be yielded, or a subset:

```
CALL graph_project('g', 'Person', 'KNOWS')
YIELD nodeCount, relationshipCount
RETURN nodeCount, relationshipCount
```

## Querying Projections

### USE Clause

Switch to a projection for subsequent queries:

```
USE my_projection
MATCH (n)-[e]->(m)
RETURN count(*) AS edge_count
```

The projection name is written as an unquoted identifier (no `GRAPH` keyword, no quotes).

### Supported Query Patterns

| Pattern | Requirement | Example |
|---------|-------------|---------|
| `(n)` node scan | Always works | `MATCH (n) RETURN count(*)` |
| `(n:Label)` label filter | Label must exist in projection | `MATCH (n:Person) RETURN n` |
| `(n)-[e]->(m)` directed edge | Always works | `MATCH (n)-[e]->(m) RETURN n, m` |
| `(n)-[e]-(m)` undirected edge | Always works | `MATCH (n)~[e]~(m) RETURN n, m` |
| `(n)-[e:Type]->(m)` typed edge | Type must exist in projection | `MATCH (n)-[e:KNOWS]->(m) RETURN n, m` |
| `n.property` property access | Property must be included during projection | `MATCH (n) RETURN n.age` |

Self-loop queries (`(n)-[e]->(n)` where source and target are the same variable) are not supported on projections.

### Returning to the Main Graph

Use `CURRENT_GRAPH` or `HOME_GRAPH` to switch back:

```
USE CURRENT_GRAPH
MATCH (n) RETURN count(*)
```

## PROJECT() Aggregate

Projections can also be created inline using the `PROJECT()` aggregate function inside a `RETURN` clause. This builds the projection from MATCH results rather than from B+Tree index scans.

```
MATCH (a:Person)-[e:KNOWS]->(b:Person)
RETURN PROJECT('my_graph' INCLUDE LABELS INCLUDE PROPERTIES)
```

The `INCLUDE LABELS` option preserves node/edge labels. The `INCLUDE PROPERTIES` option preserves property values. Both are optional.

## Behavior Notes

- **Endpoint filtering**: Only edges whose *both* endpoints are in the projected node set are included. For example, `graph_project('g', 'Person', 'BOUGHT')` will include zero BOUGHT edges if BOUGHT connects Person to Item and Item is not in the projection.
- **Deduplication**: Duplicate labels or types in lists (e.g., `['Person', 'Person']`) are automatically deduplicated. Each label is scanned once.
- **Missing labels**: If a label or type does not exist in the database, a warning is emitted but the projection is still created (with zero nodes/edges for that label).

## Limitations

- **Self-loops**: Queries with the pattern `(n)-[e]->(n)` (same variable for source and target) are not supported on projections.
- **Label aliasing**: The MAP syntax accepts a `label` key for aliasing (e.g., `{People: {label: 'Person'}}`), but aliasing is parsed without effect -- the source label name is used.
- **Streaming aggregation**: COUNT aggregation uses O(1) memory via streaming over sorted data. SUM/MIN/MAX require the hash-based path and may use more memory on graphs with very high parallel-edge counts.
- **Projection names**: Cannot contain `/`, `\`, null bytes, or control characters. Cannot be `.`, `..`, empty, or whitespace-only.
- **Duplicate projections**: Creating a projection with a name that already exists will fail.

## Examples

### 1. Basic social graph projection

Create a projection of people and their relationships, then count edges:

```
CALL graph_project('social', 'Person', 'KNOWS')
YIELD graphName, nodeCount, relationshipCount
RETURN graphName, nodeCount, relationshipCount
```

```
USE social
MATCH (n)-[e]->(m)
RETURN count(*) AS total_edges
```

### 2. Multi-label projection with properties

Project multiple node types with properties, then query:

```
CALL graph_project('library', ['Author', 'Book'], ['Published', 'Reviewed'], {
  nodeProperties: ['name', 'title'],
  relationshipProperties: ['year']
})
YIELD graphName, nodeCount, relationshipCount
RETURN graphName, nodeCount, relationshipCount
```

```
USE library
MATCH (a:Author)-[e:Published]->(b:Book)
RETURN a.name, b.title, e.year
```

### 3. Undirected projection for GNN

Create an undirected projection suitable for graph neural network algorithms:

```
CALL graph_project('gnn_graph', 'Person', {
  KNOWS: {orientation: 'UNDIRECTED'},
  FOLLOWS: {orientation: 'UNDIRECTED'}
})
YIELD graphName, nodeCount, relationshipCount
RETURN graphName, nodeCount, relationshipCount
```

### 4. Aggregated projection

Collapse parallel edges using COUNT aggregation:

```
CALL graph_project('transfers', 'Account', 'TRANSFER', {aggregation: 'COUNT'})
YIELD graphName, nodeCount, relationshipCount
RETURN graphName, nodeCount, relationshipCount
```

```
USE transfers
MATCH (a)-[e]->(b)
RETURN count(*) AS unique_pairs
```

### 5. GNN training projection

Create a projection with GNN metadata for training:

```gql
CALL graph_project('arxiv', ':Node', ':CITES', {
    orientation: 'UNDIRECTED',
    nodeProperties: ['label', 'split'],
    includeFeatures: 'node_features',
    labelProperty: 'label',
    splitProperty: 'split'
})
YIELD graphName, nodeCount, featureDim, numClasses
RETURN graphName, nodeCount, featureDim, numClasses
```

Then use with the GNN training pipeline:

```gql
CALL gnn_offline_sample('arxiv', 's1', [15, 10], {batchSize: 512, usePredefinedSplits: true})
YIELD totalBatches RETURN *
```

```gql
CALL gnn_train('s1', 'node_features', {hiddenDim: 256, epochs: 50, lr: 0.01})
YIELD bestValAccuracy, testAccuracy RETURN *
```
