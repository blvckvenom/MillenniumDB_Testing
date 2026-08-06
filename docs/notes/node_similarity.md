# GQL nodeSimilarity

This document describes the current semantics and usage of the GQL
`nodeSimilarity` procedure. Some implementation details, especially projection
behavior and performance characteristics, may evolve as the analytical engine
matures.

## Overview

`nodeSimilarity` computes topological similarity between nodes by comparing
their neighborhoods. It is exposed as a GQL `CALL` procedure and returns pairs
of nodes together with a numeric similarity score.

The procedure is read-only: it does not create, update, or delete graph data.
Results are returned directly as query rows.

## Syntax

Basic usage:

```gql
CALL nodeSimilarity()
RETURN node1, node2, similarity
```

Usage with arguments:

```gql
CALL nodeSimilarity(
  similarityMetric: "JACCARD",
  similarityCutoff: 0.2,
  topK: 10
)
RETURN node1, node2, similarity
```

Arguments are optional and must be passed by name. Metric names are currently
case-sensitive strings and must use their canonical uppercase form, such as
`"JACCARD"`, `"OVERLAP"`, or `"COSINE"`.

## Output Columns

| Column | Description |
| --- | --- |
| `node1` | First node in the result row |
| `node2` | Second node in the result row |
| `similarity` | Similarity score as a double |

`node1` and `node2` are GQL nodes. They can be returned directly, inspected with
`properties(node1)` or `properties(node2)`, or used in property access
expressions such as `node1.name`.

The output variables can also be renamed with `YIELD ... AS ...`:

```gql
CALL nodeSimilarity(topN: 10)
YIELD node1 AS source, node2 AS target, similarity AS score
RETURN source, target, score
```

## Neighborhood Semantics

For a node `u`, `nodeSimilarity` currently defines the neighborhood `N(u)` as:

```text
N(u) = { v | u ~ v exists }
     union { v | u -> v exists }
```

The following rules apply:

- An undirected edge `u ~ v` contributes `v` to `N(u)` and `u` to `N(v)`.
- A directed edge `u -> v` contributes `v` to `N(u)`.
- A directed edge `u -> v` does not contribute `u` to `N(v)`.
- Self-loops are included. A loop `u -> u` or `u ~ u` contributes `u` to `N(u)`.
- Parallel edges do not duplicate neighbors. Neighborhoods are sets of distinct
  nodes.
- Nodes without neighbors do not participate by default because `degreeCutoff`
  defaults to `1`.

## Pair Universe

The procedure first builds the neighborhood map and applies the configured
degree filters. It then compares the nodes that remain after that filtering
step.

Without `topK` or `bottomK`, each unordered pair is emitted at most once. For a
pair of nodes `u` and `v`, the output contains either `(u, v)` or `(v, u)`, using
MillenniumDB internal id ordering.

With `topK` or `bottomK`, candidate pairs are expanded into directed result rows
so that each source node can have its own per-node ranking. In that mode, both
`(u, v)` and `(v, u)` may appear if each node selects the other within its
ranking.

## Similarity Metrics

All metrics compare the neighborhood sets of two nodes:

$$
\mathrm{JACCARD}(u, v) =
\frac{|N(u) \cap N(v)|}{|N(u) \cup N(v)|}
$$

$$
\mathrm{OVERLAP}(u, v) =
\frac{|N(u) \cap N(v)|}{\min(|N(u)|, |N(v)|)}
$$

$$
\mathrm{COSINE}(u, v) =
\frac{|N(u) \cap N(v)|}{\sqrt{|N(u)| \cdot |N(v)|}}
$$

If a metric denominator is zero, the score is `0.0`. With the default
`degreeCutoff = 1`, denominator-zero cases should normally be removed before
score computation.

## Arguments

| Argument | Type | Default | Description |
| --- | --- | --- | --- |
| `similarityMetric` | string | `"JACCARD"` | Metric to use: `"JACCARD"`, `"OVERLAP"`, or `"COSINE"` |
| `similarityCutoff` | number | `0.0` | Minimum similarity score to emit |
| `degreeCutoff` | integer | `1` | Minimum node degree to consider |
| `upperDegreeCutoff` | integer | `UINT64_MAX` | Maximum node degree to consider |
| `topK` | integer | absent | Maximum number of most-similar rows per source node |
| `bottomK` | integer | absent | Maximum number of least-similar rows per source node |
| `topN` | integer | absent | Global maximum number of most-similar rows |
| `bottomN` | integer | absent | Global maximum number of least-similar rows |

`topK` and `bottomK` must be greater than zero when provided. `topN` and
`bottomN` may be zero; in that case the procedure returns zero rows for the
corresponding global limit. An absent ranking argument means that the
corresponding limit is not applied.

## Filtering and Ranking

The current processing order is:

```text
degree filter
-> score computation
-> similarityCutoff
-> topK or bottomK per node
-> topN or bottomN globally
```

Degree and similarity cutoffs are inclusive:

```text
degree >= degreeCutoff
degree <= upperDegreeCutoff
similarity >= similarityCutoff
```

The following argument combinations are invalid:

```text
topK + bottomK
topN + bottomN
topK + bottomN
bottomK + topN
```

These combinations are rejected to avoid mixing incompatible most-similar and
least-similar ranking modes in the same result set.

## Ordering and Tie-Breaking

Ordering and tie-breaking use MillenniumDB internal node ids, not user-defined
properties such as `_id`.

When neither `topK` nor `bottomK` is used, pairs are generated in ascending
internal id order and each unordered pair is emitted at most once.

For per-node ranking:

- `topK` sorts candidates by descending score and breaks ties by ascending
  `node2` internal id.
- `bottomK` sorts candidates by ascending score and breaks ties by ascending
  `node2` internal id.

For global ranking:

- `topN` sorts rows by descending score and breaks ties by ascending `node1`
  internal id, then ascending `node2` internal id.
- `bottomN` sorts rows by ascending score and breaks ties by ascending `node1`
  internal id, then ascending `node2` internal id.

Scores are stored as doubles and are not manually rounded by the procedure.

## Examples

Basic usage with the default metric:

```gql
CALL nodeSimilarity()
RETURN node1, node2, similarity
LIMIT 10
```

Top global Jaccard pairs:

```gql
CALL nodeSimilarity(similarityMetric: "JACCARD", topN: 10)
RETURN node1, node2, similarity
```

Top global Overlap pairs above a cutoff:

```gql
CALL nodeSimilarity(
  similarityMetric: "OVERLAP",
  similarityCutoff: 0.5,
  topN: 10
)
RETURN node1, node2, similarity
```

Top two Cosine neighbors per source node:

```gql
CALL nodeSimilarity(similarityMetric: "COSINE", topK: 2)
RETURN node1, node2, similarity
```

Degree-filtered results:

```gql
CALL nodeSimilarity(
  degreeCutoff: 2,
  upperDegreeCutoff: 100,
  similarityCutoff: 0.3
)
RETURN node1, node2, similarity
```

Aliased output:

```gql
CALL nodeSimilarity(topN: 10)
YIELD node1 AS source, node2 AS target, similarity AS score
RETURN source, target, score
```

## Limitations

- The current implementation is naive and builds adjacency and result data in
  memory.
- The procedure is not optimized for large graphs.
- Projection behavior needs semantic review, especially around edge direction.

## Implementation References

- Logical procedure declaration: `src/query/parser/op/gql/op_procedure.h`
- CALL parser wiring:
  `src/query/parser/grammar/gql/query_visitor.cc`
- Physical iterator:
  `src/query/executor/binding_iter/procedure/node_similarity.cc`
- Physical constructor wiring:
  `src/query/optimizer/property_graph_model/binding_list_iter_constructor.cc`
