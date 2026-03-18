# GNN Procedure Reference

All procedures require `ENABLE_GNN=ON` at build time. Procedure names use
underscores in GQL queries: `CALL gnn_hnsw_create(...)`, not `gnn.hnsw.create`.

YIELD column names are **camelCase** (e.g., `indexName`, `nodeCount`).

---

## HNSW Index Procedures

### gnn_hnsw_create

Create an HNSW index over a registered FeatureMatrix.

```gql
CALL gnn_hnsw_create('index_name', 'tensor_key', {metric: 'cosine', M: 16, efConstruction: 200})
YIELD indexName, dimension, nodeCount, buildTimeMs
RETURN indexName, dimension, nodeCount, buildTimeMs
```

**Parameters:**
| Name | Type | Required | Default | Description |
|------|------|----------|---------|-------------|
| indexName | STRING | yes | — | Filesystem-safe name for the index |
| tensorKey | STRING | yes | — | Registered feature name (e.g., `'node_features'`) |
| options | MAP | no | `{}` | See below |

**Options:** `metric` ('cosine'|'euclidean'|'manhattan', default 'cosine'),
`M` (1-256, default 16), `efConstruction` (positive, default 200),
`threads` (1-cores, default all).

**YIELD columns:** `indexName` (STRING), `dimension` (INT), `nodeCount` (INT), `buildTimeMs` (INT)

---

### gnn_hnsw_find_similar

Query k nearest neighbors from an HNSW index.

```gql
CALL gnn_hnsw_find_similar('index_name', 42, 10, 100)
YIELD similar_node, distance
RETURN similar_node, distance
```

**Parameters:** indexName (STRING), nodeId (INT, 0-based), k (INT), ef (INT, should be >= k).

**YIELD columns:** `similar_node` (NODE), `distance` (FLOAT). Returns k rows sorted by distance ascending.

---

### gnn_hnsw_info

```gql
CALL gnn_hnsw_info('index_name')
YIELD indexName, dimension, nodeCount, metric
RETURN *
```

---

### gnn_hnsw_list

```gql
CALL gnn_hnsw_list()
YIELD indexName, dimension, nodeCount, metric
RETURN *
```

---

### gnn_hnsw_drop

```gql
CALL gnn_hnsw_drop('index_name')
YIELD success
RETURN success
```

---

## Sampling Procedures

### gnn_offline_sample

Pre-compute mini-batches via k-hop neighborhood sampling on a projection.

```gql
CALL gnn_offline_sample('projection_name', 'sample_name', [15, 10], {
    batchSize: 1024, trainRatio: 0.7, validationRatio: 0.15,
    testRatio: 0.15, randomSeed: 42, orientation: 'REVERSE'
})
YIELD sampleName, projectionName, totalBatches, trainBatches,
      validationBatches, testBatches, uniqueNodes, computeMillis
RETURN *
```

**Parameters:**
| Name | Type | Required | Default | Description |
|------|------|----------|---------|-------------|
| projectionName | STRING | yes | — | Source graph projection |
| sampleName | STRING | yes | — | Name for the sample set |
| fanouts | LIST\<INT\> | yes | — | Neighbors per layer, e.g. `[15, 10]` for 2-hop |
| options | MAP | no | `{}` | See below |

**Options:** `batchSize` (default 1024), `trainRatio` (default 0.7),
`validationRatio` (default 0.15), `testRatio` (default 0.15),
`randomSeed` (default 42), `orientation` ('NATURAL'|'REVERSE'|'UNDIRECTED', default 'REVERSE').

Ratios must sum to 1.0 (tolerance ±0.001).

**YIELD columns:** `sampleName`, `projectionName`, `totalBatches`, `trainBatches`,
`validationBatches`, `testBatches`, `uniqueNodes`, `computeMillis` (all STRING/INT).

---

### gnn_sample_info

```gql
CALL gnn_sample_info('sample_name')
YIELD sampleName, projectionName, totalBatches, trainBatches,
      validationBatches, testBatches, uniqueNodes, totalEdges,
      batchSize, numLayers, fanouts, randomSeed
RETURN *
```

`fanouts` is returned as a comma-separated string (e.g., `"15,10"`).

---

### gnn_sample_list

```gql
CALL gnn_sample_list()
YIELD sampleName, projectionName, totalBatches, uniqueNodes
RETURN *
```

---

### gnn_sample_drop

```gql
CALL gnn_sample_drop('sample_name')
YIELD success, message
RETURN success, message
```

---

## Binary File Formats

### FeatureMatrix (.fmat)
`[Header: 64B][Data: N×D×dtype_size]`
Header: magic=0x474E4E46 ("GNNF"), version=1, num_rows(u64), num_cols(u64), dtype(u8), reserved(39B).

### RowMapping (.rmap)
`[Header: 16B][Data: N×8B]`
Header: magic=0x524D4150 ("RMAP"), version=1, count(u64).

### SampleCatalog (catalog.dat in samples/)
Magic=0x534E4E47 ("GNNS"), version=2. See `src/gnn/sampling/sample_catalog.h` for full offset table.

### Batch Index (batches.idx)
`[Header: 16B][Entries: N×16B]`
Header: magic=0x58444E49 ("INDX"), version=1, count(u64). Each entry: offset(u64), size(u64).

### Frequency (frequency.dat)
`[Header: 16B][Entries: N×16B]`
Header: magic=0x51455246 ("FREQ"), version=1, count(u64). Each entry: node_oid(u64), count(u64).

### Batch Data (batches.dat)
`[Header: 8B][GraphSample blobs...]`
Header: magic=0x48435442 ("BTCH"), version=1. Each blob: magic=0x4D534E47 ("GNSM"), version=2.
See `src/gnn/sampling/graph_sample.cc` for GraphSample serialization layout.

### PackedBatchStore (.bin per batch)
`[Header: 32B][Data: N×D×dtype_size]`
Header: magic=0x474E4E42 ("GNNB"), version=1, num_nodes(u64), feature_dim(u64), dtype(u8), reserved(7B).
