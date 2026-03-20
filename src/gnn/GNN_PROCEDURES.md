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

### gnn_materialize_batches

Materialize packed feature batches from offline sampling output (L3 MinHash
reordering + L4 packed batch files). Bridges sampling → training.

```gql
CALL gnn_materialize_batches('sample_name', 'node_features', {
    reorder: 1, strategy: 'SEGMENTED', numHashes: 2, segmentSize: 100, force: 0
})
YIELD sampleName, featureName, totalBatches, reordered,
      reorderTimeMs, packTimeMs, totalTimeMs, packedDir
RETURN *
```

**Parameters:** sampleName (STRING), featureName (STRING), options (MAP optional).

**Options:** `reorder` (0/1, default 1), `strategy` ('SEGMENTED'|'MULTIPASS_BOUNDED'),
`numHashes` (INT, default 2), `segmentSize` (INT, default 100), `force` (0/1, default 0).

**Outputs on disk:**
- `gnn_features/<name>_reordered.fmat` — L3 reordered FeatureMatrix (if reorder=1)
- `gnn_features/<name>_reordered.rmap` — L3 reordered RowMapping (if reorder=1)
- `samples/<sampleName>/packed/batch_NNNNNN.bin` — L4 per-batch packed files

---

### gnn_build_feature_store

Build a four-level hierarchical feature store (DiskGNN-faithful). Classifies nodes
by access frequency into L1 (GPU), L2 (CPU), L3 (shared), and L4 (unique) tiers,
then writes cache files and slim packed batches.

```gql
CALL gnn_build_feature_store('sample_name', 'node_features', {
    gpu_budget_mb: 0, cpu_budget_mb: 100, reorder: 1, force: 0
})
YIELD sampleName, featureName, l1Nodes, l2Nodes, l3Nodes, l4Nodes,
      gpuAvailable, buildTimeMs
RETURN *
```

**Parameters:**
| Name | Type | Required | Default | Description |
|------|------|----------|---------|-------------|
| sampleName | STRING | yes | — | Existing sample set name |
| featureName | STRING | yes | — | Registered feature name |
| options | MAP | no | `{}` | See below |

**Options:**
| Option | Type | Default | Description |
|--------|------|---------|-------------|
| gpu_budget_mb | INT | 2048 | GPU memory budget in MB (0 = skip L1) |
| cpu_budget_mb | INT | 4096 | CPU memory budget in MB |
| reorder | BOOL | 1 | Use MinHash reordering for L3 |
| force | BOOL | 0 | Overwrite existing feature store files |
| strategy | STRING | 'SEGMENTED' | MinHash strategy ('SEGMENTED' or 'MULTIPASS_BOUNDED') |
| numHashes | INT | 64 | Hash functions for MinHash |
| segmentSize | INT | 50 | Batches per segment (SEGMENTED strategy) |

**YIELD columns:**
| Column | Type | Description |
|--------|------|-------------|
| sampleName | STRING | Echo of input sample name |
| featureName | STRING | Echo of input feature name |
| l1Nodes | INT | Nodes in GPU cache (L1) |
| l2Nodes | INT | Nodes in CPU cache (L2) |
| l3Nodes | INT | Shared nodes (L3, freq > 1) |
| l4Nodes | INT | Unique nodes (L4, freq == 1) |
| gpuAvailable | BOOL | Whether CUDA was available |
| buildTimeMs | INT | Total build time in milliseconds |

**Outputs on disk:**
- `gnn_features/<name>_gpu_cache.bin` — L1 GPU cache (GNNC format)
- `gnn_features/<name>_cpu_cache.bin` — L2 CPU cache (GNNC format)
- `gnn_features/<name>_store.meta` — Store metadata (GFLS format)
- `gnn_features/<name>_reordered.fmat` — L3 reordered FeatureMatrix (if reorder=1)
- `gnn_features/<name>_reordered.rmap` — L3 reordered RowMapping (if reorder=1)
- `samples/<sampleName>/packed_slim/batch_NNNNNN.bin` — L4 slim packed batches (GNNB v2)

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

### PackedBatchStore (.bin per batch) — GNNB v1
`[Header: 32B][Data: N×D×dtype_size]`
Header: magic=0x474E4E42 ("GNNB"), version=1, num_nodes(u64), feature_dim(u64), dtype(u8), reserved(7B).

### PackedBatchStore v2 (.bin per batch) — GNNB v2
`[Header: 32B][ObjectId table: N×8B][Data: N×D×dtype_size]`
Same header as v1 but version=2. ObjectId table between header and data enables
node identification without external index. Used by `packed_slim` files from
`gnn_build_feature_store`. v1 files remain readable (v1 has no OID table).

### Cache File (.bin) — GNNC Format
Used for gpu_cache.bin and cpu_cache.bin.
```
Offset  Size       Field
0       4B         magic: 0x474E4E43 ("GNNC")
4       4B         version: uint32 = 1
8       8B         num_nodes: uint64
16      8B         feature_dim: uint64
24      1B         dtype: uint8
25      7B         reserved
32      N×8B       ObjectId table
32+N×8  N×D×dtype  Feature rows (contiguous)
```

### Store Metadata — GFLS Format
Written by `gnn_build_feature_store`, read by FourLevelStore runtime constructor.
```
Offset  Size   Field
0       4B     magic: 0x47464C53 ("GFLS")
4       4B     version: uint32 = 1
8       8B     l1_count: uint64
16      8B     l2_count: uint64
24      8B     l3_count: uint64
32      8B     l4_count: uint64
40      8B     feature_dim: uint64
48      1B     dtype: uint8
49      1B     gpu_available: uint8
50      6B     reserved
56      256B   packed_slim_dir (null-terminated path)
```
Total: 312 bytes.
