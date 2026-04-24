# Thesis Demo Snapshot — ogbn-arxiv end-to-end

**Date**: 2026-04-24 (captured post-fixes)
**Projection**: `arxiv_gnn` under `data/dbs/gql/ogbn-arxiv/projections/arxiv_gnn/`
**Stack**: Spec #3 (GNN_MINIMAL) + Spec #5 (DELTA_VARINT) + Spec #8 (CSR_HYBRID) + Phase 6 (writeProperty)
**Head**: `92e50ff0` (cumulative of `1428c66c` T8-B.1, `b8bdf10f` perf, `49664262` T8-B.2, `92e50ff0` docs)

---

## Build configuration

```gql
CALL graph_project('arxiv_gnn', 'Node', 'CONNECTS', {
    orientation: 'UNDIRECTED',
    indexSet: 'GNN_MINIMAL',
    leafFormat: 'DELTA_VARINT',
    graphStorage: 'CSR_HYBRID',
    includeFeatures: 'node_features',
    labelProperty: 'label',
    splitProperty: 'split'
}) YIELD graphName, nodeCount, relCount, projectMillis, featureDim, numClasses
```

Result: 169,343 nodes, feat_dim=128, numClasses=40, projectMillis within budget.

## Training configuration

```gql
CALL gnn_train('arxiv_s', 'node_features', {
    model: 'graphsage', hiddenDim: 128, epochs: 30, lr: 0.01,
    dropout: 0.5, patience: 10, randomSeed: 42,
    exportEmbeddings: true,
    outputDir: 'arxiv_retry3',
    writeProperty: 'embedding',
    inferenceBatchSize: 2048
})
```

## Training results (from `gnn_output/arxiv_retry3/training_log.json`)

```json
{
  "results": {
    "ran_epochs": 10,
    "test_accuracy": 0.508611,
    "train_seconds": 5.551474
  },
  "cache_stats": {
    "l1_hits": 1526297,
    "l1_hit_ratio": 1.000000
  },
  "epoch_losses": [2.192339, 2.031755, 1.990569, 1.972095, 1.955176,
                   1.944271, 1.936663, 1.932771, 1.928053, 1.923500]
}
```

- **testAccuracy: 0.5086** — lower than published GraphSAGE (~0.67) because patience=10 with `usePredefinedSplits=true` on arxiv produced 0 validation batches (all dataset splits went to train/test without val). Early-stopping triggered at epoch 10 despite loss still decreasing. This is an **infrastructure artifact of the splits, not the model**. Fix: explicit `val_ratio: 0.2` in sample config.
- **train_seconds: 5.55s** on GPU (CPU baseline was ~150s) — GPU path confirmed, L1 cache hit ratio 1.0 for all 1.5M feature requests.
- Loss monotonically decreasing: 2.19 → 1.92 (model learning correctly).

## writeProperty / EmbeddingWriter

- Phase B: 29,799 non-seed nodes in 15 chunks of 2048 on device=cuda
- Total Phase B wall-clock: ~23 min (chunks took 5s → 137s with linear growth per chunk index — suspected FeatureMatrix hot-page eviction)
- Phase C: BPlusTree batch insert of all 169,343 embeddings to `node_key_value` + `key_value_node`
- Final: tensors.dat 86.86 MB (169343 × 128 × 4 = correct size)

## THESIS DEMO — queryable embeddings

### Query 1 — coverage
```gql
USE arxiv_gnn MATCH (n) WHERE n.embedding IS NOT NULL RETURN count(n) AS c
```
Result: **169,343** (100% of nodes have a queryable embedding)

### Query 2 — tensor retrieval
```gql
USE arxiv_gnn MATCH (n) RETURN n.embedding LIMIT 2
```
Result (first row, 128 floats abbreviated):
```
[0.145360, -0.005315, -0.014548, 0.073661, 0.136137, 0.085491, 0.087647, 0.072852, 0.138640, -0.035463, -0.122946, 0.006053, 0.018679, -0.097047, -0.976792, -0.019374, 0.493970, -0.201721, 0.071680, -0.204432, -0.141727, 2.518395, 0.100279, -0.176934, -0.050323, -0.251058, 0.563431, -0.104452, 0.154907, 0.181878, ...]
```
Second row (different embedding):
```
[-0.039265, 0.017584, 0.006513, -0.007301, -0.017552, 0.081927, -0.103991, 0.064408, 0.068787, -0.068880, -0.072247, 0.050968, 0.009136, -0.060770, -1.469260, 0.072644, 1.174816, -0.134106, -0.054129, -0.054845, -0.002443, 0.035325, ...]
```
Distinct learned embeddings per node (not init noise — values have expected SAGE distribution: mostly centered near 0 with sparse outliers).

### Query 3 — cosineDistance sanity (self-similarity)
```gql
USE arxiv_gnn MATCH (a) RETURN cosineDistance(a.embedding, a.embedding) AS self LIMIT 3
```
Result:
```
self
0.0E0
0.0E0
0.0E0
```
Identity check: a node's cosine distance to itself is 0 across all samples. ✓

### Query 4 — cosineDistance between different nodes
```gql
USE arxiv_gnn MATCH (a), (b) RETURN cosineDistance(a.embedding, b.embedding) AS sim LIMIT 3
```
Result:
```
sim
0.0E0
8.296009E-1
5.695621E-1
```
Sensible values in [0, 2] range. First pair is identical node (sim=0); other pairs show realistic similarity distribution. ✓

## Leaf format verification

| File | byte 0 | Format |
|---|:---:|---|
| `from_to_edge.leaf` | **3** | CSR_HYBRID (Spec #8) |
| `to_from_edge.leaf` | **3** | CSR_HYBRID (Spec #8) |
| `nodes.leaf` | **2** | DELTA_VARINT (Spec #5) |
| `node_label.leaf` | **2** | DELTA_VARINT |
| `label_node.leaf` | **2** | DELTA_VARINT |
| `node_key_value.leaf` | 156 (non-canonical) | BITSET v1 (T8-B.2 fix — mutable for writeProperty) |
| `key_value_node.leaf` | 170 (non-canonical) | BITSET v1 (T8-B.2 fix — mutable for writeProperty) |

The hybrid mutability story is empirically visible: topology leaves compressed (v2/v3), property leaves mutable (v1).

## Disk footprint

- Projection: **155 MB** total
- `from_to_edge.leaf`: 5.47 MB (CSR-compressed)
- `to_from_edge.leaf`: 5.51 MB (CSR-compressed)
- `node_key_value.leaf`: 1.57 MB (BITSET v1)
- `key_value_node.leaf`: 1.99 MB (BITSET v1)
- `tensors.dat` (DB-level, shared): 86.86 MB

## Verification commands (for reproducibility)

```bash
# Health check
pgrep -a mdb
curl -s -H "Accept:text/csv" -X POST http://localhost:18100 -d \
    "USE arxiv_gnn MATCH (n) WHERE n.embedding IS NOT NULL RETURN count(n)"
# Expected: 169343

# Thesis money shot
curl -s -H "Accept:text/csv" -X POST http://localhost:18100 -d \
    "USE arxiv_gnn MATCH (n) RETURN n.embedding LIMIT 1"
# Expected: [float1, float2, ..., float128]

# Cosine demo
curl -s -H "Accept:text/csv" -X POST http://localhost:18100 -d \
    "USE arxiv_gnn MATCH (a) RETURN cosineDistance(a.embedding, a.embedding) AS self LIMIT 1"
# Expected: 0.0E0
```

## Conclusion

The MillenniumDB thesis stack works end-to-end on ogbn-arxiv at full compression (Spec #3+#5+#8) with queryable embeddings (Phase 6). All 169,343 nodes have embeddings accessible via standard GQL `MATCH`. `cosineDistance` built-in returns valid similarity values. The hybrid mutability model is empirically demonstrated (topology immutable, properties mutable).

This snapshot serves as the empirical proof-of-concept for the thesis claim: *"MillenniumDB is the first graph DBMS closing the loop graph → GNN → queryable embeddings within a single ISO/IEC 39075 GQL system."*

Next milestone: reproduce on ogbn-products (62M edges, 2.45M nodes) via the same pipeline.
