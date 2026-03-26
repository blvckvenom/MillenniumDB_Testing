# GNN Training Pipeline — Design Specification

**Date:** 2026-03-26
**Status:** Draft
**Scope:** Extension of `graph_project` for GNN data + Phase 4 Training Pipeline + Phase 5 GraphSAGE MEAN

---

## 1. Objectives

### What this spec covers

Two pieces developed in parallel:

1. **Extension of `graph_project`** — Three new fields in the `configuration` Map to produce GNN-ready files (`gnn_meta.bin`, `labels.bin`, `splits.bin`) inside the projection directory.

2. **Phase 4+5: Training Pipeline + GraphSAGE MEAN** — Minimum components for end-to-end training: LabelStore, SplitStore, BatchAssembler, TrainingLoop, GraphSAGE MEAN model, `gnn_train` procedure, and export for external validation.

### What this spec does NOT cover

- Converting scalar graph properties to FeatureMatrix columns (future extension)
- Feature normalization (z-score, min-max)
- `relationshipWeightProperty` / edge weights in GNN aggregation
- GCN, GAT, GIN models (only GraphSAGE MEAN)
- Phase 6 EmbeddingWriter (only basic `.npy` export)
- `mutate` mode (writing embeddings back to projection)

### Target datasets

- **Rapid validation:** Cora (2,708 nodes, trains in seconds)
- **Functional validation:** ogbn-arxiv (169,343 nodes)
- **Scale benchmark (future):** ogbn-papers100M (111M nodes)

### Success criteria

1. GraphSAGE MEAN training end-to-end on Cora with accuracy > 70%
2. Trained model exportable as `.pt`, loadable from Python with PyTorch
3. Generated embeddings exportable as `.npy`, verifiable externally with kNN/sklearn

---

## 2. Extension of `graph_project`

### 2.1 API changes

Three new optional fields in the 4th parameter (`configuration` Map):

| Field | Type | Default | Description |
|---|---|---|---|
| `includeFeatures` | STRING | `""` | Name of a registered FeatureMatrix (from `--with-tensors` at import) |
| `labelProperty` | STRING | `""` | Node property containing the classification class |
| `splitProperty` | STRING | `""` | Node property containing train/val/test assignment |

All are optional. Without them, `graph_project` behaves identically to today (backward compatible).

### 2.2 Usage examples

**Minimal (no changes from today):**
```gql
CALL graph_project('g', ':Node', ':CITES')
```

**Full GNN configuration:**
```gql
CALL graph_project('arxiv', ':Node', ':CITES', {
    orientation: 'UNDIRECTED',
    includeFeatures: 'node_features',
    labelProperty: 'label',
    splitProperty: 'split'
})
YIELD graphName, nodeCount, relationshipCount, featureDim, numClasses, projectMillis
```

**Without labels (unsupervised):**
```gql
CALL graph_project('hetero', ['Paper', 'Author', 'Venue'], ['CITES', 'WRITES'], {
    orientation: 'UNDIRECTED',
    includeFeatures: 'node_features'
})
```

### 2.3 New YIELD fields

| Field | Type | Description |
|---|---|---|
| `featureDim` | INTEGER | Dimension of the FeatureMatrix (0 if `includeFeatures` not specified) |
| `numClasses` | INTEGER | Unique classes found (0 if `labelProperty` not specified) |

Added to existing fields: `graphName`, `nodeCount`, `relationshipCount`, `projectMillis`.

### 2.4 Disk output

```
projections/{name}/
├── [existing B+Trees — unchanged]
│
├── gnn_meta.bin              ← GNN metadata (feature paths, dimensions)
├── labels.bin                ← [N] int64 (-1 if node lacks the property)
└── splits.bin                ← [N] uint8 (0=TRAIN, 1=VAL, 2=TEST, 255=UNLABELED)
```

The FeatureMatrix is NOT copied. `gnn_meta.bin` stores the **path** to the existing files in `gnn_features/`.

### 2.5 Internal logic

Within `NativeProjectionBuilder::scan_nodes_by_labels()`, which already visits each node and calls `extract_node_properties()`:

```
For each node (existing flow):
  1. storage->add_node(node_id)                           [EXISTING]
  2. extract_node_properties(node_id)                     [EXISTING]

  NEW — during extract_node_properties, when reading each property:
  3. If key == labelProperty:
       labels_buffer[node_index] = to_int64(value_id)
       unique_classes.insert(value)

  4. If key == splitProperty:
       splits_buffer[node_index] = parse_split(value_id)
       // "train"→0, "val"/"validation"→1, "test"→2, other→255

In finalize():
  5. If includeFeatures not empty:
       Verify FeatureMatrix exists in gnn_features/
       Verify num_rows >= projection nodeCount
       Write gnn_meta.bin with paths

  6. If labelProperty not empty:
       Write labels.bin

  7. If splitProperty not empty:
       Write splits.bin
```

### 2.6 Validations

| What | When | On failure |
|---|---|---|
| `includeFeatures` is registered in catalog | Argument parsing | Error: `"Feature 'X' not found. Import with: --with-tensors file.npy"` |
| FeatureMatrix has >= N rows | `finalize()` | Error: `"FeatureMatrix has X rows but projection has Y nodes"` |
| `labelProperty` exists on at least one node | After scan | Warning (not error) — nodes without label get -1 |
| `splitProperty` values recognized | During scan | Warning for unrecognized values → 255 |

### 2.7 Node-to-row mapping

The importer assigns `node_i → row_i` sequentially (import.cc lines 1244-1246). `labels.bin[i]` and `splits.bin[i]` use the same ordering as the RowMapping — `labels.bin[i]` = label of the node whose ObjectId is in `RowMapping[i]`.

If the projection includes all nodes (as in ogbn-arxiv where all are `:Node`), the mapping is trivial: row i = node i.

---

## 3. Phase 4 — Training Pipeline

### 3.1 Architecture

```
LabelStore ──┐
SplitStore ──┤
             ├──→ BatchAssembler ──→ TrainingLoop ──→ Output
FourLevelStore┤                          ↑
SampleStorage─┘                    GraphSAGE Model
```

BatchAssembler is the central point: it unites the four data sources into a `MiniBatch` struct that TrainingLoop consumes.

### 3.2 LabelStore

Reads `labels.bin` from the projection directory. Mmap read-only, O(1) access.

```cpp
class LabelStore {
public:
    static LabelStore open(const fs::path& labels_path);

    uint64_t num_nodes() const;
    uint64_t num_classes() const;

    int64_t get(uint64_t row_index) const;
    torch::Tensor gather(const std::vector<uint64_t>& row_indices) const;
    // Returns tensor [B] int64, -1 for unlabeled nodes
};
```

### 3.3 SplitStore

Reads `splits.bin` from the projection directory. Mmap read-only, O(1) access.

```cpp
class SplitStore {
public:
    static SplitStore open(const fs::path& splits_path);

    enum Split : uint8_t { TRAIN=0, VAL=1, TEST=2, UNLABELED=255 };

    uint64_t num_nodes() const;
    Split get(uint64_t row_index) const;
    torch::Tensor gather_mask(
        const std::vector<uint64_t>& row_indices, Split target
    ) const;  // Returns tensor [B] bool
};
```

### 3.4 MiniBatch

The contract between BatchAssembler and TrainingLoop:

```cpp
struct MiniBatch {
    torch::Tensor features;       // [N_batch, D] float32 — all nodes in subgraph
    std::vector<torch::Tensor> edge_indices;  // each [2, E_k] int64 — per layer
    torch::Tensor labels;         // [num_seeds] int64 — seed node labels
    torch::Tensor label_mask;     // [num_seeds] bool — true if label != -1
    uint64_t num_seeds;           // target nodes (layer 0)
    uint64_t num_nodes;           // total nodes in computational subgraph
    SplitType split;              // TRAIN / VAL / TEST
    uint64_t batch_id;
};
```

### 3.5 BatchAssembler

```cpp
class BatchAssembler {
public:
    BatchAssembler(
        FourLevelStore& feature_store,
        SampleStorage& samples,
        LabelStore& labels,
        SplitStore& splits,
        const RowMapping& row_mapping
    );

    MiniBatch assemble(uint64_t batch_id);
};
```

`assemble(batch_id)` does:
1. Read GraphSample from SampleStorage → node ObjectIds + edge indices per layer
2. Collect all unique nodes in the computational subgraph
3. Load features from FourLevelStore (or FeatureMatrix fallback)
4. Build `edge_indices` tensors from GraphSample `src_indices`/`dst_indices`
5. Gather labels for seed nodes via LabelStore + RowMapping
6. Build `label_mask` = (labels != -1)
7. Package into MiniBatch

### 3.6 TrainingLoop

```cpp
class TrainingLoop {
public:
    struct Config {
        uint64_t epochs = 50;
        double learning_rate = 0.01;
        double weight_decay = 0.0;
        double tolerance = 1e-4;
        uint64_t patience = 5;
        std::string output_dir;
    };

    struct Result {
        uint64_t ran_epochs;
        bool converged;
        double best_val_accuracy;
        std::vector<double> epoch_losses;
        double train_seconds;
    };

    TrainingLoop(
        torch::nn::Module& model,
        BatchAssembler& assembler,
        const SampleCatalog& catalog,
        Config config
    );

    Result train();
};
```

`train()` implements:
```
for epoch in 1..max_epochs:
    // Train phase
    model.train()
    for batch_id in catalog.train_batch_ids:
        mini = assembler.assemble(batch_id)
        embeddings = model.forward(mini.features, mini.edge_indices)
        predictions = classifier(embeddings[0:num_seeds])
        loss = cross_entropy(predictions[label_mask], labels[label_mask])
        loss.backward()
        optimizer.step()

    // Validation phase
    model.eval()
    val_accuracy = evaluate(model, assembler, val_batch_ids)

    // Early stopping
    if val_accuracy > best: save checkpoint, reset patience
    else: patience++; if patience >= max: break

    // Convergence check
    if |loss_prev - loss_curr| < tolerance: break
```

### 3.7 FourLevelStore fallback

`gnn_train` attempts to use FourLevelStore if it exists (steps 5+6 of the pipeline). If the user did not run `materialize_batches` + `build_feature_store`, the trainer falls back to reading directly from FeatureMatrix:

```
If FourLevelStore exists:
  features = store.load_batch_features(node_ids)    // fast, cached

If not (fallback):
  features = feature_matrix.extract_rows(row_ids)   // slower, but works
```

This enables a minimal flow for rapid testing with Cora (skip steps 5+6).

---

## 4. Phase 5 — GraphSAGE MEAN Model

### 4.1 The operation

One GraphSAGE MEAN layer:
```
h_v^(k) = σ( W^(k) · CONCAT(h_v^(k-1), MEAN({h_u^(k-1) : u ∈ N(v)})) )
```

Using existing `scatter_sum` from `sparse_ops.h`:
```
1. neighbor_features = x[src]                    // gather by edge source
2. agg = scatter_sum(neighbor_features, dst, N)  // sum by destination
3. degree = scatter_sum(ones, dst, N)            // count neighbors
4. agg = agg / degree                            // mean = sum / count
5. combined = concat(x, agg)                     // [N, 2*D_in]
6. out = relu(linear(combined))                  // [N, D_out]
7. out = l2_normalize(out)                       // unit norm
```

### 4.2 Model structure

```cpp
class GraphSAGEModel : public torch::nn::Module {
public:
    GraphSAGEModel(
        int64_t input_dim,
        int64_t hidden_dim,
        int64_t num_classes,
        int64_t num_layers,
        double dropout
    );

    torch::Tensor forward(
        torch::Tensor x,                              // [N, D]
        const std::vector<torch::Tensor>& edge_indices // per-layer [2, E_k]
    );

private:
    std::vector<torch::nn::Linear> layers;   // num_layers SAGEConv layers
    torch::nn::Linear classifier{nullptr};    // final → num_classes
    double dropout_;
};
```

Forward processes layers from outside in:
```
For 2 layers, fanouts [15, 10]:
  Layer 1: aggregate features from hop-2 nodes → hop-1 nodes  (edge_indices[1])
  Layer 2: aggregate from hop-1 nodes → seed nodes             (edge_indices[0])
  Classifier: linear(seed_embeddings) → [num_seeds, num_classes]
```

### 4.3 Configurable parameters

| Parameter | Default | Description |
|---|---|---|
| `hiddenDim` | 256 | Hidden layer dimension |
| `numLayers` | (inferred) | Equals length of sample fanouts |
| `dropout` | 0.5 | Dropout between layers (training only) |

---

## 5. Procedure `gnn_train`

### 5.1 Syntax

```gql
CALL gnn_train(sampleName, featureName [, options])
YIELD modelName, ranEpochs, didConverge, bestValAccuracy,
      trainAccuracy, testAccuracy, trainSeconds
```

### 5.2 Parameters

| Name | Type | Required | Default | Description |
|---|---|---|---|---|
| `sampleName` | STRING | yes | — | Existing sample set |
| `featureName` | STRING | yes | — | Registered feature name |
| `options` | MAP | no | `{}` | See table below |

**Options:**

| Key | Type | Default | Description |
|---|---|---|---|
| `model` | STRING | `'graphsage'` | Model type (only `'graphsage'` initially) |
| `hiddenDim` | INT | 256 | Hidden dimension |
| `dropout` | FLOAT | 0.5 | Dropout rate |
| `epochs` | INT | 50 | Maximum epochs |
| `lr` | FLOAT | 0.01 | Learning rate (Adam) |
| `weightDecay` | FLOAT | 0.0 | L2 regularization |
| `patience` | INT | 5 | Early stopping patience |
| `tolerance` | FLOAT | 0.0001 | Loss convergence threshold |
| `outputDir` | STRING | `'default'` | Subdirectory for model + embeddings |
| `exportEmbeddings` | BOOL | true | Generate embeddings.npy after training |

### 5.3 YIELD fields

| Field | Type | Description |
|---|---|---|
| `modelName` | STRING | Model identifier |
| `ranEpochs` | INT | Epochs executed |
| `didConverge` | BOOL | Stopped by convergence or early stopping |
| `bestValAccuracy` | FLOAT | Best validation accuracy |
| `trainAccuracy` | FLOAT | Final training accuracy |
| `testAccuracy` | FLOAT | Test accuracy (evaluated at end) |
| `trainSeconds` | FLOAT | Total training time |

### 5.4 Disk output

```
projections/{projection_name}/gnn_output/{outputDir}/
├── graphsage_model.pt        ← serialized model (torch::save)
├── training_log.json         ← epoch losses, accuracies, config
├── embeddings.npy            ← [N, hidden_dim] float32 (if exportEmbeddings=true)
└── node_ids.npy              ← [N] int64 — ObjectId per row
```

### 5.5 Internal logic

```
1. Parse arguments
2. Open SampleStorage → get projection_name from catalog
3. Open GnnMeta, LabelStore, SplitStore from projection directory
4. Open FeatureMatrix + FourLevelStore (or fallback to FeatureMatrix only)
5. Infer num_layers from sample fanouts
6. Create GraphSAGEModel(input_dim, hidden_dim, num_classes, num_layers, dropout)
7. Create BatchAssembler(store, samples, labels, splits, row_mapping)
8. Run TrainingLoop → Result
9. Evaluate test accuracy
10. Export model (.pt) + training_log.json + embeddings.npy + node_ids.npy
11. Yield results
```

---

## 6. Binary Formats

### 6.1 gnn_meta.bin

```
Offset  Size  Field
0       8     magic: "GNNM\0\0\0\0"
8       4     version: uint32 (1)
12      4     feature_dim: uint32
16      8     num_nodes: uint64
24      8     num_classes: uint64
32      1     has_labels: uint8 (0/1)
33      1     has_splits: uint8 (0/1)
34      2     reserved
36      4     feature_name_len: uint32
40      N     feature_name: char[N]
40+N    4     fmat_path_len: uint32
44+N    M     fmat_path: char[M] (relative path to .fmat)
44+N+M  4     rmap_path_len: uint32
48+N+M  P     rmap_path: char[P] (relative path to .rmap)
```

### 6.2 labels.bin

```
Offset  Size    Field
0       8       magic: "GNNL\0\0\0\0"
8       8       num_nodes: uint64
16      8       num_classes: uint64
24      N×8     labels: int64[N]  (-1 = unlabeled)
```

### 6.3 splits.bin

```
Offset  Size    Field
0       8       magic: "GNNS\0\0\0\0"
8       8       num_nodes: uint64
16      N×1     splits: uint8[N]  (0=TRAIN, 1=VAL, 2=TEST, 255=UNLABELED)
```

### 6.4 training_log.json

```json
{
  "model": "graphsage",
  "input_dim": 128,
  "hidden_dim": 256,
  "num_classes": 40,
  "num_layers": 2,
  "dropout": 0.5,
  "learning_rate": 0.01,
  "ran_epochs": 23,
  "did_converge": true,
  "best_val_accuracy": 0.6842,
  "test_accuracy": 0.6731,
  "epoch_losses": [2.45, 1.89, 1.52],
  "epoch_val_accuracies": [0.32, 0.45, 0.58],
  "train_seconds": 142.5,
  "sample_name": "arxiv_s1",
  "feature_name": "node_features",
  "projection_name": "arxiv"
}
```

---

## 7. File Inventory

### 7.1 New files

```
src/gnn/training/
  label_store.h, label_store.cc
  split_store.h, split_store.cc
  mini_batch.h
  batch_assembler.h, batch_assembler.cc
  training_loop.h, training_loop.cc
  npy_writer.h, npy_writer.cc

src/gnn/models/
  graphsage_model.h, graphsage_model.cc

src/gnn/projection/
  gnn_meta.h                              (reader + writer)

src/query/procedure/builtin/
  gnn_train_procedure.h, gnn_train_procedure.cc

src/gnn/tests/
  test_label_store.cc
  test_split_store.cc
  test_batch_assembler.cc
  test_graphsage_model.cc
  test_training_loop.cc
```

**Total: 22 new files.**

### 7.2 Modified files

```
src/query/procedure/builtin/project_procedure.cc
src/graph_models/gql/projection/native_projection_builder.h
src/graph_models/gql/projection/native_projection_builder.cc
src/graph_models/gql/gql_model.cc                            (register gnn_train)
src/gnn/CMakeLists.txt                                        (add subdirectories)
```

**Total: 5 modified files.**

### 7.3 LOC estimates

| Component | Files | LOC | Complexity |
|---|---|---|---|
| graph_project extension | 3 modified + 1 new | ~180 | Low |
| LabelStore | 2 new | ~80 | Low |
| SplitStore | 2 new | ~60 | Low |
| GnnMeta | 1 new | ~80 | Low |
| MiniBatch | 1 new | ~30 | Trivial |
| BatchAssembler | 2 new | ~200 | Medium |
| TrainingLoop | 2 new | ~250 | Medium |
| GraphSAGE Model | 2 new | ~150 | Medium |
| NpyWriter | 2 new | ~80 | Low |
| gnn_train procedure | 2 new | ~180 | Medium |
| Tests | 5 new | ~400 | Medium |
| **Total** | **22 new + 5 modified** | **~1,690** | |

---

## 8. Implementation Order

### Phase A — Unblock testing

1. Python script to generate `labels.bin` + `splits.bin` from existing `.npy` files (~40 LOC)
2. LabelStore + SplitStore + tests (~200 LOC)
3. GnnMeta reader (~40 LOC)

### Phase B — Training core (parallel with C)

4. MiniBatch struct (~30 LOC)
5. BatchAssembler + tests (~280 LOC)
6. GraphSAGE Model + tests (~230 LOC)
7. TrainingLoop + tests (~330 LOC)
8. NpyWriter (~80 LOC)

### Phase C — graph_project extension (parallel with B)

9. Parse new fields in project_procedure.cc (~30 LOC)
10. labels/splits writers in native_projection_builder.cc (~110 LOC)
11. GnnMeta writer (~40 LOC)
12. Integration tests (~100 LOC)

### Phase D — Integration

13. gnn_train procedure (~180 LOC)
14. End-to-end test with Cora (~100 LOC)
15. Remove temporary Python script

Phases A+B and C are **fully parallel**. D depends on both completing.

---

## 9. End-to-End Flow

```bash
# 1. Import graph + embeddings
mdb import ogbn_arxiv.gql data/dbs/gql/arxiv \
    --with-tensors ogbn_arxiv_features.npy

# 2. Start server
mdb server data/dbs/gql/arxiv
```

```gql
-- 3. Project with GNN data
CALL graph_project('arxiv', ':Node', ':CONNECTS', {
    orientation: 'UNDIRECTED',
    includeFeatures: 'node_features',
    labelProperty: 'label',
    splitProperty: 'split'
})
YIELD graphName, nodeCount, featureDim, numClasses
RETURN *

-- 4. Sample mini-batches
CALL gnn_offline_sample('arxiv', 's1', [15, 10], {batchSize: 512})
YIELD totalBatches, uniqueNodes RETURN *

-- 5. Materialize features (L3+L4)
CALL gnn_materialize_batches('s1', 'node_features', {reorder: 1})
YIELD totalBatches, packTimeMs RETURN *

-- 6. Build feature store (L1+L2)
CALL gnn_build_feature_store('s1', 'node_features', {cpu_budget_mb: 100})
YIELD l1Nodes, l2Nodes, l3Nodes, l4Nodes RETURN *

-- 7. Train GraphSAGE
CALL gnn_train('s1', 'node_features', {
    hiddenDim: 256, epochs: 50, lr: 0.01, patience: 5
})
YIELD bestValAccuracy, testAccuracy, trainSeconds RETURN *
```

```bash
# 8. External verification (Python)
python3 -c "
import numpy as np
emb = np.load('.../gnn_output/default/embeddings.npy')
print(f'Embeddings shape: {emb.shape}')
"
```

Minimal flow for rapid testing (skip steps 5+6):
```gql
CALL graph_project('cora', ':Paper', ':CITES', {
    orientation: 'UNDIRECTED', includeFeatures: 'node_features',
    labelProperty: 'label', splitProperty: 'split'
})
CALL gnn_offline_sample('cora', 's', [15, 10], {batchSize: 64})
CALL gnn_train('s', 'node_features', {epochs: 20})
YIELD bestValAccuracy, testAccuracy RETURN *
```

---

## 10. Future Extensions (Out of Scope)

- `featureProperties: ['age', 'salary']` — materialize scalar properties as FeatureMatrix columns
- `relationshipWeightProperty` — edge weights in GNN aggregation
- Feature normalization (z-score, min-max)
- GCN, GAT, GIN models
- Multi-GPU training
- `mutate` mode (write embeddings back to projection)
- Generate features from topology (Degree, PageRank as node features)
- Learnable embeddings for featureless nodes
