# GNN Training Pipeline — Design Specification

**Date:** 2026-03-26
**Status:** Reviewed
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
Pre-scan setup:
  Open RowMapping from gnn_features/ (needed for correct indexing)
  Allocate labels_buffer[num_feature_rows] initialized to -1
  Allocate splits_buffer[num_feature_rows] initialized to 255

For each node (existing flow):
  1. storage->add_node(node_id)                           [EXISTING]
  2. extract_node_properties(node_id)                     [EXISTING]

  NEW — during extract_node_properties, when reading each property:
  3. row_index = row_mapping.find(node_id)   // O(log N) lookup
     If not found: skip GNN extraction (node has no features)

  4. If key == labelProperty:
       labels_buffer[row_index] = to_int64(value_id)
       unique_classes.insert(value)

  5. If key == splitProperty:
       splits_buffer[row_index] = parse_split(value_id)
       // "train"→0, "val"/"validation"→1, "test"→2, other→255

In finalize():
  6. If includeFeatures not empty:
       Verify FeatureMatrix exists in gnn_features/
       Verify num_rows >= projection nodeCount
       Write gnn_meta.bin (stores feature_name, not paths — see 6.1)

  7. If labelProperty not empty:
       Write labels.bin (with version field)

  8. If splitProperty not empty:
       Write splits.bin (with version field)
```

**Critical: Indexing by RowMapping, not scan order.** The projection scans nodes grouped by label (all `:Paper` first, then `:Author`, etc.), which may not match the sequential ObjectId order used by the RowMapping. Using `row_mapping.find(node_id)` guarantees `labels_buffer[i]` corresponds to `FeatureMatrix[i]` regardless of scan order.

### 2.6 Validations

| What | When | On failure |
|---|---|---|
| `includeFeatures` is registered in catalog | Argument parsing | Error: `"Feature 'X' not found. Import with: --with-tensors file.npy"` |
| FeatureMatrix has >= N rows | `finalize()` | Error: `"FeatureMatrix has X rows but projection has Y nodes"` |
| `labelProperty` exists on at least one node | After scan | Warning (not error) — nodes without label get -1 |
| `splitProperty` values recognized | During scan | Warning for unrecognized values → 255 |

Note: the row count validation checks size only, not row identity. It ensures the FeatureMatrix has enough rows for all projected nodes.

### 2.7 Node-to-row mapping

The importer assigns `node_i → row_i` sequentially (import.cc lines 1244-1246). Both `labels.bin` and `splits.bin` are indexed by **RowMapping row index**, not by scan order or ObjectId value.

`labels.bin[i]` = label of the node whose ObjectId is in `RowMapping[i]`.
`splits.bin[i]` = split of the node whose ObjectId is in `RowMapping[i]`.

For single-label projections (ogbn-arxiv: all nodes are `:Node`), scan order happens to match RowMapping order. For multi-label projections, the `row_mapping.find()` lookup in step 3 ensures correctness regardless of scan order.

### 2.8 Interaction with sampling splits

When `splitProperty` is provided, the projection's `splits.bin` contains **predefined** train/val/test assignments (from the dataset). This conflicts with `gnn_offline_sample`'s ratio-based random splitting (`trainRatio: 0.7`, etc.).

**Resolution:** `gnn_offline_sample` gains a new option `usePredefinedSplits: true`. When enabled:
- The sampler reads `splits.bin` from the projection directory
- `SeedSelector` uses the predefined splits instead of random ratio-based splitting
- The `trainRatio`/`validationRatio`/`testRatio` parameters are ignored
- Each `GraphSample` gets its `SplitType` from the node's predefined split

When `usePredefinedSplits` is not set (default), behavior is unchanged — ratio-based random splitting as today.

This requires modifying `SeedSelector` (~30 LOC) to accept an external split assignment vector. Added to the file inventory (Section 7.2).

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

**Note on SampleCatalog:** `SampleCatalog` is an existing component (from Phase 2, `src/gnn/sampling/sample_catalog.h`) that stores metadata about a sample set including `total_batches`, `train_batches`, `validation_batches`, `test_batches`, `projection_name`, `fanouts`, etc. The TrainingLoop uses `catalog.train_batches` and `catalog.validation_batches` to iterate over the correct batch IDs per split. These counts are set during `gnn_offline_sample` based on either ratio-based splitting or predefined splits (Section 2.8).

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
2. Collect all unique nodes in the computational subgraph → `all_unique_nodes` vector
3. Build **global index map**: `oid_to_global[node_oid] = position in all_unique_nodes`
4. Load features from FourLevelStore (or FeatureMatrix fallback) → `[N_batch, D]`
5. **Remap edge indices** per layer:
   - GraphSample stores `src_indices[i]` as index into `nodes_per_layer[k+1]` (layer-local)
   - Convert: `global_src = oid_to_global[nodes_per_layer[k+1][src_indices[i]]]`
   - Convert: `global_dst = oid_to_global[nodes_per_layer[k][dst_indices[i]]]`
   - Build `edge_indices[k]` as tensor `[2, E_k]` with global indices
6. Gather labels for seed nodes via LabelStore + RowMapping:
   - `seed_row_indices[i] = row_mapping.find(nodes_per_layer[0][i])` → RowMapping index
   - `labels = label_store.gather(seed_row_indices)` → `[num_seeds]` int64
7. Build `label_mask` = (labels != -1)
8. Package into MiniBatch

**Edge index semantics:** In `edge_indices[k]`, row 0 = message source (layer k+1 nodes, further from seeds), row 1 = message destination (layer k nodes, closer to seeds). Messages flow from source toward seeds. The forward pass iterates k = num_layers-1 down to 0.

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
        int64_t random_seed = -1;     // -1 = non-deterministic
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
        optimizer.zero_grad()
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
  features = store.load_batch_features(batch_id)    // takes batch_id, not node_ids
  // Returns torch::Tensor [N_batch, D] in all_unique_nodes order

If not (fallback):
  // FeatureMatrix::extract_rows is void with pre-allocated buffer:
  row_ids = translate ObjectIds → row indices via RowMapping
  buffer = allocate(row_ids.size() * fm.row_bytes())
  fm.extract_rows(row_ids, buffer.data())
  features = torch::from_blob(buffer, {N, D}, kFloat32)
```

This enables a minimal flow for rapid testing with Cora (skip steps 5+6).

---

## 4. Phase 5 — GraphSAGE MEAN Model

### 4.1 The operation

One GraphSAGE MEAN layer:
```
h_v^(k) = σ( W^(k) · CONCAT(h_v^(k-1), MEAN({h_u^(k-1) : u ∈ N(v)})) )
```

Note: The spec uses the general GraphSAGE framework (Algorithm 1, lines 4-5: CONCAT(self, AGG(neighbors))) with MEAN as the aggregate function. This differs from the paper's "mean aggregator" (Equation 2) which includes the self node inside the mean, but is mathematically equivalent to PyG's SAGEConv formulation (`W_1*x_i + W_2*MEAN(x_j)`) and is the most widely used in practice.

Using existing `scatter_sum` from `sparse_ops.h`:
```
1. neighbor_features = x[src]                    // gather by edge source
2. agg = scatter_sum(neighbor_features, dst, N)  // sum by destination
3. degree = scatter_sum(ones, dst, N)            // count neighbors
4. degree = clamp_min(degree, 1)                 // guard against isolated nodes (degree 0)
5. agg = agg / degree                            // mean = sum / count
6. combined = concat(x, agg)                     // [N, 2*D_in]
7. out = relu(linear(combined))                  // [N, D_out]
8. out = dropout(out, p)                         // only during training, NOT on final layer
9. if normalize: out = l2_normalize(out)         // optional, default off (PyG convention)
```

Alternatively, step 2-5 can be replaced by the existing `scatter_mean` from `sparse_ops.h` which already handles the zero-degree case internally.

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
| `dropout` | 0.5 | Dropout rate (training only) |
| `normalize` | false | L2-normalize output of each layer (paper=true, PyG=false) |

**Dropout placement:** Applied after ReLU activation for all SAGEConv layers except the final one. No dropout on classifier logits. This follows the standard PyG pattern (`ogbn_train.py`).

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
| `normalize` | BOOL | false | L2-normalize layer outputs (paper=true, PyG=false) |
| `randomSeed` | INT | -1 | Random seed for reproducibility (-1 = non-deterministic) |
| `exportEmbeddings` | BOOL | true | Generate embeddings.npy after training |

**Device placement:** All tensors are placed on `torch::kCPU` initially. GPU training is a future extension. For Cora and ogbn-arxiv, CPU training is sufficient.

### 5.3 YIELD fields

| Field | Type | Description |
|---|---|---|
| `modelName` | STRING | Model identifier |
| `ranEpochs` | INT | Epochs executed |
| `didConverge` | BOOL | Stopped by convergence or early stopping |
| `bestValAccuracy` | FLOAT | Best validation accuracy |
| `testAccuracy` | FLOAT | Test accuracy (evaluated at end, -1 if no labels) |
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
3. Open GnnMeta from projection directory
   Open LabelStore IF labels.bin exists (otherwise unsupervised mode — no loss, no accuracy)
   Open SplitStore IF splits.bin exists (otherwise all batches treated as TRAIN)
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

Stores only the **feature name** (not file paths). Readers reconstruct paths at read time via `db_folder + "/gnn_features/" + feature_name + ".fmat"`, consistent with how all existing GNN procedures locate features.

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
40      N     feature_name: char[N] (e.g., "node_features")
```

Total: 40 + N bytes (~55 bytes typical).

### 6.2 labels.bin

```
Offset  Size    Field
0       8       magic: "GNNL\0\0\0\0"
8       4       version: uint32 (1)
12      4       reserved: uint32
16      8       num_nodes: uint64
24      8       num_classes: uint64
32      N×8     labels: int64[N]  (-1 = unlabeled)
```

### 6.3 splits.bin

```
Offset  Size    Field
0       8       magic: "GNNS\0\0\0\0"
8       4       version: uint32 (1)
12      4       reserved: uint32
16      8       num_nodes: uint64
24      N×1     splits: uint8[N]  (0=TRAIN, 1=VAL, 2=TEST, 255=UNLABELED)
```

### 6.4 training_log.json

```json
{
  "version": 1,
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

**Total: 21 new files.**

### 7.2 Modified files

```
src/query/procedure/builtin/project_procedure.cc
src/graph_models/gql/projection/native_projection_builder.h
src/graph_models/gql/projection/native_projection_builder.cc
src/graph_models/gql/gql_model.cc                            (register gnn_train)
src/gnn/sampling/seed_selector.h                              (accept predefined splits)
src/gnn/sampling/seed_selector.cc
src/gnn/CMakeLists.txt                                        (add subdirectories)
```

**Total: 7 modified files.**

### 7.3 LOC estimates

| Component | Files | LOC | Complexity |
|---|---|---|---|
| graph_project extension | 3 modified + 1 new | ~180 | Low |
| SeedSelector predefined splits | 2 modified | ~30 | Low |
| LabelStore | 2 new | ~80 | Low |
| SplitStore | 2 new | ~60 | Low |
| GnnMeta | 1 new | ~80 | Low |
| MiniBatch | 1 new | ~30 | Trivial |
| BatchAssembler | 2 new | ~300 | Medium-High |
| TrainingLoop | 2 new | ~250 | Medium |
| GraphSAGE Model | 2 new | ~150 | Medium |
| NpyWriter | 2 new | ~80 | Low |
| gnn_train procedure | 2 new | ~180 | Medium |
| Tests | 5 new | ~400 | Medium |
| **Total** | **21 new + 7 modified** | **~1,820** | |

Note: BatchAssembler increased from 200 to 300 LOC due to the global index remapping logic (Section 3.5 step 3-5) and FourLevelStore fallback path.

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

-- 4. Sample mini-batches (usePredefinedSplits respects the splits from step 3)
CALL gnn_offline_sample('arxiv', 's1', [15, 10], {
    batchSize: 512, usePredefinedSplits: true
})
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
CALL gnn_offline_sample('cora', 's', [15, 10], {batchSize: 64, usePredefinedSplits: true})
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
