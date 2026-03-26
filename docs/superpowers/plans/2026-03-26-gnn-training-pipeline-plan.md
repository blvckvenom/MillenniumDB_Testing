# GNN Training Pipeline Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Enable end-to-end GraphSAGE MEAN training within MillenniumDB, from graph projection through to exported embeddings.

**Architecture:** Extend `graph_project` to emit GNN metadata (labels, splits, feature reference) during its existing node scan. Build a training pipeline (BatchAssembler → TrainingLoop → GraphSAGE) that consumes these files plus the existing FourLevelStore/SampleStorage infrastructure. All tensors on CPU; GPU is a future extension.

**Tech Stack:** C++17, LibTorch (torch::nn, torch::optim), GTest, existing mdb_gnn_core library, ANTLR4 (no grammar changes needed)

**Spec:** `docs/superpowers/specs/2026-03-26-gnn-training-pipeline-design.md`

---

## Phase A — Unblock Testing

These tasks produce the binary formats (labels.bin, splits.bin, gnn_meta.bin) and their readers. A temporary Python script generates test data so Phase B can proceed without waiting for Phase C.

---

### Task 1: GnnMeta reader/writer

**Files:**
- Create: `src/gnn/projection/gnn_meta.h`
- Test: `src/gnn/tests/test_label_store.cc` (will also test GnnMeta in Task 2)

The `GnnMeta` struct handles the `gnn_meta.bin` binary format (spec Section 6.1). Header-only since it's small.

- [ ] **Step 1: Write gnn_meta.h with read/write and round-trip test**

Format: magic("GNNM",8) + version(4) + feature_dim(4) + num_nodes(8) + num_classes(8) + has_labels(1) + has_splits(1) + reserved(2) + feature_name_len(4) + feature_name(N).

```cpp
// gnn_meta.h — key structures
struct GnnMeta {
    static constexpr char MAGIC[8] = {'G','N','N','M',0,0,0,0};
    static constexpr uint32_t VERSION = 1;

    std::string feature_name;   // e.g., "node_features"
    uint32_t feature_dim = 0;
    uint64_t num_nodes = 0;
    uint64_t num_classes = 0;
    bool has_labels = false;
    bool has_splits = false;

    static GnnMeta read(const std::filesystem::path& path);
    void write(const std::filesystem::path& path) const;
    static bool exists(const std::filesystem::path& dir);
};
```

Implement `write()`: open file, write magic, version, fields, string. Implement `read()`: open file, validate magic+version, read fields. Implement `exists()`: check `dir / "gnn_meta.bin"`.

Include a round-trip test in the same file:
```cpp
TEST(GnnMetaTest, WriteAndReadRoundtrip) {
    auto dir = fs::temp_directory_path() / "gnn_meta_test";
    fs::create_directories(dir);
    GnnMeta meta;
    meta.feature_name = "node_features";
    meta.feature_dim = 128;
    meta.num_nodes = 1000;
    meta.num_classes = 7;
    meta.has_labels = true;
    meta.has_splits = true;
    meta.write(dir / "gnn_meta.bin");

    auto loaded = GnnMeta::read(dir / "gnn_meta.bin");
    EXPECT_EQ(loaded.feature_name, "node_features");
    EXPECT_EQ(loaded.feature_dim, 128);
    EXPECT_EQ(loaded.num_nodes, 1000);
    EXPECT_EQ(loaded.num_classes, 7);
    EXPECT_TRUE(loaded.has_labels);
    EXPECT_TRUE(loaded.has_splits);
    fs::remove_all(dir);
}

TEST(GnnMetaTest, ExistsReturnsFalseWhenMissing) {
    EXPECT_FALSE(GnnMeta::exists("/tmp/nonexistent_dir_12345"));
}
```

- [ ] **Step 2: Run test to verify round-trip**

Run: `cd build/Debug && cmake --build . --target test_label_store -j$(nproc) && ctest -R LabelStore -V`
(GnnMeta tests are compiled into the same executable.)

- [ ] **Step 3: Commit**

```bash
git add -f src/gnn/projection/gnn_meta.h
git commit -m "feat(gnn): add GnnMeta reader/writer for projection GNN metadata"
```

---

### Task 2: LabelStore

**Files:**
- Create: `src/gnn/training/label_store.h`
- Create: `src/gnn/training/label_store.cc`
- Create: `src/gnn/tests/test_label_store.cc`
- Modify: `src/gnn/CMakeLists.txt` (add training/ sources + test executable)

LabelStore reads `labels.bin` (spec Section 6.2) via mmap. Format: magic("GNNL",8) + version(4) + reserved(4) + num_nodes(8) + num_classes(8) + data(N*8 int64).

- [ ] **Step 1: Add training/ directory to CMakeLists.txt**

In `src/gnn/CMakeLists.txt`, add `training/label_store.cc` to the `mdb_gnn_core` source list. Add test executable:
```cmake
add_executable(test_label_store tests/test_label_store.cc)
target_link_libraries(test_label_store mdb_gnn_core GTest::gtest_main)
set_target_properties(test_label_store PROPERTIES
    BUILD_RPATH "${CMAKE_SOURCE_DIR}/third_party/libtorch/lib"
    INSTALL_RPATH "${CMAKE_SOURCE_DIR}/third_party/libtorch/lib"
)
add_test(NAME LabelStoreTests COMMAND test_label_store)
```

- [ ] **Step 2: Write failing tests**

```cpp
// test_label_store.cc
#include "gnn/training/label_store.h"
#include "gnn/projection/gnn_meta.h"
#include "test_helpers.h"

using LabelStoreTest = mdb::gnn::test::GnnStorageTest;

TEST_F(LabelStoreTest, WriteAndReadRoundtrip) {
    auto path = test_path("labels.bin");
    // Write manually: 5 nodes, 3 classes, labels [0, 1, 2, -1, 0]
    LabelStore::write(path, {0, 1, 2, -1, 0}, 3);
    auto store = LabelStore::open(path);
    EXPECT_EQ(store.num_nodes(), 5);
    EXPECT_EQ(store.num_classes(), 3);
    EXPECT_EQ(store.get(0), 0);
    EXPECT_EQ(store.get(3), -1);
}

TEST_F(LabelStoreTest, GatherReturnsTensor) {
    auto path = test_path("labels.bin");
    LabelStore::write(path, {10, 20, 30, -1, 40}, 5);
    auto store = LabelStore::open(path);
    auto tensor = store.gather({0, 2, 3});
    EXPECT_EQ(tensor.size(0), 3);
    EXPECT_EQ(tensor[0].item<int64_t>(), 10);
    EXPECT_EQ(tensor[1].item<int64_t>(), 30);
    EXPECT_EQ(tensor[2].item<int64_t>(), -1);
}

TEST_F(LabelStoreTest, InvalidMagicThrows) {
    auto path = test_path("bad.bin");
    // Write garbage
    std::ofstream f(path, std::ios::binary);
    f.write("BADMAGIC", 8);
    f.close();
    EXPECT_THROW(LabelStore::open(path), std::runtime_error);
}
```

- [ ] **Step 3: Run tests to verify they fail**

Run: `cd build/Debug && cmake --build . --target test_label_store -j$(nproc) && ctest -R LabelStore -V`
Expected: FAIL — LabelStore class does not exist yet.

- [ ] **Step 4: Implement LabelStore**

```cpp
// label_store.h
class LabelStore {
public:
    static LabelStore open(const fs::path& path);
    static void write(const fs::path& path, const std::vector<int64_t>& labels, uint64_t num_classes);

    uint64_t num_nodes() const;
    uint64_t num_classes() const;
    int64_t get(uint64_t row_index) const;
    torch::Tensor gather(const std::vector<uint64_t>& row_indices) const;

    LabelStore(LabelStore&&) noexcept;
    LabelStore& operator=(LabelStore&&) noexcept;
    ~LabelStore();
private:
    LabelStore() = default;
    static constexpr char MAGIC[8] = {'G','N','N','L',0,0,0,0};
    static constexpr uint32_t VERSION = 1;
    static constexpr size_t HEADER_SIZE = 32; // 8+4+4+8+8

    void* mmap_ptr_ = nullptr;
    size_t mmap_size_ = 0;
    uint64_t num_nodes_ = 0;
    uint64_t num_classes_ = 0;
    const int64_t* data_ptr() const;
};
```

`open()`: open file, mmap read-only, validate magic+version, read num_nodes and num_classes from header.
`write()`: create file, write header, write int64 array.
`get()`: bounds check, return `data_ptr()[row_index]`.
`gather()`: allocate tensor `[B]` int64, fill from data_ptr.

- [ ] **Step 5: Run tests to verify they pass**

Run: `cd build/Debug && cmake --build . --target test_label_store -j$(nproc) && ctest -R LabelStore -V`
Expected: PASS (3 tests)

- [ ] **Step 6: Commit**

```bash
git add -f src/gnn/training/label_store.h src/gnn/training/label_store.cc \
    src/gnn/tests/test_label_store.cc src/gnn/CMakeLists.txt
git commit -m "feat(gnn): add LabelStore for reading labels.bin with mmap"
```

---

### Task 3: SplitStore

**Files:**
- Create: `src/gnn/training/split_store.h`
- Create: `src/gnn/training/split_store.cc`
- Create: `src/gnn/tests/test_split_store.cc`
- Modify: `src/gnn/CMakeLists.txt` (add sources + test)

Same pattern as LabelStore but for `splits.bin` (spec Section 6.3). Format: magic("GNNS",8) + version(4) + reserved(4) + num_nodes(8) + data(N*1 uint8).

- [ ] **Step 1: Write failing tests**

Tests: WriteAndReadRoundtrip, GatherMaskReturnsCorrectBools, ParsesSplitStrings, InvalidMagicThrows.

Key test: `gather_mask({0,1,2,3}, Split::TRAIN)` on data `[0,1,2,0]` returns `[true, false, false, true]`.

- [ ] **Step 2: Run tests — verify FAIL**

- [ ] **Step 3: Implement SplitStore**

```cpp
class SplitStore {
public:
    enum Split : uint8_t { TRAIN=0, VAL=1, TEST=2, UNLABELED=255 };

    static SplitStore open(const fs::path& path);
    static void write(const fs::path& path, const std::vector<uint8_t>& splits);
    static Split parse_split_string(const std::string& s);
    // "train"→0, "val"/"validation"→1, "test"→2, other→255

    uint64_t num_nodes() const;
    Split get(uint64_t row_index) const;
    torch::Tensor gather_mask(const std::vector<uint64_t>& row_indices, Split target) const;

    // Move-only (owns mmap)
    // ...
};
```

- [ ] **Step 4: Run tests — verify PASS**

- [ ] **Step 5: Commit**

```bash
git add -f src/gnn/training/split_store.h src/gnn/training/split_store.cc \
    src/gnn/tests/test_split_store.cc src/gnn/CMakeLists.txt
git commit -m "feat(gnn): add SplitStore for reading splits.bin with mmap"
```

---

### Task 4: Python script to generate test data

**Files:**
- Create: `scripts/generate_gnn_labels.py`

Temporary script that reads existing `_labels.npy` and generates `labels.bin` + `splits.bin` in the projection directory. This unblocks Phase B development.

- [ ] **Step 1: Write the script**

```python
#!/usr/bin/env python3
"""Generate labels.bin and splits.bin for GNN training.

Usage:
  python3 scripts/generate_gnn_labels.py <labels.npy> <output_dir> [--split-file splits.npy]

If no split file provided, uses predefined splits from the .gql properties
by assigning 70/15/15 random split.
"""
import struct, sys, numpy as np

def write_labels_bin(path, labels, num_classes):
    with open(path, 'wb') as f:
        f.write(b'GNNL\x00\x00\x00\x00')  # magic
        f.write(struct.pack('<I', 1))        # version
        f.write(struct.pack('<I', 0))        # reserved
        f.write(struct.pack('<Q', len(labels)))
        f.write(struct.pack('<Q', num_classes))
        f.write(labels.astype(np.int64).tobytes())

def write_splits_bin(path, splits):
    with open(path, 'wb') as f:
        f.write(b'GNNS\x00\x00\x00\x00')  # magic
        f.write(struct.pack('<I', 1))        # version
        f.write(struct.pack('<I', 0))        # reserved
        f.write(struct.pack('<Q', len(splits)))
        f.write(splits.astype(np.uint8).tobytes())
```

Also write `write_gnn_meta_bin()` to generate `gnn_meta.bin`.

- [ ] **Step 2: Test with Cora**

```bash
python3 scripts/generate_gnn_labels.py \
    data/example/gql/cora/cora_labels.npy \
    /tmp/test_gnn_labels/
# Verify files exist and sizes are correct
ls -la /tmp/test_gnn_labels/
```

- [ ] **Step 3: Commit**

```bash
git add scripts/generate_gnn_labels.py
git commit -m "chore(gnn): add temporary script to generate labels.bin and splits.bin"
```

---

## Phase B — Training Core

These tasks build the training pipeline components. They use the Python script from Task 4 for test data. Independent of Phase C.

---

### Task 5: MiniBatch struct

**Files:**
- Create: `src/gnn/training/mini_batch.h`

Header-only struct (spec Section 3.4).

- [ ] **Step 1: Write mini_batch.h**

```cpp
#pragma once
#include <cstdint>
#include <vector>
#include <torch/torch.h>
#include "gnn/sampling/graph_sample.h" // for SplitType

namespace mdb::gnn {

struct MiniBatch {
    torch::Tensor features;                        // [N_batch, D] float32
    std::vector<torch::Tensor> edge_indices;       // each [2, E_k] int64
    torch::Tensor labels;                          // [num_seeds] int64
    torch::Tensor label_mask;                      // [num_seeds] bool
    uint64_t num_seeds = 0;
    uint64_t num_nodes = 0;
    SplitType split = SplitType::TRAIN;
    uint64_t batch_id = 0;
};

} // namespace mdb::gnn
```

- [ ] **Step 2: Verify it compiles**

Add to CMakeLists.txt sources (header-only, but ensure include path works). Build:
```bash
cd build/Debug && cmake --build . --target mdb_gnn_core -j$(nproc)
```

- [ ] **Step 3: Commit**

```bash
git add -f src/gnn/training/mini_batch.h src/gnn/CMakeLists.txt
git commit -m "feat(gnn): add MiniBatch struct for training pipeline"
```

---

### Task 6: BatchAssembler

**Files:**
- Create: `src/gnn/training/batch_assembler.h`
- Create: `src/gnn/training/batch_assembler.cc`
- Create: `src/gnn/tests/test_batch_assembler.cc`
- Modify: `src/gnn/CMakeLists.txt`

The most complex component (~300 LOC). Unifies features + topology + labels + splits into MiniBatch (spec Section 3.5).

- [ ] **Step 1: Write batch_assembler.h interface**

```cpp
class BatchAssembler {
public:
    // Full mode: FourLevelStore + labels + splits
    BatchAssembler(
        FourLevelStore& feature_store,
        SampleStorage& samples,
        LabelStore* labels,            // nullable (unsupervised)
        SplitStore* splits,            // nullable
        const RowMapping& row_mapping
    );

    // Fallback mode: FeatureMatrix directly (no FourLevelStore)
    BatchAssembler(
        const FeatureMatrix& feature_matrix,
        SampleStorage& samples,
        LabelStore* labels,
        SplitStore* splits,
        const RowMapping& row_mapping
    );

    MiniBatch assemble(uint64_t batch_id);

private:
    // Internal: remap layer-local indices to global
    std::vector<torch::Tensor> build_edge_indices(
        const GraphSample& sample,
        const std::unordered_map<uint64_t, int64_t>& oid_to_global
    );

    // Internal: load features via FourLevelStore or FeatureMatrix fallback
    torch::Tensor load_features(
        const std::vector<ObjectId>& unique_nodes,
        uint64_t batch_id
    );
};
```

- [ ] **Step 2: Write failing tests**

Test with a small synthetic graph (5 nodes, 6 edges, 2 layers). Create FeatureMatrix, RowMapping, SampleStorage, LabelStore, SplitStore in temp directory. Verify:
- `mini.features.sizes() == {N_unique, D}`
- `mini.edge_indices.size() == num_layers`
- `mini.labels.size(0) == num_seeds`
- `mini.label_mask` correctly masks -1 labels
- Edge indices are correctly remapped to global

- [ ] **Step 3: Run tests — verify FAIL**

- [ ] **Step 4: Implement batch_assembler.cc**

Key logic in `assemble()`:
1. `auto sample = samples_.read_sample(batch_id);`
2. Build `oid_to_global` map from `sample.all_unique_nodes`
3. `auto features = load_features(sample.all_unique_nodes, batch_id);`
4. `auto edge_indices = build_edge_indices(sample, oid_to_global);`
5. Gather labels: translate seed OIDs → RowMapping indices → LabelStore::gather
6. Build label_mask: `labels != -1`
7. Package MiniBatch

Key in `build_edge_indices()`:
```cpp
for (int k = 0; k < num_layers; k++) {
    const auto& edges = sample.edges_per_layer[k];
    auto src_tensor = torch::empty({(int64_t)edges.size()}, torch::kInt64);
    auto dst_tensor = torch::empty({(int64_t)edges.size()}, torch::kInt64);
    auto src_acc = src_tensor.accessor<int64_t, 1>();
    auto dst_acc = dst_tensor.accessor<int64_t, 1>();
    for (size_t i = 0; i < edges.size(); i++) {
        auto src_oid = sample.nodes_per_layer[k+1][edges.src_indices[i]];
        auto dst_oid = sample.nodes_per_layer[k][edges.dst_indices[i]];
        src_acc[i] = oid_to_global.at(src_oid.id);
        dst_acc[i] = oid_to_global.at(dst_oid.id);
    }
    result.push_back(torch::stack({src_tensor, dst_tensor}));
}
```

- [ ] **Step 5: Run tests — verify PASS**

- [ ] **Step 6: Commit**

```bash
git add -f src/gnn/training/batch_assembler.h src/gnn/training/batch_assembler.cc \
    src/gnn/tests/test_batch_assembler.cc src/gnn/CMakeLists.txt
git commit -m "feat(gnn): add BatchAssembler to unify features, topology, labels into MiniBatch"
```

---

### Task 7: GraphSAGE MEAN model

**Files:**
- Create: `src/gnn/models/graphsage_model.h`
- Create: `src/gnn/models/graphsage_model.cc`
- Create: `src/gnn/tests/test_graphsage_model.cc`
- Modify: `src/gnn/CMakeLists.txt`

Implements spec Section 4. Uses `scatter_sum` from `sparse_ops.h`.

- [ ] **Step 1: Write graphsage_model.h**

```cpp
struct GraphSAGEConfig {
    int64_t input_dim;
    int64_t hidden_dim = 256;
    int64_t num_classes;
    int64_t num_layers;
    double dropout = 0.5;
    bool normalize = false;  // L2 normalize (default off, PyG convention)
};

class GraphSAGEModel : public torch::nn::Module {
public:
    explicit GraphSAGEModel(const GraphSAGEConfig& config);

    // Returns logits [num_seeds, num_classes] for seed nodes only
    torch::Tensor forward(
        torch::Tensor x,                              // [N, D]
        const std::vector<torch::Tensor>& edge_indices, // per-layer [2, E_k]
        int64_t num_seeds                              // how many seed nodes
    );

private:
    torch::Tensor sage_conv(
        torch::Tensor x,
        torch::Tensor edge_index,
        torch::nn::Linear& linear
    );

    std::vector<torch::nn::Linear> convs_;
    torch::nn::Linear classifier_{nullptr};
    GraphSAGEConfig config_;
};
```

- [ ] **Step 2: Write failing tests**

Tests:
- `ForwardProducesCorrectShape`: 10 nodes, 2 layers, verify output shape `[num_seeds, num_classes]`
- `GradientsFlow`: verify `loss.backward()` produces non-zero gradients on all parameters
- `DropoutDiffers`: `model.train()` vs `model.eval()` produce different outputs (dropout effect)

- [ ] **Step 3: Run tests — verify FAIL**

- [ ] **Step 4: Implement graphsage_model.cc**

`sage_conv()` implements spec Section 4.1 steps 1-9:
```cpp
torch::Tensor GraphSAGEModel::sage_conv(
    torch::Tensor x, torch::Tensor edge_index, torch::nn::Linear& linear) {
    auto src = edge_index[0];  // [E]
    auto dst = edge_index[1];  // [E]
    int64_t N = x.size(0);

    auto neighbor_feat = x.index_select(0, src);
    auto agg = ops::scatter_sum(neighbor_feat, dst, N);    // mdb::gnn::ops namespace
    auto ones = torch::ones({src.size(0), 1}, x.options());
    auto degree = ops::scatter_sum(ones, dst, N).clamp_min(1.0);
    agg = agg / degree;

    auto combined = torch::cat({x, agg}, 1);
    return linear->forward(combined);
}
```

`forward()` iterates layers from outside in:
```cpp
torch::Tensor GraphSAGEModel::forward(
    torch::Tensor x, const std::vector<torch::Tensor>& edge_indices, int64_t num_seeds) {
    for (int k = (int)convs_.size() - 1; k >= 0; k--) {
        x = sage_conv(x, edge_indices[k], convs_[k]);
        if (k > 0) {  // not final conv layer
            x = torch::relu(x);
            if (is_training()) x = torch::dropout(x, config_.dropout, true);
            if (config_.normalize)
                x = x / x.norm(2, 1, true).clamp_min(1e-6);
        }
    }
    // Classify only seed nodes
    auto seed_embeddings = x.slice(0, 0, num_seeds);
    return classifier_->forward(seed_embeddings);
}
```

- [ ] **Step 5: Run tests — verify PASS**

- [ ] **Step 6: Commit**

```bash
git add -f src/gnn/models/graphsage_model.h src/gnn/models/graphsage_model.cc \
    src/gnn/tests/test_graphsage_model.cc src/gnn/CMakeLists.txt
git commit -m "feat(gnn): add GraphSAGE MEAN model with configurable layers and dropout"
```

---

### Task 8: NpyWriter

**Files:**
- Create: `src/gnn/training/npy_writer.h`
- Create: `src/gnn/training/npy_writer.cc`

Writes tensors as NumPy `.npy` files for external validation. Placed near the existing `src/import/npy_loader.h` conceptually but in the gnn module since it's GNN-specific output.

- [ ] **Step 1: Implement NpyWriter**

```cpp
class NpyWriter {
public:
    // Write a 2D float32 tensor as .npy
    static void write_float32(const fs::path& path, const torch::Tensor& tensor);
    // Write a 1D int64 tensor as .npy
    static void write_int64(const fs::path& path, const torch::Tensor& tensor);
};
```

NumPy v1.0 format: magic `\x93NUMPY` + major(1) + minor(0) + header_len(2) + header_str + data. Header string: `{'descr': '<f4', 'fortran_order': False, 'shape': (N, D), }` padded to 64-byte alignment.

- [ ] **Step 2: Test round-trip with NumPy**

Write a tensor, load it in Python, verify values match. Can be a manual test or a scripted check.

- [ ] **Step 3: Commit**

```bash
git add -f src/gnn/training/npy_writer.h src/gnn/training/npy_writer.cc src/gnn/CMakeLists.txt
git commit -m "feat(gnn): add NpyWriter for exporting tensors as NumPy .npy files"
```

---

### Task 9: TrainingLoop

**Files:**
- Create: `src/gnn/training/training_loop.h`
- Create: `src/gnn/training/training_loop.cc`
- Create: `src/gnn/tests/test_training_loop.cc`
- Modify: `src/gnn/CMakeLists.txt`

Implements spec Section 3.6. Epoch iteration with early stopping and validation.

- [ ] **Step 1: Write training_loop.h**

```cpp
class TrainingLoop {
public:
    struct Config {
        uint64_t epochs = 50;
        double learning_rate = 0.01;
        double weight_decay = 0.0;
        double tolerance = 1e-4;
        uint64_t patience = 5;
        int64_t random_seed = -1;
        std::string output_dir;
    };

    struct Result {
        uint64_t ran_epochs = 0;
        bool converged = false;
        double best_val_accuracy = 0.0;
        std::vector<double> epoch_losses;
        double train_seconds = 0.0;
    };

    // Note: spec uses torch::nn::Module& for generality. We use GraphSAGEModel&
    // since it's the only model for now. If GCN/GAT/GIN are added later, refactor
    // to a GnnModel base class or use torch::nn::Module& + dynamic_cast.
    TrainingLoop(
        GraphSAGEModel& model,
        BatchAssembler& assembler,
        const SampleCatalog& catalog,
        Config config
    );

    Result train();

    // Evaluate accuracy on a set of batches
    double evaluate(uint64_t start_batch, uint64_t count);

private:
    GraphSAGEModel& model_;
    BatchAssembler& assembler_;
    const SampleCatalog& catalog_;
    Config config_;
};
```

- [ ] **Step 2: Write failing tests**

Test with a tiny synthetic dataset (10 nodes, known labels). Verify:
- `LossDecreases`: train for 5 epochs, `epoch_losses[4] < epoch_losses[0]`
- `EarlyStopsOnConvergence`: with tolerance=999, stops after 2 epochs
- `PatienceWorks`: validation accuracy plateaus → stops after patience epochs
- `RandomSeedReproducible`: two runs with same seed produce same losses

Note: these tests will need a small BatchAssembler setup. Use the FeatureMatrix fallback path (no FourLevelStore needed).

- [ ] **Step 3: Run tests — verify FAIL**

- [ ] **Step 4: Implement training_loop.cc**

Key: `train()` follows spec Section 3.6 pseudocode exactly. Use `torch::optim::Adam`. Compute `cross_entropy` only on nodes with `label_mask == true`. Track `best_val_accuracy` for early stopping. Save checkpoint via `torch::save(model, path)` when validation improves.

- [ ] **Step 5: Run tests — verify PASS**

- [ ] **Step 6: Commit**

```bash
git add -f src/gnn/training/training_loop.h src/gnn/training/training_loop.cc \
    src/gnn/tests/test_training_loop.cc src/gnn/CMakeLists.txt
git commit -m "feat(gnn): add TrainingLoop with early stopping and validation"
```

---

## Phase C — graph_project Extension

Independent of Phase B. Extends `graph_project` to produce `gnn_meta.bin`, `labels.bin`, `splits.bin`.

---

### Task 10: Parse new configuration fields

**Files:**
- Modify: `src/query/procedure/builtin/project_procedure.cc`

Add parsing of `includeFeatures`, `labelProperty`, `splitProperty` from the 4th argument Map, alongside existing `nodeProperties`, `orientation`, etc.

- [ ] **Step 1: Add field parsing in project_procedure.cc**

In the `if (config_dict)` block (around line 128), after existing field parsing:
```cpp
std::string include_features;
std::string label_property;
std::string split_property;

if (config_dict) {
    // ... existing parsing ...

    // GNN fields (optional)
    if (auto v = get_string_from_dict(config_dict, "includeFeatures")) {
        include_features = *v;
    }
    if (auto v = get_string_from_dict(config_dict, "labelProperty")) {
        label_property = *v;
    }
    if (auto v = get_string_from_dict(config_dict, "splitProperty")) {
        split_property = *v;
    }
}
```

Validate `includeFeatures` against catalog:
```cpp
if (!include_features.empty()) {
    const auto& names = gql_model.catalog.gnn_feature_names;
    if (std::find(names.begin(), names.end(), include_features) == names.end()) {
        throw std::runtime_error(
            format_not_found_error("feature", include_features, names,
                "Import with: mdb import data.gql <db> --with-tensors features.npy"));
    }
}
```

Pass these to `NativeProjectionBuilder` (requires adding parameters to its constructor).

- [ ] **Step 2: Add new YIELD fields**

After existing yields, add:
```cpp
ctx.yield("featureDim", ctx.create_int(static_cast<int64_t>(stats.feature_dim)));
ctx.yield("numClasses", ctx.create_int(static_cast<int64_t>(stats.num_classes)));
```

- [ ] **Step 3: Verify existing tests still pass**

```bash
cd build/Debug && cmake --build . -j$(nproc) && ctest -R ProjectionTests -V
```

- [ ] **Step 4: Commit**

```bash
git add -f src/query/procedure/builtin/project_procedure.cc
git commit -m "feat(gnn): parse includeFeatures, labelProperty, splitProperty in graph_project"
```

---

### Task 11: Labels/splits writers in NativeProjectionBuilder

**Files:**
- Modify: `src/graph_models/gql/projection/native_projection_builder.h`
- Modify: `src/graph_models/gql/projection/native_projection_builder.cc`

Add GNN data extraction during the existing node scan (spec Section 2.5).

- [ ] **Step 1: Add GNN fields to NativeProjectionBuilder**

In the header, add members:
```cpp
// GNN data extraction (optional)
std::string include_features_;
std::string label_property_;
std::string split_property_;
std::unique_ptr<RowMapping> gnn_row_mapping_;
std::vector<int64_t> labels_buffer_;
std::vector<uint8_t> splits_buffer_;
std::unordered_set<int64_t> unique_classes_;
uint32_t feature_dim_ = 0;
```

- [ ] **Step 2: Initialize GNN buffers in constructor**

If `include_features` is non-empty:
```cpp
auto rmap_path = fs::path(db_folder) / "gnn_features" / (include_features + ".rmap");
gnn_row_mapping_ = std::make_unique<RowMapping>(RowMapping::open(rmap_path));
auto fmat_path = fs::path(db_folder) / "gnn_features" / (include_features + ".fmat");
auto fm = FeatureMatrix::open(fmat_path);
feature_dim_ = fm.num_cols();

if (!label_property.empty()) {
    labels_buffer_.resize(gnn_row_mapping_->size(), -1);
}
if (!split_property.empty()) {
    splits_buffer_.resize(gnn_row_mapping_->size(), 255);
}
```

- [ ] **Step 3: Add extraction in extract_node_properties()**

Inside the property loop, after `storage->add_node_property(...)`:
```cpp
if (gnn_row_mapping_) {
    auto row_opt = gnn_row_mapping_->find(node_id);
    if (row_opt.has_value()) {
        uint64_t row = *row_opt;
        if (source_key_name == label_property_ && !label_property_.empty()) {
            labels_buffer_[row] = to_int64(value_id);
            unique_classes_.insert(labels_buffer_[row]);
        }
        if (source_key_name == split_property_ && !split_property_.empty()) {
            splits_buffer_[row] = SplitStore::parse_split_string(
                unpack_string_value(value_id));
        }
    }
}
```

- [ ] **Step 4: Write files in finalize()**

After existing B+Tree finalization:
```cpp
if (!include_features_.empty()) {
    GnnMeta meta;
    meta.feature_name = include_features_;
    meta.feature_dim = feature_dim_;
    meta.num_nodes = gnn_row_mapping_->size();
    meta.num_classes = unique_classes_.size();
    meta.has_labels = !label_property_.empty();
    meta.has_splits = !split_property_.empty();
    meta.write(fs::path(projection_dir) / "gnn_meta.bin");
}
if (!label_property_.empty() && !labels_buffer_.empty()) {
    LabelStore::write(
        fs::path(projection_dir) / "labels.bin",
        labels_buffer_, unique_classes_.size());
}
if (!split_property_.empty() && !splits_buffer_.empty()) {
    SplitStore::write(
        fs::path(projection_dir) / "splits.bin",
        splits_buffer_);
}
```

- [ ] **Step 5: Run existing projection tests**

```bash
cd build/Debug && cmake --build . -j$(nproc) && ctest -R Projection -V
```
Verify no regressions — existing tests don't use the new fields.

- [ ] **Step 6: Commit**

```bash
git add -f src/graph_models/gql/projection/native_projection_builder.h \
    src/graph_models/gql/projection/native_projection_builder.cc
git commit -m "feat(gnn): extract labels and splits during graph_project node scan"
```

---

### Task 12: SeedSelector predefined splits

**Files:**
- Modify: `src/gnn/sampling/seed_selector.h`
- Modify: `src/gnn/sampling/seed_selector.cc`
- Modify: `src/gnn/sampling/sampling_config.h` (add `use_predefined_splits` field)
- Modify: `src/query/procedure/builtin/gnn_offline_sample_procedure.cc` (parse option)

Add `usePredefinedSplits` support (spec Section 2.8). When enabled, SeedSelector reads `splits.bin` instead of using ratio-based random splitting.

- [ ] **Step 1: Add predefined splits to SamplingConfig**

In `sampling_config.h`:
```cpp
bool use_predefined_splits = false;  ///< Use splits.bin from projection
```

- [ ] **Step 2: Modify SeedSelector::Impl to accept external splits**

In `seed_selector.cc`, inside the `Impl` class, modify `get_seed_split()`:
```cpp
if (config_.use_predefined_splits) {
    // Load splits.bin from projection directory
    auto proj_dir = ProjectionManager::get_instance()
        .get_projection_dir(config_.projection_name);
    auto store = SplitStore::open(fs::path(proj_dir) / "splits.bin");

    for (auto& oid : all_seeds) {
        auto row = row_mapping_->find(oid);
        if (!row) continue;
        auto split = store.get(*row);
        switch (split) {
            case SplitStore::TRAIN: result.train_seeds.push_back(oid); break;
            case SplitStore::VAL:   result.validation_seeds.push_back(oid); break;
            case SplitStore::TEST:  result.test_seeds.push_back(oid); break;
            default: break; // skip UNLABELED nodes
        }
    }
} else {
    // Existing ratio-based splitting logic
}
```

- [ ] **Step 3: Parse option in gnn_offline_sample_procedure.cc**

Add in the options parsing block:
```cpp
if (auto v = opts.get_bool("usePredefinedSplits")) {
    config.use_predefined_splits = *v;
}
```

- [ ] **Step 4: Verify existing sampling tests still pass**

```bash
cd build/Debug && cmake --build . -j$(nproc) && ctest -R Sampling -V
```

- [ ] **Step 5: Commit**

```bash
git add -f src/gnn/sampling/seed_selector.h src/gnn/sampling/seed_selector.cc \
    src/gnn/sampling/sampling_config.h \
    src/query/procedure/builtin/gnn_offline_sample_procedure.cc
git commit -m "feat(gnn): add usePredefinedSplits option to gnn_offline_sample"
```

---

### Task 13: Integration test for graph_project GNN extension

**Files:**
- Create: `tests/gql/test_suites/projection_gnn/` (test suite directory)

End-to-end test: import Cora, project with GNN fields, verify labels.bin + splits.bin + gnn_meta.bin are correct.

- [ ] **Step 1: Write integration test**

Create test files that:
1. Import Cora with `--with-tensors`
2. Run `graph_project('cora', ':Paper', ':CITES', {includeFeatures: 'node_features', labelProperty: 'label', splitProperty: 'split', orientation: 'UNDIRECTED'})`
3. Verify YIELD fields: `featureDim == 1433`, `numClasses == 7`
4. Verify files exist in projection directory

- [ ] **Step 2: Run integration test**

```bash
./scripts/run-tests gql
```

- [ ] **Step 3: Commit**

```bash
git add tests/gql/test_suites/projection_gnn/
git commit -m "test(gnn): add integration tests for graph_project GNN extension"
```

---

## Phase D — Integration

Depends on Phases B and C completing.

---

### Task 14: gnn_train procedure

**Files:**
- Create: `src/query/procedure/builtin/gnn_train_procedure.h`
- Create: `src/query/procedure/builtin/gnn_train_procedure.cc`
- Modify: `src/graph_models/gql/gql_model.cc` (register procedure)
- Modify: `src/gnn/CMakeLists.txt` (if needed)

The orchestrator procedure (spec Section 5).

- [ ] **Step 1: Write gnn_train_procedure.h**

Follow the pattern of existing GNN procedures (e.g., `gnn_materialize_batches_procedure.h`):
```cpp
namespace GQL::Procedures {
class GnnTrainProcedure : public Procedure {
public:
    void execute(ProcedureContext& ctx) override;
    std::vector<YieldColumn> get_yield_columns() const override;
};
}
```

- [ ] **Step 2: Implement execute()**

Follow spec Section 5.5 steps 1-11. Key logic:
1. Parse sampleName, featureName, options (hiddenDim, lr, epochs, etc.)
2. Open SampleStorage → get projection_name
3. Open GnnMeta, conditionally open LabelStore/SplitStore
4. Open FeatureMatrix + try FourLevelStore (fallback to FM)
5. Infer num_layers from fanouts
6. Create GraphSAGEModel
7. Create BatchAssembler
8. Run TrainingLoop
9. Evaluate test accuracy
10. Export model + embeddings + training_log.json
11. Yield results

- [ ] **Step 3: Register in gql_model.cc**

Inside `#ifdef ENABLE_GNN`:
```cpp
#include "query/procedure/builtin/gnn_train_procedure.h"
// ...
catalog.register_procedure("gnn_train",
    std::make_unique<GQL::Procedures::GnnTrainProcedure>());
```

The export step (step 10 in spec Section 5.5) must:
- Save model: `torch::save(model, output_dir / "graphsage_model.pt")`
- Save embeddings: use `NpyWriter::write_float32(output_dir / "embeddings.npy", all_embeddings)`
- Save node IDs: use `NpyWriter::write_int64(output_dir / "node_ids.npy", all_node_ids)`
- Save training log: write `training_log.json` via simple string formatting (no JSON library needed — flat structure, use `std::ofstream` with manual formatting matching spec Section 6.4)
- Create output directory: `fs::create_directories(output_dir)` before writing

Test accuracy computation: after `train()` returns, call `loop.evaluate(catalog.train_batches + catalog.validation_batches, catalog.test_batches)` on test batch range, similar to how validation is computed inside the training loop.

- [ ] **Step 4: Build and verify compilation**

```bash
cd build/Debug && cmake --build . -j$(nproc)
```

- [ ] **Step 5: Commit**

```bash
git add -f src/query/procedure/builtin/gnn_train_procedure.h \
    src/query/procedure/builtin/gnn_train_procedure.cc \
    src/graph_models/gql/gql_model.cc src/gnn/CMakeLists.txt
git commit -m "feat(gnn): add gnn_train GQL procedure for GraphSAGE training"
```

---

### Task 15: End-to-end test with Cora

**Files:**
- Create: `tests/gql/test_suites/gnn_training/` (test suite directory)

The ultimate validation: import Cora → project → sample → train → verify accuracy > 70%.

- [ ] **Step 1: Write E2E test script**

Test flow:
```gql
CALL graph_project('cora', ':Paper', ':CITES', {
    orientation: 'UNDIRECTED',
    includeFeatures: 'node_features',
    labelProperty: 'label',
    splitProperty: 'split'
})
YIELD graphName, nodeCount, featureDim, numClasses
RETURN *
-- Expected: nodeCount=2708, featureDim=1433, numClasses=7
```

```gql
CALL gnn_offline_sample('cora', 'cora_s', [15, 10], {
    batchSize: 64, usePredefinedSplits: true
})
YIELD totalBatches RETURN *
```

```gql
CALL gnn_train('cora_s', 'node_features', {
    hiddenDim: 128, epochs: 30, lr: 0.01, dropout: 0.5,
    randomSeed: 42, exportEmbeddings: true
})
YIELD bestValAccuracy, testAccuracy, ranEpochs, didConverge
RETURN *
-- Expected: testAccuracy > 0.70
```

- [ ] **Step 2: Run E2E test**

```bash
./scripts/run-tests gql
```

- [ ] **Step 3: Verify exported embeddings externally**

```bash
python3 -c "
import numpy as np
emb = np.load('data/dbs/gql/cora_test/projections/cora/gnn_output/default/embeddings.npy')
ids = np.load('data/dbs/gql/cora_test/projections/cora/gnn_output/default/node_ids.npy')
print(f'Embeddings: {emb.shape}, IDs: {ids.shape}')
assert emb.shape[0] == 2708
assert emb.shape[1] == 128  # hidden_dim
"
```

- [ ] **Step 4: Commit**

```bash
git add tests/gql/test_suites/gnn_training/
git commit -m "test(gnn): add end-to-end training test with Cora dataset"
```

---

### Task 16: Remove temporary Python script

**Files:**
- Delete: `scripts/generate_gnn_labels.py`

Now that `graph_project` generates labels.bin and splits.bin natively, the temporary script is no longer needed.

- [ ] **Step 1: Remove script**

```bash
git rm scripts/generate_gnn_labels.py
git commit -m "chore(gnn): remove temporary label/split generation script"
```

---

## Cumulative CMakeLists.txt Changes

All `.cc` files added to `mdb_gnn_core` sources in `src/gnn/CMakeLists.txt`:

```cmake
# Training pipeline (Phase B)
training/label_store.cc          # Task 2
training/split_store.cc          # Task 3
training/batch_assembler.cc      # Task 6
training/npy_writer.cc           # Task 8
training/training_loop.cc        # Task 9

# Models (Phase B)
models/graphsage_model.cc        # Task 7

# Procedure (Phase D) — in the existing procedure source list section
${CMAKE_SOURCE_DIR}/src/query/procedure/builtin/gnn_train_procedure.cc  # Task 14
```

Test executables — note linkage requirements:

```cmake
# Simple tests (only mdb_gnn_core):
test_label_store, test_split_store, test_graphsage_model

# Tests needing SampleStorage (heavier linkage):
add_executable(test_batch_assembler tests/test_batch_assembler.cc)
target_link_libraries(test_batch_assembler
    GTest::gtest_main
    mdb_gnn_core
    millenniumdb
    antlr4_cpp_runtime
    ssl crypto
    ${TORCH_LIBRARIES}
)

# Same heavy linkage for test_training_loop
```

---

## Summary

| Phase | Tasks | Dependencies | Estimated LOC |
|---|---|---|---|
| A (Unblock) | 1-4 | None | ~320 |
| B (Training Core) | 5-9 | Phase A | ~1,060 |
| C (graph_project) | 10-13 | Phase A (for LabelStore/SplitStore writers) | ~340 |
| D (Integration) | 14-16 | Phases B + C | ~380 |
| **Total** | **16 tasks** | | **~2,100** |

**Parallelism:** Phases B and C are independent after Phase A completes. Phase D requires both.
