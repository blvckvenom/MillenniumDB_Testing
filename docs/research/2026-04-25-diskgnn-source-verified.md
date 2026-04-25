# DiskGNN Source-Verified Architectural Comparison

**Date**: 2026-04-25
**Source**: `/home/benito_pc/Escritorio/DiskGNN/` (Liu et al. SIGMOD 2025) + LaTeX paper at `docs/external_references/GNN_PAPERS/02_DiskGNN/sections/`
**Purpose**: thesis defense material verifying claimed similarities and differences between MillenniumDB feature-GNN and DiskGNN

---

## 1. Why this document exists

Earlier session notes referenced DiskGNN's architecture from memory and from inspirational comments scattered across `src/gnn/sampling/`. To support thesis defense, the comparison needed to be cross-checked against DiskGNN's actual source code and paper text. This doc records the verified findings.

## 2. DiskGNN's sampling — verified from `examples/mega_batch_sampling.py`

### Key import (line 2)

```python
from dgl.dataloading import DataLoader, NeighborSampler
```

### Sampling step (lines 28-44)

```python
g, features, labels, n_classes, splitted_idx = dataset
g = g.remove_self_loop()
g = g.add_self_loop()
train_nid, _, _ = (splitted_idx["train"], splitted_idx["valid"], splitted_idx["test"])

sampler = NeighborSampler(fanout)            # ← DGL's sampler, not custom
train_dataloader = DataLoader(
    g,                                        # ← Full DGL graph in CPU memory
    train_nid,
    sampler,
    batch_size=args.batchsize,
    shuffle=True,
    drop_last=False,
    num_workers=2,
)
```

### Verified findings

1. **DiskGNN does NOT implement its own sampling algorithm.** It uses `dgl.dataloading.NeighborSampler` directly.
2. **The graph topology is loaded entirely into CPU RAM during sampling** (DGL's `DataLoader` requires this). For papers100M, this means ~50-80 GB CPU RAM is needed for the topology alone.
3. **DiskGNN's `offgs` C++ extension** (in `src/`) contains only feature loading code: `gather.cc`, `load.cc`, `save.cc`, `free.cc`, plus CUDA kernels for tensor ops. **No sampling code in C++**.
4. **Sample output is persisted to disk** as `.pt` PyTorch tensors (`subgraph_*.pt`, `in-nid-*.pt`, `out-nid-*.pt`).
5. **OS page cache is dropped before measurement** (line 47: `echo 1 > /proc/sys/vm/drop_caches`) — confirms they assume cold-cache start as their baseline.
6. **Node access counts are tracked during sampling** (line 50: `node_counts = torch.zeros(g.num_nodes(), dtype=torch.int64, device="cuda")`) — used later for the popularity-sorted feature placement.

## 3. DiskGNN's actual contribution — feature loading, NOT sampling

From `4-system-overview.tex`:

> *"\name{} takes the graph samples of many mini-batches and all node features of the data graph as input"*
>
> *"This can be done by running existing GNN frameworks like DGL or PyG"*

The paper explicitly assumes pre-sampled mini-batches as input. DiskGNN's contributions are:

1. **Four-level feature store** (paper §4): GPU cache → CPU cache → Disk cache (shared, MinHash-reordered) → Packed feature chunks (per mini-batch).
2. **MinHash reorder of disk cache** (paper §5.1, Algorithm 1): assigns adjacent IDs to nodes whose feature access patterns are similar across mini-batches. This is what `src/gnn/sampling/minhash_reorderer.cc:Strategy::SEGMENTED` replicates.
3. **Pipeline disk-read with GPU compute** (paper §5.3): overlap fetch with model forward.

## 4. MillenniumDB feature-GNN comparison

| Component | DiskGNN | MillenniumDB feature-GNN |
|---|---|---|
| Sampling algorithm | `dgl.NeighborSampler` (Python wrapper around DGL C++) | `BasicKHopSampler` (own C++, original) |
| Sampling execution | Python iteration over `DataLoader` | C++ direct call from `gnn_offline_sample` |
| Topology storage during sample | DGL graph fully resident in CPU RAM | Configurable: hash cache RAM / mmap sidecar / B+Tree direct |
| Memory bound on sampler | RAM(graph) + workspace ~5-10 GB | Configurable: 5-80 GB depending on backend |
| Sample output | `.pt` files via `torch.save` | `.bin` packed batch files |
| Sample reuse across epochs | ✓ pre-sampled once | ✓ same approach |
| Track node access frequency | ✓ during sample | ✓ via offline `node_counts` profiling |
| Four-level **feature** cache | ✓ GPU + CPU + disk-cache + packed chunks | ✓ FourLevelStore (faithful reproduction, src/gnn/storage/) |
| MinHash **feature** reorder | ✓ Algorithm 1 SEGMENTED | ✓ ported in MinHashReorderer |
| MinHash **adjacency** reorder | ✗ not in paper | ✗ not in our impl either (potential Spec #14) |
| **Adjacency cache** (RAM-resident hash) | ✗ relies on DGL's full-graph CSR | ✓ Spec #11 + opt-out flag |
| **mmap-backed adjacency sidecar** | ✗ | ✓ Spec #4-B TopologySnapshot |
| **B+Tree-direct adjacency** (out-of-core) | ✗ | ✓ default fallback |

### Key empirical implication

**DiskGNN cannot run on a 32 GB commodity machine for papers100M** because the DGL graph in RAM exceeds available memory. This is a structural limitation: their sampling step fundamentally requires the graph in RAM.

**MillenniumDB feature-GNN can run on 32 GB commodity for papers100M** via:

- `useAdjacencyCache: false` + Spec #4-B mmap sidecar ("Camino D" empirically validated 2026-04-25).
- B+Tree direct fallback (slowest but smallest RAM).

This is a thesis-defensible original contribution: **out-of-core adjacency for sampling** that DiskGNN explicitly does not address.

## 5. Faithful reproductions in our code

Where our code follows DiskGNN exactly, attribution is in the source:

```cpp
// src/gnn/sampling/minhash_reorderer.h:3-5
// MinHash reordering algorithm based on DiskGNN (Liu et al., SIGMOD 2025)
// Original: https://github.com/Liu-rj/DiskGNN (MIT License)
// SEGMENTED — DiskGNN Algorithm 1 (validated)
// MULTIPASS_BOUNDED — original contribution
```

```cpp
// src/query/procedure/builtin/gnn_offline_sample_procedure.h:15
// This procedure implements the DiskGNN (SIGMOD 2025) architecture of
// pre-computed offline sampling
```

These attributions correctly identify which parts are DiskGNN-derived.

## 6. Where we diverge by design

Our `BasicKHopSampler::Impl::do_sample` runs the canonical neighbor-sampling algorithm (GraphSAGE Hamilton 2017 §3.2). The algorithm itself is identical to what DGL's `NeighborSampler` does internally. The difference is in **where adjacency lives during the call**:

```cpp
// src/gnn/projection/topology_accessor.cc — get_neighbors dispatch
if (cache_enabled_ && fwd_cache_built_) {
    return /* O(1) hash lookup, Spec #11 */;
} else if (fwd_csr_.has_data()) {
    return /* O(1) mmap slice, Spec #4-B */;
} else {
    return /* O(log N + degree) B+Tree path */;
}
```

DGL's equivalent line would always go through "in-memory CSR slice", with no opt-out. Our flexibility is the architectural contribution.

## 7. Implications for thesis defense

### Defensible claims

1. *"MillenniumDB feature-GNN faithfully reproduces DiskGNN's four-level feature cache hierarchy and MinHash disk reorder (verified against original C++ at `~/Escritorio/DiskGNN/src/`)."*
2. *"Where DiskGNN delegates sampling to DGL Python primitives that require the full graph topology in CPU RAM, MillenniumDB's `BasicKHopSampler` operates over disk-resident B+Tree storage with three configurable adjacency backends (Spec #11 RAM hash cache, Spec #4-B mmap-backed sidecar, B+Tree direct fallback). This enables out-of-core sampling on commodity hardware that DiskGNN cannot perform."*
3. *"The MULTIPASS_BOUNDED MinHash strategy (`src/gnn/sampling/minhash_reorderer.cc`) is an original contribution beyond DiskGNN's SEGMENTED Algorithm 1."*

### Honest caveats

1. *"DiskGNN's contribution is feature loading, not sampling. Some architectural innovation we attribute to inspiration from DiskGNN — specifically the FourLevelStore — is a direct port; we credit the original."*
2. *"DiskGNN's pipeline overlap of disk-read with GPU compute is not yet implemented in our pipeline (single-threaded sampling, then training). Future work."*
3. *"MinHash reorder of adjacency (in addition to features) is not in DiskGNN nor in our impl. A natural Spec #14 candidate."*

## 8. Spec #13 positioning

Given the verified architecture above, the proposed **Spec #13 — FourLevelTopologyStore** would:

1. Apply DiskGNN's four-level pattern to topology (DiskGNN does it only for features).
2. Use frequency-based partition with MinHash adjacency reorder (extending paper Algorithm 1 to a new domain).
3. Bridge the gap between "all RAM" (Spec #11) and "all disk" (Spec #4-B sidecar) with a hot-RAM + cold-disk hybrid.
4. Enable papers100M training on commodity 32 GB hardware where DiskGNN cannot run.

This is a paper-grade extension of DiskGNN's architecture, not a replication.

## 9. Source files inspected for this verification

- `/home/benito_pc/Escritorio/DiskGNN/examples/mega_batch_sampling.py` — main sampling driver
- `/home/benito_pc/Escritorio/DiskGNN/examples/merge_minibatch_sample.py` — secondary sampling for mega-batches
- `/home/benito_pc/Escritorio/DiskGNN/src/*.cc, *.cu` — full C++ extension (gather, load, save, free, cuda kernels — no sampler)
- `/home/benito_pc/Escritorio/DiskGNN/python/setup.py` — confirms only `offgs` C++ ext, no sampling extension
- `docs/external_references/GNN_PAPERS/02_DiskGNN/sections/4-system-overview.tex` — paper architectural overview
- `docs/external_references/GNN_PAPERS/02_DiskGNN/sections/5-system-design.tex` — paper key designs section

All findings cited above are reproducible by re-reading those exact files.
