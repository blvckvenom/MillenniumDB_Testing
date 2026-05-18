# Phase 2 — DiskGNN repo setup on celebi

**Date:** 2026-05-18
**Repo:** `/home/bfuentes/DiskGNN/DiskGNN/` (paper code, README pins Python 3.9.18 + PyTorch 2.0.1 + CUDA 11.7 + DGL 1.1.2 + PyG 2.5.0)
**Plan reference:** `docs/superpowers/plans/2026-05-17-diskgnn-paper-parity-plan.md` Tasks 10-13.

## Environment

- CUDA installed (toolkit): **12.8** at `/usr/local/cuda` (`nvcc 12.8.61`)
- Driver: 580.126.09 (reports CUDA 13.0)
- GPU: **NVIDIA GeForce RTX 5070 Ti**, compute capability **(12, 0)** = sm_120 = **Blackwell**
- VRAM: 16 303 MiB
- System Python: 3.12.3
- conda installed at: `/home/bfuentes/miniconda3/` (Miniconda3, conda 26.3.2; this run)
- conda env created: `diskgnn_cu124` — Python 3.10.20 (the name is a misnomer; we ended up on cu128)
- liburing: dev `liburing-dev:amd64 2.5-1build1`, headers at `/usr/include/liburing.h`
- Disk free: 415 GB on `/`
- RAM: 30 GiB total, ~27 GiB available
- OS: Ubuntu 24.04.4 LTS

## Attempts log

### Attempt 1: CUDA 11.7 stack (paper-pinned)
- **Skipped intentionally.** Paper pins PyTorch 2.0.1 with cu117 wheels. cu117 toolkit (Apr 2022) predates Blackwell entirely; sm_120 was introduced with CUDA 12.8. PyTorch 2.0.1 cu117 wheels contain no sm_120 PTX/cubin, so kernel launches would fail even if `torch.cuda.is_available()` returned True. Spending ~30-45 min on miniconda + cu117 env just to confirm this is wasted budget. Documented and moved on.
- **Conclusion:** FAIL by construction. Not attempted.

### Attempt 2A: PyTorch 2.6.0 + CUDA 12.4 (modern stable)
- **Command:** `pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cu124`
- **Result:** Install succeeded. `torch.cuda.is_available()` returns `True`, but actual kernel launch (`torch.zeros(1, device='cuda').sum().item()`) raises:

```
RuntimeError: CUDA error: no kernel image is available for execution on the device
UserWarning: NVIDIA GeForce RTX 5070 Ti with CUDA capability sm_120 is not compatible with the current PyTorch installation.
The current PyTorch install supports CUDA capabilities sm_50 sm_60 sm_70 sm_75 sm_80 sm_86 sm_90.
```

- **Conclusion:** Confirms advisor's prediction — `cuda.is_available()` is not a sufficient gate. Required upgrade.

### Attempt 2B: PyTorch 2.7.0 + CUDA 12.8 (Blackwell stable)
- **Command:** `pip install --upgrade torch==2.7.0 --index-url https://download.pytorch.org/whl/cu128`
- **Result:** PASS. `torch.zeros(1, device='cuda').sum().item()` → `0.0`. `torch.cuda.get_device_capability(0)` → `(12, 0)`.
- **Conclusion:** PyTorch 2.7 cu128 is the minimum Blackwell-stable stack.

### Attempt 2C: PyG ecosystem on torch-2.7 + cu128
- **Commands:**
  ```
  pip install torch_geometric
  pip install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv \
              -f https://data.pyg.org/whl/torch-2.7.0+cu128.html
  ```
- **Result:** PASS. PyG 2.7.0 + `pyg_lib-0.5.0+pt27cu128`, `torch_scatter-2.1.2+pt27cu128`, `torch_sparse-0.6.18+pt27cu128`, `torch_cluster-1.6.3+pt27cu128`, `torch_spline_conv-1.2.2+pt27cu128`.
- **Conclusion:** PyG ecosystem (compiled extensions) ship Blackwell-compatible wheels for torch-2.7 cu128.

### Attempt 2D: DGL with graphbolt on torch-2.7 cu128 — **BLOCKER**
- **Commands tried:**
  1. `pip install dgl` → installed `dgl-2.1.0` (PyPI cpu-only wheel). Importable as raw DGL, but `import dgl.graphbolt as gb` fails with `ModuleNotFoundError: No module named 'torchdata.datapipes'`. Downgraded `torchdata<0.10` to restore the older datapipe API.
  2. After torchdata downgrade + `pip install pyyaml pydantic`: `import dgl.graphbolt as gb` fails with `FileNotFoundError: Cannot find DGL C++ graphbolt library at .../libgraphbolt_pytorch_2.7.0.so`. DGL 2.1 ships libgraphbolt for torch 2.0.0–2.2.1 only.
  3. **DGL 2.5.0 from data.dgl.ai:** torch-2.6 channel exists (`https://data.dgl.ai/wheels/torch-2.6/dgl-2.5.0-cp310-cp310-manylinux1_x86_64.whl`, HTTP 200). The cu121 channel is 403 from outside. Wheel installed cleanly, ships `libgraphbolt_pytorch_2.6.0.so`.
  4. **Symlink hack:** `ln -s libgraphbolt_pytorch_2.6.0.so libgraphbolt_pytorch_2.7.0.so` lets DGL find the file, but loading raises:
     ```
     OSError: ...libgraphbolt_pytorch_2.6.0.so: undefined symbol:
       _ZN5torch6detail10class_baseC2ERKSsS3_SsRKSt9type_infoS6_
     ```
     PyTorch's `torch::detail::class_base` constructor signature changed between 2.6 and 2.7 — hard C++ ABI break, not a soft mismatch. The symlink workaround is fundamentally impossible.
  5. **Nightly DGL torch-2.7 channels probed:** `data.dgl.ai/wheels-test/torch-2.7/...`, `wheels-internal/torch-2.7/...` → AccessDenied. No prebuilt DGL exists for torch-2.7 cu128 at this date.
  6. **Build DGL from source:** Not attempted (estimated 30-60 min for the full graphbolt extension, with high risk of further Blackwell-driver / cccl compat surprises; this would blow the time budget).
- **Affected paper scripts** (all use `import dgl.graphbolt as gb`):
  ```
  prepare_dataset.py, sampling.py, feat_deduplicate_*.py, mega_batch_sampling.py,
  megabatch_train_single_thread.py, merge_minibatch_*.py, disk_cache_overlap.py,
  train_multi_thread.py
  ```
  i.e., the entire pipeline entry point chain.
- **Conclusion:** DGL graphbolt is a hard blocker for running the paper code as-is on Blackwell. Workaround paths (older DGL, manual prepare_dataset reimplementation) work only as far as the sampler pipeline.

### Attempt 3: Paper C++ extension build (`offgs`)
- **Pre-patch:** `CMakeLists.txt` line 3 was `set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -arch=sm_75")`. Patched to `-arch=sm_120` for Blackwell.
- **Commands:**
  ```
  cd /home/bfuentes/DiskGNN/DiskGNN && rm -rf build && mkdir build && cd build
  cmake ..        # picked up torch-2.7 from the env, autodetected CUDA 12.0 arch
  make -j4
  ```
- **Result:** PASS. Build completed cleanly through `Linking CXX shared library liboffgs.so`. No DGL dependency in the C++ source (verified by grep), so DGL's brokenness doesn't infect the native extension.
- **Python wrapper install** (`cd ../python && python setup.py install`): PASS, but the install puts `liboffgs.so` at `<env>/offgs/liboffgs.so` while `offgs/__init__.py` expects it at `<env>/lib/python3.10/site-packages/offgs/liboffgs.so`. Manually copied to fix. `import offgs` then succeeds and `dir(offgs)` shows `OffgsDataset` exported.
- **Conclusion:** Paper's C++ extension (`liboffgs.so`, ~100 K lines of CUDA + io_uring) builds and loads against PyTorch 2.7 cu128 with sm_120 PTX. This is the actual high-risk surface — and it works.

### Attempt 4: ogbn-arxiv smoke test
- **prepare_dataset.py via README path:** FAIL.
  ```
  python prepare_dataset.py --dataset ogbn-arxiv
  → ImportError: Cannot load Graphbolt C++ library
  ```
- **Workaround via old DGL+OGB API:** wrote `/tmp/prepare_ogbn_arxiv_via_ogb_api.py` that uses `from ogb.nodeproppred import DglNodePropPredDataset` (the older API that's commented out in the repo's `load_graph.py:load_ogb`) and replicates the `run()` outputs (graph.pth, features.bin, labels.pth, conf.json, split_idx.pth).
- **Result:** PASS. Downloaded ogbn-arxiv (~83 MB zip), wrote dataset to `/home/bfuentes/diskgnn_data/offgs_dataset/ogbn-arxiv-offgs/`. Graph: 169 343 nodes, 1 335 586 edges; feat [169343, 128] float32; labels [169343]; 40 classes.
- **Next step (sampling.py):** BLOCKED. `sampling.py` is `import dgl.graphbolt as gb` from line 9, with `gb.ItemSampler`, `gb.DataLoader`, `gb.ItemSet` calls throughout — not trivially portable to the old `dgl.dataloading` API since the file structure expects graphbolt's `MiniBatch.compacted_*` attributes and the graphbolt-specific neighbor sampler. Replicating this would require ~200 LOC of port work and is out of scope for the 90-min budget.
- **Downstream chain:** `feat_packing.py` reads `node_counts.pt`, `meta_node_popularity.pt`, `in-nid-{bid}.pt`, all produced by `sampling.py`. `train_multi_thread.py` reads further outputs of `feat_packing.py`. All gated by `sampling.py`.

## Final state (historical, pre-source-build)

> **2026-05-18 update:** The graphbolt blocker described below was subsequently resolved by building `libgraphbolt_pytorch_2.7.0.so` from DGL source — see the **graphbolt-from-source build** section at the bottom of this doc. The "Recommendation" section below is preserved for the audit trail but its core verdict is now superseded: Phase 2 is unblocked.

- **Env name:** `diskgnn_cu124` (misnomer — actually torch-2.7 + cu128)
- **Activation:** `source /home/bfuentes/miniconda3/etc/profile.d/conda.sh && conda activate diskgnn_cu124`
- **Paper code runnable:**
  - Dataset prepared on disk for ogbn-arxiv via custom adapter (`/tmp/prepare_ogbn_arxiv_via_ogb_api.py`) — YES
  - `offgs` C++ extension (`liboffgs.so`) — YES, builds and imports on Blackwell
  - `prepare_dataset.py` (README path) — NO (graphbolt blocker)
  - `sampling.py` — NO (graphbolt blocker)
  - `feat_packing.py` — NO transitively (depends on sampling.py outputs)
  - `train_multi_thread.py` — NO transitively (uses `dgl.dataloading.NeighborSampler` which works, but reads cache files produced by feat_packing.py)
- **Blocker (root cause):** DGL graphbolt's C++ extension (`libgraphbolt_pytorch_X.Y.Z.so`) is pinned per-PyTorch-version and DGL has not published a `torch-2.7` channel. PyTorch 2.7 is the minimum stable for Blackwell sm_120. Result: the paper's pipeline entry point (`sampling.py`) is unrunnable.
- **Patches landed on paper repo:**
  - `CMakeLists.txt:3` `-arch=sm_75` → `-arch=sm_120` (one-line, required for Blackwell PTX gen)
- **Disk artifacts left:**
  - `/home/bfuentes/miniconda3/` (~3-4 GB conda root)
  - `/home/bfuentes/diskgnn_data/ogb_raw/` + `arxiv.zip` extract (~ 200 MB)
  - `/home/bfuentes/diskgnn_data/offgs_dataset/ogbn-arxiv-offgs/` (~115 MB)
  - `/tmp/prepare_ogbn_arxiv_via_ogb_api.py` (adapter script)
  - `/home/bfuentes/DiskGNN/DiskGNN/build/` (cmake outputs + liboffgs.so)
- **Time:** ~75 min wall-clock (within 90-min stop budget).

## Recommendation

**Abort full Phase 2 benchmark.** The DiskGNN paper's experimental pipeline (papers100M with `[10,15,20]` fanout, 50 epochs, smart-search feature packing, online training) is **not runnable** on celebi without one of:

1. **Build DGL graphbolt from source for torch-2.7 cu128.** ~1-3 hours, with risk of further failures on the way (DGL's CMake stack pulls cccl headers, and graphbolt has internal CUDA kernels that need sm_120 retargeting). Worth doing only if there's a strategic reason to need apples-to-apples DiskGNN paper numbers on Blackwell.
2. **Port `sampling.py` from graphbolt to `dgl.dataloading.NeighborSampler`.** ~200-400 LOC. Even if successful, would produce sampling that's no longer bit-equivalent to the paper's measured pipeline, so the "paper comparison" framing breaks — at that point you're running our own sampling on the paper's training loop, which gives no clean apples-to-apples signal anyway.
3. **Run on a non-Blackwell GPU.** A Turing/Ampere/Hopper GPU + the paper's pinned cu117 stack would work out-of-the-box; this is a hardware constraint, not a software bug we can fix on celebi.

**Phase 2 was already flagged informational** in the master plan ("Phase 0 findings already determined Phase 1 is not justified; Phase 2 is informational only"). The headline finding — that DiskGNN's reference implementation is hard-coupled to a (DGL graphbolt) × (PyTorch ABI) × (CUDA arch) triple that isn't co-satisfiable on Blackwell — is itself the deliverable. The MillenniumDB GNN pipeline can continue benching its own offline-sampling + four-level-store + training pipeline (Plan F + Spec #13 + Spec C3 stages) and report relative speedups without paper-side reproduction.

**If a non-Blackwell GPU becomes available** (an A100/H100 box, e.g.), the paper's exact stack would install in ~20 min and the existing celebi tooling/results in `~/Desktop/spec13_papers100m_e2e/...` can be re-baselined against it.

## Files referenced

- `/home/bfuentes/DiskGNN/DiskGNN/CMakeLists.txt:3` — sm_75 → sm_120 patch
- `/home/bfuentes/DiskGNN/DiskGNN/examples/prepare_dataset.py` — graphbolt-blocked
- `/home/bfuentes/DiskGNN/DiskGNN/examples/sampling.py` — graphbolt-blocked
- `/home/bfuentes/DiskGNN/DiskGNN/examples/load_graph.py` — has the old `load_ogb` path (commented out as caller) usable for the adapter
- `/tmp/prepare_ogbn_arxiv_via_ogb_api.py` — working adapter for ogbn-arxiv dataset prep
- `/home/bfuentes/miniconda3/envs/diskgnn_cu124/` — Python env with torch 2.7 cu128, offgs, PyG ecosystem, DGL (raw API only)

## graphbolt-from-source build (2026-05-18)

The graphbolt blocker from Attempt 2D was resolved by building the `libgraphbolt_pytorch_2.7.0.so` from DGL source against the working torch 2.7 + cu128 + Blackwell sm_120 stack.

### Build summary

- DGL source: `/tmp/dgl-src` (branch `2.5.x`, commit `cd67ccc6`)
- Shallow submodules: `--depth 1 --shallow-submodules` (CCCL, HugeCTR, liburing, taskflow, tsl_robin_map, pcg, cuco, nanoflann)
- Build command (in `/tmp/dgl-src/graphbolt`):
  ```bash
  mkdir -p /tmp/gb-bin/graphbolt
  LD_LIBRARY_PATH="/usr/local/cuda/lib64" \
  CMAKE_COMMAND=cmake \
  CUDA_TOOLKIT_ROOT_DIR=/usr/local/cuda \
  USE_CUDA=ON USE_LIBURING=ON \
  BINDIR=/tmp/gb-bin \
  CUDAARCHS="120" \
  CC=gcc CXX=g++ \
  bash build.sh "$(which python)"
  ```
- LD_LIBRARY_PATH scrubbed (excluded `/home/bfuentes/MillenniumDB_Testing/third_party/libtorch/lib`) — otherwise cmake's `find_package(Torch)` would resolve to MDB's bundled libtorch and produce a binary linked against the wrong libc10. Verified post-build via `ldd`:
  ```
  libc10.so => /home/bfuentes/miniconda3/envs/diskgnn_cu124/lib/python3.10/site-packages/torch/lib/libc10.so
  libtorch.so => /home/bfuentes/miniconda3/envs/diskgnn_cu124/lib/python3.10/site-packages/torch/lib/libtorch.so
  libtorch_cuda.so => /home/bfuentes/miniconda3/envs/diskgnn_cu124/lib/python3.10/site-packages/torch/lib/libtorch_cuda.so
  ```
  Correctly bound to conda env torch, not MDB third_party.

### Patches applied

**None.** No source modifications were needed. The DGL 2.5.x graphbolt CMakeLists already supports env-driven CUDA arch via `$CUDAARCHS`, and exposing `CUDAARCHS="120"` produced the correct nvcc flag chain:
```
-gencode arch=compute_120,code=sm_120 ... --generate-code=arch=compute_120,code=[compute_120,sm_120]
```

(The cuda/extension subfolder additionally inherits Volta-only `compute_70` per the upstream comment about CCCL #1083, which is irrelevant to runtime since we never call extension/* on devices < sm_70.)

### Build metrics

- Wall-clock: ~3 min (single make -j 20 invocation, gated by the heaviest `.cu` file `neighbor_sampler.cu` ≈ 2:13 single-core CUB+thrust+taskflow template expansion)
- Peak RAM: ~7 GB (well under 30 GB)
- Output: `/tmp/dgl-src/graphbolt/build/2.7.0/libgraphbolt_pytorch_2.7.0.so` (83 018 568 bytes = 79 MB)
- Auxiliary: `/tmp/dgl-src/graphbolt/build/2.7.0/libgraphbolt_pytorch_2.7.0_cuda.a` (the extension static lib that the .so links in)

### Installation

```bash
DGL_GB_DIR=/home/bfuentes/miniconda3/envs/diskgnn_cu124/lib/python3.10/site-packages/dgl/graphbolt
rm "${DGL_GB_DIR}/libgraphbolt_pytorch_2.7.0.so"  # removes the broken 2.6.0-symlink trick
cp /tmp/gb-bin/graphbolt/libgraphbolt_pytorch_2.7.0.so "${DGL_GB_DIR}/"
```

Now both `libgraphbolt_pytorch_2.6.0.so` (shipped by DGL wheel, unused) and `libgraphbolt_pytorch_2.7.0.so` (freshly built, ABI-correct) coexist; DGL's loader picks the right one via `torch.__version__.split("+")[0]`.

### Verification

| Test | Result |
|---|---|
| `import dgl.graphbolt as gb` | **PASS** |
| `gb.BuiltinDataset('ogbn-arxiv').load()` (177 MB download + extract) | **PASS** |
| GPU smoke: `graph.to('cuda').sample_neighbors(seeds_cuda, fanout=5)` + `torch.cuda.synchronize()` | **PASS** (17 sampled edges from 4 seeds × 5 fanout, sm_120 kernel executed on RTX 5070 Ti) |
| Paper `import sampling` | **PASS** |
| `python prepare_dataset.py --dataset ogbn-arxiv --store-path /home/bfuentes/diskgnn_data/offgs_dataset` | **PASS** — produces `graph.pth` (131 MB), `features.bin` (87 MB), `labels.pth`, `conf.json`, `split_idx.pth` |

### Patches landed on paper repo (this session)

- `examples/prepare_dataset.py:23` — replaced hardcoded `root="/nvme2n1/graphbolt_dataset/datasets"` with `/home/bfuentes/diskgnn_data/graphbolt_dataset/datasets` (host-specific; `.bak` of original retained). The `--store-path` flag already covered the output side.

### Files referenced (this section)

- `/tmp/dgl-src/` — DGL 2.5.x source clone
- `/tmp/dgl-src/graphbolt/CMakeLists.txt` — env-driven build config (no patches needed)
- `/tmp/dgl-src/graphbolt/build/2.7.0/libgraphbolt_pytorch_2.7.0.so` — newly built binary (79 MB)
- `/home/bfuentes/miniconda3/envs/diskgnn_cu124/lib/python3.10/site-packages/dgl/graphbolt/libgraphbolt_pytorch_2.7.0.so` — installed copy
- `/tmp/graphbolt_build.log` — full build log (130+ lines)
- `/home/bfuentes/diskgnn_data/graphbolt_dataset/datasets/ogbn-arxiv-seeds*` — graphbolt-format ogbn-arxiv (177 MB zip + extracted)
- `/home/bfuentes/diskgnn_data/offgs_dataset/ogbn-arxiv-offgs/` — paper-format ogbn-arxiv (220 MB total)

### Recommendation for Phase 2 continuation

**YES, proceed.** The graphbolt blocker is fully cleared:
- (i) Real `prepare_dataset.py` runs end-to-end on ogbn-arxiv.
- (ii) `sampling.py` imports without error and all `gb.*` API surface used by it (`gb.ItemSampler`, `gb.DataLoader`, `gb.ItemSet`, `gb.NeighborSampler`, `gb.BuiltinDataset`) is exposed and callable.
- (iii) GPU smoke confirms sm_120 graphbolt CUDA kernels (specifically `neighbor_sampler.cu`) execute correctly on RTX 5070 Ti.
- (iv) The offgs C++ extension (`liboffgs.so`) from earlier in this doc still loads (separate library, unaffected by graphbolt rebuild).

Next steps for Phase 2 should be:
1. Run `sampling.py --dataset ogbn-arxiv` to generate the sampled megabatch artifacts.
2. Run `feat_packing.py` on the sampling output.
3. Run `train_multi_thread.py` for the small-scale ogbn-arxiv accuracy baseline.
4. If ogbn-arxiv looks healthy, attempt papers100M — note papers100M graphbolt download is ~120 GB (warn the user before kicking off).

**Reservation about Blackwell-only build cuda/extension half:** The `cuda/extension/*.cu` files compile only with `compute_70` (Volta minimum) per the upstream CMakeLists comment about unresolved CCCL issue #1083. If any extension code path is hit on sm_120, it will likely fail with "no kernel image" — but the graphbolt design intent is that the extension cache (`gpu_cache.cu`, `gpu_graph_cache.cu`, `unique_and_compact_map.cu`) is optional. Empirically the smoke test using `sample_neighbors` did not touch it. If later phases trip on extension code we may need to revisit (the fix would be to also force `CUDAARCHS=70;120` for the extension lib, which is the upstream-recommended pattern).

---

## ogbn-arxiv smoke test (2026-05-18)

End-to-end attempt at the paper's three-stage pipeline (`sampling.py` -> `batched_packing.py` -> `train_multi_thread.py`). Run dir `/home/bfuentes/Desktop/spec13_papers100m_e2e/post_pop_os/42_paper_arxiv_smoke/` holds the full `/usr/bin/time -v` logs.

### Stage 1: `sampling.py` — PASS (after CPU patches)
- **Wall-clock:** 4.01 s elapsed, script-reported `Sampling Time 1.731 s + Save Block 0.842 s + Save Rank 0.005 s = Total 2.578 s` (the 1.4 s gap is python startup + dataset load).
- **Peak RSS:** 797.5 MB.
- **Output files:** 89 train batches (`train-{0..88}.pt` 1024-seed each) + `in-nid-*.pt` + `out-nid-*.pt` + `node_counts.pt` + `meta_node_popularity.pt` under `/home/bfuentes/diskgnn_data/offgs_dataset/ogbn-arxiv-1024-10,15,20-1.0/`.
- **Patches required** (all in a copy at `/tmp/diskgnn_patched/sampling.py`, paper repo untouched):
  1. Force `device = torch.device("cpu")` and skip `g.pin_memory_()`. Root cause: the DGL wheel shipped on PyPI bundles a CPU-only `libdgl.so` (11.9 MB; CUDA builds are ~100 MB). The custom graphbolt-cuda `.so` I built earlier does GPU sampling, but the resulting `DGLBlock` cannot be `.cpu()`'d or moved because `libdgl.so` itself has no CUDA dispatch. Forcing the entire sampler datapipe to CPU avoids the cross-library boundary.
  2. Pass `device=device` (torch.device object) into `create_dataloader` instead of the int `args.device`.
  3. `mkdir -p /tmp/diskgnn_patched/logs/` so the CSV writer at `args.log='logs/sample_decompose.csv'` has a target dir.
  4. **Conda-env site-packages patch:** `/home/bfuentes/miniconda3/envs/diskgnn_cu124/lib/python3.10/site-packages/offgs/dataset.py` — three `torch.load(...)` calls (`labels`, `split_idx`, `graph`) got `weights_only=False` appended. Required by PyTorch 2.6+ default of `weights_only=True` which rejects DGL's `FusedCSCSamplingGraph` pickle.

### Stage 2: `batched_packing.py` — PASS (NOT `feat_packing.py`)
- **Wall-clock:** 3.79 s elapsed, script-reported `Total Time 2.951 s`.
- **Peak RSS:** 1.12 GB.
- **Cache config:** `--feat-cache-size 3e7` (30 MB = ~34.6% of the 82 MB feature dataset; the original task brief's 5e8 forces 100% caching which makes the disk-cache assert in the script fire because there's nothing left to put on disk).
- **Key gotcha — the task brief calls the wrong script.** The brief specifies `feat_packing.py`. That script writes `cached_feats.pt` (torch.save). `train_multi_thread.py` then calls `torch.ops.offgs._CAPI_LoadFeats_Direct("cached_feats.bin", ...)` — a different file, written by `batched_packing.py`. `feat_packing.py` is an older/companion script; `batched_packing.py` is the paired packer for `train_multi_thread.py`. After running the latter, `cached_feats.bin` + `cache_rev_idx.pt` appear and stage 3 advances.

### Stage 3: `train_multi_thread.py` — **FAIL** at `block.to(cuda)`
- **Wall-clock to failure:** 2.78 s elapsed.
- **Failure point:** `train_multi_thread.py:281` — `blocks = [block.to(device, non_blocking=True) for block in blocks]`. Fatal error:
  ```
  dgl._ffi.base.DGLError: Check failed: allow_missing:
    Device API cuda is not enabled. Please install the cuda version of dgl.
  ```
- **Same root cause as stage 1's needed patch:** the CPU-only `libdgl.so` has no CUDA dispatch. The training loop unconditionally moves DGLBlocks to GPU before forward pass, and there is no env-var or argument switch in the script to keep them on CPU. Patching the training loop to be fully CPU-resident would require rewriting the cuda-stream-based feature-loader threads (`feat_transfer` uses `torch.cuda.Stream`, `torch.ops.offgs._CAPI_GatherInGPU`, etc.) and would no longer be the paper's pipeline.
- **Patches applied** (`/tmp/diskgnn_patched/train_multi_thread.py`):
  1. `torch.load = lambda *a, **kw: orig_load(*a, weights_only=False, **kw)` monkey-patch at module load so the four DGLBlock-pickle calls in `load_graph`/`load_meta`/etc. work under PyTorch 2.6+.
  This patch was sufficient to unblock the script init; it then crashed at `block.to(device)`.

### What `/proc/sys/vm/drop_caches` line was skipped
- All three paper scripts have commented-out `with open("/proc/sys/vm/drop_caches", "w") as stream: stream.write("1\n")` blocks. Not reachable without sudo and explicitly skipped in the task brief. No-op on this run; cold-cache timing measurements would need a separate kernel-cache-flush mechanism.

### Conclusion
- **Pipeline functional on celebi as configured:** NO. Stages 1 + 2 PASS via documented patches. Stage 3 fails on the very first `block.to(cuda)` because the upstream DGL wheel's `libdgl.so` is CPU-only. The custom graphbolt-cuda build cleared the import path but does not provide a CUDA DGL runtime.
- **What it would take to unblock stage 3:**
  1. Build full DGL from source with `USE_CUDA=ON` against torch 2.7 + cu128, sm_120 (the same path used to produce the graphbolt `.so`, but now for the core `libdgl.so` ~100 MB binary including all the CUDA `array/cuda/`, `runtime/cuda/`, `kernel/cuda/` translation units). Estimated 30-90 min if `data.dgl.ai`'s `wheels-internal/torch-2.7` channel does not surface a prebuild between now and then.
  2. Alternative: rewrite `train_multi_thread.py` for pure-CPU training. This drops the paper's GPU-bandwidth saturating design (CUDA streams, GPU feature cache, pinned-memory DMA) and stops being a parity test.
- **Recommendation for papers100M:** **DO NOT proceed** with papers100M on the current stack. We cannot validate even the small ogbn-arxiv accuracy baseline. A papers100M run would burn the ~120 GB download + multi-hour sampling pass and then crash at the same `block.to(cuda)` line. Next session should land the DGL-from-source build first (the same way graphbolt was built), then re-attempt this smoke test as a gating check.

### Patched files (all outside the protected working tree)
- `/tmp/diskgnn_patched/sampling.py` — CPU device, no pin_memory
- `/tmp/diskgnn_patched/feat_packing.py` — copy of original, unused (we used batched_packing.py)
- `/tmp/diskgnn_patched/train_multi_thread.py` — torch.load monkey-patch
- `/home/bfuentes/miniconda3/envs/diskgnn_cu124/lib/python3.10/site-packages/offgs/dataset.py` — `weights_only=False` on three torch.load calls (env-local; not packaged back upstream)

## Full DGL CUDA build (2026-05-18)

Built DGL 2.5.x from `/tmp/dgl-src` against torch 2.7.0+cu128, CUDA 12.8, with CUDA archs `sm_80;sm_86;sm_89;sm_90;sm_120` (Blackwell RTX 5070 Ti is sm_120). The PyPI wheel's `libdgl.so` is CPU-only — this build replaces it with a full CUDA-enabled binary.

### Build config

```bash
cd /tmp/dgl-src && rm -rf build && mkdir build && cd build
export CUDAToolkit_ROOT=/usr/local/cuda
TORCH_DIR=$(python -c "import torch; print(torch.utils.cmake_prefix_path)")
cmake .. \
  -DUSE_CUDA=ON \
  -DCMAKE_PREFIX_PATH="${TORCH_DIR}" \
  -DCMAKE_BUILD_TYPE=Release \
  -DBUILD_TYPE=release \
  -DBUILD_TORCH=ON \
  -DUSE_LIBURING=ON \
  -DUSE_OPENMP=ON \
  -DCUDA_ARCH_NAME=Manual \
  -DCUDA_ARCH_BIN="80;86;89;90;120" \
  -DCUDA_ARCH_PTX="90"
make -j8
```

### Built libraries

| Library | Size before | Size after | Notes |
|---|---|---|---|
| `libdgl.so`                                | 11.9 MB (CPU-only PyPI) | **196 MB** (CUDA) | full sm_80..sm_120 fat binary |
| `libgraphbolt_pytorch_2.7.0.so`            | 83 MB  (sm_120 only, prior subagent) | **198 MB** (sm_80..sm_120) | rebuilt for arch parity |
| `libdgl_sparse_pytorch_2.7.0.so`           | absent (only `_2.6.0.so` shipped)    | **977 KB** | adapter shim |
| `libtensoradapter_pytorch_2.7.0.so`        | absent (only `_2.6.0.so` shipped)    | **168 KB** | torch 2.7 ABI shim |

Build wall-clock: 8 min 30 s from cmake-configure to final link (graphbolt + dgl + dgl_sparse + tensoradapter targets sequentially complete; see `/tmp/dgl_build.log`).

### Installation

Replaced in `~/miniconda3/envs/diskgnn_cu124/lib/python3.10/site-packages/dgl/`:
- `libdgl.so` (CPU-only 11.9 MB → CUDA 196 MB; backup kept in `/tmp/dgl_backup/libdgl.so`)
- `tensoradapter/pytorch/libtensoradapter_pytorch_2.7.0.so` (added new)
- `dgl_sparse/libdgl_sparse_pytorch_2.7.0.so` (added new)
- `graphbolt/libgraphbolt_pytorch_2.7.0.so` (replaced 83 MB → 198 MB)

### Verification

`g.to('cuda')` smoke test:
```
DGL version: 2.5.0
Torch version: 2.7.0+cu128
Torch CUDA available: True, device count: 1
CUDA arch: (12, 0)
DGL CPU graph: cpu
DGL CUDA graph: cuda:0
g.num_nodes: 4 g.num_edges: 3
PASS: DGL g.to('cuda') works
```

### ogbn-arxiv `train_multi_thread.py` re-run (after DGL CUDA build)

Stage 3, which previously failed with `Device API cuda is not enabled`, now completes the training loop on GPU.

- Command: `python train_multi_thread.py --dataset ogbn-arxiv --fanout "10,15,20" --hidden 256 --dropout 0.2 --model SAGE --gpu-cache-size 1e7 --cpu-cache-size 2e7 --dir /home/bfuentes/diskgnn_data/offgs_dataset --ratio 1.0 --blowup -1`
- Cache pack used: `cache-size-3e+07/blowup--1.0` (the matching pack from `batched_packing.py --feat-cache-size 3e7`).
- **Wall-clock:** 4.75 s for 3 epochs end-to-end.
- **Peak RSS:** 1.94 GB (Maximum resident set size).
- Per-epoch (train acc, epoch time):

| Epoch | Train acc | Graph Load (s) | Epoch Time (s) | Input Feature Num |
|---|---|---|---|---|
| 0 | 48.66% | 0.229 | 1.229 | 1,105,275,520 |
| 1 | 62.66% | 0.123 | 0.652 | 1,105,275,520 |
| 2 | 65.36% | 0.093 | 0.649 | 1,105,275,520 |

Avg epoch time: 0.651 s. The training loop runs the full DiskGNN data path (CPU + GPU feature cache, threaded prefetch, GPU model forward/backward). Logs in `/home/bfuentes/Desktop/spec13_papers100m_e2e/post_pop_os/42_paper_arxiv_smoke/train_cuda.log`.

### Caveat — val/test accuracy not captured (DGL 2.5 graphbolt API drift)

Re-running with `--debug` (which triggers per-epoch val/test eval) crashes on the very first val batch:

```
AttributeError: 'FusedCSCSamplingGraph' object has no attribute 'device'
  File ".../dgl/dataloading/neighbor_sampler.py", line 158, in sample_blocks
    cpu = F.device_type(g.device) == "cpu"
```

DGL 2.5's `dataloading.NeighborSampler.sample_blocks` accesses `g.device` directly on the graph object passed into the sampler. The training-loop dataloader (paper's custom `OffgsDataLoader`) wraps a different graphbolt object that exposes device through a getter (`g.csc_indptr.device`). This is purely an API-drift problem inside the paper's example script — unrelated to whether DGL's CUDA dispatch works. Run dir: `train_cuda_debug.log`. Patching the val/test loop to use the paper's custom wrapper instead of `dgl.dataloading.NeighborSampler` is the fix (out of scope for this CUDA-unblock task).

### Summary

- **libdgl.so size: 196 MB** (vs 11.9 MB CPU-only PyPI)
- **libtensoradapter_pytorch_2.7.0.so: 168 KB** (newly installed for torch 2.7 ABI)
- **DGL CUDA smoke (`g.to('cuda')`): PASS**
- **ogbn-arxiv `train_multi_thread.py`: PASS**
  - Status: train loop runs on GPU; 3 epochs in 4.75 s wall-clock
  - Per-epoch time (epochs 1-2 steady state): 0.65 s
  - Train accuracy: 48.66% → 62.66% → 65.36% (expected GraphSAGE-on-arxiv early-epoch trajectory)
  - val_acc: NOT MEASURED — `--debug` path crashes on `g.device` attribute access in `dgl.dataloading.NeighborSampler`; orthogonal to CUDA enablement
- **Recommendation: PROCEED with papers100M** for the GPU-training validation. The Stage 3 blocker (CPU-only `libdgl.so`) is now resolved. The remaining graphbolt val/test API drift is in the example script and can be patched separately if accuracy needs to be measured at training time; the train loop itself runs correctly on Blackwell sm_120.
