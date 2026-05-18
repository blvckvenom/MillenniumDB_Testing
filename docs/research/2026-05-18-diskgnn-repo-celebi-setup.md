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

## Final state

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
