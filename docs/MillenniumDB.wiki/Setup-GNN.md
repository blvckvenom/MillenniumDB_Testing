# Building with GNN support

The graph neural network pipeline is compiled out by default. Everything on this
page is in addition to [[Setup]], which still has to be satisfied first.

## What the option turns on

`-D ENABLE_GNN=ON` compiles `src/gnn/` into `mdb_gnn_core` and registers the GNN
procedures with the GQL executor. Without it the binary builds and runs
normally, and `CALL gnn_train(...)` reports an unknown procedure.

## Extra requirements

| Requirement | Why |
|---|---|
| LibTorch (C++ distribution of PyTorch) | the model, the optimiser, and the tensor operations |
| NVIDIA driver with a kernel module for the **running** kernel | otherwise CUDA is invisible |
| CUDA Toolkit matching the LibTorch build | mixing a `cu124` LibTorch with a `cu128` toolkit fails at link time |
| ~40 GB free | the cu128 LibTorch unpacks to 6 GB with its zip another 3.6 GB, the CUDA Toolkit takes 8.8 GB plus a 5.4 GB installer left in `/tmp`, and the automated path builds both Release (~2 GB) and a Debug tree of about 13 GB |

The LibTorch build has to match the GPU generation. Ada cards (RTX 40 series)
take the `cu124` distribution; Blackwell cards (RTX 50 series) need `cu128`.
A `cu124` LibTorch on a Blackwell card compiles cleanly and fails at run time
with *no kernel image is available for execution on the device*, because the
toolkit has no code for compute capability 12.0. That is a loud failure. The
silent fallback described at the end of this page is the different case where
CUDA is absent altogether.

## The automated path

```bash
scripts/onboard.sh
```

Installs the driver, the CUDA Toolkit, Boost 1.82, the matching LibTorch, the
Python virtual environments, then configures and builds Release with
`ENABLE_GNN=ON` and runs the unit tests. Every step records its own completion
marker, so re-running it resumes rather than redoing.

All commands that need root are collected into a single batch and run through
one `pkexec` call, so the whole thing asks for the administrator password once.
Preview that batch without authenticating:

```bash
scripts/onboard.sh --dry-run
```

Useful flags:

```bash
scripts/onboard.sh --gpu=5070      # force the Blackwell stack (CUDA 12.8, LibTorch cu128)
scripts/onboard.sh --gpu=4070      # force the Ada stack (CUDA 12.4, LibTorch cu124)
scripts/onboard.sh --no-gpu        # CPU-only LibTorch, no CUDA Toolkit
scripts/onboard.sh --skip-driver   # driver and CUDA are already installed
scripts/onboard.sh --resume        # continue after the reboot the driver install needs
```

By default the run also pre-stages the raw OGB archives — arxiv, products, mag,
papers100M and proteins, about 52 GB — so that a later experiment does not wait
on a download. Narrow or skip that:

```bash
scripts/onboard.sh --download-dataset=arxiv,products   # only these
scripts/onboard.sh --download-dataset=                 # none
scripts/onboard.sh --skip-cora                         # skip the 5 MB Cora source too
```

Cora is different from the rest: it is downloaded *and* converted, because every
command this page and [[Creating and running a GNN database]] recommend needs
`data/example/gql/cora/` to exist, and that directory is not in version control.

The script detects the GPU generation from three sources in order: the model
name reported by `nvidia-smi`, the model name in `lspci`, and the PCI device id.
The third exists because recent cards are often unknown to the installed
`pci.ids` database and print as a bare hexadecimal id, which no name pattern
matches. When all three are inconclusive the script stops rather than guessing a
CUDA version.

## The manual path

```bash
# LibTorch for a Blackwell card. Ada cards take the cu124 build instead.
wget https://download.pytorch.org/libtorch/cu128/libtorch-cxx11-abi-shared-with-deps-2.7.0%2Bcu128.zip
unzip libtorch-*.zip -d "$HOME/libs"

# The build and the editor configuration both look for third_party/libtorch, so
# link it there rather than pointing only CMake at the unpacked directory.
mkdir -p third_party
ln -sfn "$HOME/libs/libtorch" third_party/libtorch

cmake -B build/Release \
      -D CMAKE_BUILD_TYPE=Release \
      -D ENABLE_GNN=ON \
      -D CMAKE_PREFIX_PATH="$PWD/third_party/libtorch"
cmake --build build/Release -j "$(nproc)"
```

Boost 1.82 is expected at `third_party/boost_1_82`. `scripts/onboard.sh` creates
both locations; a manual build has to arrange them itself.

## Verifying the build

```bash
scripts/validate-gnn-environment.sh
```

Exit code 0 means every check passed, 1 means something is missing that blocks
GNN work, and 2 means it will run with limitations. CUDA counts as a limitation
rather than a blocker, so a CPU-only host exits 2, not 0.

For an end-to-end check that a GPU is genuinely being used, run the smallest
benchmark:

```bash
scripts/gnn_benchmark.sh cora
```

It takes a few seconds and refuses to run without a usable GPU unless `--cpu` is
passed. See [[Creating and running a GNN database]] for what it does.

## When the GPU stops working

MillenniumDB does not fail when CUDA is unavailable. It moves the work to the
CPU and continues, so a broken GPU looks like a slow machine rather than an
error. The signs, in the order they appear:

| Symptom | Where |
|---|---|
| `gpuAvailable` reads `false` | `gnn_build_feature_store` output |
| `l1Nodes` is 0 and `l1HitRatio` is 0.0, **on a store built with a non-zero `gpu_budget_mb`** | `gnn_build_feature_store`, `gnn_train` |
| `useAddrTablesEffective` reads `false` | `gnn_train` output |
| epochs take several times longer than expected | anywhere |

The qualification on the second row matters. `scripts/gnn_benchmark.sh cora`
builds its store with `gpu_budget_mb:0`, so a perfectly healthy GPU also reports
`l1Nodes=0` and `l1HitRatio=0.0` there. Pass `--gpu-cache-mb 512` to see the
cache exercised, or read the `gpuAvailable` line instead, which answers the
question directly.

The most common cause on a rolling-release distribution is a kernel upgrade: the
driver package is still installed, but no kernel module was built for the kernel
that is now running. Distinguish the two cases:

```bash
nvidia-smi                                   # fails when the module is missing
dpkg -l | grep nvidia-driver                 # the driver is still listed
ls /lib/modules/$(uname -r)/updates/dkms/    # empty when nothing was rebuilt
```

Two packaging models exist and they are repaired differently. When the driver is
managed by DKMS, `sudo dkms autoinstall` rebuilds the module against the running
kernel. When it comes as prebuilt per-kernel packages, DKMS has nothing to
rebuild and the fix is to install the module package matching the kernel series,
for example `linux-modules-nvidia-<version>-generic-hwe-24.04`. Checking
`/var/lib/dkms/` tells you which model is in use: an NVIDIA entry means DKMS, no
entry means prebuilt packages.

Re-running `scripts/onboard.sh` handles both.
