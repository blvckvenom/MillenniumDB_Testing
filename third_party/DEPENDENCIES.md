# Third-Party Dependencies

This document describes the external libraries required by MillenniumDB that are not tracked in the repository.

## Required Dependencies

### 1. Boost 1.82+

**Location:** `third_party/boost_1_82/`

**Purpose:** Core C++ utilities (filesystem, program_options, asio, etc.)

**Installation:**
```bash
# Option 1: Download pre-built headers
cd third_party
wget https://boostorg.jfrog.io/artifactory/main/release/1.82.0/source/boost_1_82_0.tar.gz
tar -xzf boost_1_82_0.tar.gz
mv boost_1_82_0 boost_1_82
rm boost_1_82_0.tar.gz

# Option 2: System package (Ubuntu/Debian)
sudo apt-get install libboost-all-dev
# Note: May require adjusting CMakeLists.txt paths
```

**Documentation:** See `docs/MillenniumDB.wiki/Setup.md` for detailed Boost setup.

---

### 2. LibTorch 2.5.1+ (PyTorch C++ Frontend)

**Location:** `third_party/libtorch` (symlink recommended)

**Purpose:** Tensor operations and neural network inference for GNN features.

**Installation:**
```bash
# Download LibTorch (CUDA 12.4 version)
cd /path/to/your/libs  # e.g., ~/libs
wget https://download.pytorch.org/libtorch/cu124/libtorch-cxx11-abi-shared-with-deps-2.5.1%2Bcu124.zip
unzip libtorch-cxx11-abi-shared-with-deps-2.5.1+cu124.zip
rm libtorch-cxx11-abi-shared-with-deps-2.5.1+cu124.zip

# Create symlink in third_party
cd /path/to/millenniumdb/third_party
ln -s /path/to/your/libs/libtorch libtorch

# CPU-only version (if no CUDA)
wget https://download.pytorch.org/libtorch/cpu/libtorch-cxx11-abi-shared-with-deps-2.5.1%2Bcpu.zip
```

**Requirements:**
- CUDA 12.4+ (for GPU version)
- cuDNN 8.x (for GPU version)
- ~2GB disk space

**Verify installation:**
```bash
ls third_party/libtorch/lib/libtorch.so
```

---

### 3. libnpy (Header-only)

**Location:** `third_party/libnpy/`

**Purpose:** Read/write NumPy `.npy` files for loading pre-computed embeddings and feature vectors.

**Installation:**
```bash
cd third_party
mkdir -p libnpy
wget -O libnpy/npy.hpp https://raw.githubusercontent.com/llohse/libnpy/master/include/npy.hpp
```

**Source:** https://github.com/llohse/libnpy

**Usage:** Single header include:
```cpp
#include "third_party/libnpy/npy.hpp"
```

---

### 4. ska_sort (Header-only)

**Location:** `third_party/ska_sort/`

**Purpose:** High-performance radix sort implementation for efficient sorting of large datasets during projection operations.

**Installation:**
```bash
cd third_party
mkdir -p ska_sort
wget -O ska_sort/ska_sort.hpp https://raw.githubusercontent.com/skarupke/ska_sort/master/ska_sort.hpp
```

**Source:** https://github.com/skarupke/ska_sort

**Usage:** Single header include:
```cpp
#include "third_party/ska_sort/ska_sort.hpp"
```

---

## Tracked Dependencies

### ANTLR4 Runtime 4.13.1

**Location:** `third_party/antlr4-runtime-4.13.1/` (tracked in repository)

**Purpose:** Parser runtime for SPARQL, MQL, and GQL query languages.

**Note:** This dependency is included in the repository and does not require separate installation.

---

## Quick Setup Script

For convenience, here's a script to install all header-only dependencies:

```bash
#!/bin/bash
# Run from repository root

cd third_party

# libnpy
mkdir -p libnpy
wget -O libnpy/npy.hpp https://raw.githubusercontent.com/llohse/libnpy/master/include/npy.hpp

# ska_sort
mkdir -p ska_sort
wget -O ska_sort/ska_sort.hpp https://raw.githubusercontent.com/skarupke/ska_sort/master/ska_sort.hpp

echo "Header-only dependencies installed."
echo "Remember to install Boost and LibTorch separately (see above)."
```

---

## Version Compatibility Matrix

| Dependency | Minimum Version | Tested Version | Notes |
|------------|-----------------|----------------|-------|
| Boost | 1.74 | 1.82 | Header-only usage |
| LibTorch | 2.0 | 2.5.1+cu124 | CUDA 12.4 |
| libnpy | - | latest | Header-only |
| ska_sort | - | latest | Header-only |
| ANTLR4 | 4.13.1 | 4.13.1 | Exact version required |
