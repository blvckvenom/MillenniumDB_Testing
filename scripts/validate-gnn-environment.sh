#!/bin/bash
# ============================================================================
# GNN Environment Validation Script
# ============================================================================
#
# Validates that the development environment is properly configured for
# GNN development with MillenniumDB.
#
# Usage:
#   ./scripts/validate-gnn-environment.sh
#
# Exit codes:
#   0 - All checks passed
#   1 - Critical failure (cannot proceed)
#   2 - Warnings present (can proceed with limitations)
# ============================================================================

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Counters
WARNINGS=0
ERRORS=0

# ============================================================================
# Helper Functions
# ============================================================================

print_header() {
    echo ""
    echo -e "${BLUE}=== $1 ===${NC}"
}

print_ok() {
    echo -e "  ${GREEN}✓${NC} $1"
}

print_warn() {
    echo -e "  ${YELLOW}⚠${NC} $1"
    # Post-increment returns the OLD value, so at 0 it returns 0, which is a
    # false exit status and kills the script under 'set -e' on the very first
    # warning. Assignment always succeeds.
    WARNINGS=$((WARNINGS + 1))
}

print_error() {
    echo -e "  ${RED}✗${NC} $1"
    ERRORS=$((ERRORS + 1))
}

# ============================================================================
# Check Script Location
# ============================================================================

print_header "Environment Validation for GNN Development"

# Determine project root (script should be in scripts/)
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$( cd "$SCRIPT_DIR/.." && pwd )"

echo "  Project root: $PROJECT_ROOT"

if [ ! -f "$PROJECT_ROOT/CMakeLists.txt" ]; then
    print_error "Cannot find CMakeLists.txt - are you in the MillenniumDB directory?"
    exit 1
fi

cd "$PROJECT_ROOT"

# ============================================================================
# Check 1: LibTorch Installation
# ============================================================================

print_header "LibTorch Installation"

LIBTORCH_DIR="$PROJECT_ROOT/third_party/libtorch"

if [ -d "$LIBTORCH_DIR" ]; then
    print_ok "LibTorch directory found"

    # Check version
    if [ -f "$LIBTORCH_DIR/build-version" ]; then
        VERSION=$(cat "$LIBTORCH_DIR/build-version")
        print_ok "LibTorch version: $VERSION"

        # Extract major version
        MAJOR_VERSION=$(echo "$VERSION" | cut -d'.' -f1)
        if [ "$MAJOR_VERSION" -lt 2 ]; then
            print_warn "LibTorch 2.x recommended (found $VERSION)"
        fi
    else
        print_warn "LibTorch version unknown (build-version not found)"
    fi

    # Check for required libraries
    if [ -f "$LIBTORCH_DIR/lib/libtorch.so" ]; then
        print_ok "libtorch.so found"
    else
        print_error "libtorch.so not found in $LIBTORCH_DIR/lib/"
    fi

    # Check for headers
    if [ -d "$LIBTORCH_DIR/include/torch" ]; then
        print_ok "LibTorch headers found"
    else
        print_error "LibTorch headers not found"
    fi
else
    print_error "LibTorch NOT found in third_party/libtorch/"
    echo "         Download from: https://pytorch.org/get-started/locally/"
    echo "         Extract to: $LIBTORCH_DIR"
fi

# ============================================================================
# Check 2: CUDA Installation (Optional)
# ============================================================================

print_header "CUDA Support (Optional)"

if command -v nvcc &> /dev/null; then
    NVCC_VERSION=$(nvcc --version | grep "release" | sed 's/.*release \([0-9.]*\).*/\1/')
    print_ok "CUDA compiler found: nvcc $NVCC_VERSION"

    # Check nvidia-smi
    if command -v nvidia-smi &> /dev/null; then
        GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)
        if [ -n "$GPU_NAME" ]; then
            print_ok "GPU detected: $GPU_NAME"
        else
            print_warn "nvidia-smi available but no GPU detected"
        fi
    else
        print_warn "nvidia-smi not found - cannot detect GPU"
    fi

    # Check CUDA toolkit version in LibTorch
    if [ -f "$LIBTORCH_DIR/build-version" ]; then
        if grep -q "cu" "$LIBTORCH_DIR/build-version"; then
            CUDA_IN_LIBTORCH=$(cat "$LIBTORCH_DIR/build-version" | grep -oE 'cu[0-9]+' | sed 's/cu//')
            print_ok "LibTorch CUDA version: $CUDA_IN_LIBTORCH"
        else
            print_warn "LibTorch appears to be CPU-only build"
        fi
    fi
else
    print_warn "CUDA not found - GPU acceleration disabled"
    echo "         Install CUDA Toolkit: https://developer.nvidia.com/cuda-downloads"
fi

# ============================================================================
# Check 3: C++ Compiler
# ============================================================================

print_header "C++ Compiler"

if command -v g++ &> /dev/null; then
    GCC_VERSION=$(g++ -dumpversion)
    print_ok "g++ version: $GCC_VERSION"

    # Check minimum version (8.1 required)
    MAJOR_VERSION=$(echo "$GCC_VERSION" | cut -d'.' -f1)
    if [ "$MAJOR_VERSION" -lt 8 ]; then
        print_error "GCC 8.1+ required (found $GCC_VERSION)"
    fi
else
    print_error "g++ not found"
fi

# Check for C++17 support
echo -e '  Testing C++17 support...'
if echo 'int main() { if constexpr (true) return 0; }' | g++ -std=c++17 -x c++ - -o /dev/null 2>/dev/null; then
    print_ok "C++17 supported"
else
    print_error "C++17 not supported"
fi

# ============================================================================
# Check 4: CMake
# ============================================================================

print_header "CMake"

if command -v cmake &> /dev/null; then
    CMAKE_VERSION=$(cmake --version | head -1 | sed 's/.*version //')
    print_ok "CMake version: $CMAKE_VERSION"

    # Check minimum version (3.18 required for LibTorch)
    MAJOR=$(echo "$CMAKE_VERSION" | cut -d'.' -f1)
    MINOR=$(echo "$CMAKE_VERSION" | cut -d'.' -f2)
    if [ "$MAJOR" -lt 3 ] || ([ "$MAJOR" -eq 3 ] && [ "$MINOR" -lt 18 ]); then
        print_error "CMake 3.18+ required (found $CMAKE_VERSION)"
    fi
else
    print_error "CMake not found"
fi

# ============================================================================
# Check 5: Boost
# ============================================================================

print_header "Boost Library"

BOOST_DIR="$PROJECT_ROOT/third_party/boost_1_82"

if [ -d "$BOOST_DIR/include/boost" ]; then
    print_ok "Boost 1.82 headers found"
else
    print_error "Boost 1.82 not found in third_party/boost_1_82/"
    echo "         Download: https://archives.boost.io/release/1.82.0/source/boost_1_82_0.tar.gz"
fi

# ============================================================================
# Check 6: Python (for tests)
# ============================================================================

print_header "Python (for testing)"

if command -v python3 &> /dev/null; then
    PYTHON_VERSION=$(python3 --version 2>&1 | sed 's/Python //')
    print_ok "Python version: $PYTHON_VERSION"

    # Check for venv
    if python3 -c "import venv" 2>/dev/null; then
        print_ok "Python venv module available"
    else
        print_warn "Python venv module not found"
    fi
else
    print_warn "Python 3 not found (needed for tests)"
fi

# ============================================================================
# Check 7: GNN Module Files
# ============================================================================

print_header "GNN Module Structure"

GNN_DIR="$PROJECT_ROOT/src/gnn"

if [ -d "$GNN_DIR" ]; then
    print_ok "GNN module directory exists"

    # Check for key files
    KEY_FILES=(
        "CMakeLists.txt"
        "core/cuda_context.h"
            "storage/gnn_tensor_store.h"
        "storage/gnn_tensor_converter.h"
        "projection/feature_accessor.h"
    )

    for file in "${KEY_FILES[@]}"; do
        if [ -f "$GNN_DIR/$file" ]; then
            print_ok "Found $file"
        else
            print_warn "Missing $file"
        fi
    done
else
    print_error "GNN module directory not found at $GNN_DIR"
fi

# ============================================================================
# Check 8: Try Configuration
# ============================================================================

print_header "CMake Configuration Test"

BUILD_DIR="$PROJECT_ROOT/build/gnn-env-test"
rm -rf "$BUILD_DIR"

echo "  Running cmake..."
if cmake -B "$BUILD_DIR" -D CMAKE_BUILD_TYPE=Debug -D ENABLE_GNN=ON 2>&1 | tail -5; then
    if [ -f "$BUILD_DIR/Makefile" ] || [ -f "$BUILD_DIR/build.ninja" ]; then
        print_ok "CMake configuration succeeded"
    else
        print_error "CMake configuration produced no build files"
    fi
else
    print_error "CMake configuration failed"
fi

# Cleanup
rm -rf "$BUILD_DIR"

# ============================================================================
# Summary
# ============================================================================

print_header "Summary"

echo ""
if [ $ERRORS -gt 0 ]; then
    echo -e "${RED}✗ $ERRORS error(s) found - cannot proceed with GNN development${NC}"
    exit 1
elif [ $WARNINGS -gt 0 ]; then
    echo -e "${YELLOW}⚠ $WARNINGS warning(s) found - can proceed with limitations${NC}"
    exit 2
else
    echo -e "${GREEN}✓ All checks passed - ready for GNN development${NC}"
    exit 0
fi
