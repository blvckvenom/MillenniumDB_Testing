#!/usr/bin/env bash
# ============================================================================
# MillenniumDB feature-GNN — Automated Onboarding
# ============================================================================
# Installs every dependency needed to build and test this repository on a
# fresh Ubuntu / Pop!_OS 22.04+ machine, including NVIDIA driver, CUDA Toolkit,
# Boost 1.82, LibTorch (version chosen for the target GPU), header-only deps,
# Python test venvs, builds Release with ENABLE_GNN=ON, and runs the unit
# test suite as a smoke check.
#
# Usage:
#   scripts/onboard.sh                  # auto-detect everything
#   scripts/onboard.sh --gpu=5070       # force Blackwell stack (CUDA 12.8 + LibTorch 2.7 cu128)
#   scripts/onboard.sh --gpu=4070       # force Ada stack (CUDA 12.4 + LibTorch 2.5.1 cu124)
#   scripts/onboard.sh --no-gpu         # CPU-only build (LibTorch cpu, no CUDA Toolkit)
#   scripts/onboard.sh --skip-driver    # assume NVIDIA driver + CUDA are already installed
#   scripts/onboard.sh --skip-tests     # don't run ./scripts/run-tests at the end
#   scripts/onboard.sh --resume         # skip apt+driver phase (for use after reboot)
#
# Idempotent: every step checks its own completion marker and skips if already done.
# ============================================================================

set -euo pipefail

# ---------------------------------------------------------------------------
# Flags
# ---------------------------------------------------------------------------
GPU_CHOICE="auto"
SKIP_DRIVER=0
SKIP_TESTS=0
RESUME=0
for arg in "$@"; do
    case "$arg" in
        --gpu=*)        GPU_CHOICE="${arg#*=}" ;;
        --no-gpu)       GPU_CHOICE="none" ;;
        --skip-driver)  SKIP_DRIVER=1 ;;
        --skip-tests)   SKIP_TESTS=1 ;;
        --resume)       RESUME=1 ;;
        -h|--help)
            grep -E '^# ' "$0" | sed 's/^# \{0,1\}//' | head -25
            exit 0
            ;;
        *) echo "Unknown flag: $arg" >&2; exit 1 ;;
    esac
done

# ---------------------------------------------------------------------------
# Colors + logging
# ---------------------------------------------------------------------------
if [ -t 1 ]; then
    RED=$'\033[0;31m'; GREEN=$'\033[0;32m'; YELLOW=$'\033[1;33m'
    BLUE=$'\033[0;34m'; BOLD=$'\033[1m'; NC=$'\033[0m'
else
    RED=''; GREEN=''; YELLOW=''; BLUE=''; BOLD=''; NC=''
fi

log_step()  { echo -e "\n${BLUE}${BOLD}━━━ $* ━━━${NC}"; }
log_ok()    { echo -e "  ${GREEN}✓${NC} $*"; }
log_warn()  { echo -e "  ${YELLOW}⚠${NC} $*"; }
log_err()   { echo -e "  ${RED}✗${NC} $*" >&2; }
log_info()  { echo -e "  ${BLUE}ℹ${NC} $*"; }
die()       { log_err "$*"; exit 1; }

# ---------------------------------------------------------------------------
# Prerequisites
# ---------------------------------------------------------------------------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MDB_HOME="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$MDB_HOME"

[ -f "$MDB_HOME/CMakeLists.txt" ] || die "Not in MillenniumDB root ($MDB_HOME)"
log_info "Project root: $MDB_HOME"

# pkexec is the global preference; fall back to sudo if not available.
if command -v pkexec >/dev/null 2>&1; then
    PRIV="pkexec"
elif command -v sudo >/dev/null 2>&1; then
    PRIV="sudo"
    log_warn "pkexec not found, falling back to sudo"
else
    die "Neither pkexec nor sudo available"
fi

# ---------------------------------------------------------------------------
# Step 1 — Detect GPU and resolve stack versions
# ---------------------------------------------------------------------------
log_step "Step 1: GPU detection"

detect_gpu_model() {
    if ! command -v lspci >/dev/null 2>&1; then
        $PRIV apt-get install -y pciutils >/dev/null 2>&1 || true
    fi
    local gpu
    gpu="$(lspci 2>/dev/null | grep -iE 'vga|3d' | grep -i nvidia || true)"
    echo "$gpu"
}

GPU_LINE="$(detect_gpu_model)"
if [ -n "$GPU_LINE" ]; then
    log_ok "NVIDIA GPU found: $GPU_LINE"
else
    log_warn "No NVIDIA GPU detected via lspci"
fi

if [ "$GPU_CHOICE" = "auto" ]; then
    if echo "$GPU_LINE" | grep -qiE '50[789]0|5060|rtx.5'; then
        GPU_CHOICE="5070"
    elif echo "$GPU_LINE" | grep -qiE '40[6-9]0|4050|rtx.4'; then
        GPU_CHOICE="4070"
    elif [ -n "$GPU_LINE" ]; then
        log_warn "GPU detected but unknown generation; defaulting to 4070 stack (CUDA 12.4)"
        GPU_CHOICE="4070"
    else
        GPU_CHOICE="none"
    fi
fi

case "$GPU_CHOICE" in
    5070)
        CUDA_VER="12.8.0"
        CUDA_VER_SHORT="12.8"
        CUDA_RUN_URL="https://developer.download.nvidia.com/compute/cuda/12.8.0/local_installers/cuda_12.8.0_570.86.10_linux.run"
        DRIVER_PKG="nvidia-driver-570"
        LIBTORCH_URL="https://download.pytorch.org/libtorch/cu128/libtorch-cxx11-abi-shared-with-deps-2.7.0%2Bcu128.zip"
        LIBTORCH_ZIP="libtorch-cxx11-abi-shared-with-deps-2.7.0+cu128.zip"
        LIBTORCH_EXPECTED="2.7.0+cu128"
        ENABLE_GPU=1
        ;;
    4070)
        CUDA_VER="12.4.0"
        CUDA_VER_SHORT="12.4"
        CUDA_RUN_URL="https://developer.download.nvidia.com/compute/cuda/12.4.0/local_installers/cuda_12.4.0_550.54.14_linux.run"
        DRIVER_PKG="nvidia-driver-550"
        LIBTORCH_URL="https://download.pytorch.org/libtorch/cu124/libtorch-cxx11-abi-shared-with-deps-2.5.1%2Bcu124.zip"
        LIBTORCH_ZIP="libtorch-cxx11-abi-shared-with-deps-2.5.1+cu124.zip"
        LIBTORCH_EXPECTED="2.5.1+cu124"
        ENABLE_GPU=1
        ;;
    none)
        LIBTORCH_URL="https://download.pytorch.org/libtorch/cpu/libtorch-cxx11-abi-shared-with-deps-2.5.1%2Bcpu.zip"
        LIBTORCH_ZIP="libtorch-cxx11-abi-shared-with-deps-2.5.1+cpu.zip"
        LIBTORCH_EXPECTED="2.5.1+cpu"
        ENABLE_GPU=0
        ;;
    *) die "Invalid --gpu choice: $GPU_CHOICE (use 5070 | 4070 | none)" ;;
esac

log_ok "Stack selected: GPU=$GPU_CHOICE, LibTorch=$LIBTORCH_EXPECTED, CUDA=${CUDA_VER_SHORT:-none}"

# ---------------------------------------------------------------------------
# Step 2 — APT packages
# ---------------------------------------------------------------------------
if [ "$RESUME" -eq 0 ]; then
    log_step "Step 2: APT system packages"
    APT_PKGS=(
        git g++ cmake make pkg-config
        libssl-dev libncurses-dev less
        python3 python3-venv python3-pip
        libicu-dev libtbb-dev liburing-dev
        openjdk-21-jdk zsh unzip wget curl ca-certificates
        clangd nodejs npm
        pciutils build-essential
    )
    $PRIV apt-get update
    $PRIV apt-get install -y "${APT_PKGS[@]}"
    log_ok "APT packages installed"
else
    log_info "Skipping APT phase (--resume)"
fi

# ---------------------------------------------------------------------------
# Step 3 — NVIDIA driver + CUDA Toolkit
# ---------------------------------------------------------------------------
if [ "$ENABLE_GPU" -eq 1 ] && [ "$SKIP_DRIVER" -eq 0 ] && [ "$RESUME" -eq 0 ]; then
    log_step "Step 3: NVIDIA driver + CUDA Toolkit $CUDA_VER_SHORT"

    DRIVER_OK=0
    if command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi -q >/dev/null 2>&1; then
        CURRENT_DRIVER="$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1)"
        log_info "Current driver: $CURRENT_DRIVER"
        # Compare major version against required (570 for 5070, 550 for 4070)
        REQUIRED_MAJOR="${DRIVER_PKG##*-}"
        CURRENT_MAJOR="${CURRENT_DRIVER%%.*}"
        if [ "$CURRENT_MAJOR" -ge "$REQUIRED_MAJOR" ]; then
            log_ok "Driver $CURRENT_DRIVER satisfies required >= $REQUIRED_MAJOR"
            DRIVER_OK=1
        fi
    fi

    if [ "$DRIVER_OK" -eq 0 ]; then
        log_info "Installing $DRIVER_PKG (reboot required after this step)"
        $PRIV apt-get install -y "$DRIVER_PKG" nvidia-utils-"${DRIVER_PKG##*-}"
        log_warn "Driver installed. REBOOT and re-run: $0 --resume --gpu=$GPU_CHOICE"
        touch "$MDB_HOME/.onboard_needs_reboot"
        exit 0
    fi

    if [ -f "$MDB_HOME/.onboard_needs_reboot" ]; then
        rm "$MDB_HOME/.onboard_needs_reboot"
    fi

    # CUDA Toolkit check
    CUDA_INSTALL_DIR="/usr/local/cuda-$CUDA_VER_SHORT"
    if [ -d "$CUDA_INSTALL_DIR" ] && [ -x "$CUDA_INSTALL_DIR/bin/nvcc" ]; then
        log_ok "CUDA Toolkit $CUDA_VER_SHORT already installed at $CUDA_INSTALL_DIR"
    else
        log_info "Downloading CUDA Toolkit $CUDA_VER installer"
        cd /tmp
        CUDA_RUN="$(basename "$CUDA_RUN_URL")"
        [ -f "$CUDA_RUN" ] || wget -q --show-progress "$CUDA_RUN_URL"
        log_info "Installing CUDA Toolkit (silent, toolkit-only, no driver)"
        $PRIV sh "$CUDA_RUN" --toolkit --silent --override --no-opengl-libs
        cd "$MDB_HOME"
        log_ok "CUDA Toolkit $CUDA_VER_SHORT installed"
    fi

    # Symlink /usr/local/cuda -> /usr/local/cuda-X.Y
    if [ ! -L /usr/local/cuda ] || [ "$(readlink /usr/local/cuda)" != "cuda-$CUDA_VER_SHORT" ]; then
        $PRIV ln -sfn "cuda-$CUDA_VER_SHORT" /usr/local/cuda
        log_ok "Symlinked /usr/local/cuda -> cuda-$CUDA_VER_SHORT"
    fi
elif [ "$ENABLE_GPU" -eq 0 ]; then
    log_info "Skipping GPU stack (--no-gpu or no NVIDIA detected)"
else
    log_info "Skipping driver install (--skip-driver or --resume)"
fi

# ---------------------------------------------------------------------------
# Step 4 — Boost 1.82 headers
# ---------------------------------------------------------------------------
log_step "Step 4: Boost 1.82"
BOOST_INCLUDE="$MDB_HOME/third_party/boost_1_82/include/boost"
if [ -d "$BOOST_INCLUDE" ] && [ -f "$BOOST_INCLUDE/version.hpp" ]; then
    log_ok "Boost 1.82 headers already present"
else
    cd /tmp
    [ -f boost_1_82_0.tar.gz ] || wget -q --show-progress \
        https://archives.boost.io/release/1.82.0/source/boost_1_82_0.tar.gz
    rm -rf boost_1_82_0
    tar -xf boost_1_82_0.tar.gz
    mkdir -p "$MDB_HOME/third_party/boost_1_82/include"
    mv boost_1_82_0/boost "$MDB_HOME/third_party/boost_1_82/include/"
    rm -rf boost_1_82_0
    cd "$MDB_HOME"
    log_ok "Boost 1.82 installed"
fi

# ---------------------------------------------------------------------------
# Step 5 — LibTorch
# ---------------------------------------------------------------------------
log_step "Step 5: LibTorch $LIBTORCH_EXPECTED"
LIBS_DIR="$HOME/libs"
mkdir -p "$LIBS_DIR"
LIBTORCH_DIR="$LIBS_DIR/libtorch"
INSTALL_LIBTORCH=1
if [ -d "$LIBTORCH_DIR" ] && [ -f "$LIBTORCH_DIR/build-version" ]; then
    CURRENT_TORCH="$(cat "$LIBTORCH_DIR/build-version")"
    if [ "$CURRENT_TORCH" = "$LIBTORCH_EXPECTED" ]; then
        log_ok "LibTorch $CURRENT_TORCH already installed"
        INSTALL_LIBTORCH=0
    else
        log_warn "Found LibTorch $CURRENT_TORCH, need $LIBTORCH_EXPECTED — replacing"
        rm -rf "$LIBTORCH_DIR"
    fi
fi

if [ "$INSTALL_LIBTORCH" -eq 1 ]; then
    cd "$LIBS_DIR"
    [ -f "$LIBTORCH_ZIP" ] || wget -q --show-progress "$LIBTORCH_URL"
    log_info "Unzipping LibTorch (this takes ~1 minute)"
    unzip -q "$LIBTORCH_ZIP"
    cd "$MDB_HOME"
    log_ok "LibTorch $LIBTORCH_EXPECTED installed at $LIBTORCH_DIR"
fi

# Symlink into third_party
TP_LIBTORCH="$MDB_HOME/third_party/libtorch"
if [ -L "$TP_LIBTORCH" ] && [ "$(readlink -f "$TP_LIBTORCH")" = "$(readlink -f "$LIBTORCH_DIR")" ]; then
    log_ok "third_party/libtorch symlink correct"
else
    rm -rf "$TP_LIBTORCH"
    ln -sfn "$LIBTORCH_DIR" "$TP_LIBTORCH"
    log_ok "Linked third_party/libtorch -> $LIBTORCH_DIR"
fi

# ---------------------------------------------------------------------------
# Step 6 — Header-only deps (libnpy, ska_sort)
# ---------------------------------------------------------------------------
log_step "Step 6: Header-only third_party deps"
cd "$MDB_HOME/third_party"

if [ -f libnpy/include/npy.hpp ]; then
    log_ok "libnpy present"
else
    mkdir -p libnpy/include
    wget -q -O libnpy/include/npy.hpp \
        https://raw.githubusercontent.com/llohse/libnpy/master/include/npy.hpp
    log_ok "libnpy downloaded"
fi

if [ -f ska_sort/ska_sort.hpp ]; then
    log_ok "ska_sort present"
else
    mkdir -p ska_sort
    wget -q -O ska_sort/ska_sort.hpp \
        https://raw.githubusercontent.com/skarupke/ska_sort/master/ska_sort.hpp
    log_ok "ska_sort downloaded"
fi
cd "$MDB_HOME"

# ---------------------------------------------------------------------------
# Step 7 — ANTLR4 jar (only needed to regenerate parsers; optional)
# ---------------------------------------------------------------------------
log_step "Step 7: ANTLR4 jar (optional, for grammar regeneration)"
ANTLR_JAR="/usr/local/lib/antlr-4.13.1-complete.jar"
if [ -f "$ANTLR_JAR" ]; then
    log_ok "ANTLR4 jar present at $ANTLR_JAR"
else
    $PRIV curl -fsSL -o "$ANTLR_JAR" \
        https://www.antlr.org/download/antlr-4.13.1-complete.jar \
        && log_ok "ANTLR4 jar installed" \
        || log_warn "Failed to install ANTLR jar (skipping — only needed if you edit .g4 grammars)"
fi

# ---------------------------------------------------------------------------
# Step 8 — Python test venvs
# ---------------------------------------------------------------------------
log_step "Step 8: Python test venvs"
setup_venv() {
    local dir="$1"
    if [ ! -f "$dir/requirements.txt" ]; then
        log_warn "$dir has no requirements.txt"
        return
    fi
    if [ -d "$dir/.venv" ]; then
        log_ok "venv at $dir/.venv already exists"
        return
    fi
    python3 -m venv "$dir/.venv"
    # shellcheck disable=SC1091
    source "$dir/.venv/bin/activate"
    pip install -q --upgrade pip
    pip install -q -r "$dir/requirements.txt"
    deactivate
    log_ok "venv created at $dir/.venv"
}
setup_venv "$MDB_HOME/tests"
setup_venv "$MDB_HOME/tests_Proj_GNN"

# ---------------------------------------------------------------------------
# Step 9 — Environment variables for current shell and ~/.bashrc
# ---------------------------------------------------------------------------
log_step "Step 9: Environment variables"
BASHRC_MARKER="# === MillenniumDB onboarding env ==="
if ! grep -q "$BASHRC_MARKER" "$HOME/.bashrc" 2>/dev/null; then
    {
        echo ""
        echo "$BASHRC_MARKER"
        echo "export MDB_HOME=\"$MDB_HOME\""
        if [ "$ENABLE_GPU" -eq 1 ]; then
            echo "export CUDA_HOME=/usr/local/cuda"
            echo 'export PATH=$CUDA_HOME/bin:$PATH'
            echo 'export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH'
        fi
        echo 'export LD_LIBRARY_PATH=$MDB_HOME/third_party/libtorch/lib:${LD_LIBRARY_PATH:-}'
        echo "# === end MillenniumDB ==="
    } >> "$HOME/.bashrc"
    log_ok "Appended env to ~/.bashrc"
else
    log_ok "~/.bashrc already has MillenniumDB env block"
fi

# Export for the current shell so the build below can see CUDA
export MDB_HOME
if [ "$ENABLE_GPU" -eq 1 ]; then
    export CUDA_HOME=/usr/local/cuda
    export PATH="$CUDA_HOME/bin:$PATH"
    export LD_LIBRARY_PATH="$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}"
fi
export LD_LIBRARY_PATH="$MDB_HOME/third_party/libtorch/lib:${LD_LIBRARY_PATH:-}"

# ---------------------------------------------------------------------------
# Step 10 — Build
# ---------------------------------------------------------------------------
log_step "Step 10: Build Release (ENABLE_GNN=ON)"
cd "$MDB_HOME"

CMAKE_FLAGS=(-D CMAKE_BUILD_TYPE=Release -D ENABLE_GNN=ON)
if [ "$ENABLE_GPU" -eq 0 ]; then
    # Force-disable CUDA detection when user asked for CPU-only
    CMAKE_FLAGS+=(-D CMAKE_DISABLE_FIND_PACKAGE_CUDAToolkit=ON)
fi

cmake -B build/Release "${CMAKE_FLAGS[@]}"
cmake --build build/Release -j "$(nproc)"

# Generate compile_commands.json symlink for clangd/cclsp
ln -sf build/Release/compile_commands.json "$MDB_HOME/compile_commands.json"

if "$MDB_HOME/build/Release/bin/mdb" help >/dev/null 2>&1; then
    log_ok "Binary works: $MDB_HOME/build/Release/bin/mdb"
else
    die "Binary failed smoke test (mdb help)"
fi

# ---------------------------------------------------------------------------
# Step 11 — Validate GNN environment + run tests
# ---------------------------------------------------------------------------
log_step "Step 11: Validation"
if [ -x "$MDB_HOME/scripts/validate-gnn-environment.sh" ]; then
    bash "$MDB_HOME/scripts/validate-gnn-environment.sh" || \
        log_warn "validate-gnn-environment reported warnings — review output above"
fi

if [ "$SKIP_TESTS" -eq 0 ]; then
    log_step "Step 12: Test suite (unit tests only — full suite is long)"
    bash "$MDB_HOME/scripts/run-tests" unit || \
        log_warn "Some unit tests failed — review ./tests/ output"
else
    log_info "Skipping tests (--skip-tests)"
fi

# ---------------------------------------------------------------------------
# Done
# ---------------------------------------------------------------------------
echo ""
echo -e "${GREEN}${BOLD}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "${GREEN}${BOLD}  Onboarding complete ✓${NC}"
echo -e "${GREEN}${BOLD}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo ""
echo "  Binary:   $MDB_HOME/build/Release/bin/mdb help"
echo "  Stack:    LibTorch=$LIBTORCH_EXPECTED, GPU=$GPU_CHOICE"
echo ""
echo "  Next steps:"
echo "    source ~/.bashrc               # reload env in new shells"
echo "    ./scripts/run-tests gql        # full GQL test suite"
echo "    ./scripts/run-tests            # every suite (unit+sparql+mql+gql)"
if [ "$ENABLE_GPU" -eq 1 ]; then
    echo "    ./scripts/verify_gnn_integration.sh   # GNN pipeline smoke test"
fi
echo ""
