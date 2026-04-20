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
#   scripts/onboard.sh --dry-run        # show root commands that WOULD run, skip pkexec
#   scripts/onboard.sh --download-dataset=papers100M       # raw OGB download at end
#   scripts/onboard.sh --download-dataset=arxiv,products   # multiple datasets
#
# Idempotent: every step checks its own completion marker and skips if already done.
#
# Privilege model:
#   All root-requiring commands are accumulated into a single batch script and
#   invoked with ONE pkexec call (ONE admin password prompt for the whole run).
#   Use --dry-run to preview the batch contents without authenticating.
# ============================================================================

set -euo pipefail

# ---------------------------------------------------------------------------
# Flags
# ---------------------------------------------------------------------------
GPU_CHOICE="auto"
SKIP_DRIVER=0
SKIP_TESTS=0
RESUME=0
DRY_RUN=0
DOWNLOAD_DATASETS=""   # comma-separated names, empty = no dataset pre-staging
for arg in "$@"; do
    case "$arg" in
        --gpu=*)                GPU_CHOICE="${arg#*=}" ;;
        --no-gpu)               GPU_CHOICE="none" ;;
        --skip-driver)          SKIP_DRIVER=1 ;;
        --skip-tests)           SKIP_TESTS=1 ;;
        --resume)               RESUME=1 ;;
        --dry-run)              DRY_RUN=1 ;;
        --download-dataset=*)   DOWNLOAD_DATASETS="${arg#*=}" ;;
        -h|--help)
            grep -E '^# ' "$0" | sed 's/^# \{0,1\}//' | head -35
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
# Root-command batching
# ---------------------------------------------------------------------------
# All root-requiring commands go into a single batch script that is invoked
# once with pkexec/sudo. Admin types their password a single time regardless
# of how many individual apt/install/symlink operations the script needs.
#
# Each queue_root call appends a command + a human-readable log marker to
# the batch file. execute_root_batch then invokes the batch under $PRIV.
# If the batch file is empty (nothing was queued), execute_root_batch is a
# no-op — so on --resume or idempotent re-runs where all root work is
# already done, no prompt is shown.
# ---------------------------------------------------------------------------

ROOT_BATCH_FILE=""  # Populated by init_root_batch after LOG_DIR exists.

init_root_batch() {
    ROOT_BATCH_FILE="$LOG_DIR/root-phase.sh"
    cat > "$ROOT_BATCH_FILE" << 'BATCH_HDR'
#!/usr/bin/env bash
# Auto-generated batch script. Runs under pkexec/sudo as root.
# Each command is prefixed by a ">>> <label>" echo for traceability.
set -euo pipefail
echo "=== MillenniumDB root batch starting (uid=$(id -u), $(date)) ==="
BATCH_HDR
    chmod +x "$ROOT_BATCH_FILE"
}

# queue_root "<human label>" -- <shell command...>
# The command is stored verbatim; $LOG_DIR and $MDB_HOME are resolved
# by the parent shell, so the batch file contains absolute paths.
queue_root() {
    local label="$1"; shift
    {
        echo ""
        echo "# --- $label ---"
        printf 'echo ">>> %s"\n' "$label"
        printf '%s\n' "$*"
    } >> "$ROOT_BATCH_FILE"
}

execute_root_batch() {
    local label="${1:-pre-reboot root phase}"

    # Empty batch (only the header stub) → nothing queued → no prompt needed.
    # Header is ~4 lines; any queue_root adds at least 4 more lines.
    if [ ! -f "$ROOT_BATCH_FILE" ] || [ "$(wc -l < "$ROOT_BATCH_FILE")" -le 6 ]; then
        log_info "No root commands queued — skipping pkexec prompt"
        return 0
    fi

    log_step "Root batch: $label"
    log_info "The following commands will run as root under ONE pkexec prompt:"
    grep -E '^# --- ' "$ROOT_BATCH_FILE" | sed 's/^# --- /    - /; s/ ---$//'
    log_info "Full batch script: $ROOT_BATCH_FILE"
    log_info "Admin can inspect it before typing the password."

    if [ "$DRY_RUN" -eq 1 ]; then
        log_warn "[DRY RUN] Skipping '$PRIV bash $ROOT_BATCH_FILE'"
        log_warn "[DRY RUN] Full batch contents below:"
        echo "───────────────── batch script begin ─────────────────"
        cat "$ROOT_BATCH_FILE"
        echo "────────────────── batch script end ──────────────────"
        return 0
    fi

    # Explicit warning so the user knows the password prompt coming up
    # expects the ADMINISTRATOR's password, not their own. Critical in
    # shared workstations where the user running the script has no sudo
    # and an admin is physically present to authenticate.
    local cmd_count
    cmd_count="$(grep -cE '^# --- ' "$ROOT_BATCH_FILE")"
    echo ""
    echo -e "${YELLOW}${BOLD}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo -e "${YELLOW}${BOLD}  ADMINISTRATOR PASSWORD REQUIRED${NC}"
    echo -e "${YELLOW}${BOLD}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo ""
    echo "  A GUI authentication dialog is about to appear on screen."
    echo "  It asks for the ADMINISTRATOR's password — NOT ${USER:-the"
    echo "  current user}'s password."
    echo ""
    echo "  If you are NOT the administrator of this workstation:"
    echo "    - hand the keyboard to whoever is, OR"
    echo "    - ask them to type their password in the dialog now."
    echo ""
    echo "  The one prompt covers $cmd_count root operations listed above."
    echo "  After authentication, admin is NOT needed again for the rest"
    echo "  of the onboarding run."
    echo ""
    echo -e "${YELLOW}  ⏳ Waiting for password...${NC}"
    echo ""

    # Real execution. tee stdout+stderr to a log file alongside the batch
    # so we can read what happened after the fact (e.g. if admin closed the
    # terminal window early).
    local batch_log="${ROOT_BATCH_FILE}.log"
    if $PRIV bash "$ROOT_BATCH_FILE" 2>&1 | tee "$batch_log"; then
        log_ok "Root batch completed (log: $batch_log)"
    else
        die "Root batch failed — see $batch_log for details"
    fi
}

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
# Step 0 — OS detection + persistent log directory
# ---------------------------------------------------------------------------
log_step "Step 0: OS detection + log directory"

# Detect Linux distribution + version; some package names vary per release
# (notably openjdk-21 is not in Ubuntu 22.04 by default, only 24.04+).
OS_ID="unknown"
OS_VERSION_ID="unknown"
if [ -f /etc/os-release ]; then
    # shellcheck disable=SC1091
    source /etc/os-release
    OS_ID="${ID:-unknown}"
    OS_VERSION_ID="${VERSION_ID:-unknown}"
fi

case "${OS_ID}:${OS_VERSION_ID}" in
    ubuntu:24.*|ubuntu:25.*|pop:24.*|pop:25.*|debian:13|debian:trixie|debian:sid)
        JDK_PKG="openjdk-21-jdk"
        ;;
    ubuntu:22.*|ubuntu:23.*|pop:22.*|pop:23.*|debian:12|debian:bookworm)
        JDK_PKG="openjdk-17-jdk"
        ;;
    *)
        JDK_PKG="default-jdk"
        log_warn "Unrecognized OS (${OS_ID} ${OS_VERSION_ID}); falling back to default-jdk"
        ;;
esac
log_ok "OS: ${OS_ID} ${OS_VERSION_ID}, JDK package: ${JDK_PKG}"

# Central log directory — cmake configure/build outputs tee here for post-
# mortem when the script is long-running and stdout is scrolled past.
LOG_DIR="/tmp/mdb-onboard-$(date +%Y%m%d-%H%M%S)"
mkdir -p "$LOG_DIR"
log_ok "Log dir: $LOG_DIR"

# Initialize the root-commands batch file. queue_root appends to it from
# Steps 2, 3, 7; execute_root_batch runs the whole thing at the end of the
# root-needing phase.
init_root_batch
log_ok "Root batch file initialized: $ROOT_BATCH_FILE"

[ "$DRY_RUN" -eq 1 ] && log_warn "DRY RUN mode: no pkexec/sudo will be invoked"

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
    log_step "Step 2: APT system packages (queued)"
    APT_PKGS=(
        git g++ cmake make pkg-config
        libssl-dev libncurses-dev less
        python3 python3-venv python3-pip
        libicu-dev libtbb-dev liburing-dev
        "$JDK_PKG" zsh unzip wget curl ca-certificates
        clangd nodejs npm
        pciutils build-essential
    )
    queue_root "apt-get update" "apt-get update"
    queue_root "apt-get install ${#APT_PKGS[@]} packages (git, g++, cmake, $JDK_PKG, ...)" \
        "apt-get install -y ${APT_PKGS[*]}"
    log_ok "APT commands queued for root batch"
else
    log_info "Skipping APT phase (--resume)"
fi

# ---------------------------------------------------------------------------
# Step 3 — NVIDIA driver + CUDA Toolkit
# ---------------------------------------------------------------------------
if [ "$ENABLE_GPU" -eq 1 ] && [ "$SKIP_DRIVER" -eq 0 ] && [ "$RESUME" -eq 0 ]; then
    log_step "Step 3: NVIDIA driver + CUDA Toolkit $CUDA_VER_SHORT"

    # Secure Boot + NVIDIA DKMS is the #1 cause of "driver installs OK but
    # nvidia-smi fails post-reboot" on fresh UEFI workstations. Warn loudly
    # — the user must either disable SB in UEFI, or enroll the MOK key when
    # the driver install prompts for a passphrase (MokManager screen on
    # next boot). Without one of those, the kernel refuses to load the
    # unsigned nvidia.ko and every CUDA call fails silently.
    if command -v mokutil >/dev/null 2>&1; then
        SB_STATE="$(mokutil --sb-state 2>/dev/null | head -1 || echo unknown)"
        if echo "$SB_STATE" | grep -qi enabled; then
            log_warn "Secure Boot is ENABLED."
            log_warn "  NVIDIA DKMS will build the module but not sign it."
            log_warn "  After reboot, nvidia-smi may fail until you:"
            log_warn "    a) disable Secure Boot in UEFI firmware, OR"
            log_warn "    b) enroll the MOK key — when dpkg asks for a passphrase"
            log_warn "       during driver install, remember it; on next boot"
            log_warn "       MokManager appears and you enroll it with that passphrase."
        else
            log_ok "Secure Boot: $SB_STATE"
        fi
    else
        log_info "mokutil not installed; cannot check Secure Boot state"
    fi

    # Flag: will be set to 1 if we're installing a driver this run. The
    # reboot checkpoint at the end of the script fires only when this is 1.
    NEEDS_REBOOT=0

    # --- Driver check + queue ---
    DRIVER_OK=0
    if command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi -q >/dev/null 2>&1; then
        CURRENT_DRIVER="$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1)"
        log_info "Current driver: $CURRENT_DRIVER"
        REQUIRED_MAJOR="${DRIVER_PKG##*-}"
        CURRENT_MAJOR="${CURRENT_DRIVER%%.*}"
        if [ "$CURRENT_MAJOR" -ge "$REQUIRED_MAJOR" ]; then
            log_ok "Driver $CURRENT_DRIVER satisfies required >= $REQUIRED_MAJOR"
            DRIVER_OK=1
        fi
    fi

    if [ "$DRIVER_OK" -eq 0 ]; then
        log_info "Queueing $DRIVER_PKG for install (reboot will be required)"
        queue_root "install NVIDIA driver $DRIVER_PKG" \
            "apt-get install -y $DRIVER_PKG nvidia-utils-${DRIVER_PKG##*-}"
        NEEDS_REBOOT=1
    fi

    # --- CUDA Toolkit pre-download (as user, no root) + queue install ---
    CUDA_INSTALL_DIR="/usr/local/cuda-$CUDA_VER_SHORT"
    if [ -d "$CUDA_INSTALL_DIR" ] && [ -x "$CUDA_INSTALL_DIR/bin/nvcc" ]; then
        log_ok "CUDA Toolkit $CUDA_VER_SHORT already installed at $CUDA_INSTALL_DIR"
    else
        CUDA_RUN="$(basename "$CUDA_RUN_URL")"
        CUDA_RUN_PATH="/tmp/$CUDA_RUN"
        if [ ! -f "$CUDA_RUN_PATH" ]; then
            log_info "Downloading CUDA Toolkit $CUDA_VER installer (as user)"
            if [ "$DRY_RUN" -eq 1 ]; then
                log_warn "[DRY RUN] Would wget $CUDA_RUN_URL to $CUDA_RUN_PATH"
            else
                cd /tmp
                wget -q --show-progress "$CUDA_RUN_URL"
                cd "$MDB_HOME"
            fi
        else
            log_ok "CUDA installer already cached at $CUDA_RUN_PATH"
        fi
        log_info "Queueing CUDA Toolkit install (toolkit-only, no driver, silent)"
        queue_root "install CUDA Toolkit $CUDA_VER_SHORT from $CUDA_RUN_PATH" \
            "sh $CUDA_RUN_PATH --toolkit --silent --no-opengl-libs"
    fi

    # --- CUDA symlink queue (idempotent on execution) ---
    # We queue unconditionally; the batched shell re-checks and no-ops if
    # the symlink already points where we want. Cheap, always correct.
    queue_root "symlink /usr/local/cuda -> cuda-$CUDA_VER_SHORT" \
        "ln -sfn cuda-$CUDA_VER_SHORT /usr/local/cuda"

elif [ "$ENABLE_GPU" -eq 0 ]; then
    log_info "Skipping GPU stack (--no-gpu or no NVIDIA detected)"
    NEEDS_REBOOT=0
else
    log_info "Skipping driver/CUDA queueing (--skip-driver or --resume)"
    NEEDS_REBOOT=0
fi

# ---------------------------------------------------------------------------
# Step 3b — ANTLR4 jar queue (root-owned, grouped with Step 3 for batching)
# ---------------------------------------------------------------------------
# Moved here from the old Step 7 position so that ALL root-requiring
# operations queue BEFORE the execute_root_batch call below. That keeps
# the contract "queue everything, then one pkexec prompt, then user ops".
log_step "Step 3b: ANTLR4 jar (optional, for grammar regeneration)"
ANTLR_JAR="/usr/local/lib/antlr-4.13.1-complete.jar"
ANTLR_URL="https://www.antlr.org/download/antlr-4.13.1-complete.jar"
if [ -f "$ANTLR_JAR" ]; then
    log_ok "ANTLR4 jar present at $ANTLR_JAR"
else
    log_info "Queueing ANTLR 4.13.1 jar download for root batch"
    queue_root "download ANTLR 4.13.1 jar to $ANTLR_JAR" \
        "mkdir -p $(dirname "$ANTLR_JAR") && curl -fsSL -o $ANTLR_JAR $ANTLR_URL"
fi

# ---------------------------------------------------------------------------
# Root batch execution — the ONE pkexec prompt for the whole onboarding.
# After this, no more root operations are needed; remaining steps (Boost,
# LibTorch, venvs, build, tests) all run as the invoking user.
# ---------------------------------------------------------------------------
execute_root_batch "Install APT packages + NVIDIA driver + CUDA Toolkit + ANTLR jar"

# In DRY-RUN mode, stop here. The subsequent user-phase steps (Boost,
# LibTorch, venvs, cmake build) perform filesystem mutations on the
# user's home directory — they are NOT wrapped under DRY_RUN and would
# execute for real. Exit early so a dry-run preview is side-effect-free.
if [ "$DRY_RUN" -eq 1 ]; then
    echo ""
    log_step "DRY RUN summary"
    log_info "Root phase: previewed above (one pkexec prompt, batch contents shown)."
    log_info "Remaining steps (Boost, LibTorch, venvs, cmake build, tests) would"
    log_info "run as \$USER without further privilege prompts; they are skipped"
    log_info "here to avoid any filesystem mutations."
    if [ -n "$DOWNLOAD_DATASETS" ]; then
        log_info "After build, would pre-stage OGB datasets: $DOWNLOAD_DATASETS"
        log_info "  (raw download only; happens BEFORE the reboot checkpoint"
        log_info "   so datasets finish even if a driver-install reboot is pending)"
    fi
    log_ok  "DRY RUN complete. No system changes were made."
    exit 0
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

# Step 7 (ANTLR jar queueing) was moved up into Step 3b so that all root
# operations queue before execute_root_batch. The root batch + DRY-RUN exit
# now live directly after Step 3b. Nothing to do at this position.

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

# Tee cmake output: visible live AND persisted to $LOG_DIR for post-hoc
# review. `set -o pipefail` (line 23) makes the pipeline fail if cmake
# fails, so tee doesn't swallow compiler errors.
cmake -B build/Release "${CMAKE_FLAGS[@]}" 2>&1 | tee "$LOG_DIR/cmake-configure.log"
cmake --build build/Release -j "$(nproc)" 2>&1 | tee "$LOG_DIR/cmake-build.log"

# Generate compile_commands.json symlink for clangd/cclsp
ln -sf build/Release/compile_commands.json "$MDB_HOME/compile_commands.json"

if "$MDB_HOME/build/Release/bin/mdb" help >/dev/null 2>&1; then
    log_ok "Binary works: $MDB_HOME/build/Release/bin/mdb"
else
    die "Binary failed smoke test (mdb help)"
fi

# ---------------------------------------------------------------------------
# Pre-stage OGB raw datasets BEFORE the reboot checkpoint.
# ---------------------------------------------------------------------------
# This runs before the NEEDS_REBOOT branch deliberately: if the driver was
# freshly installed and a reboot is imminent, we still want the 50 GB
# papers100M download to happen NOW (it is unrelated to the GPU driver
# and takes 1-2 hours). That way the user can reboot, --resume for tests,
# and have the dataset already on disk when they sit down to experiment.
#
# download-dataset.sh is idempotent (checks raw/data.npz exists) so on
# --resume this re-invocation is a cheap no-op.
if [ -n "$DOWNLOAD_DATASETS" ]; then
    log_step "Pre-staging OGB datasets: $DOWNLOAD_DATASETS"
    log_info "Raw download only — no conversion, no import, no venv."
    log_info "Use stream_convert_ogb.py + 'mdb import' when ready to experiment."
    IFS=',' read -r -a _DATASET_ARR <<< "$DOWNLOAD_DATASETS"
    for _ds in "${_DATASET_ARR[@]}"; do
        _ds_trimmed="$(echo "$_ds" | xargs)"   # strip whitespace
        [ -z "$_ds_trimmed" ] && continue
        log_info "Dataset: $_ds_trimmed"
        if bash "$MDB_HOME/scripts/download-dataset.sh" "$_ds_trimmed"; then
            log_ok "Dataset $_ds_trimmed staged"
        else
            log_warn "Dataset $_ds_trimmed failed — re-run: scripts/download-dataset.sh $_ds_trimmed"
        fi
    done
fi

# ---------------------------------------------------------------------------
# Step 11 — Validate GNN environment + run tests
# ---------------------------------------------------------------------------
# If the driver was freshly installed this run, the kernel module is NOT
# yet loaded. nvidia-smi will fail and CUDA tests will segfault. Skip
# validation + tests here, emit reboot checkpoint, and exit 0. The user
# reboots and re-runs with --resume to reach this point again with
# NEEDS_REBOOT=0, which then runs the tests.
if [ "${NEEDS_REBOOT:-0}" -eq 1 ]; then
    touch "$MDB_HOME/.onboard_needs_reboot"
    echo ""
    echo -e "${YELLOW}${BOLD}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo -e "${YELLOW}${BOLD}  Build complete, REBOOT REQUIRED before tests${NC}"
    echo -e "${YELLOW}${BOLD}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo ""
    echo "  The NVIDIA driver was freshly installed. It requires a reboot"
    echo "  to load the kernel module before tests can exercise the GPU."
    if [ -n "$DOWNLOAD_DATASETS" ]; then
        echo ""
        echo "  Good news: the dataset(s) [$DOWNLOAD_DATASETS] already finished"
        echo "  downloading before this checkpoint — they are ready on disk."
    fi
    echo ""
    echo "  Next steps:"
    echo "    sudo reboot"
    echo "    # after login:"
    echo "    cd $MDB_HOME && ./scripts/onboard.sh --resume --gpu=$GPU_CHOICE"
    echo ""
    if command -v mokutil >/dev/null 2>&1 && \
       mokutil --sb-state 2>/dev/null | head -1 | grep -qi enabled; then
        echo -e "${YELLOW}  NOTE: Secure Boot is enabled. During this boot, the MokManager"
        echo -e "        screen may appear asking to enroll the MOK key. Use the"
        echo -e "        passphrase you set during driver install.${NC}"
        echo ""
    fi
    exit 0
fi

# Consume the reboot marker if we're resuming after a driver install.
if [ -f "$MDB_HOME/.onboard_needs_reboot" ]; then
    rm -f "$MDB_HOME/.onboard_needs_reboot"
    log_ok "Reboot marker consumed — driver should now be loaded"
fi

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
echo "  Logs:     $LOG_DIR/"
echo ""
echo "  Next steps:"
echo "    source ~/.bashrc               # reload env in new shells"
echo "    ./scripts/run-tests gql        # full GQL test suite"
echo "    ./scripts/run-tests            # every suite (unit+sparql+mql+gql)"
if [ "$ENABLE_GPU" -eq 1 ]; then
    echo "    ./scripts/verify_gnn_integration.sh   # GNN pipeline smoke test"
fi
echo ""
# Dataset pre-staging hook is now executed earlier, before the NEEDS_REBOOT
# branch, so datasets download even in the path where the script exits for
# a driver-reboot checkpoint.
