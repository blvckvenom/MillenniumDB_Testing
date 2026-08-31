#!/usr/bin/env bash
# ============================================================================
# MillenniumDB — Automated Onboarding
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
#   scripts/onboard.sh --download-dataset=papers100M       # narrow OGB download
#   scripts/onboard.sh --download-dataset=arxiv,products   # multiple datasets
#   scripts/onboard.sh --download-dataset=                 # skip OGB downloads
#   scripts/onboard.sh --skip-cora                         # skip Cora LINQS tgz
#
# Dataset pre-staging (default behavior):
#   Raw downloads all 5 OGB benchmarks + Cora raw for thesis work. Total
#   ~52 GB on disk. No conversion, no import — raw source zips/tarballs
#   only, ready for stream_convert_ogb.py + 'mdb import' at experiment
#   time. Override with --download-dataset=list or --skip-cora to narrow.
#
# Idempotent: every step checks its own completion marker and skips if already done.
#
# Privilege model:
#   All root-requiring commands are accumulated into a single batch script and
#   invoked with ONE pkexec call (ONE admin password prompt for the whole run).
#   Use --dry-run to preview the batch contents without authenticating.
#
# Prerequisites (one-time, BEFORE this script can run):
#   1. Ubuntu 22.04+/24.04+ or Pop!_OS 22.04+ installed (admin did this).
#   2. git installed — NOT in Ubuntu minimal install by default:
#         sudo apt update && sudo apt install -y git
#      On Ubuntu Desktop "normal installation" git usually comes preinstalled.
#   3. This repo cloned — requires git + network:
#         git clone <repository-url>
#         cd MillenniumDB_Testing
#   4. (Recommended) Secure Boot disabled in UEFI firmware to avoid the MOK
#      enrollment dance post-reboot. The script warns if SB is enabled.
#
# Everything else (CUDA, driver, Boost, LibTorch, Python deps, build) is
# installed automatically by this script.
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
# comma-separated OGB dataset names to pre-stage raw (no conversion, no
# import). Default downloads all 5 OGB benchmarks for thesis work
# (~52 GB total on disk). Override with --download-dataset=arxiv,products
# to narrow, or --download-dataset= (empty) to skip entirely. Cora (5 MB,
# LINQS source) is handled separately below and downloads unless
# --skip-cora is passed.
DOWNLOAD_DATASETS="arxiv,products,mag,papers100M,proteins"
SKIP_CORA=0
for arg in "$@"; do
    case "$arg" in
        --gpu=*)                GPU_CHOICE="${arg#*=}" ;;
        --no-gpu)               GPU_CHOICE="none" ;;
        --skip-driver)          SKIP_DRIVER=1 ;;
        --skip-tests)           SKIP_TESTS=1 ;;
        --resume)               RESUME=1 ;;
        --dry-run)              DRY_RUN=1 ;;
        --download-dataset=*)   DOWNLOAD_DATASETS="${arg#*=}" ;;
        --skip-cora)            SKIP_CORA=1 ;;
        -h|--help)
            grep -E '^# ' "$0" | sed 's/^# \{0,1\}//' | head -50
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

# GPU generation resolution.
#
# The stack (CUDA version + LibTorch build) is chosen from the GPU
# generation, and choosing wrong is expensive: CUDA 12.4 on a Blackwell card
# compiles cleanly and only fails at RUNTIME, because the toolkit does not
# know sm_120. So the generation is resolved from three independent sources,
# most authoritative first, and if all three are inconclusive the script
# ABORTS instead of guessing.
#
#   1. nvidia-smi     exact marketing name, but needs a working driver.
#   2. lspci name     needs the local pci.ids database to know the device.
#   3. PCI device id  works on any machine, but is a documented heuristic.
#
# Source 3 exists because pciutils ships a static device database: a card
# newer than the installed pci.ids prints as "Device 2c05" with no model
# name, and every name-matching rule silently misses it.

detect_gpu_line() {
    if ! command -v lspci >/dev/null 2>&1; then
        $PRIV apt-get install -y pciutils >/dev/null 2>&1 || true
    fi
    lspci 2>/dev/null | grep -iE 'vga|3d' | grep -i nvidia || true
}

detect_gpu_pci_id() {
    lspci -nn 2>/dev/null | grep -iE 'vga|3d' \
        | grep -oiE '10de:[0-9a-f]{4}' | head -1 | cut -d: -f2 || true
}

# gpu_gen_from_name <human readable gpu name> -> 5070 | 4070 | (empty)
gpu_gen_from_name() {
    if echo "$1" | grep -qiE 'rtx.?5[0-9]{3}|[^0-9]5[0-9]{3}[^0-9]?(ti|super)?|blackwell'; then
        echo 5070
    elif echo "$1" | grep -qiE 'rtx.?4[0-9]{3}|[^0-9]4[0-9]{3}[^0-9]?(ti|super)?|ada|lovelace'; then
        echo 4070
    else
        echo ""
    fi
}

# gpu_gen_from_pci_id <4 hex digits> -> 5070 | 4070 | (empty)
# HEURISTIC, documented on purpose. NVIDIA assigns device ids in blocks per
# architecture: Ada (AD10x) lands in 0x2600-0x28FF, Blackwell (GB20x) in
# 0x2B00-0x2FFF. Verified against 10de:2c05 (RTX 5070 Ti). A card outside
# both ranges yields empty, which triggers the explicit abort below rather
# than a silent wrong answer.
gpu_gen_from_pci_id() {
    case "$1" in
        ''|*[!0-9a-fA-F]*) echo ""; return ;;
    esac
    local id=$((16#$1))
    if   [ "$id" -ge $((16#2B00)) ] && [ "$id" -le $((16#2FFF)) ]; then echo 5070
    elif [ "$id" -ge $((16#2600)) ] && [ "$id" -le $((16#28FF)) ]; then echo 4070
    else echo ""; fi
}

GPU_LINE="$(detect_gpu_line)"
GPU_PCI_ID="$(detect_gpu_pci_id)"

if [ -n "$GPU_LINE" ]; then
    log_ok "NVIDIA GPU found: $GPU_LINE"
    [ -n "$GPU_PCI_ID" ] && log_info "PCI device id: 10de:$GPU_PCI_ID"
else
    log_warn "No NVIDIA GPU detected via lspci"
fi

if [ "$GPU_CHOICE" = "auto" ]; then
    RESOLVED=""
    RESOLVED_SRC=""

    if [ -z "$GPU_LINE" ]; then
        RESOLVED="none"
        RESOLVED_SRC="lspci (no NVIDIA display device on the bus)"
    fi

    if [ -z "$RESOLVED" ] && command -v nvidia-smi >/dev/null 2>&1; then
        SMI_NAME="$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1 || true)"
        if [ -n "$SMI_NAME" ]; then
            RESOLVED="$(gpu_gen_from_name "$SMI_NAME")"
            [ -n "$RESOLVED" ] && RESOLVED_SRC="nvidia-smi: $SMI_NAME"
        fi
    fi

    if [ -z "$RESOLVED" ]; then
        RESOLVED="$(gpu_gen_from_name "$GPU_LINE")"
        [ -n "$RESOLVED" ] && RESOLVED_SRC="lspci device name"
    fi

    if [ -z "$RESOLVED" ] && [ -n "$GPU_PCI_ID" ]; then
        RESOLVED="$(gpu_gen_from_pci_id "$GPU_PCI_ID")"
        if [ -n "$RESOLVED" ]; then
            RESOLVED_SRC="PCI device id heuristic (10de:$GPU_PCI_ID)"
            log_warn "Generation inferred from the PCI device id, not from a model name."
            log_warn "  Your pci.ids database does not know this card. Refresh it with:"
            log_warn "    sudo update-pciids"
            log_warn "  Override with --gpu=5070 or --gpu=4070 if this is wrong."
        fi
    fi

    if [ -z "$RESOLVED" ]; then
        log_err "Could not determine the GPU generation."
        log_err "  lspci line : ${GPU_LINE:-<none>}"
        log_err "  PCI id     : ${GPU_PCI_ID:-<none>}"
        log_err ""
        log_err "  Refusing to guess: picking the wrong CUDA version compiles fine"
        log_err "  and then fails at runtime, which is far harder to diagnose."
        log_err ""
        log_err "  Fix it with either:"
        log_err "    sudo update-pciids            # refresh the PCI device database"
        log_err "    scripts/onboard.sh --gpu=5070 # Blackwell: RTX 50xx, CUDA 12.8"
        log_err "    scripts/onboard.sh --gpu=4070 # Ada:       RTX 40xx, CUDA 12.4"
        log_err "    scripts/onboard.sh --no-gpu   # build CPU-only on purpose"
        die "GPU generation unresolved"
    fi

    GPU_CHOICE="$RESOLVED"
    log_ok "GPU generation resolved from $RESOLVED_SRC -> stack $GPU_CHOICE"
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
        liblz4-dev          # Spec #1 spill-file compression (3-5× disk savings)
        libgtest-dev        # Required by custom test targets
                            # (serial_scan_test, radix_partition_sort_test).
                            # Without this CMakeLists.txt silently skips them.
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

    # Two very different situations both make nvidia-smi fail, and they need
    # opposite fixes:
    #
    #   a) No driver at all           -> install DRIVER_PKG.
    #   b) Driver userspace installed, but its KERNEL MODULE was never built
    #      for the kernel currently running -> install only the per-kernel
    #      module (or rebuild via DKMS). This is what a kernel-series upgrade
    #      leaves behind. Installing DRIVER_PKG here is wrong: the userspace
    #      half is already present and is often NEWER than DRIVER_PKG, so a
    #      blind install can queue a downgrade and still not load a module.
    #
    # Case (b) is worth detecting explicitly because its symptom is silence:
    # CUDA reports zero devices, and the whole GNN pipeline degrades to CPU
    # without a single error message.
    MODULE_MISSING=0
    if [ "$DRIVER_OK" -eq 0 ]; then
        RUNNING_KERNEL="$(uname -r)"
        INSTALLED_DRIVER_PKG="$(dpkg-query -W -f='${Package} ${Status}\n' 'nvidia-driver-*' 2>/dev/null \
            | awk '$NF=="installed"{print $1; exit}' || true)"
        MODULE_FOR_KERNEL="$(find "/lib/modules/$RUNNING_KERNEL" -name 'nvidia.ko*' 2>/dev/null | head -1 || true)"

        if [ -n "$INSTALLED_DRIVER_PKG" ] && [ -z "$MODULE_FOR_KERNEL" ]; then
            MODULE_MISSING=1
            log_warn "Driver userspace is installed ($INSTALLED_DRIVER_PKG) but no kernel"
            log_warn "  module exists for the running kernel $RUNNING_KERNEL."
            OTHER_KERNEL_MOD="$(find /lib/modules -name 'nvidia.ko*' 2>/dev/null | head -1 || true)"
            if [ -n "$OTHER_KERNEL_MOD" ]; then
                log_info "  A module does exist for another kernel: $OTHER_KERNEL_MOD"
                log_info "  Booting that kernel from GRUB restores the GPU without installing anything."
            fi

            if [ -d /var/lib/dkms/nvidia ]; then
                log_info "  Packaging model: DKMS. Queueing a module rebuild."
                queue_root "rebuild NVIDIA kernel module via DKMS" "dkms autoinstall"
            else
                # Prebuilt per-kernel modules (Ubuntu / Pop!_OS default).
                # Derive the package prefix from whatever is already installed
                # instead of hardcoding a branch or an -open/-server variant.
                EXISTING_MOD_PKG="$(dpkg-query -W -f='${Package} ${Status}\n' 'linux-modules-nvidia-*' 2>/dev/null \
                    | awk '$NF=="installed"{print $1}' \
                    | grep -E -- '-[0-9]+\.[0-9]+\.[0-9]+-[0-9]+-[a-z]+$' | head -1 || true)"
                if [ -n "$EXISTING_MOD_PKG" ]; then
                    MOD_PREFIX="$(echo "$EXISTING_MOD_PKG" | sed -E 's/-[0-9]+\.[0-9]+\.[0-9]+-[0-9]+-[a-z]+$//')"
                    WANTED_MOD_PKG="${MOD_PREFIX}-${RUNNING_KERNEL}"
                    if apt-cache show "$WANTED_MOD_PKG" >/dev/null 2>&1; then
                        log_info "  Packaging model: prebuilt per-kernel modules."
                        log_info "  Queueing $WANTED_MOD_PKG"
                        queue_root "install NVIDIA kernel module for $RUNNING_KERNEL" \
                            "apt-get install -y $WANTED_MOD_PKG"
                    else
                        log_err "  No package $WANTED_MOD_PKG in the configured repositories."
                        log_err "  Either boot a kernel that has its module, or install a kernel"
                        log_err "  series that NVIDIA modules are published for."
                    fi
                else
                    log_warn "  Could not infer the module package name; install it manually:"
                    log_warn "    apt-cache search \"linux-modules-nvidia.*$RUNNING_KERNEL\""
                fi
            fi
        fi
    fi

    if [ "$DRIVER_OK" -eq 0 ] && [ "$MODULE_MISSING" -eq 0 ]; then
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
# numpy venv for dataset conversion scripts (stream_convert_ogb.py etc.).
# Without this, running stream_convert_ogb.py with system python3 fails
# with "ModuleNotFoundError: No module named 'numpy'".
setup_venv "$MDB_HOME/scripts/gnn_datasets"

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
# Pre-stage Cora raw (LINQS source, 5 MB). Separate from OGB because Cora's
# upstream format (tgz with .content + .cites text files) differs from
# OGB's (zip with data.npz). Skipped with --skip-cora.
#
# Idempotent: downloads to $MDB_HOME/cora_data/raw/cora.tgz and skips if
# already present, then converts it to the .gql + .npy form the pipeline reads.
#
# The conversion is not optional. data/example/gql/cora/ is gitignored, so a
# fresh clone has no fixture, and every command this script recommends at the
# end -- cora_fastloop.sh, run-tests gnn, gnn_benchmark.sh cora -- aborts
# without it.
# ---------------------------------------------------------------------------
if [ "$SKIP_CORA" -eq 0 ]; then
    log_step "Pre-staging Cora raw (LINQS, 5 MB)"
    CORA_URL="https://linqs-data.soe.ucsc.edu/public/lbc/cora.tgz"
    CORA_DIR="$MDB_HOME/cora_data/raw"
    mkdir -p "$CORA_DIR"
    if [ -f "$CORA_DIR/cora.tgz" ] && [ -s "$CORA_DIR/cora.tgz" ]; then
        log_ok "Cora raw already present at $CORA_DIR/cora.tgz"
    else
        if wget -q --show-progress -O "$CORA_DIR/cora.tgz" "$CORA_URL"; then
            log_ok "Cora raw downloaded to $CORA_DIR/cora.tgz"
        else
            log_warn "Cora download failed — re-run: wget -O $CORA_DIR/cora.tgz $CORA_URL"
        fi
    fi

    # Convert to the .gql + feature/label .npy triple the GNN procedures read.
    if [ -f "$MDB_HOME/data/example/gql/cora/cora.gql" ]; then
        log_ok "Cora already converted at data/example/gql/cora/"
    else
        CORA_PY="$MDB_HOME/scripts/gnn_datasets/.venv/bin/python"
        [ -x "$CORA_PY" ] || CORA_PY="$(command -v python3)"
        if "$CORA_PY" "$MDB_HOME/scripts/download_gnn_datasets.py" \
               --dataset cora --output "$MDB_HOME/data/example/gql"; then
            log_ok "Cora converted to data/example/gql/cora/"
        else
            log_warn "Cora conversion failed — re-run: python3 scripts/download_gnn_datasets.py --dataset cora"
        fi
    fi
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

# Runtime GPU gate.
#
# validate-gnn-environment warns about a dead GPU but does not stop, and a
# warning scrolls past between two screens of output. That is how a host ends
# up printing "Onboarding complete" while every training run silently uses
# the CPU. When a GPU stack was deliberately selected, verify it actually
# works and make the exit code say so.
ONBOARD_GPU_OK=1
if [ "$ENABLE_GPU" -eq 1 ]; then
    if [ "$NEEDS_REBOOT" -eq 1 ]; then
        log_info "GPU verification deferred: a driver install is pending a reboot."
        log_info "  After rebooting, re-run: scripts/onboard.sh --resume"
    elif command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi -L >/dev/null 2>&1; then
        log_ok "GPU runtime verified: $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
    else
        ONBOARD_GPU_OK=0
        log_err "GPU stack was selected but the GPU is NOT usable at runtime."
        log_err "  nvidia-smi cannot talk to the driver, so CUDA sees zero devices."
        log_err ""
        log_err "  Everything still BUILDS and RUNS, silently, on the CPU. Symptoms"
        log_err "  you would otherwise chase for hours:"
        log_err "    gnn_build_feature_store -> gpuAvailable=false, l1Nodes=0"
        log_err "    gnn_train               -> l1HitRatio=0.0, useAddrTablesEffective=false"
        log_err ""
        log_err "  Diagnose which half is missing:"
        log_err "    uname -r                              # kernel in use"
        log_err "    find /lib/modules -name 'nvidia.ko*'  # kernels that HAVE the module"
        log_err "    ls /var/lib/dkms/                     # 'nvidia' here => DKMS model"
        log_err ""
        log_err "  Re-running scripts/onboard.sh now queues the right fix for both cases."
        log_err "  To build CPU-only on purpose instead: scripts/onboard.sh --no-gpu"
    fi
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
if [ "${ONBOARD_GPU_OK:-1}" -eq 1 ]; then
    echo -e "${GREEN}${BOLD}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo -e "${GREEN}${BOLD}  Onboarding complete ✓${NC}"
    echo -e "${GREEN}${BOLD}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
else
    echo -e "${YELLOW}${BOLD}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo -e "${YELLOW}${BOLD}  Onboarding finished, but the GPU is NOT usable ⚠${NC}"
    echo -e "${YELLOW}${BOLD}  The build is fine. Training would run on the CPU.${NC}"
    echo -e "${YELLOW}${BOLD}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
fi
echo ""
echo "  Binary:   $MDB_HOME/build/Release/bin/mdb help"
echo "  Stack:    LibTorch=$LIBTORCH_EXPECTED, GPU=$GPU_CHOICE"
echo "  Logs:     $LOG_DIR/"
echo ""
echo "  Next steps:"
echo "    source ~/.bashrc               # reload env in new shells"
echo "    ./scripts/cora_fastloop.sh     # 5 s end-to-end GNN gate (start here)"
echo "    ./scripts/run-tests gql        # full GQL test suite"
echo "    ./scripts/run-tests            # every suite (unit+sparql+mql+gql)"
if [ "$ENABLE_GPU" -eq 1 ]; then
    echo "    ./scripts/gnn_benchmark.sh cora        # download, train, verify the GPU is used"
fi
echo ""
# Dataset pre-staging hook is now executed earlier, before the NEEDS_REBOOT
# branch, so datasets download even in the path where the script exits for
# a driver-reboot checkpoint.

# The exit code carries the GPU verdict, so CI and wrapper scripts can react
# to a CPU-only outcome instead of parsing the banner text.
#   0 = everything verified
#   2 = built fine, but the GPU is not usable at runtime
[ "${ONBOARD_GPU_OK:-1}" -eq 1 ] || exit 2
exit 0
