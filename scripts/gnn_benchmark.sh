#!/usr/bin/env bash
# ============================================================================
# gnn_benchmark.sh — one command from a clean checkout to a trained GNN.
# ============================================================================
# Downloads a benchmark graph, converts it to the GQL import format, imports it
# with its feature matrix, starts a server, and runs the four procedures that
# make up the training pipeline. Prints a stage-by-stage timing table and a
# pass/fail verdict.
#
# Usage:
#   scripts/gnn_benchmark.sh cora            # ~1 min   (2.7 K nodes)
#   scripts/gnn_benchmark.sh arxiv           # ~5 min   (169 K nodes)
#   scripts/gnn_benchmark.sh products        # ~40 min  (2.4 M nodes)
#   scripts/gnn_benchmark.sh papers100M      # hours    (111 M nodes, ~200 GB disk;
#                                            #           pass --db to a real filesystem)
#
# Options:
#   --cpu             run without a GPU (otherwise a missing GPU is an error)
#   --skip-download   the .gql and .npy files are already in place
#   --refresh         re-convert even if the .gql exists (repairs stale files)
#   --keep            do not delete the database directory on exit
#   --db <path>       where to build the database (default: a fresh temp dir)
#   --port <n>        server port (default 7879)
#   --epochs <n>      override the training epochs
#   --gpu-cache-mb <n>  size of the GPU feature cache (default: per dataset)
#
# Why a GPU is required by default: MillenniumDB falls back to CPU silently when
# CUDA is unavailable. Nothing errors, the run simply takes several times longer.
# This script turns that quiet fallback into a loud one, and --cpu is how you say
# you meant it. The check that decides is gnn_build_feature_store's gpuAvailable,
# not nvidia-smi: only the first says whether THIS build can use the device.
# ============================================================================
set -u -o pipefail

REPO="${MDB_HOME:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
MDB="$REPO/build/Release/bin/mdb"
PORT=7879
ALLOW_CPU=0
SKIP_DOWNLOAD=0
REFRESH=0
KEEP=0
DB=""
EPOCHS=""
GPU_CACHE=""
DATASET=""

while [ $# -gt 0 ]; do
    case "$1" in
        cora|arxiv|products|papers100M)
            [ -z "$DATASET" ] || { echo "pick one dataset; got '$DATASET' and '$1'" >&2; exit 1; }
            DATASET="$1" ;;
        --cpu)            ALLOW_CPU=1 ;;
        --skip-download)  SKIP_DOWNLOAD=1 ;;
        --refresh)        REFRESH=1 ;;
        --keep)           KEEP=1 ;;
        --db)             [ $# -ge 2 ] || { echo "--db needs a path" >&2; exit 1; }
                          DB="$2"; shift ;;
        --port)           [ $# -ge 2 ] || { echo "--port needs a number" >&2; exit 1; }
                          case "$2" in ''|*[!0-9]*) echo "--port takes an integer, got '$2'" >&2; exit 1 ;; esac
                          PORT="$2"; shift ;;
        --epochs)         [ $# -ge 2 ] || { echo "--epochs needs a number" >&2; exit 1; }
                          case "$2" in ''|*[!0-9]*|0) echo "--epochs takes a positive integer, got '$2'" >&2; exit 1 ;; esac
                          EPOCHS="$2"; shift ;;
        --gpu-cache-mb)   [ $# -ge 2 ] || { echo "--gpu-cache-mb needs a number" >&2; exit 1; }
                          case "$2" in ''|*[!0-9]*) echo "--gpu-cache-mb takes an integer, got '$2'" >&2; exit 1 ;; esac
                          GPU_CACHE="$2"; shift ;;
        # Located by the header's own banner lines rather than by a fixed
        # range, which silently desyncs the moment the header grows -- that is
        # how the explanation below lost its last two lines. The pattern avoids
        # interval quantifiers, which mawk rejects without --re-interval.
        -h|--help)        awk 'NR==1{next} /^# =+$/{n++} n>=3{exit} {print}' "$0"; exit 0 ;;
        *) echo "unknown argument: $1  (try --help)" >&2; exit 1 ;;
    esac
    shift
done

# ---------------------------------------------------------------------------
# Per-dataset configuration.
#
# The values are not interchangeable. Node and edge labels come from what each
# converter writes into the .gql file; the projection orientation differs
# because papers100M is symmetrised during sampling rather than during
# projection; and the cache budgets are sized for each graph's feature matrix.
# ---------------------------------------------------------------------------
case "$DATASET" in
cora)
    DS_DIR=cora;             PREFIX=cora
    NODE_LABEL=Paper;        EDGE_LABEL=CITES
    PROJ_ORIENT=UNDIRECTED;  FANOUT="[5,10]";      BATCH=64
    BUILD_OPTS="gpu_budget_mb:0,cpu_budget_mb:100,force:1"
    PROJ_OPTS=""
    TRAIN_OPTS=""
    NEED_GB=1;               NEED_RAM_GB=2
    DEF_EPOCHS=30;           ACC_FLOOR=0.65;  DETERMINISM=1
    ;;
arxiv)
    DS_DIR=ogbn-arxiv;       PREFIX=ogbn_arxiv
    NODE_LABEL=Node;         EDGE_LABEL=CONNECTS
    PROJ_ORIENT=UNDIRECTED;  FANOUT="[10,15]";     BATCH=1024
    BUILD_OPTS="gpu_budget_mb:2048,cpu_budget_mb:512"
    PROJ_OPTS=""
    TRAIN_OPTS="sampleCacheMb:1024,"
    NEED_GB=8;               NEED_RAM_GB=6
    DEF_EPOCHS=30;           ACC_FLOOR=0.60;  DETERMINISM=0
    ;;
products)
    DS_DIR=ogbn-products;    PREFIX=ogbn_products
    NODE_LABEL=Node;         EDGE_LABEL=CONNECTS
    PROJ_ORIENT=UNDIRECTED;  FANOUT="[10,15]";     BATCH=1024
    BUILD_OPTS="gpu_budget_mb:2048,cpu_budget_mb:512"
    PROJ_OPTS=""
    # The training phase, not the store, is what exhausts host memory here: an
    # unbounded sample cache plus prefetch queues took 29.3 GB and the kernel
    # killed the server. These bound it.
    TRAIN_OPTS="sampleCacheMb:2048,prefetchNumWorkers:2,prefetchQueueSize:2,"
    NEED_GB=60;              NEED_RAM_GB=24
    DEF_EPOCHS=30;           ACC_FLOOR=0.70;  DETERMINISM=0
    ;;
papers100M)
    DS_DIR=ogbn-papers100M;  PREFIX=papers100M
    NODE_LABEL=Node;         EDGE_LABEL=CITES
    PROJ_ORIENT=NATURAL;     FANOUT="[20,15,10]";  BATCH=1024
    BUILD_OPTS="bakeBlocks:true"
    # Matching REPRODUCE.md. Without buildTopologySnapshot:false the GNN-intent
    # default turns the sidecars on and writes two ~27 GB files nobody asked for;
    # GNN_MINIMAL drops the five topology indexes sampling never reads.
    PROJ_OPTS="indexSet:'GNN_MINIMAL',buildTopologySnapshot:false,"
    TRAIN_OPTS="sampleCacheMb:2048,prefetchNumWorkers:8,prefetchQueueSize:6,"
    NEED_GB=200;             NEED_RAM_GB=28
    DEF_EPOCHS=50;           ACC_FLOOR=0.60;  DETERMINISM=0
    ;;
"")
    echo "Pick a dataset: cora | arxiv | products | papers100M   (try --help)" >&2
    exit 1 ;;
esac
[ -n "$EPOCHS" ] || EPOCHS=$DEF_EPOCHS

# The cache budgets are fixed per dataset rather than sized to the card, so that
# two machines produce comparable numbers: a 24 GB card given a 24 GB budget
# would post hit ratios an 8 GB card cannot, and the gap would read as speed
# when it is only budget. --gpu-cache-mb is the escape for a card that cannot
# hold the default.
if [ -n "$GPU_CACHE" ]; then
    BUILD_OPTS=$(echo "$BUILD_OPTS" | sed -E "s/gpu_budget_mb:[0-9]+/gpu_budget_mb:$GPU_CACHE/")
    case "$BUILD_OPTS" in *gpu_budget_mb:*) ;; *) BUILD_OPTS="gpu_budget_mb:$GPU_CACHE,$BUILD_OPTS" ;; esac
fi

GQL_DIR="$REPO/data/example/gql/$DS_DIR"
GQL="$GQL_DIR/$PREFIX.gql"
NPY="$GQL_DIR/${PREFIX}_features.npy"
# An older converter nested the output one level deeper. Adopt whichever layout
# is on disk rather than re-downloading into a second copy.
if [ ! -f "$GQL" ] && [ -f "$GQL_DIR/$DS_DIR/$PREFIX.gql" ]; then
    GQL="$GQL_DIR/$DS_DIR/$PREFIX.gql"
    NPY="$GQL_DIR/$DS_DIR/${PREFIX}_features.npy"
fi
PROJ="${DATASET}_bench"
SAMPLE="${DATASET}_s"

# Only a directory this script created may be removed on exit. A --db the user
# supplied is theirs, and the exit trap fires on every abort including the
# preflight, long before anything was written into it.
CREATED_DB=0
if [ -z "$DB" ]; then
    DB=$(mktemp -d "/tmp/gnn_bench_${DATASET}.XXXXXX"); CREATED_DB=1
elif [ -d "$DB" ] && [ -n "$(ls -A "$DB" 2>/dev/null)" ]; then
    echo "--db $DB already exists and is not empty; refusing to use it" >&2; exit 1
fi
RESP=$(mktemp "/tmp/gnn_bench_resp.XXXXXX")
SRVPID=""
RC=1
declare -a TIMINGS=()

cleanup() {
    [ -n "$SRVPID" ] && kill "$SRVPID" 2>/dev/null
    [ -n "$SRVPID" ] && wait "$SRVPID" 2>/dev/null
    rm -f "$RESP" 2>/dev/null
    if [ "$KEEP" = "1" ]; then echo "[keep] database left at $DB"
    elif [ "$CREATED_DB" = 1 ]; then
        # The import log survives a failure. The failure message tells the reader
        # to look at it, and deleting it on the way out made that advice useless.
        # The server log lives inside $DB. Rescue it before the directory goes,
        # or a training failure leaves nothing to read but the kernel log.
        if [ "$RC" != 0 ] && [ -s "$DB/server.log" ]; then
            cp "$DB/server.log" "$DB.server.log" 2>/dev/null
            echo "[bench] server log kept at $DB.server.log"
        fi
        rm -rf "$DB" 2>/dev/null
        if [ "$RC" = 0 ]; then rm -f "$DB.import.log" "$DB.server.log" 2>/dev/null
        elif [ -s "$DB.import.log" ]; then echo "[bench] import log kept at $DB.import.log"; fi
    fi
    exit $RC
}
trap cleanup EXIT INT TERM

say()  { printf '\033[1;36m[bench]\033[0m %s\n' "$*"; }
ok()   { printf '  \033[0;32m+\033[0m %s\n' "$*"; }
warn() { printf '  \033[1;33m!\033[0m %s\n' "$*"; }
fail() { printf '\033[0;31m[bench] FAILED:\033[0m %s\n' "$*" >&2; RC=1; exit 1; }
# Whole seconds turned most cora stages into a row of zeros and made the
# total wobble between 2 and 4 for identical work.
now()  { date +%s.%N; }
elapsed() { python3 -c "print(f'{$(now) - $1:.1f}')"; }

# gql <label> <query> — POST a query, leave the body in $RESP, return non-zero
# when the server reported a problem. Two traps are handled here:
#
#   curl exits 0 even when the server answers with an error, so the body itself
#   has to be inspected; and a function that calls exit only ends the script
#   when it is not being captured. Writing to a file instead of stdout keeps
#   this out of a command substitution, so the caller's `|| fail` really aborts.
gql() {
    local label=$1 query=$2 code
    code=$(curl -s --max-time 86400 -o "$RESP" -w '%{http_code}' \
                -X POST "http://localhost:$PORT/gql" \
                -H "Content-Type: text/plain" --data-binary "$query" 2>/dev/null)
    # The status is the fact; the keyword scan is only a second opinion. The
    # server answers 4xx/5xx with a bare message and no status line in the body,
    # and most of those messages contain none of the words below -- "epochs must
    # be positive" is one -- so a body-only check reports a failed call as green.
    case "$code" in
        2*) ;;
        *)  printf '\n  HTTP %s\n' "$code" >&2; cat "$RESP" >&2; printf '\n'; return 1 ;;
    esac
    if grep -qiE "(error|exception|failed|not found|bad query|unexpected)" "$RESP"; then
        printf '\n'; cat "$RESP" >&2; printf '\n'
        return 1
    fi
    return 0
}

# col <colname> — read one column out of the two-line CSV left in $RESP.
col() {
    python3 -c "
import sys
rows=[r for r in open('$RESP').read().splitlines() if r.strip()]
if len(rows)<2: print('NA'); raise SystemExit
h=rows[0].split(','); d=rows[-1].split(',')
print(d[h.index('$1')] if '$1' in h else 'NA')
"
}

stage() { TIMINGS+=("$1|$(printf '%.1f' "$2")"); }

# The server answers with whatever printf %g produced, so a plain accuracy can
# arrive as 7.52E-1. Render it back to something a person reads at a glance.
num() { python3 -c "
try:    print(f'{float(\"$1\"):.4f}')
except: print('$1')
"; }

# ---------------------------------------------------------------------------
say "1/6  preflight"
# ---------------------------------------------------------------------------
[ -x "$MDB" ] || fail "no binary at $MDB — build first:
    cmake -B build/Release -D CMAKE_BUILD_TYPE=Release -D ENABLE_GNN=ON
    cmake --build build/Release -j \$(nproc)
  or run scripts/onboard.sh, which does the whole environment."
ok "binary: $MDB"

# The converter needs numpy and requests. Prefer the project venv the dataset
# scripts already create, so a fresh machine does not have to install anything
# system-wide, and fall back to whatever python3 is on PATH.
PY=""
for candidate in "$REPO/scripts/gnn_datasets/.venv/bin/python" "$(command -v python3)"; do
    [ -x "$candidate" ] || continue
    if "$candidate" -c "import numpy, requests" 2>/dev/null; then PY="$candidate"; break; fi
done
# arxiv and products load their source through the ogb package, which pulls in
# torch. Prove it here rather than after the download stage has already started.
case "$DATASET" in
  arxiv|products)
    if [ -n "$PY" ] && ! "$PY" -c "import ogb" 2>/dev/null; then
        fail "$DATASET is converted through the ogb package, which is not installed
  in $PY.

  Install it in the project virtual environment, not system-wide -- a
  distribution python refuses pip installs as externally managed:

      python3 -m venv $REPO/scripts/gnn_datasets/.venv
      $REPO/scripts/gnn_datasets/.venv/bin/pip install numpy requests tqdm ogb torch

  It is about 2.5 GB. cora and papers100M do not need it."
    fi ;;
esac

[ -n "$PY" ] || fail "no python with numpy and requests.

  Create the project venv once:

      python3 -m venv scripts/gnn_datasets/.venv
      scripts/gnn_datasets/.venv/bin/pip install numpy requests tqdm
      scripts/gnn_datasets/.venv/bin/pip install ogb torch   # OGB datasets only"
ok "python: $PY"

GPU_NAME=""
if command -v nvidia-smi >/dev/null 2>&1; then
    GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)
fi
if [ -n "$GPU_NAME" ]; then
    ok "GPU: $GPU_NAME (compute $(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | head -1))"
elif [ "$ALLOW_CPU" = "1" ]; then
    warn "no GPU — continuing because --cpu was given; expect a much slower run"
else
    fail "no usable GPU detected.

  nvidia-smi is missing or reports no device. On this project that usually means
  the NVIDIA kernel module was not rebuilt after a kernel upgrade: the driver is
  installed, the module for the running kernel is not. Check with

      nvidia-smi ; ls /lib/modules/\$(uname -r)/updates/dkms/ 2>/dev/null

  Re-run scripts/onboard.sh to repair it, or pass --cpu to proceed anyway."
fi

# ---------------------------------------------------------------------------
say "2/6  dataset"
# ---------------------------------------------------------------------------
t0=$(now)
if [ "$SKIP_DOWNLOAD" = "1" ]; then
    ok "download skipped by request"
elif [ -f "$GQL" ] && [ -f "$NPY" ] && [ "$REFRESH" = "0" ]; then
    ok "already present at $GQL_DIR"
else
    case "$DATASET" in
      papers100M) warn "this downloads ~50 GB and expands past 120 GB" ;;
      products)   warn "this downloads ~1.5 GB" ;;
    esac
    DS_ARG=$DS_DIR
    [ "$DATASET" = cora ] && DS_ARG=cora
    FORCE_ARG=""
    [ "$REFRESH" = "1" ] && FORCE_ARG="--force"
    # --output must be explicit: the converter defaults to a path relative to
    # the current directory, so without this the files land wherever the caller
    # happened to be standing rather than next to the repository.
    "$PY" "$REPO/scripts/download_gnn_datasets.py" --dataset "$DS_ARG" $FORCE_ARG \
        --output "$REPO/data/example/gql" \
        || fail "download/convert failed"
fi
[ -f "$GQL" ] || fail "expected $GQL after the download stage"
[ -f "$NPY" ] || fail "expected $NPY after the download stage"
# The pipeline reads the published train/val/test division out of the node
# lines. Files produced before the converter started emitting it parse fine and
# import fine, and only fail four stages later inside graph_project, so the
# check belongs here rather than there.
if ! grep -qm1 'split:' "$GQL"; then
    fail "$GQL carries no split: property.

  It predates the converter that writes the published train/val/test division.
  Regenerate it:

      scripts/gnn_benchmark.sh $DATASET --refresh

  or, if this file is a fixture some other script depends on, point --db and the
  data directory somewhere else instead of overwriting it."
fi
ok "graph: $(du -h "$GQL" | cut -f1)   features: $(du -h "$NPY" | cut -f1)   splits present"
stage dataset "$(elapsed "$t0")"

# The database is what actually fills the disk: base plus features plus one
# projection. Refuse now rather than hours in, and refuse /tmp when /tmp is a
# ramdisk, which it is on many systems.
DB_PARENT=$(dirname "$DB")
FS_TYPE=$(stat -f -c %T "$DB_PARENT" 2>/dev/null || echo unknown)
if [ "$FS_TYPE" = tmpfs ] && [ "$NEED_GB" -gt 4 ]; then
    fail "$DB_PARENT is a tmpfs (a ramdisk); $DATASET needs about ${NEED_GB} GB on real
  storage. Pass --db /path/on/disk."
fi
AVAIL_GB=$(df -P -BG "$DB_PARENT" 2>/dev/null | awk 'NR==2 {gsub("G","",$4); print $4}')
if [ -n "${AVAIL_GB:-}" ] && [ "$AVAIL_GB" -lt "$NEED_GB" ]; then
    fail "$DATASET needs about ${NEED_GB} GB for the database and $DB_PARENT has ${AVAIL_GB} GB.
  Pass --db to point at a filesystem with room."
fi
ok "disk: ${AVAIL_GB:-?} GB free at $DB_PARENT, need about ${NEED_GB} GB"

# Disk was checked; memory was not, and memory is what the training phase runs
# out of. The kernel kills the server outright, so the failure arrives with no
# message from MillenniumDB at all.
RAM_GB=$(awk '/MemTotal/{printf "%d", $2/1048576}' /proc/meminfo 2>/dev/null)
if [ -n "${RAM_GB:-}" ] && [ "$RAM_GB" -lt "$NEED_RAM_GB" ]; then
    fail "$DATASET training needs about ${NEED_RAM_GB} GB of host memory and this
  machine has ${RAM_GB} GB. The kernel will kill the server mid-run rather than
  report an error. Lower the epochs, or run a smaller dataset."
fi
ok "memory: ${RAM_GB:-?} GB total, need about ${NEED_RAM_GB} GB"

# ---------------------------------------------------------------------------
say "3/6  import"
# ---------------------------------------------------------------------------
# --with-tensors is what registers the feature matrix and writes the .rmap that
# maps rows back to node ids. Without it the projection has no features to carry
# and there is no way to add them afterwards short of reimporting.
t0=$(now)
"$MDB" import "$GQL" "$DB" --with-tensors "$NPY" >"$DB.import.log" 2>&1 \
    || fail "import failed — see $DB.import.log"
ok "database: $DB"
stage import "$(elapsed "$t0")"

# ---------------------------------------------------------------------------
say "4/6  server"
# ---------------------------------------------------------------------------
# Refuse a port somebody else owns rather than killing whatever holds it. The
# old approach matched an unanchored pattern, so --port 3987 would kill a server
# on 39871 and two runs of this script would kill each other.
if command -v ss >/dev/null 2>&1 && ss -ltn "sport = :$PORT" 2>/dev/null | grep -q LISTEN; then
    fail "port $PORT is already in use. Pass --port <n> to pick another."
fi

"$MDB" server "$DB" -p "$PORT" -t 86400 --browser false >"$DB/server.log" 2>&1 &
SRVPID=$!
ready=0
for _ in $(seq 1 60); do
    # Liveness first. When the probe below is the loop condition, a reply from a
    # server this script did not start satisfies it, and the rest of the run
    # measures a stranger's database.
    kill -0 "$SRVPID" 2>/dev/null || fail "server exited on startup — see $DB/server.log"
    if curl -s --max-time 2 -X POST "http://localhost:$PORT/gql" \
            -H "Content-Type: text/plain" --data-binary "RETURN 1" 2>/dev/null | grep -q 1; then
        ready=1; break
    fi
    sleep 0.5
done
[ "$ready" = 1 ] || fail "server not answering after 30 s"
ok "listening on :$PORT"

# ---------------------------------------------------------------------------
say "5/6  pipeline"
# ---------------------------------------------------------------------------
t0=$(now)
# PROJ_OPTS carries its own trailing comma and goes first, so that an empty value
# leaves the map well formed instead of a dangling separator.
gql graph_project "CALL graph_project('$PROJ','$NODE_LABEL','$EDGE_LABEL',{
        ${PROJ_OPTS}orientation:'$PROJ_ORIENT',
        includeFeatures:'node_features',
        labelProperty:'label',
        splitProperty:'split'
    }) YIELD graphName,nodeCount,relationshipCount,featureDim,numClasses RETURN *" \
    || fail "graph_project did not create the projection"
ok "project: $(col nodeCount) nodes, $(col relationshipCount) edges, dim $(col featureDim), $(col numClasses) classes"
stage graph_project "$(elapsed "$t0")"

t0=$(now)
gql gnn_offline_sample "CALL gnn_offline_sample('$PROJ','$SAMPLE',$FANOUT,{
        batchSize:$BATCH,
        randomSeed:42,
        orientation:'UNDIRECTED'
    }) YIELD totalBatches,trainBatches,validationBatches,testBatches,uniqueNodes RETURN *" \
    || fail "gnn_offline_sample did not produce a sample"
ok "sample: $(col totalBatches) batches (train $(col trainBatches)), $(col uniqueNodes) unique nodes"
stage gnn_offline_sample "$(elapsed "$t0")"

t0=$(now)
gql gnn_build_feature_store "CALL gnn_build_feature_store('$SAMPLE','node_features',{
        $BUILD_OPTS
    }) YIELD l1Nodes,l2Nodes,l3Nodes,l4Nodes,gpuAvailable RETURN *" \
    || fail "gnn_build_feature_store did not build the store"
GPU_SEEN=$(col gpuAvailable)
ok "store: L1=$(col l1Nodes) L2=$(col l2Nodes) L3=$(col l3Nodes) L4=$(col l4Nodes)"
stage gnn_build_feature_store "$(elapsed "$t0")"

# The runtime check that matters. nvidia-smi answering only proves a device
# exists; this is MillenniumDB reporting whether its own build could use it.
if [ "$GPU_SEEN" = "true" ]; then
    ok "CUDA visible to MillenniumDB"
elif [ "$ALLOW_CPU" = "1" ]; then
    warn "running on CPU (gpuAvailable=$GPU_SEEN)"
else
    fail "the host has a GPU but MillenniumDB reports gpuAvailable=$GPU_SEEN.

  The binary was built without CUDA, or LibTorch cannot see the device. Confirm
  the build carried -D ENABLE_GNN=ON against a CUDA LibTorch, then re-run.
  Pass --cpu to accept the slower path."
fi

# patience:999 and tolerance:0 together disable both early exits. They are
# separate checks — patience watches validation accuracy, tolerance watches the
# change in loss between consecutive epochs — and on a small training split the
# loss flattens long before the model is done, so raising only patience still
# stops the run short of the epochs that were asked for.
t0=$(now)
gql gnn_train "CALL gnn_train('$SAMPLE','node_features',{
        ${TRAIN_OPTS}model:'graphsage',
        hiddenDim:256,
        epochs:$EPOCHS,
        randomSeed:42,
        patience:999,
        tolerance:0
    }) YIELD testAccuracy,bestValAccuracy,ranEpochs,trainSeconds,l1HitRatio RETURN *" \
    || fail "gnn_train did not finish"
ACC=$(col testAccuracy)
VAL=$(col bestValAccuracy)
L1H=$(col l1HitRatio)
ok "train: test $(num "$ACC"), best val $(num "$VAL"), $(col ranEpochs) epochs in $(num "$(col trainSeconds)") s, L1 hit $(num "$L1H")"
stage gnn_train "$(elapsed "$t0")"

# ---------------------------------------------------------------------------
say "6/6  verdict"
# ---------------------------------------------------------------------------
printf '\n  %-26s %10s\n' "stage" "seconds"
printf '  %-26s %10s\n' "--------------------------" "-------"
TOTAL=0
for row in "${TIMINGS[@]}"; do
    printf '  %-26s %10s\n' "${row%%|*}" "${row##*|}"
    TOTAL=$(python3 -c "print(f'{$TOTAL + ${row##*|}:.1f}')")
done
printf '  %-26s %10s\n\n' "total" "$TOTAL"

VERDICT=PASS

# The accuracy floor is a smoke gate, not a result. The OGB datasets carry their
# published division, but cora's is a positional slice of the source file rather
# than the Planetoid split, so its number is not comparable with published Cora
# figures. Read the floor as "the pipeline produced a model", nothing more.
ABOVE=$(python3 -c "print('yes' if float('$ACC') > $ACC_FLOOR else 'no')" 2>/dev/null || echo no)
if [ "$ABOVE" = yes ]; then
    ok "accuracy $(num "$ACC") clears the $ACC_FLOOR floor for $DATASET"
else
    warn "accuracy $(num "$ACC") is at or below the $ACC_FLOOR floor"
    VERDICT=CHECK
fi

# Only the small graphs get this. Sampling above cora runs multi-threaded, and
# the thread that claims a batch affects the draw, so repeat runs there differ
# by design and an equality check would fail for the wrong reason.
if [ "$DETERMINISM" = 1 ]; then
    say "     determinism: same seed, second run"
    gql gnn_train_repeat "CALL gnn_train('$SAMPLE','node_features',{
            ${TRAIN_OPTS}model:'graphsage',
            hiddenDim:256,
            epochs:$EPOCHS,
            randomSeed:42,
            patience:999,
            tolerance:0
        }) YIELD testAccuracy RETURN *" \
        || fail "the determinism re-run did not finish"
    ACC2=$(col testAccuracy)
    SAME=$(python3 -c "print('yes' if abs(float('$ACC')-float('$ACC2')) < 1e-6 else 'no')" 2>/dev/null || echo no)
    if [ "$SAME" = yes ]; then
        ok "reproducible: $(num "$ACC") twice with seed 42"
    else
        warn "same seed gave $(num "$ACC") then $(num "$ACC2")"
        VERDICT=CHECK
    fi
fi

if [ "$VERDICT" = PASS ]; then
    printf '\033[0;32m[bench] PASS\033[0m  %s end to end in %s s\n' "$DATASET" "$TOTAL"
    RC=0
else
    printf '\033[1;33m[bench] CHECK\033[0m  %s completed but the accuracy gate did not match\n' "$DATASET"
    RC=2
fi
exit $RC
