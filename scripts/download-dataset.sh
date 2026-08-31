#!/usr/bin/env bash
# ============================================================================
# OGB Dataset Downloader (raw only, no conversion, no import)
# ============================================================================
# Downloads a raw Open Graph Benchmark dataset zip + unzips it. Leaves the
# extracted NPZ + splits under $MDB_HOME/ogb_data/<name>/. Nothing is
# converted to GQL and nothing is imported into MillenniumDB — that is the
# job of stream_convert_ogb.py and `mdb import` respectively.
#
# Use case: pre-stage a workstation so that when you actually sit down to
# run experiments, the 50-GB download is already done and you can jump
# straight to convert + import.
#
# Usage:
#   scripts/download-dataset.sh papers100M           # ~50 GB download
#   scripts/download-dataset.sh arxiv                # ~300 MB (smoke test)
#   scripts/download-dataset.sh products             # ~1.5 GB
#   scripts/download-dataset.sh mag                  # ~500 MB
#   scripts/download-dataset.sh --list               # print catalogue
#
# Idempotent: if $MDB_HOME/ogb_data/<name>/raw/ already contains a non-empty
# data.npz, the script is a no-op.
#
# No Python, no OGB library dependency. Plain wget + unzip. If either tool
# is missing the script aborts with a clear error pointing to onboard.sh.
# ============================================================================

set -euo pipefail

# ---------------------------------------------------------------------------
# Dataset catalogue (URL + expected unzip directory name)
# ---------------------------------------------------------------------------
# All URLs are the stable SNAP mirror; the "expected_dir" is the top-level
# folder name that the zip extracts into — OGB names them with underscores
# (ogbn_papers100M) even though the zip is called papers100M-bin.zip.

catalogue() {
    cat <<'EOF'
# dataset       size_gb  zip_url                                                                               expected_dir
arxiv           0.3      http://snap.stanford.edu/ogb/data/nodeproppred/arxiv.zip                              ogbn_arxiv
products        1.5      http://snap.stanford.edu/ogb/data/nodeproppred/products.zip                           ogbn_products
mag             0.5      http://snap.stanford.edu/ogb/data/nodeproppred/mag.zip                                ogbn_mag
papers100M      50       http://snap.stanford.edu/ogb/data/nodeproppred/papers100M-bin.zip                     ogbn_papers100M
proteins        0.2      http://snap.stanford.edu/ogb/data/nodeproppred/proteins.zip                           ogbn_proteins
EOF
}

usage() {
    sed -n '3,24p' "$0" | sed 's/^# \{0,1\}//'
    echo ""
    echo "Available datasets:"
    catalogue | awk 'NR>1 {printf "  %-14s  ~%s GB\n", $1, $2}'
}

# ---------------------------------------------------------------------------
# Args
# ---------------------------------------------------------------------------
if [ $# -eq 0 ] || [ "${1:-}" = "--help" ] || [ "${1:-}" = "-h" ]; then
    usage
    exit 0
fi

if [ "${1:-}" = "--list" ] || [ "${1:-}" = "-l" ]; then
    catalogue
    exit 0
fi

DATASET="$1"
LINE="$(catalogue | awk -v d="$DATASET" 'NR>1 && $1==d')"
if [ -z "$LINE" ]; then
    echo "ERROR: unknown dataset '$DATASET'" >&2
    echo "Run '$0 --list' to see available datasets." >&2
    exit 1
fi

SIZE_GB="$(echo "$LINE" | awk '{print $2}')"
URL="$(echo "$LINE" | awk '{print $3}')"
DIR_NAME="$(echo "$LINE" | awk '{print $4}')"

# ---------------------------------------------------------------------------
# Preflight
# ---------------------------------------------------------------------------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MDB_HOME="${MDB_HOME:-$(cd "$SCRIPT_DIR/.." && pwd)}"
OUTPUT_DIR="$MDB_HOME/ogb_data"
mkdir -p "$OUTPUT_DIR"

TARGET="$OUTPUT_DIR/$DIR_NAME"

for tool in wget unzip; do
    if ! command -v "$tool" >/dev/null 2>&1; then
        echo "ERROR: '$tool' not installed." >&2
        echo "Run onboard.sh first, or: sudo apt install $tool" >&2
        exit 1
    fi
done

# ---------------------------------------------------------------------------
# Idempotency — skip if raw already present
# ---------------------------------------------------------------------------
if [ -f "$TARGET/raw/data.npz" ] && [ -s "$TARGET/raw/data.npz" ]; then
    RAW_SIZE="$(du -sh "$TARGET" | cut -f1)"
    echo "✓ $DATASET already downloaded at $TARGET ($RAW_SIZE)"
    echo "  Nothing to do. Delete $TARGET to re-download."
    exit 0
fi

# ---------------------------------------------------------------------------
# Disk check — warn if <1.5× expected size free
# ---------------------------------------------------------------------------
FREE_GB="$(df -BG "$OUTPUT_DIR" | awk 'NR==2 {gsub(/G/,"",$4); print $4}')"
NEEDED_GB="$(awk -v s="$SIZE_GB" 'BEGIN {printf "%d", s*2 + 5}')"  # zip + extract + headroom
if [ "${FREE_GB:-0}" -lt "$NEEDED_GB" ]; then
    echo "⚠ WARNING: only ${FREE_GB} GB free on $(df -h "$OUTPUT_DIR" | awk 'NR==2{print $6}')." >&2
    echo "  $DATASET needs ~${NEEDED_GB} GB transient (zip + unzipped). Continuing anyway." >&2
fi

# ---------------------------------------------------------------------------
# Download + unzip
# ---------------------------------------------------------------------------
ZIP_NAME="$(basename "$URL")"
ZIP_PATH="$OUTPUT_DIR/$ZIP_NAME"

echo "=========================================================="
echo "  Downloading OGB dataset: $DATASET"
echo "  URL:        $URL"
echo "  Size:       ~${SIZE_GB} GB"
echo "  Target dir: $TARGET"
echo "  Zip cache:  $ZIP_PATH"
echo "=========================================================="

cd "$OUTPUT_DIR"

if [ ! -f "$ZIP_NAME" ] || [ ! -s "$ZIP_NAME" ]; then
    echo "→ Downloading (wget)..."
    # -c resumes partial downloads if wget was killed previously.
    # --show-progress keeps visible progress for long downloads.
    wget -c --show-progress "$URL"
else
    echo "→ Zip already cached ($(du -h "$ZIP_NAME" | cut -f1)); skipping download"
fi

echo "→ Unzipping $ZIP_NAME..."
unzip -q -o "$ZIP_NAME"

# Some datasets (papers100M) extract as "papers100M-bin/"; OGB's own
# downloader renames that directory to "ogbn_papers100M/" after the fact.
# We replicate that rename so our catalogue "expected_dir" is accurate.
ALT_NAMES=(
    "papers100M-bin:ogbn_papers100M"
    "arxiv:ogbn_arxiv"
    "products:ogbn_products"
    "mag:ogbn_mag"
    "proteins:ogbn_proteins"
)
for pair in "${ALT_NAMES[@]}"; do
    src="${pair%:*}"
    dst="${pair#*:}"
    # No "&& [ ! -d $dst ]" guard here. An empty destination left by an aborted
    # run does not trip the file-based idempotency check above, so with the guard
    # the rename was skipped and the run paid the whole download and extraction
    # before failing with a complaint about the archive layout.
    if [ -d "$src" ]; then
        if [ -d "$dst" ]; then
            # A plain mv into an existing directory nests it as $dst/$src.
            # An empty leftover is safe to discard; a populated one is not
            # something this script should silently overwrite.
            if rmdir "$dst" 2>/dev/null; then
                mv "$src" "$dst"
            else
                echo "ERROR: $dst already exists and is not empty." >&2
                echo "       Remove or move it, then re-run." >&2
                exit 1
            fi
        else
            mv "$src" "$dst"
        fi
        break
    fi
done

# Sanity check
if [ ! -f "$TARGET/raw/data.npz" ]; then
    echo "ERROR: expected $TARGET/raw/data.npz after unzip, but file not found." >&2
    echo "Unzipped contents of $OUTPUT_DIR:" >&2
    ls -la "$OUTPUT_DIR" >&2
    exit 1
fi

# Cleanup zip (raw is now self-contained)
rm -f "$ZIP_NAME"

FINAL_SIZE="$(du -sh "$TARGET" | cut -f1)"
echo ""
echo "=========================================================="
echo "✓ $DATASET ready at $TARGET ($FINAL_SIZE)"
echo "=========================================================="
echo ""
echo "Next steps when you're ready to experiment:"
echo "  # 1) Convert raw NPZ to MillenniumDB .gql + .npy"
echo "  scripts/gnn_datasets/.venv/bin/python scripts/gnn_datasets/stream_convert_ogb.py \\"
echo "      --raw-dir $TARGET/raw/ \\"
echo "      --split-dir $TARGET/split/time/ \\"
echo "      --output data/example/gql/ogbn-$DATASET/ \\"
echo "      --dataset-name $DATASET \\"
echo "      --directed"
echo ""
echo "  # 2) Import into MillenniumDB with node features registered"
echo "  ./build/Release/bin/mdb import --format gql \\"
echo "      data/example/gql/ogbn-$DATASET/${DATASET}.gql \\"
echo "      data/dbs/gql/$DATASET \\"
echo "      --with-tensors data/example/gql/ogbn-$DATASET/${DATASET}_features.npy"
echo ""
