#!/usr/bin/env bash
# Thesis demo script — end-to-end MillenniumDB GNN pipeline.
#
# Demonstrates the unified graph → GNN → queryable embeddings pipeline at
# a configurable dataset scale. Runs locally on a single machine with a
# commodity GPU (tested on GTX 1660 Super 6 GB + 32 GB RAM + 12 cores).
#
# Usage:
#   scripts/thesis_demo.sh [--dataset arxiv|products|cora]
#                          [--port 18123]
#                          [--epochs 30]
#                          [--skip-import]
#                          [--skip-rebuild]
#
# Prereqs:
#   - mdb binary built at build/Release/bin/mdb (run `scripts/onboard.sh` first)
#   - Raw data at data/example/gql/ogbn-<dataset>/ (ogbn_<ds>.gql + .npy files)
#     or cora_gnn DB already imported
#
# Produces:
#   - Projection at data/dbs/gql/<dataset>/projections/<dataset>_gnn/
#   - Training outputs at gnn_output/<run_name>/ (model.pt, embeddings.npy,
#     checkpoints/, training_log.json)
#   - Queryable n.embedding property accessible via GQL `USE <proj> MATCH ...`

set -euo pipefail

cd "$(dirname "$0")/.."

DATASET="arxiv"
PORT=18123
EPOCHS=30
SKIP_IMPORT=0
SKIP_REBUILD=0

for arg in "$@"; do
    case "$arg" in
        --dataset=*) DATASET="${arg#*=}" ;;
        --port=*)    PORT="${arg#*=}" ;;
        --epochs=*)  EPOCHS="${arg#*=}" ;;
        --skip-import)  SKIP_IMPORT=1 ;;
        --skip-rebuild) SKIP_REBUILD=1 ;;
        -h|--help)
            sed -n '/^# Usage:/,/^# Produces:/p' "$0" | sed 's/^# \?//'
            exit 0
            ;;
    esac
done

# Dataset-specific config
case "$DATASET" in
    arxiv)
        DB_DIR="data/dbs/gql/ogbn-arxiv"
        GQL_FILE="data/example/gql/ogbn-arxiv/ogbn_arxiv.gql"
        FEAT_NPY="data/example/gql/ogbn-arxiv/ogbn_arxiv_features.npy"
        PROJ_NAME="arxiv_gnn"
        SAMPLE_NAME="arxiv_s"
        NODE_LABEL="Node"
        EDGE_TYPE="CONNECTS"
        FANOUTS="[10, 5]"
        BATCH_SIZE=256
        INFERENCE_BATCH=2048
        GPU_MB=4096
        CPU_MB=16384
        ;;
    products)
        DB_DIR="data/dbs/gql/ogbn-products"
        GQL_FILE="data/example/gql/ogbn-products/ogbn_products.gql"
        FEAT_NPY="data/example/gql/ogbn-products/ogbn_products_features.npy"
        PROJ_NAME="products_gnn"
        SAMPLE_NAME="products_s"
        NODE_LABEL="Node"
        EDGE_TYPE="CONNECTS"
        FANOUTS="[10, 5]"
        BATCH_SIZE=512
        INFERENCE_BATCH=2048
        GPU_MB=4096
        CPU_MB=16384
        ;;
    cora)
        DB_DIR="data/dbs/gql/cora_gnn"
        GQL_FILE=""  # already imported
        FEAT_NPY=""
        PROJ_NAME="cora_thesis"
        SAMPLE_NAME="cora_s"
        NODE_LABEL="Paper"
        EDGE_TYPE="CITES"
        FANOUTS="[10, 5]"
        BATCH_SIZE=64
        INFERENCE_BATCH=512
        GPU_MB=1024
        CPU_MB=4096
        SKIP_IMPORT=1
        ;;
    *)
        echo "ERROR: unknown dataset '$DATASET' (expected: arxiv | products | cora)" >&2
        exit 1
        ;;
esac

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-11}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-11}"

TS=$(date +%Y%m%d_%H%M%S)
LOG="/tmp/thesis_demo_${DATASET}_${TS}.log"
RUN_NAME="thesis_demo_${TS}"

echo "=== Thesis demo: $DATASET (port $PORT, epochs $EPOCHS, GPU ${GPU_MB}MB, log $LOG) ==="

# ---------- Import ----------
if [ "$SKIP_IMPORT" -eq 0 ] && [ -n "$GQL_FILE" ]; then
    if [ ! -d "$DB_DIR/gnn_features" ]; then
        echo "=== [$(date +%H:%M:%S)] Import ($GQL_FILE) with --with-tensors ==="
        pkill -9 mdb 2>/dev/null || true
        rm -rf "$DB_DIR"
        ./build/Release/bin/mdb import "$GQL_FILE" "$DB_DIR" --with-tensors "$FEAT_NPY"
    else
        echo "=== DB already imported at $DB_DIR (has gnn_features/). Skipping. ==="
    fi
fi

# ---------- Server ----------
pkill -9 mdb 2>/dev/null || true; sleep 2
./build/Release/bin/mdb server "$DB_DIR" --port "$PORT" --browser false --threads 11 > "$LOG" 2>&1 &
SRV=$!
sleep 6
echo "Server PID=$SRV"

# Background: stream training-relevant lines
(tail -f -n 0 "$LOG" 2>/dev/null | grep --line-buffered -iE "epoch|EmbeddingWriter|chunk|elapsed|error|exception|terminate|CUDA" &)
TAIL=$!

query() {
    curl -s -m 3000 -H "Accept:text/csv" -X POST "http://localhost:${PORT}" -d "$1"
}

# ---------- Project ----------
if [ "$SKIP_REBUILD" -eq 0 ]; then
    echo "=== [$(date +%H:%M:%S)] Drop old projection (if any) ==="
    query "CALL drop_projection('$PROJ_NAME')" 2>&1 | head -2 || true

    echo "=== [$(date +%H:%M:%S)] graph_project FULL STACK ==="
    query "CALL graph_project('$PROJ_NAME', '$NODE_LABEL', '$EDGE_TYPE', {
        orientation: 'UNDIRECTED',
        indexSet: 'GNN_MINIMAL',
        leafFormat: 'DELTA_VARINT',
        graphStorage: 'CSR_HYBRID',
        includeFeatures: 'node_features',
        labelProperty: 'label',
        splitProperty: 'split'
    }) YIELD graphName, nodeCount, relCount, projectMillis, featureDim, numClasses
    RETURN graphName, nodeCount, relCount, projectMillis, featureDim, numClasses"
fi

echo "=== [$(date +%H:%M:%S)] Sample (batchSize=$BATCH_SIZE) ==="
query "CALL gnn_offline_sample('$PROJ_NAME', '$SAMPLE_NAME', $FANOUTS, {
    batchSize: $BATCH_SIZE, randomSeed: 42,
    usePredefinedSplits: true, orientation: 'UNDIRECTED'
}) YIELD sampleName, totalBatches, trainBatches, testBatches
RETURN sampleName, totalBatches, trainBatches, testBatches"

echo "=== [$(date +%H:%M:%S)] Materialize ==="
query "CALL gnn_materialize_batches('$SAMPLE_NAME', 'node_features', {reorder: 1, numHashes: 2})
YIELD sampleName, totalBatches, reordered RETURN sampleName, totalBatches, reordered"

echo "=== [$(date +%H:%M:%S)] Feature store (GPU ${GPU_MB}MB + CPU ${CPU_MB}MB) ==="
query "CALL gnn_build_feature_store('$SAMPLE_NAME', 'node_features', {
    gpu_budget_mb: $GPU_MB, cpu_budget_mb: $CPU_MB,
    reorder: 1, force: 1
}) YIELD sampleName, l1Nodes, l2Nodes, l3Nodes, l4Nodes, buildTimeMs
RETURN sampleName, l1Nodes, l2Nodes, l3Nodes, l4Nodes, buildTimeMs"

echo "=== [$(date +%H:%M:%S)] Train ($EPOCHS epochs, inferenceBatchSize=$INFERENCE_BATCH) ==="
query "CALL gnn_train('$SAMPLE_NAME', 'node_features', {
    model: 'graphsage',
    hiddenDim: 128, epochs: $EPOCHS, lr: 0.01,
    dropout: 0.5, patience: 10, randomSeed: 42,
    exportEmbeddings: true,
    outputDir: '$RUN_NAME',
    writeProperty: 'embedding',
    inferenceBatchSize: $INFERENCE_BATCH
}) YIELD modelName, ranEpochs, didConverge, bestValAccuracy, testAccuracy, trainSeconds, nodesWritten, l1HitRatio
RETURN modelName, ranEpochs, didConverge, bestValAccuracy, testAccuracy, trainSeconds, nodesWritten, l1HitRatio"

# ---------- Demo queries ----------
echo ""
echo "=== [$(date +%H:%M:%S)] THESIS DEMO QUERIES ==="
echo "-- Label read (sanity) --"
query "USE $PROJ_NAME MATCH (n) RETURN n.label LIMIT 3"
echo "-- Embedding read (MONEY SHOT) --"
query "USE $PROJ_NAME MATCH (n) RETURN n.embedding LIMIT 2"
echo "-- cosineDistance --"
query "USE $PROJ_NAME MATCH (a) WHERE a.id = 0 MATCH (b) WHERE b.id = 1
       RETURN cosineDistance(a.embedding, b.embedding) AS sim"
echo "-- Top-5 similar --"
query "USE $PROJ_NAME MATCH (seed) WHERE seed.id = 0
       MATCH (other) WHERE other.id <> 0
       RETURN other.id AS id, cosineDistance(seed.embedding, other.embedding) AS dist
       ORDER BY dist ASC LIMIT 5"

echo "=== [$(date +%H:%M:%S)] DONE ==="

# ---------- Cleanup ----------
kill $TAIL 2>/dev/null || true
kill $SRV 2>/dev/null || true
wait $SRV 2>/dev/null || true

echo ""
echo "Artifacts:"
echo "  Log:         $LOG"
echo "  Outputs:     $DB_DIR/projections/$PROJ_NAME/gnn_output/$RUN_NAME/"
ls "$DB_DIR/projections/$PROJ_NAME/gnn_output/$RUN_NAME/" 2>/dev/null | sed 's/^/    /'
