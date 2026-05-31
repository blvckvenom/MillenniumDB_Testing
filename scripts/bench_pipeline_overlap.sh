#!/usr/bin/env bash
# scripts/bench_pipeline_overlap.sh — Fix #21 + Fix #22 A/B/C/D bench
#
# Runs four configurations on the same papers100M_paper_und sample
# and reports total wall-clock + per-phase breakdown.
#
# Configs:
#   A baseline — current behavior, no pipeline, no fadvise
#   B pipeline only — Fix #21 active, Fix #22 disabled
#   C pipeline + fadvise — Fix #21 + Fix #22 both active
#   D + external_sort — config C + MDB_GNN_REORDER_STRATEGY=external_sort
#
# Required state:
#   - papers100M_paper_und sample exists under data/dbs/gql/papers100M/samples/
#   - node_features.fmat + .rmap exist in data/dbs/gql/papers100M/gnn_features/
#   - server NOT running on port 29950
#   - ~110 GB free disk space for the rebuilt outputs
#
# Usage:
#   ./scripts/bench_pipeline_overlap.sh
#   OUT_DIR=~/Desktop/spec13_papers100m_e2e/post_pop_os/27_pipeline_overlap \
#     ./scripts/bench_pipeline_overlap.sh

set -euo pipefail

# Resolve repo root from this script's location.
REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DB="${DB:-$REPO_DIR/data/dbs/gql/papers100M}"
MDB="${MDB:-$REPO_DIR/build/Release/bin/mdb}"
PORT="${PORT:-29950}"
OUT_DIR="${OUT_DIR:-$HOME/Desktop/spec13_papers100m_e2e/post_pop_os/27_pipeline_overlap}"
mkdir -p "$OUT_DIR"

if [[ ! -x "$MDB" ]]; then
    echo "FATAL: mdb binary not found at $MDB"
    exit 1
fi
if [[ ! -d "$DB" ]]; then
    echo "FATAL: database not found at $DB"
    exit 1
fi

# Pre-flight: write the query once.
cat > "$OUT_DIR/query.gql" <<'EOF'
CALL gnn_build_feature_store("papers100M_paper_und", "node_features",
    {gpu_budget_mb: 2048, cpu_budget_mb: 5290}) YIELD * RETURN *
EOF

run_one() {
    local label="$1"; shift
    local env_args=("$@")
    echo "=== $label ==="
    echo "env: ${env_args[*]}"

    # Cleanup partial outputs (preserves source + caches for Fix #14 idempotency)
    rm -f "$DB/gnn_features/node_features_reordered.fmat" \
          "$DB/gnn_features/node_features_reordered.rmap" \
          "$DB/gnn_features/node_features_reordered.rmap.idx" \
          "$DB/gnn_features/node_features_store.meta"
    rm -rf "$DB/samples/papers100M_paper_und/packed_slim"

    # Launch server with given env
    local log="$OUT_DIR/$label.serverlog"
    env "${env_args[@]}" nohup "$MDB" server "$DB" --port "$PORT" --timeout 86400 \
        > "$log" 2>&1 &
    local pid=$!
    sleep 7

    local t0
    t0=$(date +%s)
    curl -s --max-time 86400 -X POST -H 'Content-Type: application/gql' \
        --data-binary @"$OUT_DIR/query.gql" \
        -o "$OUT_DIR/$label.csv" \
        -w "%{http_code} time=%{time_total}\n" \
        "http://localhost:$PORT/gql" > "$OUT_DIR/$label.curl"
    local elapsed=$(( $(date +%s) - t0 ))

    kill -15 "$pid" 2>/dev/null || true
    sleep 5
    kill -9 "$pid" 2>/dev/null || true
    echo "$label: ${elapsed}s"
    grep -E '^\[FourLevelStore\]|create_reordered.*done|L4 packed_slim done|pass[12] done' \
         "$log" | tail -30 > "$OUT_DIR/$label.phases" || true
}

# A: baseline — Fix #14/#16/#17 + chunked, no pipeline, no fadvise extra
run_one A_baseline MDB_GNN_NO_FADVISE=1

# B: Fix #21 only — pipeline overlap on, fadvise still disabled
run_one B_pipeline_only MDB_GNN_PIPELINE_OVERLAP=1 MDB_GNN_NO_FADVISE=1

# C: Fix #21 + Fix #22 — both on
run_one C_pipeline_plus_fadvise MDB_GNN_PIPELINE_OVERLAP=1

# D: optional — add external_sort
run_one D_extsort_plus_all MDB_GNN_PIPELINE_OVERLAP=1 \
    MDB_GNN_REORDER_STRATEGY=external_sort

echo
echo "=== Summary ==="
for f in A_baseline B_pipeline_only C_pipeline_plus_fadvise D_extsort_plus_all; do
    if [[ -f "$OUT_DIR/$f.curl" ]]; then
        printf '%s ' "$f"; cat "$OUT_DIR/$f.curl"
    fi
done
