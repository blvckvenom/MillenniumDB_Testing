#!/bin/bash
# Tier 2/3 benchmark: measure wall clock and peak RSS of graph_project under
# CLASSIC vs RADIX backends on a given dataset. Output CSV, one line per run.
#
# Usage:   ./scripts/bench_projection_radix.sh <db_path> <node_label> <edge_type> [num_reps]
# Env:     PORT=19879  NUM_REPS=3
#
# CSV columns: dataset,backend,rep,node_count,edge_count,wall_clock_s,peak_rss_mb
#
# Spec reference: §8.5 (Tier 2 validation), §8.6 (Tier 3 scaling).
set -euo pipefail

DB="${1:-data/dbs/gql/ogbn-arxiv}"
NODE_LABEL="${2:-Paper}"
EDGE_TYPE="${3:-CITES}"
NUM_REPS="${NUM_REPS:-${4:-3}}"
PORT="${PORT:-19879}"
MDB=${MDB:-./build/Release/bin/mdb}
DATASET=$(basename "$DB")
OUT="/tmp/bench_${DATASET}_$(date +%s).csv"

echo "dataset,backend,rep,node_count,edge_count,wall_clock_s,peak_rss_mb" > "$OUT"
echo "=== Bench $DATASET: $NUM_REPS reps per backend ==="
echo "CSV: $OUT"
echo

cleanup() {
    [[ -n "${SRV_PID:-}" ]] && kill -0 "$SRV_PID" 2>/dev/null && kill "$SRV_PID"
    wait 2>/dev/null || true
}
trap cleanup EXIT

run_one() {
    local backend="$1"
    local rep="$2"
    local proj="bench_${backend}_r${rep}"

    # Launch server
    MDB_PROJECTION_SORTER="$backend" "$MDB" server "$DB" --port "$PORT" --timeout 1800 \
        > "/tmp/bench_srv_${backend}_r${rep}.log" 2>&1 &
    SRV_PID=$!

    # Wait for ready
    for i in $(seq 1 20); do
        curl -sSf -o /dev/null --data-binary "RETURN 1" -H "Accept: text/csv" \
            "http://127.0.0.1:$PORT/" 2>/dev/null && break
        sleep 0.5
    done

    # Time the projection via HTTP
    local t0=$(date +%s.%N)
    local response
    response=$(curl -sS --data-binary \
        "CALL graph_project('$proj', '$NODE_LABEL', '$EDGE_TYPE') YIELD graphName, nodeCount, relCount RETURN *" \
        -H "Accept: text/csv" "http://127.0.0.1:$PORT/")
    local t1=$(date +%s.%N)
    local wall=$(awk -v t0="$t0" -v t1="$t1" 'BEGIN { printf "%.3f", t1 - t0 }')

    # Parse response: 2nd line like: "bench_classic_r1",169343,1166243
    local counts
    counts=$(echo "$response" | sed -n '2p' | tr -d '"')
    local nc=$(echo "$counts" | cut -d',' -f2)
    local ec=$(echo "$counts" | cut -d',' -f3)

    # Peak RSS from /proc
    local rss_kb=0
    if [[ -f "/proc/$SRV_PID/status" ]]; then
        rss_kb=$(awk '/^VmHWM:/ {print $2}' /proc/$SRV_PID/status)
    fi
    local rss_mb=$(awk -v kb="$rss_kb" 'BEGIN { printf "%.0f", kb/1024 }')

    # Cleanup projection + server
    curl -sSf --data-binary "RETURN 1" -H "Accept: text/csv" "http://127.0.0.1:$PORT/" > /dev/null 2>&1 || true
    kill "$SRV_PID" 2>/dev/null; wait "$SRV_PID" 2>/dev/null; SRV_PID=
    "$MDB" drop-projection "$DB" "$proj" > /dev/null 2>&1 || true

    echo "$DATASET,$backend,$rep,$nc,$ec,$wall,$rss_mb" | tee -a "$OUT"
}

for rep in $(seq 1 "$NUM_REPS"); do
    for backend in classic radix; do
        run_one "$backend" "$rep"
    done
done

echo
echo "=== Summary ==="
awk -F',' 'NR>1 {
    key=$2;
    wall_sum[key]+=$6; rss_max[key]=($7>rss_max[key]?$7:rss_max[key]); n[key]++
}
END {
    for (k in n) printf "  %-8s  avg_wall=%.3fs  peak_rss=%s MB  over %d runs\n", k, wall_sum[k]/n[k], rss_max[k], n[k]
}' "$OUT"

echo
echo "CSV: $OUT"
