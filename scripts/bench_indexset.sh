#!/bin/bash
# Spec #3 T3.13 — IndexSet benchmark harness.
#
# For each (dataset, IndexSet preset) pair in
#   {cora_gnn, ogbn-arxiv, ogbn-products} × {ALL, GNN_MINIMAL, READONLY_TRAVERSAL}
# measure:
#   - wall clock of graph_project (curl request)
#   - final projection dir size (bytes)
#   - .leaf file count
#   - peak RSS of the mdb server process (VmHWM sampled from /proc)
#
# Stack: SERIAL_SCAN + RADIX (MDB_PROJECTION_SERIAL_SCAN=1 MDB_PROJECTION_SORTER=radix)
#
# Strict: papers100M is NEVER projected here: this bench is scoped to graphs that
# finish in minutes, not to a 111M-node graph.
#
# Output:
#   CSV at /tmp/bench_indexset_<ts>.csv with columns
#     dataset,indexSet,wall_clock_sec,proj_bytes,leaf_count,peak_rss_mb,num_nodes,num_edges
#   plus pretty-printed summary table to stdout with reduction percentages vs
#   the ALL baseline per dataset.
#
# Usage: ./scripts/bench_indexset.sh
# Env:
#   MDB            Path to mdb binary (default: ./build/Release/bin/mdb)
#   PORT_BASE      Starting port (default: 19891; +1 per run to avoid collision)
#   DATASETS       Override dataset list (space-separated basenames under data/dbs/gql)
#   MODES          Override preset list  (space-separated)

set -euo pipefail

MDB=${MDB:-./build/Release/bin/mdb}
PORT_BASE=${PORT_BASE:-19891}
BENCH_PROJ=${BENCH_PROJ:-bench_indexset_tmp}

# Dataset configuration: name  db_path  node_label  edge_type
# (Paper/CITES for cora_gnn; Node/CONNECTS for the OGB datasets — these were
# imported via scripts/gnn_datasets/stream_convert_ogb.py without --directed,
# so edges carry the `CONNECTS` label.)
declare -a DATASETS_DEFAULT=(
    "cora_gnn|data/dbs/gql/cora_gnn|Paper|CITES"
    "ogbn-arxiv|data/dbs/gql/ogbn-arxiv|Node|CONNECTS"
    "ogbn-products|data/dbs/gql/ogbn-products|Node|CONNECTS"
)

if [[ -n "${DATASETS:-}" ]]; then
    # Rebuild entries for requested datasets (must match a known config)
    NEW_DATASETS=()
    for ds in $DATASETS; do
        for entry in "${DATASETS_DEFAULT[@]}"; do
            if [[ "${entry%%|*}" == "$ds" ]]; then
                NEW_DATASETS+=("$entry")
                break
            fi
        done
    done
    DATASETS_LIST=("${NEW_DATASETS[@]}")
else
    DATASETS_LIST=("${DATASETS_DEFAULT[@]}")
fi

MODES_LIST=(${MODES:-ALL GNN_MINIMAL READONLY_TRAVERSAL})

# Strict guard: papers100M is out of scope here.
for entry in "${DATASETS_LIST[@]}"; do
    name="${entry%%|*}"
    if [[ "$name" == "papers100M" || "$name" == *papers100M* ]]; then
        echo "ERROR: papers100M is out of scope for bench_indexset.sh: projecting it takes tens of minutes. Aborting." >&2
        exit 3
    fi
done

TS=$(date +%s)
CSV_OUT="/tmp/bench_indexset_${TS}.csv"
echo "dataset,indexSet,wall_clock_sec,proj_bytes,leaf_count,peak_rss_mb,num_nodes,num_edges" > "$CSV_OUT"

SRV_PID=
SAMPLER_PID=
CURRENT_DB=
CURRENT_PROJ=

cleanup() {
    # Kill peak-RSS sampler if present
    if [[ -n "${SAMPLER_PID:-}" ]] && kill -0 "$SAMPLER_PID" 2>/dev/null; then
        kill "$SAMPLER_PID" 2>/dev/null || true
        wait "$SAMPLER_PID" 2>/dev/null || true
    fi
    # Kill server if present
    if [[ -n "${SRV_PID:-}" ]] && kill -0 "$SRV_PID" 2>/dev/null; then
        kill "$SRV_PID" 2>/dev/null || true
        # Give it a moment, then SIGKILL if still alive
        for _ in 1 2 3 4 5 6 7 8 9 10; do
            kill -0 "$SRV_PID" 2>/dev/null || break
            sleep 0.3
        done
        kill -9 "$SRV_PID" 2>/dev/null || true
        wait "$SRV_PID" 2>/dev/null || true
    fi
    # Best-effort drop of the most recent projection
    if [[ -n "${CURRENT_DB:-}" && -n "${CURRENT_PROJ:-}" ]]; then
        "$MDB" drop-projection "$CURRENT_DB" "$CURRENT_PROJ" >/dev/null 2>&1 || true
    fi
}
trap cleanup EXIT INT TERM

# Launch the mdb server on $1=db_path, $2=port, $3=logfile.
# Exports SERIAL_SCAN=1 + SORTER=radix (the target stack for Spec #1 + #2).
start_server() {
    local db="$1" port="$2" logfile="$3"
    MDB_PROJECTION_SERIAL_SCAN=1 \
    MDB_PROJECTION_SORTER=radix \
        "$MDB" server "$db" --port "$port" --timeout 3600 \
        > "$logfile" 2>&1 &
    SRV_PID=$!
    CURRENT_DB="$db"
    # Wait for readiness (up to ~20 s).
    for _ in $(seq 1 40); do
        if curl -sSf -o /dev/null --data-binary "RETURN 1" -H "Accept: text/csv" \
                "http://127.0.0.1:$port/" 2>/dev/null; then
            return 0
        fi
        sleep 0.5
    done
    echo "ERROR: server on port $port did not come up in 20 s" >&2
    cat "$logfile" >&2 || true
    return 2
}

stop_server() {
    if [[ -n "${SRV_PID:-}" ]] && kill -0 "$SRV_PID" 2>/dev/null; then
        kill "$SRV_PID" 2>/dev/null || true
        for _ in 1 2 3 4 5 6 7 8 9 10; do
            kill -0 "$SRV_PID" 2>/dev/null || break
            sleep 0.3
        done
        kill -9 "$SRV_PID" 2>/dev/null || true
        wait "$SRV_PID" 2>/dev/null || true
    fi
    SRV_PID=
    CURRENT_DB=
}

# Peak RSS sampler — writes to $1 until the tracked pid exits.
# Polls /proc/<pid>/status every 1 s and records VmHWM (kB).
start_peak_rss_sampler() {
    local pid="$1" out="$2"
    : > "$out"
    (
        while kill -0 "$pid" 2>/dev/null; do
            if [[ -r /proc/$pid/status ]]; then
                # VmHWM is the high-water mark — already a running max.
                awk '/^VmHWM:/ {print $2}' /proc/$pid/status 2>/dev/null >> "$out" || true
            fi
            sleep 1
        done
    ) &
    SAMPLER_PID=$!
}

stop_peak_rss_sampler() {
    if [[ -n "${SAMPLER_PID:-}" ]] && kill -0 "$SAMPLER_PID" 2>/dev/null; then
        kill "$SAMPLER_PID" 2>/dev/null || true
        wait "$SAMPLER_PID" 2>/dev/null || true
    fi
    SAMPLER_PID=
}

# Drop a projection before benchmarking a fresh run (no partial state).
drop_proj_offline() {
    local db="$1" proj="$2"
    "$MDB" drop-projection "$db" "$proj" >/dev/null 2>&1 || true
}

# Core per-run logic: launch server, issue CALL graph_project, sample RSS,
# capture wall clock, measure disk, drop projection.
run_one() {
    local dataset="$1" db="$2" node_label="$3" edge_type="$4" mode="$5" port="$6"
    local proj="${BENCH_PROJ}_${mode}"
    local logfile="/tmp/bench_indexset_${dataset}_${mode}_${port}.log"
    local rssfile="/tmp/bench_indexset_${dataset}_${mode}_${port}.rss"

    CURRENT_PROJ="$proj"
    drop_proj_offline "$db" "$proj"

    echo "--- [$dataset / $mode] starting server on port $port ---"
    start_server "$db" "$port" "$logfile"
    start_peak_rss_sampler "$SRV_PID" "$rssfile"

    local query
    query="CALL graph_project('$proj', '$node_label', '$edge_type', {orientation: 'NATURAL', indexSet: '$mode'}) YIELD graphName, nodeCount, relationshipCount RETURN *"

    # Wall clock of the curl round-trip — this includes B+Tree build.
    local t0 t1 wall
    t0=$(date +%s.%N)
    local response
    response=$(curl -sS --max-time 3000 --data-binary "$query" \
        -H "Accept: text/csv" "http://127.0.0.1:$port/" || true)
    t1=$(date +%s.%N)
    wall=$(LC_NUMERIC=C awk -v t0="$t0" -v t1="$t1" 'BEGIN { printf "%.3f", t1 - t0 }')

    # Pull last VmHWM sample (it's already a high-water mark, so the last entry
    # observable-before-teardown is the max).
    stop_peak_rss_sampler
    local rss_kb=0
    if [[ -s "$rssfile" ]]; then
        rss_kb=$(tail -n 1 "$rssfile")
    fi
    local rss_mb
    rss_mb=$(LC_NUMERIC=C awk -v kb="$rss_kb" 'BEGIN { printf "%.0f", kb/1024 }')

    # Parse CSV response: header line, then one data row:
    #   "graphName","nodeCount","relationshipCount"
    #   "bench_indexset_tmp_ALL",2708,5429
    local data_line nc ec
    data_line=$(echo "$response" | sed -n '2p' | tr -d '"')
    nc=$(echo "$data_line" | awk -F',' '{print $2}')
    ec=$(echo "$data_line" | awk -F',' '{print $3}')
    # Defensive fallbacks
    [[ -z "$nc" ]] && nc=0
    [[ -z "$ec" ]] && ec=0

    local proj_dir="$db/projections/$proj"
    local proj_bytes=0 leaf_count=0
    if [[ -d "$proj_dir" ]]; then
        proj_bytes=$(du -sb "$proj_dir" | awk '{print $1}')
        leaf_count=$(find "$proj_dir" -maxdepth 1 -name '*.leaf' | wc -l)
    fi

    # Clean up: stop server first (releases file handles), then drop the proj.
    stop_server
    drop_proj_offline "$db" "$proj"
    CURRENT_PROJ=

    printf 'RESULT  %-14s %-20s wall=%ss  disk=%s B  leafs=%s  rss=%s MB  N=%s  E=%s\n' \
        "$dataset" "$mode" "$wall" "$proj_bytes" "$leaf_count" "$rss_mb" "$nc" "$ec"

    # Detect a silent failure — server accepted the call but no projection exists.
    if [[ "$proj_bytes" -eq 0 || "$leaf_count" -eq 0 ]]; then
        echo "WARN  projection '$proj' not materialized on $dataset — see $logfile"
    fi

    echo "$dataset,$mode,$wall,$proj_bytes,$leaf_count,$rss_mb,$nc,$ec" >> "$CSV_OUT"
}

echo "================================================================"
echo "IndexSet bench — $(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "  git HEAD   : $(git -C "$(dirname "$0")/.." rev-parse --short HEAD 2>/dev/null || echo unknown)"
echo "  binary     : $MDB"
echo "  stack      : MDB_PROJECTION_SERIAL_SCAN=1  MDB_PROJECTION_SORTER=radix"
echo "  CSV output : $CSV_OUT"
echo "================================================================"

PORT="$PORT_BASE"
for entry in "${DATASETS_LIST[@]}"; do
    IFS='|' read -r name db nl et <<<"$entry"
    if [[ ! -d "$db" ]]; then
        echo "WARN: dataset $name not found at $db — skipping."
        continue
    fi
    for mode in "${MODES_LIST[@]}"; do
        run_one "$name" "$db" "$nl" "$et" "$mode" "$PORT"
        PORT=$((PORT + 1))
    done
done

# --- Pretty-printed summary with reduction vs ALL -----------------------------
echo
echo "================================================================"
echo "Summary (reduction percentages vs ALL baseline, per dataset)"
echo "================================================================"
LC_NUMERIC=C awk -F',' '
NR == 1 { next }
{
    ds = $1; mode = $2
    wall[ds "|" mode] = $3 + 0
    bytes[ds "|" mode] = $4 + 0
    leafs[ds "|" mode] = $5 + 0
    rss[ds "|" mode] = $6 + 0
    ds_seen[ds] = 1
    mode_seen[mode] = 1
}
END {
    # Print header
    printf "%-14s %-20s %-11s %-13s %-8s %-10s %-20s %-20s\n", \
        "dataset", "mode", "wall_s", "disk_MB", "leafs", "rss_MB", \
        "disk_vs_ALL", "wall_vs_ALL"
    printf "%-14s %-20s %-11s %-13s %-8s %-10s %-20s %-20s\n", \
        "------", "----", "------", "-------", "-----", "------", "-----------", "-----------"

    # Iterate datasets in a stable order.
    n = 0
    for (ds in ds_seen) order[++n] = ds
    # Bubble sort for deterministic CSV-order-preserving output.
    for (i = 1; i <= n; i++)
        for (j = i + 1; j <= n; j++)
            if (order[i] > order[j]) { t = order[i]; order[i] = order[j]; order[j] = t }

    modes[1] = "ALL"; modes[2] = "GNN_MINIMAL"; modes[3] = "READONLY_TRAVERSAL"
    for (i = 1; i <= n; i++) {
        ds = order[i]
        base_wall  = wall[ds "|ALL"]  + 0
        base_bytes = bytes[ds "|ALL"] + 0
        for (m = 1; m <= 3; m++) {
            mode = modes[m]
            key  = ds "|" mode
            if (!(key in wall)) continue
            disk_mb = bytes[key] / (1024 * 1024)
            wall_s  = wall[key]
            if (mode == "ALL" || base_bytes == 0) {
                disk_red = "   (baseline)"
            } else {
                d = (1 - bytes[key] / base_bytes) * 100
                disk_red = sprintf("%+7.1f%% vs ALL", -d)   # negative = smaller
            }
            if (mode == "ALL" || base_wall == 0) {
                wall_red = "   (baseline)"
            } else {
                w = (1 - wall[key] / base_wall) * 100
                wall_red = sprintf("%+7.1f%% vs ALL", -w)
            }
            printf "%-14s %-20s %-11.3f %-13.2f %-8d %-10d %-20s %-20s\n", \
                ds, mode, wall_s, disk_mb, leafs[key], rss[key], disk_red, wall_red
        }
    }
}' "$CSV_OUT"

echo
echo "CSV: $CSV_OUT"
