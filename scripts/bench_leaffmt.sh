#!/usr/bin/env bash
#
# Spec #5 T5.13 — leafFormat benchmark harness.
#
# For each (dataset, leafFormat) pair in
#   {cora_gnn, ogbn-arxiv, ogbn-products} x {BITSET, DELTA_VARINT}
# measure:
#   - wall clock of graph_project (curl round-trip)
#   - peak RSS of the mdb server process (VmHWM sampled from /proc)
#   - total .leaf disk bytes (sum across all indexes in the projection)
#   - per-index .leaf size (second CSV section, for breakdown)
#   - wall clock of a full-range edge scan as read-side throughput proxy
#     (USE proj MATCH (n)-[e]->(m) RETURN count(e))
#
# Stack: SERIAL_SCAN + RADIX (MDB_PROJECTION_SERIAL_SCAN=1 MDB_PROJECTION_SORTER=radix)
# IndexSet: GNN_MINIMAL (topology-only subset matches Gate C scope).
#
# Strict: papers100M is NEVER projected (benito_pc not authorized; celebi-only).
# Missing datasets are SKIPPED (logged) rather than abort, so dev machines
# without the big OGB copies can still run the cora_gnn smoke.
#
# Output:
#   CSV at /tmp/bench_leaffmt_<ts>.csv
#     Section 1: one row per (dataset, format) with summary measurements.
#     Section 2: one row per (dataset, format, index_name) with per-index bytes.
#   Markdown summary to stdout with size ratio per dataset and Gate C flag
#   (ogbn-products DELTA_VARINT / BITSET ratio <= 0.80 target).
#
# Usage: ./scripts/bench_leaffmt.sh
# Env:
#   MDB_BIN     Path to mdb binary (default: build/Release/bin/mdb)
#   DATASETS    Override dataset list (space-separated basenames)
#   FORMATS     Override format list (default: BITSET DELTA_VARINT)
#   BENCH_DIR   Output directory for CSV (default: /tmp)
#   PORT_BASE   Starting port (default: 19781; +1 per run)
#   SKIP_SCAN   Set to 1 to skip the read-scan phase (faster iteration)
#
# Spec reference:
#   docs/superpowers/specs/2026-04-25-delta-varint-leaf-design.md §7.3
#   docs/superpowers/plans/2026-04-25-delta-varint-leaf-plan.md  §T5.13

set -euo pipefail
export LC_ALL=C

MDB_BIN=${MDB_BIN:-build/Release/bin/mdb}
BENCH_DIR=${BENCH_DIR:-/tmp}
PORT_BASE=${PORT_BASE:-19781}
SKIP_SCAN=${SKIP_SCAN:-0}
BENCH_PROJ=${BENCH_PROJ:-bench_leaffmt_tmp}

# Dataset configuration: name  db_path  node_label  edge_type
# (Mirror of scripts/bench_indexset.sh — identical labels.)
declare -a DATASETS_DEFAULT=(
    "cora_gnn|data/dbs/gql/cora_gnn|Paper|CITES"
    "ogbn-arxiv|data/dbs/gql/ogbn-arxiv|Node|CONNECTS"
    "ogbn-products|data/dbs/gql/ogbn-products|Node|CONNECTS"
)

if [[ -n "${DATASETS:-}" ]]; then
    NEW_DATASETS=()
    for ds in $DATASETS; do
        if [[ "$ds" == "papers100M" || "$ds" == *papers100M* ]]; then
            echo "ERROR: papers100M is out of scope for bench_leaffmt.sh (benito_pc not authorized; celebi-only dataset). Aborting." >&2
            exit 3
        fi
        matched=0
        for entry in "${DATASETS_DEFAULT[@]}"; do
            if [[ "${entry%%|*}" == "$ds" ]]; then
                NEW_DATASETS+=("$entry")
                matched=1
                break
            fi
        done
        if [[ "$matched" -eq 0 ]]; then
            echo "ERROR: unknown dataset '$ds' (not in built-in config)." >&2
            exit 3
        fi
    done
    DATASETS_LIST=("${NEW_DATASETS[@]}")
else
    DATASETS_LIST=("${DATASETS_DEFAULT[@]}")
fi

FORMATS_LIST=(${FORMATS:-BITSET DELTA_VARINT})

# Strict guard: papers100M never runs here.
for entry in "${DATASETS_LIST[@]}"; do
    name="${entry%%|*}"
    if [[ "$name" == "papers100M" || "$name" == *papers100M* ]]; then
        echo "ERROR: papers100M is out of scope for bench_leaffmt.sh (benito_pc not authorized; celebi-only dataset). Aborting." >&2
        exit 3
    fi
done

# --- prereqs ----------------------------------------------------------------
if [[ ! -x "$MDB_BIN" ]]; then
    echo "ERROR: mdb binary not found at $MDB_BIN" >&2
    echo "       Build it with: cmake --build build/Release -j \$(nproc)" >&2
    exit 2
fi
if [[ ! -x /usr/bin/time ]]; then
    echo "ERROR: /usr/bin/time not found (needed for peak RSS via 'time -v')" >&2
    echo "       Install with: apt-get install time" >&2
    exit 2
fi
if ! command -v curl >/dev/null 2>&1; then
    echo "ERROR: curl not installed (required to issue queries)" >&2
    exit 2
fi

TS=$(date +%s)
CSV_OUT="$BENCH_DIR/bench_leaffmt_${TS}.csv"
CSV_PERIDX="$BENCH_DIR/bench_leaffmt_${TS}_peridx.csv"
echo "dataset,format,indexSet,num_nodes,num_edges,project_millis,wall_clock_sec,peak_rss_mb,total_leaf_bytes,scan_millis,scan_count" > "$CSV_OUT"
echo "dataset,format,index_name,leaf_bytes" > "$CSV_PERIDX"

SRV_PID=
SAMPLER_PID=
CURRENT_DB=
CURRENT_PROJ=

cleanup() {
    if [[ -n "${SAMPLER_PID:-}" ]] && kill -0 "$SAMPLER_PID" 2>/dev/null; then
        kill "$SAMPLER_PID" 2>/dev/null || true
        wait "$SAMPLER_PID" 2>/dev/null || true
    fi
    if [[ -n "${SRV_PID:-}" ]] && kill -0 "$SRV_PID" 2>/dev/null; then
        kill "$SRV_PID" 2>/dev/null || true
        for _ in 1 2 3 4 5 6 7 8 9 10; do
            kill -0 "$SRV_PID" 2>/dev/null || break
            sleep 0.3
        done
        kill -9 "$SRV_PID" 2>/dev/null || true
        wait "$SRV_PID" 2>/dev/null || true
    fi
    if [[ -n "${CURRENT_DB:-}" && -n "${CURRENT_PROJ:-}" ]]; then
        "$MDB_BIN" drop-projection "$CURRENT_DB" "$CURRENT_PROJ" >/dev/null 2>&1 || true
    fi
}
trap cleanup EXIT INT TERM

start_server() {
    local db="$1" port="$2" logfile="$3"
    MDB_PROJECTION_SERIAL_SCAN=1 \
    MDB_PROJECTION_SORTER=radix \
        "$MDB_BIN" server "$db" --port "$port" --timeout 3600 \
        > "$logfile" 2>&1 &
    SRV_PID=$!
    CURRENT_DB="$db"
    for _ in $(seq 1 60); do
        if curl -sSf -o /dev/null --data-binary "RETURN 1" -H "Accept: text/csv" \
                "http://127.0.0.1:$port/" 2>/dev/null; then
            return 0
        fi
        sleep 0.5
    done
    echo "ERROR: server on port $port did not come up in 30s" >&2
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

start_peak_rss_sampler() {
    local pid="$1" out="$2"
    : > "$out"
    (
        while kill -0 "$pid" 2>/dev/null; do
            if [[ -r /proc/$pid/status ]]; then
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

drop_proj_offline() {
    local db="$1" proj="$2"
    "$MDB_BIN" drop-projection "$db" "$proj" >/dev/null 2>&1 || true
}

# run_one: one (dataset, format) cell.
#   $1 dataset, $2 db, $3 node_label, $4 edge_type, $5 format, $6 port
run_one() {
    local dataset="$1" db="$2" node_label="$3" edge_type="$4" format="$5" port="$6"
    local fmt_lc
    fmt_lc=$(echo "$format" | tr '[:upper:]' '[:lower:]')
    local proj="${BENCH_PROJ}_${dataset//-/_}_${fmt_lc}"
    local logfile="$BENCH_DIR/bench_leaffmt_${dataset}_${fmt_lc}_${port}.log"
    local rssfile="$BENCH_DIR/bench_leaffmt_${dataset}_${fmt_lc}_${port}.rss"

    CURRENT_PROJ="$proj"
    drop_proj_offline "$db" "$proj"

    echo "--- [$dataset / $format] starting server on port $port ---"
    start_server "$db" "$port" "$logfile"
    start_peak_rss_sampler "$SRV_PID" "$rssfile"

    local query
    query="CALL graph_project('$proj', '$node_label', '$edge_type', {orientation: 'NATURAL', indexSet: 'GNN_MINIMAL', leafFormat: '$format'}) YIELD graphName, nodeCount, relCount, projectMillis RETURN *"

    local t0 t1 wall
    t0=$(date +%s.%N)
    local response
    response=$(curl -sS --max-time 3000 --data-binary "$query" \
        -H "Accept: text/csv" "http://127.0.0.1:$port/" || true)
    t1=$(date +%s.%N)
    wall=$(awk -v t0="$t0" -v t1="$t1" 'BEGIN { printf "%.3f", t1 - t0 }')

    stop_peak_rss_sampler
    local rss_kb=0
    if [[ -s "$rssfile" ]]; then
        rss_kb=$(tail -n 1 "$rssfile")
    fi
    local rss_mb
    rss_mb=$(awk -v kb="$rss_kb" 'BEGIN { printf "%.0f", kb/1024 }')

    # Parse CSV: header, then data row.
    #   "graphName","nodeCount","relCount","projectMillis"
    #   "bench_leaffmt_xxx",2708,5429,123
    local data_line nc ec pms
    data_line=$(echo "$response" | sed -n '2p' | tr -d '"')
    nc=$(echo "$data_line" | awk -F',' '{print $2}')
    ec=$(echo "$data_line" | awk -F',' '{print $3}')
    pms=$(echo "$data_line" | awk -F',' '{print $4}')
    [[ -z "$nc" ]] && nc=0
    [[ -z "$ec" ]] && ec=0
    [[ -z "$pms" ]] && pms=0

    local proj_dir="$db/projections/$proj"
    local total_leaf=0
    if [[ -d "$proj_dir" ]]; then
        # -bc emits the grand total on the last line.
        total_leaf=$(du -bc "$proj_dir"/*.leaf 2>/dev/null | tail -1 | cut -f1 || echo 0)
    fi
    [[ -z "$total_leaf" ]] && total_leaf=0

    # Per-index sizes into the secondary CSV.
    if [[ -d "$proj_dir" ]]; then
        for lf in "$proj_dir"/*.leaf; do
            [[ -e "$lf" ]] || continue
            local lname lbytes
            lname=$(basename "$lf" .leaf)
            lbytes=$(stat -c %s "$lf" 2>/dev/null || echo 0)
            echo "$dataset,$format,$lname,$lbytes" >> "$CSV_PERIDX"
        done
    fi

    # --- scan phase: full-range edge scan as read-side proxy ---
    local scan_ms=0 scan_count=0
    if [[ "$SKIP_SCAN" != "1" ]]; then
        local scan_query="USE $proj MATCH (n)-[e]->(m) RETURN count(e)"
        local s0 s1
        s0=$(date +%s.%N)
        local scan_resp
        scan_resp=$(curl -sS --max-time 1800 --data-binary "$scan_query" \
            -H "Accept: text/csv" "http://127.0.0.1:$port/" || true)
        s1=$(date +%s.%N)
        scan_ms=$(awk -v t0="$s0" -v t1="$s1" 'BEGIN { printf "%.1f", (t1 - t0) * 1000 }')
        # Response is CSV: header "count(e)" then a numeric row.
        scan_count=$(echo "$scan_resp" | sed -n '2p' | tr -d '"' | tr -d ' ')
        [[ -z "$scan_count" ]] && scan_count=0
    fi

    stop_server
    drop_proj_offline "$db" "$proj"
    CURRENT_PROJ=

    printf 'RESULT  %-14s %-14s wall=%ss  peakRSS=%sMB  leaf=%sB  N=%s E=%s  scan=%sms (count=%s)\n' \
        "$dataset" "$format" "$wall" "$rss_mb" "$total_leaf" "$nc" "$ec" "$scan_ms" "$scan_count"

    if [[ "$total_leaf" -eq 0 ]]; then
        echo "WARN  no .leaf files materialized for $dataset/$format — see $logfile" >&2
    fi

    echo "$dataset,$format,GNN_MINIMAL,$nc,$ec,$pms,$wall,$rss_mb,$total_leaf,$scan_ms,$scan_count" >> "$CSV_OUT"
}

# --- main -------------------------------------------------------------------
echo "================================================================"
echo "leafFormat bench — $(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "  git HEAD    : $(git -C "$(dirname "$0")/.." rev-parse --short HEAD 2>/dev/null || echo unknown)"
echo "  binary      : $MDB_BIN"
echo "  stack       : MDB_PROJECTION_SERIAL_SCAN=1  MDB_PROJECTION_SORTER=radix"
echo "  indexSet    : GNN_MINIMAL"
echo "  formats     : ${FORMATS_LIST[*]}"
echo "  scan phase  : $([[ "$SKIP_SCAN" == "1" ]] && echo SKIPPED || echo ENABLED)"
echo "  CSV summary : $CSV_OUT"
echo "  CSV perIdx  : $CSV_PERIDX"
echo "================================================================"

PORT="$PORT_BASE"
for entry in "${DATASETS_LIST[@]}"; do
    IFS='|' read -r name db nl et <<<"$entry"
    if [[ ! -d "$db" ]]; then
        echo "[skip] dataset $name not present at $db"
        continue
    fi
    for fmt in "${FORMATS_LIST[@]}"; do
        run_one "$name" "$db" "$nl" "$et" "$fmt" "$PORT"
        PORT=$((PORT + 1))
    done
done

# --- markdown summary -------------------------------------------------------
echo
echo "================================================================"
echo "Summary (DELTA_VARINT / BITSET size ratio per dataset)"
echo "================================================================"
echo
echo "| Dataset        | Format        | Leaf bytes      | Ratio    | Build s  | Scan ms   | Peak RSS MB |"
echo "|----------------|---------------|----------------:|---------:|---------:|----------:|------------:|"
awk -F',' '
NR == 1 { next }
{
    ds = $1; fmt = $2
    nodes[ds "|" fmt] = $4 + 0
    edges[ds "|" fmt] = $5 + 0
    pms[ds "|" fmt]   = $6 + 0
    wall[ds "|" fmt]  = $7 + 0
    rss[ds "|" fmt]   = $8 + 0
    bytes[ds "|" fmt] = $9 + 0
    scan[ds "|" fmt]  = $10 + 0
    ds_seen[ds] = 1
}
END {
    n = 0
    for (ds in ds_seen) order[++n] = ds
    for (i = 1; i <= n; i++)
        for (j = i + 1; j <= n; j++)
            if (order[i] > order[j]) { t = order[i]; order[i] = order[j]; order[j] = t }

    formats[1] = "BITSET"
    formats[2] = "DELTA_VARINT"

    for (i = 1; i <= n; i++) {
        ds = order[i]
        base_key = ds "|BITSET"
        base_bytes = bytes[base_key] + 0
        for (m = 1; m <= 2; m++) {
            fmt = formats[m]
            key = ds "|" fmt
            if (!(key in bytes)) continue
            if (fmt == "BITSET" || base_bytes == 0) {
                ratio_s = "1.00"
            } else {
                r = bytes[key] / base_bytes
                ratio_s = sprintf("%.3f", r)
            }
            # Human-readable size
            b = bytes[key]
            if (b >= 1024*1024*1024)
                hsz = sprintf("%.2f G", b / (1024*1024*1024))
            else if (b >= 1024*1024)
                hsz = sprintf("%.2f M", b / (1024*1024))
            else if (b >= 1024)
                hsz = sprintf("%.2f k", b / 1024)
            else
                hsz = sprintf("%d B", b)
            printf "| %-14s | %-13s | %15s | %8s | %8.2f | %9.1f | %11d |\n", \
                ds, fmt, hsz, ratio_s, wall[key], scan[key], rss[key]
        }
    }
}' "$CSV_OUT"

echo
echo "Gate C criterion: ogbn-products DELTA_VARINT / BITSET ratio <= 0.80"
awk -F',' '
NR == 1 { next }
$1 == "ogbn-products" { bytes[$2] = $9 + 0 }
END {
    if (("BITSET" in bytes) && ("DELTA_VARINT" in bytes) && bytes["BITSET"] > 0) {
        r = bytes["DELTA_VARINT"] / bytes["BITSET"]
        if (r <= 0.80) {
            printf "  PASS  ratio = %.4f (target <= 0.80)\n", r
        } else {
            printf "  FAIL  ratio = %.4f (target <= 0.80) — Spec #5 compression insufficient\n", r
        }
    } else {
        print "  SKIP  ogbn-products not measured in this run"
    }
}' "$CSV_OUT"

echo
echo "CSV summary : $CSV_OUT"
echo "CSV perIdx  : $CSV_PERIDX"
