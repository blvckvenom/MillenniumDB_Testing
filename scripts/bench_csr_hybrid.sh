#!/usr/bin/env bash
#
# Spec #8 T8.12 — graphStorage 4-mode benchmark harness.
#
# For each (dataset, leafFormat, graphStorage) triple in
#   {cora_gnn, ogbn-arxiv, ogbn-products}
#   x {BITSET, DELTA_VARINT}
#   x {BTREE, CSR_HYBRID}
# measure:
#   - wall clock of graph_project (curl round-trip)
#   - peak RSS of the mdb server process (VmHWM sampled from /proc)
#   - total .leaf disk bytes
#   - total topology_*.csr sidecar bytes (0 unless buildTopologySnapshot=true)
#   - total projection directory bytes (du -sb) — full footprint incl. .dir etc.
#   - wall clock of a full-range edge scan as read-side throughput proxy
#     (USE proj MATCH (n)-[e]->(m) RETURN count(e))
#     NOTE: on graphStorage=CSR_HYBRID, BPTLeafCSRWriter currently emits
#     flags=0 with no edge_id stream and the reader returns edge_id=0 for
#     every tuple.  count(e) under CSR_HYBRID is therefore affected by
#     this deferred Spec #8-B item; the CSV column `edge_id_supported`
#     flags this per row so downstream consumers do not silently compare
#     scan_edge_count across storage modes.
#   - gnn_offline_sample throughput: uniqueNodes / wall_clock seconds on
#     a fixed small seed set with fanout [15, 10] and a stable random seed.
#     This is the primary Gate D GNN metric and exercises
#     TopologyAccessor -> BptIter -> BPTLeafCSR.
#
# IndexSet: GNN_MINIMAL (topology-only subset matches Gate C/D scope).
# Sorter  : radix (default for this bench, matches bench_leaffmt.sh).
#
# Strict: papers100M is NEVER projected (benito_pc not authorized; celebi-only).
# Missing datasets are SKIPPED (logged) rather than abort, so dev machines
# without the big OGB copies can still run the cora_gnn smoke.
#
# Output:
#   CSV at /tmp/bench_csr_hybrid_<ts>.csv
#     One row per (dataset, format, storage) = up to 12 rows.
#   Markdown summary to stdout with per-dataset comparison table, including
#   the compression-stack row (BITSET+BTREE -> DELTA_VARINT+CSR_HYBRID).
#
# Usage: ./scripts/bench_csr_hybrid.sh
# Env:
#   MDB_BIN        Path to mdb binary (default: build/Release/bin/mdb)
#   DATASETS       Override dataset list (space-separated basenames)
#   FORMATS        Override format list (default: BITSET DELTA_VARINT)
#   STORAGE_MODES  Override storage list (default: BTREE CSR_HYBRID)
#   BENCH_DIR      Output directory for CSV (default: /tmp)
#   PORT_BASE      Starting port (default: 19860; +1 per run)
#   SKIP_SCAN      Set to 1 to skip the read-scan phase
#   SKIP_SAMPLE    Set to 1 to skip the gnn_offline_sample phase
#
# Gate D criteria (checked in summary):
#   1. CSR_HYBRID total storage <= 0.90 x DELTA_VARINT+BTREE (>= 10% reduction)
#   2. Sampling throughput(CSR_HYBRID) / throughput(DELTA_VARINT+BTREE) >= 0.80
#   3. BITSET+BTREE and DELTA_VARINT+BTREE byte totals byte-identical to T5.13
#      (non-CSR paths untouched) — reported, not auto-checked against a
#      reference number here.
#
# Spec reference:
#   docs/superpowers/specs/2026-04-25-csr-hybrid-design.md §3.12
#   docs/superpowers/plans/2026-04-25-csr-hybrid-plan.md   §T8.12

set -euo pipefail
export LC_ALL=C

MDB_BIN=${MDB_BIN:-build/Release/bin/mdb}
BENCH_DIR=${BENCH_DIR:-/tmp}
PORT_BASE=${PORT_BASE:-19860}
SKIP_SCAN=${SKIP_SCAN:-0}
SKIP_SAMPLE=${SKIP_SAMPLE:-0}
BENCH_PROJ=${BENCH_PROJ:-bench_csrhyb_tmp}
BENCH_SAMPLE=${BENCH_SAMPLE:-bench_csrhyb_sample}
FANOUTS=${FANOUTS:-"[15, 10]"}
NUM_SEEDS=${NUM_SEEDS:-512}

# Dataset configuration: name  db_path  node_label  edge_type
# (Mirror of scripts/bench_leaffmt.sh — identical labels.)
declare -a DATASETS_DEFAULT=(
    "cora_gnn|data/dbs/gql/cora_gnn|Paper|CITES"
    "ogbn-arxiv|data/dbs/gql/ogbn-arxiv|Node|CONNECTS"
    "ogbn-products|data/dbs/gql/ogbn-products|Node|CONNECTS"
)

if [[ -n "${DATASETS:-}" ]]; then
    NEW_DATASETS=()
    for ds in $DATASETS; do
        if [[ "$ds" == "papers100M" || "$ds" == *papers100M* ]]; then
            echo "ERROR: papers100M is out of scope for bench_csr_hybrid.sh (benito_pc not authorized; celebi-only dataset). Aborting." >&2
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
STORAGE_LIST=(${STORAGE_MODES:-BTREE CSR_HYBRID})

# Strict guard: papers100M never runs here.
for entry in "${DATASETS_LIST[@]}"; do
    name="${entry%%|*}"
    if [[ "$name" == "papers100M" || "$name" == *papers100M* ]]; then
        echo "ERROR: papers100M is out of scope for bench_csr_hybrid.sh (benito_pc not authorized; celebi-only dataset). Aborting." >&2
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
CSV_OUT="$BENCH_DIR/bench_csr_hybrid_${TS}.csv"
echo "dataset,format,storage,indexSet,num_nodes,num_edges_query,project_millis,wall_clock_sec,peak_rss_mb,total_leaf_bytes,total_csr_bytes,total_proj_bytes,scan_millis,scan_edge_count,edge_id_supported,sample_seeds_per_sec" > "$CSV_OUT"

SRV_PID=
SAMPLER_PID=
CURRENT_DB=
CURRENT_PROJ=
CURRENT_SAMPLE=

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
    if [[ -n "${CURRENT_DB:-}" && -n "${CURRENT_SAMPLE:-}" ]]; then
        rm -rf "$CURRENT_DB/samples/$CURRENT_SAMPLE" 2>/dev/null || true
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

# Samples live at <db>/samples/<name>; no CLI drop, so remove offline.
drop_sample_offline() {
    local db="$1" sample="$2"
    rm -rf "$db/samples/$sample" 2>/dev/null || true
}

# run_one: one (dataset, format, storage) cell.
#   $1 dataset, $2 db, $3 node_label, $4 edge_type,
#   $5 format, $6 storage, $7 port
run_one() {
    local dataset="$1" db="$2" node_label="$3" edge_type="$4"
    local format="$5" storage="$6" port="$7"
    local fmt_lc stor_lc
    fmt_lc=$(echo "$format"  | tr '[:upper:]' '[:lower:]')
    stor_lc=$(echo "$storage" | tr '[:upper:]' '[:lower:]')
    # Shorten 'csr_hybrid' to 'csr' so projection names stay comfortably
    # under common filesystem/limit boundaries.
    case "$stor_lc" in
        csr_hybrid) stor_lc="csr" ;;
        btree)      stor_lc="btree" ;;
    esac
    local proj="${BENCH_PROJ}_${dataset//-/_}_${fmt_lc}_${stor_lc}"
    local sample="${BENCH_SAMPLE}_${dataset//-/_}_${fmt_lc}_${stor_lc}"
    local logfile="$BENCH_DIR/bench_csrhyb_${dataset}_${fmt_lc}_${stor_lc}_${port}.log"
    local rssfile="$BENCH_DIR/bench_csrhyb_${dataset}_${fmt_lc}_${stor_lc}_${port}.rss"

    CURRENT_PROJ="$proj"
    CURRENT_SAMPLE="$sample"
    drop_proj_offline "$db" "$proj"
    drop_sample_offline "$db" "$sample"

    echo "--- [$dataset / $format x $storage] starting server on port $port ---"
    start_server "$db" "$port" "$logfile"
    start_peak_rss_sampler "$SRV_PID" "$rssfile"

    local query
    query="CALL graph_project('$proj', '$node_label', '$edge_type', {orientation: 'NATURAL', indexSet: 'GNN_MINIMAL', leafFormat: '$format', graphStorage: '$storage'}) YIELD graphName, nodeCount, relCount, projectMillis RETURN *"

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
    #   "bench_csrhyb_xxx",2708,5429,123
    local data_line nc ec pms
    data_line=$(echo "$response" | sed -n '2p' | tr -d '"')
    nc=$(echo "$data_line" | awk -F',' '{print $2}')
    ec=$(echo "$data_line" | awk -F',' '{print $3}')
    pms=$(echo "$data_line" | awk -F',' '{print $4}')
    [[ -z "$nc" ]] && nc=0
    [[ -z "$ec" ]] && ec=0
    [[ -z "$pms" ]] && pms=0

    local proj_dir="$db/projections/$proj"
    local total_leaf=0 total_csr=0 total_proj=0
    if [[ -d "$proj_dir" ]]; then
        total_leaf=$(du -bc "$proj_dir"/*.leaf 2>/dev/null | tail -1 | cut -f1 || echo 0)
        [[ -z "$total_leaf" ]] && total_leaf=0
        if ls "$proj_dir"/topology_*.csr >/dev/null 2>&1; then
            total_csr=$(du -bc "$proj_dir"/topology_*.csr 2>/dev/null | tail -1 | cut -f1 || echo 0)
        fi
        [[ -z "$total_csr" ]] && total_csr=0
        total_proj=$(du -sb "$proj_dir" 2>/dev/null | cut -f1 || echo 0)
        [[ -z "$total_proj" ]] && total_proj=0
    fi

    # --- scan phase: full-range edge scan as read-side proxy ---
    local scan_ms=0 scan_count=0
    local edge_id_supported="yes"
    if [[ "$storage" == "CSR_HYBRID" ]]; then
        # T8.5 deferred: BPTLeafCSRWriter emits flags=0 and no edge_id
        # stream, so the reader returns edge_id=0 for every tuple and
        # count(e) is affected.  Record the raw number but mark the
        # column so downstream consumers don't cross-compare.
        edge_id_supported="deferred"
    fi
    if [[ "$SKIP_SCAN" != "1" ]]; then
        local scan_query="USE $proj MATCH (n)-[e]->(m) RETURN count(e)"
        local s0 s1
        s0=$(date +%s.%N)
        local scan_resp
        scan_resp=$(curl -sS --max-time 1800 --data-binary "$scan_query" \
            -H "Accept: text/csv" "http://127.0.0.1:$port/" || true)
        s1=$(date +%s.%N)
        scan_ms=$(awk -v t0="$s0" -v t1="$s1" 'BEGIN { printf "%.1f", (t1 - t0) * 1000 }')
        scan_count=$(echo "$scan_resp" | sed -n '2p' | tr -d '"' | tr -d ' ')
        [[ -z "$scan_count" ]] && scan_count=0
    fi

    # --- sample phase: gnn_offline_sample throughput (seeds/sec proxy) ---
    local seeds_per_sec=0
    if [[ "$SKIP_SAMPLE" != "1" ]]; then
        drop_sample_offline "$db" "$sample"
        local sample_query
        sample_query="CALL gnn_offline_sample('$proj', '$sample', $FANOUTS, {batchSize: $NUM_SEEDS, randomSeed: 42, orientation: 'UNDIRECTED'}) YIELD sampleName, totalBatches, trainBatches, validationBatches, testBatches, uniqueNodes, computeMillis RETURN *"
        local sa0 sa1 sample_wall
        sa0=$(date +%s.%N)
        local sample_resp
        sample_resp=$(curl -sS --max-time 3000 --data-binary "$sample_query" \
            -H "Accept: text/csv" "http://127.0.0.1:$port/" || true)
        sa1=$(date +%s.%N)
        sample_wall=$(awk -v t0="$sa0" -v t1="$sa1" 'BEGIN { printf "%.3f", t1 - t0 }')

        local sdata unique_nodes=0
        sdata=$(echo "$sample_resp" | sed -n '2p' | tr -d '"')
        unique_nodes=$(echo "$sdata" | awk -F',' '{print $6}' | tr -d ' ')
        [[ -z "$unique_nodes" ]] && unique_nodes=0

        if [[ -n "$sample_resp" ]] && echo "$sample_resp" | head -1 | grep -q sampleName; then
            seeds_per_sec=$(awk -v u="$unique_nodes" -v w="$sample_wall" \
                'BEGIN { if (w > 0) printf "%.0f", u / w; else print 0 }')
        else
            echo "WARN  sampling failed on $dataset/$format/$storage — see $logfile" >&2
            seeds_per_sec=0
        fi
    fi

    stop_server
    drop_proj_offline "$db" "$proj"
    drop_sample_offline "$db" "$sample"
    CURRENT_PROJ=
    CURRENT_SAMPLE=

    printf 'RESULT  %-14s %-14s %-11s wall=%ss  peakRSS=%sMB  leaf=%sB  csr=%sB  N=%s E=%s  scan=%sms  sample=%s s/s\n' \
        "$dataset" "$format" "$storage" "$wall" "$rss_mb" "$total_leaf" "$total_csr" "$nc" "$ec" "$scan_ms" "$seeds_per_sec"

    if [[ "$total_leaf" -eq 0 ]]; then
        echo "WARN  no .leaf files materialized for $dataset/$format/$storage — see $logfile" >&2
    fi

    echo "$dataset,$format,$storage,GNN_MINIMAL,$nc,$ec,$pms,$wall,$rss_mb,$total_leaf,$total_csr,$total_proj,$scan_ms,$scan_count,$edge_id_supported,$seeds_per_sec" >> "$CSV_OUT"
}

# --- main -------------------------------------------------------------------
echo "================================================================"
echo "CSR_HYBRID 4-mode bench — $(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "  git HEAD    : $(git -C "$(dirname "$0")/.." rev-parse --short HEAD 2>/dev/null || echo unknown)"
echo "  binary      : $MDB_BIN"
echo "  stack       : MDB_PROJECTION_SERIAL_SCAN=1  MDB_PROJECTION_SORTER=radix"
echo "  indexSet    : GNN_MINIMAL"
echo "  formats     : ${FORMATS_LIST[*]}"
echo "  storage     : ${STORAGE_LIST[*]}"
echo "  fanouts     : $FANOUTS    batchSize=$NUM_SEEDS"
echo "  scan phase  : $([[ "$SKIP_SCAN" == "1" ]] && echo SKIPPED || echo ENABLED)"
echo "  sample phase: $([[ "$SKIP_SAMPLE" == "1" ]] && echo SKIPPED || echo ENABLED)"
echo "  CSV         : $CSV_OUT"
echo "================================================================"

PORT="$PORT_BASE"
for entry in "${DATASETS_LIST[@]}"; do
    IFS='|' read -r name db nl et <<<"$entry"
    if [[ ! -d "$db" ]]; then
        echo "[skip] dataset $name not present at $db"
        continue
    fi
    for fmt in "${FORMATS_LIST[@]}"; do
        for stor in "${STORAGE_LIST[@]}"; do
            run_one "$name" "$db" "$nl" "$et" "$fmt" "$stor" "$PORT"
            PORT=$((PORT + 1))
        done
    done
done

# --- markdown summary -------------------------------------------------------
echo
echo "================================================================"
echo "Summary — per-dataset 4-mode comparison"
echo "================================================================"

awk -F',' '
function hsize(b,   s) {
    if (b >= 1024*1024*1024) s = sprintf("%.2f G", b / (1024*1024*1024))
    else if (b >= 1024*1024) s = sprintf("%.2f M", b / (1024*1024))
    else if (b >= 1024)      s = sprintf("%.2f k", b / 1024)
    else                     s = sprintf("%d B",   b)
    return s
}
NR == 1 { next }
{
    ds = $1; fmt = $2; stor = $3
    key = ds "|" fmt "|" stor
    wall[key] = $8 + 0
    rss[key]  = $9 + 0
    lbytes[key] = $10 + 0
    cbytes[key] = $11 + 0
    pbytes[key] = $12 + 0
    scan[key]   = $13 + 0
    sseeds[key] = $16 + 0
    ds_seen[ds] = 1
    if (!(ds "|" fmt "|" stor in row_seen)) {
        row_seen[ds "|" fmt "|" stor] = 1
    }
}
END {
    # Stable ordering: alphabetize datasets.
    n = 0
    for (ds in ds_seen) order[++n] = ds
    for (i = 1; i <= n; i++)
        for (j = i + 1; j <= n; j++)
            if (order[i] > order[j]) { t = order[i]; order[i] = order[j]; order[j] = t }

    # Mode display order: BITSET+BTREE, DELTA_VARINT+BTREE,
    #                      BITSET+CSR_HYBRID, DELTA_VARINT+CSR_HYBRID.
    modes[1] = "BITSET|BTREE"
    modes[2] = "DELTA_VARINT|BTREE"
    modes[3] = "BITSET|CSR_HYBRID"
    modes[4] = "DELTA_VARINT|CSR_HYBRID"
    labels[1] = "BITSET + BTREE"
    labels[2] = "DELTA_VARINT + BTREE"
    labels[3] = "BITSET + CSR_HYBRID"
    labels[4] = "DELTA_VARINT + CSR_HYBRID"

    for (i = 1; i <= n; i++) {
        ds = order[i]
        printf "\n"
        printf "| %-14s | leaf bytes | csr bytes | total proj | build s  | scan ms | sample s/sec |\n", ds
        printf "|----------------|-----------:|----------:|-----------:|---------:|--------:|-------------:|\n"
        for (m = 1; m <= 4; m++) {
            split(modes[m], parts, "|")
            fmt = parts[1]; stor = parts[2]
            key = ds "|" fmt "|" stor
            if (!(key in lbytes)) {
                printf "| %-14s | (missing)  |           |            |          |         |              |\n", labels[m]
                continue
            }
            lb = lbytes[key]; cb = cbytes[key]; pb = pbytes[key]
            printf "| %-14s | %10s | %9s | %10s | %8.2f | %7.1f | %12d |\n", \
                labels[m], hsize(lb), hsize(cb), hsize(pb), wall[key], scan[key], sseeds[key]
        }
        # Compression stack row: BITSET+BTREE -> DELTA_VARINT+CSR_HYBRID.
        base_key = ds "|BITSET|BTREE"
        stack_key = ds "|DELTA_VARINT|CSR_HYBRID"
        if ((base_key in pbytes) && (stack_key in pbytes) && pbytes[base_key] > 0) {
            r = pbytes[stack_key] / pbytes[base_key]
            printf "|   stack red.   | %10s | %9s | %s / %s = %.3f (%.1f%% reduction)                        |\n", \
                hsize(lbytes[stack_key]) " vs " hsize(lbytes[base_key]), \
                hsize(cbytes[stack_key]), \
                hsize(pbytes[stack_key]), hsize(pbytes[base_key]), r, (1 - r) * 100
        }
    }
}' "$CSV_OUT"

echo
echo "================================================================"
echo "Gate D criteria"
echo "================================================================"

awk -F',' '
NR == 1 { next }
{
    ds = $1; fmt = $2; stor = $3
    key = ds "|" fmt "|" stor
    lbytes[key] = $10 + 0
    cbytes[key] = $11 + 0
    pbytes[key] = $12 + 0
    sseeds[key] = $16 + 0
    ds_seen[ds] = 1
}
END {
    n = 0
    for (ds in ds_seen) order[++n] = ds
    for (i = 1; i <= n; i++)
        for (j = i + 1; j <= n; j++)
            if (order[i] > order[j]) { t = order[i]; order[i] = order[j]; order[j] = t }

    # Criterion 1: CSR_HYBRID best-case total storage vs DELTA_VARINT+BTREE
    # baseline.  Best-case = DELTA_VARINT + CSR_HYBRID (both opts stacked).
    print ""
    print "Criterion 1  total_proj(DELTA_VARINT+CSR_HYBRID) <= 0.90 x total_proj(DELTA_VARINT+BTREE)"
    for (i = 1; i <= n; i++) {
        ds = order[i]
        b = pbytes[ds "|DELTA_VARINT|BTREE"]
        c = pbytes[ds "|DELTA_VARINT|CSR_HYBRID"]
        if (b == 0 || c == 0) {
            printf "  SKIP  %s (missing data)\n", ds
            continue
        }
        r = c / b
        status = (r <= 0.90) ? "PASS" : "FAIL"
        printf "  %s  %-14s ratio = %.4f  (leaf+csr+dir: %d vs %d B)\n", status, ds, r, c, b
    }

    # Criterion 2: sampling throughput — CSR_HYBRID >= 0.80 x DELTA_VARINT+BTREE
    print ""
    print "Criterion 2  sample_seeds_per_sec(*+CSR_HYBRID) >= 0.80 x (DELTA_VARINT+BTREE)"
    for (i = 1; i <= n; i++) {
        ds = order[i]
        baseline_thru = sseeds[ds "|DELTA_VARINT|BTREE"]
        for (k = 1; k <= 2; k++) {
            ft = (k == 1) ? "BITSET" : "DELTA_VARINT"
            c_thru = sseeds[ds "|" ft "|CSR_HYBRID"]
            if (baseline_thru == 0 || c_thru == 0) {
                printf "  SKIP  %-14s %s+CSR_HYBRID (missing data)\n", ds, ft
                continue
            }
            r = c_thru / baseline_thru
            status = (r >= 0.80) ? "PASS" : "FAIL"
            printf "  %s  %-14s %s+CSR_HYBRID / DELTA_VARINT+BTREE = %.3f (%d / %d seeds/s)\n", \
                status, ds, ft, r, c_thru, baseline_thru
        }
    }

    # Criterion 3: Non-CSR paths untouched — reported for review, not auto-checked.
    print ""
    print "Criterion 3  BTREE byte totals (for cross-check against T5.13 baseline)"
    printf "  %-14s %-13s %-11s %15s\n", "dataset", "format", "storage", "leaf_bytes"
    for (i = 1; i <= n; i++) {
        ds = order[i]
        for (k = 1; k <= 2; k++) {
            ft = (k == 1) ? "BITSET" : "DELTA_VARINT"
            key = ds "|" ft "|BTREE"
            if (!(key in lbytes)) continue
            printf "  %-14s %-13s %-11s %15d\n", ds, ft, "BTREE", lbytes[key]
        }
    }
}' "$CSV_OUT"

echo
echo "CSV : $CSV_OUT"
