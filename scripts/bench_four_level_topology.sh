#!/usr/bin/env bash
#
# Spec #13 T13.15 — Four-Level Topology Store benchmark harness.
#
# For each (dataset, mode) pair in
#   {cora_gnn, ogbn-arxiv} × {spec13, spec11, caminoD, bpt}
# measure:
#   - sample wall clock of CALL gnn_offline_sample with the mode's GQL config
#   - peak RSS of the mdb server process (VmHWM sampled from /proc)
#   - L1 / L2 / L3 / L4 node counts parsed from the [FourLevelTopologyStore]
#     stderr line emitted at end of build() (only meaningful for spec13;
#     emitted as 0 for the other three modes since they don't construct it)
#   - L1 + L2 RAM bytes (parsed from "ram_used=" segment of the same line)
#   - num_seeds (uniqueNodes from YIELD) and split totals
#
# Modes (all share the same base GQL invocation, only the trailing options
# map differs — strict mutual exclusion of cache backends):
#
#   spec13    → useFourLevelTopologyStore:true,  useAdjacencyCache:true,
#               useL3MmapSidecar:true   (L1+L2+L3+L4 dispatch — auto behavior)
#   spec11    → useFourLevelTopologyStore:false, useAdjacencyCache:true
#               (Spec #11 monolithic RAM hash — no Four-Level Store)
#   caminoD   → useFourLevelTopologyStore:false, useAdjacencyCache:false
#               (Spec #4-B sidecar mmap fallback; projection MUST have
#                buildTopologySnapshot:true)
#   bpt       → useFourLevelTopologyStore:false, useAdjacencyCache:false
#               (B+Tree direct; projection WITHOUT sidecar so the cold path
#                is exercised)
#
# To isolate sample wall-clock from projection-build noise, two projections
# are built per dataset and reused across modes:
#   <BENCH_PROJ>_with_snap → buildTopologySnapshot:true
#                            (used by spec13, spec11, caminoD)
#   <BENCH_PROJ>_no_snap   → no sidecar
#                            (used by bpt)
# Both projections use indexSet:'GNN_MINIMAL' (the topology-only subset).
#
# Stack: MDB_PROJECTION_SORTER=radix.
#
# Strict: papers100M is NEVER projected here (celebi-only dataset, see
# docs/research/2026-04-25-spec13-papers100m-procedure.md for the manual
# celebi procedure that complements this bench at papers scale).
# ogbn-products is OPT-IN via DATASETS=ogbn-products because spec11 risks
# OOM and the bench is meant to be routine.
#
# Output:
#   CSV at /tmp/bench_four_level_topology_<ts>.csv with columns
#     dataset,mode,sample_wall_sec,peak_rss_mb,
#     l1_node_count,l2_node_count,l3_node_count,l4_node_count,
#     l1_l2_total_ram_mb,
#     num_seeds,total_batches,train_batches,validation_batches,test_batches
#   plus pretty-printed summary table to stdout (per-dataset, sorted by
#   sample_wall_sec ascending → fastest mode at the top).
#
# Usage: ./scripts/bench_four_level_topology.sh
# Smoke: DATASETS=cora_gnn ./scripts/bench_four_level_topology.sh
#
# Env:
#   MDB             Path to mdb binary (default: build/Release/bin/mdb)
#   PORT_BASE       Starting port (default: 19990; +1 per (dataset,mode))
#   DATASETS        Override dataset list (space-separated basenames under
#                   data/dbs/gql; allow opt-in for ogbn-products)
#   MODES           Override mode list (space-separated subset of:
#                   spec13 spec11 caminoD bpt)
#   NUM_SEEDS       batchSize passed to gnn_offline_sample (default: 512)
#   FANOUTS         Fanout list passed to gnn_offline_sample (default: "[15, 10]")

set -euo pipefail
export LC_ALL=C

MDB=${MDB:-build/Release/bin/mdb}
PORT_BASE=${PORT_BASE:-19990}
BENCH_PROJ=${BENCH_PROJ:-bench_four_level_tmp}
BENCH_SAMPLE=${BENCH_SAMPLE:-bench_four_level_sample}
NUM_SEEDS=${NUM_SEEDS:-512}
FANOUTS=${FANOUTS:-"[15, 10]"}

# Dataset configuration: name | db_path | node_label | edge_type
# ogbn-products is intentionally absent from the default — opt in via DATASETS.
declare -a DATASETS_DEFAULT=(
    "cora_gnn|data/dbs/gql/cora_gnn|Paper|CITES"
    "ogbn-arxiv|data/dbs/gql/ogbn-arxiv|Node|CONNECTS"
)

declare -a DATASETS_KNOWN=(
    "cora_gnn|data/dbs/gql/cora_gnn|Paper|CITES"
    "ogbn-arxiv|data/dbs/gql/ogbn-arxiv|Node|CONNECTS"
    "ogbn-products|data/dbs/gql/ogbn-products|Node|CONNECTS"
)

if [[ -n "${DATASETS:-}" ]]; then
    NEW_DATASETS=()
    for ds in $DATASETS; do
        if [[ "$ds" == "papers100M" || "$ds" == *papers100M* ]]; then
            echo "ERROR: papers100M is out of scope for bench_four_level_topology.sh" >&2
            echo "       (celebi-only dataset; see docs/research/2026-04-25-spec13-papers100m-procedure.md)" >&2
            exit 3
        fi
        for entry in "${DATASETS_KNOWN[@]}"; do
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

# Strict guard: papers100M is out of scope here.
for entry in "${DATASETS_LIST[@]}"; do
    name="${entry%%|*}"
    if [[ "$name" == "papers100M" || "$name" == *papers100M* ]]; then
        echo "ERROR: papers100M is out of scope for bench_four_level_topology.sh" >&2
        echo "       (celebi-only dataset; see docs/research/2026-04-25-spec13-papers100m-procedure.md)" >&2
        exit 3
    fi
done

MODES_LIST=(${MODES:-spec13 spec11 caminoD bpt})

TS=$(date +%s)
CSV_OUT="/tmp/bench_four_level_topology_${TS}.csv"
echo "dataset,mode,sample_wall_sec,peak_rss_mb,l1_node_count,l2_node_count,l3_node_count,l4_node_count,l1_l2_total_ram_mb,num_seeds,total_batches,train_batches,validation_batches,test_batches" > "$CSV_OUT"

SRV_PID=
SAMPLER_PID=
CURRENT_DB=
CURRENT_PROJ_WITH=
CURRENT_PROJ_NO=
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
    # Best-effort cleanup of in-flight sample + projections.
    if [[ -n "${CURRENT_DB:-}" && -n "${CURRENT_SAMPLE:-}" ]]; then
        rm -rf "$CURRENT_DB/samples/$CURRENT_SAMPLE" 2>/dev/null || true
    fi
    if [[ -n "${CURRENT_DB:-}" && -n "${CURRENT_PROJ_WITH:-}" ]]; then
        "$MDB" drop-projection "$CURRENT_DB" "$CURRENT_PROJ_WITH" >/dev/null 2>&1 || true
    fi
    if [[ -n "${CURRENT_DB:-}" && -n "${CURRENT_PROJ_NO:-}" ]]; then
        "$MDB" drop-projection "$CURRENT_DB" "$CURRENT_PROJ_NO" >/dev/null 2>&1 || true
    fi
}
trap cleanup EXIT INT TERM

start_server() {
    local db="$1" port="$2" logfile="$3"
    MDB_PROJECTION_SORTER=radix \
        "$MDB" server "$db" --port "$port" --timeout 7200 \
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
    echo "ERROR: server on port $port did not come up in 30 s" >&2
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

# Peak RSS sampler — writes VmHWM samples (kB) to $2 every second until
# the tracked pid exits. VmHWM is the kernel's running high-water mark,
# so the final line is the peak across the entire observation window.
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

rss_file_to_mb() {
    local rssfile="$1"
    local rss_kb=0
    if [[ -s "$rssfile" ]]; then
        rss_kb=$(tail -n 1 "$rssfile")
    fi
    LC_NUMERIC=C awk -v kb="$rss_kb" 'BEGIN { printf "%.0f", kb/1024 }'
}

drop_proj_offline() {
    local db="$1" proj="$2"
    "$MDB" drop-projection "$db" "$proj" >/dev/null 2>&1 || true
}

drop_sample_offline() {
    local db="$1" sample="$2"
    rm -rf "$db/samples/$sample" 2>/dev/null || true
}

# Build a projection on a running server. $1=port $2=name $3=node_label
# $4=edge_type $5=snapshot(true|false). Echoes nodeCount,relCount on stdout.
build_projection() {
    local port="$1" proj="$2" node_label="$3" edge_type="$4" snap="$5"
    local q="CALL graph_project('$proj', '$node_label', '$edge_type', {orientation: 'NATURAL', indexSet: 'GNN_MINIMAL', buildTopologySnapshot: $snap}) YIELD graphName, nodeCount, relationshipCount RETURN *"
    local resp
    resp=$(curl -sS --max-time 7200 --data-binary "$q" \
        -H "Accept: text/csv" "http://127.0.0.1:$port/" || true)
    local data_line nc ec
    data_line=$(echo "$resp" | sed -n '2p' | tr -d '"')
    nc=$(echo "$data_line" | awk -F',' '{print $2}')
    ec=$(echo "$data_line" | awk -F',' '{print $3}')
    [[ -z "$nc" ]] && nc=0
    [[ -z "$ec" ]] && ec=0
    echo "${nc},${ec}"
}

# Compose the GQL options map for a given mode. Echoes the option-map
# string (without surrounding {}) — the caller wraps it.
mode_options() {
    local mode="$1"
    case "$mode" in
        spec13)
            echo "batchSize: $NUM_SEEDS, randomSeed: 42, orientation: 'UNDIRECTED', useFourLevelTopologyStore: true, useAdjacencyCache: true, useL3MmapSidecar: true"
            ;;
        spec11)
            echo "batchSize: $NUM_SEEDS, randomSeed: 42, orientation: 'UNDIRECTED', useFourLevelTopologyStore: false, useAdjacencyCache: true"
            ;;
        caminoD)
            echo "batchSize: $NUM_SEEDS, randomSeed: 42, orientation: 'UNDIRECTED', useFourLevelTopologyStore: false, useAdjacencyCache: false"
            ;;
        bpt)
            echo "batchSize: $NUM_SEEDS, randomSeed: 42, orientation: 'UNDIRECTED', useFourLevelTopologyStore: false, useAdjacencyCache: false"
            ;;
        *)
            echo "ERROR: unknown mode '$mode'" >&2
            return 1
            ;;
    esac
}

# Pick the projection name a given mode samples from.
mode_projection_name() {
    local base="$1" mode="$2"
    case "$mode" in
        spec13|spec11|caminoD)  echo "${base}_with_snap" ;;
        bpt)                    echo "${base}_no_snap" ;;
        *)                      echo "ERROR" ;;
    esac
}

# Parse the "FourLevelTopologyStore: built — l1_fwd=… l2_fwd=… l1_rev=…
# l2_rev=… l3_fwd=… l3_rev=… ram_used=… bytes" line from the server log.
# Echoes "<l1_n>,<l2_n>,<l3_n>,<ram_mb>" — emits 0 for everything when the
# line is missing (modes other than spec13).
parse_four_level_log() {
    local logfile="$1"
    local line
    line=$(grep -E '^FourLevelTopologyStore: built' "$logfile" | tail -n 1 || true)
    if [[ -z "$line" ]]; then
        echo "0,0,0,0"
        return
    fi
    # Extract l1_fwd, l2_fwd, l1_rev, l2_rev, l3_fwd, l3_rev, ram_used.
    local l1f l2f l1r l2r l3f l3r ram_b
    l1f=$(echo "$line" | sed -n 's/.*l1_fwd=\([0-9]*\).*/\1/p')
    l2f=$(echo "$line" | sed -n 's/.*l2_fwd=\([0-9]*\).*/\1/p')
    l1r=$(echo "$line" | sed -n 's/.*l1_rev=\([0-9]*\).*/\1/p')
    l2r=$(echo "$line" | sed -n 's/.*l2_rev=\([0-9]*\).*/\1/p')
    l3f=$(echo "$line" | sed -n 's/.*l3_fwd=\([0-9]*\).*/\1/p')
    l3r=$(echo "$line" | sed -n 's/.*l3_rev=\([0-9]*\).*/\1/p')
    ram_b=$(echo "$line" | sed -n 's/.*ram_used=\([0-9]*\).*/\1/p')
    [[ -z "$l1f" ]] && l1f=0; [[ -z "$l2f" ]] && l2f=0
    [[ -z "$l1r" ]] && l1r=0; [[ -z "$l2r" ]] && l2r=0
    [[ -z "$l3f" ]] && l3f=0; [[ -z "$l3r" ]] && l3r=0
    [[ -z "$ram_b" ]] && ram_b=0
    local l1_total=$((l1f + l1r))
    local l2_total=$((l2f + l2r))
    local l3_total=$((l3f + l3r))
    local ram_mb
    ram_mb=$(LC_NUMERIC=C awk -v b="$ram_b" 'BEGIN { printf "%.0f", b/1048576 }')
    echo "${l1_total},${l2_total},${l3_total},${ram_mb}"
}

# Run one (dataset, mode) sample. Assumes server is running on $port and
# both projections (with_snap, no_snap) already exist in $db.
run_one_sample() {
    local dataset="$1" db="$2" mode="$3" port="$4" logfile="$5"
    local proj sample rss_sample
    proj=$(mode_projection_name "$BENCH_PROJ" "$mode")
    sample="${BENCH_SAMPLE}_${mode}"
    rss_sample="/tmp/bench_four_level_${dataset}_${mode}_${port}.sample.rss"
    CURRENT_SAMPLE="$sample"
    drop_sample_offline "$db" "$sample"

    # Tier-distribution + ram lines emitted to stderr live in the server
    # logfile. Capture the file size BEFORE the call so we can scope the
    # parse to lines emitted during this sample only.
    local log_size_before=0
    [[ -f "$logfile" ]] && log_size_before=$(stat -c '%s' "$logfile" 2>/dev/null || echo 0)

    local opts
    opts=$(mode_options "$mode")
    local sample_query="CALL gnn_offline_sample('$proj', '$sample', $FANOUTS, {$opts}) YIELD sampleName, totalBatches, trainBatches, validationBatches, testBatches, uniqueNodes, computeMillis RETURN *"

    start_peak_rss_sampler "$SRV_PID" "$rss_sample"
    local ts0 ts1 sample_resp
    ts0=$(date +%s.%N)
    sample_resp=$(curl -sS --max-time 7200 --data-binary "$sample_query" \
        -H "Accept: text/csv" "http://127.0.0.1:$port/" || true)
    ts1=$(date +%s.%N)
    stop_peak_rss_sampler

    local sample_wall sample_rss_mb
    sample_wall=$(LC_NUMERIC=C awk -v t0="$ts0" -v t1="$ts1" 'BEGIN { printf "%.3f", t1 - t0 }')
    sample_rss_mb=$(rss_file_to_mb "$rss_sample")

    # Parse YIELD CSV — header on line 1, single data row on line 2.
    local sdata total_batches train_b val_b test_b unique_nodes
    sdata=$(echo "$sample_resp" | sed -n '2p' | tr -d '"')
    total_batches=$(echo "$sdata" | awk -F',' '{print $2}')
    train_b=$(echo "$sdata"       | awk -F',' '{print $3}')
    val_b=$(echo "$sdata"         | awk -F',' '{print $4}')
    test_b=$(echo "$sdata"        | awk -F',' '{print $5}')
    unique_nodes=$(echo "$sdata"  | awk -F',' '{print $6}')
    [[ -z "$total_batches" ]] && total_batches=0
    [[ -z "$train_b" ]] && train_b=0
    [[ -z "$val_b" ]] && val_b=0
    [[ -z "$test_b" ]] && test_b=0
    [[ -z "$unique_nodes" ]] && unique_nodes=0

    if [[ -z "$sample_resp" ]] || ! echo "$sample_resp" | head -1 | grep -q sampleName; then
        echo "WARN  sampling failed on $dataset mode=$mode — see $logfile" >&2
    fi

    # Slice the server log to only the bytes emitted during this run, then
    # parse the FourLevelTopologyStore tier-distribution line if present.
    local sliced_log="/tmp/bench_four_level_${dataset}_${mode}_${port}.sliced.log"
    if [[ -f "$logfile" ]]; then
        tail -c "+$((log_size_before + 1))" "$logfile" > "$sliced_log" 2>/dev/null || true
    else
        : > "$sliced_log"
    fi

    local parsed l1_n l2_n l3_n l4_n ram_mb
    parsed=$(parse_four_level_log "$sliced_log")
    l1_n=$(echo "$parsed" | awk -F',' '{print $1}')
    l2_n=$(echo "$parsed" | awk -F',' '{print $2}')
    l3_n=$(echo "$parsed" | awk -F',' '{print $3}')
    ram_mb=$(echo "$parsed" | awk -F',' '{print $4}')

    # L4 = none (modes other than spec13 don't construct the dispatcher;
    # for spec13 the current implementation accounts L4 separately and we
    # surface 0 here unless future versions emit a dedicated counter).
    l4_n=0

    drop_sample_offline "$db" "$sample"
    CURRENT_SAMPLE=

    printf 'RESULT  %-14s mode=%-9s sample=%ss  rss=%sMB  l1=%s l2=%s l3=%s l4=%s ram_mb=%s  uniq=%s  batches=%s\n' \
        "$dataset" "$mode" "$sample_wall" "$sample_rss_mb" \
        "$l1_n" "$l2_n" "$l3_n" "$l4_n" "$ram_mb" \
        "$unique_nodes" "$total_batches"

    echo "$dataset,$mode,$sample_wall,$sample_rss_mb,$l1_n,$l2_n,$l3_n,$l4_n,$ram_mb,$unique_nodes,$total_batches,$train_b,$val_b,$test_b" >> "$CSV_OUT"
}

run_dataset() {
    local dataset="$1" db="$2" node_label="$3" edge_type="$4" port="$5"
    local proj_with="${BENCH_PROJ}_with_snap"
    local proj_no="${BENCH_PROJ}_no_snap"
    local logfile="/tmp/bench_four_level_${dataset}_${port}.log"

    # Wipe any leftover projections from prior bench aborts.
    drop_proj_offline "$db" "$proj_with"
    drop_proj_offline "$db" "$proj_no"

    echo "--- [$dataset] starting server on port $port ---"
    start_server "$db" "$port" "$logfile"

    # Build BOTH projections up-front so all four mode runs can reuse them.
    CURRENT_PROJ_WITH="$proj_with"
    CURRENT_PROJ_NO="$proj_no"

    echo "    building $proj_with (snapshot=true) …"
    local nc_ec_with
    nc_ec_with=$(build_projection "$port" "$proj_with" "$node_label" "$edge_type" "true")
    echo "      → nodes,relsh = $nc_ec_with"

    echo "    building $proj_no (snapshot=false) …"
    local nc_ec_no
    nc_ec_no=$(build_projection "$port" "$proj_no" "$node_label" "$edge_type" "false")
    echo "      → nodes,relsh = $nc_ec_no"

    # Run each mode sequentially against the running server. The server
    # is left running across modes — sample wall-clock is what we measure;
    # projection build is amortised across all four.
    for mode in "${MODES_LIST[@]}"; do
        echo "  --- [$dataset / mode=$mode] sampling ---"
        run_one_sample "$dataset" "$db" "$mode" "$port" "$logfile"
    done

    stop_server
    drop_proj_offline "$db" "$proj_with"
    drop_proj_offline "$db" "$proj_no"
    CURRENT_PROJ_WITH=
    CURRENT_PROJ_NO=
}

echo "================================================================"
echo "Four-Level Topology Store bench — $(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "  git HEAD   : $(git -C "$(dirname "$0")/.." rev-parse --short HEAD 2>/dev/null || echo unknown)"
echo "  binary     : $MDB"
echo "  stack      : MDB_PROJECTION_SORTER=radix"
echo "  modes      : ${MODES_LIST[*]}"
echo "  datasets   : $(printf '%s ' "${DATASETS_LIST[@]%%|*}")"
echo "  fanouts    : $FANOUTS"
echo "  batchSize  : $NUM_SEEDS"
echo "  CSV output : $CSV_OUT"
echo "================================================================"

PORT="$PORT_BASE"
for entry in "${DATASETS_LIST[@]}"; do
    IFS='|' read -r name db nl et <<<"$entry"
    if [[ ! -d "$db" ]]; then
        echo "WARN: dataset $name not found at $db — skipping."
        continue
    fi
    run_dataset "$name" "$db" "$nl" "$et" "$PORT"
    PORT=$((PORT + 1))
done

# --- Pretty-printed summary: per dataset, sorted by sample_wall_sec asc ----
echo
echo "================================================================"
echo "Summary (per dataset, sorted by sample_wall_sec ascending)"
echo "================================================================"
LC_NUMERIC=C awk -F',' '
NR == 1 { next }
{
    ds = $1
    rec[ds, ++cnt[ds]] = $0
    ds_seen[ds] = 1
}
END {
    n = 0
    for (ds in ds_seen) order[++n] = ds
    for (i = 1; i <= n; i++)
        for (j = i + 1; j <= n; j++)
            if (order[i] > order[j]) { t = order[i]; order[i] = order[j]; order[j] = t }

    for (i = 1; i <= n; i++) {
        ds = order[i]
        printf "\n[%s]\n", ds
        printf "  %-9s %-12s %-9s %-9s %-9s %-9s %-9s %-13s %-10s\n", \
            "mode", "sample_sec", "rss_MB", "l1_nodes", "l2_nodes", "l3_nodes", "l4_nodes", "l1_l2_ram_MB", "uniq_nodes"
        printf "  %-9s %-12s %-9s %-9s %-9s %-9s %-9s %-13s %-10s\n", \
            "----", "----------", "------", "--------", "--------", "--------", "--------", "------------", "----------"

        # Collect rows for this dataset.
        m = 0
        for (k = 1; k <= cnt[ds]; k++) {
            row = rec[ds, k]
            split(row, f, ",")
            m++
            mode_arr[m]  = f[2]
            wall_arr[m]  = f[3] + 0
            rss_arr[m]   = f[4] + 0
            l1_arr[m]    = f[5] + 0
            l2_arr[m]    = f[6] + 0
            l3_arr[m]    = f[7] + 0
            l4_arr[m]    = f[8] + 0
            ram_arr[m]   = f[9] + 0
            uniq_arr[m]  = f[10] + 0
        }
        # Insertion sort by wall.
        for (a = 1; a <= m; a++)
            for (b = a + 1; b <= m; b++)
                if (wall_arr[a] > wall_arr[b]) {
                    for (col_name in arrs) {}
                    t = wall_arr[a]; wall_arr[a] = wall_arr[b]; wall_arr[b] = t
                    t = mode_arr[a]; mode_arr[a] = mode_arr[b]; mode_arr[b] = t
                    t = rss_arr[a];  rss_arr[a]  = rss_arr[b];  rss_arr[b]  = t
                    t = l1_arr[a];   l1_arr[a]   = l1_arr[b];   l1_arr[b]   = t
                    t = l2_arr[a];   l2_arr[a]   = l2_arr[b];   l2_arr[b]   = t
                    t = l3_arr[a];   l3_arr[a]   = l3_arr[b];   l3_arr[b]   = t
                    t = l4_arr[a];   l4_arr[a]   = l4_arr[b];   l4_arr[b]   = t
                    t = ram_arr[a];  ram_arr[a]  = ram_arr[b];  ram_arr[b]  = t
                    t = uniq_arr[a]; uniq_arr[a] = uniq_arr[b]; uniq_arr[b] = t
                }
        for (a = 1; a <= m; a++) {
            printf "  %-9s %-12.3f %-9d %-9d %-9d %-9d %-9d %-13d %-10d\n", \
                mode_arr[a], wall_arr[a], rss_arr[a], \
                l1_arr[a], l2_arr[a], l3_arr[a], l4_arr[a], \
                ram_arr[a], uniq_arr[a]
        }
        # Wipe per-dataset accumulators.
        for (a = 1; a <= m; a++) {
            mode_arr[a]=""; wall_arr[a]=0; rss_arr[a]=0
            l1_arr[a]=0;    l2_arr[a]=0;   l3_arr[a]=0;   l4_arr[a]=0
            ram_arr[a]=0;   uniq_arr[a]=0
        }
    }
}' "$CSV_OUT"

echo
echo "CSV: $CSV_OUT"
