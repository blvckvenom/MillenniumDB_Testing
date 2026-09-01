#!/bin/bash
# Spec #4-B T4.12 — topology-snapshot benchmark harness.
#
# For each (dataset, snapshot mode) pair in
#   {cora_gnn, ogbn-arxiv, ogbn-products} × {on, off}
# measure:
#   Projection build:
#     - wall clock of graph_project (curl request)
#     - snapshot_bytes = size(topology_fwd.csr) + size(topology_rev.csr)
#     - proj_total_bytes = all .leaf + .dir + .csr under projection dir
#     - peak RSS of the mdb server during build (VmHWM from /proc)
#     - nodeCount / relationshipCount (from YIELD)
#   Sampling:
#     - wall clock of CALL gnn_offline_sample with fanout [10, 15] (DGL order; hop order 15 then 10)
#     - seeds_per_sec = uniqueNodes_sampled / wall_sec
#     - peak RSS of the mdb server during the sampling call
#
# Both phases use MDB_PROJECTION_SERIAL_SCAN=1 MDB_PROJECTION_SORTER=radix and
# indexSet='GNN_MINIMAL' — the difference between modes is purely the CSR
# sidecar build (`buildTopologySnapshot: true`) and the resulting B+Tree ↔
# mmap slice dispatch at sample time.
#
# Strict: papers100M is NEVER projected here: this bench is scoped to graphs that
# finish in minutes, not to a 111M-node graph.
#
# Output:
#   CSV at /tmp/bench_topology_snapshot_<ts>.csv with columns
#     dataset,snapshot,proj_wall_sec,snap_bytes,proj_total_bytes,proj_peak_rss_mb,
#     sample_wall_sec,seeds_per_sec,sample_peak_rss_mb,num_nodes,num_edges
#   plus pretty-printed summary table to stdout.
#
# Usage: ./scripts/bench_topology_snapshot.sh
# Env:
#   MDB            Path to mdb binary (default: ./build/Release/bin/mdb)
#   PORT_BASE      Starting port (default: 19981; +1 per run)
#   DATASETS       Override dataset list (space-separated basenames under data/dbs/gql)
#   MODES          Override mode list (space-separated: on off)
#   NUM_SEEDS      Target seed count (default: 10000; informational only —
#                  the sampler uses all labeled nodes in the projection)
#   FANOUTS        Fanout list used in the sampling call (default: "[10, 15]", DGL order)

set -euo pipefail

MDB=${MDB:-./build/Release/bin/mdb}
PORT_BASE=${PORT_BASE:-19981}
BENCH_PROJ=${BENCH_PROJ:-bench_topo_snap_tmp}
BENCH_SAMPLE=${BENCH_SAMPLE:-bench_topo_snap_sample}
NUM_SEEDS=${NUM_SEEDS:-10000}
# Fanout notation: since 2026-07-06 gnn_offline_sample reads the list in DGL
# order -- the LAST element is the hop adjacent to the seeds -- and reverses it
# internally. The default below was flipped to preserve the hop order this bench
# was written to measure; results produced before that flip sampled the mirror
# order and are not comparable with results produced after.
FANOUTS=${FANOUTS:-"[10, 15]"}

# Dataset configuration: name  db_path  node_label  edge_type
declare -a DATASETS_DEFAULT=(
    "cora_gnn|data/dbs/gql/cora_gnn|Paper|CITES"
    "ogbn-arxiv|data/dbs/gql/ogbn-arxiv|Node|CONNECTS"
    "ogbn-products|data/dbs/gql/ogbn-products|Node|CONNECTS"
)

if [[ -n "${DATASETS:-}" ]]; then
    NEW_DATASETS=()
    for ds in $DATASETS; do
        if [[ "$ds" == "papers100M" || "$ds" == *papers100M* ]]; then
            echo "ERROR: papers100M is out of scope for bench_topology_snapshot.sh: projecting it takes tens of minutes. Aborting." >&2
            exit 3
        fi
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

MODES_LIST=(${MODES:-on off})

# Strict guard: papers100M is out of scope here.
for entry in "${DATASETS_LIST[@]}"; do
    name="${entry%%|*}"
    if [[ "$name" == "papers100M" || "$name" == *papers100M* ]]; then
        echo "ERROR: papers100M is out of scope for bench_topology_snapshot.sh: projecting it takes tens of minutes. Aborting." >&2
        exit 3
    fi
done

TS=$(date +%s)
CSV_OUT="/tmp/bench_topology_snapshot_${TS}.csv"
echo "dataset,snapshot,proj_wall_sec,snap_bytes,proj_total_bytes,proj_peak_rss_mb,sample_wall_sec,seeds_per_sec,sample_peak_rss_mb,num_nodes,num_edges" > "$CSV_OUT"

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
    # Best-effort cleanup of in-flight sample + projection
    if [[ -n "${CURRENT_DB:-}" && -n "${CURRENT_SAMPLE:-}" ]]; then
        rm -rf "$CURRENT_DB/samples/$CURRENT_SAMPLE" 2>/dev/null || true
    fi
    if [[ -n "${CURRENT_DB:-}" && -n "${CURRENT_PROJ:-}" ]]; then
        "$MDB" drop-projection "$CURRENT_DB" "$CURRENT_PROJ" >/dev/null 2>&1 || true
    fi
}
trap cleanup EXIT INT TERM

start_server() {
    local db="$1" port="$2" logfile="$3"
    MDB_PROJECTION_SERIAL_SCAN=1 \
    MDB_PROJECTION_SORTER=radix \
        "$MDB" server "$db" --port "$port" --timeout 3600 \
        > "$logfile" 2>&1 &
    SRV_PID=$!
    CURRENT_DB="$db"
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

# Convert last VmHWM sample (kB) from file into MB, default 0 on empty.
rss_file_to_mb() {
    local rssfile="$1"
    local rss_kb=0
    if [[ -s "$rssfile" ]]; then
        rss_kb=$(tail -n 1 "$rssfile")
    fi
    LC_NUMERIC=C awk -v kb="$rss_kb" 'BEGIN { printf "%.0f", kb/1024 }'
}

# Read the VmHWM peak at the moment the sampler started (since VmHWM is
# cumulative, we snapshot /proc/<pid>/status BEFORE a phase and subtract
# later if a per-phase peak is wanted). Here we just snapshot a baseline.
snapshot_vmhwm_kb() {
    local pid="$1"
    if [[ -r /proc/$pid/status ]]; then
        awk '/^VmHWM:/ {print $2}' /proc/$pid/status 2>/dev/null || echo 0
    else
        echo 0
    fi
}

drop_proj_offline() {
    local db="$1" proj="$2"
    "$MDB" drop-projection "$db" "$proj" >/dev/null 2>&1 || true
}

# Delete on-disk sample storage (since there is no CLI drop for samples).
drop_sample_offline() {
    local db="$1" sample="$2"
    rm -rf "$db/samples/$sample" 2>/dev/null || true
}

# run_one: one (dataset, mode) cell of the matrix.
#   $1 dataset, $2 db_path, $3 node_label, $4 edge_type, $5 mode(on|off), $6 port
run_one() {
    local dataset="$1" db="$2" node_label="$3" edge_type="$4" mode="$5" port="$6"
    local proj="${BENCH_PROJ}_${mode}"
    local sample="${BENCH_SAMPLE}_${mode}"
    local logfile="/tmp/bench_topo_snap_${dataset}_${mode}_${port}.log"
    local rss_build="/tmp/bench_topo_snap_${dataset}_${mode}_${port}.build.rss"
    local rss_sample="/tmp/bench_topo_snap_${dataset}_${mode}_${port}.sample.rss"

    local snap_flag
    if [[ "$mode" == "on" ]]; then
        snap_flag="true"
    else
        snap_flag="false"
    fi

    CURRENT_PROJ="$proj"
    CURRENT_SAMPLE="$sample"
    drop_proj_offline "$db" "$proj"
    drop_sample_offline "$db" "$sample"

    echo "--- [$dataset / snapshot=$mode] starting server on port $port ---"
    start_server "$db" "$port" "$logfile"

    # ── Phase 1: projection build ──────────────────────────────────────
    start_peak_rss_sampler "$SRV_PID" "$rss_build"

    local build_query
    build_query="CALL graph_project('$proj', '$node_label', '$edge_type', {orientation: 'NATURAL', indexSet: 'GNN_MINIMAL', buildTopologySnapshot: $snap_flag}) YIELD graphName, nodeCount, relationshipCount, topologySnapshotBytes RETURN *"

    local t0 t1 proj_wall
    t0=$(date +%s.%N)
    local build_resp
    build_resp=$(curl -sS --max-time 3000 --data-binary "$build_query" \
        -H "Accept: text/csv" "http://127.0.0.1:$port/" || true)
    t1=$(date +%s.%N)
    proj_wall=$(LC_NUMERIC=C awk -v t0="$t0" -v t1="$t1" 'BEGIN { printf "%.3f", t1 - t0 }')

    stop_peak_rss_sampler
    local proj_rss_mb
    proj_rss_mb=$(rss_file_to_mb "$rss_build")

    # Parse CSV response:
    #   "graphName","nodeCount","relationshipCount","topologySnapshotBytes"
    #   "bench_topo_snap_tmp_on",2708,5429,88720
    local data_line nc ec topo_yield
    data_line=$(echo "$build_resp" | sed -n '2p' | tr -d '"')
    nc=$(echo "$data_line" | awk -F',' '{print $2}')
    ec=$(echo "$data_line" | awk -F',' '{print $3}')
    topo_yield=$(echo "$data_line" | awk -F',' '{print $4}')
    [[ -z "$nc" ]] && nc=0
    [[ -z "$ec" ]] && ec=0
    [[ -z "$topo_yield" ]] && topo_yield=0

    local proj_dir="$db/projections/$proj"
    local proj_total_bytes=0 snap_bytes=0
    if [[ -d "$proj_dir" ]]; then
        # Sum all .leaf / .dir / .csr files under the projection.
        proj_total_bytes=$(find "$proj_dir" -maxdepth 1 \
            \( -name '*.leaf' -o -name '*.dir' -o -name '*.csr' \) \
            -printf '%s\n' 2>/dev/null | awk '{s+=$1} END {print s+0}')
        # Explicit CSR pair (0 when snapshot off).
        local fwd_path="$proj_dir/topology_fwd.csr"
        local rev_path="$proj_dir/topology_rev.csr"
        local fwd_bytes=0 rev_bytes=0
        [[ -f "$fwd_path" ]] && fwd_bytes=$(stat -c '%s' "$fwd_path" 2>/dev/null || echo 0)
        [[ -f "$rev_path" ]] && rev_bytes=$(stat -c '%s' "$rev_path" 2>/dev/null || echo 0)
        snap_bytes=$((fwd_bytes + rev_bytes))
    fi

    # Sanity: when snapshot=off, CSRs must be absent; when on, both present.
    if [[ "$mode" == "on" && "$snap_bytes" -eq 0 ]]; then
        echo "WARN  snapshot=on but topology_*.csr missing on $dataset — see $logfile" >&2
    fi
    if [[ "$mode" == "off" && "$snap_bytes" -ne 0 ]]; then
        echo "WARN  snapshot=off but topology_*.csr present on $dataset ($snap_bytes B)" >&2
    fi

    # ── Phase 2: sampling ──────────────────────────────────────────────
    # Fresh peak-RSS tracker so we observe per-phase peaks (note: VmHWM
    # is cumulative, so the sampling peak is reported as max(build_peak,
    # sampling_peak). For the common case where sampling allocates less
    # than build, this will equal the build peak — expected behaviour.)
    local sample_wall=0 seeds_per_sec=0 sample_rss_mb=0 unique_nodes=0

    start_peak_rss_sampler "$SRV_PID" "$rss_sample"

    local sample_query
    sample_query="CALL gnn_offline_sample('$proj', '$sample', $FANOUTS, {batchSize: $NUM_SEEDS, randomSeed: 42, orientation: 'UNDIRECTED'}) YIELD sampleName, totalBatches, trainBatches, validationBatches, testBatches, uniqueNodes, computeMillis RETURN *"

    local ts0 ts1
    ts0=$(date +%s.%N)
    local sample_resp
    sample_resp=$(curl -sS --max-time 3000 --data-binary "$sample_query" \
        -H "Accept: text/csv" "http://127.0.0.1:$port/" || true)
    ts1=$(date +%s.%N)
    sample_wall=$(LC_NUMERIC=C awk -v t0="$ts0" -v t1="$ts1" 'BEGIN { printf "%.3f", t1 - t0 }')

    stop_peak_rss_sampler
    sample_rss_mb=$(rss_file_to_mb "$rss_sample")

    # Sampling CSV:
    #   "sampleName","totalBatches","trainBatches","validationBatches","testBatches","uniqueNodes","computeMillis"
    local sdata
    sdata=$(echo "$sample_resp" | sed -n '2p' | tr -d '"')
    unique_nodes=$(echo "$sdata" | awk -F',' '{print $6}')
    [[ -z "$unique_nodes" ]] && unique_nodes=0

    # seeds/sec: total seeds sampled across all splits / wall clock.
    # The bench treats `uniqueNodes` (= seeds + neighbor expansion) as the
    # throughput numerator, since that is what the sidecar path accelerates.
    if [[ -z "$sample_resp" ]] || ! echo "$sample_resp" | head -1 | grep -q sampleName; then
        echo "WARN  sampling failed on $dataset mode=$mode — see $logfile" >&2
        seeds_per_sec=0
    else
        seeds_per_sec=$(LC_NUMERIC=C awk -v u="$unique_nodes" -v w="$sample_wall" \
            'BEGIN { if (w > 0) printf "%.0f", u / w; else print 0 }')
    fi

    # ── Cleanup ────────────────────────────────────────────────────────
    stop_server
    drop_sample_offline "$db" "$sample"
    drop_proj_offline "$db" "$proj"
    CURRENT_PROJ=
    CURRENT_SAMPLE=

    printf 'RESULT  %-14s snap=%-3s  build=%ss  snap_B=%s  total_B=%s  proj_rss=%s MB  sample=%ss  seeds/s=%s  sample_rss=%s MB  N=%s  E=%s  uniq=%s\n' \
        "$dataset" "$mode" "$proj_wall" "$snap_bytes" "$proj_total_bytes" \
        "$proj_rss_mb" "$sample_wall" "$seeds_per_sec" "$sample_rss_mb" \
        "$nc" "$ec" "$unique_nodes"

    echo "$dataset,$mode,$proj_wall,$snap_bytes,$proj_total_bytes,$proj_rss_mb,$sample_wall,$seeds_per_sec,$sample_rss_mb,$nc,$ec" >> "$CSV_OUT"
}

echo "================================================================"
echo "Topology-snapshot bench — $(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "  git HEAD   : $(git -C "$(dirname "$0")/.." rev-parse --short HEAD 2>/dev/null || echo unknown)"
echo "  binary     : $MDB"
echo "  stack      : MDB_PROJECTION_SERIAL_SCAN=1  MDB_PROJECTION_SORTER=radix"
echo "  indexSet   : GNN_MINIMAL (both modes)"
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
    for mode in "${MODES_LIST[@]}"; do
        run_one "$name" "$db" "$nl" "$et" "$mode" "$PORT"
        PORT=$((PORT + 1))
    done
done

# --- Pretty-printed summary: snapshot=on vs snapshot=off per dataset ---------
echo
echo "================================================================"
echo "Summary (snapshot=on relative to snapshot=off, per dataset)"
echo "================================================================"
LC_NUMERIC=C awk -F',' '
NR == 1 { next }
{
    ds = $1; mode = $2
    key = ds "|" mode
    pwall[key]  = $3 + 0
    sbytes[key] = $4 + 0
    ptot[key]   = $5 + 0
    prss[key]   = $6 + 0
    swall[key]  = $7 + 0
    sps[key]    = $8 + 0
    srss[key]   = $9 + 0
    nc[key]     = $10 + 0
    ec[key]     = $11 + 0
    ds_seen[ds] = 1
}
END {
    printf "%-14s %-4s %-10s %-10s %-11s %-9s %-11s %-11s %-9s %-14s %-14s\n", \
        "dataset", "snap", "build_s", "snap_MB", "total_MB", "rss_MB", \
        "sample_s", "seeds/s", "srss_MB", "build_vs_off", "speedup_on/off"
    printf "%-14s %-4s %-10s %-10s %-11s %-9s %-11s %-11s %-9s %-14s %-14s\n", \
        "------", "----", "-------", "-------", "--------", "------", \
        "--------", "-------", "-------", "------------", "--------------"

    n = 0
    for (ds in ds_seen) order[++n] = ds
    for (i = 1; i <= n; i++)
        for (j = i + 1; j <= n; j++)
            if (order[i] > order[j]) { t = order[i]; order[i] = order[j]; order[j] = t }

    modes[1] = "off"; modes[2] = "on"
    for (i = 1; i <= n; i++) {
        ds = order[i]
        base_build = pwall[ds "|off"]
        base_sps   = sps[ds "|off"]
        for (m = 1; m <= 2; m++) {
            mode = modes[m]
            k = ds "|" mode
            if (!(k in pwall) && !(k in swall)) continue
            snap_mb  = sbytes[k] / (1024 * 1024)
            total_mb = ptot[k]   / (1024 * 1024)
            build_overhead = "   (baseline)"
            speedup        = "   (baseline)"
            if (mode == "on") {
                if (base_build > 0) {
                    d = (pwall[k] / base_build - 1) * 100
                    build_overhead = sprintf("%+7.1f%%", d)
                }
                if (base_sps > 0 && sps[k] > 0) {
                    speedup = sprintf("%7.2fx", sps[k] / base_sps)
                }
            }
            printf "%-14s %-4s %-10.3f %-10.2f %-11.2f %-9d %-11.3f %-11d %-9d %-14s %-14s\n", \
                ds, mode, pwall[k], snap_mb, total_mb, prss[k], \
                swall[k], sps[k], srss[k], build_overhead, speedup
        }
    }
}' "$CSV_OUT"

echo
echo "CSV: $CSV_OUT"
