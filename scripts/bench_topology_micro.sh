#!/bin/bash
# Spec #4-B T4.17 — shell wrapper around bench_topology_micro.
#
# Drives the isolated TopologyAccessor micro-benchmark across the
# {cora_gnn, ogbn-arxiv, ogbn-products} dataset matrix. For datasets
# that don't already have a projection, runs `mdb server` + curl to
# build one via `graph_project` with `indexSet: 'GNN_MINIMAL'` and
# `buildTopologySnapshot: true`. Once the projection exists on disk,
# invokes the standalone C++ benchmark binary against it (the binary
# handles CSR-parking / BPT-fallback / CSR-restore internally).
#
# Strict: papers100M is never projected here.
#
# Output:
#   - Human-readable block per dataset, printed by the bench binary.
#   - Aggregated CSV at /tmp/bench_topology_micro_<ts>.csv with one
#     row per dataset.
#
# Usage:
#   ./scripts/bench_topology_micro.sh              # default matrix
#   DATASETS="cora_gnn" ./scripts/bench_topology_micro.sh
#   NUM_SEEDS=5000 FANOUTS="15,10" ./scripts/bench_topology_micro.sh
#
# Env:
#   MDB            Path to mdb binary (default: ./build/Release/bin/mdb)
#   BENCH          Path to bench binary (default: ./build/Release/bin/bench_topology_micro)
#   PORT_BASE      Starting port for graph_project server (default: 19971; +1 per dataset)
#   NUM_SEEDS      Seeds for the timed pass (default: 10000)
#   WARMUP         Warm-up seeds (default: 100)
#   FANOUTS        CSV fanouts for `--fanouts` (default: "15,10")
#   ORIENTATION    NATURAL|REVERSE|UNDIRECTED (default: UNDIRECTED)
#   DATASETS       Space-separated dataset basenames under data/dbs/gql/

set -euo pipefail

MDB=${MDB:-./build/Release/bin/mdb}
BENCH=${BENCH:-./build/Release/bin/bench_topology_micro}
PORT_BASE=${PORT_BASE:-19971}
NUM_SEEDS=${NUM_SEEDS:-10000}
WARMUP=${WARMUP:-100}
FANOUTS=${FANOUTS:-"15,10"}
ORIENTATION=${ORIENTATION:-UNDIRECTED}

# Fanouts string for GQL call (spaces, brackets).
FANOUTS_GQL=$(echo "[$FANOUTS]" | sed 's/,/, /g')

# Dataset configuration: name|db_path|node_label|edge_type|proj_name
declare -a DATASETS_DEFAULT=(
    "cora_gnn|data/dbs/gql/cora_gnn|Paper|CITES|cora_proj"
    "ogbn-arxiv|data/dbs/gql/ogbn-arxiv|Node|CONNECTS|bench_topo_micro_arxiv"
    "ogbn-products|data/dbs/gql/ogbn-products|Node|CONNECTS|bench_topo_micro_products"
)

if [[ -n "${DATASETS:-}" ]]; then
    NEW_DATASETS=()
    for ds in $DATASETS; do
        if [[ "$ds" == "papers100M" || "$ds" == *papers100M* ]]; then
            echo "ERROR: papers100M is out of scope (celebi-only dataset)." >&2
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

for entry in "${DATASETS_LIST[@]}"; do
    name="${entry%%|*}"
    if [[ "$name" == "papers100M" || "$name" == *papers100M* ]]; then
        echo "ERROR: papers100M is out of scope here." >&2
        exit 3
    fi
done

TS=$(date +%s)
CSV_OUT="/tmp/bench_topology_micro_${TS}.csv"
echo "dataset,orientation,fanouts,num_seeds,bpt_sec,csr_sec,bpt_sps,csr_sps,speedup" > "$CSV_OUT"

SRV_PID=
cleanup() {
    if [[ -n "${SRV_PID:-}" ]] && kill -0 "$SRV_PID" 2>/dev/null; then
        kill "$SRV_PID" 2>/dev/null || true
        for _ in 1 2 3 4 5 6 7 8 9 10; do
            kill -0 "$SRV_PID" 2>/dev/null || break
            sleep 0.3
        done
        kill -9 "$SRV_PID" 2>/dev/null || true
        wait "$SRV_PID" 2>/dev/null || true
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
    for _ in $(seq 1 60); do
        if curl -sSf -o /dev/null --data-binary "RETURN 1" \
                -H "Accept: text/csv" \
                "http://127.0.0.1:$port/" 2>/dev/null; then
            return 0
        fi
        sleep 0.5
    done
    echo "ERROR: server did not come up on port $port" >&2
    tail -60 "$logfile" >&2 || true
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
}

ensure_projection() {
    local db="$1" proj="$2" node_label="$3" edge_type="$4" port="$5"

    local proj_dir="$db/projections/$proj"
    if [[ -f "$proj_dir/catalog.dat" ]]; then
        echo "  [skip-build] projection '$proj' already exists at $proj_dir"
        return 0
    fi

    local logfile="/tmp/bench_topo_micro_build_${proj}_${port}.log"
    echo "  [build] starting mdb server on port $port → $logfile"
    start_server "$db" "$port" "$logfile"

    local build_query
    build_query="CALL graph_project('$proj', '$node_label', '$edge_type', {orientation: 'NATURAL', indexSet: 'GNN_MINIMAL', buildTopologySnapshot: true}) YIELD graphName, nodeCount, relationshipCount, topologySnapshotBytes RETURN *"

    echo "  [build] graph_project '$proj' ..."
    local t0 t1
    t0=$(date +%s.%N)
    local resp
    resp=$(curl -sS --max-time 14400 --data-binary "$build_query" \
        -H "Accept: text/csv" "http://127.0.0.1:$port/" || true)
    t1=$(date +%s.%N)
    local wall
    wall=$(LC_NUMERIC=C awk -v t0="$t0" -v t1="$t1" 'BEGIN { printf "%.2f", t1 - t0 }')
    echo "  [build] done in ${wall}s"
    echo "$resp" | sed -n '1,4p' | sed 's/^/      /'

    stop_server

    if [[ ! -f "$proj_dir/catalog.dat" ]]; then
        echo "ERROR: projection build did not produce catalog.dat at $proj_dir" >&2
        tail -80 "$logfile" >&2 || true
        return 2
    fi
}

port=$PORT_BASE
for entry in "${DATASETS_LIST[@]}"; do
    IFS='|' read -r name db node_label edge_type proj <<< "$entry"
    port=$((port + 1))

    echo ""
    echo "=========================================================="
    echo "Dataset: $name  (db=$db, proj=$proj)"
    echo "=========================================================="

    if [[ ! -d "$db" ]]; then
        echo "  [skip] db folder missing: $db" >&2
        continue
    fi

    if ! ensure_projection "$db" "$proj" "$node_label" "$edge_type" "$port"; then
        echo "  [skip] projection build failed for $name" >&2
        continue
    fi

    # Run the isolated bench.
    run_out="/tmp/bench_topo_micro_${name}_${TS}.out"
    if ! "$BENCH" "$db" "$proj" \
            --num-seeds "$NUM_SEEDS" \
            --warmup "$WARMUP" \
            --fanouts "$FANOUTS" \
            --orientation "$ORIENTATION" \
            --dataset-label "$name" 2>&1 | tee "$run_out"; then
        echo "  [fail] bench binary non-zero exit on $name" >&2
        continue
    fi

    # Extract the BENCH_CSV line and strip the tag prefix.
    csv_line=$(grep '^BENCH_CSV,' "$run_out" | tail -n 1 | sed 's/^BENCH_CSV,//')
    if [[ -n "$csv_line" ]]; then
        echo "$csv_line" >> "$CSV_OUT"
    fi
done

echo ""
echo "=========================================================="
echo "CSV summary: $CSV_OUT"
echo "=========================================================="
cat "$CSV_OUT"
