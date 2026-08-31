#!/bin/bash
# Golden compare: 4 (scan × sort) mode combinations + the builder-level
# PARALLEL edge scan, all on cora_gnn.
#
# Runs graph_project once per mode:
#   cc  classic scan  / classic sort   (baseline)
#   cr  classic scan  / radix   sort
#   sc  serial scan   / classic sort
#   sr  serial scan   / radix   sort
#   par PARALLEL edge scan (MDB_PROJECTION_PARALLEL_SCAN=1) / classic sort
# Every .leaf / .dir B+Tree file from the baseline (cc) is byte-compared
# against the same file from the other modes.
#
# A clean run proves invariant I4: bit-identical B+Tree output across all
# mode combinations, confirming each path is a pure-refactor alternative to
# the classic path. The PARALLEL mode adds the builder-level parallel edge
# scan to that invariant (its has_node + orientation run in TBB workers, the
# order-sensitive tail replays single-threaded in ascending-partition order).
#
# Usage:  ./scripts/test_projection_radix.sh
# Exit:   0 on ALL MATCH, 1 on mismatch, 2 on setup/server error.
set -euo pipefail

# Resolved from this script rather than from the caller's directory, so the
# benchmark runs from anywhere. MDB_HOME still overrides.
MDB_HOME="${MDB_HOME:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
MDB=${MDB:-$MDB_HOME/build/Release/bin/mdb}
DB=${DB:-data/dbs/gql/cora_gnn}
PORT=${PORT:-19880}

# Cora schema. Override via NODE_LABEL=... EDGE_TYPE=... envs if needed.
NODE_LABEL=${NODE_LABEL:-Paper}
EDGE_TYPE=${EDGE_TYPE:-CITES}

# 7 projections:
#   4 (scan=classic|serial × sort=classic|radix) + 1 PARALLEL
#   + 2 RADIX GPU on/off (Task 4.3): rgpu uses the GPU-eligible radix path
#     (default), rcpu forces MDB_FORCE_CPU_SORT=1 so the radix Phase 2
#     per-partition sort routes everything to CPU. Byte-comparing rgpu and
#     rcpu against the cc baseline proves the GPU per-partition sort produces
#     output bit-identical to the CPU path (gpu_sort_preconditions_hold
#     guarantees order-equivalence).
# PARALLEL_MODES[i]="1" sets MDB_PROJECTION_PARALLEL_SCAN=1 for that run.
# FORCE_CPU_MODES[i]="1" sets MDB_FORCE_CPU_SORT=1 for that run.
PROJS=(test_golden_cc test_golden_cr test_golden_sc test_golden_sr test_golden_par test_golden_rgpu test_golden_rcpu)
SCAN_MODES=(0 0 1 1 0 0 0)
SORT_MODES=(classic radix classic radix classic radix radix)
PARALLEL_MODES=(0 0 0 0 1 0 0)
FORCE_CPU_MODES=(0 0 0 0 0 0 1)

cleanup() {
    if [[ -n "${SRV_PID:-}" ]] && kill -0 "$SRV_PID" 2>/dev/null; then
        kill "$SRV_PID" 2>/dev/null || true
        wait "$SRV_PID" 2>/dev/null || true
    fi
    for p in "${PROJS[@]}"; do
        "$MDB" drop-projection "$DB" "$p" 2>/dev/null || true
    done
}
trap cleanup EXIT

SRV_PID=

run_projection() {
    local scan="$1"
    local sort="$2"
    local proj="$3"
    local parallel="${4:-0}"
    local force_cpu="${5:-0}"

    echo ">>> [scan=$scan sort=$sort parallel=$parallel force_cpu=$force_cpu] Starting server on port $PORT..."
    # MDB_FORCE_CPU_SORT, when set to "1", disables the radix Phase 2 GPU
    # per-partition sort (the server-side switch matched at
    # radix_partition_sort.cc + external_record_sort.h). Export it only when
    # requested so the GPU-eligible run sees an unset env var, matching how
    # production runs the path.
    local extra_env=()
    if [[ "$force_cpu" == "1" ]]; then
        extra_env+=(MDB_FORCE_CPU_SORT=1)
    fi
    env "${extra_env[@]}" \
        MDB_PROJECTION_SERIAL_SCAN="$scan" \
        MDB_PROJECTION_SORTER="$sort" \
        MDB_PROJECTION_PARALLEL_SCAN="$parallel" \
        "$MDB" server "$DB" --port "$PORT" --timeout 600 \
        > "/tmp/mdb_golden_${scan}_${sort}_p${parallel}_c${force_cpu}.log" 2>&1 &
    SRV_PID=$!

    # Wait for server to accept connections (max 10 s).
    for i in $(seq 1 20); do
        if curl -sSf -o /dev/null "http://127.0.0.1:$PORT/" \
                --data-binary "RETURN 1" -H "Accept: text/csv" 2>/dev/null; then
            break
        fi
        sleep 0.5
    done

    echo ">>> [scan=$scan sort=$sort parallel=$parallel force_cpu=$force_cpu] Creating projection '$proj'..."
    local query="CALL graph_project('$proj', '$NODE_LABEL', '$EDGE_TYPE') YIELD graphName, nodeCount, relCount RETURN *"
    local response
    response=$(curl -sS --data-binary "$query" -H "Accept: text/csv" "http://127.0.0.1:$PORT/" || true)
    echo "$response" | head -5

    kill "$SRV_PID" 2>/dev/null || true
    wait "$SRV_PID" 2>/dev/null || true
    SRV_PID=
    # Surface the radix telemetry line (Task 4.2) so the run log shows the
    # observed gpu/cpu partition split.
    grep -h "\[RADIX\] partitions sorted:" \
        "/tmp/mdb_golden_${scan}_${sort}_p${parallel}_c${force_cpu}.log" 2>/dev/null || true
    echo ">>> [scan=$scan sort=$sort parallel=$parallel force_cpu=$force_cpu] Server stopped; projection written."
    echo
}

echo "=== Golden compare — scan×sort combinations + PARALLEL + GPU on/off on $(basename "$DB") ==="
echo "    Modes: cc (classic/classic)  cr (classic/radix)"
echo "           sc (serial/classic)   sr (serial/radix)"
echo "           par (PARALLEL edge scan / classic sort)"
echo "           rgpu (radix, GPU-eligible)   rcpu (radix, MDB_FORCE_CPU_SORT=1)"
echo

for i in "${!PROJS[@]}"; do
    run_projection "${SCAN_MODES[$i]}" "${SORT_MODES[$i]}" "${PROJS[$i]}" \
        "${PARALLEL_MODES[$i]}" "${FORCE_CPU_MODES[$i]}"
done

# Verify all projection directories exist before comparing.
for p in "${PROJS[@]}"; do
    pdir="$DB/projections/$p"
    if [[ ! -d "$pdir" ]]; then
        echo "ERROR: projection directory missing: $pdir" >&2
        exit 2
    fi
done

BASE="${PROJS[0]}"
BASE_DIR="$DB/projections/$BASE"

echo "=== Byte-comparing B+Tree files (baseline: $BASE) ==="
MISMATCH=0
TOTAL=0

for other in "${PROJS[@]:1}"; do
    OTHER_DIR="$DB/projections/$other"
    echo "--- $BASE vs $other ---"
    for f in "$BASE_DIR"/*.leaf "$BASE_DIR"/*.dir; do
        [[ -e "$f" ]] || continue
        name=$(basename "$f")
        f2="$OTHER_DIR/$name"
        TOTAL=$((TOTAL + 1))

        if [[ ! -f "$f2" ]]; then
            echo "MISSING in $other: $name"
            MISMATCH=$((MISMATCH + 1))
        elif ! cmp -s "$f" "$f2"; then
            sz_b=$(stat -c %s "$f")
            sz_o=$(stat -c %s "$f2")
            echo "MISMATCH: $name (${BASE}=${sz_b}B, ${other}=${sz_o}B)"
            MISMATCH=$((MISMATCH + 1))
        else
            sz=$(stat -c %s "$f")
            echo "MATCH:    $name (${sz}B)"
        fi
    done
    echo
done

NPAIRS=$(( ${#PROJS[@]} - 1 ))
if [[ $MISMATCH -eq 0 && $TOTAL -gt 0 ]]; then
    echo "ALL ${#PROJS[@]} MODES MATCH — golden compare PASSED on $TOTAL files x $NPAIRS pairs"
    exit 0
else
    echo "FAILED — $MISMATCH mismatches across $TOTAL files x $NPAIRS pairs"
    exit 1
fi
