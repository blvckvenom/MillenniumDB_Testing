#!/bin/bash
# Task 16 — papers100M Run 7: SERIAL_SCAN=1 + SORTER=radix (thesis config).
#
# Launches the mdb server with both Spec #2 (MDB_PROJECTION_SERIAL_SCAN=1)
# and Spec #1 (MDB_PROJECTION_SORTER=radix) active, drives a graph_project
# over HTTP, and samples RSS + disk at 30 s cadence. PSI safety monitor
# (scripts/psi_safety_monitor.sh) runs in parallel and aborts on any of:
# PSI memory "some avg10" > 5%, RSS > 18 GB, sustained swap rate > 50 MB/s.
#
# The RSS threshold is bumped from Run 6's 15 GB to 18 GB because Tier-3
# ogbn-products empirically showed 11 GB peak RSS under serial mode (vs
# predicted 1.5 GB) — the source of the extra RAM is not yet identified
# but the PSI monitor's 18 GB ceiling leaves 13 GB headroom on 31 GB RAM
# machines, which is enough for OS + interactive apps.
#
# Run 6 (radix only) aborted at +73 min due to disk-full (not OOM). Run 7
# tests the central Spec #2 claim: serial mode bounds peak scratch disk
# to O(max single index) ≈ 40 GB instead of O(sum all indexes) ≈ 160 GB.
#
# Usage:  ./scripts/run_papers100m_run7.sh
# Env:    PORT=19879
# Output: /tmp/papers100m_run7/
set -euo pipefail

MDB=./build/Release/bin/mdb
DB=data/dbs/gql/papers100M
PROJ_NAME=papers100m_serial_radix_run7
PORT="${PORT:-19879}"
OUTDIR="/tmp/papers100m_run7"
mkdir -p "$OUTDIR"

SRV_LOG="$OUTDIR/server.log"
POST_LOG="$OUTDIR/post.log"
SAMPLE_CSV="$OUTDIR/samples.csv"
PSI_LOG="$OUTDIR/psi_monitor.log"

cleanup() {
    [[ -n "${PSI_PID:-}" ]] && kill -0 "$PSI_PID" 2>/dev/null && kill "$PSI_PID" 2>/dev/null
    [[ -n "${POST_PID:-}" ]] && kill -0 "$POST_PID" 2>/dev/null && kill "$POST_PID" 2>/dev/null
    [[ -n "${SRV_PID:-}" ]] && kill -0 "$SRV_PID" 2>/dev/null && kill "$SRV_PID" 2>/dev/null
    wait 2>/dev/null || true
}
trap cleanup EXIT

echo "=== Task 16 papers100M Run 7 (SERIAL_SCAN=1 + SORTER=radix) — $(date) ==="
echo "Pre-flight:"
df -h / | tail -1 | awk '{print "  disk free: " $4}'
free -g | awk '/^Mem:/ {print "  ram avail: " $NF " GB"}'
echo "Target DB: $(du -sh "$DB" | awk '{print $1}')"
echo "Output dir: $OUTDIR"
echo

# 1. Launch mdb server under SERIAL_SCAN=1 + SORTER=radix
echo "[$(date +%H:%M:%S)] starting mdb server (SERIAL_SCAN=1 + SORTER=radix)..."
MDB_PROJECTION_SERIAL_SCAN=1 MDB_PROJECTION_SORTER=radix \
    "$MDB" server "$DB" --port "$PORT" --timeout 14400 \
    > "$SRV_LOG" 2>&1 &
SRV_PID=$!
echo "server PID: $SRV_PID"

# Wait for server ready (max 30 s)
for i in $(seq 1 60); do
    if curl -sSf -o /dev/null --data-binary "RETURN 1" -H "Accept: text/csv" \
            "http://127.0.0.1:$PORT/" 2>/dev/null; then
        echo "[$(date +%H:%M:%S)] server ready."
        break
    fi
    sleep 0.5
done

# 2. Launch PSI safety monitor on server PID (18 GB ceiling, see header)
echo "[$(date +%H:%M:%S)] arming PSI safety monitor (threshold: 18 GB RSS, 5% psi, 50 MB/s swap)..."
THRESHOLD_RSS_GB=18 THRESHOLD_PSI=5.0 THRESHOLD_SWAP_MB_S=50 \
    ./scripts/psi_safety_monitor.sh "$SRV_PID" > "$PSI_LOG" 2>&1 &
PSI_PID=$!

# 3. Launch projection POST in background (expected 120-180 min under serial)
echo "[$(date +%H:%M:%S)] launching graph_project POST (async, ~120-180 min expected)..."
(
    t0=$(date +%s.%N)
    curl -sS --data-binary \
        "CALL graph_project('$PROJ_NAME', 'Node', 'CITES', {orientation: 'NATURAL'}) YIELD graphName, nodeCount, relCount, projectMillis RETURN *" \
        -H "Accept: text/csv" "http://127.0.0.1:$PORT/" \
        > "$POST_LOG" 2>&1
    t1=$(date +%s.%N)
    awk -v t0="$t0" -v t1="$t1" 'BEGIN { printf "wall_clock_seconds=%.3f\n", t1-t0 }' >> "$POST_LOG"
) &
POST_PID=$!

# 4. Sample RSS + disk at 30 s cadence until POST finishes or server exits
echo "ts,srv_pid,peak_rss_mb,cur_rss_mb,disk_free_gb,psi_some_pct" > "$SAMPLE_CSV"
echo "[$(date +%H:%M:%S)] sampling every 30 s (ctrl-c to abort)..."
echo "--- SAMPLES ---"

while kill -0 "$POST_PID" 2>/dev/null && kill -0 "$SRV_PID" 2>/dev/null; do
    ts=$(date +%H:%M:%S)
    hwm=$(awk '/^VmHWM:/ {print $2}' "/proc/$SRV_PID/status" 2>/dev/null || echo 0)
    rss=$(awk '/^VmRSS:/ {print $2}' "/proc/$SRV_PID/status" 2>/dev/null || echo 0)
    free_gb=$(df -BG / | tail -1 | awk '{sub(/G/,"",$4); print $4}')
    psi=$(awk '/^some/ { for (i=1;i<=NF;i++) if ($i ~ /^avg10=/) { split($i,a,"="); print a[2] } }' /proc/pressure/memory 2>/dev/null || echo "0")
    peak_mb=$((hwm / 1024))
    cur_mb=$((rss / 1024))
    printf "[%s] peak=%s MB  cur=%s MB  disk_free=%sGB  psi=%s%%\n" \
        "$ts" "$peak_mb" "$cur_mb" "$free_gb" "$psi"
    echo "$ts,$SRV_PID,$peak_mb,$cur_mb,$free_gb,$psi" >> "$SAMPLE_CSV"
    sleep 30
done

echo "--- END SAMPLES ---"
echo "[$(date +%H:%M:%S)] POST exited, collecting results..."

# 5. Capture final state
final_peak_kb=$(awk '/^VmHWM:/ {print $2}' "/proc/$SRV_PID/status" 2>/dev/null || echo 0)
final_peak_mb=$((final_peak_kb / 1024))
post_output=$(cat "$POST_LOG")

echo
echo "=== RESULT ==="
echo "server PID: $SRV_PID"
echo "peak RSS: $final_peak_mb MB"
echo "POST output:"
echo "$post_output" | sed 's/^/  /'

# Kill PSI monitor + server
kill "$PSI_PID" 2>/dev/null || true
kill "$SRV_PID" 2>/dev/null || true
wait 2>/dev/null || true

echo "=== logs ==="
ls -lh "$OUTDIR"

# 6. Drop projection so repeat runs have clean state
"$MDB" drop-projection "$DB" "$PROJ_NAME" > /dev/null 2>&1 || true

echo "=== complete ==="
