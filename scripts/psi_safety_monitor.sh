#!/bin/bash
# PSI-based safety monitor for long-running mdb projection builds.
#
# Watches /proc/pressure/memory (PSI) and /proc/PID/status (VmHWM + VmRSS)
# at 2-second cadence. Aborts the target process if ANY of:
#
#   1. PSI memory "some avg10" > threshold_psi (default 5.0%).
#      Interpretation: for ≥10% of the last 10 s, tasks have been stalled
#      waiting for memory. This is the earliest signal that kernel reclaim
#      is about to start harvesting user-app pages. Aborting mdb here
#      preserves interactive apps.
#
#   2. Process RSS > threshold_rss_gb (default 15 GB).
#      Our radix backend's algorithmic bound is ~2.5 GB. Anything over
#      15 GB indicates a design violation (glibc retention, unbounded
#      buffer, regression) and we abort for investigation.
#
#   3. Swap usage delta > threshold_swap_mb_s (default 50 MB/s sustained
#      for 5 s). Indicates swap thrashing onset, which precedes system-wide
#      unresponsiveness.
#
# On abort: SIGTERM first; if process hasn't exited after 5 s, SIGKILL.
# Logs all decisions to /tmp/psi_monitor_PID.log.
#
# Usage:   ./scripts/psi_safety_monitor.sh <target_pid>
# Env:     THRESHOLD_PSI=5.0 THRESHOLD_RSS_GB=15 THRESHOLD_SWAP_MB_S=50
#
# This replaces the need for `systemd-run --user --scope -p MemoryMax=20G`
# as the safety mechanism: instead of hard-limiting memory (which kills
# mdb cleanly but reveals nothing about why), we monitor multiple early
# indicators and abort with diagnostic context.
set -u

TARGET_PID="${1:-}"
if [[ -z "$TARGET_PID" ]]; then
    echo "Usage: $0 <target_pid>" >&2
    exit 2
fi
if ! kill -0 "$TARGET_PID" 2>/dev/null; then
    echo "Process $TARGET_PID not running." >&2
    exit 2
fi

THRESHOLD_PSI="${THRESHOLD_PSI:-5.0}"
THRESHOLD_RSS_GB="${THRESHOLD_RSS_GB:-15}"
THRESHOLD_SWAP_MB_S="${THRESHOLD_SWAP_MB_S:-50}"
CADENCE="${CADENCE:-2}"
LOG="/tmp/psi_monitor_${TARGET_PID}.log"

echo "=== PSI safety monitor for PID $TARGET_PID ===" | tee "$LOG"
echo "thresholds: psi=${THRESHOLD_PSI}%  rss=${THRESHOLD_RSS_GB} GB  swap_rate=${THRESHOLD_SWAP_MB_S} MB/s" | tee -a "$LOG"
echo "cadence: ${CADENCE}s  log: $LOG" | tee -a "$LOG"
echo | tee -a "$LOG"

abort_reason=""
prev_swap_kb=$(awk '/^SwapCached:/ {print $2}' /proc/meminfo)
prev_ts=$(date +%s)
sustained_swap=0

while kill -0 "$TARGET_PID" 2>/dev/null; do
    ts=$(date +%s)
    elapsed=$((ts - prev_ts))

    # --- PSI memory pressure ---
    psi_some=$(awk '/^some/ { for (i=1;i<=NF;i++) if ($i ~ /^avg10=/) { split($i,a,"="); print a[2] } }' /proc/pressure/memory 2>/dev/null || echo "0")
    psi_num=$(awk -v x="$psi_some" 'BEGIN { print (x+0.0) }')

    # --- Process RSS (VmHWM = peak RSS, VmRSS = current) ---
    vmrss_kb=$(awk '/^VmRSS:/ {print $2}' /proc/$TARGET_PID/status 2>/dev/null || echo "0")
    vmhwm_kb=$(awk '/^VmHWM:/ {print $2}' /proc/$TARGET_PID/status 2>/dev/null || echo "0")
    rss_gb=$(awk -v kb="$vmrss_kb" 'BEGIN { printf "%.2f", kb/1024/1024 }')
    hwm_gb=$(awk -v kb="$vmhwm_kb" 'BEGIN { printf "%.2f", kb/1024/1024 }')

    # --- Swap delta ---
    swap_kb=$(awk '/^SwapCached:/ {print $2}' /proc/meminfo)
    swap_delta_kb=$((swap_kb - prev_swap_kb))
    if [[ $elapsed -gt 0 ]]; then
        swap_rate_mb_s=$(awk -v dk="$swap_delta_kb" -v e="$elapsed" 'BEGIN { printf "%.1f", (dk/1024)/e }')
    else
        swap_rate_mb_s="0.0"
    fi
    prev_swap_kb=$swap_kb
    prev_ts=$ts

    printf "[%s] psi=%5s%%  rss=%6s GB  peak=%6s GB  swap=%6s MB/s\n" \
        "$(date +%H:%M:%S)" "$psi_num" "$rss_gb" "$hwm_gb" "$swap_rate_mb_s" | tee -a "$LOG"

    # --- Abort checks ---
    if awk "BEGIN { exit ($psi_num > $THRESHOLD_PSI) ? 0 : 1 }"; then
        abort_reason="PSI memory pressure ${psi_num}% > ${THRESHOLD_PSI}%"
        break
    fi
    if awk "BEGIN { exit ($rss_gb > $THRESHOLD_RSS_GB) ? 0 : 1 }"; then
        abort_reason="RSS ${rss_gb} GB > ${THRESHOLD_RSS_GB} GB (algorithmic bound ~2.5 GB)"
        break
    fi
    if awk "BEGIN { exit ($swap_rate_mb_s > $THRESHOLD_SWAP_MB_S) ? 0 : 1 }"; then
        sustained_swap=$((sustained_swap + CADENCE))
        if [[ $sustained_swap -ge 5 ]]; then
            abort_reason="sustained swap rate ${swap_rate_mb_s} MB/s > ${THRESHOLD_SWAP_MB_S} MB/s for ${sustained_swap}s"
            break
        fi
    else
        sustained_swap=0
    fi

    sleep "$CADENCE"
done

if [[ -n "$abort_reason" ]]; then
    echo | tee -a "$LOG"
    echo "!!! ABORT !!! reason: $abort_reason" | tee -a "$LOG"
    echo "sending SIGTERM to $TARGET_PID..." | tee -a "$LOG"
    kill -TERM "$TARGET_PID" 2>/dev/null || true
    for i in 1 2 3 4 5; do
        kill -0 "$TARGET_PID" 2>/dev/null || { echo "exited cleanly." | tee -a "$LOG"; exit 1; }
        sleep 1
    done
    echo "still running after 5 s; SIGKILL." | tee -a "$LOG"
    kill -KILL "$TARGET_PID" 2>/dev/null || true
    exit 1
else
    echo | tee -a "$LOG"
    echo "target $TARGET_PID exited on its own. monitor done." | tee -a "$LOG"
    exit 0
fi
