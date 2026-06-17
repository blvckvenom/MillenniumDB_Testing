#!/usr/bin/env bash
# papers100m_edge_scan_profile.sh — FACTUAL edge-scan bottleneck split.
#
# Builds a throwaway projection with MDB_PROJECTION_EDGE_SCAN_PROFILE=1 (the
# env-gated timers added to scan_edges_impl_classic_), samples the mdb server's
# aggregate CPU% + running-thread count DURING the edge scan, captures the
# [EDGE_SCAN_PROFILE] summary line, then KILLS the build the moment the edge
# scan completes (we do NOT need the multi-minute finalize/leaf-write for this
# question). NEVER touches the canonical papers100M_e2e_opt store.
#
# Decision rule:
#   serial_consumer_ms ~>= 80% of edge_scan_wall  -> CONSUMER-bound: the serial
#       tail (detector + add_edge + flush) gates throughput; a serial-tail lever
#       (e.g. skip the redundant windowed detector, lean on sort-time unique)
#       pays off.  Cross-check: aggregate CPU stays ~100-300% during the scan.
#   serial_consumer_ms << edge_scan_wall -> PARALLEL-WALK-bound: the B+Tree walk
#       + endpoint resolution dominate; a serial-tail lever cannot help.
#       Cross-check: aggregate CPU ~1600%+ during the scan.
set -u -o pipefail
REPO=/home/bfuentes/MillenniumDB_Testing
MDB="$REPO/build/Release/bin/mdb"
DB="$REPO/data/dbs/gql/papers100M"
PROBE=papers100M_escan_probe
PROBEDIR="$DB/projections/$PROBE"
PORT=7899
OUT="$REPO/docs/research/2026-06-16-cpugpu-build"
mkdir -p "$OUT"
RES="$OUT/edge_scan_profile_results.txt"
SRVLOG="$OUT/edge_scan_profile_server.log"
CPULOG="$OUT/edge_scan_profile_cpu.log"
SRVPID=""; CPID=""
cleanup(){ for p in "$CPID" "$SRVPID"; do [ -n "$p" ] && kill "$p" 2>/dev/null || true; done
           for q in $(pgrep -f "bin/mdb server .*-p $PORT" 2>/dev/null||true); do kill -9 "$q" 2>/dev/null||true; done
           rm -rf "$PROBEDIR" 2>/dev/null || true; }
trap cleanup EXIT INT TERM
log(){ echo "[escan $(date '+%m-%d %H:%M:%S')] $*" | tee -a "$RES"; }
gql(){ curl -s --max-time 16000 -X POST "http://localhost:$PORT/gql" -H "Content-Type: text/plain" --data-binary "$1"; }

export MDB_GNN_TOPOLOGY_UINT32=1
export MDB_PROJECTION_SORTER=radix
export MDB_SORT_BUFFER_MB=4096
export MDB_PROJECTION_EDGE_SCAN_PROFILE=1   # the timers added this session

: > "$RES"; : > "$SRVLOG"; : > "$CPULOG"   # truncate ALL logs — else a stale [EDGE_SCAN_PROFILE] line from a prior run is matched instantly and the build is killed before it scans
log "START edge-scan profile: probe=$PROBE port=$PORT (profile timers ON)"
rm -rf "$PROBEDIR"
for p in $(pgrep -f "bin/mdb server .*-p $PORT" 2>/dev/null||true); do kill -9 "$p" 2>/dev/null||true; done
sleep 1
"$MDB" server "$DB" -p $PORT -t 16000 --browser false --versioned-buffer 3GB >>"$SRVLOG" 2>&1 & SRVPID=$!
ready=0
for _ in $(seq 1 240); do gql "RETURN 1" 2>/dev/null | grep -q 1 && { ready=1; break; }; kill -0 "$SRVPID" 2>/dev/null||{ log "server died on startup"; exit 1; }; sleep 1; done
[ "$ready" = 1 ] || { log "server not ready"; exit 1; }
log "server up pid=$SRVPID"

# CPU sampler: every 3s log aggregate %CPU (can exceed 100 = multi-core) +
# running-thread count. The edge-scan phase is identifiable as the window AFTER
# the node scan and BEFORE any .leaf file appears.
( while kill -0 "$SRVPID" 2>/dev/null; do
    pcpu=$(ps -o %cpu= -p "$SRVPID" 2>/dev/null | tr -d ' '); pcpu=${pcpu:-NA}
    rth=$(ps -L -o stat= -p "$SRVPID" 2>/dev/null | grep -c '^R')
    leaf=$( [ -e "$PROBEDIR/nodes.leaf" ] && echo "POSTSCAN" || echo "scanning" )
    echo "$(date '+%H:%M:%S') aggCPU%=$pcpu runThreads=$rth phase=$leaf" >> "$CPULOG"
    sleep 3
  done ) & CPID=$!

# Fire graph_project in the background so we can watch for the PROFILE line and
# kill right after the edge scan (skip the long finalize).
log "graph_project (RADIX, profile ON) into $PROBE — will kill after edge scan"
( gql "CALL graph_project('$PROBE','Node','CITES',{orientation:'NATURAL',indexSet:'GNN_MINIMAL',buildTopologySnapshot:false}) YIELD graphName,nodeCount,relationshipCount RETURN *" >>"$SRVLOG" 2>&1 ) &

# Wait for the [EDGE_SCAN_PROFILE] summary (printed at end of scan_edges_impl_classic_),
# or bail if the server dies.
seen=0
for _ in $(seq 1 4800); do   # up to 4h
  if grep -q "\[EDGE_SCAN_PROFILE\] edges_to_consumer" "$SRVLOG" 2>/dev/null; then seen=1; break; fi
  kill -0 "$SRVPID" 2>/dev/null || { log "server died before edge scan finished"; tail -8 "$SRVLOG" | sed 's/^/    /' | tee -a "$RES"; exit 1; }
  sleep 3
done
[ "$seen" = 1 ] || { log "edge-scan profile line not seen within budget"; exit 1; }

log "=== EDGE-SCAN PROFILE (factual split) ==="
grep "\[EDGE_SCAN_PROFILE\]" "$SRVLOG" | sed 's/^/    /' | tee -a "$RES"
log "=== aggregate CPU% during the scanning window (phase=scanning rows) ==="
grep "phase=scanning" "$CPULOG" 2>/dev/null | tail -25 | sed 's/^/    /' | tee -a "$RES"
log "peak aggregate CPU% while scanning:"
awk '/phase=scanning/{split($2,a,"="); c=a[2]+0; if(c>m)m=c} END{print "    peak_aggCPU%="m}' "$CPULOG" 2>/dev/null | tee -a "$RES"

log "killing build right after edge scan (skip finalize) + dropping probe"
# cleanup() handles server kill + probe rm on EXIT
log "DONE edge-scan profile"
