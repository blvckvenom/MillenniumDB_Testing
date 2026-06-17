#!/usr/bin/env bash
# papers100m_cpugpu_build_measure.sh — validate Phases 4/5/6 at papers100M scale.
# Builds a THROWAWAY projection papers100M_cpugpu_probe (NEVER touches the canonical
# papers100M_e2e_opt) with RADIX (GPU-eligible) + leaf compression default-on, while
# sampling GPU% / CPU% / mdb RSS, then measures edge-leaf compression vs the canonical
# uncompressed baseline and the [RADIX] gpu/cpu partition split. Drops the probe at end.
set -u -o pipefail
REPO=/home/bfuentes/MillenniumDB_Testing
MDB="$REPO/build/Release/bin/mdb"
DB="$REPO/data/dbs/gql/papers100M"
PROBE=papers100M_cpugpu_probe
PROBEDIR="$DB/projections/$PROBE"
CANON="$DB/projections/papers100M_e2e_opt"   # READ-ONLY baseline, never modified
PORT=7898
OUT="$REPO/docs/research/2026-06-16-cpugpu-build"
mkdir -p "$OUT"
RES="$OUT/measure_results.txt"; SRVLOG="$OUT/server_measure.log"
GPULOG="$OUT/gpu_util.log"; RSSLOG="$OUT/rss_watch.log"
SRVPID=""; GPID=""; WPID=""
cleanup(){ for p in "$GPID" "$WPID" "$SRVPID"; do [ -n "$p" ] && kill "$p" 2>/dev/null || true; done; }
trap cleanup EXIT INT TERM
log(){ echo "[meas $(date '+%m-%d %H:%M:%S')] $*" | tee -a "$RES"; }
now(){ date +%s; }
diskG(){ df -BG --output=avail "$REPO" | tail -1 | tr -dc 0-9; }
memMB(){ awk '/MemAvailable/{print int($2/1024)}' /proc/meminfo; }
gql(){ curl -s --max-time 16000 -X POST "http://localhost:$PORT/gql" -H "Content-Type: text/plain" --data-binary "$1"; }
chk(){ echo "$1" | grep -qiE "error|exception|failed|abort|out of range|not found" && { log "  BODY: $1"; log "  STAGE FAILED"; return 1; }; return 0; }
nonempty(){ [ -n "$(echo "$1" | tr -d '[:space:]')" ] || { log "  EMPTY RESPONSE — server died"; return 1; }; }

export MDB_GNN_TOPOLOGY_UINT32=1
export MDB_PROJECTION_SORTER=radix      # GPU-eligible per-partition path
export MDB_SORT_BUFFER_MB=4096          # bounded
# leaf compression is default-on (Phase 6); do NOT set MDB_PROJECTION_NO_LEAF_COMPRESSION

: > "$RES"
log "START measure: RADIX + compression(default-on), probe=$PROBE; disk $(diskG)G free, mem $(memMB)MB"
log "canonical baseline (uncompressed, read-only): $(ls -l --block-size=M "$CANON/from_to_edge.leaf" 2>/dev/null | awk '{print $5}')  from_to_edge.leaf"

rm -rf "$PROBEDIR"   # only the probe, never the canonical
for p in $(pgrep -f "bin/mdb server .*-p $PORT" 2>/dev/null||true); do kill -9 "$p" 2>/dev/null||true; done
sleep 1
MDB_GNN_TOPOLOGY_UINT32=1 MDB_PROJECTION_SORTER=radix MDB_SORT_BUFFER_MB=4096 \
  "$MDB" server "$DB" -p $PORT -t 16000 --browser false --versioned-buffer 3GB >>"$SRVLOG" 2>&1 & SRVPID=$!
ready=0
for _ in $(seq 1 240); do gql "RETURN 1" 2>/dev/null | grep -q 1 && { ready=1; break; }; kill -0 "$SRVPID" 2>/dev/null||{ log "server died on startup"; exit 1; }; sleep 1; done
[ "$ready" = 1 ] || { log "server not ready"; exit 1; }
log "server up pid=$SRVPID"

# Concurrent samplers (GPU util every 2s, mdb RSS + sysMem + disk every 3s).
( while true; do echo "$(date '+%H:%M:%S') $(nvidia-smi --query-gpu=utilization.gpu,utilization.memory,memory.used --format=csv,noheader,nounits 2>/dev/null | head -1)" >> "$GPULOG"; sleep 2; done ) & GPID=$!
( while true; do rss=$(ps -o rss= -p "$SRVPID" 2>/dev/null|tr -dc 0-9); rss=${rss:-0}; echo "$(date '+%H:%M:%S') mdbRSS_MB=$((rss/1024)) sysMemAvailMB=$(memMB) diskG=$(diskG)" >> "$RSSLOG"; sleep 3; done ) & WPID=$!

log "graph_project (RADIX + compression) into $PROBE"
t0=$(now)
b=$(gql "CALL graph_project('$PROBE','Node','CITES',{orientation:'NATURAL',indexSet:'GNN_MINIMAL',buildTopologySnapshot:false}) YIELD graphName,nodeCount,relationshipCount,projectMillis RETURN *")
nonempty "$b" || { log "graph_project died; server tail:"; tail -6 "$SRVLOG" | sed 's/^/    /' | tee -a "$RES"; exit 1; }
chk "$b" || exit 1
[ -s "$PROBEDIR/from_to_edge.leaf" ] || { log "from_to_edge.leaf 0 bytes"; exit 1; }
log "GRAPH_PROJECT $(( $(now)-t0 ))s :: $b"

# Stop samplers before measuring.
kill "$GPID" "$WPID" 2>/dev/null || true; GPID=""; WPID=""

log "=== RESULTS ==="
log "probe leaf sizes (COMPRESSED):"
ls -l --block-size=M "$PROBEDIR"/{from_to_edge,to_from_edge,nodes,node_label,label_node}.leaf 2>/dev/null | awk '{print "    "$5"  "$9}' | tee -a "$RES"
probe_ft=$(stat -c%s "$PROBEDIR/from_to_edge.leaf" 2>/dev/null||echo 0)
canon_ft=$(stat -c%s "$CANON/from_to_edge.leaf" 2>/dev/null||echo 0)
if [ "$canon_ft" -gt 0 ]; then
  log "from_to_edge.leaf: compressed=$((probe_ft/1024/1024))M vs canonical-uncompressed=$((canon_ft/1024/1024))M  ratio=$(awk "BEGIN{printf \"%.3f\", $probe_ft/$canon_ft}")x"
fi
log "GPU/CPU partition split ([RADIX] lines from server log):"
grep "\[RADIX\] partitions sorted" "$SRVLOG" 2>/dev/null | tail -20 | sed 's/^/    /' | tee -a "$RES"
log "peak GPU util (top samples):"
sort -t, -k1 -rn "$GPULOG" 2>/dev/null | head -3 | sed 's/^/    /' | tee -a "$RES" || true
log "peak mdb RSS (max):"
awk -F'mdbRSS_MB=' '{split($2,a," "); if(a[1]+0>m)m=a[1]} END{print "    peak mdbRSS_MB="m}' "$RSSLOG" 2>/dev/null | tee -a "$RES"
log "min sysMemAvail (tightest):"
awk -F'sysMemAvailMB=' '{split($2,a," "); if(min==""||a[1]+0<min)min=a[1]} END{print "    min sysMemAvailMB="min}' "$RSSLOG" 2>/dev/null | tee -a "$RES"

log "dropping probe projection $PROBE (reclaim disk)"
gql "CALL graph_drop('$PROBE')" 2>&1 | tail -1 | sed 's/^/    /' | tee -a "$RES"
rm -rf "$PROBEDIR" 2>/dev/null || true
log "disk after drop: $(diskG)G free"
log "DONE measure"
