#!/usr/bin/env bash
# papers100m_bloomfix_rebuild_v5.sh — RADIX backend (bounded by construction) +
# per-process RSS logging to settle MDB-OOM-vs-exogenous (celebi is shared; the
# previous watchers logged SYSTEM MemAvailable, not mdb's RSS). If mdb RSS stays
# low while system RAM crashes -> the OOM is another tenant, not graph_project.
# RADIX avoids the CLASSIC ExternalRecordSort path entirely and is byte-identical
# (BITSET+BTREE, 20/20 on cora_gnn). No swap.
set -u -o pipefail
REPO=/home/bfuentes/MillenniumDB_Testing
MDB="$REPO/build/Release/bin/mdb"
DB="$REPO/data/dbs/gql/papers100M"
PROJ=papers100M_e2e_opt
PROJDIR="$DB/projections/$PROJ"
SAMPLE=e2e5ep
PORT=7897
OUT="$REPO/docs/research/2026-06-15-bloomfix-rebuild"
mkdir -p "$OUT"
RES="$OUT/rebuild_v5_results.txt"; SRVLOG="$OUT/server_v5.log"; WLOG="$OUT/watch_v5.log"
SRVPID=""; WPID=""
cleanup(){ [ -n "$WPID" ] && kill "$WPID" 2>/dev/null||true; [ -n "$SRVPID" ] && kill "$SRVPID" 2>/dev/null||true; }
trap cleanup EXIT INT TERM
log(){ echo "[rb5 $(date '+%m-%d %H:%M:%S')] $*" | tee -a "$RES"; }
now(){ date +%s; }
diskG(){ df -BG --output=avail "$REPO" | tail -1 | tr -dc 0-9; }
memMB(){ awk '/MemAvailable/{print int($2/1024)}' /proc/meminfo; }
gql(){ curl -s --max-time 20000 -X POST "http://localhost:$PORT/gql" -H "Content-Type: text/plain" --data-binary "$1"; }
chk(){ echo "$1" | grep -qiE "error|exception|failed|abort|out of range|not found" && { log "  BODY: $1"; log "  STAGE FAILED"; return 1; }; return 0; }
nonempty(){ [ -n "$(echo "$1" | tr -d '[:space:]')" ] || { log "  EMPTY RESPONSE — server died"; return 1; }; }

export MDB_GNN_TOPOLOGY_UINT32=1
export MDB_GNN_REORDER_STRATEGY=external_sort
export MDB_PROJECTION_SORTER=radix   # bounded-by-construction backend (workflow recommendation)
export MDB_SORT_BUFFER_MB=4096

: > "$RES"
log "Phase 0: rm -rf partial projection dir (keep node_features.fmat)"
rm -rf "$PROJDIR"
log "disk after clean: $(diskG)G free; mem $(memMB)MB"

for p in $(pgrep -f "bin/mdb server .*-p $PORT" 2>/dev/null||true); do kill -9 "$p" 2>/dev/null||true; done
sleep 1
MDB_GNN_TOPOLOGY_UINT32=1 MDB_PROJECTION_SORTER=radix MDB_SORT_BUFFER_MB=4096 "$MDB" server "$DB" -p $PORT -t 20000 --browser false --versioned-buffer 3GB >>"$SRVLOG" 2>&1 & SRVPID=$!
ready=0
for _ in $(seq 1 240); do gql "RETURN 1" 2>/dev/null | grep -q 1 && { ready=1; break; }; kill -0 "$SRVPID" 2>/dev/null||{ log "server died on startup"; exit 1; }; sleep 1; done
[ "$ready" = 1 ] || { log "server not ready"; exit 1; }
log "server up pid=$SRVPID (RADIX backend)"

# Watcher: log BOTH system MemAvailable AND mdb's RSS (KB->MB) every 8s.
( while true; do
    rss=$(ps -o rss= -p "$SRVPID" 2>/dev/null | tr -dc 0-9); rss=${rss:-0}
    echo "$(date '+%H:%M:%S') sysMemAvailMB=$(memMB) mdbRSS_MB=$((rss/1024)) diskG=$(diskG)" >> "$WLOG"
    sleep 8
  done ) & WPID=$!

log "START v5 RADIX rebuild (RSS-logged, NO swap)"

log "Phase 2: graph_project (RADIX, bounded; Bloom fix => relCount RECOVERS)"
t0=$(now)
b=$(gql "CALL graph_project('$PROJ','Node','CITES',{orientation:'NATURAL',indexSet:'GNN_MINIMAL',buildTopologySnapshot:false,includeFeatures:'node_features',labelProperty:'label',splitProperty:'split'}) YIELD graphName,nodeCount,relationshipCount,featureDim,numClasses,projectMillis RETURN *")
nonempty "$b" || { log "graph_project died; mdb RSS at end vs system (watch_v5.log tail):"; tail -5 "$WLOG" | sed 's/^/    /' | tee -a "$RES"; exit 1; }
chk "$b" || exit 1
[ -s "$PROJDIR/from_to_edge.leaf" ] || { log "from_to_edge.leaf 0 bytes"; exit 1; }
log "GRAPH_PROJECT $(( $(now)-t0 ))s :: $b"
echo "$b" > "$OUT/graph_project_v5.txt"

log "Phase 3: gnn_build_topology_snapshot"
t0=$(now)
b=$(gql "CALL gnn_build_topology_snapshot('$PROJ') YIELD projectionName,fwdBytes,revBytes,durationMillis RETURN *")
nonempty "$b" || exit 1; chk "$b" || exit 1
[ -s "$PROJDIR/topology_fwd.csr" ] || { log "topology_fwd.csr missing"; exit 1; }
log "TOPO_SNAPSHOT $(( $(now)-t0 ))s :: $b"

log "Phase 4: gnn_offline_sample e2e5ep"
t0=$(now)
b=$(gql "CALL gnn_offline_sample('$PROJ','$SAMPLE',[10,15,20],{batchSize:1024,randomSeed:42,orientation:'UNDIRECTED',usePredefinedSplits:true,useFourLevelTopologyStore:true,useAdjacencyCache:true,useL3MmapSidecar:true,l1CacheMb:512,l2CacheMb:6144,numWorkers:16,force:true}) YIELD totalBatches,trainBatches,validationBatches,testBatches,uniqueNodes,numWorkersUsed,computeMillis,sampleContentFp RETURN *")
nonempty "$b" || exit 1; chk "$b" || exit 1
log "SAMPLE $(( $(now)-t0 ))s :: $b"
echo "$b" > "$OUT/sample_v5.txt"

log "Phase 5: gnn_build_feature_store + bakeBlocks"
t0=$(now)
b=$(gql "CALL gnn_build_feature_store('$SAMPLE','node_features',{gpu_budget_mb:2048,cpu_budget_mb:5290,buildAddrTables:true,bakeBlocks:true}) YIELD l1Nodes,l2Nodes,l3Nodes,l4Nodes,slimMb,totalDiskMb,addrTablesBuiltOk,blocksBuiltOk,buildTimeMs RETURN *")
nonempty "$b" || exit 1; chk "$b" || exit 1
log "BUILD $(( $(now)-t0 ))s :: $b"

log "Phase 6: gnn_train seed42 50ep N=8"
t0=$(now)
b=$(gql "CALL gnn_train('$SAMPLE','node_features',{model:'graphsage',hiddenDim:256,epochs:50,lr:0.001,dropout:0.2,normalize:false,patience:999,randomSeed:42,useAsyncPrefetcher:true,prefetchNumWorkers:8,prefetchQueueSize:8,sampleCacheMb:2048,exportEmbeddings:false,outputDir:'bloomfix50'}) YIELD testAccuracy,bestValAccuracy,ranEpochs,trainSeconds,l1HitRatio,l2HitRatio RETURN *")
nonempty "$b" || exit 1; chk "$b" || exit 1
log "TRAIN $(( $(now)-t0 ))s :: $b"
echo "$b" > "$OUT/train_result_v5.txt"

log "DONE v5 — relCount vs 1,612,996,613 ; testAcc vs 0.6406"
