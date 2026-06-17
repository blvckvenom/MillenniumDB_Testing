#!/usr/bin/env bash
# papers100m_rebuild_store.sh — rebuild the e2e5ep FourLevelStore derivatives
# (reordered.fmat / L1+L2 caches / packed_slim / addr_tables / meta) which were
# overwritten by a different sample's build (e2e5ep_sl self-loop ablation), then
# re-bake the self-contained blocks against the fresh store_fp.
#
# The source node_features.fmat + .rmap are INTACT (only the derived store is
# stale). force:true regenerates all derivatives for e2e5ep; a second call with
# bakeBlocks + force_*:false re-bakes the v2 self-contained blocks (the
# documented post-force step — otherwise blocks stay stale and train falls back
# to online / zero-fill). Then a 1-epoch sanity train confirms no zero-fill.
set -u -o pipefail
REPO=/home/bfuentes/MillenniumDB_Testing
MDB="$REPO/build/Release/bin/mdb"; DB="$REPO/data/dbs/gql/papers100M"
SAMPLE="${SAMPLE:-e2e5ep}"; PORT="${PORT:-7907}"
OUT="$REPO/docs/research/2026-06-16-accuracy-target"; mkdir -p "$OUT"
RES="$OUT/rebuild_store.txt"; SRVLOG="$OUT/rebuild_store_srv.log"; : > "$RES"; : > "$SRVLOG"
log(){ echo "[rebuild $(date +%H:%M:%S)] $*" | tee -a "$RES"; }
gql(){ curl -s --max-time 172800 -X POST "http://localhost:$PORT/gql" -H "Content-Type: text/plain" --data-binary "$1"; }
chk(){ echo "$1" | grep -qiE "error|exception|failed" && { echo "BODY: $1" | tee -a "$RES"; return 1; }; return 0; }
SRV=""
trap '[ -n "$SRV" ] && kill "$SRV" 2>/dev/null; [ -n "$SRV" ] && wait "$SRV" 2>/dev/null' EXIT INT TERM
for p in $(pgrep -f "bin/mdb server .*-p $PORT" 2>/dev/null||true); do kill -9 "$p" 2>/dev/null; done; sleep 2
"$MDB" server "$DB" -p $PORT -t 172800 --browser false --versioned-buffer 3GB >>"$SRVLOG" 2>&1 & SRV=$!
for _ in $(seq 1 240); do gql "RETURN 1" 2>/dev/null|grep -q 1 && break; kill -0 $SRV 2>/dev/null||{ log "server died"; exit 1; }; sleep 1; done
log "server up pid=$SRV; rebuilding store for $SAMPLE (force full)"

log "STEP 1/3: full rebuild (force:true, gpu 2048 / cpu 5290, buildAddrTables) — ~31 min"
B1=$(gql "CALL gnn_build_feature_store('$SAMPLE','node_features',{force:true,gpu_budget_mb:2048,cpu_budget_mb:5290,buildAddrTables:true}) YIELD l1Nodes,l2Nodes,l3Nodes,l4Nodes,slimMb,reorderedMb,totalDiskMb RETURN *")
chk "$B1" || { log "STEP 1 FAILED"; exit 1; }
log "STEP 1 done: $(echo "$B1"|tail -1)"

log "STEP 2/3: re-bake self-contained blocks (bakeBlocks, force_*:false) — ~4 min"
B2=$(gql "CALL gnn_build_feature_store('$SAMPLE','node_features',{bakeBlocks:true,force_caches:false,force_reorder:false,force_packed_slim:false,force_meta:false}) YIELD blocksMb,blocksBuiltOk RETURN *")
chk "$B2" || { log "STEP 2 FAILED"; exit 1; }
log "STEP 2 done: $(echo "$B2"|tail -1)"

log "STEP 3/3: 1-epoch sanity train (must be self-contained, NO zero-fill)"
T=$(gql "CALL gnn_train('$SAMPLE','node_features',{model:'graphsage',hiddenDim:256,epochs:1,lr:0.001,dropout:0.2,patience:999,randomSeed:42,prefetchNumWorkers:8,prefetchQueueSize:8,sampleCacheMb:2048,saveOnBestVal:false,saveFinal:false,exportEmbeddings:false}) YIELD testAccuracy,bestValAccuracy,ranEpochs RETURN *")
chk "$T" || { log "STEP 3 train FAILED"; exit 1; }
ZF=$(grep -c "ZERO-FILLED" "$SRVLOG" 2>/dev/null || echo 0)
MODE=$(grep -i "feature-load mode" "$SRVLOG" | tail -1)
log "STEP 3 done: $(echo "$T"|tail -1)"
log "  feature-load: $MODE"
log "  ZERO-FILLED warnings in srv log: $ZF  (MUST be 0)"
if [ "$ZF" = "0" ]; then log "REBUILD OK — store healthy for $SAMPLE"; else log "REBUILD STILL BROKEN — $ZF zero-fill warnings"; fi
