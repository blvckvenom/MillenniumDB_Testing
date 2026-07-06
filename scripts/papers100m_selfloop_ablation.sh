#!/usr/bin/env bash
# papers100m_selfloop_ablation.sh — FACTUAL test of the -1p accuracy-gap candidate.
#
# Hypothesis (GAP_REOPEN_2026-06-16): DiskGNN's papers100M graph prep does
# dgl.add_self_loop, so each node aggregates ITSELF in the neighbor-mean; MDB does not
# (self handled separately by SAGEConv's CONCAT). This re-runs the EXACT canonical
# pipeline (e_converge50_clean = testAcc 0.6406 / bestVal 0.6742) changing ONLY
# MDB_GNN_SAMPLE_SELF_LOOP=1 (the staged add_self_loop emulation in the sampler).
#
# A rise toward ~0.65 => self-loop IS the residual cause (gap factually closed).
# No movement => ruled out; correct the prior doc + attribute to deeper diffs / variance.
#
# NEVER modifies the canonical e2e5ep sample/store or papers100M_e2e_opt projection
# (writes a NEW sample e2e5ep_sl + its own store). ~75 min.
set -u -o pipefail
REPO=/home/bfuentes/MillenniumDB_Testing
MDB="$REPO/build/Release/bin/mdb"; DB="$REPO/data/dbs/gql/papers100M"
PROJ=papers100M_e2e_opt; SAMPLE=e2e5ep_sl; PORT="${PORT:-7915}"
OUT="$REPO/docs/research/2026-06-10-thesis-measurements/selfloop_ablation"; mkdir -p "$OUT"
RES="$OUT/RESULT.txt"; SRVLOG="$OUT/server.log"
: > "$RES"; : > "$SRVLOG"
log(){ echo "[sl $(date +%H:%M:%S)] $*" | tee -a "$RES"; }
gql(){ curl -s --max-time 172800 -X POST "http://localhost:$PORT/gql" -H "Content-Type: text/plain" --data-binary "$1"; }
err(){ echo "$1" | grep -qiE "error|exception|abort|not found|out of range" && { echo "  BODY: $1" | tee -a "$RES"; return 1; }; return 0; }
SRV=""; trap '[ -n "$SRV" ] && kill -9 "$SRV" 2>/dev/null' EXIT INT TERM
for p in $(pgrep -f "bin/mdb server .*-p $PORT" 2>/dev/null||true); do kill -9 "$p" 2>/dev/null; done; sleep 2

# Self-loop ON for the SERVER (the static flag is read at first sample call).
export MDB_GNN_SAMPLE_SELF_LOOP=1
export MDB_GNN_TOPOLOGY_UINT32=1
log "START self-loop ablation: SAMPLE=$SAMPLE  MDB_GNN_SAMPLE_SELF_LOOP=1  (canonical baseline: testAcc 0.6406 / bestVal 0.6742)"
log "disk free: $(df -BG --output=avail "$REPO"|tail -1|tr -dc 0-9)G"

"$MDB" server "$DB" -p $PORT -t 172800 --browser false --versioned-buffer 3GB >>"$SRVLOG" 2>&1 & SRV=$!
for _ in $(seq 1 240); do gql "RETURN 1" 2>/dev/null|grep -q 1 && break; kill -0 $SRV 2>/dev/null||{ log "server died"; exit 1; }; sleep 1; done
log "server up pid=$SRV (self-loop env active)"

# ---- Phase A: re-sample with self-loop (same config as canonical e2e5ep) ----
log "Phase A: gnn_offline_sample $SAMPLE [10,15,20] UNDIRECTED self-loop ON  (ETA ~3-5 min)"
tA=$(date +%s)
b=$(gql "CALL gnn_offline_sample('$PROJ','$SAMPLE',[20,15,10],{batchSize:1024,randomSeed:42,orientation:'UNDIRECTED',usePredefinedSplits:true,useFourLevelTopologyStore:true,useAdjacencyCache:true,useL3MmapSidecar:true,l1CacheMb:512,l2CacheMb:6144,numWorkers:16,force:true}) YIELD totalBatches,trainBatches,validationBatches,testBatches,uniqueNodes,numWorkersUsed,computeMillis,sampleContentFp RETURN *")
err "$b" || { log "Phase A FAILED"; exit 1; }
log "  A done $(( $(date +%s)-tA ))s :: $b"
log "  SANITY: totalBatches/uniqueNodes should match canonical (1512 / ~48.7M); sampleContentFp MUST differ from e2e5ep (self-edges) — proves the flag engaged."

# ---- Phase B: build feature store + bake blocks ----
log "Phase B: gnn_build_feature_store $SAMPLE + bakeBlocks  (ETA ~31 min)"
tB=$(date +%s)
b=$(gql "CALL gnn_build_feature_store('$SAMPLE','node_features',{gpu_budget_mb:2048,cpu_budget_mb:5290,buildAddrTables:true,bakeBlocks:true}) YIELD l1Nodes,l2Nodes,l3Nodes,l4Nodes,slimMb,totalDiskMb,addrTablesBuiltOk,blocksBuiltOk,buildTimeMs RETURN *")
err "$b" || { log "Phase B FAILED"; exit 1; }
log "  B done $(( $(date +%s)-tB ))s :: $b"
log "  disk free after build: $(df -BG --output=avail "$REPO"|tail -1|tr -dc 0-9)G"

# ---- Phase C: train 50 epochs (EXACT canonical converge50 HP) ----
log "Phase C: gnn_train 50ep (lr0.001 dropout0.2 hidden256 seed42 N=8)  (ETA ~41 min)"
tC=$(date +%s)
HP="model:'graphsage',hiddenDim:256,epochs:50,lr:0.001,dropout:0.2,normalize:false,patience:999,randomSeed:42,useAsyncPrefetcher:true,prefetchNumWorkers:8,prefetchQueueSize:8,sampleCacheMb:2048,saveOnBestVal:true,saveFinal:true,outputDir:'sl_converge50',exportEmbeddings:false"
b=$(gql "CALL gnn_train('$SAMPLE','node_features',{$HP}) YIELD ranEpochs,trainSeconds,bestValAccuracy,testAccuracy RETURN *")
err "$b" || log "  (train error body; check below)"
log "  C done $(( $(date +%s)-tC ))s"
grep -i "feature-load mode" "$SRVLOG" | tail -1 | sed 's/^/  [mode] /' | tee -a "$RES"
log "  YIELDS: $(echo "$b"|tail -1)"

log "=== VERDICT ==="
log "  canonical (self-loop OFF): testAcc 0.6406 / bestVal 0.6742"
log "  self-loop ON:              $(echo "$b"|tail -1)"
log "  -> rise toward ~0.65 = self-loop IS the residual; flat = ruled out."
log "DONE self-loop ablation."
