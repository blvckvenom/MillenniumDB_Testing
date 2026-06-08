#!/usr/bin/env bash
# papers100m_packed_full.sh — same-day A/B bench for the Packed-Full Feature Store
# (Task 7). Mirrors the PROVEN harness in papers100m_selfcontained_measure.sh.
#
# Flow (papers100M e2e5ep):
#   PHASE 1   — measure the current v2 / self-contained baseline (4-tier intact),
#               N=1 sequential + N=8, in-session (server session 1).
#   PHASE 1.5 — DESTRUCTIVE (guarded): delete the regenerable 4-tier (reordered
#               fmat/rmap/rmap.idx + packed_slim + addr_tables) to free disk.
#               NEVER deletes node_features.{rmap,rmap.idx,fmat}, blocks/,
#               store.meta, or the gpu/cpu caches (projection prerequisites).
#   PHASE 2   — build the packed-full pack ADDITIVELY (server session 2), then
#               measure N=1 sequential + N=8. Mode MUST be self-contained+packed-full.
#   PHASE 3   — per-stage summary (assembler_kernel drop) + epoch comparison.
#
# Compares to the documented self-contained baselines (N=1 253 s, N=8 59 s) and
# DiskGNN celebi 123.5 s/ep. Same-day, same-machine, sequential, but PHASE 1 and
# PHASE 2 are DIFFERENT server sessions (a rebuild sits between), so page-cache
# state differs — not same-process.
#
# Safe to re-run: PHASE 1 auto-skips if addr_tables is already gone; the
# packed-full build overwrites via O_TRUNC.
set -u -o pipefail
REPO=/home/bfuentes/MillenniumDB_Testing
MDB="$REPO/build/Release/bin/mdb"; DB="$REPO/data/dbs/gql/papers100M"
SAMPLE="${SAMPLE:-e2e5ep}"; PORT="${PORT:-7910}"
OUT="$REPO/docs/research/2026-06-08-packed-full"; mkdir -p "$OUT"
RES="$OUT/RESULT_packed_full.txt"; SRVLOG="$OUT/server_pf.log"
: > "$RES"; : > "$SRVLOG"
log(){ echo "[pf $(date +%H:%M:%S)] $*" | tee -a "$RES"; }
gql(){ curl -s --max-time 172800 -X POST "http://localhost:$PORT/gql" -H "Content-Type: text/plain" --data-binary "$1"; }
SD="$DB/samples/$SAMPLE"; FEAT="$DB/gnn_features"; SRV=""
trap '[ -n "$SRV" ] && kill -9 "$SRV" 2>/dev/null' EXIT INT TERM
# curl exits 0 on server errors — MUST grep the BODY.
err(){ echo "$1" | grep -qiE "error|exception" && { echo "  BODY: $1" | tee -a "$RES"; return 1; }; return 0; }
disk(){ df -h "$DB" | awk 'NR==2{print "  disk: "$4" free ("$5")"}' | tee -a "$RES"; }
mode(){ grep -i "feature-load mode" "$SRVLOG" | tail -1 | sed 's/^/  [mode] /' | tee -a "$RES"; }
ep(){ grep -E "epoch=" "$SRVLOG" | tail -"$1" | sed 's/^/  [epoch] /' | tee -a "$RES"; }
# split: 0=train. Cols: 3 sample_read 9 assembler_kernel 11 active 12 edge 13 h2d 14 fwd 15 bwd (us)
summ(){ awk -F, -v tag="$1" 'NR>1 && $2==0{sr+=$3;ak+=$9;ac+=$11;ed+=$12;h+=$13;fw+=$14;bw+=$15;n++}
  END{ if(!n){print "  "tag": no train rows"; exit} f=1e6;
    printf "  %-16s train_batches=%d  sample_read=%7.1f  assembler_kernel=%7.1f  active_idx=%6.1f  edge_build=%6.1f  h2d=%4.1f  fwd=%4.1f  bwd=%4.1f  (sum=%.1f s)\n",
      tag,n,sr/f,ak/f,ac/f,ed/f,h/f,fw/f,bw/f,(sr+ak+ac+ed+h+fw+bw)/f }' "$2" | tee -a "$RES"; }

# Hyperparameters — identical to the self-contained harness.
HP="model:'graphsage',hiddenDim:256,lr:0.001,dropout:0.2,normalize:false,patience:999,tolerance:0,randomSeed:42,sampleCacheMb:2048,saveOnBestVal:false,saveFinal:false"

# ---- server lifecycle helpers -------------------------------------------------
start_server(){
  for p in $(pgrep -f "bin/mdb server .*-p $PORT" 2>/dev/null||true); do kill -9 "$p" 2>/dev/null; done; sleep 2
  "$MDB" server "$DB" -p $PORT -t 172800 --browser false --versioned-buffer 3GB >>"$SRVLOG" 2>&1 & SRV=$!
  for _ in $(seq 1 240); do
    gql "RETURN 1" 2>/dev/null | grep -q 1 && return 0
    kill -0 $SRV 2>/dev/null || { log "server died (see $SRVLOG)"; exit 1; }
    sleep 1
  done
  log "server did not become ready in 240s"; exit 1
}
stop_server(){
  [ -n "$SRV" ] && kill "$SRV" 2>/dev/null
  for _ in $(seq 1 60); do kill -0 "$SRV" 2>/dev/null || break; sleep 1; done
  kill -9 "$SRV" 2>/dev/null; SRV=""; sleep 2
}

# ---- prerequisite check -------------------------------------------------------
prereq_ok(){
  local ok=1
  for f in "$MDB" "$FEAT/node_features.fmat" "$FEAT/node_features.rmap" \
           "$FEAT/node_features_store.meta" "$SD/blocks"; do
    if [ ! -e "$f" ]; then log "PREREQ MISSING: $f"; ok=0; fi
  done
  [ "$ok" -eq 1 ]
}

# ---- N=1 sequential 1ep (per-stage via profileLog) ----------------------------
run_seq(){ # $1=tag $2=extra-opts $3=csv
  log "N=1 seq 1ep — $1"
  local T="CALL gnn_train('$SAMPLE','node_features',{$HP,epochs:1,useAsyncPrefetcher:false,prefetchNumWorkers:1,profileLog:'$3',outputDir:'pf_$1'$2}) YIELD ranEpochs,trainSeconds RETURN *"
  local R; R=$(gql "$T"); err "$R" || exit 1; log "  yields: $(echo "$R"|tail -1)"; mode; ep 1
}

# ---- N=8 2ep overlapped --------------------------------------------------------
run_n8(){ # $1=tag $2=extra-opts
  log "N=8 2ep — $1"
  local T="CALL gnn_train('$SAMPLE','node_features',{$HP,epochs:2,useAsyncPrefetcher:true,prefetchNumWorkers:8,prefetchQueueSize:6,outputDir:'pf8_$1'$2}) YIELD ranEpochs,trainSeconds RETURN *"
  local R; R=$(gql "$T"); err "$R" || exit 1; log "  yields: $(echo "$R"|tail -1)"; mode; ep 2
}

# ==============================================================================
log "===== papers100M Packed-Full A/B bench (sample=$SAMPLE, port=$PORT) ====="
prereq_ok || { log "ABORT: prerequisites missing (see above)."; exit 1; }
log "prereqs OK (mdb, node_features.fmat, node_features.rmap, store.meta, blocks/)"
disk

# ------------------------------------------------------------------------------
# PHASE 1 — v2 baseline (server session 1, 4-tier intact)
# ------------------------------------------------------------------------------
log "================= PHASE 1 — v2 / self-contained baseline ================="
PHASE1_RAN=0
if [ -d "$SD/addr_tables" ]; then
  log "addr_tables present — measuring v2 baseline in-session"
  start_server
  run_seq v2_baseline "" "$OUT/pf_v2_seq.csv"
  run_n8  v2_baseline ""
  stop_server
  PHASE1_RAN=1
else
  log "addr_tables ALREADY GONE — PHASE 1 skipped (script re-run after a prior"
  log "  destructive pass). Documented 4-tier baseline: N=1 253 s, N=8 59 s."
fi

# ------------------------------------------------------------------------------
# PHASE 1.5 — free disk (DESTRUCTIVE, guarded)
# ------------------------------------------------------------------------------
log "================= PHASE 1.5 — free disk (DESTRUCTIVE) ================="
# Re-assert the prereq guard right before any delete: NEVER remove the projection
# prerequisites (rmap/rmap.idx/fmat), blocks/, store.meta, or the gpu/cpu caches.
GUARD_OK=1
for f in "$FEAT/node_features.fmat" "$FEAT/node_features.rmap" \
         "$FEAT/node_features_store.meta" "$SD/blocks"; do
  if [ ! -e "$f" ]; then log "DESTRUCTIVE GUARD FAILED — missing $f"; GUARD_OK=0; fi
done
if [ "$GUARD_OK" -ne 1 ]; then
  log "ABORT before delete: a prerequisite is missing. Nothing deleted."
  exit 1
fi
log "destructive guard OK — deleting ONLY the regenerable 4-tier"
disk
for d in "$FEAT/node_features_reordered.fmat" \
         "$FEAT/node_features_reordered.rmap" \
         "$FEAT/node_features_reordered.rmap.idx" \
         "$SD/packed_slim" \
         "$SD/addr_tables"; do
  if [ -e "$d" ]; then
    log "  rm -rf $d  ($(du -sh "$d" 2>/dev/null | cut -f1))"
    rm -rf "$d"
  else
    log "  (skip, absent) $d"
  fi
done
log "deletion done"
disk

# ------------------------------------------------------------------------------
# PHASE 2 — packed-full (server session 2)
# ------------------------------------------------------------------------------
log "================= PHASE 2 — packed-full ================="
start_server
log "build packed-full pack (additive)"
t0=$(date +%s)
BR=$(gql "CALL gnn_build_feature_store('$SAMPLE','node_features',{packFullFeatures:true}) YIELD packedFullMb,buildTimeMs RETURN *")
err "$BR" || exit 1
t1=$(date +%s)
log "  build yields: $(echo "$BR"|tail -1)"
log "  build_seconds=$((t1-t0))  packed_full=$(du -sh "$SD/packed_full" 2>/dev/null|cut -f1)"
disk

run_seq packfull "" "$OUT/pf_packfull_seq.csv"
PF_MODE=$(grep -i "feature-load mode" "$SRVLOG" | tail -1)
if echo "$PF_MODE" | grep -qi "self-contained+packed-full"; then
  log "  OK — packed-full activated ($PF_MODE)"
else
  log "  WARNING — packed-full did NOT activate. mode=[$PF_MODE]"
  log "  Expected 'self-contained+packed-full'. Check packed_full presence + store_fp match."
fi

run_n8 packfull ""

stop_server

# ------------------------------------------------------------------------------
# PHASE 3 — summary
# ------------------------------------------------------------------------------
log "================= PHASE 3 — per-stage summary (N=1 seq, train split) ================="
if [ -f "$OUT/pf_v2_seq.csv" ]; then
  summ "v2_baseline" "$OUT/pf_v2_seq.csv"
else
  log "  v2_baseline: no per-stage CSV (PHASE 1 skipped). Baseline N=1 253 s / N=8 59 s."
fi
summ "packfull" "$OUT/pf_packfull_seq.csv"
log "Interpretation:"
log "  Expect packed-full assembler_kernel to DROP sharply vs v2 (~143 -> ~28-40 s)"
log "  and sample_read=active_idx=edge_build=0 (self-contained graph path)."
log "  N=1 + N=8 epochs are logged above per phase ([epoch] lines)."
log "  Compare to self-contained baselines N=1 253 s, N=8 59 s and DiskGNN celebi 123.5 s/ep."
log "  CAVEAT: PHASE 1 (v2) and PHASE 2 (packed-full) ran in DIFFERENT server"
log "  sessions with a rebuild between them, so page-cache state differs — same-day,"
log "  same-machine, sequential, NOT same-process."
log "PHASE1_RAN=$PHASE1_RAN"
log "DONE."
