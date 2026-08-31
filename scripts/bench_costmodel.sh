#!/usr/bin/env bash
# bench_costmodel.sh — per-stage cost-model instrumentation for the GNN pipeline.
#
# Runs the full pipeline on one or more datasets (cora, arxiv) and captures, per
# stage (project / sample / materialize / build / train), from the SERVER
# process /proc:
#   * wall_s            — wall-clock seconds
#   * dread_MB/dwrite_MB — actual block-device I/O (read_bytes/write_bytes, post page-cache)
#   * lread_MB/lwrite_MB — logical syscall I/O (rchar/wchar, includes cache hits)
#   * peakrss_MB        — VmHWM after the stage (cumulative process peak)
# plus dataset shape (nodes/edges/featDim/classes) and artifact sizes, and the
# gnn_train accuracy/telemetry yields. Emits one CSV row per (dataset, stage) to
# docs/research/<date>-costmodel/costmodel.csv and prints a table.
#
# The companion analyzer scripts/costmodel_analyze.py turns these two points
# (cora, arxiv) into per-stage rates, a roofline classification, a papers100M
# extrapolation, and gpu/cpu budget recommendations.
#
# Usage: bench_costmodel.sh [cora] [arxiv]   (default: both)
# Fanout notation: since 2026-07-06 gnn_offline_sample reads the list in DGL
# order -- the LAST element is the hop adjacent to the seeds -- and reverses it
# internally. The literals below were flipped to preserve the hop order this
# bench was written to measure; a CSV produced before that flip sampled the
# mirror order and is not comparable with one produced after.
set -u -o pipefail

REPO="${MDB_HOME:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
MDB="$REPO/build/Release/bin/mdb"
PORT=7885
OUTDIR="$REPO/docs/research/2026-05-31-costmodel"
CSV="$OUTDIR/costmodel.csv"
DB=""; SRVPID=""; RC=0
mkdir -p "$OUTDIR"

cleanup() {
  [ -n "$SRVPID" ] && kill "$SRVPID" 2>/dev/null || true
  [ -n "$SRVPID" ] && wait "$SRVPID" 2>/dev/null || true
  [ -n "$DB" ] && rm -rf "$DB" 2>/dev/null || true
}
trap cleanup EXIT INT TERM
say() { echo "[costmodel] $*"; }
gql() { curl -s --max-time 3600 -X POST "http://localhost:$PORT/gql" \
             -H "Content-Type: text/plain" --data-binary "$1"; }
now() { date +%s.%N; }
col() {
  echo "$1" | python3 -c "
import sys
ls=[l for l in sys.stdin.read().splitlines() if l.strip()]
if len(ls)<2: print(''); raise SystemExit
h=ls[0].split(','); d=ls[-1].split(',')
try: print(d[h.index('$2')])
except Exception: print('')
"
}
# Read four /proc/<pid>/io counters (bytes): rchar wchar read_bytes write_bytes.
io_snap() {
  awk '/^rchar:/{r=$2}/^wchar:/{w=$2}/^read_bytes:/{rb=$2}/^write_bytes:/{wb=$2}
       END{printf "%s %s %s %s", r, w, rb, wb}' "/proc/$SRVPID/io" 2>/dev/null || echo "0 0 0 0"
}
rss_peak_mb() {  # VmHWM in MB
  awk '/^VmHWM:/{printf "%.1f", $2/1024}' "/proc/$SRVPID/status" 2>/dev/null || echo 0
}
mb() { echo "$1" | awk '{printf "%.2f", $1/1048576}'; }   # bytes -> MB
dur() { echo "$1 $2" | awk '{printf "%.3f", $2-$1}'; }

# Run one instrumented stage: $1=stage label, $2=gql query. Captures io+time+rss
# deltas around the call and appends a CSV row. Echoes the response body.
DATASET=""
stage() {
  local label="$1" query="$2"
  read -r r0 w0 rb0 wb0 <<<"$(io_snap)"; local t0; t0=$(now)
  local body; body=$(gql "$query")
  local t1; t1=$(now); read -r r1 w1 rb1 wb1 <<<"$(io_snap)"
  local rss; rss=$(rss_peak_mb)
  printf '%s,%s,%s,%s,%s,%s,%s,%s\n' \
    "$DATASET" "$label" "$(dur "$t0" "$t1")" \
    "$(mb $((rb1-rb0)))" "$(mb $((wb1-wb0)))" \
    "$(mb $((r1-r0)))"  "$(mb $((w1-w0)))" "$rss" >> "$CSV"
  echo "$body"
}

run_dataset() {
  # $1=name $2=gql $3=npy $4=nodeLabel $5=edgeLabel $6=fanout $7=batch $8=epochs $9=hidden ${10}=gpu_mb ${11}=cpu_mb
  local name="$1" GQLF="$2" NPYF="$3" NODE="$4" EDGE="$5" FAN="$6" BATCH="$7" EP="$8" HID="$9" GMB="${10}" CMB="${11}"
  DATASET="$name"
  [ -f "$GQLF" ] || { say "skip $name — gql missing ($GQLF)"; return; }

  DB=$(mktemp -d /tmp/costmodel_${name}.XXXXXX)
  local t0; t0=$(now)
  local imp_args=(import "$GQLF" "$DB")
  [ -n "$NPYF" ] && [ -f "$NPYF" ] && imp_args+=(--with-tensors "$NPYF")
  "$MDB" "${imp_args[@]}" >/dev/null 2>&1 || { say "$name import failed"; rm -rf "$DB"; DB=""; return; }
  local t1; t1=$(now)
  printf '%s,%s,%s,%s,%s,%s,%s,%s\n' "$name" "import" "$(dur "$t0" "$t1")" "" "" "" "" "" >> "$CSV"

  for p in $(pgrep -f "build/Release/bin/mdb server .*-p $PORT" 2>/dev/null || true); do kill -9 "$p" 2>/dev/null || true; done
  "$MDB" server "$DB" -p $PORT -t 3600 --browser false >"$DB/server.log" 2>&1 &
  SRVPID=$!
  local ready=0
  for _ in $(seq 1 60); do gql "RETURN 1" 2>/dev/null | grep -q 1 && { ready=1; break; }
    kill -0 "$SRVPID" 2>/dev/null || { say "$name server died"; return; }; sleep 0.5; done
  [ "$ready" = 1 ] || { say "$name server not ready"; return; }

  local b
  b=$(stage project "CALL graph_project('${name}_g','$NODE','$EDGE',{orientation:'UNDIRECTED',includeFeatures:'node_features',labelProperty:'label'$( [ "$name" = arxiv ] && echo ",splitProperty:'split'")}) YIELD graphName,nodeCount,relationshipCount,featureDim,numClasses RETURN *")
  say "$name shape: nodes=$(col "$b" nodeCount) edges=$(col "$b" relationshipCount) D=$(col "$b" featureDim) C=$(col "$b" numClasses)"

  stage sample "CALL gnn_offline_sample('${name}_g','${name}_s',$FAN,{batchSize:$BATCH,randomSeed:42,orientation:'UNDIRECTED'}) YIELD totalBatches,uniqueNodes RETURN *" >/dev/null
  stage materialize "CALL gnn_materialize_batches('${name}_s','node_features',{reorder:1,numHashes:2,force:1}) YIELD totalBatches RETURN *" >/dev/null
  b=$(stage build "CALL gnn_build_feature_store('${name}_s','node_features',{gpu_budget_mb:$GMB,cpu_budget_mb:$CMB}) YIELD l1Nodes,l2Nodes,l3Nodes,l4Nodes,slimMb,reorderedMb,gpuCacheMb,cpuCacheMb,totalDiskMb RETURN *")
  say "$name tiers: L1=$(col "$b" l1Nodes) L2=$(col "$b" l2Nodes) L3=$(col "$b" l3Nodes) L4=$(col "$b" l4Nodes) | slimMb=$(col "$b" slimMb) reordMb=$(col "$b" reorderedMb) gpuMb=$(col "$b" gpuCacheMb) cpuMb=$(col "$b" cpuCacheMb) diskMb=$(col "$b" totalDiskMb)"
  b=$(stage train "CALL gnn_train('${name}_s','node_features',{model:'graphsage',hiddenDim:$HID,epochs:$EP,lr:0.01,dropout:0.5,patience:999,randomSeed:42}) YIELD testAccuracy,bestValAccuracy,ranEpochs,trainSeconds,assembleSeconds,forwardSeconds,backwardSeconds,useAddrTablesEffective,l1HitRatio RETURN *")
  say "$name train: testAcc=$(col "$b" testAccuracy) bestVal=$(col "$b" bestValAccuracy) ep=$(col "$b" ranEpochs) trainS=$(col "$b" trainSeconds) asm=$(col "$b" assembleSeconds) fwd=$(col "$b" forwardSeconds) bwd=$(col "$b" backwardSeconds) addr=$(col "$b" useAddrTablesEffective) l1=$(col "$b" l1HitRatio)"

  kill "$SRVPID" 2>/dev/null || true; wait "$SRVPID" 2>/dev/null || true; SRVPID=""
  rm -rf "$DB" 2>/dev/null || true; DB=""
}

# --- driver ---
echo "dataset,stage,wall_s,dread_MB,dwrite_MB,lread_MB,lwrite_MB,peakrss_MB" > "$CSV"
WANT="${*:-cora arxiv}"
echo "$WANT" | grep -qw cora  && run_dataset cora  "$REPO/data/example/gql/cora/cora.gql" "$REPO/data/example/gql/cora/cora_features.npy" Paper CITES "[5,10]" 64 30 256 512 256
echo "$WANT" | grep -qw arxiv && run_dataset arxiv "$REPO/data/example/gql/ogbn-arxiv/ogbn-arxiv/ogbn_arxiv.gql" "$REPO/data/example/gql/ogbn-arxiv/ogbn-arxiv/ogbn_arxiv_features.npy" Node CONNECTS "[10,15]" 1024 30 256 2048 512

say "CSV written: $CSV"
column -t -s, "$CSV"
exit $RC
