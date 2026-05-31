#!/usr/bin/env bash
# arxiv_e2e.sh — mid-scale (ogbn-arxiv) end-to-end validation of the GNN pipeline
# with the post-refactor defaults. Runs project -> sample -> materialize -> build
# -> train on a fresh /tmp DB, reports per-stage wall-clock, and validates the
# contracts we fixed at code level actually hold at mid-scale:
#
#   * bestValAccuracy > 0           (auto usePredefinedSplits / F#2 + valid-split mapping)
#   * testAccuracy in the published GraphSAGE-MEAN ballpark (~0.66-0.72)
#   * useAddrTablesEffective = true (STEP 1 SHA path + STEP 6 telemetry)
#   * a 2nd feature-store build with no force REUSES (content-fingerprint staleness)
#
# Dataset: data/example/gql/ogbn-arxiv/ogbn-arxiv/ (169,343 nodes, 128 feat, 40 classes,
# OGB time-split train/valid/test). Generate it once with:
#   <conda diskgnn_cu124>/bin/python scripts/gnn_datasets/download_ogb.py \
#       --dataset ogbn-arxiv --output data/example/gql/ogbn-arxiv/ \
#       --ogb-root /home/bfuentes/diskgnn_data/ogb_raw
#
# Exit 0 = PASS.
set -u -o pipefail

REPO=/home/bfuentes/MillenniumDB_Testing
MDB="$REPO/build/Release/bin/mdb"
DSET="$REPO/data/example/gql/ogbn-arxiv/ogbn-arxiv"
GQL="$DSET/ogbn_arxiv.gql"
NPY="$DSET/ogbn_arxiv_features.npy"
PORT=7884
DB=$(mktemp -d /tmp/arxiv_e2e.XXXXXX)
LOG="$DB/server.log"
SRVPID=""
RC=1

cleanup() {
  [ -n "$SRVPID" ] && kill "$SRVPID" 2>/dev/null || true
  [ -n "$SRVPID" ] && wait "$SRVPID" 2>/dev/null || true
  rm -rf "$DB" 2>/dev/null || true
  exit $RC
}
trap cleanup EXIT INT TERM
say()  { echo "[arxiv] $*"; }
fail() { echo "[arxiv] FAIL: $*" >&2; RC=1; exit 1; }
gql()  { curl -s --max-time 1800 -X POST "http://localhost:$PORT/gql" \
              -H "Content-Type: text/plain" --data-binary "$1"; }
now()  { date +%s.%N; }
dur()  { echo "$1 $2" | awk '{printf "%.1f", $2-$1}'; }

# Extract a named column from a 2-line CSV YIELD body.
col() {
  local body="$1" name="$2"
  echo "$body" | python3 -c "
import sys
ls=[l for l in sys.stdin.read().splitlines() if l.strip()]
if len(ls)<2: print(''); raise SystemExit
h=ls[0].split(','); d=ls[-1].split(',')
try: print(d[h.index('$name')])
except Exception: print('')
"
}

[ -f "$GQL" ] || fail "arxiv gql not found at $GQL (run download_ogb.py first)"
[ -f "$NPY" ] || fail "arxiv features not found at $NPY"

# --- import + server ---
t0=$(now)
"$MDB" import "$GQL" "$DB" --with-tensors "$NPY" >/dev/null 2>&1 || fail "import failed"
t1=$(now); say "import: $(dur "$t0" "$t1")s"
for p in $(pgrep -f "build/Release/bin/mdb server .*-p $PORT" 2>/dev/null || true); do kill -9 "$p" 2>/dev/null || true; done
"$MDB" server "$DB" -p $PORT -t 1800 --browser false >"$LOG" 2>&1 &
SRVPID=$!
ready=0
for _ in $(seq 1 60); do
  gql "RETURN 1" 2>/dev/null | grep -q 1 && { ready=1; break; }
  kill -0 "$SRVPID" 2>/dev/null || fail "server died (see $LOG)"
  sleep 0.5
done
[ "$ready" = "1" ] || fail "server not ready"

chk() { echo "$1" | grep -qiE "error|exception|failed" && { say "body: $1"; fail "$2"; }; return 0; }

# --- 1. project (GNN intent -> STEP 3 auto GNN_MINIMAL + topology snapshot) ---
t0=$(now)
b=$(gql "CALL graph_project('arxiv_gnn','Node','CONNECTS',{orientation:'UNDIRECTED',includeFeatures:'node_features',labelProperty:'label',splitProperty:'split'}) YIELD graphName,nodeCount,relationshipCount,featureDim,numClasses RETURN *")
chk "$b" "graph_project errored"
t1=$(now)
say "project: $(dur "$t0" "$t1")s  nodes=$(col "$b" nodeCount) rels=$(col "$b" relationshipCount) featDim=$(col "$b" featureDim) classes=$(col "$b" numClasses)"

# --- 2. sample (auto usePredefinedSplits via F#2, auto numWorkers via F#1) ---
t0=$(now)
b=$(gql "CALL gnn_offline_sample('arxiv_gnn','arxiv_s',[15,10],{batchSize:1024,randomSeed:42,orientation:'UNDIRECTED'}) YIELD totalBatches,trainBatches,validationBatches,testBatches,uniqueNodes,numWorkersUsed RETURN *")
chk "$b" "gnn_offline_sample errored"
t1=$(now)
say "sample: $(dur "$t0" "$t1")s  batches=$(col "$b" totalBatches) (tr=$(col "$b" trainBatches) val=$(col "$b" validationBatches) te=$(col "$b" testBatches)) uniqNodes=$(col "$b" uniqueNodes) workers=$(col "$b" numWorkersUsed)"
VAL_B=$(col "$b" validationBatches)
[ "${VAL_B:-0}" -gt 0 ] 2>/dev/null || fail "0 validation batches — predefined splits not applied (val mapping regression)"

# --- 3. materialize ---
t0=$(now)
b=$(gql "CALL gnn_materialize_batches('arxiv_s','node_features',{reorder:1,numHashes:2,force:1}) YIELD totalBatches RETURN *")
chk "$b" "gnn_materialize_batches errored"
t1=$(now); say "materialize: $(dur "$t0" "$t1")s"

# --- 4. build feature store (GPU L1) ---
t0=$(now)
b=$(gql "CALL gnn_build_feature_store('arxiv_s','node_features',{gpu_budget_mb:2048,cpu_budget_mb:512}) YIELD l1Nodes,l2Nodes,l3Nodes,l4Nodes RETURN *")
chk "$b" "gnn_build_feature_store errored"
t1=$(now)
say "build: $(dur "$t0" "$t1")s  L1=$(col "$b" l1Nodes) L2=$(col "$b" l2Nodes) L3=$(col "$b" l3Nodes) L4=$(col "$b" l4Nodes)"

# --- 5. train ---
t0=$(now)
b=$(gql "CALL gnn_train('arxiv_s','node_features',{model:'graphsage',hiddenDim:256,epochs:30,lr:0.01,dropout:0.5,patience:10,randomSeed:42}) YIELD testAccuracy,bestValAccuracy,ranEpochs,trainSeconds,useAddrTablesEffective,l1HitRatio RETURN *")
chk "$b" "gnn_train errored"
t1=$(now)
TEST=$(col "$b" testAccuracy); BVAL=$(col "$b" bestValAccuracy); EP=$(col "$b" ranEpochs)
TRS=$(col "$b" trainSeconds); ADDR=$(col "$b" useAddrTablesEffective); L1H=$(col "$b" l1HitRatio)
say "train: $(dur "$t0" "$t1")s  testAcc=$TEST bestVal=$BVAL epochs=$EP trainS=$TRS useAddrTables=$ADDR l1Hit=$L1H"

# --- 6. STEP 8 reuse: rebuild feature store, no force -> must reuse ---
off=$(wc -c < "$LOG")
b=$(gql "CALL gnn_build_feature_store('arxiv_s','node_features',{gpu_budget_mb:2048,cpu_budget_mb:512}) YIELD l1Nodes RETURN *")
chk "$b" "gnn_build_feature_store (reuse) errored"
newlog=$(tail -c +$((off + 1)) "$LOG" 2>/dev/null || true)
echo "$newlog" | grep -q "reusing existing artifacts" \
  && say "reuse: feature store skipped rebuild (content fingerprint match)" \
  || fail "2nd build did NOT reuse (STEP 8 staleness gate)"

# --- validations ---
ok=1
awk "BEGIN{exit !($TEST>0.60)}" || { say "testAcc $TEST <= 0.60 (below GraphSAGE ballpark)"; ok=0; }
awk "BEGIN{exit !($BVAL>0.0)}"  || { say "bestVal $BVAL not > 0 (splits not applied)"; ok=0; }
[ "$ADDR" = "true" ] || { say "useAddrTablesEffective=$ADDR (expected true)"; ok=0; }

if [ "$ok" = "1" ]; then
  say "PASS — testAcc=$TEST bestVal=$BVAL useAddrTables=$ADDR l1Hit=$L1H, STEP 8 reuse OK"
  RC=0
else
  fail "validations failed (see lines above)"
fi
exit $RC
