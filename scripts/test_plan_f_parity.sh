#!/usr/bin/env bash
# test_plan_f_parity.sh — Plan F numWorkers determinism gate (semantic).
#
# INVARIANT: numWorkers must NOT change the trained model. gnn_offline_sample
# with numWorkers={0,1,2,4} on the same (projection, seed, fanout) must yield
# the SAME testAccuracy after materialize+build+train — i.e. the sampled
# subgraph content per batch_id depends only on (random_seed, batch_id),
# independent of how many worker threads produced it.
#
# Why testAccuracy and not batches.dat bytes: with numWorkers>=2 the physical
# layout of batches.dat is written in worker-completion order (non-deterministic
# offsets), but batch_index maps batch_id->offset correctly and training random-
# accesses by batch_id, so the layout variance is BENIGN. The semantic invariant
# (content per batch_id, hence the trained model) is what must hold. We also
# assert W=0 and W=1 are byte-identical (both write in batch_id order), pinning
# the single-threaded paths.
#
# Exit 0 = PASS (identical testAcc across all numWorkers + W0==W1 bytes).
set -u -o pipefail

REPO=/home/bfuentes/MillenniumDB_Testing
MDB="$REPO/build/Release/bin/mdb"
CORA_GQL="$REPO/data/example/gql/cora/cora.gql"
CORA_NPY="$REPO/data/example/gql/cora/cora_features.npy"
PORT=7882
DB=$(mktemp -d /tmp/cora_planf.XXXXXX)
SRVPID=""
RC=1

cleanup() {
  [ -n "$SRVPID" ] && kill "$SRVPID" 2>/dev/null || true
  [ -n "$SRVPID" ] && wait "$SRVPID" 2>/dev/null || true
  rm -rf "$DB" 2>/dev/null || true
  exit $RC
}
trap cleanup EXIT INT TERM
say()  { echo "[planf] $*"; }
fail() { echo "[planf] FAIL: $*" >&2; RC=1; exit 1; }
gql() { curl -s --max-time 120 -X POST "http://localhost:$PORT/gql" \
             -H "Content-Type: text/plain" --data-binary "$1"; }
acc_of() {  # extract testAccuracy column from a 2-line CSV body
  echo "$1" | python3 -c "
import sys
ls=[l for l in sys.stdin.read().splitlines() if l.strip()]
if len(ls)<2: print('NaN'); raise SystemExit
h=ls[0].split(','); d=ls[-1].split(',')
try: print('%.6f'%float(d[h.index('testAccuracy')]))
except Exception: print('NaN')
"
}

"$MDB" import "$CORA_GQL" "$DB" --with-tensors "$CORA_NPY" >/dev/null 2>&1 || fail "import failed"
for p in $(pgrep -f "build/Release/bin/mdb server .*-p $PORT" 2>/dev/null || true); do kill -9 "$p" 2>/dev/null || true; done
"$MDB" server "$DB" -p $PORT -t 600 --browser false >"$DB/server.log" 2>&1 &
SRVPID=$!
ready=0
for _ in $(seq 1 30); do
  gql "RETURN 1" 2>/dev/null | grep -q 1 && { ready=1; break; }
  kill -0 "$SRVPID" 2>/dev/null || fail "server died (see $DB/server.log)"
  sleep 0.5
done
[ "$ready" = "1" ] || fail "server not ready"

gql "CALL graph_project('cora','Paper','CITES',{orientation:'UNDIRECTED',includeFeatures:'node_features',labelProperty:'label'}) YIELD graphName RETURN *" >/dev/null

TRAIN_Q_TPL="CALL gnn_train('%s','node_features',{epochs:5,randomSeed:42,patience:999,saveOnBestVal:false,saveFinal:false}) YIELD testAccuracy RETURN *"
declare -A ACC SHA
for W in 0 1 2 4; do
  s="s$W"
  body=$(gql "CALL gnn_offline_sample('cora','$s',[5,10],{batchSize:64,randomSeed:42,orientation:'UNDIRECTED',numWorkers:$W}) YIELD totalBatches RETURN *")
  echo "$body" | grep -qiE "error|exception|failed" && fail "sample W=$W errored: $body"
  SHA[$W]=$(sha256sum "$DB/samples/$s/batches.dat" 2>/dev/null | awk '{print $1}')
  gql "CALL gnn_materialize_batches('$s','node_features',{reorder:1,numHashes:2,force:1}) YIELD totalBatches RETURN *" >/dev/null
  gql "CALL gnn_build_feature_store('$s','node_features',{gpu_budget_mb:0,cpu_budget_mb:100,force:1}) YIELD l2Nodes RETURN *" >/dev/null
  tq=$(printf "$TRAIN_Q_TPL" "$s")
  ACC[$W]=$(acc_of "$(gql "$tq")")
  say "numWorkers=$W -> testAcc=${ACC[$W]}  sha=${SHA[$W]:0:16}…"
done

# Semantic gate: identical testAcc across all numWorkers.
ref=${ACC[0]}
ok=1
[ "$ref" = "NaN" ] && fail "could not parse testAcc for W=0"
for W in 1 2 4; do
  [ "${ACC[$W]}" != "$ref" ] && { say "MISMATCH: W=$W testAcc=${ACC[$W]} != W=0 $ref"; ok=0; }
done
# Layout gate: single-threaded paths (W=0, W=1) are byte-identical.
[ "${SHA[0]}" != "${SHA[1]}" ] && { say "MISMATCH: W=0 and W=1 batches.dat differ (single-threaded paths must match)"; ok=0; }

if [ "$ok" = "1" ]; then
  say "PASS — numWorkers {0,1,2,4} train to identical testAcc ($ref); W0==W1 byte-identical"
  RC=0
else
  fail "Plan F determinism violated — see MISMATCH lines"
fi
exit $RC
