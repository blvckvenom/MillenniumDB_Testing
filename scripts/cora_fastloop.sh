#!/usr/bin/env bash
# cora_fastloop.sh — fast (~5s) end-to-end GNN pipeline gate on the cora dataset.
#
# This is the iteration harness for the small-dataset-first refactor: every
# refactor step must keep this green. Runs the full pipeline on a FRESH /tmp DB
# (clean reset, no stale-artifact reuse), asserts testAccuracy > 0.70, and
# re-runs training with the same seed to assert determinism within 0.001.
#
# Usage:  scripts/cora_fastloop.sh [--keep]
#   --keep   do not delete the /tmp DB on exit (for debugging)
#
# Exit code 0 = PASS (accuracy gate + determinism gate), non-zero = FAIL.
set -u -o pipefail

# --------------------------------------------------------------------------
REPO=/home/bfuentes/MillenniumDB_Testing
MDB="$REPO/build/Release/bin/mdb"
CORA_GQL="$REPO/data/example/gql/cora/cora.gql"
CORA_NPY="$REPO/data/example/gql/cora/cora_features.npy"
PORT=7878
ACC_GATE=0.70          # testAccuracy must exceed this
DET_TOL=0.001          # |testAcc(run1) - testAcc(run2)| must be <= this
KEEP=0
[ "${1-}" = "--keep" ] && KEEP=1

DB=$(mktemp -d /tmp/cora_fastloop.XXXXXX)
SRVPID=""
RC=1

cleanup() {
  [ -n "$SRVPID" ] && kill "$SRVPID" 2>/dev/null || true
  [ -n "$SRVPID" ] && wait "$SRVPID" 2>/dev/null || true
  if [ "$KEEP" = "0" ]; then rm -rf "$DB" 2>/dev/null || true
  else echo "[keep] DB left at $DB"; fi
  exit $RC
}
trap cleanup EXIT INT TERM

say() { echo "[cora_fastloop] $*"; }
fail() { echo "[cora_fastloop] FAIL: $*" >&2; RC=1; exit 1; }

# run_gql <name> <query> -> echoes the server response body to stdout (and a copy
# to stderr). Detects server-side errors in the BODY (curl exits 0 even on error).
run_gql() {
  local name=$1 query=$2 body
  body=$(curl -s --max-time 120 -X POST "http://localhost:$PORT/gql" \
               -H "Content-Type: text/plain" --data-binary "$query")
  echo "  [$name] $body" >&2
  if echo "$body" | grep -qiE "(error|exception|failed|already exists|not found|bad query|unexpected|out of range)"; then
    fail "$name returned an error response (see above)"
  fi
  echo "$body"
}

# col <csv-2-line-output> <colname> -> prints the float value of that column
col() {
  python3 -c "
import sys
lines=[l for l in sys.stdin.read().splitlines() if l.strip()]
if len(lines)<2: print('NaN'); sys.exit()
h=lines[0].split(','); d=lines[-1].split(',')
try: print(float(d[h.index('$1')]))
except Exception: print('NaN')
"
}

# --------------------------------------------------------------------------
[ -x "$MDB" ]      || fail "mdb binary not found at $MDB"
[ -f "$CORA_GQL" ] || fail "cora.gql not found at $CORA_GQL"
[ -f "$CORA_NPY" ] || fail "cora_features.npy not found at $CORA_NPY"

T0=$(date +%s)

say "import cora -> $DB"
"$MDB" import "$CORA_GQL" "$DB" --with-tensors "$CORA_NPY" >/dev/null 2>&1 \
  || fail "import failed"

# free the port if a stale server is holding it
for p in $(pgrep -f "build/Release/bin/mdb server .*-p $PORT" 2>/dev/null || true); do
  kill -9 "$p" 2>/dev/null || true
done

say "start server on :$PORT"
"$MDB" server "$DB" -p $PORT -t 600 --browser false >"$DB/server.log" 2>&1 &
SRVPID=$!

# poll readiness (cheap query) up to ~15s
ready=0
for _ in $(seq 1 30); do
  if curl -s --max-time 2 -X POST "http://localhost:$PORT/gql" \
        -H "Content-Type: text/plain" --data-binary "RETURN 1" 2>/dev/null | grep -q 1; then
    ready=1; break
  fi
  kill -0 "$SRVPID" 2>/dev/null || fail "server died on startup (see $DB/server.log)"
  sleep 0.5
done
[ "$ready" = "1" ] || fail "server not ready after 15s"

say "graph_project"
run_gql project "CALL graph_project('cora','Paper','CITES',{orientation:'UNDIRECTED',includeFeatures:'node_features',labelProperty:'label'}) YIELD graphName, nodeCount, relationshipCount, featureDim, numClasses RETURN *" >/dev/null

say "gnn_offline_sample"
run_gql sample "CALL gnn_offline_sample('cora','s',[5,10],{batchSize:64,randomSeed:42,orientation:'UNDIRECTED'}) YIELD totalBatches, trainBatches, validationBatches, testBatches, uniqueNodes RETURN *" >/dev/null

say "gnn_materialize_batches"
run_gql materialize "CALL gnn_materialize_batches('s','node_features',{reorder:1,numHashes:2,force:1}) YIELD totalBatches, reordered, totalTimeMs RETURN *" >/dev/null

say "gnn_build_feature_store"
run_gql build "CALL gnn_build_feature_store('s','node_features',{gpu_budget_mb:0,cpu_budget_mb:100,force:1}) YIELD l1Nodes, l2Nodes, l3Nodes, l4Nodes, buildTimeMs RETURN *" >/dev/null

TRAIN_Q="CALL gnn_train('s','node_features',{epochs:5,randomSeed:42,patience:999,saveOnBestVal:false,saveFinal:false}) YIELD testAccuracy, bestValAccuracy, ranEpochs, trainSeconds RETURN *"

say "gnn_train (run 1)"
OUT1=$(run_gql train1 "$TRAIN_Q")
ACC1=$(echo "$OUT1" | col testAccuracy)

say "gnn_train (run 2, determinism)"
OUT2=$(run_gql train2 "$TRAIN_Q")
ACC2=$(echo "$OUT2" | col testAccuracy)

T1=$(date +%s)

# --------------------------------------------------------------------------
say "testAccuracy run1=$ACC1 run2=$ACC2  (gate >$ACC_GATE, det-tol $DET_TOL)  wall=$((T1-T0))s"
VERDICT=$(python3 -c "
a1='$ACC1'; a2='$ACC2'
try:
    a1=float(a1); a2=float(a2)
except: print('FAIL parse'); raise SystemExit
import math
if math.isnan(a1) or math.isnan(a2): print('FAIL nan'); raise SystemExit
if a1 <= $ACC_GATE: print(f'FAIL acc {a1:.4f} <= $ACC_GATE'); raise SystemExit
if abs(a1-a2) > $DET_TOL: print(f'FAIL determinism |{a1:.6f}-{a2:.6f}|={abs(a1-a2):.6f} > $DET_TOL'); raise SystemExit
print(f'PASS acc={a1:.4f} det=|{a1-a2:.2e}|')
")
echo "[cora_fastloop] $VERDICT"
case "$VERDICT" in
  PASS*) RC=0 ;;
  *)     RC=1 ;;
esac
exit $RC
