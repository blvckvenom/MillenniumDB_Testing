#!/usr/bin/env bash
# symmetric_csr_cora_gate.sh — cora E2E accuracy gate for the symmetric
# (pre-merged undirected) topology path.
#
# Asserts the canonical bit-identical constant 0.8574939 (dropout=0, N=1) for the
# symmetric ON vs OFF configurations, a genuine A/B WITHIN the four-level path
# (same neighbor emission order):
#
#   ON  : useSymmetricTopology:'on'  — the pre-merged undirected sym tier (single
#         dispatch replaces the per-node out+in+merge).
#   OFF : useSymmetricTopology:'off' — the sym tier is NOT built; UNDIRECTED
#         fetches use the runtime out+in+merge fallback (same dedup rule).
#
# ON==OFF==0.8574939 means the symmetric single-slice is byte-identical to the
# runtime merge. (cora has 151 mutual citations, so the sym path genuinely
# exercises the edge-id-keyed dedup that preserves the duplicated neighbors.)
#
# Every CALL body is checked for MDB error text (HTTP 200 + error-in-body).
set -u -o pipefail
REPO=/home/bfuentes/MillenniumDB_Testing
MDB="$REPO/build/Release/bin/mdb"
CORA_GQL="$REPO/data/example/gql/cora/cora.gql"
CORA_NPY="$REPO/data/example/gql/cora/cora_features.npy"
PORT="${PORT:-7889}"; CANON=0.8574939
DB=$(mktemp -d /tmp/sym_cora.XXXXXX); SRVPID=""
cleanup(){ [ -n "$SRVPID" ] && kill "$SRVPID" 2>/dev/null; rm -rf "$DB" 2>/dev/null; }
trap cleanup EXIT INT TERM
gql(){ curl -s --max-time 180 -X POST "http://localhost:$PORT/gql" -H 'Content-Type: text/plain' --data-binary "$1"; }
fail(){ echo "[sym_gate] FAIL: $*" >&2; exit 1; }
chk(){ echo "$1" | grep -qiE "error|exception|failed|out of range" && { echo "  BODY: $1" >&2; fail "$2 error"; }; }
col(){ python3 -c "
import sys
L=[l for l in sys.stdin.read().splitlines() if l.strip()]
if len(L)<2: print('NaN'); sys.exit()
h=L[0].split(','); d=L[-1].split(',')
try: print(repr(float(d[h.index('$1')])))
except: print('NaN')"; }

"$MDB" import "$CORA_GQL" "$DB" --with-tensors "$CORA_NPY" >/dev/null 2>&1 || fail import
for p in $(pgrep -f "bin/mdb server .*-p $PORT" 2>/dev/null || true); do kill -9 "$p" 2>/dev/null; done
"$MDB" server "$DB" -p $PORT -t 600 --browser false >"$DB/server.log" 2>&1 & SRVPID=$!
for _ in $(seq 1 30); do gql "RETURN 1" 2>/dev/null | grep -q 1 && break; kill -0 $SRVPID 2>/dev/null || fail server_died; sleep 0.5; done

b=$(gql "CALL graph_project('cora','Paper','CITES',{orientation:'UNDIRECTED',includeFeatures:'node_features',labelProperty:'label'}) YIELD nodeCount RETURN *"); chk "$b" project

# $1 = tag, $2 = extra sample options -> echoes testAccuracy (or NaN).
run_once(){
  local tag="$1" sopts="$2" s="s_$1"
  gql "CALL gnn_sample_drop('$s')" >/dev/null 2>&1 || true
  b=$(gql "CALL gnn_offline_sample('cora','$s',[10,5],{batchSize:64,randomSeed:42,orientation:'UNDIRECTED'${sopts}}) YIELD totalBatches RETURN *"); chk "$b" "${tag}_sample"
  b=$(gql "CALL gnn_build_feature_store('$s','node_features',{gpu_budget_mb:2,cpu_budget_mb:1,buildAddrTables:true,force:1}) YIELD l1Nodes RETURN *"); chk "$b" "${tag}_build"
  local T="epochs:5,randomSeed:42,dropout:0,patience:999,saveOnBestVal:false,saveFinal:false,prefetchNumWorkers:1,prefetchQueueSize:6,lrSchedule:''"
  local out; out=$(gql "CALL gnn_train('$s','node_features',{${T}}) YIELD testAccuracy RETURN *"); chk "$out" "${tag}_train"
  echo "$out" | col testAccuracy
}

ACC_ON=$(run_once on  ",useSymmetricTopology:'on'");  [ "$ACC_ON"  = "NaN" ] && fail "ON: no testAccuracy"
ACC_OFF=$(run_once off ",useSymmetricTopology:'off'"); [ "$ACC_OFF" = "NaN" ] && fail "OFF: no testAccuracy"

echo "ON=$ACC_ON OFF=$ACC_OFF CANON=$CANON"
for pair in "ON:$ACC_ON" "OFF:$ACC_OFF"; do
  name="${pair%%:*}"; val="${pair#*:}"
  python3 -c "import sys; sys.exit(0 if float('$val')==float('$CANON') else 1)" || fail "$name testAcc $val != $CANON"
done
python3 -c "import sys; sys.exit(0 if float('$ACC_ON')==float('$ACC_OFF') else 1)" || fail "ON ($ACC_ON) != OFF ($ACC_OFF)"
echo "[sym_gate] PASS: symmetric ON==OFF==$CANON"
