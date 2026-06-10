#!/usr/bin/env bash
# cora_v2_probe.sh — print cora v2-path testAccuracy for a given (workers, odirect).
# Deterministic N=1 is the bit-identical mode. Prints just the float testAcc.
#   WORKERS=1 ODIRECT=0 bash scripts/cora_v2_probe.sh
# Every CALL response body is checked for MDB error text (the server returns
# HTTP 200 + error in the body, so curl exit codes are useless) and the script
# exits 1 instead of printing NaN — a dual-failure A/B can never pass NaN==NaN.
# EXPECT (default auto): with DROPOUT=0 WORKERS=1 the result is also checked
# against the canonical bit-identical constant 0.8574939. EXPECT=skip disables,
# EXPECT=<float> overrides.
set -u -o pipefail
REPO=/home/bfuentes/MillenniumDB_Testing
MDB="$REPO/build/Release/bin/mdb"
CORA_GQL="$REPO/data/example/gql/cora/cora.gql"; CORA_NPY="$REPO/data/example/gql/cora/cora_features.npy"
PORT="${PORT:-7886}"; WORKERS="${WORKERS:-1}"; ODIRECT="${ODIRECT:-0}"; DROPOUT="${DROPOUT:-0.5}"; BAKE="${BAKE:-0}"; PACKFULL="${PACKFULL:-0}"; EXPECT="${EXPECT:-auto}"
[ "$PACKFULL" = "1" ] && BAKE=1
DB=$(mktemp -d /tmp/cora_v2probe.XXXXXX); SRVPID=""
cleanup(){ [ -n "$SRVPID" ] && kill "$SRVPID" 2>/dev/null; [ -n "$SRVPID" ] && wait "$SRVPID" 2>/dev/null; rm -rf "$DB" 2>/dev/null; }
trap cleanup EXIT INT TERM
gql(){ curl -s --max-time 120 -X POST "http://localhost:$PORT/gql" -H "Content-Type: text/plain" --data-binary "$1"; }
fail(){ echo "[v2_probe] FAIL: $*" >&2; exit 1; }
chk(){ echo "$1" | grep -qiE "error|exception|failed|out of range|corresponding feature row" && { echo "  BODY: $1" >&2; fail "$2 error"; }; }
col(){ python3 -c "
import sys
L=[l for l in sys.stdin.read().splitlines() if l.strip()]
if len(L)<2: print('NaN'); sys.exit()
h=L[0].split(','); d=L[-1].split(',')
try: print(repr(float(d[h.index('$1')])))
except: print('NaN')"; }
acc_check(){ # $1=acc $2=label $3=body — refuse NaN; canonical check when applicable
  if [ "$1" = "NaN" ]; then echo "  BODY: ${3:-}" >&2; fail "$2: no testAccuracy in response"; fi
  local exp=""
  case "$EXPECT" in
    auto) case "$DROPOUT" in 0|0.0) [ "$WORKERS" = "1" ] && exp=0.8574939 ;; esac ;;
    ""|skip|off) ;;
    *) exp="$EXPECT" ;;
  esac
  [ -z "$exp" ] && return 0
  python3 -c "import sys; sys.exit(0 if float('$1') == float('$exp') else 1)" \
    || fail "$2: testAccuracy=$1 != expected $exp (canonical N=1 dropout=0 gate)"
}
"$MDB" import "$CORA_GQL" "$DB" --with-tensors "$CORA_NPY" >/dev/null 2>&1 || { echo IMPORT_FAIL; exit 1; }
for p in $(pgrep -f "bin/mdb server .*-p $PORT" 2>/dev/null||true); do kill -9 "$p" 2>/dev/null; done
MDB_GNN_L4_O_DIRECT="$ODIRECT" "$MDB" server "$DB" -p $PORT -t 600 --browser false >"$DB/server.log" 2>&1 & SRVPID=$!
for _ in $(seq 1 30); do gql "RETURN 1" 2>/dev/null|grep -q 1 && break; kill -0 $SRVPID 2>/dev/null||{ echo SERVER_DIED; exit 1; }; sleep 0.5; done
b=$(gql "CALL graph_project('cora','Paper','CITES',{orientation:'UNDIRECTED',includeFeatures:'node_features',labelProperty:'label'}) YIELD nodeCount RETURN *"); chk "$b" project
b=$(gql "CALL gnn_offline_sample('cora','s',[10,5],{batchSize:64,randomSeed:42,orientation:'UNDIRECTED'}) YIELD totalBatches RETURN *"); chk "$b" sample
b=$(gql "CALL gnn_materialize_batches('s','node_features',{reorder:1,numHashes:2,force:1}) YIELD totalBatches RETURN *"); chk "$b" materialize
BAKEOPT=""; [ "$BAKE" = "1" ] && BAKEOPT=",bakeBlocks:true"
BR=$(gql "CALL gnn_build_feature_store('s','node_features',{gpu_budget_mb:2,cpu_budget_mb:1,buildAddrTables:true,force:1${BAKEOPT}}) YIELD l1Nodes,l2Nodes,l3Nodes,l4Nodes,blocksMb,blocksBuiltOk RETURN *"); chk "$BR" build
[ "$BAKE" = "1" ] && echo "build(BAKE=1): $(echo "$BR" | tail -1)" >&2
if [ "$PACKFULL" = "1" ]; then
  PFR=$(gql "CALL gnn_build_feature_store('s','node_features',{packFullFeatures:true}) YIELD packedFullMb RETURN *"); chk "$PFR" packfull_build
  echo "packed_full build: $(echo "$PFR" | tail -1)" >&2
fi
TRAINOPTS="epochs:5,randomSeed:42,dropout:${DROPOUT},patience:999,saveOnBestVal:false,saveFinal:false,prefetchNumWorkers:${WORKERS},prefetchQueueSize:6"
if [ "$PACKFULL" = "1" ]; then
  OUT_ON=$(gql "CALL gnn_train('s','node_features',{${TRAINOPTS}}) YIELD testAccuracy RETURN *"); chk "$OUT_ON" train_packfull_on
  MODE_ON=$(grep -i "feature-load mode" "$DB/server.log" 2>/dev/null | tail -1)
  OUT_OFF=$(gql "CALL gnn_train('s','node_features',{${TRAINOPTS},noPackedFull:true}) YIELD testAccuracy RETURN *"); chk "$OUT_OFF" train_packfull_off
  MODE_OFF=$(grep -i "feature-load mode" "$DB/server.log" 2>/dev/null | tail -1)
  echo "[mode ON ] $MODE_ON"  >&2
  echo "[mode OFF] $MODE_OFF" >&2
  A_ON=$(echo "$OUT_ON" | col testAccuracy);   acc_check "$A_ON"  packfull_on  "$OUT_ON"
  A_OFF=$(echo "$OUT_OFF" | col testAccuracy); acc_check "$A_OFF" packfull_off "$OUT_OFF"
  echo "PACKFULL_ON=$A_ON"
  echo "PACKFULL_OFF=$A_OFF"
else
  OUT=$(gql "CALL gnn_train('s','node_features',{${TRAINOPTS}}) YIELD testAccuracy RETURN *"); chk "$OUT" train
  grep -i "feature-load mode" "$DB/server.log" 2>/dev/null | tail -1 | sed 's/^/[mode] /' >&2 || true
  ACC=$(echo "$OUT" | col testAccuracy); acc_check "$ACC" train "$OUT"
  echo "$ACC"
fi
