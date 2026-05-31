#!/usr/bin/env bash
# test_feature_store_staleness_cora.sh — feature-store content-fingerprint staleness gate.
#
# CONTRACT: gnn_build_feature_store must REUSE an existing feature store when
# the sample is unchanged, and RECOMPUTE (never silently reuse) when the
# sample content changes — even when scalar counts could collide. Concretely:
#
#   1. POSITIVE  — rebuild the SAME sample without force  => "reusing", FP unchanged,
#                  reordered.fmat mtime unchanged.
#   2. NEGATIVE(fanout)      — re-sample with a different fanout, rebuild without
#                  force => "recomputing", FP changes.
#   3. NEGATIVE(orientation) — re-sample SAME fanout/seed/batch but a DIFFERENT
#                  orientation => FP changes + "recomputing". This is the case
#                  a count-only fingerprint design would have falsely reused.
#
# After every recompute, gnn_train must complete (shapes consistent) — guarding
# the silent-stale-shape failure this gate exists to catch.
#
# Exit 0 = PASS.
set -u -o pipefail

REPO=/home/bfuentes/MillenniumDB_Testing
MDB="$REPO/build/Release/bin/mdb"
CORA_GQL="$REPO/data/example/gql/cora/cora.gql"
CORA_NPY="$REPO/data/example/gql/cora/cora_features.npy"
PORT=7883
DB=$(mktemp -d /tmp/cora_staleness.XXXXXX)
GNN_DIR="$DB/gnn_features"
FP_FILE="$GNN_DIR/node_features_store.fp"
FMAT="$GNN_DIR/node_features_reordered.fmat"
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
say()  { echo "[staleness] $*"; }
fail() { echo "[staleness] FAIL: $*" >&2; RC=1; exit 1; }
gql()  { curl -s --max-time 180 -X POST "http://localhost:$PORT/gql" \
              -H "Content-Type: text/plain" --data-binary "$1"; }

# Read the uint64 fingerprint at offset 8 of <feature>_store.fp (0 if absent).
read_fp() {
  python3 - "$FP_FILE" <<'PY'
import sys, struct, os
p = sys.argv[1]
if not os.path.exists(p):
    print("0"); raise SystemExit
b = open(p, "rb").read()
if len(b) < 16:
    print("0"); raise SystemExit
print(struct.unpack("<I", b[0:4])[0] == 0x47464650 and struct.unpack("<Q", b[8:16])[0] or 0)
PY
}
mtime() { stat -c %Y "$1" 2>/dev/null || echo 0; }

# Run a build WITHOUT force and classify the server-log verdict for this call
# only (by byte offset). Echoes one of: reuse | recompute | unknown.
build_classify() {
  local off; off=$(wc -c < "$LOG" 2>/dev/null || echo 0)
  local body
  body=$(gql "CALL gnn_build_feature_store('s','node_features',{gpu_budget_mb:0,cpu_budget_mb:100}) YIELD l2Nodes RETURN *")
  echo "$body" | grep -qiE "error|exception|failed" && { say "build body: $body"; echo "error"; return; }
  local newlog; newlog=$(tail -c +$((off + 1)) "$LOG" 2>/dev/null || true)
  if echo "$newlog" | grep -q "reusing existing artifacts"; then echo "reuse"
  elif echo "$newlog" | grep -q "recomputing stale artifacts"; then echo "recompute"
  else echo "unknown"; fi
}

resample() {  # $1=fanout-literal  $2=orientation
  gql "CALL gnn_offline_sample('cora','s',$1,{batchSize:64,randomSeed:42,orientation:'$2',force:true}) YIELD totalBatches RETURN *" \
    | grep -qiE "error|exception|failed" && fail "re-sample ($1,$2) errored"
  gql "CALL gnn_materialize_batches('s','node_features',{reorder:1,numHashes:2,force:1}) YIELD totalBatches RETURN *" \
    | grep -qiE "error|exception|failed" && fail "materialize errored"
  return 0
}

train_ok() {
  local body
  body=$(gql "CALL gnn_train('s','node_features',{epochs:2,randomSeed:42,patience:999,saveOnBestVal:false,saveFinal:false}) YIELD testAccuracy RETURN *")
  echo "$body" | grep -qiE "error|exception|failed" && fail "gnn_train did not complete: $body"
  return 0
}

# --- setup ---
"$MDB" import "$CORA_GQL" "$DB" --with-tensors "$CORA_NPY" >/dev/null 2>&1 || fail "import failed"
for p in $(pgrep -f "build/Release/bin/mdb server .*-p $PORT" 2>/dev/null || true); do kill -9 "$p" 2>/dev/null || true; done
"$MDB" server "$DB" -p $PORT -t 600 --browser false >"$LOG" 2>&1 &
SRVPID=$!
ready=0
for _ in $(seq 1 30); do
  gql "RETURN 1" 2>/dev/null | grep -q 1 && { ready=1; break; }
  kill -0 "$SRVPID" 2>/dev/null || fail "server died (see $LOG)"
  sleep 0.5
done
[ "$ready" = "1" ] || fail "server not ready"

gql "CALL graph_project('cora','Paper','CITES',{orientation:'UNDIRECTED',includeFeatures:'node_features',labelProperty:'label'}) YIELD graphName RETURN *" >/dev/null

# === 0. Initial sample [10,5] UNDIRECTED + first build ===
resample "[10,5]" "UNDIRECTED"
v0=$(build_classify); [ "$v0" = "error" ] && fail "initial build errored"
[ -f "$FP_FILE" ] || fail "store.fp not written after first build"
FP_A=$(read_fp); m_A=$(mtime "$FMAT")
say "first build: verdict=$v0  FP_A=$FP_A  reordered.mtime=$m_A"
[ "$FP_A" = "0" ] && fail "FP_A is 0 (sample content fingerprint not propagated)"

# === 1. POSITIVE — rebuild unchanged sample, no force => reuse ===
v1=$(build_classify)
FP_A2=$(read_fp); m_A2=$(mtime "$FMAT")
say "rebuild-unchanged: verdict=$v1  FP=$FP_A2  reordered.mtime=$m_A2"
[ "$v1" = "reuse" ]    || fail "unchanged rebuild did not REUSE (verdict=$v1)"
[ "$FP_A2" = "$FP_A" ] || fail "FP changed across an unchanged rebuild ($FP_A -> $FP_A2)"
[ "$m_A2" = "$m_A" ]   || fail "reordered.fmat was rewritten on a reuse (mtime $m_A -> $m_A2)"

# === 2. NEGATIVE(fanout) — re-sample [5,5,5], rebuild no force => recompute ===
resample "[5,5,5]" "UNDIRECTED"
v2=$(build_classify); [ "$v2" = "error" ] && fail "fanout-changed build errored"
FP_B=$(read_fp); m_B=$(mtime "$FMAT")
say "fanout-changed: verdict=$v2  FP_B=$FP_B  reordered.mtime=$m_B"
[ "$v2" = "recompute" ] || fail "fanout change did not RECOMPUTE (verdict=$v2)"
[ "$FP_B" != "$FP_A" ]  || fail "FP did not change after a fanout change (still $FP_A)"
train_ok
say "fanout-changed: gnn_train completed OK"

# === 3. NEGATIVE(orientation) — same fanout/seed, flip UNDIRECTED->NATURAL ===
# Establish a known UNDIRECTED baseline at [10,5] first...
resample "[10,5]" "UNDIRECTED"; vU=$(build_classify); FP_U=$(read_fp)
say "undirected [10,5] baseline: verdict=$vU  FP_U=$FP_U"
# ...then re-sample with the SAME fanout/seed/batch but NATURAL orientation.
resample "[10,5]" "NATURAL"
v3=$(build_classify); [ "$v3" = "error" ] && fail "orientation-flip build errored"
FP_C=$(read_fp)
say "orientation-flip (NATURAL): verdict=$v3  FP_C=$FP_C"
[ "$FP_C" != "$FP_U" ]  || fail "orientation flip did NOT change FP — count-only false-positive ($FP_U)"
[ "$v3" = "recompute" ] || fail "orientation flip did not RECOMPUTE (verdict=$v3)"
train_ok
say "orientation-flip: gnn_train completed OK"

say "PASS — reuse-on-match, recompute-on-(fanout|orientation), trains consistent"
RC=0
exit $RC
