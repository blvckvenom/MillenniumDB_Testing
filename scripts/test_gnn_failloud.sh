#!/usr/bin/env bash
# test_gnn_failloud.sh — STEP 4 negative tests: a GNN-intent graph_project must
# FAIL LOUD (not silently produce a broken projection) when:
#   (1) includeFeatures is set but the prerequisite <feature>.rmap is absent;
#   (2) labelProperty is set but no node yields an integer label;
#   (3) splitProperty is set but no node has a recognized split token.
# Also asserts the happy path (real feature + real label, no bogus split) STILL
# succeeds — i.e. the guards do not reject valid GNN projections.
#
# Exit 0 = PASS (3 bad configs errored + 1 good config succeeded).
set -u -o pipefail

REPO=/home/bfuentes/MillenniumDB_Testing
MDB="$REPO/build/Release/bin/mdb"
CORA_GQL="$REPO/data/example/gql/cora/cora.gql"
CORA_NPY="$REPO/data/example/gql/cora/cora_features.npy"
PORT=7879
DB=$(mktemp -d /tmp/cora_failloud.XXXXXX)
SRVPID=""
RC=1

cleanup() {
  [ -n "$SRVPID" ] && kill "$SRVPID" 2>/dev/null || true
  [ -n "$SRVPID" ] && wait "$SRVPID" 2>/dev/null || true
  rm -rf "$DB" 2>/dev/null || true
  exit $RC
}
trap cleanup EXIT INT TERM
say()  { echo "[failloud] $*"; }
fail() { echo "[failloud] FAIL: $*" >&2; RC=1; exit 1; }

gql() {  # gql <query> -> echoes body
  curl -s --max-time 120 -X POST "http://localhost:$PORT/gql" \
       -H "Content-Type: text/plain" --data-binary "$1"
}
is_err() { echo "$1" | grep -qiE "(error|exception|failed|not found|does not exist|was set but|was requested|already exists|out of range|unavailable)"; }

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

# --- Guard 1: bogus feature name -> no .rmap -> must error ---
B1=$(gql "CALL graph_project('g1','Paper','CITES',{orientation:'UNDIRECTED',includeFeatures:'no_such_feature'}) YIELD graphName RETURN *")
say "guard1 (missing .rmap): $B1"
is_err "$B1" || fail "guard1 did NOT error on missing .rmap"

# --- Guard 2: real feature + bogus label property -> no classes -> must error ---
B2=$(gql "CALL graph_project('g2','Paper','CITES',{orientation:'UNDIRECTED',includeFeatures:'node_features',labelProperty:'no_such_label'}) YIELD graphName RETURN *")
say "guard2 (no label matched): $B2"
is_err "$B2" || fail "guard2 did NOT error on unmatched labelProperty"

# --- Guard 3: real feature + bogus split property -> all UNLABELED -> must error ---
B3=$(gql "CALL graph_project('g3','Paper','CITES',{orientation:'UNDIRECTED',includeFeatures:'node_features',splitProperty:'no_such_split'}) YIELD graphName RETURN *")
say "guard3 (no split matched): $B3"
is_err "$B3" || fail "guard3 did NOT error on unmatched splitProperty"

# --- Happy path: real feature + real label, no split -> must SUCCEED ---
B4=$(gql "CALL graph_project('g4','Paper','CITES',{orientation:'UNDIRECTED',includeFeatures:'node_features',labelProperty:'label'}) YIELD graphName, nodeCount, featureDim, numClasses RETURN *")
say "happy path: $B4"
is_err "$B4" && fail "happy-path GNN projection was wrongly rejected"
echo "$B4" | grep -q "g4" || fail "happy-path projection did not return its name"

say "PASS — 3 guards fired + happy path succeeded"
RC=0
exit 0
