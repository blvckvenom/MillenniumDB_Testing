#!/usr/bin/env bash
# symmetric_csr_cora_build_gate.sh — builds the symmetric (pre-merged undirected)
# topology sidecar on a real cora projection and asserts the build-time
# self-verify passes for EVERY node (symStatus=='built'), the sidecar is
# non-empty, and a simple graph is not falsely refused as a multigraph.
#
# The post-hoc bake cross-checks every baked row against the live accessor's
# UNDIRECTED node list and aborts on any mismatch, so symStatus=='built' over
# all 2708 cora nodes is the real correctness proof on non-toy data.
#
# Scope: this gate validates the PRODUCTION of topology_sym.csr. Its CONSUMPTION
# by the sampler and the ON==OFF accuracy parity (testAcc == 0.8574939) are the
# job of scripts/symmetric_csr_cora_gate.sh once the accessor/store/engine wiring
# lands.
#
# Every CALL body is checked for MDB error text (the server returns HTTP 200 +
# error in the body, so curl exit codes are useless).
set -u -o pipefail
REPO=/home/bfuentes/MillenniumDB_Testing
MDB="$REPO/build/Release/bin/mdb"
CORA_GQL="$REPO/data/example/gql/cora/cora.gql"
CORA_NPY="$REPO/data/example/gql/cora/cora_features.npy"
PORT="${PORT:-7887}"
DB=$(mktemp -d /tmp/sym_cora_build.XXXXXX); SRVPID=""
cleanup(){ [ -n "$SRVPID" ] && kill "$SRVPID" 2>/dev/null; rm -rf "$DB" 2>/dev/null; }
trap cleanup EXIT INT TERM
gql(){ curl -s --max-time 120 -X POST "http://localhost:$PORT/gql" -H 'Content-Type: text/plain' --data-binary "$1"; }
fail(){ echo "[sym_build_gate] FAIL: $*" >&2; exit 1; }
chk(){ echo "$1" | grep -qiE "error|exception|failed|out of range" && { echo "  BODY: $1" >&2; fail "$2 error"; }; }
col(){ python3 -c "
import sys
L=[l for l in sys.stdin.read().splitlines() if l.strip()]
if len(L)<2: print('NaN'); sys.exit()
h=L[0].split(','); d=L[-1].split(',')
try: print(d[h.index('$1')].strip().strip('\"'))
except: print('NaN')"; }

"$MDB" import "$CORA_GQL" "$DB" --with-tensors "$CORA_NPY" >/dev/null 2>&1 || fail import
for p in $(pgrep -f "bin/mdb server .*-p $PORT" 2>/dev/null || true); do kill -9 "$p" 2>/dev/null; done
"$MDB" server "$DB" -p $PORT -t 600 --browser false >"$DB/server.log" 2>&1 & SRVPID=$!
for _ in $(seq 1 30); do gql "RETURN 1" 2>/dev/null | grep -q 1 && break; kill -0 $SRVPID 2>/dev/null || fail server_died; sleep 0.5; done

b=$(gql "CALL graph_project('cora','Paper','CITES',{orientation:'UNDIRECTED',includeFeatures:'node_features',labelProperty:'label'}) YIELD nodeCount RETURN *"); chk "$b" project

SB=$(gql "CALL gnn_build_topology_snapshot('cora','symmetric') YIELD symStatus,symBytes,parallelEdgeRefused RETURN *"); chk "$SB" sym_build
STATUS=$(echo "$SB" | col symStatus)
BYTES=$(echo "$SB" | col symBytes)
REFUSED=$(echo "$SB" | col parallelEdgeRefused)
echo "symStatus=$STATUS symBytes=$BYTES parallelEdgeRefused=$REFUSED"

[ "$STATUS" = "built" ] || fail "symStatus=$STATUS (want 'built') — self-verify or eligibility failed"
[ "$REFUSED" = "false" ] || fail "parallelEdgeRefused=$REFUSED (cora is simple, want false)"
python3 -c "import sys; sys.exit(0 if int('$BYTES') > 50000 else 1)" \
    || fail "symBytes=$BYTES too small — the sidecar looks empty (vacuous self-verify)"

echo "[sym_build_gate] PASS: cora symmetric sidecar built + self-verified over all nodes (symBytes=$BYTES)"
