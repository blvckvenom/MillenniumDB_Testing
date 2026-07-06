#!/usr/bin/env bash
# test_topology_uint32_cora.sh — cora losslessness gate for the Spec #6 uint32
# topology CSR sidecar.
#
# Proves on REAL cora topology that the NARROW (uint32, tag-stripped) sidecar
# reconstructs byte-for-byte the SAME adjacency (ROW_PTR + tagged COL_IDX +
# tagged EDGE_IDS) as the legacy WIDE (uint64) sidecar.
#
# Why not compare batches.dat: gnn_offline_sample's batches.dat is only
# SEMANTICALLY deterministic (uniqueNodes / totalBatches match), not
# byte-deterministic across invocations (verified: wide-vs-wide already
# differs). So the rigorous, deterministic losslessness target is the sidecar
# adjacency itself, compared element-wise. Both sidecars are built via the
# SAME sequential procedure (gnn_build_topology_snapshot) so the only variable
# is id_width — any element-wise difference is a width bug, not a build-path
# ordering artifact.
#
# Method (A/B on ONE projection, same build path):
#   project (no snapshot) -> server env OFF: gnn_build_topology_snapshot => WIDE
#     sidecar; save a copy. -> server env ON: gnn_build_topology_snapshot =>
#     NARROW sidecar (overwrites in place).
#   Gate: id_width 8 -> 4, narrow file smaller, AND the reconstructed
#         (ROW_PTR, tagged COL_IDX, tagged EDGE_IDS) of narrow == wide,
#         element-for-element, for BOTH directions. Plus a smoke sample on the
#         narrow sidecar that must complete with the same uniqueNodes/batches.
#
# Exit 0 = PASS. Non-zero = FAIL.
set -u -o pipefail

REPO=/home/bfuentes/MillenniumDB_Testing
MDB="$REPO/build/Release/bin/mdb"
CORA_GQL="$REPO/data/example/gql/cora/cora.gql"
CORA_NPY="$REPO/data/example/gql/cora/cora_features.npy"
PORT=7889
KEEP=0
[ "${1-}" = "--keep" ] && KEEP=1

DB=$(mktemp -d /tmp/cora_u32_gate.XXXXXX)
SRVPID=""
RC=1

cleanup() {
  [ -n "$SRVPID" ] && kill "$SRVPID" 2>/dev/null || true
  [ -n "$SRVPID" ] && wait "$SRVPID" 2>/dev/null || true
  if [ "$KEEP" = "0" ]; then rm -rf "$DB" 2>/dev/null || true
  else echo "[u32_gate] DB left at $DB"; fi
  exit $RC
}
trap cleanup EXIT INT TERM

say()  { echo "[u32_gate] $*"; }
fail() { echo "[u32_gate] FAIL: $*" >&2; RC=1; exit 1; }

run_gql() {
  local name=$1 query=$2 body
  body=$(curl -s --max-time 180 -X POST "http://localhost:$PORT/gql" \
               -H "Content-Type: text/plain" --data-binary "$query")
  echo "  [$name] $body" >&2
  if echo "$body" | grep -qiE "(error|exception|failed|not found|bad query|unexpected|out of range)"; then
    fail "$name returned an error response (see above)"
  fi
  echo "$body"
}

start_server() {
  local narrow=$1
  for p in $(pgrep -f "build/Release/bin/mdb server .*-p $PORT" 2>/dev/null || true); do
    kill -9 "$p" 2>/dev/null || true
  done
  if [ "$narrow" = "1" ]; then
    MDB_GNN_TOPOLOGY_UINT32=1 "$MDB" server "$DB" -p $PORT -t 600 --browser false \
      >"$DB/server_narrow.log" 2>&1 &
  else
    "$MDB" server "$DB" -p $PORT -t 600 --browser false \
      >"$DB/server_wide.log" 2>&1 &
  fi
  SRVPID=$!
  local ready=0
  for _ in $(seq 1 40); do
    if curl -s --max-time 2 -X POST "http://localhost:$PORT/gql" \
          -H "Content-Type: text/plain" --data-binary "RETURN 1" 2>/dev/null | grep -q 1; then
      ready=1; break
    fi
    kill -0 "$SRVPID" 2>/dev/null || fail "server died on startup"
    sleep 0.5
  done
  [ "$ready" = "1" ] || fail "server not ready after 20s"
}

stop_server() {
  [ -n "$SRVPID" ] && kill "$SRVPID" 2>/dev/null || true
  [ -n "$SRVPID" ] && wait "$SRVPID" 2>/dev/null || true
  SRVPID=""
  sleep 0.5
}

id_width_of() { python3 -c "
with open('$1','rb') as f: f.seek(12); b=f.read(1)
print(b[0] if b else -1)
"; }

[ -x "$MDB" ]      || fail "mdb binary not found at $MDB"
[ -f "$CORA_GQL" ] || fail "cora.gql not found"
[ -f "$CORA_NPY" ] || fail "cora_features.npy not found"

say "import cora -> $DB"
"$MDB" import "$CORA_GQL" "$DB" --with-tensors "$CORA_NPY" >/dev/null 2>&1 \
  || fail "import failed"

SAMPLE_FLAGS="batchSize:64,randomSeed:42,orientation:'UNDIRECTED',useFourLevelTopologyStore:true,useL3MmapSidecar:true"

# ----- Phase A: project (no snapshot) + WIDE sidecar via the procedure -----
say "Phase A — server env OFF: project + gnn_build_topology_snapshot (wide)"
start_server 0
run_gql project "CALL graph_project('cora','Paper','CITES',{orientation:'UNDIRECTED',includeFeatures:'node_features',labelProperty:'label'}) YIELD graphName, nodeCount, relationshipCount RETURN *" >/dev/null
run_gql build_wide "CALL gnn_build_topology_snapshot('cora') YIELD graphName RETURN *" >/dev/null
WIDE_SAMPLE=$(run_gql sample_wide "CALL gnn_offline_sample('cora','s_wide',[5,10],{$SAMPLE_FLAGS}) YIELD totalBatches, uniqueNodes RETURN *")
stop_server

PROJ_DIR=$(dirname "$(find "$DB" -name topology_fwd.csr 2>/dev/null | head -1)")
[ -n "$PROJ_DIR" ] || fail "could not locate projection dir"
say "projection dir: $PROJ_DIR"
cp "$PROJ_DIR/topology_fwd.csr" "$DB/fwd.wide.csr"
cp "$PROJ_DIR/topology_rev.csr" "$DB/rev.wide.csr"
WIDE_IDW=$(id_width_of "$DB/fwd.wide.csr")
[ "$WIDE_IDW" = "8" ] || fail "Phase A sidecar id_width=$WIDE_IDW, expected 8"
WIDE_FWD_SZ=$(stat -c %s "$DB/fwd.wide.csr")

# ----- Phase B: NARROW sidecar via the SAME procedure -----
say "Phase B — server env MDB_GNN_TOPOLOGY_UINT32=1: gnn_build_topology_snapshot (narrow)"
start_server 1
run_gql build_narrow "CALL gnn_build_topology_snapshot('cora') YIELD graphName RETURN *" >/dev/null
NARROW_IDW=$(id_width_of "$PROJ_DIR/topology_fwd.csr")
[ "$NARROW_IDW" = "4" ] || fail "Phase B sidecar id_width=$NARROW_IDW, expected 4 (env opt-in not honored?)"
NARROW_SAMPLE=$(run_gql sample_narrow "CALL gnn_offline_sample('cora','s_narrow',[5,10],{$SAMPLE_FLAGS}) YIELD totalBatches, uniqueNodes RETURN *")
stop_server

NARROW_FWD_SZ=$(stat -c %s "$PROJ_DIR/topology_fwd.csr")
say "sidecar fwd bytes: wide=$WIDE_FWD_SZ narrow=$NARROW_FWD_SZ"
[ "$NARROW_FWD_SZ" -lt "$WIDE_FWD_SZ" ] || fail "narrow sidecar not smaller than wide ($NARROW_FWD_SZ >= $WIDE_FWD_SZ)"

# Semantic sample invariants (these ARE deterministic across invocations).
WS=$(echo "$WIDE_SAMPLE"   | tail -1)
NS=$(echo "$NARROW_SAMPLE" | tail -1)
say "sample(wide)=$WS  sample(narrow)=$NS"
[ "$WS" = "$NS" ] || fail "sample semantic counts differ: wide='$WS' narrow='$NS'"

# ----- Rigorous element-wise adjacency comparison (the real losslessness gate) -----
say "comparing reconstructed adjacency narrow-vs-wide (both directions)"
python3 - "$DB/fwd.wide.csr" "$PROJ_DIR/topology_fwd.csr" \
          "$DB/rev.wide.csr" "$PROJ_DIR/topology_rev.csr" <<'PY'
import sys, struct
def parse(path):
    with open(path,'rb') as f:
        hdr=f.read(64); body=f.read()
    idw=hdr[12]; flags=hdr[13]; dst_tag=hdr[14]; eid_tag=hdr[15]
    N=struct.unpack_from('<Q',hdr,16)[0]; M=struct.unpack_from('<Q',hdr,24)[0]
    has_eid=flags & 1
    row=list(struct.unpack_from('<%dQ'%(N+1), body, 0)); off=(N+1)*8
    if idw==8:
        col=list(struct.unpack_from('<%dQ'%M, body, off)); off+=M*8
        eid=list(struct.unpack_from('<%dQ'%M, body, off)) if has_eid else []
    elif idw==4:
        craw=struct.unpack_from('<%dI'%M, body, off); off+=M*4
        col=[(dst_tag<<56)|c for c in craw]
        if has_eid:
            eraw=struct.unpack_from('<%dI'%M, body, off)
            eid=[(eid_tag<<56)|e for e in eraw]
        else: eid=[]
    else:
        raise SystemExit("bad id_width %d in %s"%(idw,path))
    return idw,N,M,row,col,eid

def cmp_dir(wide_path, narrow_path, label):
    wi,wN,wM,wrow,wcol,weid = parse(wide_path)
    ni,nN,nM,nrow,ncol,neid = parse(narrow_path)
    assert wi==8 and ni==4, "%s: id_width wide=%d narrow=%d"%(label,wi,ni)
    if (wN,wM)!=(nN,nM): raise SystemExit("%s: N/M mismatch w=(%d,%d) n=(%d,%d)"%(label,wN,wM,nN,nM))
    if wrow!=nrow:       raise SystemExit("%s: ROW_PTR differs"%label)
    if wcol!=ncol:
        # find first diff for diagnostics
        for i,(a,b) in enumerate(zip(wcol,ncol)):
            if a!=b: raise SystemExit("%s: COL_IDX[%d] wide=%d narrow=%d"%(label,i,a,b))
        raise SystemExit("%s: COL_IDX length differs"%label)
    if weid!=neid:
        for i,(a,b) in enumerate(zip(weid,neid)):
            if a!=b: raise SystemExit("%s: EDGE_IDS[%d] wide=%d narrow=%d"%(label,i,a,b))
        raise SystemExit("%s: EDGE_IDS length differs"%label)
    print("  %s: N=%d M=%d  ROW_PTR/COL_IDX/EDGE_IDS element-identical (wide uint64 == narrow uint32)"%(label,wN,wM))

cmp_dir(sys.argv[1], sys.argv[2], "fwd")
cmp_dir(sys.argv[3], sys.argv[4], "rev")
print("ADJACENCY_LOSSLESS_OK")
PY
PYRC=$?
[ "$PYRC" = "0" ] || fail "adjacency comparison failed (narrow sidecar is NOT lossless)"

say "PASS — narrow(uint32) sidecar reconstructs wide(uint64) adjacency exactly; sample semantics match; id_width 8->4; narrow smaller"
RC=0
exit $RC
