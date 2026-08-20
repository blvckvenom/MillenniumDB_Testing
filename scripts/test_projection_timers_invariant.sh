#!/usr/bin/env bash
# test_projection_timers_invariant.sh — gate on the MDB_BENCHMARK=1 phase
# breakdown emitted by graph_project (ProjectionTimers).
#
# WHY THIS EXISTS
#
# The breakdown is the entry point of every projection cost model: it is what
# says which phase to attack. It has silently reported impossible numbers
# before, and the failure was invisible because nothing checked it:
#
#   - percentages summing to 106.2%, because the total was anchored inside
#     finalize() and did not contain the two scans that run before it;
#   - a phase reported as half of an interval by a hardcoded 0.5, i.e. a
#     constant printed in the position of a measurement;
#   - the serialized scan mode double-counting, because its scans run inside
#     the interval attributed to sort+index-build.
#
# Each of those inflates or deflates a phase's share, which is exactly the
# quantity used to decide what to optimize. A breakdown that cannot be trusted
# is worse than none.
#
# WHAT IT ASSERTS  (per scan mode: classic, serialized, parallel)
#
#   A1 containment  residual >= 0. A negative residual means two timers covered
#                   the same interval.
#   A2 closure      the printed percentages sum to 100 within rounding.
#   A3 no warning   the builder did not emit its own overlap warning.
#   A4 liveness     total and edge_scan are both strictly positive, so a gate
#                   cannot pass by having every timer silently report zero.
#
# The parser DISCOVERS the phase names rather than hardcoding them: any line of
# the form "  <name>:  <ms> ms  (<pct>%)" is a phase, `total` is the
# denominator and `residual` is the slack. Adding a phase timer later is
# therefore covered by this gate automatically, with no edit here.
#
# Usage:  scripts/test_projection_timers_invariant.sh [--keep]
# Exit 0 = all modes pass.

set -u -o pipefail

REPO=/home/bfuentes/MillenniumDB_Testing
MDB="$REPO/build/Release/bin/mdb"
CORA_GQL="$REPO/data/example/gql/cora/cora.gql"
CORA_NPY="$REPO/data/example/gql/cora/cora_features.npy"
PORT=7893
KEEP=0
[ "${1-}" = "--keep" ] && KEEP=1

DB=$(mktemp -d /tmp/proj_timers.XXXXXX)
SRVPID=""
RC=1
FAILURES=0

cleanup() {
  [ -n "$SRVPID" ] && kill "$SRVPID" 2>/dev/null || true
  [ -n "$SRVPID" ] && wait "$SRVPID" 2>/dev/null || true
  if [ "$KEEP" = "0" ]; then rm -rf "$DB" 2>/dev/null || true
  else echo "[keep] DB left at $DB"; fi
  exit $RC
}
trap cleanup EXIT INT TERM

say()  { echo "[timers] $*"; }
fail() { echo "[timers] FAIL: $*" >&2; RC=1; exit 1; }

[ -x "$MDB" ]      || fail "mdb binary not found at $MDB"
[ -f "$CORA_GQL" ] || fail "cora.gql not found at $CORA_GQL"

say "import cora -> $DB"
"$MDB" import "$CORA_GQL" "$DB" --with-tensors "$CORA_NPY" >/dev/null 2>&1 \
  || fail "import failed"

# ---------------------------------------------------------------------------
# check_mode <label> <env-assignments...>
#   Starts a server with MDB_BENCHMARK=1 plus the given mode variables, projects
#   cora, and runs the invariants over the [BENCHMARK] block it printed.
# ---------------------------------------------------------------------------
check_mode() {
  local label=$1; shift
  local log="$DB/srv_${label}.log"

  for p in $(pgrep -f "build/Release/bin/mdb server .*-p $PORT" 2>/dev/null || true); do
    kill -9 "$p" 2>/dev/null || true
  done
  sleep 0.3

  env MDB_BENCHMARK=1 "$@" \
    "$MDB" server "$DB" -p $PORT -t 600 --browser false >"$log" 2>&1 &
  SRVPID=$!

  local ready=0
  for _ in $(seq 1 40); do
    if curl -s --max-time 2 -X POST "http://localhost:$PORT/gql" \
          -H "Content-Type: text/plain" --data-binary "RETURN 1" 2>/dev/null | grep -q 1; then
      ready=1; break
    fi
    kill -0 "$SRVPID" 2>/dev/null || { echo "[timers] server died ($label), see $log" >&2; return 1; }
    sleep 0.5
  done
  [ "$ready" = "1" ] || { echo "[timers] server not ready ($label)" >&2; return 1; }

  # A distinct projection name per mode: graph_project throws if it exists.
  local body
  body=$(curl -s --max-time 300 -X POST "http://localhost:$PORT/gql" \
    -H "Content-Type: text/plain" --data-binary \
    "CALL graph_project('cora_$label','Paper','CITES',{orientation:'UNDIRECTED',includeFeatures:'node_features',labelProperty:'label'}) YIELD graphName, nodeCount, relationshipCount RETURN *")
  # curl exits 0 on a server-side error, so the body has to be inspected.
  if echo "$body" | grep -qiE "(error|exception|failed|already exists|not found)"; then
    echo "[timers] graph_project failed ($label): $body" >&2
    kill "$SRVPID" 2>/dev/null; wait "$SRVPID" 2>/dev/null; SRVPID=""
    return 1
  fi

  kill "$SRVPID" 2>/dev/null; wait "$SRVPID" 2>/dev/null; SRVPID=""

  python3 - "$log" "$label" <<'PY'
import re, sys

log, label = sys.argv[1], sys.argv[2]
text = open(log, errors="replace").read()

# "[BENCHMARK]   edge_scan:      123.4 ms  ( 45.6%)" -> ("edge_scan", 123.4, 45.6)
row = re.compile(r"^\[BENCHMARK\]\s+([A-Za-z_+]+):\s+(-?[\d.]+) ms\s+\(\s*(-?[\d.]+)%\)", re.M)
rows = {m.group(1): (float(m.group(2)), float(m.group(3))) for m in row.finditer(text)}

if not rows:
    print(f"  [{label}] FAIL: no [BENCHMARK] block in the server log")
    sys.exit(1)
if "total" not in rows:
    print(f"  [{label}] FAIL: block has no 'total' row")
    sys.exit(1)

total_ms, _ = rows["total"]
residual_ms, residual_pct = rows.get("residual", (None, None))
phases = {k: v for k, v in rows.items() if k not in ("total", "residual")}

print(f"  [{label}] total={total_ms:.1f} ms over {len(phases)} phases: "
      + ", ".join(f"{k}={v[1]:.1f}%" for k, v in sorted(phases.items())))

bad = []

# A1 containment
if residual_ms is None:
    bad.append("no 'residual' row: unaccounted time is invisible again")
elif residual_ms < -0.5:  # -0.5 ms absorbs float noise, not a real overlap
    bad.append(f"A1 containment: residual {residual_ms:.1f} ms < 0 "
               f"(two timers cover the same interval)")

# A2 closure -- five rows printed at 0.1% precision can drift half a tenth each
if residual_pct is not None:
    s = sum(v[1] for v in phases.values()) + residual_pct
    if abs(s - 100.0) > 0.35:
        bad.append(f"A2 closure: percentages sum to {s:.2f}, not 100")

# A3 no self-reported overlap
if "WARNING" in text and "negative residual" in text:
    bad.append("A3: builder emitted its own overlap warning")

# A4 liveness -- a breakdown of all-zero timers must not pass
if total_ms <= 0:
    bad.append("A4 liveness: total is not positive")
if phases.get("edge_scan", (0, 0))[0] <= 0:
    bad.append("A4 liveness: edge_scan is zero, the timer never fired")

if bad:
    for b in bad:
        print(f"  [{label}] FAIL: {b}")
    sys.exit(1)
print(f"  [{label}] residual={residual_ms:.1f} ms ({residual_pct:.1f}%) -- A1-A4 pass")
PY
}

# The three modes are governed by init_scan_mode(). They are checked separately
# because the timers nest differently in each: the serialized path runs its
# scans inside the interval attributed to sort+index-build, which is precisely
# the overlap A1 exists to catch.
say "mode classic (default)"
check_mode classic    MDB_PROJECTION_SERIAL_SCAN=0 || FAILURES=$((FAILURES+1))
say "mode serialized"
check_mode serialized MDB_PROJECTION_SERIAL_SCAN=1 || FAILURES=$((FAILURES+1))
say "mode parallel"
check_mode parallel   MDB_PROJECTION_PARALLEL_SCAN=1 || FAILURES=$((FAILURES+1))

if [ "$FAILURES" -eq 0 ]; then
  say "PASS: the phase breakdown holds its invariants in all three scan modes"
  RC=0
else
  say "FAIL: $FAILURES of 3 modes broke an invariant"
  RC=1
fi
