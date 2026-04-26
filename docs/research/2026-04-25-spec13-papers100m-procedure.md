# Spec #13 papers100M validation procedure (T13.16)

**Status**: PROCEDURE — to be executed on the celebi bench machine.
**Scope**: papers100M (`ogbn-papers100M`, ~111 M nodes, ~1.6 B undirected edges).
**Date authored**: 2026-04-25
**Reference design**: `docs/superpowers/specs/2026-04-25-four-level-topology-store-design.md` §4
**Reference plan**: `docs/superpowers/plans/2026-04-25-four-level-topology-store-plan.md` §T13.16
**Reference ADR**: `Partial_Idea/decisions/010_four_level_topology_store.md`

papers100M is **never** projected on the local commodity box (`benito_pc`); it
lives only on celebi, the user's bench machine. The local bench harness
`scripts/bench_four_level_topology.sh` refuses papers100M with
`exit 3`. This document is the celebi-side counterpart of T13.16: a
recipe for running, capturing, and grading the Spec #13 measurement
against the design's projection.

---

## 1. Pre-requisites

On celebi:

1. **Source tree**: `feature-GNN` branch checked out at the commit that
   includes Spec #13 Phase 0-4 + the present Phase 6 bench harness.
   ```bash
   git fetch && git checkout feature-GNN
   git log --oneline | head -5
   ```
   The HEAD must include the `bench(gnn): four-level-topology bench …`
   commit (Phase 6 deliverable).

2. **Release build**:
   ```bash
   cmake -B build/Release -D CMAKE_BUILD_TYPE=Release
   cmake --build build/Release --target mdb -j $(nproc)
   build/Release/bin/mdb help   # smoke
   ```

3. **Imported papers100M database** at `data/dbs/gql/papers100M`.
   The database must have been imported with `mdb import --with-tensors`
   so the `feat` tensor property is available for downstream training
   (Spec #13 itself only needs topology, but the procedure measures the
   full sample-side pipeline).
   ```bash
   ls -lh data/dbs/gql/papers100M/{catalog.dat,*.leaf} | head
   du -sh data/dbs/gql/papers100M
   ```

4. **Hardware budget**: ≥ 30 GB available RAM and ≥ 200 GB free disk.
   ```bash
   free -h            # MemAvailable should report ≥ 30 GB
   df -h .            # at least 200 GB free
   nproc              # for parallelism reporting
   ```

5. **Tooling**: `curl`, `awk`, `sed`, `grep`. `jq` is optional.

---

## 2. Build the projection

papers100M is too large to project on commodity hardware in any other
configuration than `indexSet:'GNN_MINIMAL'`, and Spec #13 specifically
needs the Spec #4-B sidecar as its L3 tier. Run:

```bash
build/Release/bin/mdb server data/dbs/gql/papers100M --port 19990 \
    --timeout 86400 &
SRV_PID=$!

# Wait for HTTP up
for _ in $(seq 1 120); do
    curl -sSf -o /dev/null --data-binary "RETURN 1" \
        -H "Accept: text/csv" "http://127.0.0.1:19990/" 2>/dev/null && break
    sleep 1
done

curl -sS --max-time 86400 \
    --data-binary "CALL graph_project('papers100M_proj', 'Paper', 'CITES', {
        orientation: 'NATURAL',
        indexSet: 'GNN_MINIMAL',
        leafFormat: 'DELTA_VARINT',
        graphStorage: 'BTREE',
        buildTopologySnapshot: true,
        includeFeatures: 'feat',
        labelProperty: 'label',
        splitProperty: 'split'
    }) YIELD graphName, nodeCount, relationshipCount, topologySnapshotBytes
       RETURN *" \
    -H "Accept: text/csv" "http://127.0.0.1:19990/" \
    | tee /tmp/papers100m_projection.csv
```

**Why these knobs:**
- `indexSet: 'GNN_MINIMAL'` — drops the 5 indexes the GNN pipeline does
  not need. Saves ~60% disk and ~50% wall-clock per ADR 005.
- `leafFormat: 'DELTA_VARINT'` — ~80% leaf size reduction at zero read
  cost (Spec #5).
- `graphStorage: 'BTREE'` — **NOT** `CSR_HYBRID`. Spec #4-B sidecar is
  incompatible with `CSR_HYBRID` per `project_procedure.cc:275`. The
  sidecar IS the L3 tier in Spec #13, so it is mandatory here.
- `buildTopologySnapshot: true` — emits `topology_fwd.csr` +
  `topology_rev.csr`, which the Four-Level Store opens as L3 mmap.
- `includeFeatures` / `labelProperty` / `splitProperty` — required
  downstream by `gnn_offline_sample` + training. Adjust property names
  to whatever your papers100M import used.

**Expected build wall-clock**: 60-90 min on celebi-class hardware
(dominated by leaf sort + sidecar emission). Capture from
`/tmp/papers100m_projection.csv` once the curl returns.

---

## 3. Run the Four-Level sample

After the projection finishes:

```bash
curl -sS --max-time 86400 \
    --data-binary "CALL gnn_offline_sample('papers100M_proj', 'samples_spec13',
        [10, 5], {
        batchSize: 1024,
        randomSeed: 42,
        orientation: 'UNDIRECTED',
        useFourLevelTopologyStore: true,
        useAdjacencyCache: true,
        useL3MmapSidecar: true
        -- l1CacheMb / l2CacheMb left unset → auto-detect via /proc/meminfo
    }) YIELD sampleName, totalBatches, trainBatches, validationBatches,
             testBatches, uniqueNodes, computeMillis
       RETURN *" \
    -H "Accept: text/csv" "http://127.0.0.1:19990/" \
    | tee /tmp/papers100m_sample.csv
```

**Auto-detected budgets** on a 30 GB-RAM celebi node (per design §2.2:
70% of available RAM as cache budget, of which 25%/75% to L1/L2):
- L1 ≈ 5 GB → ~5 % of nodes (top hubs by degree)
- L2 ≈ 14 GB → ~15 % of nodes (warm tier)
- L3 → ~80 % of nodes resident only in mmap sidecar
- L4 → 0 (B+Tree is the L3 fallback, not actively dispatched
  unless the sidecar is missing)

If `MemAvailable` differs, override explicitly:

```
useFourLevelTopologyStore: true,
useAdjacencyCache: true,
useL3MmapSidecar: true,
l1CacheMb: 5120,
l2CacheMb: 14336
```

For comparison, run the **Camino-D-only** baseline (sidecar mmap, no
in-RAM tiers):

```
{ batchSize: 1024, randomSeed: 42, orientation: 'UNDIRECTED',
  useFourLevelTopologyStore: false, useAdjacencyCache: false }
```

The Spec #11 mode (`useAdjacencyCache: true,
useFourLevelTopologyStore: false`) is **NOT** runnable at papers100M
scale on commodity hardware — its monolithic RAM hash needs ~80 GB.
Document the OOM if attempted; do not block on it.

---

## 4. Logs and metrics to capture

While the sample runs, capture in parallel:

1. **`[FourLevelTopologyStore]` build line** (stderr from server):
   ```
   FourLevelTopologyStore: built — l1_fwd=… l2_fwd=… l1_rev=… l2_rev=…
                                  l3_fwd=… l3_rev=… ram_used=… bytes
   ```
   Emitted once per build. Confirms tier population matched the auto-
   detected budgets. Log via:
   ```bash
   grep -E '^FourLevelTopologyStore: built' /tmp/papers100m_server.log
   ```

2. **Peak RSS** of the mdb server process (kernel high-water mark):
   ```bash
   awk '/^VmHWM:/ {print $2 " kB"}' /proc/$SRV_PID/status
   ```
   Sample once per minute via a background loop and take the max.

3. **`samplingDurationMs`** (sample wall-clock from YIELD):
   pulled out of the YIELD response. The bench script uses
   `awk -F',' '{print $7}' /tmp/papers100m_sample.csv | sed -n 2p`.

4. **`topology_*.csr` mmap residency** (the L3 working set):
   ```bash
   for f in data/dbs/gql/papers100M/projections/papers100M_proj/topology_*.csr; do
       echo "$f"
       fincore "$f" 2>/dev/null || vmtouch "$f" 2>/dev/null || \
           awk '{ printf "  size=%s bytes\n", $0 }' <(stat -c '%s' "$f")
   done
   ```

---

## 5. Comparison template

Fill in the table after each run. Keep the file checked into
`docs/research/` next to this procedure as
`docs/research/<date>-papers100m-results.md`.

| mode      | proj_build_sec | sample_wall_sec | peak_rss_gb | l1_nodes | l2_nodes | l3_nodes | notes                       |
|-----------|---------------:|----------------:|------------:|---------:|---------:|---------:|-----------------------------|
| spec13    |                |                 |             |          |          |          | auto-detected budgets       |
| spec13    |                |                 |             |          |          |          | l1=5 GB, l2=14 GB explicit  |
| caminoD   |                |                 |             |        - |        - |        - | sidecar-only mmap           |
| spec11    |                |       (OOM expected)        |       - |        - |        - |        - | not runnable at this scale  |
| bpt       |                |                 |             |        - |        - |        - | B+Tree direct (slowest)     |

Cells marked `-` are intentionally empty for modes that do not
construct the relevant tier.

---

## 6. Decision criteria

Following design doc §4 and the success metrics in plan §"Success
metrics":

| Measured `sample_wall_sec` | Verdict | Action |
|---|---|---|
| ≤ 30 min (1800 s) | **THESIS-DEFENSIBLE.** Closes the papers100M-on-commodity-RAM gap. | Update `docs/research/2026-04-25-gate-e-report.md` with the measured number; mark T13.16 ✓ in the plan. |
| 30 - 60 min | Acceptable but warrants follow-up: investigate L3 cold-fault dominance. | Profile with `perf stat -e major-faults`; if major-faults dominate, enable Phase 5 MinHash L3 reorder more aggressively (write `node_counts.bin` from a prior sample run, then rebuild). |
| > 60 min | **REGRESSION.** Sample is no faster than Camino D; Spec #13's L1/L2 are not paying off. | Phase 5+ implementation: aggressive L3 reorder + possibly bumping the L1/L2 split (less L1, more L2 → more nodes covered at the 50-200 ns tier). |

The design's projection is **~16 min** for fanout `[10, 5]` on
papers100M (370M lookups × 2.6 µs avg latency, see design §4). A
measured value within ±50% of that target is a clear win.

If the measurement comes in at 2-3× projected (≈ 30-50 min), the
plan §T13.16 already provides the escape hatch:

> If 2-3× slower: investigate L3 cold-fault dominance, possibly
> enable L3 reorder more aggressively.

---

## 7. After the run

1. **Drop the projection** to free disk (papers100M_proj is ~250 GB):
   ```bash
   curl -sS --data-binary "CALL drop_projection('papers100M_proj') RETURN *" \
       -H "Accept: text/csv" "http://127.0.0.1:19990/"
   # or after stopping the server:
   build/Release/bin/mdb drop-projection data/dbs/gql/papers100M papers100M_proj
   ```

2. **Stop the server**:
   ```bash
   kill -TERM $SRV_PID
   wait $SRV_PID 2>/dev/null
   ```

3. **Archive logs** under `docs/research/` (only the relevant
   summary; the full server log is multi-GB):
   - `topology_fwd.csr` size + sha256
   - `topology_rev.csr` size + sha256
   - `[FourLevelTopologyStore] built` line verbatim
   - `samplingDurationMs` from YIELD
   - peak VmHWM
   - the filled-in table from §5

4. **Update upstream docs** if the run is the canonical T13.16
   measurement:
   - Plan checklist `T13.16` → ✓
   - `docs/research/2026-04-25-gate-e-report.md` row "papers100M sample
     within projected window"
   - `docs/research/2026-04-24-defense-qa.md` Spec #13 framing for
     thesis defense

---

## 8. Connection to the local bench

`scripts/bench_four_level_topology.sh` exercises the same code path on
cora and arxiv (and opt-in products). On those datasets all four
modes (spec13, spec11, caminoD, bpt) fit on commodity hardware, so the
local bench is the regression-detection harness for Spec #13. The
papers100M procedure here is the scale-validation companion: same
modes, same tier-distribution log line, same YIELD shape.

When both deliverables are filled in, the Gate E sign-off (T13.17)
has the empirical evidence it needs across the full dataset spectrum.
