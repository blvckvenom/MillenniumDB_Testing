#!/usr/bin/env python3
"""costmodel_analyze.py — turn bench_costmodel.sh's per-stage CSV into a cost
model: per-stage time/throughput decomposition, a roofline-style I/O
classification, an empirically-grounded gpu/cpu budget policy (resolves F#3),
and a coarse papers100M feasibility projection.

Usage:
    python3 scripts/costmodel_analyze.py [docs/research/2026-05-31-costmodel/costmodel.csv]

Honest by design: time/throughput numbers are measured; the papers100M
projection is a structural scaling from the cora+arxiv points and is labelled
approximate. The budget policy is exact arithmetic over measured feature
footprint vs detected VRAM/RAM, not a fit.
"""
import csv, subprocess, sys
from pathlib import Path

# Measured dataset shapes (N nodes, D feat dim, E directed edges).
SHAPES = {
    "cora":       dict(N=2_708,       D=1433, E=5_429),
    "arxiv":      dict(N=169_343,     D=128,  E=1_166_243),
    "papers100M": dict(N=111_059_956, D=128,  E=1_615_685_872),  # projection target
}
def feat_mb(s):  # feature matrix footprint in MB (float32)
    return s["N"] * s["D"] * 4 / 1e6

NVME_SEQ_MBPS = 3000.0  # PCIe Gen4 NVMe sequential reference (order-of-magnitude)

def detect_hw():
    vram = ram = 0.0
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=memory.total", "--format=csv,noheader,nounits"],
            text=True, stderr=subprocess.DEVNULL)
        vram = float(out.strip().splitlines()[0])
    except Exception:
        pass
    try:
        for line in Path("/proc/meminfo").read_text().splitlines():
            if line.startswith("MemTotal:"):
                ram = float(line.split()[1]) / 1024  # kB -> MB
                break
    except Exception:
        pass
    return vram, ram

def load(csv_path):
    rows = {}
    with open(csv_path) as f:
        for r in csv.DictReader(f):
            ds, st = r["dataset"], r["stage"]
            def fnum(k):
                try: return float(r[k])
                except Exception: return 0.0
            rows.setdefault(ds, {})[st] = dict(
                wall=fnum("wall_s"), dread=fnum("dread_MB"), dwrite=fnum("dwrite_MB"),
                lread=fnum("lread_MB"), lwrite=fnum("lwrite_MB"), rss=fnum("peakrss_MB"))
    return rows

STAGES = ["import", "project", "sample", "materialize", "build", "train"]

def fmt(x, w=9, p=1):
    return f"{x:>{w}.{p}f}"

def classify(d):
    """Roofline-ish: compare physical disk throughput to NVMe ref + the
    logical/physical ratio (high ratio => page-cache-served => compute-bound)."""
    phys = d["dread"] + d["dwrite"]
    log = d["lread"] + d["lwrite"]
    if d["wall"] <= 0: return "-"
    phys_bw = phys / d["wall"]
    if phys < 1 and log < 1:
        return "compute/in-mem"
    if log > 2 * max(phys, 0.1):
        return "cache-served"
    if phys_bw < 0.3 * NVME_SEQ_MBPS:
        return "disk-bound(random)"
    return "disk-bound(seq)"

def main():
    csv_path = sys.argv[1] if len(sys.argv) > 1 else \
        "docs/research/2026-05-31-costmodel/costmodel.csv"
    rows = load(csv_path)
    vram, ram = detect_hw()
    out = []
    p = out.append

    p(f"# GNN pipeline cost model\n")
    p(f"Source: {csv_path}")
    p(f"Hardware: VRAM={vram:.0f} MB, RAM={ram:.0f} MB, NVMe seq ref={NVME_SEQ_MBPS:.0f} MB/s\n")

    # --- 1. Per-stage decomposition ---
    for ds in [d for d in ("cora", "arxiv") if d in rows]:
        total = sum(rows[ds].get(s, {}).get("wall", 0) for s in STAGES)
        p(f"## {ds} — per-stage decomposition (E2E {total:.2f}s)\n")
        p(f"{'stage':<12}{'wall_s':>9}{'%E2E':>7}{'dRead':>9}{'dWrite':>9}"
          f"{'physMB/s':>10}{'peakRSS':>9}  class")
        for st in STAGES:
            d = rows[ds].get(st)
            if not d: continue
            phys = d["dread"] + d["dwrite"]
            bw = phys / d["wall"] if d["wall"] > 0 else 0
            pct = 100 * d["wall"] / total if total else 0
            p(f"{st:<12}{fmt(d['wall'])}{fmt(pct,7)}{fmt(d['dread'])}{fmt(d['dwrite'])}"
              f"{fmt(bw,10)}{fmt(d['rss'])}  {classify(d)}")
        p("")

    # --- 2. Findings ---
    p("## Findings\n")
    if "arxiv" in rows:
        a = rows["arxiv"]
        tr = a.get("train", {})
        e2e = sum(a.get(s, {}).get("wall", 0) for s in STAGES)
        p(f"- **train dominates**: arxiv train {tr.get('wall',0):.1f}s = "
          f"{100*tr.get('wall',0)/e2e:.0f}% of E2E. Within train, assemble is the bottleneck "
          f"(measured asm >> fwd+bwd in the train yields).")
        p(f"- **train is per-epoch disk re-read of sample structure**: arxiv train read "
          f"{tr.get('dread',0):.0f} MB physical from disk across epochs while l1HitRatio=1.0 "
          f"(all features in GPU). The disk traffic is packed_slim/batches.dat re-read each "
          f"epoch, not features → caching the sample structure in RAM across epochs is the "
          f"main train-time lever.")
        mat = a.get("materialize", {})
        p(f"- **materialize is write-heavy**: {mat.get('dwrite',0):.0f} MB written "
          f"({mat.get('wall',0):.1f}s) — the packed scratch + reorder prep. Build then reads "
          f"~{a.get('build',{}).get('dread',0):.0f} MB back.")
    p("")

    # --- 3. Budget policy (F#3) ---
    p("## Budget policy (proposed defaults — resolves F#3)\n")
    p("Policy: gpu_budget = min(0.60 * VRAM, feat_MB); "
      "cpu_budget = min(0.50 * RAM, max(0, feat_MB - gpu_budget)); "
      "remainder -> L3 mmap / L4 disk.\n")
    p(f"{'dataset':<12}{'featMB':>10}{'gpuBudMB':>10}{'cpuBudMB':>10}"
      f"{'L1%':>7}{'L2%':>7}{'L3/4%':>8}")
    for ds, s in SHAPES.items():
        fm = feat_mb(s)
        gpu = min(0.60 * vram, fm) if vram else 0.0
        cpu = min(0.50 * ram, max(0.0, fm - gpu)) if ram else 0.0
        l1 = 100 * gpu / fm if fm else 0
        l2 = 100 * cpu / fm if fm else 0
        l34 = max(0.0, 100 - l1 - l2)
        p(f"{ds:<12}{fmt(fm,10)}{fmt(gpu,10)}{fmt(cpu,10)}"
          f"{fmt(l1,7)}{fmt(l2,7)}{fmt(l34,8)}")
    p("")
    p("Interpretation: cora/arxiv fit entirely in L1 (GPU) at these budgets → "
      "l1HitRatio=1.0 as measured. papers100M's 56.8 GB feature set far exceeds VRAM, "
      "so the L1% line is the realistic GPU-resident fraction; the rest must serve from "
      "L2 RAM + L3/L4 disk — which is exactly where the four-level store + addr-tables earn "
      "their keep, and where train-time becomes cache-hit-ratio-bound rather than compute-bound.\n")

    # --- 4. Coarse papers100M projection (affine 2-point fit) ---
    p("## papers100M projection (affine 2-point fit, coarse)\n")
    if "cora" in rows and "arxiv" in rows:
        pp = SHAPES["papers100M"]
        # driver value per (stage, dataset): edges for topology stages, feat_MB for build.
        def driver(stage, ds):
            return feat_mb(SHAPES[ds]) if stage == "build" else SHAPES[ds]["E"]
        def affine(stage):
            x1, y1 = driver(stage, "cora"),  rows["cora"].get(stage, {}).get("wall", 0)
            x2, y2 = driver(stage, "arxiv"), rows["arxiv"].get(stage, {}).get("wall", 0)
            if x2 == x1: return 0.0, y2, 0.0
            b = (y2 - y1) / (x2 - x1); a0 = y1 - b * x1
            xp = driver(stage, "papers100M")
            return a0, b, a0 + b * xp
        p("Affine fit time = a + b*driver over (cora, arxiv); driver = edges (topology "
          "stages) or feat_MB (build).\n")
        p(f"{'stage':<12}{'driver':>13}{'a(s)':>9}{'b':>12}{'proj_s':>10}{'proj_min':>10}")
        for st in ("project", "sample", "materialize", "build"):
            drv = "feat_MB" if st == "build" else "edges"
            a0, b, ps = affine(st)
            ps = max(ps, 0.0)
            p(f"{st:<12}{drv:>13}{fmt(a0,9,2)}{b:>12.2e}{fmt(ps,10)}{fmt(ps/60,10)}")
        p(f"{'train':<12}{'cache-bound':>13}{'-':>9}{'-':>12}{'N/A':>10}{'N/A':>10}")
        p("\n  REALITY CHECK vs known papers100M measurements ("
          "Plan F): the real sample is ~78-153s and pack/build ~327s — far below the affine "
          "projection above. The linear-in-edges driver OVER-predicts the topology stages because "
          "(i) sampling parallelizes across workers (Plan F numWorkers) and (ii) cost is really "
          "batches x capped-receptive-field, not total edges. So treat the topology-stage "
          "projections as loose UPPER BOUNDS; the build (feature-driven) projection is the more "
          "trustworthy one.")
        p("\n  train is deliberately NOT extrapolated: at papers100M scale the feature set spills "
          "out of GPU/RAM, so train time is governed by the L1/L2/L3/L4 cache-hit ratios under the "
          "budget policy above (17/28/55%), not a constant MB/s. Measure it in the Phase 3 run.")
    p("")

    report = "\n".join(out)
    print(report)
    md = Path(csv_path).with_name("costmodel_report.md")
    md.write_text(report + "\n")
    print(f"\n[written] {md}")

if __name__ == "__main__":
    main()
