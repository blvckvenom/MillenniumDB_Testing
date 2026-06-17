#!/usr/bin/env python3
"""substrate_seed_impact.py — quantify the accuracy-relevant magnitude of the
MDB-vs-DiskGNN undirected substrate difference, restricted to LABELED SEED nodes
(train/val/test), plus a mechanism trace of isolated-in-MDB-only nodes.

Deep-neighborhood diffs wash out under fanout sampling + mean-aggregation; what
can move accuracy is a difference in the *seed* nodes' own neighborhoods (the
classified nodes). This script measures exactly that.
"""
import torch, numpy as np, struct, random, sys, time

PROJ = "/home/bfuentes/MillenniumDB_Testing/data/dbs/gql/papers100M/projections/papers100M_e2e_opt"
FWD  = PROJ + "/topology_fwd.csr"
REV  = PROJ + "/topology_rev.csr"
SPL  = PROJ + "/splits.bin"
GPTH = "/home/bfuentes/diskgnn_data/offgs_dataset/ogbn-papers100M-offgs/graph.pth"

def open_csr(path):
    with open(path, 'rb') as f: hdr = f.read(64)
    assert hdr[:8] == b'TOPOCSR1'
    N = struct.unpack_from('<Q', hdr, 16)[0]; M = struct.unpack_from('<Q', hdr, 24)[0]
    rp = np.memmap(path, dtype='<u8', mode='r', offset=64, shape=(N + 1,))
    col = np.memmap(path, dtype='<u4', mode='r', offset=64 + 8 * (N + 1), shape=(M,))
    return N, M, rp, col

t0 = time.time()
Nf, Mf, rpf, colf = open_csr(FWD)
Nr, Mr, rpr, colr = open_csr(REV)
N = Nf
g = torch.load(GPTH, mmap=True, weights_only=False)
indptr = g.csc_indptr.numpy(); indices = g.indices.numpy()

fwd_deg = rpf[1:].astype(np.int64) - rpf[:-1].astype(np.int64)
rev_deg = rpr[1:].astype(np.int64) - rpr[:-1].astype(np.int64)
mdb_deg = fwd_deg + rev_deg
dgn_deg = (indptr[1:] - indptr[:-1]).astype(np.int64)

# ---- splits ----
with open(SPL, 'rb') as f: shdr = f.read(24)
assert shdr[:4] == b'GNNS', shdr[:4]
Ns = struct.unpack_from('<Q', shdr, 16)[0]; assert Ns == N
splits = np.memmap(SPL, dtype=np.uint8, mode='r', offset=24, shape=(N,))
TR = (splits == 0); VA = (splits == 1); TE = (splits == 2)
print("=== split histogram (validate vs OGB train1207179 val125265 test214338) ===")
print(f"  TRAIN={int(TR.sum()):,} VAL={int(VA.sum()):,} TEST={int(TE.sum()):,} "
      f"UNLABELED={int((splits==255).sum()):,}")
SEED = TR | VA | TE
print(f"  total labeled seeds = {int(SEED.sum()):,}")

# ---- global isolation ----
print("\n=== GLOBAL isolation (degree==0 on one side) ===")
miso = (mdb_deg == 0); diso = (dgn_deg == 0)
print(f"  isolated in BOTH        = {int((miso & diso).sum()):,}")
print(f"  isolated in MDB ONLY    = {int((miso & ~diso).sum()):,}  (MDB sees 0 nbrs, DGN sees >0)")
print(f"  isolated in DGN ONLY    = {int((diso & ~miso).sum()):,}")

def seed_report(mask, name):
    md = mdb_deg[mask]; dd = dgn_deg[mask]
    n = int(mask.sum())
    eq = int((md == dd).sum())
    iso_mdb_only = int(((md == 0) & (dd > 0)).sum())
    print(f"\n=== SEED subset: {name} (n={n:,}) ===")
    print(f"  mean degree  MDB={md.mean():.3f}  DGN={dd.mean():.3f}  (Δ={dd.mean()-md.mean():+.4f})")
    print(f"  identical multiset-degree = {eq:,} ({100.0*eq/n:.3f}%)")
    print(f"  seed isolated in MDB ONLY = {iso_mdb_only:,} ({100.0*iso_mdb_only/n:.4f}%)")
    print(f"  seeds DGN>MDB={int((dd>md).sum()):,}  DGN<MDB={int((dd<md).sum()):,}")

seed_report(SEED, "ALL labeled seeds")
seed_report(TE,   "TEST seeds (drive test acc)")
seed_report(VA,   "VAL seeds (drive best-val selection)")

# ---- per-seed neighbor-SET sample (TEST seeds) ----
print("\n=== per-TEST-seed neighbor-SET sample ===")
te_ids = np.nonzero(TE)[0]
random.seed(11)
samp = sorted(random.sample(list(te_ids), min(20000, len(te_ids))))
set_eq = 0; jac_sum = 0.0; onlyD = 0; onlyM = 0
for r in samp:
    ms = set(np.concatenate([colf[rpf[r]:rpf[r+1]], colr[rpr[r]:rpr[r+1]]]).astype(np.int64).tolist())
    ds = set(indices[indptr[r]:indptr[r+1]].astype(np.int64).tolist())
    inter = len(ms & ds); uni = len(ms | ds) or 1
    jac_sum += inter / uni; onlyD += len(ds - ms); onlyM += len(ms - ds)
    if ms == ds: set_eq += 1
print(f"  sampled TEST seeds        = {len(samp):,}")
print(f"  exact set-equal           = {set_eq:,} ({100.0*set_eq/len(samp):.3f}%)")
print(f"  mean Jaccard              = {jac_sum/len(samp):.6f}")
print(f"  only-in-DGN total={onlyD:,}  only-in-MDB total={onlyM:,}")

# ---- mechanism trace: nodes isolated-in-MDB-only ----
print("\n=== MECHANISM TRACE: 6 nodes isolated in MDB only ===")
cand = np.nonzero(miso & ~diso)[0]
random.seed(3)
for r in (random.sample(list(cand), 6) if len(cand) >= 6 else list(cand)):
    d = indices[indptr[r]:indptr[r+1]].astype(np.int64)
    print(f"  node {r}: MDB deg=0 (fwd={fwd_deg[r]},rev={rev_deg[r]}); DGN nbrs={d.tolist()}")
    for x in d.tolist():
        # does MDB record the edge under X's row (in either direction)?
        xf = set(colf[rpf[x]:rpf[x+1]].astype(np.int64).tolist())
        xr = set(colr[rpr[x]:rpr[x+1]].astype(np.int64).tolist())
        print(f"     nbr X={x}: DGNdeg={dgn_deg[x]} MDBdeg={mdb_deg[x]} | r in X.fwd?={r in xf} r in X.rev?={r in xr}")
print(f"\n[done in {time.time()-t0:.1f}s]")
