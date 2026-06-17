#!/usr/bin/env python3
"""compare_undirected_substrate.py — is MDB's undirected sampling substrate the
same graph DiskGNN samples from?

This is the decisive test for the "último candidato vivo" of the papers100M
accuracy gap: undirected-graph-construction / neighbor-sampling micro-differences.

MDB substrate  = topology_fwd.csr (out-nbrs) + topology_rev.csr (in-nbrs), the
                 exact CSR the GNN sampler reads for UNDIRECTED k-hop.
DiskGNN substrate = graph.pth FusedCSCSamplingGraph CSC (indptr, indices), the
                 exact graph GraphBolt sample_neighbor reads.

Node-id alignment: MDB node row r == OGB node r (proven by feature/label
alignment, fmat[r]==npy[r]); DiskGNN uses OGB ids directly. So row r is the same
physical paper in both. The script verifies this (neighbor-set overlap must be
high) before trusting any difference.

Outputs:
  (1) EXACT global multiset-degree comparison over ALL 111M nodes (vectorized).
  (2) per-node neighbor-SET comparison over a stratified sample (random + hubs).
  (3) self-loop counts both sides.
"""
import torch, numpy as np, struct, random, sys, time

PROJ = "/home/bfuentes/MillenniumDB_Testing/data/dbs/gql/papers100M/projections/papers100M_e2e_opt"
FWD  = PROJ + "/topology_fwd.csr"
REV  = PROJ + "/topology_rev.csr"
GPTH = "/home/bfuentes/diskgnn_data/offgs_dataset/ogbn-papers100M-offgs/graph.pth"
NSAMPLE = int(sys.argv[1]) if len(sys.argv) > 1 else 20000

def open_csr(path):
    with open(path, 'rb') as f:
        hdr = f.read(64)
    assert hdr[:8] == b'TOPOCSR1', f"bad magic {hdr[:8]!r}"
    ver      = struct.unpack_from('<I', hdr, 8)[0]
    id_width = hdr[12]
    flags    = hdr[13]
    N        = struct.unpack_from('<Q', hdr, 16)[0]
    M        = struct.unpack_from('<Q', hdr, 24)[0]
    assert id_width == 4, f"expected uint32 COL, got id_width={id_width}"
    rp_off  = 64
    row_ptr = np.memmap(path, dtype='<u8', mode='r', offset=rp_off, shape=(N + 1,))
    col_off = rp_off + 8 * (N + 1)
    col     = np.memmap(path, dtype='<u4', mode='r', offset=col_off, shape=(M,))
    return dict(N=N, M=M, ver=ver, flags=flags, row_ptr=row_ptr, col=col)

t0 = time.time()
print("=== open MDB CSR sidecars ===")
fwd = open_csr(FWD); rev = open_csr(REV)
N = fwd['N']
print(f"  fwd: N={fwd['N']:,} M={fwd['M']:,} flags={fwd['flags']} ver={fwd['ver']}")
print(f"  rev: N={rev['N']:,} M={rev['M']:,} flags={rev['flags']} ver={rev['ver']}")
print(f"  col max fwd={int(fwd['col'][:100000000].max())} (must be < N={N})")
assert rev['N'] == N

print("=== open DiskGNN graph.pth ===")
g = torch.load(GPTH, mmap=True, weights_only=False)
indptr  = g.csc_indptr.numpy()        # int64 [N+1]
indices = g.indices.numpy()           # int32 [M]
print(f"  dgn: N={len(indptr)-1:,} M={len(indices):,}")
assert len(indptr) - 1 == N, "node-count mismatch!"

# ---------- (1) EXACT global multiset-degree comparison over ALL nodes ----------
print(f"\n=== (1) GLOBAL multiset-degree over ALL {N:,} nodes ===")
fwd_deg = (fwd['row_ptr'][1:].astype(np.int64) - fwd['row_ptr'][:-1].astype(np.int64))
rev_deg = (rev['row_ptr'][1:].astype(np.int64) - rev['row_ptr'][:-1].astype(np.int64))
mdb_deg = fwd_deg + rev_deg                                   # multiset (fwd ++ rev)
dgn_deg = (indptr[1:] - indptr[:-1]).astype(np.int64)
diff    = dgn_deg - mdb_deg
n_eq    = int((diff == 0).sum())
print(f"  MDB total undirected (sum mdb_deg) = {int(mdb_deg.sum()):,}")
print(f"  DGN total undirected (sum dgn_deg) = {int(dgn_deg.sum()):,}")
print(f"  nodes with IDENTICAL multiset-degree = {n_eq:,} / {N:,} = {100.0*n_eq/N:.4f}%")
print(f"  nodes where DGN>MDB = {int((diff>0).sum()):,}; DGN<MDB = {int((diff<0).sum()):,}")
print(f"  sum |diff| = {int(np.abs(diff).sum()):,}  (total edge-slot discrepancy)")
nz = diff[diff != 0]
if nz.size:
    print(f"  among differing nodes: diff min={int(nz.min())} max={int(nz.max())} "
          f"mean={nz.mean():.3f} median={int(np.median(nz))}")
    # histogram of small diffs
    for d in (-2, -1, 1, 2, 3):
        print(f"    diff=={d:+d}: {int((diff==d).sum()):,} nodes")
    print(f"    |diff|>3 : {int((np.abs(diff)>3).sum()):,} nodes")

# ---------- (2) per-node neighbor-SET comparison on a stratified sample ----------
print(f"\n=== (2) per-node neighbor-SET sample (N={NSAMPLE} random + 200 hubs) ===")
random.seed(7)
samp = set(random.sample(range(N), min(NSAMPLE, N)))
hubs = np.argpartition(dgn_deg, -200)[-200:].tolist()
samp |= set(hubs)
samp = sorted(samp)

set_equal = 0; set_overlap_sum = 0.0; n = 0
mdb_self = 0; dgn_self = 0
mdb_has_dup = 0   # MDB multiset has a repeated neighbor (reciprocal double)
dgn_has_dup = 0
only_mdb_total = 0; only_dgn_total = 0
worst = []  # (jaccard, r, |mdb|, |dgn|, only_mdb, only_dgn)
for r in samp:
    mf = fwd['col'][fwd['row_ptr'][r]:fwd['row_ptr'][r+1]]
    mr = rev['col'][rev['row_ptr'][r]:rev['row_ptr'][r+1]]
    mdb_ms = np.concatenate([mf, mr]).astype(np.int64)
    d = indices[indptr[r]:indptr[r+1]].astype(np.int64)
    ms = set(mdb_ms.tolist()); ds = set(d.tolist())
    if r in ms: mdb_self += 1
    if r in ds: dgn_self += 1
    if len(mdb_ms) != len(ms): mdb_has_dup += 1
    if len(d) != len(ds): dgn_has_dup += 1
    inter = len(ms & ds); uni = len(ms | ds) or 1
    only_mdb = len(ms - ds); only_dgn = len(ds - ms)
    only_mdb_total += only_mdb; only_dgn_total += only_dgn
    jac = inter / uni
    set_overlap_sum += jac
    if ms == ds: set_equal += 1
    n += 1
    if jac < 1.0:
        worst.append((jac, r, len(ms), len(ds), only_mdb, only_dgn))

print(f"  sampled nodes                  = {n:,}")
print(f"  exact set-equal neighbor sets  = {set_equal:,} ({100.0*set_equal/n:.3f}%)")
print(f"  mean Jaccard(MDB,DGN) overlap  = {set_overlap_sum/n:.6f}")
print(f"  self-loops: MDB={mdb_self}  DGN={dgn_self} (of {n})")
print(f"  nodes w/ duplicate nbr in multiset: MDB={mdb_has_dup}  DGN={dgn_has_dup}")
print(f"  neighbors only-in-MDB total={only_mdb_total:,}  only-in-DGN total={only_dgn_total:,}")
worst.sort()
print("  10 worst-overlap sampled nodes (jaccard, node, |MDB_set|, |DGN_set|, onlyMDB, onlyDGN):")
for w in worst[:10]:
    print(f"    jac={w[0]:.4f} node={w[1]} |MDB|={w[2]} |DGN|={w[3]} onlyMDB={w[4]} onlyDGN={w[5]}")
print(f"\n[done in {time.time()-t0:.1f}s]")
