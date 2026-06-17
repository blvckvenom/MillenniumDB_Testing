#!/usr/bin/env python3
"""Memory-bounded (MDB-style, disk/RAM-CSR, NOT DGL in-RAM) measurement of the
papers100M bidirected+self-loop receptive field for fanout [10,15,20], to compare
DIRECTLY against MDB's measured 1300.4 nodes/seed (which DGL's in-RAM build OOMs on
30 GB — see DGL_OOM_REPORT.md).

Builds the bidirected+self-loop CSR in RAM via a vectorized counting sort (no numba,
no 3.3B argsort): peak ~16 GB instead of DGL's ~38 GB. Then runs a DGL-equivalent
k-hop sampler (uniform WITHOUT replacement, min(fanout, degree)) — the same algorithm
as dgl.NeighborSampler and MDB's Fisher-Yates sampler.
"""
import numpy as np, time, argparse, os
ap = argparse.ArgumentParser()
ap.add_argument("--npz", default="/home/bfuentes/diskgnn_data/papers100M-bin/raw/data.npz")
ap.add_argument("--ei", default="/home/bfuentes/diskgnn_data/edge_index.npy",
                help="uncompressed edge_index .npy (TRUE mmap-able; the npz copy is DEFLATED and loads 25.9GB into RAM)")
ap.add_argument("--fanout", default="10,15,20")
ap.add_argument("--batchsize", type=int, default=1024)
ap.add_argument("--num-batches", type=int, default=300)
ap.add_argument("--self-loop", type=int, default=1)
ap.add_argument("--seed", type=int, default=42)
ap.add_argument("--order", default="forward", choices=["forward","reverse"],
                help="forward: seeds sample fanout[0] first; reverse: fanout[-1] first (DGL block order)")
args = ap.parse_args()
fan = [int(x) for x in args.fanout.split(",")]
rng = np.random.default_rng(args.seed)
t0 = time.time()

# edge_index: prefer the uncompressed .npy (TRUE mmap, slices don't load 25.9GB).
if os.path.exists(args.ei):
    ei = np.load(args.ei, mmap_mode="r")        # mmap-able; reads only the accessed slices
    print(f"[t{time.time()-t0:.0f}] edge_index via uncompressed mmap {args.ei} {ei.shape} {ei.dtype}", flush=True)
else:
    raise SystemExit(f"need uncompressed {args.ei} (the npz edge_index is DEFLATED -> not mmap-able)")
E = ei.shape[1]
N = int(np.asarray(np.load(args.npz)["num_nodes_list"]).reshape(-1)[0])   # tiny array, cheap to decompress
print(f"[t{time.time()-t0:.0f}] N={N:,} E_directed={E:,} self_loop={args.self_loop}", flush=True)

# ---- degree (out+in [+self]) ----
deg = np.zeros(N, dtype=np.int64)
CH = 80_000_000
for s in range(0, E, CH):
    e = min(s+CH, E)
    np.add.at(deg, np.asarray(ei[0,s:e]), 1)   # out
    np.add.at(deg, np.asarray(ei[1,s:e]), 1)   # in (reverse edge)
if args.self_loop:
    deg += 1
M = int(deg.sum())
print(f"[t{time.time()-t0:.0f}] bidirected{'+self' if args.self_loop else ''} edges M={M:,}  RAM-indices={M*4/1e9:.1f}GB", flush=True)

# ---- CSR via vectorized counting sort ----
indptr = np.zeros(N+1, dtype=np.int64); np.cumsum(deg, out=indptr[1:])
indices = np.empty(M, dtype=np.int32)
cursor = indptr[:-1].copy()

def scatter(u_arr, v_arr):
    # place v at the next free slot of u, handling duplicate u within the chunk
    sidx = np.argsort(u_arr, kind="stable")
    us = u_arr[sidx]; vs = v_arr[sidx]
    first = np.searchsorted(us, us)                 # first-occurrence index per value (us sorted)
    rank = np.arange(us.shape[0], dtype=np.int64) - first
    pos = cursor[us] + rank
    indices[pos] = vs
    np.add.at(cursor, u_arr, 1)

for s in range(0, E, CH):
    e = min(s+CH, E)
    su = np.asarray(ei[0,s:e]).astype(np.int64); sv = np.asarray(ei[1,s:e]).astype(np.int64)
    scatter(su, sv)          # out edge u->v
    scatter(sv, su)          # reverse edge v->u
    print(f"  CSR {e/E*100:.0f}% (t{time.time()-t0:.0f})", flush=True)
if args.self_loop:
    alln = np.arange(N, dtype=np.int64); scatter(alln, alln)
print(f"[t{time.time()-t0:.0f}] CSR built. validate...", flush=True)

# ---- validate CSR against raw edges for a few nodes ----
def neighbors(v): return indices[indptr[v]:indptr[v+1]]
chk0 = np.asarray(ei[:, :200000])
for tv in [int(chk0[0,0]), int(chk0[0,123]), int(chk0[1,77])]:
    raw_out = set(chk0[1, chk0[0,:]==tv].tolist())
    csr = set(int(x) for x in neighbors(tv))
    ok = raw_out.issubset(csr)   # CSR (out+in) must contain all raw out-neighbors of tv
    print(f"  node {tv}: csr_deg={indptr[tv+1]-indptr[tv]} contains raw_out({len(raw_out)})={ok}", flush=True)

# ---- DGL-equivalent k-hop sampler (uniform w/o replacement, min(fanout,deg)) ----
order_fan = fan if args.order=="forward" else fan[::-1]
train = np.fromiter((int(x) for x in
        __import__("pandas").read_csv(os.path.join(os.path.dirname(os.path.dirname(args.npz)),
        "split","time","train.csv.gz"), header=None).values.reshape(-1)), dtype=np.int64) \
        if os.path.exists(os.path.join(os.path.dirname(os.path.dirname(args.npz)),"split","time","train.csv.gz")) \
        else rng.integers(0, N, 1_200_000)
print(f"[t{time.time()-t0:.0f}] train seeds={train.shape[0]:,} fanout_order={order_fan}", flush=True)

rf = []
t1 = time.time()
for b in range(args.num_batches):
    seeds = train[rng.integers(0, train.shape[0], args.batchsize)]
    frontier = np.unique(seeds)
    acc = [frontier]                              # all unique nodes touched (numpy, no python set)
    for f in order_fan:
        nxt = []
        for u in frontier.tolist():               # frontier <= ~150k nodes -> feasible
            lo = indptr[u]; hi = indptr[u+1]; d = hi-lo
            if d <= 0: continue
            if d <= f:
                nxt.append(indices[lo:hi])
            else:
                nxt.append(indices[lo:hi][rng.choice(d, size=f, replace=False)])  # uniform w/o replacement
        if not nxt: break
        frontier = np.unique(np.concatenate(nxt))
        acc.append(frontier)
    rf.append(int(np.unique(np.concatenate(acc)).shape[0]))
    if (b+1) % 10 == 0: print(f"  batch {b+1}/{args.num_batches} rf~{np.mean(rf):.0f} (t{time.time()-t1:.0f})", flush=True)
rf = np.array(rf); ps = rf/args.batchsize
print("\n================ DGL-EQUIVALENT RESULT (papers100M, MDB-style lowmem CSR) ================")
print(f"fanout={fan} order={args.order} self_loop={args.self_loop} batches={len(rf)}")
print(f"receptive field PER BATCH: mean={rf.mean():,.0f} std={rf.std():,.0f}")
print(f"receptive field PER SEED : mean={ps.mean():.1f} std={ps.std():.1f}")
print(f"--- MDB reference (e2e5ep blocks): per-seed mean=1300.4, per-batch mean=1,330,052 ---")
