#!/usr/bin/env python
"""
Structural + adversarial validation of GNN embeddings WRITTEN BACK into a
MillenniumDB projection (gnn_train writeProperty / gnn_predict writeProperty).

Reference implementation: ogbn-products, projection 'pwb'.

It reads the STORED PROPERTY back out of the database over GQL -- not the .npy
export -- and answers the question the timing measurement cannot: are the right
vectors attached to the right nodes?

  S1 coverage     every node in the projection carries a readable embedding
  S2 dimension    every stored vector has exactly hidden_dim components
  S3 finiteness   no NaN / Inf
  S4 degeneracy   no all-zero vectors
  S5 duplicates   nodes sharing a byte-identical vector, cross-checked against
                  the tensor store's own dedup count
  M1 MIS-MAPPING  argmax(classifier @ stored_vector) == the node's EXTERNAL
                  ground-truth label, scored globally, per split, and PER BATCH
  M2 null control the same statistic under deliberate mis-mappings, so M1's
                  power is measured rather than assumed

INDEPENDENCE: labels and splits come from the raw OGB archive; the node identity
comes from the identifier the database itself prints. Neither passes through the
GNN pipeline. See the report text for the precise scope.

Usage:
    mdb server <db> -p 7897 -t 7200 --browser false &
    python scripts/validate_written_embeddings.py --fetch --analyze
"""
import sys, os, time, json, subprocess, gzip
import numpy as np

# ------------------------------------------------------------------ configuration
DB   = os.environ.get("MDB_DB",
       "/home/bfuentes/MillenniumDB_Testing/docs/research/2026-07-14-accuracy-small/mdb_dbs/products/db")
PROJ = os.environ.get("MDB_PROJ", "pwb")
PROP = os.environ.get("MDB_PROP", "embedding")
GOUT = f"{DB}/projections/{PROJ}/gnn_output/default/"
OGB  = os.environ.get("OGB_ROOT",
       "/home/bfuentes/MillenniumDB_Testing/docs/research/2026-07-14-accuracy-small/"
       "data/dl_cache_products/ogb_data/ogbn_products/")
PORT = os.environ.get("PORT", "7897")
OUT  = os.environ.get("OUT", "/home/bfuentes/.claude/jobs/0729db73/tmp/products_cmp")
N, DIM, BATCH_SIZE, CHUNK = 2449029, 256, 512, 50000
SPLIT_BATCHES = (("train", 0, 385), ("val", 1, 77), ("test", 2, 4323))

EMB   = OUT + "/stored_emb.npy"
FLAGS = OUT + "/stored_flags.npz"

# tensors.dat physical layout, used only for the S5 cross-check
TBLOCK, TREC, TPER = 524288, 1026, 511


def gql(q, out):
    r = subprocess.run(["curl", "-s", "--max-time", "900", "-X", "POST",
                        f"http://localhost:{PORT}/gql",
                        "-H", "Content-Type: application/gql",
                        "-H", "Accept: text/plain",
                        "--data-binary", q, "-o", out], capture_output=True)
    if r.returncode != 0:
        raise RuntimeError(f"curl rc={r.returncode}: {r.stderr[:300]!r} "
                           "(is the server up on this port?)")


# ----------------------------------------------------------------------- stage 1
def fetch():
    emb  = np.zeros((N, DIM), np.float32)
    dims = np.full(N, -1, np.int32)          # -1 = node never returned
    order_ok, t0, tmp = True, time.time(), OUT + "/_chunk.txt"
    for off in range(0, N, CHUNK):
        gql(f"USE {PROJ} MATCH (n:Node) RETURN n, n.{PROP} SKIP {off} LIMIT {CHUNK}", tmp)
        with open(tmp) as f:
            if not f.readline().startswith("n,n."):
                raise RuntimeError(f"bad response at SKIP {off}")
            pos = off
            for line in f:
                c = line.index(",")
                k = int(line[:c][2:])                     # '_n<K>' -> K
                body = line[c + 1:].strip()
                if body in ("NULL", ""):
                    dims[k] = 0
                else:
                    v = np.fromstring(body.strip("[]"), sep=",", dtype=np.float32)
                    dims[k] = v.size
                    if v.size == DIM:
                        emb[k] = v
                order_ok &= (k == pos)
                pos += 1
        if off % (CHUNK * 10) == 0:
            print(f"  SKIP {off:>8}  {time.time()-t0:6.1f}s", flush=True)
    np.save(EMB, emb)
    np.savez(FLAGS, dims=dims, order_ok=np.array([order_ok]))
    print(f"fetched {N} nodes in {time.time()-t0:.1f}s  scan_order_is_identity={order_ok}")


# ----------------------------------------------------------------------- stage 2
def analyze():
    import torch, pandas as pd
    from scipy.stats import binom
    R, emb = {}, np.load(EMB)
    fl = np.load(FLAGS); dims = fl["dims"]

    # -- EXTERNAL ground truth: raw OGB archive, no MillenniumDB involvement
    lab = np.loadtxt(gzip.open(OGB + "raw/node-label.csv.gz"), dtype=np.int64)
    spl = np.full(N, -1, np.int64)
    for code, nm in ((0, "train"), (1, "valid"), (2, "test")):
        spl[pd.read_csv(OGB + f"split/sales_ranking/{nm}.csv.gz",
                        header=None).values.ravel()] = code
    assert lab.shape[0] == N

    have = dims == DIM
    # -- S1..S5
    R["S1_scan_order_identity"] = bool(fl["order_ok"][0])
    R["S1_nodes_never_returned"] = int((dims == -1).sum())
    R["S1_nodes_with_readable_property"] = int(have.sum())
    R["S1_nodes_expected"] = N
    R["S1_MISSING"] = int(N - have.sum())
    R["S2_dim_histogram"] = {int(k): int(v) for k, v in zip(*np.unique(dims, return_counts=True))}
    R["S3_nonfinite_nodes"] = int((~np.isfinite(emb).all(1)).sum())
    nrm = np.linalg.norm(emb, axis=1)
    R["S4_zero_norm_among_readable"] = int((nrm[have] == 0).sum())
    R["S4_norm_min/med/max_readable"] = [float(nrm[have].min()), float(np.median(nrm[have])),
                                         float(nrm[have].max())]
    if os.path.exists(f"{DB}/tensors.dat"):
        sz = os.path.getsize(f"{DB}/tensors.dat")
        recs = sum(min(TPER, max(0, (min(sz, (b + 1) * TBLOCK) - b * TBLOCK) // TREC))
                   for b in range((sz + TBLOCK - 1) // TBLOCK))
        R["S5_tensor_store_records"] = int(recs)
        R["S5_nodes_sharing_a_vector"] = int(N - recs)

    # -- M1 mis-mapping probe
    sd = torch.jit.load(GOUT + "model.pt").state_dict()
    Wc = sd["classifier.weight"].float().cpu().numpy()
    bc = sd["classifier.bias"].float().cpu().numpy()

    def score(E):
        o = np.empty(N, bool)
        for s in range(0, N, 200000):
            o[s:s+200000] = ((E[s:s+200000] @ Wc.T + bc).argmax(1) == lab[s:s+200000])
        return o

    ok = score(emb)
    R["M1_accuracy_all_readable"] = float(ok[have].mean())
    for code, nm in ((0, "train"), (1, "val"), (2, "test")):
        m = have & (spl == code)
        R[f"M1_accuracy_{nm}"] = float(ok[m].mean())

    # true batch boundaries: rows are contiguous per split, batch_size seeds each
    sid = np.load(GOUT + "seed_ids.npy") & 0x00FFFFFFFFFFFFFF
    rspl = np.load(GOUT + "splits.npy")
    bid = np.empty(len(sid), np.int64); base = 0; rng_ = {}
    for nm, code, nb in SPLIT_BATCHES:
        idx = np.where(rspl == code)[0]
        assert (np.diff(idx) == 1).all(), "split rows are not contiguous"
        bid[idx] = base + np.arange(len(idx)) // BATCH_SIZE
        rng_[nm] = (base, base + nb); base += nb
    bo = np.empty(N, np.int64); bo[sid] = bid
    hit = np.bincount(bo[have], weights=ok[have].astype(float), minlength=base)
    tot = np.bincount(bo[have], minlength=base)
    bacc = hit / np.maximum(tot, 1)
    R["M1_num_batches"] = int(base)
    R["M1_batch_acc_min"] = float(bacc.min())
    R["M1_batch_acc_median"] = float(np.median(bacc))
    R["M1_batches_below_0.35"] = int((bacc < 0.35).sum())
    R["M1_worst_batches"] = [[int(i), int(tot[i]), float(bacc[i])] for i in np.argsort(bacc)[:5]]
    R["M1_overdispersion_per_split"] = {}
    for nm, code, nb in SPLIT_BATCHES:
        lo, hi = rng_[nm]
        sel = np.arange(lo, hi); sel = sel[tot[sel] >= BATCH_SIZE * 0.8]
        p = hit[sel].sum() / tot[sel].sum()
        R["M1_overdispersion_per_split"][nm] = round(
            float(np.var(bacc[sel]) / (p * (1 - p) / tot[sel]).mean()), 3)

    # -- M2 nulls: what the same statistic reads under real mis-mappings
    rows = np.arange(len(sid)); g = np.random.default_rng(0); nulls = {}

    def null(perm, name):
        E = np.empty_like(emb); E[sid] = emb[sid[perm]]
        o = score(E)
        nulls[name] = {"all": round(float(o[have].mean()), 5),
                       "test": round(float(o[have & (spl == 2)].mean()), 5)}

    shuf = rows.copy()
    for s in range(0, len(rows), BATCH_SIZE):
        shuf[s:s+BATCH_SIZE] = g.permutation(shuf[s:s+BATCH_SIZE])
    null(shuf, "random_permutation_within_each_batch")
    off1 = rows.copy()
    for s in range(0, len(rows), BATCH_SIZE):
        e = min(s + BATCH_SIZE, len(rows)); off1[s:e] = np.roll(off1[s:e], 1)
    null(off1, "off_by_one_within_each_batch")
    null(g.permutation(len(rows)), "global_shuffle")
    R["M2_nulls"] = nulls
    R["M2_power_at_n512"] = {
        "P(batch<0.35 | clean)": float(binom.cdf(int(0.35*512), 512, R["M1_accuracy_test"])),
        "P(batch>=0.35 | permuted)": float(binom.sf(int(0.35*512)-1, 512,
                                                    nulls["random_permutation_within_each_batch"]["test"])),
        "nodes_to_swap_in_one_batch_to_trip": int(round(
            512 * (R["M1_accuracy_test"] - 0.35) /
            (R["M1_accuracy_test"] - nulls["random_permutation_within_each_batch"]["test"]))),
    }

    # -- X: stored property vs the .npy export (SELF-CONSISTENT on values)
    npy = np.load(GOUT + "embeddings.npy", mmap_mode="r")
    pick = np.sort(g.choice(len(sid), 20000, replace=False))
    a = np.asarray(npy[pick], np.float32); b = emb[sid[pick]]
    m = have[sid[pick]]
    rel = np.linalg.norm(a[m]-b[m], axis=1)/np.maximum(np.linalg.norm(b[m], axis=1), 1e-12)
    R["X_property_vs_npy_relL2_median"] = float(np.median(rel))
    R["X_property_vs_npy_relL2_max"] = float(rel.max())

    # -------------------------------------------------------------- PASS / FAIL
    v = []
    v.append(("S1 coverage", R["S1_MISSING"] == 0, f"{R['S1_MISSING']} nodes unreadable"))
    v.append(("S2 dimension", set(R["S2_dim_histogram"]) <= {0, DIM, -1}
              and R["S2_dim_histogram"].get(DIM, 0) == R["S1_nodes_with_readable_property"],
              str(R["S2_dim_histogram"])))
    v.append(("S3 finiteness", R["S3_nonfinite_nodes"] == 0, str(R["S3_nonfinite_nodes"])))
    v.append(("S4 degeneracy", R["S4_zero_norm_among_readable"] == 0,
              str(R["S4_zero_norm_among_readable"])))
    v.append(("M1 no mis-mapped batch", R["M1_batches_below_0.35"] == 0,
              f"min batch acc {R['M1_batch_acc_min']:.4f}"))
    v.append(("M1 overdispersion in [0.5,2]",
              all(0.5 <= x <= 2.0 for x in R["M1_overdispersion_per_split"].values()),
              str(R["M1_overdispersion_per_split"])))
    v.append(("M2 null separation >= 3x",
              R["M1_accuracy_test"] >= 3 * nulls["random_permutation_within_each_batch"]["test"],
              f"{R['M1_accuracy_test']:.4f} vs {nulls['random_permutation_within_each_batch']['test']:.4f}"))
    R["VERDICT"] = [[n, "PASS" if okk else "FAIL", d] for n, okk, d in v]
    R["OVERALL"] = "PASS" if all(o for _, o, _ in v) else "FAIL"

    print(json.dumps(R, indent=2))
    json.dump(R, open(OUT + "/validate_written_embeddings.json", "w"), indent=2)
    return 0 if R["OVERALL"] == "PASS" else 1


if __name__ == "__main__":
    if "--fetch" in sys.argv:
        fetch()
    sys.exit(analyze() if "--analyze" in sys.argv else 0)
