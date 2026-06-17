#!/usr/bin/env python3
"""Compare DGL NeighborSampler receptive field vs MDB on papers100M [10,15,20].

Loads ONLY the topology (edge_index) from the OGB papers100M raw npz (skips the
57GB node_feat to fit in 30GB RAM), builds the graph the way DiskGNN's
load_graph.py does (add_reverse_edges + remove_self_loop + add_self_loop),
then runs dgl NeighborSampler([10,15,20]) over batches of 1024 train seeds and
reports the per-seed receptive field (input_nodes count) distribution.

MDB reference (e2e5ep block headers): per-seed receptive field mean = 1300.4.

Run in the diskgnn_cu124 env AFTER the download+extract:
  ~/miniconda3/envs/diskgnn_cu124/bin/python scripts/diskgnn_dgl_receptive_field.py \
      --raw /home/bfuentes/diskgnn_data/ogb_raw/ogbn_papers100M
"""
import argparse, os, time, numpy as np

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", required=True, help="dir containing raw/data.npz + split/")
    ap.add_argument("--fanout", default="10,15,20")
    ap.add_argument("--batchsize", type=int, default=1024)
    ap.add_argument("--num-batches", type=int, default=300, help="batches to sample for the distribution")
    ap.add_argument("--self-loop", type=int, default=1, help="1=add_self_loop like DiskGNN, 0=no self-loop (MDB-style)")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    import torch, dgl
    from dgl.dataloading import DataLoader, NeighborSampler
    torch.manual_seed(args.seed); np.random.seed(args.seed)
    fanout = [int(x) for x in args.fanout.split(",")]

    # ---- load ONLY topology from the raw npz (skip node_feat) ----
    npz = os.path.join(args.raw, "raw", "data.npz")
    print(f"[load] {npz}", flush=True)
    t0 = time.time()
    z = np.load(npz, mmap_mode="r")
    print(f"[load] npz keys: {list(z.keys())}", flush=True)
    import gc
    eikey = "edge_index" if "edge_index" in z.files else [k for k in z.files if "edge" in k][0]
    ei = z[eikey]                              # [2, E], int64 mmap (NOT loaded to RAM)
    E = ei.shape[1]
    if "num_nodes_list" in z.files:
        num_nodes = int(np.asarray(z["num_nodes_list"]).reshape(-1)[0])
    else:
        num_nodes = int(max(ei[0].max(), ei[1].max())) + 1
    print(f"[load] key={eikey} num_nodes={num_nodes:,} edges={E:,} ({time.time()-t0:.0f}s)", flush=True)
    # int32 directly (node ids < 2^31): astype reads mmap + converts, no int64 RAM copy
    src = torch.from_numpy(np.asarray(ei[0]).astype(np.int32, copy=True))
    dst = torch.from_numpy(np.asarray(ei[1]).astype(np.int32, copy=True))
    del ei, z; gc.collect()
    print(f"[load] edge_index -> int32 in RAM ({2*E*4/1e9:.1f} GB) ({time.time()-t0:.0f}s)", flush=True)
    g = dgl.graph((src, dst), num_nodes=num_nodes)
    del src, dst; gc.collect()
    print(f"[graph] directed: {g.num_edges():,} edges ({time.time()-t0:.0f}s)", flush=True)
    # DiskGNN load_graph.py: add_reverse_edges + remove_self_loop + add_self_loop
    g = dgl.add_reverse_edges(g)
    print(f"[graph] +reverse: {g.num_edges():,} edges", flush=True)
    g = dgl.remove_self_loop(g)
    if args.self_loop:
        g = dgl.add_self_loop(g)
        print(f"[graph] +self_loop: {g.num_edges():,} edges (DiskGNN-faithful)", flush=True)
    else:
        print(f"[graph] NO self_loop (MDB-style): {g.num_edges():,} edges", flush=True)
    g = g.formats("csc")

    # ---- train seeds from split ----
    tr = os.path.join(args.raw, "split", "time", "train.csv.gz")
    if os.path.exists(tr):
        import pandas as pd
        train_nid = torch.from_numpy(pd.read_csv(tr, compression="gzip", header=None).values.reshape(-1)).to(torch.int32)
    else:  # fallback: random nodes
        train_nid = torch.randint(0, num_nodes, (1207000,), dtype=torch.int32)
    print(f"[seeds] train nodes: {train_nid.numel():,}", flush=True)

    sampler = NeighborSampler(fanout)
    dl = DataLoader(g, train_nid, sampler, batch_size=args.batchsize, shuffle=True,
                    drop_last=True, num_workers=0, device="cpu")
    rf = []
    t1 = time.time()
    for i, (input_nodes, output_nodes, blocks) in enumerate(dl):
        rf.append(int(input_nodes.numel()))
        if i+1 >= args.num_batches: break
    rf = np.array(rf)
    persed = rf / args.batchsize
    print("\n================ DGL RESULT ================", flush=True)
    print(f"fanout={fanout} batch={args.batchsize} self_loop={args.self_loop} batches={len(rf)} ({time.time()-t1:.0f}s)")
    print(f"receptive field PER BATCH: mean={rf.mean():,.0f} std={rf.std():,.0f} min={rf.min():,} max={rf.max():,}")
    print(f"receptive field PER SEED : mean={persed.mean():.1f} std={persed.std():.1f}")
    print(f"--- MDB reference (e2e5ep): per-seed mean=1300.4, per-batch mean=1,330,052 ---")

if __name__ == "__main__":
    main()
