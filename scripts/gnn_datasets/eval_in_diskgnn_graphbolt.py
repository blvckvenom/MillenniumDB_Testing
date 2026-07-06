#!/usr/bin/env python3
# =============================================================================
# eval_in_diskgnn_graphbolt.py — evaluate a MillenniumDB-trained GraphSAGE
# inside DiskGNN's evaluation pipeline.
#
# Everything below the weights is DiskGNN's own stack, imported from their
# repo checkout (examples/dgl_baseline/graphbolt/node_classification.py):
# their GraphBolt on-disk dataset, their create_dataloader() datapipe, their
# evaluate() loop, their SAGE class. The only foreign piece is the weight
# file, produced by `mdb_gnn.py convert --to dgl-collapsed`, whose layout
# matches their SAGE stack exactly (see mdb_gnn.collapse_head).
#
# Example:
#   python eval_in_diskgnn_graphbolt.py \
#     --weights mdb_best_dgl.pth \
#     --diskgnn ~/DiskGNN/DiskGNN \
#     --root ~/diskgnn_data/graphbolt_dataset/datasets \
#     --dataset ogbn-papers100M --fanout 10,15,20 --splits validation,test
#
# Note on fanout: the string is passed verbatim to THEIR sampler, i.e. DGL
# convention (also MillenniumDB's default since 2026-07-06).
# =============================================================================

import argparse
import importlib.util
import time
from pathlib import Path
from types import SimpleNamespace

import torch


def load_their_module(diskgnn_root):
    path = (Path(diskgnn_root).expanduser()
            / "examples/dgl_baseline/graphbolt/node_classification.py")
    spec = importlib.util.spec_from_file_location("diskgnn_node_classification", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--weights", required=True,
                    help="state_dict from mdb_gnn.py convert --to dgl-collapsed")
    ap.add_argument("--diskgnn", default="~/DiskGNN/DiskGNN")
    ap.add_argument("--root", default="~/diskgnn_data/graphbolt_dataset/datasets")
    ap.add_argument("--dataset", default="ogbn-papers100M")
    ap.add_argument("--fanout", default="10,15,20")
    ap.add_argument("--batch-size", type=int, default=1024)
    ap.add_argument("--hidden", type=int, default=256)
    ap.add_argument("--num-workers", type=int, default=0)
    ap.add_argument("--splits", default="validation,test")
    ap.add_argument("--no-pin", action="store_true",
                    help="skip graph.pin_memory_() (slower CPU sampling path)")
    args = ap.parse_args()

    nc = load_their_module(args.diskgnn)
    import dgl.graphbolt as gb

    print(f"Loading {args.dataset} from {args.root} ...", flush=True)
    t0 = time.time()
    dataset = gb.BuiltinDataset(
        args.dataset, root=str(Path(args.root).expanduser())).load()
    graph = dataset.graph
    features = dataset.feature
    if not args.no_pin:
        try:
            graph.pin_memory_()
            print("graph pinned (GPU-UVA sampling path)", flush=True)
        except Exception as e:  # not enough lockable RAM — their cpu-cuda mode
            print(f"pin_memory_ failed ({e}); sampling on CPU", flush=True)
    task = dataset.tasks[0]
    num_classes = task.metadata["num_classes"]
    in_size = features.size("node", None, "feat")[0]
    print(f"loaded in {time.time() - t0:.1f}s | in_size={in_size} "
          f"num_classes={num_classes}", flush=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = nc.SAGE(in_size, args.hidden, num_classes)
    sd = torch.load(args.weights, map_location="cpu", weights_only=True)
    model.load_state_dict(sd, strict=True)
    model = model.to(device).eval()
    print(f"loaded MDB weights into their SAGE class (strict=True): "
          f"{sum(t.numel() for t in sd.values()):,} params", flush=True)

    ns = SimpleNamespace(
        batch_size=args.batch_size,
        fanout=list(map(int, args.fanout.split(","))),
        device=device,
        num_workers=args.num_workers,
        storage_device="cpu" if args.no_pin else "pinned",
        sample_mode="sample_neighbor",
        overlap_graph_fetch=False,
    )
    # Their create_dataloader reads `args` from the MODULE globals (set by
    # argparse when their script runs as __main__); provide the same when the
    # file is imported as a module.
    nc.args = ns

    # API-drift shim: their script targets an older GraphBolt whose DataLoader
    # accepted overlap_graph_fetch; dgl 2.5 removed it (omitting it means
    # overlap disabled, which is what we pass anyway).
    _orig_dataloader = nc.gb.DataLoader

    def _dataloader_compat(datapipe, *a, **kw):
        kw.pop("overlap_graph_fetch", None)
        return _orig_dataloader(datapipe, *a, **kw)

    nc.gb.DataLoader = _dataloader_compat

    itemsets = {
        "train": task.train_set,
        "validation": task.validation_set,
        "test": task.test_set,
    }
    for split in args.splits.split(","):
        t0 = time.time()
        acc = nc.evaluate(ns, model, graph, features, itemsets[split], num_classes)
        print(f"RESULT {split}: accuracy={acc.item():.6f} "
              f"({time.time() - t0:.1f}s, fanout={ns.fanout}, "
              f"batch_size={ns.batch_size})", flush=True)


if __name__ == "__main__":
    main()
