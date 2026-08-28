#!/usr/bin/env python3
# =============================================================================
# mdb_gnn.py — use GNN models trained by MillenniumDB's gnn_train outside the
#              DBMS (plain PyTorch, DGL, PyTorch Geometric).
# =============================================================================
#
# What gnn_train leaves on disk (per run, under gnn_output/<outputDir>/):
#
#   model.pt                  LibTorch archive with the final weights. Named
#                             parameters: conv_0..conv_{L-1} (each a Linear
#                             over concat(self, mean_neighbors)) + classifier.
#   checkpoints/*.pt          Same parameters plus Adam optimizer state
#                             (param_groups.* / state.* entries, filtered out
#                             here).
#   training_log.json         Hyperparameters + accuracies + loss curve.
#
# With exportEmbeddings: true, gnn_train additionally writes a row-aligned
# export (embeddings / seed_ids / seed_rows / labels / splits .npy) plus
# export_manifest.json describing the architecture and forward conventions.
#
# Everything in this module is derived dynamically from the parameter shapes
# and the manifest — no layer count, dimension or dataset is hardcoded.
#
# Architecture note: MillenniumDB's head (last conv -> hidden, then a separate
# linear classifier, with NO activation in between) is an exact linear
# factorization of the standard DGL/OGB SAGE layout (last conv -> classes).
# collapse_head() multiplies the two matrices, so the converted checkpoint
# loads into an unmodified DGL-style SAGE class (e.g. DiskGNN's) and computes
# the same logits.
#
# CLI:
#   python3 mdb_gnn.py info     <model.pt | run_dir>
#   python3 mdb_gnn.py eval     <run_dir> [--split test|validation|train|all]
#   python3 mdb_gnn.py convert  <model.pt | run_dir> --to statedict|dgl-collapsed --out W.pth
#   python3 mdb_gnn.py selftest
#
# Requires: torch, numpy. DGL / PyTorch Geometric are imported lazily and only
# needed for their respective converters.
# =============================================================================

import argparse
import json
import re
import sys
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

PARAM_RE = re.compile(r"^(conv_(\d+)|classifier)\.(weight|bias)$")

SPLIT_CODES = {"train": 0, "validation": 1, "test": 2}


# -----------------------------------------------------------------------------
# Loading
# -----------------------------------------------------------------------------

def load_params(path):
    """Extract the model parameters from a LibTorch archive (.pt).

    Works for both model.pt (weights only) and checkpoints/*.pt (weights +
    Adam state; the optimizer entries do not match PARAM_RE and are dropped).
    Returns an OrderedDict name -> float32 CPU tensor.
    """
    mod = torch.jit.load(str(path), map_location="cpu")
    params = OrderedDict()
    for name, tensor in mod.named_parameters():
        if PARAM_RE.match(name):
            params[name] = tensor.detach().to(torch.float32).clone()
    if not params:
        raise ValueError(f"{path}: no conv_*/classifier parameters found")
    return params


@dataclass
class ArchSpec:
    num_layers: int
    input_dim: int
    hidden_dim: int
    num_classes: int
    dropout: float = 0.0
    normalize: bool = False

    @staticmethod
    def from_params(params, manifest=None):
        """Infer the architecture from parameter shapes; the manifest (when
        available) only contributes what shapes cannot express (dropout,
        normalize) and is cross-checked against the shapes."""
        conv_ids = sorted(
            int(m.group(2)) for n in params
            if (m := PARAM_RE.match(n)) and m.group(2) is not None
            and n.endswith(".weight")
        )
        if not conv_ids or conv_ids != list(range(len(conv_ids))):
            raise ValueError(f"unexpected conv layer ids: {conv_ids}")
        num_layers = len(conv_ids)
        hidden_dim, two_in = params[f"conv_{num_layers - 1}.weight"].shape
        num_classes = params["classifier.weight"].shape[0]
        spec = ArchSpec(
            num_layers=num_layers,
            input_dim=two_in // 2,
            hidden_dim=hidden_dim,
            num_classes=num_classes,
        )
        if manifest:
            m = manifest.get("model", {})
            spec.dropout = float(m.get("dropout", 0.0))
            spec.normalize = bool(m.get("normalize", False))
            for key, mine in (("input_dim", spec.input_dim),
                              ("hidden_dim", spec.hidden_dim),
                              ("num_classes", spec.num_classes),
                              ("num_layers", spec.num_layers)):
                if key in m and int(m[key]) != mine:
                    raise ValueError(
                        f"manifest/{key}={m[key]} disagrees with shapes ({mine})")
        return spec

    def layer_in_dim(self, k):
        """Input dim of conv_k. conv_{L-1} is applied first (input features);
        every other conv consumes hidden states."""
        return self.input_dim if k == self.num_layers - 1 else self.hidden_dim


def read_manifest(run_dir):
    p = Path(run_dir) / "export_manifest.json"
    if p.is_file():
        with open(p) as f:
            return json.load(f)
    return None


def resolve_model_pt(path):
    """Accept either a .pt file or a run directory containing model.pt."""
    p = Path(path)
    if p.is_dir():
        cand = p / "model.pt"
        if not cand.is_file():
            raise FileNotFoundError(f"{p}: no model.pt in directory")
        return cand
    return p


# -----------------------------------------------------------------------------
# Pure-PyTorch replica of the in-DB model
# -----------------------------------------------------------------------------

def _sage_conv(x, edge_index, num_dst, linear):
    """One MEAN aggregation layer, mirroring the C++ forward exactly:
    scatter-sum over incoming messages, divide by clamp_min(in_degree, 1),
    concat(self, agg) with self first, then the layer's Linear."""
    src, dst = edge_index[0], edge_index[1]
    neigh = x.index_select(0, src)
    agg = torch.zeros(num_dst, x.shape[1], dtype=x.dtype, device=x.device)
    agg.index_add_(0, dst, neigh)
    deg = torch.zeros(num_dst, 1, dtype=x.dtype, device=x.device)
    deg.index_add_(0, dst, torch.ones(src.shape[0], 1, dtype=x.dtype, device=x.device))
    agg = agg / deg.clamp_min(1.0)
    x_self = x[:num_dst]
    return linear(torch.cat([x_self, agg], dim=1))


class MDBGraphSAGE(nn.Module):
    """Replica of MillenniumDB's GraphSAGE (MEAN) built from an ArchSpec.

    forward()/get_embeddings() consume MillenniumDB's batch layout (features
    for the full active set, per-layer local edge_index, active-set sizes with
    the prefix invariant). inference() runs full-graph layer-wise inference
    from a global edge_index instead (OGB-style evaluation).
    """

    def __init__(self, spec):
        super().__init__()
        self.spec = spec
        for k in range(spec.num_layers):
            self.add_module(
                f"conv_{k}", nn.Linear(2 * spec.layer_in_dim(k), spec.hidden_dim))
        self.classifier = nn.Linear(spec.hidden_dim, spec.num_classes)

    def conv(self, k):
        return getattr(self, f"conv_{k}")

    def _between(self, x):
        x = torch.relu(x)
        if self.training and self.spec.dropout > 0:
            x = torch.dropout(x, self.spec.dropout, train=True)
        if self.spec.normalize:
            x = x / x.norm(2, dim=1, keepdim=True).clamp_min(1e-6)
        return x

    def get_embeddings(self, features, edge_indices, active_sizes):
        x = features
        for k in reversed(range(self.spec.num_layers)):
            x = _sage_conv(x, edge_indices[k], active_sizes[k], self.conv(k))
            if k > 0:
                x = self._between(x)
        return x

    def forward(self, features, edge_indices, active_sizes):
        return self.classifier(self.get_embeddings(features, edge_indices, active_sizes))

    @torch.no_grad()
    def inference(self, x, edge_index, return_embeddings=False):
        """Full-neighbor inference over a whole graph. edge_index is [2, E]
        global with messages flowing src -> dst (pass both directions for an
        undirected graph)."""
        n = x.shape[0]
        for k in reversed(range(self.spec.num_layers)):
            x = _sage_conv(x, edge_index, n, self.conv(k))
            if k > 0:
                x = self._between(x)
        return x if return_embeddings else self.classifier(x)

    @staticmethod
    def from_params(params, manifest=None):
        spec = ArchSpec.from_params(params, manifest)
        model = MDBGraphSAGE(spec)
        model.load_state_dict(params, strict=True)
        model.eval()
        return model


# -----------------------------------------------------------------------------
# Weight conversion
# -----------------------------------------------------------------------------

def split_halves(weight):
    """concat(self, agg) order: the first in_dim columns act on the node's own
    features, the second half on the mean-aggregated neighbors."""
    in_dim = weight.shape[1] // 2
    return weight[:, :in_dim].contiguous(), weight[:, in_dim:].contiguous()


def collapse_head(params):
    """Merge (conv_0, classifier) into a single conv emitting logits.

    Exact because nothing (no activation, no dropout at eval, no normalize)
    sits between the last conv and the classifier in the MDB forward:
        classifier(W0 @ cat + b0) = (Wc@W0) @ cat + (Wc@b0 + bc)

    Returns (spec, layers) where layers[i] = (w_self, w_neigh, bias) in
    APPLICATION order (layers[0] consumes the input features), i.e. the
    standard DGL/OGB SAGE layout whose last layer outputs num_classes.
    """
    spec = ArchSpec.from_params(params)
    L = spec.num_layers
    layers = []
    for i in range(L):
        k = L - 1 - i  # conv_k is applied i-th
        w = params[f"conv_{k}.weight"]
        b = params[f"conv_{k}.bias"]
        if k == 0:
            wc = params["classifier.weight"]
            bc = params["classifier.bias"]
            w = wc @ w
            b = wc @ b + bc
        w_self, w_neigh = split_halves(w)
        layers.append((w_self, w_neigh, b))
    return spec, layers


def preserved_layers(params):
    """Same (w_self, w_neigh, bias) list but WITHOUT collapsing: all layers
    emit hidden_dim and the classifier stays separate. Returns
    (spec, layers, (w_clf, b_clf))."""
    spec = ArchSpec.from_params(params)
    L = spec.num_layers
    layers = []
    for i in range(L):
        k = L - 1 - i
        w_self, w_neigh = split_halves(params[f"conv_{k}.weight"])
        layers.append((w_self, w_neigh, params[f"conv_{k}.bias"]))
    return spec, layers, (params["classifier.weight"], params["classifier.bias"])


def _fill_dgl_sageconv(conv, w_self, w_neigh, bias):
    """Fill one dgl.nn.SAGEConv('mean') by key introspection, so the mapping
    survives DGL moving the bias between fc_self / a standalone parameter."""
    sd = conv.state_dict()
    new = {}
    bias_placed = False
    for key in sd:
        if key.endswith("fc_self.weight"):
            new[key] = w_self.clone()
        elif key.endswith("fc_neigh.weight"):
            new[key] = w_neigh.clone()
        elif key == "bias":
            new[key] = bias.clone()
            bias_placed = True
    for key in sd:
        if key in new:
            continue
        if key.endswith("fc_self.bias") and not bias_placed:
            new[key] = bias.clone()
            bias_placed = True
        elif key.endswith(".bias"):
            new[key] = torch.zeros_like(sd[key])
        else:
            raise ValueError(f"unexpected SAGEConv parameter: {key}")
    if not bias_placed:
        raise ValueError("SAGEConv exposes no bias parameter to receive the MDB bias")
    conv.load_state_dict(new)


def to_dgl_collapsed(params):
    """Build the standard DGL SAGE stack (as used by DiskGNN's model zoo:
    SAGEConv(in->hid), hid->hid ..., hid->classes, ReLU between) filled with
    the collapsed MDB weights. Its state_dict loads into any class with that
    exact layout via load_state_dict, no code changes on the target side.
    """
    import dgl.nn.pytorch as dglnn

    spec, layers = collapse_head(params)
    if spec.normalize:
        raise ValueError(
            "normalize=true models insert an l2-normalize between hidden "
            "layers, which the plain DGL SAGE stack does not compute; use "
            "to_dgl() (structure-preserving wrapper) instead")

    class SAGE(nn.Module):
        def __init__(self, in_size, hid_size, out_size, num_layers):
            super().__init__()
            self.layers = nn.ModuleList()
            if num_layers == 1:
                self.layers.append(dglnn.SAGEConv(in_size, out_size, "mean"))
            else:
                self.layers.append(dglnn.SAGEConv(in_size, hid_size, "mean"))
                for _ in range(num_layers - 2):
                    self.layers.append(dglnn.SAGEConv(hid_size, hid_size, "mean"))
                self.layers.append(dglnn.SAGEConv(hid_size, out_size, "mean"))

        def forward(self, blocks, h):
            for i, (layer, block) in enumerate(zip(self.layers, blocks)):
                h = layer(block, h)
                if i != len(self.layers) - 1:
                    h = torch.relu(h)
            return h

    model = SAGE(spec.input_dim, spec.hidden_dim, spec.num_classes, spec.num_layers)
    for conv, (w_self, w_neigh, bias) in zip(model.layers, layers):
        _fill_dgl_sageconv(conv, w_self, w_neigh, bias)
    model.eval()
    return model


def to_dgl(params, manifest=None):
    """Structure-preserving DGL module: SAGEConv stack all emitting hidden_dim
    plus a separate linear head, honoring dropout/normalize from the manifest.
    Use this when you also want the 256-dim embeddings outside MDB."""
    import dgl.nn.pytorch as dglnn

    spec = ArchSpec.from_params(params, manifest)
    _, layers, (w_clf, b_clf) = preserved_layers(params)

    class SAGEPreserved(nn.Module):
        def __init__(self):
            super().__init__()
            self.spec = spec
            self.layers = nn.ModuleList()
            for i in range(spec.num_layers):
                in_dim = spec.input_dim if i == 0 else spec.hidden_dim
                self.layers.append(dglnn.SAGEConv(in_dim, spec.hidden_dim, "mean"))
            self.head = nn.Linear(spec.hidden_dim, spec.num_classes)

        def forward(self, blocks, h, return_embeddings=False):
            for i, (layer, block) in enumerate(zip(self.layers, blocks)):
                h = layer(block, h)
                if i != len(self.layers) - 1:
                    h = torch.relu(h)
                    if self.training and self.spec.dropout > 0:
                        h = torch.dropout(h, self.spec.dropout, train=True)
                    if self.spec.normalize:
                        h = h / h.norm(2, dim=1, keepdim=True).clamp_min(1e-6)
            return h if return_embeddings else self.head(h)

    model = SAGEPreserved()
    for conv, (w_self, w_neigh, bias) in zip(model.layers, layers):
        _fill_dgl_sageconv(conv, w_self, w_neigh, bias)
    with torch.no_grad():
        model.head.weight.copy_(w_clf)
        model.head.bias.copy_(b_clf)
    model.eval()
    return model


def to_pyg(params, manifest=None):
    """Structure-preserving PyG module (SAGEConv aggr='mean' stack + linear
    head). PyG applies lin_l to the aggregated neighbors (bias there) and
    lin_r to the root features."""
    from torch_geometric.nn import SAGEConv

    spec = ArchSpec.from_params(params, manifest)
    _, layers, (w_clf, b_clf) = preserved_layers(params)

    class PyGSAGEPreserved(nn.Module):
        def __init__(self):
            super().__init__()
            self.spec = spec
            self.layers = nn.ModuleList()
            for i in range(spec.num_layers):
                in_dim = spec.input_dim if i == 0 else spec.hidden_dim
                self.layers.append(SAGEConv(in_dim, spec.hidden_dim, aggr="mean"))
            self.head = nn.Linear(spec.hidden_dim, spec.num_classes)

        def forward(self, x, adjs, return_embeddings=False):
            """adjs: list of (edge_index, num_dst) per layer, outermost first,
            with the prefix invariant (dst nodes are the first num_dst rows)."""
            for i, (layer, (edge_index, num_dst)) in enumerate(zip(self.layers, adjs)):
                x = layer((x, x[:num_dst]), edge_index)
                if i != len(self.layers) - 1:
                    x = torch.relu(x)
                    if self.training and self.spec.dropout > 0:
                        x = torch.dropout(x, self.spec.dropout, train=True)
                    if self.spec.normalize:
                        x = x / x.norm(2, dim=1, keepdim=True).clamp_min(1e-6)
            return x if return_embeddings else self.head(x)

    model = PyGSAGEPreserved()
    with torch.no_grad():
        for conv, (w_self, w_neigh, bias) in zip(model.layers, layers):
            conv.lin_l.weight.copy_(w_neigh)
            conv.lin_l.bias.copy_(bias)
            conv.lin_r.weight.copy_(w_self)
        model.head.weight.copy_(w_clf)
        model.head.bias.copy_(b_clf)
    model.eval()
    return model


# -----------------------------------------------------------------------------
# Offline evaluation from the exported .npy files
# -----------------------------------------------------------------------------

def evaluate_offline(run_dir, split="test"):
    """Accuracy from embeddings.npy + labels.npy + splits.npy + the classifier
    weights in model.pt. No graph and no MDB-internal format needed."""
    run_dir = Path(run_dir)
    emb = np.load(run_dir / "embeddings.npy")
    labels = np.load(run_dir / "labels.npy")
    splits = np.load(run_dir / "splits.npy")
    params = load_params(run_dir / "model.pt")
    w = params["classifier.weight"].numpy()
    b = params["classifier.bias"].numpy()

    logits = emb @ w.T + b
    pred = logits.argmax(axis=1)

    results = {}
    wanted = SPLIT_CODES.items() if split == "all" else [(split, SPLIT_CODES[split])]
    for name, code in wanted:
        mask = (splits == code) & (labels >= 0)
        n = int(mask.sum())
        results[name] = {
            "n": n,
            "accuracy": float((pred[mask] == labels[mask]).mean()) if n else float("nan"),
        }
    return results


# -----------------------------------------------------------------------------
# Self-test: conversion exactness across random configurations
# -----------------------------------------------------------------------------

def _random_params(spec, gen):
    params = OrderedDict()
    for k in range(spec.num_layers):
        params[f"conv_{k}.weight"] = torch.randn(
            spec.hidden_dim, 2 * spec.layer_in_dim(k), generator=gen)
        params[f"conv_{k}.bias"] = torch.randn(spec.hidden_dim, generator=gen)
    params["classifier.weight"] = torch.randn(
        spec.num_classes, spec.hidden_dim, generator=gen)
    params["classifier.bias"] = torch.randn(spec.num_classes, generator=gen)
    return params


def _random_batch(spec, gen):
    sizes = [int(torch.randint(2, 6, (1,), generator=gen))]
    for _ in range(spec.num_layers):
        sizes.append(sizes[-1] + int(torch.randint(1, 20, (1,), generator=gen)))
    edges = []
    for k in range(spec.num_layers):
        e = int(torch.randint(1, 40, (1,), generator=gen))
        src = torch.randint(0, sizes[k + 1], (e,), generator=gen)
        dst = torch.randint(0, sizes[k], (e,), generator=gen)
        edges.append(torch.stack([src, dst]))
    x = torch.randn(sizes[-1], spec.input_dim, generator=gen)
    return x, edges, sizes


def _check(name, a, b, failures, rtol=1e-4, atol=1e-5):
    diff = (a - b).abs().max().item() if a.numel() else 0.0
    ok = torch.allclose(a, b, rtol=rtol, atol=atol)
    print(f"    {'PASS' if ok else 'FAIL'}  {name}  max|diff|={diff:.3e}")
    if not ok:
        failures.append(name)


def selftest():
    try:
        import dgl
        has_dgl = True
    except ImportError:
        has_dgl = False
    try:
        import torch_geometric  # noqa: F401
        has_pyg = True
    except ImportError:
        has_pyg = False
    print(f"selftest: dgl={'yes' if has_dgl else 'NO (skipping dgl checks)'} "
          f"pyg={'yes' if has_pyg else 'NO (skipping pyg checks)'}")

    failures = []
    gen = torch.Generator().manual_seed(20260706)
    specs = [
        ArchSpec(1, 7, 16, 5),
        ArchSpec(2, 5, 8, 3),
        ArchSpec(3, 13, 32, 9),
        ArchSpec(4, 6, 12, 4),
        ArchSpec(2, 5, 4, 6),            # num_classes > hidden_dim
        ArchSpec(3, 9, 16, 5, 0.5, True) # dropout + normalize (eval-mode)
    ]
    for spec in specs:
        print(f"  spec L={spec.num_layers} in={spec.input_dim} hid={spec.hidden_dim} "
              f"classes={spec.num_classes} normalize={spec.normalize}")
        params = _random_params(spec, gen)
        replica = MDBGraphSAGE.from_params(
            params, {"model": {"dropout": spec.dropout, "normalize": spec.normalize}})
        x, edges, sizes = _random_batch(spec, gen)
        ref_logits = replica(x, edges, sizes)
        ref_emb = replica.get_embeddings(x, edges, sizes)

        # (1) Collapse algebra in pure torch: run the collapsed stack manually.
        _, layers = collapse_head(params)
        h = x
        for i, (w_self, w_neigh, bias) in enumerate(layers):
            k = spec.num_layers - 1 - i
            lin = nn.Linear(w_self.shape[1] * 2, w_self.shape[0])
            with torch.no_grad():
                lin.weight.copy_(torch.cat([w_self, w_neigh], dim=1))
                lin.bias.copy_(bias)
            h = _sage_conv(h, edges[k], sizes[k], lin)
            if i != len(layers) - 1:
                h = torch.relu(h)
                if spec.normalize:
                    h = h / h.norm(2, dim=1, keepdim=True).clamp_min(1e-6)
        _check("collapsed == replica logits", ref_logits, h, failures)

        if has_dgl:
            import dgl
            blocks = [
                dgl.create_block(
                    (edges[k][0], edges[k][1]),
                    num_src_nodes=sizes[k + 1], num_dst_nodes=sizes[k])
                for k in reversed(range(spec.num_layers))
            ]
            if not spec.normalize:
                dgl_c = to_dgl_collapsed(params)
                _check("dgl collapsed (DiskGNN-class layout) == replica",
                       ref_logits, dgl_c(blocks, x), failures)
            dgl_p = to_dgl(params, {"model": {"dropout": spec.dropout,
                                              "normalize": spec.normalize}})
            _check("dgl preserved logits == replica",
                   ref_logits, dgl_p(blocks, x), failures)
            _check("dgl preserved embeddings == replica",
                   ref_emb, dgl_p(blocks, x, return_embeddings=True), failures)

        if has_pyg:
            adjs = [(edges[k], sizes[k]) for k in reversed(range(spec.num_layers))]
            pyg_p = to_pyg(params, {"model": {"dropout": spec.dropout,
                                              "normalize": spec.normalize}})
            _check("pyg preserved logits == replica",
                   ref_logits, pyg_p(x, adjs), failures)

    if failures:
        print(f"selftest: {len(failures)} FAILURES: {failures}")
        return 1
    print("selftest: all checks passed")
    return 0


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------

def cmd_info(args):
    pt = resolve_model_pt(args.path)
    manifest = read_manifest(Path(pt).parent)
    params = load_params(pt)
    spec = ArchSpec.from_params(params, manifest)
    total = sum(t.numel() for t in params.values())
    print(f"{pt}")
    print(f"  layers={spec.num_layers} input_dim={spec.input_dim} "
          f"hidden_dim={spec.hidden_dim} num_classes={spec.num_classes}")
    if manifest:
        print(f"  manifest: dropout={spec.dropout} normalize={spec.normalize} "
              f"sample={manifest.get('sample', {}).get('sample_name', '?')} "
              f"fanouts={manifest.get('sample', {}).get('fanouts', '?')}")
    else:
        print("  (no export_manifest.json next to the archive)")
    print(f"  parameters: {total:,}")
    for name, t in params.items():
        print(f"    {name}  {tuple(t.shape)}")
    if spec.num_classes > spec.hidden_dim:
        print("  NOTE: num_classes > hidden_dim — the separate-classifier head "
              "caps the logit map rank at hidden_dim (raise hiddenDim if this "
              "configuration is trained from scratch).")
    return 0


def cmd_eval(args):
    results = evaluate_offline(args.run_dir, split=args.split)
    for name, r in results.items():
        print(f"{name}: accuracy={r['accuracy']:.7f} (n={r['n']})")
    return 0


def cmd_convert(args):
    pt = resolve_model_pt(args.path)
    manifest = read_manifest(Path(pt).parent)
    params = load_params(pt)
    if args.to == "statedict":
        torch.save(params, args.out)
    elif args.to == "dgl-collapsed":
        model = to_dgl_collapsed(params)
        torch.save(model.state_dict(), args.out)
    else:
        raise ValueError(args.to)
    spec = ArchSpec.from_params(params, manifest)
    print(f"wrote {args.out} ({args.to}, L={spec.num_layers} "
          f"in={spec.input_dim} hid={spec.hidden_dim} classes={spec.num_classes})")
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("info", help="architecture + parameters of a .pt archive")
    p.add_argument("path")
    p.set_defaults(fn=cmd_info)

    p = sub.add_parser("eval", help="offline accuracy from an exported run dir")
    p.add_argument("run_dir")
    p.add_argument("--split", default="test",
                   choices=["train", "validation", "test", "all"])
    p.set_defaults(fn=cmd_eval)

    p = sub.add_parser("convert", help="convert weights for external frameworks")
    p.add_argument("path")
    p.add_argument("--to", required=True, choices=["statedict", "dgl-collapsed"])
    p.add_argument("--out", required=True)
    p.set_defaults(fn=cmd_convert)

    p = sub.add_parser("selftest", help="conversion exactness across random configs")
    p.set_defaults(fn=lambda a: selftest())

    args = ap.parse_args(argv)
    return args.fn(args)


if __name__ == "__main__":
    sys.exit(main())
