#!/usr/bin/env python3
"""
Download and convert OGB datasets to MillenniumDB GQL format.

This script downloads Open Graph Benchmark datasets and converts them
to the GQL format for import into MillenniumDB.

Usage:
    python download_ogb.py --dataset ogbn-arxiv --output data/example/gql/ogbn-arxiv/
    python download_ogb.py --dataset ogbn-products --output data/example/gql/ogbn-products/

Requirements:
    pip install ogb torch numpy

Dataset sizes (for reference):
    ogbn-arxiv:      169K nodes,   1.2M edges  (~300MB download)
    ogbn-products:   2.4M nodes,  61.9M edges  (~1.5GB download)
    ogbn-mag:        1.9M nodes,  21.1M edges  (~500MB download)
    ogbn-papers100M: 111M nodes,   1.6B edges  (~50GB download)

Author: GNN Integration Team
"""

import argparse
import os
import sys
from pathlib import Path
from typing import Optional, Tuple
import numpy as np

try:
    # CRITICAL: Patch torch.load BEFORE any ogb imports
    # PyTorch 2.6+ changed default weights_only=False to True
    # OGB was written before this change and doesn't work with weights_only=True
    import torch
    _original_torch_load = torch.load
    def _patched_torch_load(*args, **kwargs):
        if 'weights_only' not in kwargs:
            kwargs['weights_only'] = False
        return _original_torch_load(*args, **kwargs)
    torch.load = _patched_torch_load

    # Patch decide_download to auto-accept downloads
    import ogb.utils.url
    ogb.utils.url.decide_download = lambda url: True

    from ogb.nodeproppred import NodePropPredDataset

    # Patch again after import in case it was cached
    import ogb.nodeproppred.dataset as npd_module
    npd_module.decide_download = lambda url: True
except ImportError:
    print("Error: ogb package not found. Install with: pip install ogb")
    sys.exit(1)


DATASET_INFO = {
    "ogbn-arxiv": {
        "nodes": 169_343,
        "edges": 1_166_243,
        "description": "Citation network of CS papers from arXiv",
        "download_size": "~300MB",
    },
    "ogbn-products": {
        "nodes": 2_449_029,
        "edges": 61_859_140,
        "description": "Amazon product co-purchasing network",
        "download_size": "~1.5GB",
    },
    "ogbn-mag": {
        "nodes": 1_939_743,
        "edges": 21_111_007,
        "description": "Microsoft Academic Graph (heterogeneous)",
        "download_size": "~500MB",
    },
    "ogbn-proteins": {
        "nodes": 132_534,
        "edges": 39_561_252,
        "description": "Protein-protein association network",
        "download_size": "~200MB",
    },
    "ogbn-papers100M": {
        "nodes": 111_059_956,
        "edges": 1_615_685_872,
        "description": "Large-scale citation network",
        "download_size": "~50GB",
    },
}


def download_dataset(name: str, root: str = "./ogb_data") -> Tuple:
    """
    Download OGB dataset and return graph data.

    Returns:
        Tuple of (edge_index, node_features, labels, split_idx)
    """
    print(f"Downloading {name}...")
    print(f"  Expected size: {DATASET_INFO.get(name, {}).get('download_size', 'unknown')}")

    dataset = NodePropPredDataset(name=name, root=root)

    # Get the graph
    graph, labels = dataset[0]

    # edge_index is [2, num_edges] array
    edge_index = graph["edge_index"]

    # node features
    node_feat = graph.get("node_feat", None)

    # Get split indices
    split_idx = dataset.get_idx_split()

    print(f"  Loaded {edge_index.shape[1]:,} edges")
    print(f"  Node features shape: {node_feat.shape if node_feat is not None else 'None'}")

    return edge_index, node_feat, labels, split_idx


def convert_to_gql(
    edge_index: np.ndarray,
    node_feat: Optional[np.ndarray],
    labels: np.ndarray,
    split_idx: dict,
    output_path: Path,
    dataset_name: str,
    max_nodes: Optional[int] = None,
) -> None:
    """
    Convert OGB data to MillenniumDB GQL format.

    MillenniumDB GQL import format:
        Nodes: <id> :<label> [prop:value ...]
        Edges: <from_id>-><to_id> :<label> [prop:value ...]

    Example:
        0 :Node label:3 split:"train" feat_dim:100
        0->1 :CONNECTS
    """
    output_path.mkdir(parents=True, exist_ok=True)
    gql_file = output_path / f"{dataset_name.replace('-', '_')}.gql"

    num_nodes = edge_index.max() + 1
    if max_nodes and max_nodes < num_nodes:
        num_nodes = max_nodes
        print(f"  Limiting to {num_nodes:,} nodes")

    # Determine split for each node
    train_set = set(split_idx["train"].tolist()) if "train" in split_idx else set()
    valid_set = set(split_idx["valid"].tolist()) if "valid" in split_idx else set()
    test_set = set(split_idx["test"].tolist()) if "test" in split_idx else set()

    def get_split(node_id: int) -> str:
        if node_id in train_set:
            return "train"
        elif node_id in valid_set:
            return "valid"
        elif node_id in test_set:
            return "test"
        return "unlabeled"

    print(f"  Writing GQL to {gql_file}...")

    with open(gql_file, "w") as f:
        # Write nodes in MillenniumDB format: <id> :<label> [prop:value ...]
        for node_id in range(num_nodes):
            if node_id % 100_000 == 0:
                print(f"    Nodes: {node_id:,}/{num_nodes:,}")

            props = []

            # Add label if available
            if labels is not None and node_id < len(labels):
                label_val = labels[node_id]
                if isinstance(label_val, np.ndarray):
                    label_val = label_val.item() if label_val.size == 1 else label_val.tolist()
                props.append(f"label:{label_val}")

            # Add split as quoted string
            split = get_split(node_id)
            props.append(f'split:"{split}"')

            # Add feature dimension
            if node_feat is not None and node_id < len(node_feat):
                feat = node_feat[node_id]
                props.append(f"feat_dim:{len(feat)}")

            prop_str = " ".join(props)
            f.write(f"{node_id} :Node {prop_str}\n")

        # Write edges in MillenniumDB format: <from_id>-><to_id> :<label>
        edge_count = 0
        for i in range(edge_index.shape[1]):
            src = edge_index[0, i]
            dst = edge_index[1, i]

            # Skip edges outside our node limit
            if max_nodes and (src >= max_nodes or dst >= max_nodes):
                continue

            if edge_count % 1_000_000 == 0:
                print(f"    Edges: {edge_count:,}/{edge_index.shape[1]:,}")

            f.write(f"{src}->{dst} :CONNECTS\n")
            edge_count += 1

        print(f"  Wrote {num_nodes:,} nodes and {edge_count:,} edges")

    # Save full features to numpy file for GNN training
    if node_feat is not None:
        feat_file = output_path / f"{dataset_name.replace('-', '_')}_features.npy"
        if max_nodes:
            np.save(feat_file, node_feat[:max_nodes])
        else:
            np.save(feat_file, node_feat)
        print(f"  Saved features to {feat_file}")

    # Save labels
    if labels is not None:
        label_file = output_path / f"{dataset_name.replace('-', '_')}_labels.npy"
        if max_nodes:
            np.save(label_file, labels[:max_nodes])
        else:
            np.save(label_file, labels)
        print(f"  Saved labels to {label_file}")


def list_datasets():
    """Print available datasets and their sizes."""
    print("\nAvailable OGB Node Property Prediction Datasets:")
    print("-" * 70)
    print(f"{'Dataset':<20} {'Nodes':>12} {'Edges':>15} {'Download':<12}")
    print("-" * 70)

    for name, info in DATASET_INFO.items():
        print(f"{name:<20} {info['nodes']:>12,} {info['edges']:>15,} {info['download_size']:<12}")

    print("-" * 70)
    print("\nRecommended for benchmarking:")
    print("  - ogbn-arxiv:    Quick validation (~1 min to process)")
    print("  - ogbn-products: Full benchmark (~10 min to process)")
    print("  - ogbn-papers100M: Stress test (requires ~64GB RAM)")


def main():
    parser = argparse.ArgumentParser(
        description="Download and convert OGB datasets to GQL format"
    )
    parser.add_argument(
        "--dataset", "-d",
        type=str,
        choices=list(DATASET_INFO.keys()),
        help="Dataset to download"
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        default="./data/example/gql/",
        help="Output directory for GQL files"
    )
    parser.add_argument(
        "--ogb-root",
        type=str,
        default="./ogb_data",
        help="Root directory for OGB downloads (cached)"
    )
    parser.add_argument(
        "--max-nodes",
        type=int,
        default=None,
        help="Limit number of nodes (for testing)"
    )
    parser.add_argument(
        "--list", "-l",
        action="store_true",
        help="List available datasets"
    )

    args = parser.parse_args()

    if args.list or not args.dataset:
        list_datasets()
        return

    # Download and convert
    edge_index, node_feat, labels, split_idx = download_dataset(
        args.dataset,
        root=args.ogb_root
    )

    output_path = Path(args.output) / args.dataset
    convert_to_gql(
        edge_index,
        node_feat,
        labels,
        split_idx,
        output_path,
        args.dataset,
        max_nodes=args.max_nodes
    )

    print(f"\nDone! Import into MillenniumDB with:")
    print(f"  ./build/Release/bin/mdb import {output_path}/*.gql ./data/dbs/gql/{args.dataset}")


if __name__ == "__main__":
    main()
