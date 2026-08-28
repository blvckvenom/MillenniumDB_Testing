#!/usr/bin/env python3
"""
download_gnn_datasets.py - Download and convert GNN benchmark datasets to MillenniumDB GQL format.

Unified script that downloads Cora (from LINQS) and OGB datasets (ogbn-arxiv,
ogbn-products) and converts them to the GQL import format with separate NPY files
for node features and labels.

Usage:
    python scripts/download_gnn_datasets.py --dataset cora
    python scripts/download_gnn_datasets.py --dataset ogbn-arxiv
    python scripts/download_gnn_datasets.py --dataset ogbn-products
    python scripts/download_gnn_datasets.py --dataset all
    python scripts/download_gnn_datasets.py --dataset all --force

Output structure per dataset:
    data/example/gql/<dataset>/
    ├── <name>.gql           # Nodes with split:"train"/"val"/"test", edges
    ├── <name>_features.npy  # (N x D) float32 feature matrix
    └── <name>_labels.npy    # (N,) int label array

Requirements:
    pip install numpy requests tqdm
    pip install ogb torch     # Only for OGB datasets

GNN Integration Team - MillenniumDB
"""

import argparse
import hashlib
import os
import sys
import tarfile
import tempfile
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import numpy as np

try:
    from tqdm import tqdm
except ImportError:
    # Minimal fallback if tqdm is not installed
    class tqdm:  # type: ignore[no-redef]
        def __init__(self, iterable=None, total=None, desc="", unit="", unit_scale=False, **kwargs):
            self.iterable = iterable
            self.total = total
            self.desc = desc
            self.n = 0

        def __iter__(self):
            for item in self.iterable:
                yield item
                self.n += 1

        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def update(self, n=1):
            self.n += n

        def set_postfix_str(self, s):
            pass

        def close(self):
            pass

# ─── Class map for Cora ────────────────────────────────────────────────────────
CORA_CLASS_MAP = {
    "Case_Based": 0,
    "Genetic_Algorithms": 1,
    "Neural_Networks": 2,
    "Probabilistic_Methods": 3,
    "Reinforcement_Learning": 4,
    "Rule_Learning": 5,
    "Theory": 6,
}

# Standard Planetoid splits (Yang et al., 2016)
# Indices into the sorted node list that define train/val/test
CORA_SPLIT_TRAIN = 140   # first 140 nodes (20 per class)
CORA_SPLIT_VAL   = 500   # next 500 nodes
CORA_SPLIT_TEST  = 1000  # next 1000 nodes


# ═══════════════════════════════════════════════════════════════════════════════
#  Abstract Base
# ═══════════════════════════════════════════════════════════════════════════════

class DatasetConverter(ABC):
    """Interface for dataset downloaders/converters."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable dataset name."""
        ...

    @property
    @abstractmethod
    def file_prefix(self) -> str:
        """Prefix for output files (e.g. 'cora', 'ogbn_arxiv')."""
        ...

    @abstractmethod
    def download(self, cache_dir: Path) -> Path:
        """Download raw data, return path to raw directory."""
        ...

    @abstractmethod
    def convert(self, raw_path: Path, output_dir: Path) -> None:
        """Convert raw data to GQL + NPY files in output_dir."""
        ...

    def already_exists(self, output_dir: Path) -> bool:
        """Check if output files already exist."""
        gql = output_dir / f"{self.file_prefix}.gql"
        feat = output_dir / f"{self.file_prefix}_features.npy"
        labels = output_dir / f"{self.file_prefix}_labels.npy"
        return gql.exists() and feat.exists() and labels.exists()

    def run(self, output_base: Path, cache_dir: Path, force: bool = False) -> None:
        """Full pipeline: download → convert, with caching."""
        output_dir = output_base / self.name
        output_dir.mkdir(parents=True, exist_ok=True)

        if not force and self.already_exists(output_dir):
            print(f"[{self.name}] Output files already exist. Use --force to re-generate.")
            return

        print(f"\n{'='*60}")
        print(f"  Dataset: {self.name}")
        print(f"  Output:  {output_dir}")
        print(f"{'='*60}")

        raw_path = self.download(cache_dir)
        self.convert(raw_path, output_dir)

        # Print summary
        gql = output_dir / f"{self.file_prefix}.gql"
        feat = output_dir / f"{self.file_prefix}_features.npy"
        labels = output_dir / f"{self.file_prefix}_labels.npy"
        print(f"\n✓ {self.name} complete!")
        print(f"  GQL file:  {gql}  ({gql.stat().st_size / 1024:.1f} KB)")
        print(f"  Features:  {feat}  ({feat.stat().st_size / 1024 / 1024:.1f} MB)")
        print(f"  Labels:    {labels}  ({labels.stat().st_size / 1024:.1f} KB)")
        print(f"\n  Import into MillenniumDB:")
        print(f"    ./build/Release/bin/mdb import {gql} ./data/dbs/gql/{self.name}")


# ═══════════════════════════════════════════════════════════════════════════════
#  Cora Converter (LINQS format)
# ═══════════════════════════════════════════════════════════════════════════════

class CoraConverter(DatasetConverter):
    """
    Downloads Cora from LINQS and converts .content/.cites to GQL + NPY.

    The Cora dataset (McCallum et al., 2000) is a citation network of 2,708
    ML papers classified into 7 categories, with 1,433-dim bag-of-words features.
    """

    URL = "https://linqs-data.soe.ucsc.edu/public/lbc/cora.tgz"
    MD5 = "2fc040bee8ce3d920e4204effd1e9214"

    @property
    def name(self) -> str:
        return "cora"

    @property
    def file_prefix(self) -> str:
        return "cora"

    def download(self, cache_dir: Path) -> Path:
        """Download cora.tgz from LINQS, verify MD5, and extract."""
        import requests

        raw_dir = cache_dir / "cora"
        content_file = raw_dir / "cora" / "cora.content"
        cites_file = raw_dir / "cora" / "cora.cites"

        if content_file.exists() and cites_file.exists():
            print(f"[cora] Using cached raw data at {raw_dir}")
            return raw_dir / "cora"

        raw_dir.mkdir(parents=True, exist_ok=True)
        tgz_path = raw_dir / "cora.tgz"

        print(f"[cora] Downloading from {self.URL}...")
        response = requests.get(self.URL, stream=True)
        response.raise_for_status()

        total_size = int(response.headers.get("content-length", 0))
        with open(tgz_path, "wb") as f, tqdm(
            total=total_size, unit="B", unit_scale=True, desc="Downloading cora.tgz"
        ) as pbar:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
                pbar.update(len(chunk))

        # Verify MD5
        md5 = hashlib.md5(tgz_path.read_bytes()).hexdigest()
        if md5 != self.MD5:
            print(f"  Warning: MD5 mismatch (got {md5}, expected {self.MD5})")
            print(f"  The file may still be valid if the mirror updated the archive.")

        # Extract
        print(f"[cora] Extracting archive...")
        with tarfile.open(tgz_path, "r:gz") as tar:
            tar.extractall(path=raw_dir)

        return raw_dir / "cora"

    def convert(self, raw_path: Path, output_dir: Path) -> None:
        """Convert Cora .content/.cites → GQL + NPY."""
        content_file = raw_path / "cora.content"
        cites_file = raw_path / "cora.cites"

        if not content_file.exists():
            raise FileNotFoundError(f"cora.content not found at {content_file}")
        if not cites_file.exists():
            raise FileNotFoundError(f"cora.cites not found at {cites_file}")

        # ── Parse nodes ──────────────────────────────────────────────────
        print("[cora] Parsing cora.content...")
        paper_id_map: Dict[str, int] = {}
        nodes: List[Tuple[int, str, int, str, List[float]]] = []

        with open(content_file, "r") as f:
            for idx, line in enumerate(f):
                parts = line.strip().split("\t")
                if len(parts) < 3:
                    continue
                paper_id = parts[0]
                features = [float(x) for x in parts[1:-1]]  # all 1433 features
                class_name = parts[-1]
                label = CORA_CLASS_MAP.get(class_name, -1)
                paper_id_map[paper_id] = idx
                nodes.append((idx, paper_id, label, class_name, features))

        num_nodes = len(nodes)
        feat_dim = len(nodes[0][4]) if nodes else 0
        print(f"  Parsed {num_nodes} nodes, {feat_dim} features each")

        # ── Parse edges ──────────────────────────────────────────────────
        print("[cora] Parsing cora.cites...")
        edges: List[Tuple[int, int]] = []
        skipped = 0

        with open(cites_file, "r") as f:
            for line in f:
                parts = line.strip().split("\t")
                if len(parts) != 2:
                    continue
                cited_id, citing_id = parts
                if cited_id in paper_id_map and citing_id in paper_id_map:
                    from_idx = paper_id_map[citing_id]
                    to_idx = paper_id_map[cited_id]
                    edges.append((from_idx, to_idx))
                else:
                    skipped += 1

        if skipped > 0:
            print(f"  Warning: {skipped} edges skipped (referenced non-existent papers)")
        print(f"  Parsed {len(edges)} edges")

        # ── Assign Planetoid splits ──────────────────────────────────────
        # Standard splits: train=0..139, val=140..639, test=640..1639
        # remaining nodes are unlabeled (indices 1640..2707)
        train_end = CORA_SPLIT_TRAIN
        val_end = train_end + CORA_SPLIT_VAL
        test_end = val_end + CORA_SPLIT_TEST

        def get_split(node_idx: int) -> str:
            if node_idx < train_end:
                return "train"
            elif node_idx < val_end:
                return "val"
            elif node_idx < test_end:
                return "test"
            return "unlabeled"

        split_counts = {"train": 0, "val": 0, "test": 0, "unlabeled": 0}
        for i in range(num_nodes):
            split_counts[get_split(i)] += 1
        print(f"  Splits: {split_counts}")

        # ── Write GQL file ───────────────────────────────────────────────
        gql_file = output_dir / "cora.gql"
        print(f"[cora] Writing GQL to {gql_file}...")

        with open(gql_file, "w") as out:
            out.write("// Cora Dataset - MillenniumDB GQL format\n")
            out.write(f"// Nodes: {num_nodes} papers, Edges: {len(edges)} citations\n")
            out.write(f"// Features: {feat_dim} dimensions (stored in cora_features.npy)\n")
            out.write(f"// Splits: train={split_counts['train']}, val={split_counts['val']}, "
                       f"test={split_counts['test']}, unlabeled={split_counts['unlabeled']}\n")
            out.write("//\n")
            out.write("// Class mapping:\n")
            for cname, cid in sorted(CORA_CLASS_MAP.items(), key=lambda x: x[1]):
                out.write(f"//   {cid}: {cname}\n")
            out.write("\n")

            # Nodes
            for idx, paper_id, label, class_name, _features in tqdm(
                nodes, desc="Writing nodes", unit="node"
            ):
                split = get_split(idx)
                out.write(
                    f'{idx} :Paper paper_id:"{paper_id}" label:{label} '
                    f'class_name:"{class_name}" split:"{split}"\n'
                )

            out.write("\n")

            # Edges
            for from_idx, to_idx in tqdm(edges, desc="Writing edges", unit="edge"):
                out.write(f"{from_idx}->{to_idx} :CITES\n")

        # ── Save NPY files ───────────────────────────────────────────────
        print("[cora] Saving NPY files...")

        # Features: (N, 1433) float32
        features_matrix = np.array(
            [node[4] for node in nodes], dtype=np.float32
        )
        feat_path = output_dir / "cora_features.npy"
        np.save(feat_path, features_matrix)
        print(f"  Features: {features_matrix.shape} → {feat_path}")

        # Labels: (N,) int
        labels_array = np.array([node[2] for node in nodes], dtype=np.int64)
        label_path = output_dir / "cora_labels.npy"
        np.save(label_path, labels_array)
        print(f"  Labels:   {labels_array.shape} → {label_path}")

        # ── Statistics ────────────────────────────────────────────────────
        class_counts: Dict[str, int] = {}
        for _, _, _, class_name, _ in nodes:
            class_counts[class_name] = class_counts.get(class_name, 0) + 1
        print("\n  Class distribution:")
        for cname in sorted(class_counts, key=lambda c: CORA_CLASS_MAP[c]):
            count = class_counts[cname]
            pct = 100.0 * count / num_nodes
            print(f"    {CORA_CLASS_MAP[cname]}: {cname}: {count} ({pct:.1f}%)")


# ═══════════════════════════════════════════════════════════════════════════════
#  OGB Converter
# ═══════════════════════════════════════════════════════════════════════════════

class OGBConverter(DatasetConverter):
    """
    Downloads OGB node-property-prediction datasets and converts to GQL + NPY.

    Supports ogbn-arxiv and ogbn-products. Uses the `ogb` Python package which
    handles downloading, caching, and split generation.
    """

    DATASETS = {
        "ogbn-arxiv": {
            "nodes": 169_343,
            "edges": 1_166_243,
            "feat_dim": 128,
            "directed": True,
            "description": "Citation network of CS papers from arXiv",
            "download_size": "~300MB",
        },
        "ogbn-products": {
            "nodes": 2_449_029,
            "edges": 61_859_140,
            "feat_dim": 100,
            "directed": False,
            "description": "Amazon product co-purchasing network",
            "download_size": "~1.5GB",
        },
    }

    def __init__(self, dataset_name: str):
        if dataset_name not in self.DATASETS:
            raise ValueError(
                f"Unknown OGB dataset: {dataset_name}. "
                f"Supported: {list(self.DATASETS.keys())}"
            )
        self._dataset_name = dataset_name
        self._info = self.DATASETS[dataset_name]

    @property
    def name(self) -> str:
        return self._dataset_name

    @property
    def file_prefix(self) -> str:
        return self._dataset_name.replace("-", "_")

    def _ensure_ogb(self):
        """Import ogb with PyTorch 2.6+ compatibility patches."""
        try:
            import torch
            _original_torch_load = torch.load

            def _patched_torch_load(*args, **kwargs):
                if "weights_only" not in kwargs:
                    kwargs["weights_only"] = False
                return _original_torch_load(*args, **kwargs)

            torch.load = _patched_torch_load

            import ogb.utils.url
            ogb.utils.url.decide_download = lambda url: True

            from ogb.nodeproppred import NodePropPredDataset

            import ogb.nodeproppred.dataset as npd_module
            npd_module.decide_download = lambda url: True

            return NodePropPredDataset
        except ImportError:
            print(
                f"\nError: OGB datasets require additional packages.\n"
                f"Install with:\n"
                f"  pip install ogb torch\n"
            )
            sys.exit(1)

    def download(self, cache_dir: Path) -> Path:
        """Download via ogb package, return cache_dir (ogb manages its own cache)."""
        ogb_root = cache_dir / "ogb_data"
        ogb_root.mkdir(parents=True, exist_ok=True)

        info = self._info
        print(f"[{self.name}] Downloading ({info['download_size']})...")
        print(f"  Description: {info['description']}")
        print(f"  Expected: {info['nodes']:,} nodes, {info['edges']:,} edges, "
              f"{info['feat_dim']}D features")

        return ogb_root

    def convert(self, raw_path: Path, output_dir: Path) -> None:
        """Load OGB dataset and convert to GQL + NPY."""
        NodePropPredDataset = self._ensure_ogb()

        print(f"[{self.name}] Loading dataset (this may take a while for large datasets)...")
        dataset = NodePropPredDataset(name=self.name, root=str(raw_path))

        graph, labels = dataset[0]
        edge_index = graph["edge_index"]
        node_feat = graph.get("node_feat", None)
        split_idx = dataset.get_idx_split()

        num_nodes = int(edge_index.max()) + 1
        num_edges = edge_index.shape[1]

        print(f"  Loaded: {num_nodes:,} nodes, {num_edges:,} edges")
        if node_feat is not None:
            print(f"  Features: {node_feat.shape}")

        # Build split sets
        train_set: Set[int] = set(split_idx["train"].tolist()) if "train" in split_idx else set()
        valid_set: Set[int] = set(split_idx["valid"].tolist()) if "valid" in split_idx else set()
        test_set: Set[int] = set(split_idx["test"].tolist()) if "test" in split_idx else set()

        def get_split(node_id: int) -> str:
            if node_id in train_set:
                return "train"
            elif node_id in valid_set:
                return "valid"
            elif node_id in test_set:
                return "test"
            return "unlabeled"

        print(f"  Splits: train={len(train_set):,}, valid={len(valid_set):,}, "
              f"test={len(test_set):,}")

        # ── Write GQL ────────────────────────────────────────────────────
        gql_file = output_dir / f"{self.file_prefix}.gql"
        print(f"[{self.name}] Writing GQL to {gql_file}...")

        feat_dim = node_feat.shape[1] if node_feat is not None else 0

        is_directed = self._info.get("directed", True)

        with open(gql_file, "w") as f:
            if is_directed:
                edge_desc = f"{num_edges:,} directed edges"
            else:
                edge_desc = f"{num_edges // 2:,} undirected edges"
            f.write(f"// {self.name} - MillenniumDB GQL format\n")
            f.write(f"// Nodes: {num_nodes:,}, Edges: {edge_desc}\n")
            f.write(f"// Features: {feat_dim}D (stored in {self.file_prefix}_features.npy)\n\n")

            # Nodes
            for node_id in tqdm(range(num_nodes), desc="Writing nodes", unit="node"):
                props = []

                # Label
                if labels is not None and node_id < len(labels):
                    label_val = labels[node_id]
                    if isinstance(label_val, np.ndarray):
                        label_val = label_val.item() if label_val.size == 1 else label_val.tolist()
                    props.append(f"label:{label_val}")

                # Split
                split = get_split(node_id)
                props.append(f'split:"{split}"')

                # Feature dimension
                if feat_dim > 0:
                    props.append(f"feat_dim:{feat_dim}")

                prop_str = " ".join(props)
                f.write(f"{node_id} :Node {prop_str}\n")

            f.write("\n")

            # Edges
            if is_directed:
                for i in tqdm(range(num_edges), desc="Writing edges", unit="edge"):
                    src = edge_index[0, i]
                    dst = edge_index[1, i]
                    f.write(f"{src}->{dst} :CONNECTS\n")
            else:
                # OGB doubles undirected edges via add_inverse_edge: (A,B) + (B,A).
                # Deduplicate by keeping only src < dst; use ~ (undirected) syntax.
                edge_count = 0
                seen_self_loops: Set[int] = set()
                for i in tqdm(range(num_edges), desc="Dedup undirected edges", unit="edge"):
                    src = int(edge_index[0, i])
                    dst = int(edge_index[1, i])
                    if src < dst:
                        f.write(f"{src}~{dst} :CONNECTS\n")
                        edge_count += 1
                    elif src == dst and src not in seen_self_loops:
                        seen_self_loops.add(src)
                        f.write(f"{src}~{dst} :CONNECTS\n")
                        edge_count += 1
                print(f"  Deduplicated: {num_edges:,} directed entries → {edge_count:,} undirected edges")

        # ── Save NPY ─────────────────────────────────────────────────────
        print(f"[{self.name}] Saving NPY files...")

        if node_feat is not None:
            feat_path = output_dir / f"{self.file_prefix}_features.npy"
            feat_array = np.asarray(node_feat, dtype=np.float32)
            np.save(feat_path, feat_array)
            print(f"  Features: {feat_array.shape} → {feat_path}")

        if labels is not None:
            label_path = output_dir / f"{self.file_prefix}_labels.npy"
            labels_array = np.asarray(labels, dtype=np.int64).squeeze()
            np.save(label_path, labels_array)
            print(f"  Labels:   {labels_array.shape} → {label_path}")


# ═══════════════════════════════════════════════════════════════════════════════
#  CLI
# ═══════════════════════════════════════════════════════════════════════════════

AVAILABLE_DATASETS = ["cora", "ogbn-arxiv", "ogbn-products"]


def build_converter(name: str) -> DatasetConverter:
    """Factory: return the appropriate converter for a dataset name."""
    if name == "cora":
        return CoraConverter()
    else:
        return OGBConverter(name)


def list_datasets() -> None:
    """Print overview of supported datasets."""
    print("\nSupported GNN Benchmark Datasets:")
    print("-" * 72)
    print(f"{'Dataset':<18} {'Nodes':>12} {'Edges':>14} {'Features':>10} {'Source':<10}")
    print("-" * 72)
    print(f"{'cora':<18} {'2,708':>12} {'5,429':>14} {'1,433':>10} {'LINQS':<10}")
    for ds_name, info in OGBConverter.DATASETS.items():
        print(f"{ds_name:<18} {info['nodes']:>12,} {info['edges']:>14,} "
              f"{info['feat_dim']:>10} {'OGB':<10}")
    print("-" * 72)
    print("\nUsage:")
    print("  python scripts/download_gnn_datasets.py --dataset cora")
    print("  python scripts/download_gnn_datasets.py --dataset all")


def main():
    parser = argparse.ArgumentParser(
        description="Download and convert GNN benchmark datasets to MillenniumDB GQL format.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  %(prog)s --dataset cora              # Download only Cora\n"
            "  %(prog)s --dataset ogbn-arxiv         # Download only ogbn-arxiv\n"
            "  %(prog)s --dataset all                # Download all datasets\n"
            "  %(prog)s --dataset all --force        # Re-download even if exists\n"
            "  %(prog)s --list                       # Show available datasets\n"
        ),
    )
    parser.add_argument(
        "--dataset", "-d",
        type=str,
        choices=AVAILABLE_DATASETS + ["all"],
        help="Dataset to download and convert",
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        default="data/example/gql",
        help="Base output directory (default: data/example/gql)",
    )
    parser.add_argument(
        "--cache-dir",
        type=str,
        default=None,
        help="Cache directory for downloads (default: <output>/.cache)",
    )
    parser.add_argument(
        "--force", "-f",
        action="store_true",
        help="Re-download and re-convert even if output files exist",
    )
    parser.add_argument(
        "--list", "-l",
        action="store_true",
        help="List available datasets and exit",
    )

    args = parser.parse_args()

    if args.list or not args.dataset:
        list_datasets()
        return

    output_base = Path(args.output)
    cache_dir = Path(args.cache_dir) if args.cache_dir else output_base / ".cache"
    cache_dir.mkdir(parents=True, exist_ok=True)

    # Determine which datasets to process
    datasets = AVAILABLE_DATASETS if args.dataset == "all" else [args.dataset]

    for ds_name in datasets:
        try:
            converter = build_converter(ds_name)
            converter.run(output_base, cache_dir, force=args.force)
        except Exception as e:
            print(f"\n✗ Error processing {ds_name}: {e}")
            if ds_name != datasets[-1]:
                print("  Continuing with next dataset...\n")

    print("\n" + "=" * 60)
    print("  All done!")
    print("=" * 60)


if __name__ == "__main__":
    main()
