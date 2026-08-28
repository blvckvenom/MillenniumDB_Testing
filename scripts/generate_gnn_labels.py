#!/usr/bin/env python3
"""
Generate GNN binary files (labels.bin, splits.bin, gnn_meta.bin) for testing the GNN training pipeline.

This script reads a numpy labels file and generates three binary formats required for GNN training:
- labels.bin: Node class labels in binary format
- splits.bin: Train/val/test split assignments for each node
- gnn_meta.bin: Metadata about the dataset (feature dimension, node count, class count, etc.)

If a corresponding .gql file exists, splits are parsed from split properties. Otherwise, random 70/15/15 splits are generated.

Usage:
    python3 scripts/generate_gnn_labels.py <labels.npy> <output_dir> [--feature-name NAME] [--feature-dim DIM]

Example:
    python3 scripts/generate_gnn_labels.py \\
        data/example/gql/cora/cora_labels.npy \\
        /tmp/test_projection/ \\
        --feature-name node_features --feature-dim 1433
"""

import argparse
import struct
import sys
from pathlib import Path

try:
    import numpy as np
except ImportError:
    print("Error: numpy is required. Install with: pip install numpy", file=sys.stderr)
    sys.exit(1)


def parse_splits_from_gql(gql_file, num_nodes):
    """
    Parse split assignments from a .gql file.

    Looks for split:"train", split:"val", or split:"test" properties.
    Returns array of uint8: 0=TRAIN, 1=VAL, 2=TEST.
    Returns None if no split properties found.
    """
    split_map = {"train": 0, "val": 1, "test": 2}
    splits = None

    try:
        with open(gql_file, 'r') as f:
            for line_no, line in enumerate(f):
                # Skip comments and empty lines
                line = line.strip()
                if not line or line.startswith('//'):
                    continue

                # Parse node definition: <id> :NodeType ... split:"<value>"
                if 'split:' in line:
                    # Extract node ID (first token)
                    parts = line.split()
                    if not parts:
                        continue

                    try:
                        node_id = int(parts[0])
                    except ValueError:
                        continue

                    # Extract split value
                    if 'split:' in line:
                        # Find split:" ... "
                        start = line.find('split:"')
                        if start != -1:
                            start += len('split:"')
                            end = line.find('"', start)
                            if end != -1:
                                split_value = line[start:end].strip()
                                if split_value in split_map:
                                    if splits is None:
                                        splits = np.zeros(num_nodes, dtype=np.uint8)
                                    if node_id < num_nodes:
                                        splits[node_id] = split_map[split_value]
    except Exception as e:
        print(f"Warning: Could not parse splits from {gql_file}: {e}", file=sys.stderr)
        return None

    return splits


def generate_random_splits(num_nodes, train_ratio=0.7, val_ratio=0.15):
    """
    Generate random train/val/test splits.

    Args:
        num_nodes: Number of nodes
        train_ratio: Fraction for training (default 0.7)
        val_ratio: Fraction for validation (default 0.15)

    Returns:
        Array of uint8: 0=TRAIN, 1=VAL, 2=TEST
    """
    np.random.seed(42)  # Consistent seed for reproducibility
    splits = np.random.choice(
        [0, 1, 2],
        size=num_nodes,
        p=[train_ratio, val_ratio, 1.0 - train_ratio - val_ratio]
    ).astype(np.uint8)
    return splits


def write_labels_bin(output_file, labels):
    """
    Write labels.bin file.

    Format:
        magic: b'GNNL\x00\x00\x00\x00' (8 bytes)
        version: uint32 = 1 (4 bytes)
        reserved: uint32 = 0 (4 bytes)
        num_nodes: uint64 (8 bytes)
        num_classes: uint64 (8 bytes)
        data: int64[N] — label values
    """
    labels = np.asarray(labels, dtype=np.int64).flatten()
    num_nodes = len(labels)
    num_classes = int(np.max(labels)) + 1

    with open(output_file, 'wb') as f:
        # Header
        f.write(b'GNNL\x00\x00\x00\x00')
        f.write(struct.pack('<I', 1))  # version
        f.write(struct.pack('<I', 0))  # reserved
        f.write(struct.pack('<Q', num_nodes))
        f.write(struct.pack('<Q', num_classes))

        # Data
        f.write(labels.tobytes())

    return num_nodes, num_classes


def write_splits_bin(output_file, splits):
    """
    Write splits.bin file.

    Format:
        magic: b'GNNS\x00\x00\x00\x00' (8 bytes)
        version: uint32 = 1 (4 bytes)
        reserved: uint32 = 0 (4 bytes)
        num_nodes: uint64 (8 bytes)
        data: uint8[N] — 0=TRAIN, 1=VAL, 2=TEST
    """
    splits = np.asarray(splits, dtype=np.uint8).flatten()
    num_nodes = len(splits)

    with open(output_file, 'wb') as f:
        # Header
        f.write(b'GNNS\x00\x00\x00\x00')
        f.write(struct.pack('<I', 1))  # version
        f.write(struct.pack('<I', 0))  # reserved
        f.write(struct.pack('<Q', num_nodes))

        # Data
        f.write(splits.tobytes())

    return num_nodes


def write_gnn_meta_bin(output_file, feature_dim, num_nodes, num_classes, feature_name):
    """
    Write gnn_meta.bin file.

    Format:
        magic: b'GNNM\x00\x00\x00\x00' (8 bytes)
        version: uint32 = 1 (4 bytes)
        feature_dim: uint32 (4 bytes)
        num_nodes: uint64 (8 bytes)
        num_classes: uint64 (8 bytes)
        has_labels: uint8 = 1 (1 byte)
        has_splits: uint8 = 1 (1 byte)
        reserved: uint16 = 0 (2 bytes)
        feature_name_len: uint32 (4 bytes)
        feature_name: bytes (N bytes)
    """
    feature_name_bytes = feature_name.encode('utf-8')
    feature_name_len = len(feature_name_bytes)

    with open(output_file, 'wb') as f:
        # Header
        f.write(b'GNNM\x00\x00\x00\x00')
        f.write(struct.pack('<I', 1))  # version
        f.write(struct.pack('<I', feature_dim))
        f.write(struct.pack('<Q', num_nodes))
        f.write(struct.pack('<Q', num_classes))
        f.write(struct.pack('<B', 1))  # has_labels
        f.write(struct.pack('<B', 1))  # has_splits
        f.write(struct.pack('<H', 0))  # reserved
        f.write(struct.pack('<I', feature_name_len))

        # Feature name
        f.write(feature_name_bytes)


def main():
    parser = argparse.ArgumentParser(
        description='Generate GNN binary files for testing the GNN training pipeline',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python3 scripts/generate_gnn_labels.py \\
      data/example/gql/cora/cora_labels.npy \\
      /tmp/test_projection/ \\
      --feature-name node_features --feature-dim 1433
        """
    )

    parser.add_argument('labels_file', help='Path to labels.npy file')
    parser.add_argument('output_dir', help='Output directory for binary files')
    parser.add_argument('--feature-name', default='node_features',
                        help='Feature name for metadata (default: node_features)')
    parser.add_argument('--feature-dim', type=int, default=0,
                        help='Feature dimension (default: 0, auto-detect from features.npy)')

    args = parser.parse_args()

    # Validate inputs
    labels_file = Path(args.labels_file)
    if not labels_file.exists():
        print(f"Error: Labels file not found: {labels_file}", file=sys.stderr)
        sys.exit(1)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load labels
    try:
        labels = np.load(labels_file)
        print(f"Loaded labels from {labels_file}")
        print(f"  Shape: {labels.shape}, dtype: {labels.dtype}")
    except Exception as e:
        print(f"Error: Could not load labels file: {e}", file=sys.stderr)
        sys.exit(1)

    # Flatten if necessary (e.g., shape (N,1) -> (N,))
    labels = labels.flatten()
    num_nodes = len(labels)

    # Auto-detect feature dimension if not provided
    feature_dim = args.feature_dim
    if feature_dim == 0:
        # Try to load features.npy from same directory
        features_file = labels_file.parent / labels_file.name.replace('_labels.npy', '_features.npy')
        if features_file.exists():
            try:
                features = np.load(features_file)
                feature_dim = features.shape[1] if len(features.shape) > 1 else 1
                print(f"Auto-detected feature_dim={feature_dim} from {features_file}")
            except Exception as e:
                print(f"Warning: Could not auto-detect feature_dim: {e}", file=sys.stderr)
                feature_dim = 0
        else:
            print(f"Warning: Could not find {features_file} for auto-detection", file=sys.stderr)

    # Parse or generate splits
    gql_file = labels_file.parent / labels_file.name.replace('_labels.npy', '.gql')
    splits = None
    if gql_file.exists():
        splits = parse_splits_from_gql(gql_file, num_nodes)
        if splits is not None:
            print(f"Parsed splits from {gql_file}")
        else:
            print(f"No split properties found in {gql_file}, generating random splits")
            splits = generate_random_splits(num_nodes)
    else:
        print(f"No .gql file found at {gql_file}, generating random splits")
        splits = generate_random_splits(num_nodes)

    # Write binary files
    labels_out = output_dir / 'labels.bin'
    splits_out = output_dir / 'splits.bin'
    meta_out = output_dir / 'gnn_meta.bin'

    print(f"\nWriting binary files to {output_dir}...")

    # Write labels.bin
    num_nodes_labels, num_classes = write_labels_bin(labels_out, labels)
    print(f"  Wrote {labels_out} ({num_nodes_labels} nodes, {num_classes} classes)")

    # Write splits.bin
    num_nodes_splits = write_splits_bin(splits_out, splits)
    print(f"  Wrote {splits_out}")

    # Write gnn_meta.bin
    write_gnn_meta_bin(meta_out, feature_dim, num_nodes_labels, num_classes, args.feature_name)
    print(f"  Wrote {meta_out}")

    # Print summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(f"Nodes:          {num_nodes_labels}")
    print(f"Classes:        {num_classes}")
    print(f"Feature dim:    {feature_dim}")
    print(f"Feature name:   {args.feature_name}")

    # Split counts
    train_count = np.sum(splits == 0)
    val_count = np.sum(splits == 1)
    test_count = np.sum(splits == 2)
    print(f"\nSplit distribution:")
    print(f"  TRAIN:  {train_count:6d} ({100*train_count/num_nodes_labels:5.1f}%)")
    print(f"  VAL:    {val_count:6d} ({100*val_count/num_nodes_labels:5.1f}%)")
    print(f"  TEST:   {test_count:6d} ({100*test_count/num_nodes_labels:5.1f}%)")

    # Class distribution
    print(f"\nClass distribution:")
    for class_id in range(num_classes):
        count = np.sum(labels == class_id)
        print(f"  Class {class_id:2d}: {count:6d} ({100*count/num_nodes_labels:5.1f}%)")

    print("\n" + "="*60)
    print("Done!")


if __name__ == '__main__':
    main()
