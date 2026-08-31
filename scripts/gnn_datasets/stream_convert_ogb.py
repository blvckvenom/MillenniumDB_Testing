#!/usr/bin/env python3
"""
Streaming OGB -> MillenniumDB GQL converter.

Bypasses ogb.nodeproppred.NodePropPredDataset to avoid its eager full-array
allocation (which OOMs on commodity RAM: papers100M feature matrix alone is
53 GiB). Reads .npy entries out of the OGB data.npz zip archive chunk by
chunk via zipfile.ZipFile.open(), so peak RAM stays bounded (~500 MB)
regardless of dataset size. Generalizes to MAG240M (~750 GB features).

Output format (same as download_ogb.py to keep mdb import working):
    <dataset_name>.gql            — text, one line per node / edge
    <dataset_name>_features.npy   — (N, D) float32 memmap
    <dataset_name>_labels.npy     — (N,) or (N, 1) int64 labels

Usage:
    python stream_convert_ogb.py \\
        --raw-dir ogb_data/ogbn_papers100M/raw/ \\
        --split-dir ogb_data/ogbn_papers100M/split/time/ \\
        --output data/example/gql/ogbn-papers100M/ \\
        --dataset-name papers100M \\
        --directed
"""

import argparse
import gzip
import os
import sys
import time
import zipfile
from pathlib import Path
from typing import Iterator, Optional, Tuple

import numpy as np


CHUNK_BYTES = 64 * 1024 * 1024  # 64 MB decompression chunk
GQL_WRITE_BUF = 64 * 1024 * 1024  # 64 MB stdio buffer
EDGE_CHUNK = 1_000_000  # edges per .gql-writing batch
PROGRESS_EVERY = 5.0  # seconds


def read_npy_header(zip_path: Path, member: str) -> Tuple[Tuple[int, ...], np.dtype]:
    """Return (shape, dtype) of <member>.npy inside zip without reading data."""
    with zipfile.ZipFile(zip_path) as z:
        with z.open(member) as fp:
            version = np.lib.format.read_magic(fp)
            if version == (1, 0):
                shape, fortran, dtype = np.lib.format.read_array_header_1_0(fp)
            elif version == (2, 0):
                shape, fortran, dtype = np.lib.format.read_array_header_2_0(fp)
            else:
                raise ValueError(f"Unsupported .npy version {version}")
            if fortran:
                raise NotImplementedError(f"Fortran-order .npy not supported ({member})")
            return shape, dtype


def iter_npy_chunks(
    zip_path: Path,
    member: str,
    chunk_bytes: int = CHUNK_BYTES,
) -> Iterator[Tuple[int, np.ndarray]]:
    """
    Yield (offset_in_elements, chunk_ndarray) pairs streaming from a .npy
    stored inside a .npz zip archive. The zip entry must be DEFLATE or STORED.
    Chunk is a 1-D view; caller is responsible for reshaping / reinterpreting.
    """
    with zipfile.ZipFile(zip_path) as z:
        with z.open(member) as fp:
            version = np.lib.format.read_magic(fp)
            if version == (1, 0):
                shape, fortran, dtype = np.lib.format.read_array_header_1_0(fp)
            elif version == (2, 0):
                shape, fortran, dtype = np.lib.format.read_array_header_2_0(fp)
            else:
                raise ValueError(f"Unsupported .npy version {version}")
            if fortran:
                raise NotImplementedError(f"Fortran-order .npy not supported ({member})")

            elem_size = dtype.itemsize
            total_elems = int(np.prod(shape))
            chunk_elems = max(1, chunk_bytes // elem_size)

            offset = 0
            while offset < total_elems:
                n = min(chunk_elems, total_elems - offset)
                want = n * elem_size
                buf = fp.read(want)
                if len(buf) != want:
                    raise IOError(
                        f"Short read at elem offset {offset}: got {len(buf)} want {want}"
                    )
                # Copy so the caller can keep the array after 'buf' is GC'd.
                arr = np.frombuffer(buf, dtype=dtype, count=n).copy()
                yield offset, arr
                offset += n


def extract_features(
    zip_path: Path,
    output_path: Path,
    shape: Tuple[int, ...],
    dtype: np.dtype,
) -> None:
    """Decompress node_feat.npy from zip to a proper .npy memmap at output_path."""
    total_bytes = int(np.prod(shape)) * dtype.itemsize
    print(f"[1/6] Features -> {output_path}")
    print(f"       shape={shape}, dtype={dtype}, size={total_bytes/1e9:.2f} GB")

    out = np.lib.format.open_memmap(
        str(output_path), mode="w+", dtype=dtype, shape=shape
    )
    flat = out.reshape(-1)

    t0 = time.time()
    last_print = t0
    for offset, chunk in iter_npy_chunks(zip_path, "node_feat.npy"):
        flat[offset : offset + len(chunk)] = chunk
        now = time.time()
        if now - last_print > PROGRESS_EVERY:
            done_bytes = (offset + len(chunk)) * dtype.itemsize
            pct = 100.0 * done_bytes / total_bytes
            speed = done_bytes / (now - t0) / 1e6
            print(f"       {pct:5.1f}% ({speed:.0f} MB/s)")
            last_print = now

    del out  # flush memmap
    print(f"       DONE ({time.time() - t0:.1f}s)")


def stage_edges(
    zip_path: Path,
    staging_path: Path,
    shape: Tuple[int, ...],
    dtype: np.dtype,
) -> None:
    """Decompress edge_index.npy from zip to raw binary file (no .npy header)."""
    total_bytes = int(np.prod(shape)) * dtype.itemsize
    print(f"[2/6] Staging edges -> {staging_path}")
    print(f"       shape={shape}, dtype={dtype}, size={total_bytes/1e9:.2f} GB")

    t0 = time.time()
    last_print = t0
    written = 0

    with open(staging_path, "wb", buffering=64 * 1024 * 1024) as out:
        for _offset, chunk in iter_npy_chunks(zip_path, "edge_index.npy"):
            out.write(chunk.tobytes())
            written += chunk.nbytes
            now = time.time()
            if now - last_print > PROGRESS_EVERY:
                pct = 100.0 * written / total_bytes
                speed = written / (now - t0) / 1e6
                print(f"       {pct:5.1f}% ({speed:.0f} MB/s)")
                last_print = now

    print(f"       DONE ({time.time() - t0:.1f}s)")


def load_labels(label_npz_path: Path) -> Optional[np.ndarray]:
    """Load labels fully into RAM (typically small, e.g. 888 MB for papers100M)."""
    if not label_npz_path.exists():
        print(f"[3/6] Labels: {label_npz_path} not found, skipping")
        return None
    print(f"[3/6] Loading labels from {label_npz_path}")
    with np.load(label_npz_path) as f:
        keys = list(f.files)
        key = "node_label" if "node_label" in keys else keys[0]
        labels = f[key].copy()
    if labels.ndim == 2 and labels.shape[1] == 1:
        labels = labels[:, 0]
    print(f"       shape={labels.shape}, dtype={labels.dtype}, key='{key}'")
    return labels


def load_splits(split_dir: Path, num_nodes: int) -> dict:
    """Load {train,valid,test}.csv[.gz] as boolean arrays shape (num_nodes,)."""
    print(f"[4/6] Loading splits from {split_dir}")
    splits = {}
    for name in ("train", "valid", "test"):
        path_gz = split_dir / f"{name}.csv.gz"
        path_plain = split_dir / f"{name}.csv"
        if path_gz.exists():
            opener = lambda: gzip.open(path_gz, "rt")
            found = path_gz
        elif path_plain.exists():
            opener = lambda: open(path_plain, "r")
            found = path_plain
        else:
            print(f"       WARN: {name}.csv[.gz] not found -> empty mask")
            splits[name] = np.zeros(num_nodes, dtype=bool)
            continue
        with opener() as f:
            ids = np.fromiter(
                (int(line.strip()) for line in f if line.strip()),
                dtype=np.int64,
            )
        mask = np.zeros(num_nodes, dtype=bool)
        mask[ids] = True
        splits[name] = mask
        print(f"       {name}: {len(ids):,} nodes ({found.name})")
    return splits


def write_gql(
    output_gql: Path,
    num_nodes: int,
    feature_dim: int,
    labels: Optional[np.ndarray],
    splits: dict,
    edge_staging_path: Path,
    num_edges: int,
    directed: bool,
    edge_label: str,
    max_edges: Optional[int] = None,
) -> int:
    """Write nodes + edges to .gql. Returns number of edges written."""
    if max_edges is not None and max_edges < num_edges:
        print(f"       NOTE: truncating edges to {max_edges:,} for smoke test")
        num_edges_used = max_edges
    else:
        num_edges_used = num_edges

    print(f"[5/6] Writing GQL -> {output_gql}")
    print(
        f"       nodes={num_nodes:,}, edges={num_edges_used:,}, "
        f"directed={directed}, label='{edge_label}'"
    )

    edges_mm = np.memmap(
        edge_staging_path, dtype=np.int64, mode="r", shape=(2, num_edges)
    )

    train_mask = splits["train"]
    valid_mask = splits["valid"]
    test_mask = splits["test"]

    labels_is_float = labels is not None and labels.dtype.kind == "f"

    t0 = time.time()
    last_print = t0
    edge_count_written = 0
    seen_self = set() if not directed else None

    with open(output_gql, "w", buffering=GQL_WRITE_BUF) as f:
        # --- Nodes ---
        print("       Writing nodes...")
        node_buf = []
        NODE_FLUSH = 100_000
        for node_id in range(num_nodes):
            parts = [f"{node_id} :Node"]
            if labels is not None and node_id < len(labels):
                lv = labels[node_id]
                unlabeled = (
                    np.isnan(lv) if labels_is_float else (lv < 0)
                )
                if not unlabeled:
                    parts.append(f"label:{int(lv)}")
            if train_mask[node_id]:
                parts.append('split:"train"')
            elif valid_mask[node_id]:
                parts.append('split:"valid"')
            elif test_mask[node_id]:
                parts.append('split:"test"')
            else:
                parts.append('split:"unlabeled"')
            if feature_dim:
                parts.append(f"feat_dim:{feature_dim}")
            node_buf.append(" ".join(parts))

            if len(node_buf) >= NODE_FLUSH:
                f.write("\n".join(node_buf) + "\n")
                node_buf.clear()
                now = time.time()
                if now - last_print > PROGRESS_EVERY:
                    pct = 100.0 * (node_id + 1) / num_nodes
                    speed = (node_id + 1) / (now - t0) / 1e6
                    print(
                        f"       Nodes: {node_id+1:,}/{num_nodes:,} "
                        f"({pct:.1f}%, {speed:.2f}M nodes/s)"
                    )
                    last_print = now
        if node_buf:
            f.write("\n".join(node_buf) + "\n")
            node_buf.clear()
        print(f"       Nodes DONE ({time.time() - t0:.1f}s)")

        # --- Edges ---
        print("       Writing edges...")
        t_e = time.time()
        for start in range(0, num_edges_used, EDGE_CHUNK):
            end = min(start + EDGE_CHUNK, num_edges_used)
            srcs = edges_mm[0, start:end].tolist()  # Python ints for fast f-string
            dsts = edges_mm[1, start:end].tolist()

            if directed:
                lines = [
                    f"{srcs[i]}->{dsts[i]} :{edge_label}" for i in range(end - start)
                ]
                f.write("\n".join(lines))
                f.write("\n")
                edge_count_written += end - start
            else:
                lines = []
                for i in range(end - start):
                    s = srcs[i]
                    d = dsts[i]
                    if s < d:
                        lines.append(f"{s}~{d} :{edge_label}")
                        edge_count_written += 1
                    elif s == d and s not in seen_self:
                        seen_self.add(s)
                        lines.append(f"{s}~{d} :{edge_label}")
                        edge_count_written += 1
                if lines:
                    f.write("\n".join(lines))
                    f.write("\n")

            now = time.time()
            if now - last_print > PROGRESS_EVERY:
                pct = 100.0 * end / num_edges_used
                speed = end / (now - t_e) / 1e6
                print(
                    f"       Edges: {end:,}/{num_edges_used:,} "
                    f"({pct:.1f}%, {speed:.1f}M edges/s)"
                )
                last_print = now

        print(
            f"       Edges DONE: {edge_count_written:,} written "
            f"({time.time() - t_e:.1f}s)"
        )

    del edges_mm
    print(f"       GQL DONE ({time.time() - t0:.1f}s)")
    return edge_count_written


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--raw-dir", type=Path, required=True,
                    help="Dir containing data.npz and node-label.npz")
    ap.add_argument("--split-dir", type=Path, required=True,
                    help="Dir containing {train,valid,test}.csv[.gz]")
    ap.add_argument("--output", type=Path, required=True,
                    help="Output directory for .gql + .npy files")
    ap.add_argument("--dataset-name", type=str, required=True,
                    help="Prefix for output files, e.g. 'papers100M'")
    ap.add_argument("--directed", action="store_true",
                    help="Write edges as '<s>-><d>' (default: undirected '~' with dedup)")
    ap.add_argument("--edge-label", type=str, default=None,
                    help="Edge label (default: CITES if directed else CONNECTS)")
    ap.add_argument("--max-edges", type=int, default=None,
                    help="Truncate edges for smoke test")
    ap.add_argument("--staging-dir", type=Path, default=None,
                    help="Where to put _staging_edges.bin (default: parent of --raw-dir)")
    ap.add_argument("--keep-staging", action="store_true",
                    help="Don't delete staging file after convert")
    args = ap.parse_args()

    data_npz = args.raw_dir / "data.npz"
    label_npz = args.raw_dir / "node-label.npz"
    if not data_npz.exists():
        sys.exit(f"ERROR: {data_npz} not found")

    args.output.mkdir(parents=True, exist_ok=True)
    safe_name = args.dataset_name.replace("-", "_")
    out_features = args.output / f"{safe_name}_features.npy"
    out_labels = args.output / f"{safe_name}_labels.npy"
    out_gql = args.output / f"{safe_name}.gql"

    staging_dir = args.staging_dir or args.raw_dir.parent
    staging = staging_dir / "_staging_edges.bin"

    edge_label = args.edge_label or ("CITES" if args.directed else "CONNECTS")

    # Metadata
    feat_shape, feat_dtype = read_npy_header(data_npz, "node_feat.npy")
    edge_shape, edge_dtype = read_npy_header(data_npz, "edge_index.npy")
    num_nodes = int(feat_shape[0])
    feature_dim = int(feat_shape[1]) if len(feat_shape) > 1 else 0
    num_edges = int(edge_shape[1])

    print("=" * 60)
    print(f"Dataset:     {args.dataset_name}")
    print(f"Nodes:       {num_nodes:,}")
    print(f"Edges:       {num_edges:,}")
    print(f"Feature dim: {feature_dim}")
    print(f"Directed:    {args.directed}")
    print(f"Edge label:  {edge_label}")
    print(f"Staging at:  {staging}")
    print("=" * 60)
    print()

    t_total = time.time()

    extract_features(data_npz, out_features, feat_shape, feat_dtype)
    stage_edges(data_npz, staging, edge_shape, edge_dtype)

    labels = load_labels(label_npz)
    if labels is not None:
        np.save(out_labels, labels)
        print(f"       Saved -> {out_labels}")

    splits = load_splits(args.split_dir, num_nodes)

    edges_written = write_gql(
        out_gql, num_nodes, feature_dim, labels, splits,
        staging, num_edges, args.directed, edge_label, args.max_edges,
    )

    print(f"[6/6] Cleanup")
    if args.keep_staging:
        print(f"       Keeping {staging} (--keep-staging)")
    else:
        try:
            os.remove(staging)
            print(f"       Deleted {staging}")
        except OSError as e:
            print(f"       WARN: could not delete {staging}: {e}")

    total_sec = time.time() - t_total
    print()
    print("=" * 60)
    print(f"DONE in {total_sec / 60:.1f} min")
    print(f"  Features:     {out_features}")
    print(f"  Labels:       {out_labels if labels is not None else '(none)'}")
    print(f"  GQL:          {out_gql} ({edges_written:,} edges written)")
    print("=" * 60)
    print()
    print("Next step:")
    # --with-tensors registers the feature matrix. An import without it succeeds
    # and graph_project then fails, with a full reimport as the only recovery.
    print(f"  ./build/Release/bin/mdb import {out_gql} ./data/dbs/gql/{args.dataset_name} \\")
    print(f"      --with-tensors {out_features}")


if __name__ == "__main__":
    main()
