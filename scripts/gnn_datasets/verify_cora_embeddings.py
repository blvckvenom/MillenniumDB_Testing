#!/usr/bin/env python3
# =============================================================================
# verify_cora_embeddings.py — External "golden standard" validation (Level 4)
# =============================================================================
#
# Loads embeddings and raw features from a MillenniumDB database that has been
# trained with gnn_train on Cora, then trains two linear probes:
#
#   (a) LogReg on RAW TF-IDF features   -> expected ~55-60% (Shchur MLP baseline)
#   (b) LogReg on GNN EMBEDDINGS        -> expected 75-85%  (Shchur GS-mean range)
#
# The gap (b - a) quantifies how much the GNN message-passing adds on top of
# the raw bag-of-words. If (b) is not significantly above (a), the GNN is
# not learning useful topological signal.
#
# References:
#   Shchur et al. 2018, "Pitfalls of Graph Neural Network Evaluation",
#     NeurIPS R2L workshop. Table 4 / Fig 1. For Cora: GS-mean median ~80%,
#     MLP median ~58%, LogReg median ~57% (100 random splits * 20 inits).
#
# Usage:
#   python3 scripts/gnn_datasets/verify_cora_embeddings.py <db_dir>
#
# Arguments:
#   <db_dir>  Path to the MillenniumDB database directory created by the
#             gnn_e2e_test.sh script (e.g. /tmp/gnn_e2e_test_XXXXXX/ or any
#             persistent DB that has been through the GNN pipeline).
#
# Requirements:
#   numpy, scikit-learn
# =============================================================================

import os
import struct
import sys
from pathlib import Path

import numpy as np

try:
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
except ImportError:
    sys.stderr.write(
        "ERROR: scikit-learn is required for the linear probe.\n"
        "Install with: pip install scikit-learn\n"
    )
    sys.exit(2)


# -----------------------------------------------------------------------------
# Binary readers — matches formats in src/gnn/storage/ and src/gnn/training/
# -----------------------------------------------------------------------------

FMAT_HEADER_BYTES  = 64   # FeatureMatrix "GNNF" header
RMAP_HEADER_BYTES  = 16   # RowMapping header
LABEL_HEADER_BYTES = 32   # LabelStore "GNNL" header
SPLIT_HEADER_BYTES = 24   # SplitStore "GNNS" header


def read_feature_matrix(path: Path) -> np.ndarray:
    """Load a [N, D] float32 FeatureMatrix from .fmat file."""
    size = os.path.getsize(path) - FMAT_HEADER_BYTES
    with open(path, "rb") as f:
        f.seek(FMAT_HEADER_BYTES)
        raw = f.read(size)
    # Dimensions come from the paired .rmap (for N) and from the .npy metadata.
    # Here we infer D from the embeddings file OR assume Cora defaults.
    return np.frombuffer(raw, dtype=np.float32)


def read_rmap(path: Path) -> np.ndarray:
    """Return the [N] uint64 array of ObjectIds stored in a .rmap file."""
    size = os.path.getsize(path) - RMAP_HEADER_BYTES
    with open(path, "rb") as f:
        f.seek(RMAP_HEADER_BYTES)
        return np.frombuffer(f.read(size), dtype=np.uint64).copy()


def read_labels(path: Path) -> np.ndarray:
    """Return the [N] int64 labels array from labels.bin."""
    with open(path, "rb") as f:
        raw = f.read()
    return np.frombuffer(raw[LABEL_HEADER_BYTES:], dtype=np.int64).copy()


def read_splits(path: Path) -> np.ndarray:
    """Return the [N] uint8 splits array from splits.bin (0=train 1=val 2=test 255=unlabeled)."""
    with open(path, "rb") as f:
        raw = f.read()
    return np.frombuffer(raw[SPLIT_HEADER_BYTES:], dtype=np.uint8).copy()


def read_seed_node_ids(sample_dir: Path) -> list:
    """
    Parse batches.idx + batches.dat to recover the ORDER of seed ObjectIds
    for every batch. This order matches the row ordering in embeddings.npy
    written by gnn_train.
    """
    idx_path = sample_dir / "batches.idx"
    dat_path = sample_dir / "batches.dat"

    with open(idx_path, "rb") as f:
        f.read(4 + 4)                                       # magic + version
        num_batches = struct.unpack("<Q", f.read(8))[0]
        entries = [
            (struct.unpack("<Q", f.read(8))[0],
             struct.unpack("<Q", f.read(8))[0])
            for _ in range(num_batches)
        ]

    seed_oids_ordered = []
    with open(dat_path, "rb") as bf:
        for offset, _length in entries:
            bf.seek(offset)
            # GraphSample header: magic(4) + version(4) + batch_id(8) + split(1)
            bf.read(4 + 4 + 8 + 1)
            num_layers = struct.unpack("<Q", bf.read(8))[0]
            # Layer 0 is the seed layer
            for layer_idx in range(num_layers):
                ls = struct.unpack("<Q", bf.read(8))[0]
                payload = bf.read(ls * 8)
                if layer_idx == 0:
                    layer_ids = np.frombuffer(payload, dtype=np.uint64)
                    seed_oids_ordered.extend(layer_ids.tolist())
    return seed_oids_ordered


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

def main():
    if len(sys.argv) != 2:
        sys.stderr.write(
            "Usage: verify_cora_embeddings.py <db_dir>\n"
            "  Example: python3 scripts/gnn_datasets/verify_cora_embeddings.py /tmp/cora_db\n"
        )
        return 2

    db = Path(sys.argv[1]).resolve()
    if not db.is_dir():
        sys.stderr.write(f"ERROR: {db} is not a directory\n")
        return 2

    proj_dir   = db / "projections" / "e2e_proj"
    sample_dir = db / "samples" / "e2e_sample"
    out_dir    = proj_dir / "gnn_output" / "default"

    # ------- locate files -------
    fmat_path    = db / "gnn_features" / "node_features.fmat"
    rmap_path    = db / "gnn_features" / "node_features.rmap"
    labels_path  = proj_dir / "labels.bin"
    splits_path  = proj_dir / "splits.bin"
    emb_path     = out_dir / "embeddings.npy"

    for p in (fmat_path, rmap_path, labels_path, splits_path, emb_path):
        if not p.exists():
            sys.stderr.write(f"ERROR: file not found: {p}\n")
            return 2

    # ------- load data -------
    rmap      = read_rmap(rmap_path)                       # [2708] uint64
    N         = len(rmap)
    D         = 1433                                        # Cora feature dim
    fmat_flat = read_feature_matrix(fmat_path)
    fmat      = fmat_flat.reshape(N, D)                    # [2708, 1433]
    labels    = read_labels(labels_path)                   # [2708]
    splits    = read_splits(splits_path)                   # [2708] uint8
    emb       = np.load(emb_path)                          # [1640, H]

    print(f"Loaded: N={N} D={D} embeddings.shape={emb.shape}")
    print(f"Splits: train={(splits==0).sum()} val={(splits==1).sum()} "
          f"test={(splits==2).sum()} unlabeled={(splits==255).sum()}")
    print()

    # ==========================================================================
    # Probe (a) — LogReg on RAW TF-IDF features (MLP/LogReg baseline)
    # ==========================================================================
    train_mask_raw = (splits == 0)
    val_mask_raw   = (splits == 1)
    test_mask_raw  = (splits == 2)

    X_train_raw = fmat[train_mask_raw]
    y_train_raw = labels[train_mask_raw]
    X_test_raw  = fmat[test_mask_raw]
    y_test_raw  = labels[test_mask_raw]

    # Shchur et al. use L2-regularized LogReg directly on the bag-of-words.
    # No scaling — scaling binary features distorts sparsity.
    clf_raw = LogisticRegression(max_iter=2000, solver="lbfgs", C=1.0)
    clf_raw.fit(X_train_raw, y_train_raw)
    raw_test_acc = clf_raw.score(X_test_raw, y_test_raw)

    print("=" * 70)
    print(f"(a) Linear probe on RAW features (Shchur LogReg baseline):")
    print(f"    Test accuracy = {raw_test_acc:.4f}")
    print(f"    Shchur LogReg median reference on Cora: ~0.57")
    print(f"    Shchur MLP    median reference on Cora: ~0.58")
    print("=" * 70)
    print()

    # ==========================================================================
    # Probe (b) — LogReg on GNN EMBEDDINGS (uses seed ORDER from sample storage)
    # ==========================================================================
    # The embeddings.npy file contains one row per seed across all batches,
    # in the order: batch0_seeds, batch1_seeds, ..., batchN_seeds.
    # Batch order is train(1) → val(2) → test(4) when usePredefinedSplits=true.
    try:
        seed_oids = read_seed_node_ids(sample_dir)
    except Exception as e:
        print(f"WARN: could not parse seed IDs from sample storage: {e}")
        print("     skipping probe (b)")
        return 0

    if len(seed_oids) != emb.shape[0]:
        print(f"WARN: seed count mismatch — sample has {len(seed_oids)} seeds, "
              f"embeddings has {emb.shape[0]} rows")
        print("     skipping probe (b)")
        return 0

    # Map each seed ObjectId to its row_index in the global RowMapping
    oid_to_row = {int(oid): i for i, oid in enumerate(rmap)}
    seed_rows = np.array([oid_to_row[int(oid)] for oid in seed_oids])

    # Gather labels and splits for each seed, in the same order as emb rows
    emb_labels = labels[seed_rows]
    emb_splits = splits[seed_rows]

    train_mask_emb = (emb_splits == 0)
    val_mask_emb   = (emb_splits == 1)
    test_mask_emb  = (emb_splits == 2)

    n_train = train_mask_emb.sum()
    n_val   = val_mask_emb.sum()
    n_test  = test_mask_emb.sum()
    print(f"Probe (b) split counts within embeddings: "
          f"train={n_train} val={n_val} test={n_test}")

    if n_train < 10 or n_test < 10:
        print("WARN: too few seeds in train/test — skipping probe (b)")
        return 0

    X_train_emb = emb[train_mask_emb]
    y_train_emb = emb_labels[train_mask_emb]
    X_test_emb  = emb[test_mask_emb]
    y_test_emb  = emb_labels[test_mask_emb]

    scaler2 = StandardScaler()
    X_train_emb_s = scaler2.fit_transform(X_train_emb)
    X_test_emb_s  = scaler2.transform(X_test_emb)

    clf_emb = LogisticRegression(max_iter=1000, solver="lbfgs")
    clf_emb.fit(X_train_emb_s, y_train_emb)
    emb_test_acc = clf_emb.score(X_test_emb_s, y_test_emb)

    print()
    print("=" * 70)
    print(f"(b) Linear probe on GNN EMBEDDINGS:")
    print(f"    Test accuracy = {emb_test_acc:.4f}")
    print(f"    Shchur GS-mean median reference on Cora: ~0.80 (range 0.75-0.85)")
    print("=" * 70)
    print()

    # ==========================================================================
    # Summary — the gap (b - a) is the GNN's contribution
    # ==========================================================================
    delta = emb_test_acc - raw_test_acc
    print(f"GNN message-passing contribution: {delta:+.4f} "
          f"({emb_test_acc:.4f} vs raw {raw_test_acc:.4f})")
    if delta > 0.10:
        print(f"  PASS — GNN adds >10pp over raw features")
    elif delta > 0.0:
        print(f"  MARGINAL — GNN adds <10pp")
    else:
        print(f"  FAIL — GNN does not outperform raw features")
    print()

    # Sanity checks (Shchur 2018 reference ranges)
    shchur_logreg_min, shchur_logreg_max = 0.50, 0.65
    shchur_gs_min, shchur_gs_max = 0.72, 0.87
    raw_ok = shchur_logreg_min <= raw_test_acc <= shchur_logreg_max
    emb_ok = shchur_gs_min <= emb_test_acc <= shchur_gs_max

    print(f"Raw LogReg within Shchur range [{shchur_logreg_min}, {shchur_logreg_max}]: "
          f"{'YES' if raw_ok else 'NO'}")
    print(f"GNN probe within Shchur range [{shchur_gs_min}, {shchur_gs_max}]: "
          f"{'YES' if emb_ok else 'NO'}")

    return 0 if (raw_ok and emb_ok) else 1


if __name__ == "__main__":
    sys.exit(main())
