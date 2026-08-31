# GNN Dataset Download & Conversion Scripts

Scripts for downloading GNN benchmark datasets and converting them to MillenniumDB GQL format.

## Unified Script (Recommended)

The unified script `scripts/download_gnn_datasets.py` downloads and converts all supported datasets:

```bash
# Download a specific dataset
python scripts/download_gnn_datasets.py --dataset cora
python scripts/download_gnn_datasets.py --dataset ogbn-arxiv
python scripts/download_gnn_datasets.py --dataset ogbn-products

# Download all datasets
python scripts/download_gnn_datasets.py --dataset all

# Force re-download
python scripts/download_gnn_datasets.py --dataset cora --force

# List available datasets
python scripts/download_gnn_datasets.py --list
```

### Requirements

```bash
# Core (all datasets)
pip install numpy requests tqdm

# OGB datasets only
pip install ogb torch
```

### Supported Datasets

| Dataset | Nodes | Edges | Features | Source |
|---------|------:|------:|---------:|--------|
| **cora** | 2,708 | 5,429 | 1,433 | [LINQS](https://linqs-data.soe.ucsc.edu/public/lbc/cora.tgz) |
| **ogbn-arxiv** | 169,343 | 1,166,243 | 128 | [OGB](https://ogb.stanford.edu/) |
| **ogbn-products** | 2,449,029 | 61,859,140 | 100 | [OGB](https://ogb.stanford.edu/) |

### Output Structure

Each dataset produces three files:

```
data/example/gql/<dataset>/
├── <name>.gql           # Graph topology + node metadata (split, label)
├── <name>_features.npy  # Node feature matrix (N × D), float32
└── <name>_labels.npy    # Node labels (N,), int64
```

### GQL Format

Nodes include split assignments (`train`, `val`/`valid`, `test`, `unlabeled`).
Edge syntax depends on graph type: `->` for directed, `~` for undirected:

```
// Cora nodes
0 :Paper paper_id:"31336" label:2 class_name:"Neural_Networks" split:"train"

// Cora edges (directed citations)
0->1 :CITES

// OGB nodes
0 :Node label:4 split:"train" feat_dim:128

// OGB directed edges (e.g. ogbn-arxiv citations)
0->1 :CONNECTS

// OGB undirected edges (e.g. ogbn-products co-purchasing)
0~1 :CONNECTS
```

Features are stored in separate `.npy` files (not inline in GQL) for efficiency and compatibility with `npy_loader.h`.

### NPY File Format

- **Features**: 2D float32 array, C-order (row-major)
- **Labels**: 1D int64 array

These are compatible with MillenniumDB's `NpyLoader` (`src/import/npy_loader.h`).

### Split Assignments

- **Cora**: Positional 140/500/1000 slice over cora.content order (NOT the Planetoid split) — 140 train, 500 val, 1000 test
- **OGB**: Official OGB splits provided by `dataset.get_idx_split()`

## Legacy Scripts

The individual scripts are still available for reference:

- `convert_cora_to_mdb.py` — Converts pre-downloaded Cora data (features inline in GQL)
- `download_ogb.py` — Downloads and converts individual OGB datasets

## Importing into MillenniumDB

After downloading:

```bash
# Import
./build/Release/bin/mdb import data/example/gql/cora/cora.gql ./data/dbs/gql/cora

# Start server
./build/Release/bin/mdb server ./data/dbs/gql/cora &

# Query
curl -X POST http://localhost:1234 \
  -H 'Content-Type: application/gql' \
  -d 'MATCH (p:Paper) WHERE p.split = "train" RETURN COUNT(p)'
```
