# Creating and running a GNN database

Training a graph neural network inside MillenniumDB means importing a graph
together with its node feature matrix, projecting the part of it you want to
train on, sampling neighbourhoods offline, building a feature store, and
training. This page covers all of it. The build is described in
[[Setup GNN]].

## Benchmark datasets

These are the graphs the benchmarks and reproduction scripts in this repository
use. Sizes are for the download; the converted form is larger.

| Dataset | Nodes | Edges | Features | Classes | Download | Source |
|---|---:|---:|---:|---:|---:|---|
| `cora` | 2,708 | 5,429 | 1,433 | 7 | 5 MB | LINQS |
| `ogbn-arxiv` | 169,343 | 1,166,243 | 128 | 40 | 0.3 GB | OGB |
| `ogbn-products` | 2,449,029 | 61,859,140 | 100 | 47 | 1.5 GB | OGB |
| `ogbn-papers100M` | 111,059,956 | 1,615,685,872 | 128 | 172 | 50 GB | OGB |

`scripts/download-dataset.sh` can also fetch `ogbn-mag` and `ogbn-proteins`, but
neither converter handles them and nothing here consumes them: `mag` is
heterogeneous, so it has four node types and no single edge index, and `proteins`
carries its features on edges rather than nodes.

## The short version

```bash
scripts/gnn_benchmark.sh cora          # a few seconds
scripts/gnn_benchmark.sh arxiv         # a few minutes
scripts/gnn_benchmark.sh products      # tens of minutes
scripts/gnn_benchmark.sh papers100M     # hours, and 120 GB of disk
```

One command per dataset, from nothing to a trained model: it downloads and
converts the graph, imports it with its features, starts a server, runs the four
procedures, and prints a stage-by-stage timing table with a verdict.

```
[bench] 5/6  pipeline
  + project: 2708 nodes, 5429 edges, dim 1433, 7 classes
  + sample: 27 batches (train 3), 2493 unique nodes
  + store: L1=0 L2=2493 L3=0 L4=215
  + CUDA visible to MillenniumDB
  + train: test 0.7110, best val 0.7820, 30 epochs in 0.8432 s, L1 hit 0.0000
```

It refuses to start without a usable GPU. Pass `--cpu` to say you meant it.
`--refresh` re-converts a dataset whose files predate the current converter, and
`--keep` leaves the database directory behind for inspection.

The accuracy it reports is a smoke gate, not a result, and the floors sit well
below what a healthy run produces. The OGB datasets carry their published
train/validation/test division. Cora does not: its split is a positional slice of
the source file, 140/500/1000 in the order `cora.content` happens to list its
papers, which is *not* the Planetoid split of Yang et al. (2016) and gives an
uneven 14/11/41/28/21/8/17 across the seven classes. Cora's number says the
pipeline produced a model; it is not comparable with published Cora figures.

## Step by step

### 1. Download and convert

```bash
python scripts/download_gnn_datasets.py --list
python scripts/download_gnn_datasets.py --dataset cora
python scripts/download_gnn_datasets.py --dataset all      # the three small ones
```

Each dataset produces three files under `data/example/gql/<dataset>/`:

```
cora.gql             # one line per node and per edge, with label and split
cora_features.npy    # the feature matrix, N x D, float32
cora_labels.npy      # the labels, shape (N,) — int64 where OGB gives integer
                     # labels, float32 with NaN for unlabeled on papers100M
```

Node lines carry the published division as a property:

```
0 :Paper paper_id:"31336" label:2 class_name:"Neural_Networks" split:"train"
0->1 :CITES
```

`ogbn-papers100M` is excluded from `--dataset all` deliberately and has to be
named. It also takes a different route internally: the ordinary converter loads a
dataset through `ogb.nodeproppred.NodePropPredDataset`, which allocates every
array at once, and papers100M's feature matrix alone is 53 GiB. It is instead
downloaded raw and converted by `stream_convert_ogb.py`, which reads the archive
member by member so peak memory stays near 500 MB regardless of graph size.

### 2. Import with the feature matrix

```bash
build/Release/bin/mdb import data/example/gql/cora/cora.gql data/dbs/gql/cora \
    --with-tensors data/example/gql/cora/cora_features.npy
```

`--with-tensors` is not optional for GNN work. It registers the feature matrix
and writes the row mapping that ties matrix rows back to node identifiers.
Nothing later can add it: `graph_project` reads that mapping and cannot create
it, so an import without the flag has to be redone.

### 3. Start a server

```bash
build/Release/bin/mdb server data/dbs/gql/cora -p 8080
```

Queries are sent as `POST` bodies to `/gql`:

```bash
curl -X POST http://localhost:8080/gql -H "Content-Type: text/plain" \
     --data-binary "MATCH (n:Paper) RETURN count(n)"
```

`curl` exits 0 even when the server answers with an error, so a script that
checks only the exit status will not notice a failed query. Inspect the body.

### 4. Project

```gql
CALL graph_project('cora_proj', 'Paper', 'CITES', {
    orientation: 'UNDIRECTED',
    includeFeatures: 'node_features',
    labelProperty: 'label',
    splitProperty: 'split',
    nodeProperties: ['paper_id']
}) YIELD graphName, nodeCount, relationshipCount, featureDim, numClasses RETURN *
```

A projection is a disk-resident copy of the subgraph, laid out for traversal
rather than for queries. `includeFeatures` carries the feature matrix across,
`labelProperty` names the classification target, and `splitProperty` adopts the
published division instead of generating one.

Only properties named in `nodeProperties` cross into the projection. Without it
the projection carries topology, features, labels and splits but no attributes,
and a later query that filters on one of them returns nothing.

Two options matter at scale. `indexSet: 'GNN_MINIMAL'` builds 5 of the 10
topology indexes, which is everything sampling and training need, at roughly 61%
less disk and 57% less wall-clock. `orientation` should stay `NATURAL` on
papers100M, where the symmetrisation happens during sampling instead.

### 5. Sample

```gql
CALL gnn_offline_sample('cora_proj', 'cora_s', [5, 10], {
    batchSize: 64,
    randomSeed: 42,
    orientation: 'UNDIRECTED',
    numWorkers: 0
}) YIELD totalBatches, trainBatches, validationBatches, testBatches, uniqueNodes RETURN *
```

Neighbourhoods are drawn once and written to disk, so every epoch reads the same
batches instead of re-sampling. The fanout list is read in the order DGL and
GraphBolt use: **the last element is the hop next to the seeds**. `[5, 10]` means
10 neighbours per seed, then 5 per node in that first hop.

`numWorkers` above 0 samples batches in parallel. Each batch is re-seeded as
`randomSeed XOR batch_id`, so the output does not depend on which thread claimed
which batch.

### 6. Build the feature store

```gql
CALL gnn_build_feature_store('cora_s', 'node_features', {
    gpu_budget_mb: 2048,
    cpu_budget_mb: 512
}) YIELD l1Nodes, l2Nodes, l3Nodes, l4Nodes, gpuAvailable, buildTimeMs RETURN *
```

Features are placed across four levels by how often the sample touches them: L1
in GPU memory, L2 in pinned host memory, L3 in a memory-mapped file reordered so
that nodes read together sit together, L4 in per-batch packed chunks. The yields
report how many nodes landed in each.

`gpuAvailable` is the one runtime signal that says whether this build of
MillenniumDB can see a GPU at all. `nvidia-smi` answering only proves the device
exists.

Add `bakeBlocks: true` on large graphs. It precomputes each batch's computation
graph offline; without it the index rebuild is repeated every epoch and accounts
for about 64% of batch assembly.

### 7. Train

```gql
CALL gnn_train('cora_s', 'node_features', {
    model: 'graphsage',
    hiddenDim: 256,
    epochs: 30,
    lr: 0.01,
    dropout: 0.5,
    randomSeed: 42,
    patience: 999,
    tolerance: 0
}) YIELD testAccuracy, bestValAccuracy, ranEpochs, trainSeconds, l1HitRatio RETURN *
```

There are two independent early exits. `patience` watches validation accuracy and
`tolerance` watches the change in loss between consecutive epochs. Raising only
`patience` leaves the second one armed, and on a small training split the loss
flattens long before the model is done, so a run asked for 30 epochs can stop at
9. Set both to run the full schedule.

`l1HitRatio` reports how often a feature was already in GPU memory. A ratio of
zero on a graph large enough to need caching means the GPU cache is not being
used.

## Writing embeddings back

```gql
CALL gnn_train('cora_s', 'node_features', {
    epochs: 30,
    writeProperty: 'embedding'
}) YIELD testAccuracy RETURN *
```

With `writeProperty` set, every node's embedding is stored as a queryable tensor
property on the projection, including nodes that were never seeds — those are
inferred on the fly. They can then be read and compared:

```gql
USE cora_proj MATCH (n:Paper) RETURN n.embedding LIMIT 5

USE cora_proj MATCH (a:Paper), (b:Paper)
WHERE a.paper_id = "31336"
RETURN b.paper_id, cosineDistance(a.embedding, b.embedding) AS d
ORDER BY d LIMIT 10
```

## Checkpoints and inference

Training writes `best_model` and `final_model` under the output directory. See
[[GNN Model Checkpoints]] for resuming with optimiser state preserved, and for
`gnn_predict`, `gnn_list_checkpoints`, `gnn_checkpoint_exists` and
`gnn_checkpoint_delete`.

## Reproducing the published papers100M numbers

The exact procedure, with pinned parameters and expected outputs for all four
stages, is in `docs/research/2026-07-01-papers100m-repro/REPRODUCE.md`. It is the
canonical source; the commands on this page are the general shape, not the
published configuration.
