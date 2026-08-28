# GNN Model Checkpoints

MillenniumDB GNN training produces disk-based checkpoints that enable resumption and inference without retraining. This page covers the API and operational concerns.

## Layout

```
<db>/projections/<projection>/gnn_output/<runName>/checkpoints/
  best_model.pt        (torch archive: model weights + Adam optimizer state)
  best_model.ckptmeta  (custom binary: TrainingState + SHA-256 of gnn_meta.bin)
  final_model.pt
  final_model.ckptmeta
```

## Training creates two checkpoints

`best_model` is overwritten on each strict improvement in validation accuracy. `final_model` is written once at the end of the loop. Both are enabled by default.

```gql
USE cora
CALL gnn_train('cora_sample', 'node_features', {epochs: 50, outputDir: 'exp1'})
YIELD bestCheckpointPath, finalCheckpointPath, resumedFromEpoch
```

Disable either with `saveOnBestVal: false` or `saveFinal: false`.

## Resume training

```gql
USE cora
CALL gnn_train('cora_sample', 'node_features', {
  model: 'graphsage',
  hiddenDim: 128,
  epochs: 50,
  outputDir: 'exp1',
  resumeFrom: 'best_model'   -- relative; resolved against exp1/checkpoints/
})
YIELD resumedFromEpoch, ranEpochs, bestValAccuracy
```

The resumed training preserves model weights, Adam optimizer momenta (`m` and `v`), the patience counter, best-val-accuracy tracker, and the full loss history. Absolute paths are also accepted for cross-run resumes.

**Architecture must match.** The checkpoint records `input_dim`, `hidden_dim`, `num_classes`, `num_layers`, `dropout`, `normalize`. Attempting to resume with a different `hiddenDim` throws a clear error at load time.

## Inference without retraining

```gql
USE cora
CALL gnn_predict('cora_sample', 'node_features', 'best_model', {
  writeProperty: 'prediction_embedding'  -- optional: persist to projection
})
YIELD checkpointEpoch, numSeedNodes, embeddingDim, nodesWritten
```

Predict runs in `eval` mode (no dropout) and produces bit-identical embeddings to the training export when the checkpoint came from the same deterministic run (verified by the test suite).

## Discovery and lifecycle

```gql
-- List all checkpoints under a projection (across all runs)
CALL gnn_list_checkpoints('cora')
YIELD basename, outputDir, saveKind, epoch, bestValAccuracy, creationTime;

-- Scope to one run
CALL gnn_list_checkpoints('cora', 'exp1') YIELD basename;

-- Filter by basename
CALL gnn_list_checkpoints('cora', 'exp1', 'best_model') YIELD basename;

-- Existence check
CALL gnn_checkpoint_exists('cora', 'exp1', 'best_model') YIELD exists;

-- Idempotent delete (both .pt and .ckptmeta)
CALL gnn_checkpoint_delete('cora', 'exp1', 'final_model')
YIELD ptDeleted, metaDeleted;
```

## Compatibility validation

At load time, three fields are compared between the checkpoint and the current projection:

1. **Architecture dims** (`input_dim`, `hidden_dim`, `num_classes`, `num_layers`) must match exactly.
2. **`projection_name`** stored in the checkpoint must match the current projection (prevents cross-project resumes).
3. **SHA-256 hash of the current `gnn_meta.bin`** must match what was stored when the checkpoint was saved (detects data drift: new nodes, new labels, re-sampled batches).

If any mismatch: a clear error explaining which field differs and how to recover.

## Atomicity

Each save writes `<basename>.pt.tmp` → fsync → `<basename>.ckptmeta.tmp` → rename `.pt` → rename `.ckptmeta` → fsync directory. A crash at any step leaves the previous checkpoint intact. Orphaned `.tmp` files from crashed saves are cleaned on the next save or delete.

The `.ckptmeta` file starts with the 8-byte magic `GNNCKPT\0` followed by a version number, `TrainingState` fields, and a SHA-256 of `gnn_meta.bin`.

## Paths in yields

`bestCheckpointPath` and `finalCheckpointPath` (from `gnn_train`) and `checkpointPath` (from `gnn_predict`) are **absolute paths with no file extension** — point to the basename that prefixes both `<x>.pt` and `<x>.ckptmeta`.

## Known behaviors and limits

- `gnn_list_checkpoints` ordering: descending by `creationTime`. Ties within the same second have FS-dependent order — use absolute paths for determinism.
- `gnn_checkpoint_exists` returns `true` only when BOTH `.pt` AND `.ckptmeta` are present.
- `gnn_checkpoint_delete` is idempotent: missing files are not an error.
- Only `SaveKind::Full` is produced by `gnn_train` today. `save_weights` (weights-only, no optimizer state) is available at the C++ API level for future procedures.
- Max number of checkpoints per projection is unbounded (no top-K cleanup); manage manually via `gnn_checkpoint_delete`.
