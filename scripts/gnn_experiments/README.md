# Native GQL/GNN Feature Experiments

These scripts reproduce the native feature-engineering paths for Cora using the
GNN pipeline in `build/GNN/bin/mdb`. Run them from the repository root.

Each script imports a fresh database under `/tmp`, starts an MDB server on a
fixed port, projects Cora, samples, materializes batches, builds the feature
store, and trains GraphSAGE. Feature-engineering scripts also create or append
native GNN features before projection.

## Scripts

- `run_b5_structural_cora.sh`
  - Imports Cora without tensors.
  - Creates `cora_structural_b5_zscore` with `gnn_create_structural_features`.
  - Expected outcome: low/moderate accuracy, validating native structural
    features.

- `run_cora_original.sh`
  - Imports Cora with `cora_features.npy`.
  - Uses the original `node_features` tensor without GQL-generated features.
  - Expected outcome: original Cora baseline accuracy.

- `run_b5_combined_cora.sh`
  - Imports Cora with `cora_features.npy`.
  - Creates `cora_combined_b5_zscore` by appending structural features to
    `node_features`.
  - Expected outcome: competitive with original Cora, validating
    `appendToFeature` for structural features.

- `run_b6_query_rich_cora.sh`
  - Imports Cora without tensors.
  - Creates `cora_query_rich_features_zscore` from a declarative GQL query.
  - Expected outcome: query-generated structural features only, validating the
    GQL-to-feature pipeline.

- `run_b6_query_combined_cora.sh`
  - Imports Cora with `cora_features.npy`.
  - Creates `cora_query_combined_features_zscore` by appending query-generated
    features to `node_features`.
  - Expected outcome: original Cora features plus query-generated features,
    validating declarative feature engineering with `appendToFeature`.

- `run_c1_cora_all.sh`
  - Runs the five C1 experiments in order and tees their output to
    `/tmp/c1_*.out`.
  - Does not write logs into the repository `outputs/` directory.

## C1 Comparison

C1 compares:

- no GQL-generated features with the original Cora `node_features`;
- fixed structural features from `gnn_create_structural_features`;
- query-generated structural features from `gnn_create_features_from_query`;
- fixed combined features from `node_features` plus native structural features;
- query combined features from `node_features` plus query-generated features.

Accuracy can vary slightly because training is stochastic, even with a fixed
seed. Cora import warnings on the initial comment lines are expected.

The scripts require only standard shell tools and `curl`; they do not require
`jq` or write into the repository `outputs/` directory.
