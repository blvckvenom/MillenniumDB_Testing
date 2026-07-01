#pragma once

#include <cstdint>
#include <filesystem>
#include <memory>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include <torch/torch.h>

#include "gnn/models/graphsage_model.h"
#include "gnn/projection/topology_accessor.h"
#include "gnn/sampling/graph_sample.h"
#include "gnn/sampling/sample_catalog.h"
#include "gnn/sampling/sample_storage.h"
#include "gnn/storage/row_mapping.h"
#include "gnn/training/batch_assembler.h"

namespace GQL {
class ProjectionStorage;
}

namespace mdb::gnn {

/**
 * @brief Collects seed embeddings from all training batches, infers
 *        embeddings for non-seed nodes, and (in future phases) writes
 *        them to the projection's tensor store.
 *
 * ## Overview
 *
 * After training, every node that appeared as a *seed* (layer 0) in any
 * batch already has a hidden representation that the model computed during
 * the forward pass.  Rather than re-running inference on the full graph,
 * we iterate over the pre-computed batches and collect those seed
 * embeddings, mapping each one back to its row_index in the RowMapping.
 *
 * Nodes that were never seeds still need embeddings.  Phase B performs
 * on-the-fly k-hop sampling via TopologyAccessor, builds a GraphSample,
 * assembles a MiniBatch, and runs a model forward pass to produce those
 * embeddings.
 *
 * ## Seed ObjectId recovery
 *
 * MiniBatch does not carry ObjectIds (it contains only tensors).  The
 * BatchAssembler's internal SampleStorage reference is private and not
 * exposed.  Therefore EmbeddingWriter receives its own SampleStorage
 * reference and reads GraphSamples directly to recover seed ObjectIds
 * from `nodes_per_layer[0]`.
 *
 * ## Phases
 *
 *   - Phase A: collect_seed_embeddings() iterates ALL batches,
 *     runs model.get_embeddings(), and maps results to (row_index, tensor).
 *   - Phase B: infer_non_seed_embeddings() for nodes never seen
 *     as seeds, using on-the-fly k-hop sampling from the projection topology.
 *   - Phase C: write_to_projection() persists embeddings as a
 *     new tensor property in the GQL projection.
 *
 * ## Thread safety
 *
 * Not thread-safe.  Intended for single-threaded post-training use.
 */
class EmbeddingWriter {
public:
    // =========================================================================
    // Configuration
    // =========================================================================

    /**
     * @brief Which nodes receive an embedding property.
     *
     * ALL is the historical behaviour and stays the default: seeds come from
     * the pre-computed batches (Phase A) and every remaining node in the
     * RowMapping is inferred on the fly (Phase B).  That is affordable while
     * the projection fits in RAM, and unaffordable past it -- papers100M has
     * ~109.5 M non-seed nodes, whose inference is both a multi-hundred-GB
     * resident map and weeks of random feature-row gathers.  write_all()
     * refuses such a run up front rather than discovering it mid-flight.
     *
     * SEEDS writes only the nodes that appeared as a seed in some batch and
     * skips Phase B entirely.  Note the contract is the SEED set, not the
     * labelled set: the two coincide only when the sample was built with
     * `usePredefinedSplits`, which drops unlabelled rows in SeedSelector.
     * Under the default ratio-based splitting every row is a seed, so SEEDS
     * and ALL cover the same nodes and the mode saves nothing.
     */
    enum class Coverage {
        ALL,   ///< Seeds plus on-the-fly inference for every other node (default)
        SEEDS  ///< Only nodes that were seeds in some batch (no Phase B)
    };

    struct Config {
        std::string              property_name;     ///< Target property for embeddings
        uint64_t                 batch_size = 2048; ///< Inference batch size (Phase B)
        std::vector<uint64_t>    fanouts;           ///< Sampling fanouts   (Phase B)
        EdgeOrientation          orientation = EdgeOrientation::UNDIRECTED;
        std::filesystem::path    feature_matrix_path; ///< Path to .fmat for inference feature loading
        Coverage                 coverage = Coverage::ALL; ///< Node set to write
    };

    // =========================================================================
    // Result
    // =========================================================================

    struct Result {
        uint64_t nodes_written   = 0;   ///< Nodes whose embeddings were collected/written
        uint64_t nodes_inferred  = 0;   ///< Non-seed nodes inferred in Phase B
        double   inference_ms    = 0.0; ///< Wall-clock time for inference (Phase B)
        double   write_ms        = 0.0; ///< Wall-clock time for writes (Phase C)
    };

    // =========================================================================
    // Construction
    // =========================================================================

    /**
     * @brief Construct an EmbeddingWriter.
     *
     * All reference parameters must outlive this object (non-owning).
     *
     * @param model              Trained GraphSAGE model (set to eval internally)
     * @param assembler          BatchAssembler for producing MiniBatches
     * @param sample_storage     SampleStorage for reading GraphSamples (seed ObjectIds)
     * @param row_mapping        ObjectId <-> row_index mapping
     * @param catalog            Batch counts per split (train/val/test)
     * @param projection_storage ProjectionStorage for topology access (Phase B inference)
     * @param config             Writer configuration
     */
    EmbeddingWriter(
        GraphSAGEModel&            model,
        BatchAssembler&            assembler,
        SampleStorage&             sample_storage,
        const RowMapping&          row_mapping,
        const SampleCatalog&       catalog,
        GQL::ProjectionStorage&    projection_storage,
        Config                     config
    );

    // =========================================================================
    // Public API
    // =========================================================================

    /**
     * @brief Run the full embedding write pipeline.
     *
     * Phase A: Collect seed embeddings from pre-computed batches.
     * Phase B: Infer embeddings for non-seed nodes via on-the-fly k-hop sampling.
     * Phase C: Write to projection tensor store.
     *
     * @return Summary with counts and timing.
     */
    Result write_all();

    // =========================================================================
    // Property key allocation (Phase C)
    // =========================================================================

    /// First synthetic key id usable for embedding properties.  Matches the
    /// base NativeProjectionBuilder uses for synthetic keys
    /// (RENAME_KEY_SYNTHETIC_START = 2000), so dynamically allocated ids
    /// never dip below the build-time synthetic range.
    static constexpr uint64_t EMBEDDING_KEY_SYNTHETIC_BASE = 2000;

    /**
     * @brief Compute the next free synthetic key id for a new node property.
     *
     * Returns max(EMBEDDING_KEY_SYNTHETIC_BASE, highest id in use + 1).
     * Both key namespaces are scanned because NativeProjectionBuilder
     * allocates node and edge synthetic ids from a single shared counter.
     *
     * @param node_keys Projection's node key name -> id map
     * @param edge_keys Projection's edge key name -> id map
     */
    static uint64_t next_available_key_id(
        const std::unordered_map<std::string, uint64_t>& node_keys,
        const std::unordered_map<std::string, uint64_t>& edge_keys);

    /**
     * @brief Resolve (or allocate and register) the key id for a node property.
     *
     * Re-syncs key mappings from the on-disk projection catalog first —
     * ProjectionStorage::open() restores statistics but not the key maps, so
     * without the re-sync a property persisted by an earlier session would be
     * re-allocated under a colliding id and the next save_catalog() would
     * drop every previously registered key.  An already-registered name
     * returns its existing id; otherwise a fresh id strictly above every id
     * in use is registered and returned.
     *
     * @throws std::runtime_error if the chosen id is already bound to a
     *         different property name, or the registration does not stick.
     */
    static uint64_t resolve_property_key_id(
        GQL::ProjectionStorage& storage,
        const std::string&      property_name);

    // =========================================================================
    // Memory estimation (Coverage::ALL affordability gate)
    // =========================================================================

    /// Heap bytes charged to each embedding held in the row_index -> tensor
    /// map: the unordered_map node (key, value, next pointer, and its share
    /// of the bucket array) plus the TensorImpl/StorageImpl pair backing one
    /// [hidden_dim] tensor.  Rounded up so the gate errs towards refusing.
    static constexpr uint64_t EMBEDDING_ENTRY_OVERHEAD_BYTES = 320;

    /// Heap bytes charged to each source node in the Phase B adjacency cache:
    /// the unordered_map node plus the std::vector<AdjEntry> header it holds.
    static constexpr uint64_t ADJACENCY_NODE_OVERHEAD_BYTES = 80;

    /**
     * @brief Peak resident bytes for a Coverage::SEEDS write.
     *
     * One [hidden_dim] float32 embedding per seed, held in the dedup map.
     */
    static uint64_t estimate_seed_write_bytes(uint64_t num_seeds,
                                              uint64_t hidden_dim);

    /**
     * @brief Peak resident bytes for a Coverage::ALL write.
     *
     * The seed cost extended to every node, plus the two structures only full
     * coverage builds: the worst-case `missing` row-index vector and the
     * Phase B adjacency cache.
     *
     * Exposed (like next_available_key_id) so the gate's arithmetic can be
     * exercised at papers100M shape without materializing such a fixture.
     *
     * @param num_nodes    Nodes in the RowMapping.
     * @param hidden_dim   Embedding width.
     * @param adj_entries  Directed entries the orientation makes the adjacency
     *                     cache scan (both directions when UNDIRECTED).
     */
    static uint64_t estimate_all_write_bytes(uint64_t num_nodes,
                                             uint64_t hidden_dim,
                                             uint64_t adj_entries);

private:
    // =========================================================================
    // Coverage::ALL affordability gate
    // =========================================================================

    /**
     * @brief Refuse a full-coverage write this host cannot hold.
     *
     * No-op unless Phase B would actually run (Coverage::ALL with a non-empty
     * fanout list) and /proc/meminfo is readable -- we never refuse on a
     * number we could not read.  Called at the top of write_all() so an
     * unaffordable run fails in a second instead of after the full batch scan.
     *
     * @throws std::runtime_error naming writeCoverage:'seeds' as the remedy.
     */
    void check_all_coverage_fits_() const;

    /**
     * @brief Lazily construct (once) and return the TopologyAccessor.
     *
     * Kept out of the constructor because the TopologyAccessor constructor
     * opens the CSR sidecar readers and runs their SHA-256 staleness pass --
     * minutes of I/O on a papers100M-scale projection -- while Phase B reaches
     * the topology through adj_cache_ instead, so no current path needs it.
     */
    TopologyAccessor& get_topology_();

    // =========================================================================
    // Phase A: Seed embedding collection
    // =========================================================================

    /**
     * @brief Iterate all batches, run model.get_embeddings(), and map seed
     *        embeddings back to RowMapping indices.
     *
     * For each batch_id in [0, catalog_.total_batches):
     *   1. Assemble the MiniBatch via BatchAssembler.
     *   2. Read the GraphSample via SampleStorage to recover seed ObjectIds.
     *   3. Run model.get_embeddings() on the MiniBatch.
     *   4. For each seed, look up its row_index via RowMapping::find().
     *   5. Store (row_index, embedding_tensor) in the result vector.
     *
     * Seeds whose ObjectId is not found in the RowMapping are silently
     * skipped (defensive: should never happen with consistent data).
     *
     * @return Vector of (row_index, embedding[1, hidden_dim]) pairs.
     */
    std::vector<std::pair<uint64_t, torch::Tensor>> collect_seed_embeddings();

    // =========================================================================
    // Phase B: Non-seed inference
    // =========================================================================

    /**
     * @brief Infer embeddings for nodes that were never seeds in any batch.
     *
     * For each chunk of config_.batch_size missing nodes:
     *   1. Build a GraphSample by k-hop sampling from the projection topology.
     *   2. Assemble a MiniBatch via assembler_.assemble_from_sample().
     *   3. Forward pass through model_.get_embeddings().
     *   4. Map each seed embedding back to its row_index.
     *
     * Uses TopologyAccessor for neighbor lookups, respecting config_.orientation
     * and config_.fanouts for the sampling parameters.
     *
     * @param missing  Row indices of nodes without embeddings from Phase A.
     * @return Vector of (row_index, embedding[hidden_dim]) pairs.
     */
    std::vector<std::pair<uint64_t, torch::Tensor>>
        infer_non_seed_embeddings(const std::vector<uint64_t>& missing);

    /**
     * @brief Build a GraphSample for a batch of seed nodes via k-hop sampling.
     *
     * Replicates the BasicKHopSampler algorithm (per-node uniform sampling)
     * using TopologyAccessor::get_neighbors() for neighbor lookups.
     *
     * @param seeds       Seed ObjectIds (become nodes_per_layer[0])
     * @param batch_id    Batch identifier for the GraphSample
     * @return GraphSample ready for BatchAssembler::assemble_from_sample()
     */
    GraphSample build_graph_sample(
        const std::vector<ObjectId>& seeds,
        uint64_t batch_id
    );

    // =========================================================================
    // Adjacency cache (Phase B performance path)
    // =========================================================================

    /// Adjacency pair: neighbor node id + edge id (both uint64_t raw form).
    struct AdjEntry {
        uint64_t node_id;
        uint64_t edge_id;
    };

    /// Full-scan the projection's from_to_edge + to_from_edge B+Trees once
    /// and materialize an in-memory undirected adjacency map keyed by
    /// source node id. Called lazily on the first build_graph_sample()
    /// invocation inside Phase B.
    ///
    /// Rationale: per-node `get_range({seed_id, 0, 0}, {seed_id, MAX,
    /// MAX})` lookups via the live B+Tree incur O(page_tuples) work per
    /// call under the CSR_HYBRID v3 leaf format because search_index
    /// performs a linear decode scan of the leaf page. With K seeds and
    /// L layers expanding K * avg_fan nodes, Phase B pays
    /// O(K * fan^L * page_tuples) wall-clock time per chunk. Caching
    /// a dense adjacency up-front collapses every subsequent lookup to
    /// O(avg_degree) and amortizes the full-scan O(|E|) across chunks.
    ///
    /// Memory cost: ~16 bytes per edge. arxiv (1.07 M edges): ~17 MB;
    /// products (61.9 M edges): ~1 GB — affordable wherever the projection
    /// itself already fits in RAM.
    void build_adjacency_cache_();

    /// Look up (neighbor_node_id, edge_id) pairs for a given node id
    /// from the in-memory undirected cache. Returns the (possibly empty)
    /// span of entries; an absent node id returns an empty range.
    const std::vector<AdjEntry>& get_neighbors_cached_(
        uint64_t node_id
    ) const;

    // =========================================================================
    // Phase C: Write embeddings to projection as tensor properties
    // =========================================================================

    /**
     * @brief Persist all embeddings as tensor node properties in the projection.
     *
     * For each (row_index, embedding) pair:
     *   1. Look up the node ObjectId via RowMapping.
     *   2. Serialize the embedding to raw float bytes.
     *   3. Store in TensorManager via get_or_create_id().
     *   4. Build a tensor ObjectId (MASK_TENSOR_FLOAT_EXTERN | tensor_id).
     *   5. Insert (node_oid, key_oid, tensor_oid) into the B+Tree property
     *      indexes via BPlusTree::insert().
     *
     * The property key is registered in the projection catalog under
     * config_.property_name (e.g., "embedding"). If the projection was not
     * built with node property indexes, they are created as empty B+Trees
     * first.
     *
     * @param emb_map  Map of row_index -> embedding tensor.
     * @return Number of embeddings successfully written.
     */
    uint64_t write_to_projection(
        const std::unordered_map<uint64_t, torch::Tensor>& emb_map);

    // =========================================================================
    // Members (non-owning references)
    // =========================================================================

    GraphSAGEModel&            model_;
    BatchAssembler&            assembler_;
    SampleStorage&             sample_storage_;
    const RowMapping&          row_mapping_;
    const SampleCatalog&       catalog_;
    Config                     config_;

    /// ProjectionStorage for Phase C writes (non-owning reference).
    GQL::ProjectionStorage&    projection_storage_;

    /// TopologyAccessor for on-the-fly k-hop sampling (Phase B).
    /// Owned by this object but constructed on demand by get_topology_():
    /// its constructor opens the CSR sidecar readers and runs a SHA-256
    /// staleness pass, which no path through this class currently needs.
    /// Null until first use.
    std::unique_ptr<TopologyAccessor> topology_;

    /// RNG for uniform neighbor sampling (Phase B).
    std::mt19937_64            rng_;

    /// Undirected adjacency cache: node_id -> list of (neighbor, edge_id).
    /// Populated on first use by `build_adjacency_cache_()`.
    std::unordered_map<uint64_t, std::vector<AdjEntry>> adj_cache_;

    /// Sentinel empty vector returned by get_neighbors_cached_() when a
    /// node id is not present (avoids reallocating a fresh empty vector
    /// per miss). Mutable because get_neighbors_cached_() is const.
    mutable std::vector<AdjEntry> adj_empty_sentinel_;

    /// True once build_adjacency_cache_() has finished populating
    /// adj_cache_. Defaults to false; set to true exactly once.
    bool adj_cache_built_ = false;
};

} // namespace mdb::gnn
