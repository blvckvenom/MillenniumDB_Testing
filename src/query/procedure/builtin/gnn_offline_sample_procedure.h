#pragma once

#include <string>
#include <vector>

#include "query/procedure/procedure.h"
#include "query/procedure/procedure_context.h"

namespace GQL {
namespace Procedures {

/**
 * @brief Creates pre-computed GNN mini-batches for offline training.
 *
 * This procedure implements the DiskGNN (SIGMOD 2025) architecture of
 * pre-computing ALL mini-batches before training. Benefits:
 * - Optimal I/O patterns (no random access during training)
 * - Reproducible training (deterministic sample order)
 * - Samples generated ONCE, reused across training epochs
 *
 * ## Syntax
 *
 * @code{.gql}
 *   CALL gnn.offline_sample(projectionName, sampleName, fanouts [, options])
 *   YIELD sampleName, totalBatches, trainBatches, validationBatches,
 *         testBatches, uniqueNodes, storagePath, computeMillis
 * @endcode
 *
 * ## Parameters
 *
 * | Name | Type | Required | Description |
 * |------|------|----------|-------------|
 * | projectionName | STRING | Yes | Source graph projection to sample from |
 * | sampleName | STRING | Yes | Name for the created sample set |
 * | fanouts | LIST<INT> | Yes | Neighbors per layer, e.g., [15, 10] |
 * | options | MAP | No | Configuration: batchSize, trainRatio, etc. |
 *
 * ## Options Map
 *
 * | Key | Type | Default | Description |
 * |-----|------|---------|-------------|
 * | batchSize | INT | 1024 | Number of seed nodes per batch |
 * | trainRatio | FLOAT | 0.7 | Fraction for training set |
 * | validationRatio | FLOAT | 0.15 | Fraction for validation set |
 * | testRatio | FLOAT | 0.15 | Fraction for test set |
 * | randomSeed | INT | 42 | Seed for reproducibility |
 * | orientation | STRING | 'UNDIRECTED' | Edge direction: NATURAL, REVERSE, UNDIRECTED |
 * | usePredefinedSplits | BOOL | false | Use splits.bin from projection for train/val/test |
 * | useAdjacencyCache | BOOL | true | Build an in-memory adjacency cache by performing one full B+Tree scan into an unordered_map<src, vector<AdjEntry>>; subsequent neighbor lookups become O(1) hash lookups instead of O(log N) B+Tree walks |
 * | useFourLevelTopologyStore | BOOL | true | Build the frequency-tiered Four-Level Topology Store: L1 RAM hash for hot hubs, L2 compact uint32 CSR for warm nodes, L3 mmap CSR sidecar for cold nodes, L4 direct B+Tree fallback; tier assignment is driven by per-node access-frequency counts |
 * | l1CacheMb | INT | 0 | L1 (RAM hot) budget in MiB; 0 = auto-detect from /proc/meminfo |
 * | l2CacheMb | INT | 0 | L2 (RAM warm) budget in MiB; 0 = auto-detect from /proc/meminfo |
 * | useL3MmapSidecar | BOOL | true | Open the mmap-backed CSR sidecar files (topology_{fwd,rev}.csr) as the L3 cold tier, providing O(1) neighbor slices; falls through to L4 direct B+Tree access if the sidecar files are absent |
 * | force | BOOL | false | Drop and re-create the sample set if it already exists |
 *
 * ## Examples
 *
 * @code{.gql}
 *   -- Basic usage with default options
 *   CALL gnn.offline_sample('social_graph', 'training_v1', [15, 10, 5])
 *   YIELD sampleName, totalBatches, computeMillis
 *   RETURN sampleName, totalBatches, computeMillis;
 *
 *   -- With custom options
 *   CALL gnn.offline_sample('social', 'samples_v1', [15, 10], {
 *       batchSize: 512,
 *       trainRatio: 0.8,
 *       validationRatio: 0.1,
 *       testRatio: 0.1,
 *       randomSeed: 12345,
 *       orientation: 'UNDIRECTED'
 *   })
 *   YIELD sampleName, totalBatches, uniqueNodes
 *   RETURN sampleName, totalBatches, uniqueNodes;
 * @endcode
 *
 * @see gnn.sample_list() to list existing sample sets
 * @see gnn.sample_info() to get sample set details
 * @see gnn.sample_drop() to delete sample sets
 */
class GnnOfflineSampleProcedure : public Procedure {
public:
    std::string name() const override {
        return "gnn_offline_sample";
    }

    std::string qualified_name() const override {
        return "gnn_offline_sample";
    }

    std::string description() const override {
        return "Creates pre-computed GNN mini-batches for offline training "
               "(DiskGNN architecture). Samples are generated once and can be "
               "reused across multiple training epochs.";
    }

    std::vector<Parameter> parameters() const override {
        return {
            Parameter("projectionName", ParamType::STRING, true,
                "Name of the source graph projection to sample from"),
            Parameter("sampleName", ParamType::STRING, true,
                "Name for the created sample set (stored in <db>/samples/<name>/)"),
            Parameter("fanouts", ParamType::LIST, true,
                "List of fanouts per GNN layer, e.g., [15, 10] for 2-hop sampling"),
            Parameter("options", ParamType::ANY, false,
                "Optional configuration map: batchSize, trainRatio, randomSeed, etc.")
        };
    }

    std::vector<YieldField> yield_fields() const override {
        return {
            YieldField{"sampleName", YieldType::STRING,
                "Name of the created sample set"},
            YieldField{"projectionName", YieldType::STRING,
                "Source projection name"},
            YieldField{"totalBatches", YieldType::INT,
                "Total number of batches generated"},
            YieldField{"trainBatches", YieldType::INT,
                "Number of training batches"},
            YieldField{"validationBatches", YieldType::INT,
                "Number of validation batches"},
            YieldField{"testBatches", YieldType::INT,
                "Number of test batches"},
            YieldField{"uniqueNodes", YieldType::INT,
                "Total unique nodes across all samples"},
            YieldField{"storagePath", YieldType::STRING,
                "Path where samples are stored"},
            YieldField{"computeMillis", YieldType::INT,
                "Time taken to compute samples (milliseconds)"},
            YieldField{"phase0Triggered", YieldType::BOOL,
                "True iff the cold-start topology random-walk profiler ran "
                "(triggered when node_counts.bin is absent and the sidecar is available)"},
            YieldField{"phase0Succeeded", YieldType::BOOL,
                "True iff the cold-start profiler successfully wrote node_counts.bin "
                "so the Four-Level Topology Store can warm-start on the next sample build"},
            YieldField{"phase0WalksDone", YieldType::INT,
                "Number of random walks issued by the cold-start profiler "
                "(degree-weighted Vose-alias seed selection)"},
            YieldField{"phase0LookupsDone", YieldType::INT,
                "Total neighbor lookups performed during the cold-start profiling walks"},
            YieldField{"phase0Millis", YieldType::INT,
                "Elapsed time in milliseconds for the cold-start profiling pass"},
            YieldField{"sampleContentFp", YieldType::STRING,
                "Content fingerprint of the sample (the staleness/equality check): hex of the "
                "order-independent XOR fold over batch node-sets + layer shapes "
                "+ edge endpoints. Worker/order-invariant — use as the O(1) "
                "semantic-equality gate across numWorkers and single-vs-parallel "
                "populate instead of re-running train and comparing testAccuracy."},
            YieldField{"samplingBackend", YieldType::STRING,
                "Sampling backend chosen by the hardware-based planner: "
                "CPU_OUT_OF_CORE, GPU_UVA, or GPU_VRAM_COPY. Phase 1 is inert — "
                "GPU_* is reported but the sampling ran on the CPU out-of-core path."},
            YieldField{"samplingDirections", YieldType::STRING,
                "Graph directions the GPU path would serve under the chosen "
                "backend: NONE, FORWARD_ONLY, REVERSE_ONLY, or BOTH."},
            YieldField{"samplingPlanReason", YieldType::STRING,
                "Human-readable reason for the sampling-backend decision "
                "(which gate passed or failed)."},
            YieldField{"symmetricUsed", YieldType::BOOL,
                "True iff the pre-merged undirected slice was resolved-on "
                "(useSymmetricTopology AUTO => UNDIRECTED, or ON)."},
            YieldField{"symmetricBuiltOk", YieldType::BOOL,
                "True once the symmetric slice (GPU pin) or symmetric tier (CPU) "
                "is actually active for this sample."},
            YieldField{"symmetricMs", YieldType::INT,
                "Wall-clock milliseconds spent building the in-RAM merged "
                "undirected slice (0 when not materialized)."},
            YieldField{"symmetricRamBytes", YieldType::INT,
                "Resident bytes of the merged undirected slice (0 when absent)."}
        };
    }

    void execute(ProcedureContext& ctx) override;

private:
    /**
     * @brief Parses fanouts list argument.
     * @param ctx Procedure context
     * @param arg_index Index of the fanouts argument
     * @return Vector of fanout values
     */
    std::vector<uint64_t> parse_fanouts(ProcedureContext& ctx, size_t arg_index);

    /**
     * @brief Parses optional configuration map.
     * @param ctx Procedure context
     * @param arg_index Index of the options argument
     * @param[out] batch_size Parsed batch size
     * @param[out] train_ratio Parsed train ratio
     * @param[out] val_ratio Parsed validation ratio
     * @param[out] test_ratio Parsed test ratio
     * @param[out] random_seed Parsed random seed
     * @param[out] orientation Parsed orientation string
     * @param[out] use_predefined_splits Whether to use splits.bin from projection
     * @param[out] use_predefined_splits_explicit True iff usePredefinedSplits was present in the map
     * @param[out] use_adjacency_cache Whether to build the in-memory adjacency cache (one full B+Tree scan into unordered_map for O(1) neighbor lookups)
     * @param[out] force Whether to drop and re-create an existing sample set
     */
    void parse_options(
        ProcedureContext& ctx,
        size_t arg_index,
        uint64_t& batch_size,
        double& train_ratio,
        double& val_ratio,
        double& test_ratio,
        uint64_t& random_seed,
        std::string& orientation,
        bool& use_predefined_splits,
        bool& use_predefined_splits_explicit,
        bool& use_adjacency_cache,
        bool& use_four_level_topology_store,
        uint64_t& l1_cache_mb,
        uint64_t& l2_cache_mb,
        bool& use_l3_mmap_sidecar,
        bool& auto_profile_on_cold_start,
        uint64_t& profile_num_walks,
        uint64_t& profile_walk_length,
        uint64_t& num_workers,
        bool& force,
        std::string& sampling_backend,
        std::string& symmetric_topology
    );
};

} // namespace Procedures
} // namespace GQL
