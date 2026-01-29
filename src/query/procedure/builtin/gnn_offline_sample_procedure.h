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
 * | orientation | STRING | 'REVERSE' | Edge direction: NATURAL, REVERSE, UNDIRECTED |
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
 *       orientation: 'REVERSE'
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
                "Time taken to compute samples (milliseconds)"}
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
     */
    void parse_options(
        ProcedureContext& ctx,
        size_t arg_index,
        uint64_t& batch_size,
        double& train_ratio,
        double& val_ratio,
        double& test_ratio,
        uint64_t& random_seed,
        std::string& orientation
    );
};

} // namespace Procedures
} // namespace GQL
