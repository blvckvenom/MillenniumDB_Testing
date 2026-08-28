#pragma once

#include <string>
#include <vector>

#include "query/procedure/procedure.h"
#include "query/procedure/procedure_context.h"

namespace GQL {
namespace Procedures {

/**
 * @brief Gets detailed information about a specific GNN sample set.
 *
 * Returns comprehensive metadata including configuration parameters,
 * batch counts, and storage statistics.
 *
 * ## Syntax
 *
 * @code{.gql}
 *   CALL gnn_sample_info(sampleName)
 *   YIELD sampleName, projectionName, totalBatches, trainBatches, ...
 * @endcode
 *
 * ## Parameters
 *
 * | Name | Type | Required | Description |
 * |------|------|----------|-------------|
 * | sampleName | STRING | Yes | Name of the sample set to inspect |
 *
 * ## Examples
 *
 * @code{.gql}
 *   -- Get all info about a sample set
 *   CALL gnn_sample_info('training_v1')
 *   YIELD sampleName, projectionName, totalBatches, fanouts, batchSize
 *   RETURN *;
 *
 *   -- Check specific configuration
 *   CALL gnn_sample_info('training_v1')
 *   YIELD fanouts, batchSize, randomSeed
 *   RETURN fanouts, batchSize, randomSeed;
 * @endcode
 *
 * @see gnn_sample_list() to list all sample sets
 * @see gnn_offline_sample() to create new sample sets
 * @see gnn_sample_drop() to delete sample sets
 */
class GnnSampleInfoProcedure : public Procedure {
public:
    std::string name() const override {
        return "gnn_sample_info";
    }

    std::string qualified_name() const override {
        return "gnn_sample_info";
    }

    std::string description() const override {
        return "Returns detailed information about a specific GNN sample set, "
               "including configuration parameters and batch statistics.";
    }

    std::vector<Parameter> parameters() const override {
        return {
            Parameter("sampleName", ParamType::STRING, true,
                "Name of the sample set to inspect")
        };
    }

    std::vector<YieldField> yield_fields() const override {
        return {
            YieldField{"sampleName", YieldType::STRING,
                "Name of the sample set"},
            YieldField{"projectionName", YieldType::STRING,
                "Source projection name"},
            YieldField{"totalBatches", YieldType::INT,
                "Total number of batches"},
            YieldField{"trainBatches", YieldType::INT,
                "Number of training batches"},
            YieldField{"validationBatches", YieldType::INT,
                "Number of validation batches"},
            YieldField{"testBatches", YieldType::INT,
                "Number of test batches"},
            YieldField{"uniqueNodes", YieldType::INT,
                "Total unique nodes across all samples"},
            YieldField{"totalEdges", YieldType::INT,
                "Total edges across all samples"},
            YieldField{"batchSize", YieldType::INT,
                "Number of seed nodes per batch"},
            YieldField{"numLayers", YieldType::INT,
                "Number of GNN layers (K)"},
            YieldField{"fanouts", YieldType::STRING,
                "Fanouts per layer as comma-separated string"},
            YieldField{"randomSeed", YieldType::INT,
                "Random seed used for reproducibility"},
            YieldField{"createdAt", YieldType::INT,
                "Unix timestamp when sample set was created"},
            YieldField{"storagePath", YieldType::STRING,
                "Full path to sample storage directory"}
        };
    }

    void execute(ProcedureContext& ctx) override;
};

} // namespace Procedures
} // namespace GQL
