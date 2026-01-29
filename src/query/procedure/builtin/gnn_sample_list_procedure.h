#pragma once

#include <string>
#include <vector>

#include "query/procedure/procedure.h"
#include "query/procedure/procedure_context.h"

namespace GQL {
namespace Procedures {

/**
 * @brief Lists all pre-computed GNN sample sets in the database.
 *
 * Returns metadata for each sample set without loading the full sample data.
 * Use this to discover available sample sets and their configurations.
 *
 * ## Syntax
 *
 * @code{.gql}
 *   CALL gnn.sample_list()
 *   YIELD sampleName, projectionName, totalBatches, uniqueNodes, createdAt
 * @endcode
 *
 * ## Examples
 *
 * @code{.gql}
 *   -- List all sample sets
 *   CALL gnn.sample_list()
 *   YIELD sampleName, projectionName, totalBatches
 *   RETURN sampleName, projectionName, totalBatches;
 *
 *   -- Filter by projection
 *   CALL gnn.sample_list()
 *   YIELD sampleName, projectionName, totalBatches
 *   WHERE projectionName = 'social_graph'
 *   RETURN sampleName, totalBatches;
 * @endcode
 *
 * @see gnn.offline_sample() to create new sample sets
 * @see gnn.sample_info() to get detailed information about a specific sample set
 * @see gnn.sample_drop() to delete sample sets
 */
class GnnSampleListProcedure : public Procedure {
public:
    std::string name() const override {
        return "gnn_sample_list";
    }

    std::string qualified_name() const override {
        return "gnn_sample_list";
    }

    std::string description() const override {
        return "Lists all pre-computed GNN sample sets in the database, "
               "including their source projections and batch counts.";
    }

    std::vector<Parameter> parameters() const override {
        return {};  // No parameters
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
            YieldField{"createdAt", YieldType::INT,
                "Unix timestamp when sample set was created"}
        };
    }

    void execute(ProcedureContext& ctx) override;
};

} // namespace Procedures
} // namespace GQL
