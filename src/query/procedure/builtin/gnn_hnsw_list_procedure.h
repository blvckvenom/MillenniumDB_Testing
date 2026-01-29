#pragma once

#include <string>
#include <vector>

#include "query/procedure/procedure.h"
#include "query/procedure/procedure_context.h"

namespace GQL {
namespace Procedures {

/**
 * @brief Lists all HNSW indexes in the database.
 *
 * ## Syntax
 *
 * @code{.gql}
 *   CALL gnn.hnsw.list()
 *   YIELD indexName, dimension, nodeCount, metric
 * @endcode
 *
 * ## Examples
 *
 * @code{.gql}
 *   CALL gnn.hnsw.list()
 *   YIELD indexName, nodeCount
 *   RETURN indexName, nodeCount;
 * @endcode
 */
class GnnHnswListProcedure : public Procedure {
public:
    std::string name() const override {
        return "gnn_hnsw_list";
    }

    std::string qualified_name() const override {
        return "gnn_hnsw_list";
    }

    std::string description() const override {
        return "Lists all HNSW indexes in the database.";
    }

    std::vector<Parameter> parameters() const override {
        return {};  // No parameters
    }

    std::vector<YieldField> yield_fields() const override {
        return {
            YieldField{"indexName", YieldType::STRING, "Index name"},
            YieldField{"dimension", YieldType::INT, "Embedding dimension"},
            YieldField{"nodeCount", YieldType::INT, "Number of indexed nodes"},
            YieldField{"metric", YieldType::STRING, "Distance metric"}
        };
    }

    void execute(ProcedureContext& ctx) override;
};

} // namespace Procedures
} // namespace GQL
