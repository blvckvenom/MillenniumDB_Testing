#pragma once

#include <string>
#include <vector>

#include "query/procedure/procedure.h"
#include "query/procedure/procedure_context.h"

namespace GQL {
namespace Procedures {

/**
 * @brief Gets detailed information about an HNSW index.
 *
 * ## Syntax
 *
 * @code{.gql}
 *   CALL gnn.hnsw.info(indexName)
 *   YIELD indexName, dimension, nodeCount, metric, M, efConstruction
 * @endcode
 *
 * ## Parameters
 *
 * | Name | Type | Required | Description |
 * |------|------|----------|-------------|
 * | indexName | STRING | Yes | Name of the index to inspect |
 *
 * ## Examples
 *
 * @code{.gql}
 *   CALL gnn.hnsw.info('arxiv_idx')
 *   YIELD indexName, dimension, nodeCount, M
 *   RETURN indexName, dimension, nodeCount, M;
 * @endcode
 */
class GnnHnswInfoProcedure : public Procedure {
public:
    std::string name() const override {
        return "gnn_hnsw_info";
    }

    std::string qualified_name() const override {
        return "gnn_hnsw_info";
    }

    std::string description() const override {
        return "Gets detailed information about an HNSW index.";
    }

    std::vector<Parameter> parameters() const override {
        return {
            Parameter("indexName", ParamType::STRING, true,
                "Name of the index to inspect")
        };
    }

    std::vector<YieldField> yield_fields() const override {
        return {
            YieldField{"indexName", YieldType::STRING, "Index name"},
            YieldField{"dimension", YieldType::INT, "Embedding dimension"},
            YieldField{"nodeCount", YieldType::INT, "Number of indexed nodes"},
            YieldField{"metric", YieldType::STRING, "Distance metric"},
            YieldField{"M", YieldType::INT, "Max neighbors per node"},
            YieldField{"efConstruction", YieldType::INT, "Build-time candidates"},
            YieldField{"layers", YieldType::INT, "Number of HNSW layers"}
        };
    }

    void execute(ProcedureContext& ctx) override;
};

} // namespace Procedures
} // namespace GQL
