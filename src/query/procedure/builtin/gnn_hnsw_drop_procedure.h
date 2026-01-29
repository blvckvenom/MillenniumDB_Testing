#pragma once

#include <string>
#include <vector>

#include "query/procedure/procedure.h"
#include "query/procedure/procedure_context.h"

namespace GQL {
namespace Procedures {

/**
 * @brief Drops (deletes) an HNSW index.
 *
 * ## Syntax
 *
 * @code{.gql}
 *   CALL gnn.hnsw.drop(indexName)
 *   YIELD success
 * @endcode
 *
 * ## Parameters
 *
 * | Name | Type | Required | Description |
 * |------|------|----------|-------------|
 * | indexName | STRING | Yes | Name of the index to drop |
 *
 * ## Examples
 *
 * @code{.gql}
 *   CALL gnn.hnsw.drop('arxiv_idx')
 *   YIELD success
 *   RETURN success;
 * @endcode
 */
class GnnHnswDropProcedure : public Procedure {
public:
    std::string name() const override {
        return "gnn_hnsw_drop";
    }

    std::string qualified_name() const override {
        return "gnn_hnsw_drop";
    }

    std::string description() const override {
        return "Drops (deletes) an HNSW index.";
    }

    std::vector<Parameter> parameters() const override {
        return {
            Parameter("indexName", ParamType::STRING, true,
                "Name of the index to drop")
        };
    }

    std::vector<YieldField> yield_fields() const override {
        return {
            YieldField{"success", YieldType::BOOL, "Whether the index was dropped"}
        };
    }

    void execute(ProcedureContext& ctx) override;
};

} // namespace Procedures
} // namespace GQL
