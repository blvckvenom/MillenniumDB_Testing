#pragma once

#include <string>
#include <vector>

#include "query/procedure/procedure.h"
#include "query/procedure/procedure_context.h"

namespace GQL {
namespace Procedures {

/**
 * @brief Queries an HNSW index declared over a node property.
 *
 * Results carry the node's real ObjectId, read back from the index, so they are
 * correct regardless of how the underlying rows are ordered.
 *
 * The seed is either the id of a node already in the index, in which case its
 * vector is read from the indexed property, or a literal list of numbers.
 *
 * ## Syntax
 *
 * @code{.gql}
 *   CALL gnn_hnsw_search(indexName, seed, k, ef)
 *   YIELD node, nodeId, distance
 * @endcode
 *
 * ## Examples
 *
 * @code{.gql}
 *   CALL gnn_hnsw_search('cora_emb_idx', 42, 10, 64)
 *   YIELD node, nodeId, distance RETURN node, distance;
 *
 *   CALL gnn_hnsw_search('cora_emb_idx', [0.11, -0.42, 0.07], 5, 64)
 *   YIELD node, distance RETURN node, distance;
 * @endcode
 *
 * @see gnn_hnsw_create_property() to build such an index
 */
class GnnHnswSearchProcedure : public Procedure {
public:
    std::string name() const override {
        return "gnn_hnsw_search";
    }

    std::string qualified_name() const override {
        return "gnn_hnsw_search";
    }

    std::string description() const override {
        return "Finds the nearest nodes in an HNSW index declared over a node property.";
    }

    std::vector<Parameter> parameters() const override {
        return {
            Parameter("indexName", ParamType::STRING, true,
                "Name of the HNSW index"),
            Parameter("seed", ParamType::ANY, true,
                "Node id of an indexed node, or a literal list of numbers"),
            Parameter("k", ParamType::INT, true,
                "Number of neighbours to return"),
            Parameter("ef", ParamType::INT, true,
                "Number of candidates explored, higher means better recall")
        };
    }

    std::vector<YieldField> yield_fields() const override {
        return {
            YieldField{"node", YieldType::NODE, "The neighbouring node"},
            YieldField{"nodeId", YieldType::INT, "Its numeric id"},
            YieldField{"distance", YieldType::FLOAT, "Distance under the index metric"}
        };
    }

    void execute(ProcedureContext& ctx) override;
};

} // namespace Procedures
} // namespace GQL
