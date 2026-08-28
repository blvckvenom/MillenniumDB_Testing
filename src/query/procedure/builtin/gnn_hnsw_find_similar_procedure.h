#pragma once

#include <string>
#include <vector>

#include "query/procedure/procedure.h"
#include "query/procedure/procedure_context.h"

namespace GQL {
namespace Procedures {

/**
 * @brief Finds nodes with similar embeddings using HNSW index.
 *
 * This procedure performs approximate nearest neighbor search using a
 * pre-built HNSW index to find nodes whose embeddings are similar to
 * a given seed node.
 *
 * ## Syntax
 *
 * @code{.gql}
 *   CALL gnn_hnsw_find_similar(indexName, nodeId, k, ef)
 *   YIELD similar_node, distance
 * @endcode
 *
 * ## Parameters
 *
 * | Name | Type | Required | Description |
 * |------|------|----------|-------------|
 * | indexName | STRING | Yes | Name of the HNSW index to query |
 * | nodeId | INT | Yes | Seed node ID whose similar nodes to find |
 * | k | INT | Yes | Number of similar nodes to return |
 * | ef | INT | Yes | Search candidates (higher = better recall, slower) |
 *
 * ## Output
 *
 * Returns one row per similar node, ordered by distance (ascending):
 * - similar_node: The node ObjectId
 * - distance: Distance to the seed node (lower = more similar)
 *
 * ## Examples
 *
 * @code{.gql}
 *   -- Find 10 most similar nodes to node 42
 *   CALL gnn_hnsw_find_similar('arxiv_idx', 42, 10, 100)
 *   YIELD similar_node, distance
 *   RETURN similar_node, distance;
 *
 *   -- Find similar nodes and get their properties
 *   CALL gnn_hnsw_find_similar('arxiv_idx', 42, 5, 50)
 *   YIELD similar_node, distance
 *   MATCH (n) WHERE id(n) = similar_node
 *   RETURN n.title, distance;
 * @endcode
 *
 * @see gnn_hnsw_create() to create an index
 * @see gnn_hnsw_list() to list existing indexes
 */
class GnnHnswFindSimilarProcedure : public Procedure {
public:
    std::string name() const override {
        return "gnn_hnsw_find_similar";
    }

    std::string qualified_name() const override {
        return "gnn_hnsw_find_similar";
    }

    std::string description() const override {
        return "Finds nodes with similar embeddings to a given seed node "
               "using approximate nearest neighbor search.";
    }

    std::vector<Parameter> parameters() const override {
        return {
            Parameter("indexName", ParamType::STRING, true,
                "Name of the HNSW index to query"),
            Parameter("nodeId", ParamType::INT, true,
                "Seed node ID whose similar nodes to find"),
            Parameter("k", ParamType::INT, true,
                "Number of similar nodes to return"),
            Parameter("ef", ParamType::INT, true,
                "Number of candidates (higher = better recall, slower)")
        };
    }

    std::vector<YieldField> yield_fields() const override {
        return {
            YieldField{"similar_node", YieldType::NODE,
                "Similar node"},
            YieldField{"distance", YieldType::FLOAT,
                "Distance to seed node (lower = more similar)"}
        };
    }

    void execute(ProcedureContext& ctx) override;
};

} // namespace Procedures
} // namespace GQL
