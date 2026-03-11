#pragma once

#include <string>
#include <vector>

#include "query/procedure/procedure.h"
#include "query/procedure/procedure_context.h"
#include "storage/index/hnsw/hnsw_metric.h"

namespace GQL {
namespace Procedures {

/**
 * @brief Creates an HNSW index over GNN node embeddings.
 *
 * This procedure builds an Approximate Nearest Neighbor (ANN) index using the
 * HNSW algorithm over node embeddings stored in the GNN tensor store. The index
 * enables efficient similarity search to find nodes with similar embeddings.
 *
 * ## Syntax
 *
 * @code{.gql}
 *   CALL gnn.hnsw.create(indexName, tensorKey [, options])
 *   YIELD indexName, dimension, nodeCount, buildTimeMs
 * @endcode
 *
 * ## Parameters
 *
 * | Name | Type | Required | Description |
 * |------|------|----------|-------------|
 * | indexName | STRING | Yes | Name for the HNSW index |
 * | tensorKey | STRING | Yes | Name of the feature matrix (e.g., 'node_features') |
 * | options | MAP | No | Configuration: metric, M, efConstruction |
 *
 * ## Options Map
 *
 * | Key | Type | Default | Description |
 * |-----|------|---------|-------------|
 * | metric | STRING | 'cosine' | Distance metric: 'cosine', 'euclidean', 'manhattan' |
 * | M | INT | 16 | Max neighbors per node in the graph |
 * | efConstruction | INT | 200 | Candidates during index construction |
 * | threads | INT | all cores | Number of threads for parallel construction |
 *
 * ## Examples
 *
 * @code{.gql}
 *   -- Basic usage with default options
 *   CALL gnn.hnsw.create('arxiv_idx', 'node_features')
 *   YIELD indexName, dimension, nodeCount, buildTimeMs
 *   RETURN indexName, nodeCount, buildTimeMs;
 *
 *   -- With custom options
 *   CALL gnn.hnsw.create('arxiv_idx', 'node_features', {
 *       metric: 'cosine',
 *       M: 32,
 *       efConstruction: 400
 *   })
 *   YIELD indexName, dimension, nodeCount
 *   RETURN indexName, dimension, nodeCount;
 * @endcode
 *
 * @see gnn.hnsw.find_similar() to search for similar nodes
 * @see gnn.hnsw.list() to list existing indexes
 * @see gnn.hnsw.drop() to delete an index
 */
class GnnHnswCreateProcedure : public Procedure {
public:
    std::string name() const override {
        return "gnn_hnsw_create";
    }

    std::string qualified_name() const override {
        return "gnn_hnsw_create";
    }

    std::string description() const override {
        return "Creates an HNSW index over GNN node embeddings for "
               "approximate nearest neighbor search.";
    }

    std::vector<Parameter> parameters() const override {
        return {
            Parameter("indexName", ParamType::STRING, true,
                "Name for the HNSW index"),
            Parameter("tensorKey", ParamType::STRING, true,
                "Name of the feature matrix (e.g., 'node_features')"),
            Parameter("options", ParamType::ANY, false,
                "Optional configuration: {metric, M, efConstruction}")
        };
    }

    std::vector<YieldField> yield_fields() const override {
        return {
            YieldField{"indexName", YieldType::STRING,
                "Name of the created index"},
            YieldField{"dimension", YieldType::INT,
                "Embedding dimension"},
            YieldField{"nodeCount", YieldType::INT,
                "Number of indexed nodes"},
            YieldField{"buildTimeMs", YieldType::INT,
                "Build time in milliseconds"}
        };
    }

    void execute(ProcedureContext& ctx) override;

private:
    /**
     * @brief Parses metric string to MetricType enum.
     * @param metric_str Metric name ('cosine', 'euclidean', 'inner_product')
     * @return Corresponding MetricType
     * @throws std::runtime_error if metric is unknown
     */
    HNSW::MetricType parse_metric(const std::string& metric_str);

    /**
     * @brief Parses optional configuration map.
     * @param ctx Procedure context
     * @param arg_index Index of the options argument
     * @param[out] metric Parsed metric type
     * @param[out] M Parsed M parameter
     * @param[out] ef_construction Parsed efConstruction parameter
     * @param[out] num_threads Number of threads for parallel construction
     */
    void parse_options(
        ProcedureContext& ctx,
        size_t arg_index,
        HNSW::MetricType& metric,
        uint64_t& M,
        uint64_t& ef_construction,
        size_t& num_threads
    );
};

} // namespace Procedures
} // namespace GQL
