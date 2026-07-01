#pragma once

#include <string>
#include <vector>

#include "query/procedure/procedure.h"
#include "query/procedure/procedure_context.h"

namespace GQL {
namespace Procedures {

/**
 * @brief Declares an HNSW index over a node property holding tensors.
 *
 * This is the GQL counterpart of the RDF `CREATE HNSW INDEX ... "predicate"` and
 * the Quad Model `... "property"` statements. The index is declared over a
 * property of the graph, so every entry carries the node's real ObjectId and the
 * real ObjectId of its tensor. Nothing depends on a row's position in a file.
 *
 * The GNN pipeline writes its learned embeddings back as exactly such a property
 * through `gnn_train(..., writeProperty: 'embedding')`, which is what makes those
 * embeddings indexable here.
 *
 * ## Syntax
 *
 * @code{.gql}
 *   CALL gnn_hnsw_create_property(indexName, property [, options])
 *   YIELD indexName, projection, property, dimension, nodeCount, buildTimeMs
 * @endcode
 *
 * ## Options Map
 *
 * | Key | Type | Default | Description |
 * |-----|------|---------|-------------|
 * | projection | STRING | active projection, else base graph | Graph whose property is indexed |
 * | metric | STRING | 'cosine' | 'cosine', 'euclidean' or 'manhattan' |
 * | M | INT | 16 | Max neighbors per node in the graph |
 * | efConstruction | INT | 200 | Candidates considered during construction |
 * | dimension | INT | probed from the data | Tensor width |
 * | seed | INT | 42 | Seeds layer assignment so the build is reproducible |
 *
 * @see gnn_hnsw_search() to query the resulting index
 */
class GnnHnswCreatePropertyProcedure : public Procedure {
public:
    std::string name() const override {
        return "gnn_hnsw_create_property";
    }

    std::string qualified_name() const override {
        return "gnn_hnsw_create_property";
    }

    std::string description() const override {
        return "Declares an HNSW index over a node property holding tensors, such as "
               "the embeddings written back by gnn_train.";
    }

    std::vector<Parameter> parameters() const override {
        return {
            Parameter("indexName", ParamType::STRING, true,
                "Name for the HNSW index"),
            Parameter("property", ParamType::STRING, true,
                "Node property holding the tensors (e.g. 'embedding')"),
            Parameter("options", ParamType::ANY, false,
                "Optional configuration: {projection, metric, M, efConstruction, dimension, seed}")
        };
    }

    std::vector<YieldField> yield_fields() const override {
        return {
            YieldField{"indexName", YieldType::STRING, "Name of the created index"},
            YieldField{"projection", YieldType::STRING, "Graph indexed, empty for the base graph"},
            YieldField{"property", YieldType::STRING, "Indexed node property"},
            YieldField{"dimension", YieldType::INT, "Tensor width"},
            YieldField{"nodeCount", YieldType::INT, "Number of indexed nodes"},
            YieldField{"buildTimeMs", YieldType::INT, "Build time in milliseconds"}
        };
    }

    void execute(ProcedureContext& ctx) override;
};

} // namespace Procedures
} // namespace GQL
