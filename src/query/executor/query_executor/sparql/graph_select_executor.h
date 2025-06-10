#pragma once

#include <memory>
#include <vector>

#include "query/executor/binding_iter.h"
#include "query/executor/query_executor/query_executor.h"
#include "query/executor/query_executor/sparql/ttl_writer.h"

namespace SPARQL {

class GraphSelectExecutor : public QueryExecutor {
public:
    GraphSelectExecutor(
        std::unique_ptr<BindingIter> root,
        std::vector<VarId>           projection_vars
    ) :
        root(std::move(root)),
        projection_vars(std::move(projection_vars)) {}

    uint64_t execute(std::ostream&) override;

    void analyze(std::ostream&, bool print_stats = false, int indent = 0) const override;

private:
    std::unique_ptr<BindingIter> root;
    std::unique_ptr<Binding>     binding;
    std::vector<VarId>           projection_vars;
};

} // namespace SPARQL
