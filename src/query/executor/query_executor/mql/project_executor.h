#pragma once

#include <string>
#include "query/executor/query_executor/query_executor.h"

class VirtualGraph;

namespace MQL {

class ProjectExecutor : public QueryExecutor {
public:
    ProjectExecutor(std::string name, std::string node_q, std::string edge_q);

    uint64_t execute(std::ostream& os) override;
    void analyze(std::ostream& os, bool print_stats = false, int indent = 0) const override;

private:
    std::string name;
    std::string node_query;
    std::string edge_query;
};

} // namespace MQL
