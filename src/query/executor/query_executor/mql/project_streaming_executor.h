#pragma once

#include <string>
#include <vector>

#include "query/executor/query_executor/streaming_query_executor.h"
#include "query/var_id.h"

namespace MQL {

class ProjectStreamingExecutor : public StreamingQueryExecutor {
public:
    ProjectStreamingExecutor(std::string name, std::string node_q, std::string edge_q);

    const std::vector<VarId>& get_projection_vars() const override;

    uint64_t execute(MDBServer::StreamingResponseWriter& response_writer) override;

    void analyze(std::ostream& os, bool print_stats = false, int indent = 0) const override;

private:
    std::string name;
    std::string node_query;
    std::string edge_query;
    std::vector<VarId> projection_vars;
};

} // namespace MQL
