#include "project_streaming_executor.h"

#include "network/server/protocol.h"
#include "query/query_context.h"
#include "virtual_graph/virtual_graph_factory.h"

using namespace MQL;

ProjectStreamingExecutor::ProjectStreamingExecutor(std::string name_, std::string node_q_, std::string edge_q_)
    : name(std::move(name_)), node_query(std::move(node_q_)), edge_query(std::move(edge_q_))
{
    projection_vars.push_back(get_query_ctx().get_or_create_var("status"));
}

const std::vector<VarId>& ProjectStreamingExecutor::get_projection_vars() const {
    return projection_vars;
}

uint64_t ProjectStreamingExecutor::execute(MDBServer::StreamingResponseWriter& writer)
{
    VirtualGraphFactory::project(name, node_query, edge_query);

    writer.write_map_header(2UL);
    writer.write_string("type", MDBServer::Protocol::DataType::STRING);
    writer.write_uint8(static_cast<uint8_t>(MDBServer::Protocol::ResponseType::RECORD));

    writer.write_string("payload", MDBServer::Protocol::DataType::STRING);
    writer.write_list_header(1UL);
    writer.write_string("projected graph \"" + name + "\"", MDBServer::Protocol::DataType::STRING);
    writer.seal();

    return 1;
}

void ProjectStreamingExecutor::analyze(std::ostream& os, bool, int indent) const
{
    os << std::string(indent, ' ') << "ProjectStreamingExecutor(name: " << name << ")\n";
}

