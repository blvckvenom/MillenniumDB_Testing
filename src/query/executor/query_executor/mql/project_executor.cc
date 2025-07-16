#include "project_executor.h"

#include "virtual_graph/virtual_graph_factory.h"

using namespace MQL;

ProjectExecutor::ProjectExecutor(std::string name_, std::string node_q_, std::string edge_q_)
    : name(std::move(name_)), node_query(std::move(node_q_)), edge_query(std::move(edge_q_)) {}

uint64_t ProjectExecutor::execute(std::ostream& os) {
    VirtualGraphFactory::project(name, node_query, edge_query);
    os << "projected graph \"" << name << "\"\n";
    return 1;
}

void ProjectExecutor::analyze(std::ostream& os, bool, int indent) const {
    os << std::string(indent, ' ') << "ProjectExecutor(name: " << name << ")\n";
}
