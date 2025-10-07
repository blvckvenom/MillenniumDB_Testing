#include "virtual_graph_filter.h"

#include "query/query_context.h"

VirtualGraphFilter::VirtualGraphFilter(
    std::unique_ptr<BindingIter> child,
    std::shared_ptr<VirtualGraph> graph,
    std::vector<VarId> node_vars,
    std::vector<VarId> edge_vars
) :
    child(std::move(child)),
    graph(std::move(graph)),
    node_vars(std::move(node_vars)),
    edge_vars(std::move(edge_vars))
{
}

void VirtualGraphFilter::_begin(Binding& parent)
{
    parent_binding = &parent;
    child->begin(parent);
}

bool VirtualGraphFilter::_next()
{
    while (child->next()) {
        bool ok = true;
        for (auto var : node_vars) {
            if (!graph->contains_node((*parent_binding)[var])) {
                ok = false;
                break;
            }
        }
        if (!ok) {
            continue;
        }
        for (auto var : edge_vars) {
            if (!graph->contains_edge((*parent_binding)[var])) {
                ok = false;
                break;
            }
        }
        if (ok) {
            return true;
        }
    }
    return false;
}

void VirtualGraphFilter::_reset()
{
    child->reset();
}

void VirtualGraphFilter::assign_nulls()
{
    child->assign_nulls();
}

void VirtualGraphFilter::print(std::ostream& os, int indent, bool stats) const
{
    if (stats) {
        print_generic_stats(os, indent);
    }
    os << std::string(indent, ' ') << "VirtualGraphFilter(nodes=" << node_vars.size()
       << ", edges=" << edge_vars.size() << ")\n";
    child->print(os, indent + 2, stats);
}
