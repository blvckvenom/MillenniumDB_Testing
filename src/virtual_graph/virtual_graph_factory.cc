#include "virtual_graph_factory.h"

#include <chrono>
#include <sstream>
#include <unordered_map>
#include <unordered_set>

#include "query/optimizer/quad_model/executor_constructor.h"
#include "query/parser/mql_query_parser.h"
#include "query/query_context.h"
#include "system/buffer_manager.h"

namespace {

std::unique_ptr<QueryExecutor> compile_query(const std::string& query) {
    auto version_scope = buffer_manager.init_version_readonly();

    static thread_local QueryContext qc;
    QueryContext::set_query_ctx(&qc);
    qc.prepare(*version_scope, std::chrono::seconds(60));

    auto logical_plan = MQL::QueryParser::get_query_plan(query);
    MQL::ExecutorConstructor ctor(MQL::ReturnType::TSV);
    logical_plan->accept_visitor(ctor);
    return std::move(ctor.executor);
}

std::vector<std::vector<std::string>> parse_tsv(const std::string& data) {
    std::vector<std::vector<std::string>> rows;
    std::stringstream ss(data);
    std::string line;
    while (std::getline(ss, line)) {
        std::vector<std::string> row;
        size_t start = 0;
        size_t pos;
        while ((pos = line.find('\t', start)) != std::string::npos) {
            row.push_back(line.substr(start, pos - start));
            start = pos + 1;
        }
        row.push_back(line.substr(start));
        rows.push_back(std::move(row));
    }
    return rows;
}

} // namespace

std::unordered_map<std::string, std::shared_ptr<VirtualGraph>> graphs;

static std::shared_ptr<VirtualGraph> run_project(const std::string& node_query,
                                                 const std::string& edge_query)
{
    auto vg = std::make_shared<VirtualGraph>();

    // evaluate edges first to know the set of valid nodes
    std::unordered_set<std::string> valid_nodes;
    {
        auto edge_exec = compile_query(edge_query);
        std::stringstream edge_ss;
        edge_exec->execute(edge_ss);
        auto edge_rows = parse_tsv(edge_ss.str());
        if (!edge_rows.empty()) {
            const auto& header = edge_rows.front();
            // heuristically assume the first column is the source node and the
            // last column is the target node. This matches the default RETURN
            // order used by CALL project examples.
            size_t from_idx = 0;
            size_t to_idx = header.size() >= 3 ? 2 : 1;

            // try to detect explicit column names
            for (size_t i = 0; i < header.size(); ++i) {
                if (header[i] == "from" || header[i] == "source")
                    from_idx = i;
                if (header[i] == "to" || header[i] == "target")
                    to_idx = i;
            }

            for (size_t i = 1; i < edge_rows.size(); ++i) {
                const auto& row = edge_rows[i];
                if (row.size() <= std::max(from_idx, to_idx))
                    continue;

                VirtualGraph::Edge e;
                e.from = row[from_idx];
                e.to = row[to_idx];
                valid_nodes.insert(e.from);
                valid_nodes.insert(e.to);

                for (size_t j = 0; j < row.size(); ++j) {
                    if (j == from_idx || j == to_idx)
                        continue;
                    if (j < header.size())
                        e.properties[header[j]] = row[j];
                }

                vg->edges.push_back(std::move(e));
            }
        }
    }

    // parse nodes keeping only those referenced by edges
    {
        auto node_exec = compile_query(node_query);
        std::stringstream node_ss;
        node_exec->execute(node_ss);
        auto node_rows = parse_tsv(node_ss.str());
        if (!node_rows.empty()) {
            const auto& header = node_rows.front();
            int id_idx = 0;
            for (size_t i = 0; i < header.size(); ++i) {
                if (header[i] == "id" || header[i] == "node") {
                    id_idx = i;
                    break;
                }
            }
            for (size_t i = 1; i < node_rows.size(); ++i) {
                const auto& row = node_rows[i];
                if (row.size() <= static_cast<size_t>(id_idx))
                    continue;
                const std::string& node_id = row[id_idx];
                if (!valid_nodes.empty() && !valid_nodes.count(node_id))
                    continue;
                auto it = vg->node_index.find(node_id);
                if (it == vg->node_index.end()) {
                    size_t idx = vg->nodes.size();
                    vg->node_index.insert({ node_id, idx });
                    vg->nodes.push_back({ node_id, {} });
                    it = vg->node_index.find(node_id);
                }
                auto& props = vg->nodes[it->second].properties;
                for (size_t j = 0; j < row.size(); ++j) {
                    if (j == static_cast<size_t>(id_idx))
                        continue;
                    if (j < header.size())
                        props[header[j]] = row[j];
                }
            }
        }
    }

    // ensure nodes referenced by edges exist even if not in node query
    for (const auto& id : valid_nodes) {
        if (vg->node_index.find(id) == vg->node_index.end()) {
            size_t idx = vg->nodes.size();
            vg->node_index.insert({ id, idx });
            vg->nodes.push_back({ id, {} });
        }
    }

    return vg;
}

std::shared_ptr<VirtualGraph> VirtualGraphFactory::project(
    const std::string& name,
    const std::string& node_query,
    const std::string& edge_query) {
    auto g = run_project(node_query, edge_query);
    graphs[name] = g;
    return g;
}

std::shared_ptr<VirtualGraph> VirtualGraphFactory::get(const std::string& name) {
    auto it = graphs.find(name);
    if (it != graphs.end()) {
        return it->second;
    }
    return nullptr;
}


