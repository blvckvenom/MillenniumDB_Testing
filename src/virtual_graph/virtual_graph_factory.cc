#include "virtual_graph_factory.h"

#include <algorithm>
#include <chrono>
#include <sstream>
#include <unordered_map>
#include <unordered_set>

#include "query/optimizer/quad_model/executor_constructor.h"
#include "query/parser/mql_query_parser.h"
#include "query/query_context.h"
#include "system/buffer_manager.h"
#include "graph_models/quad_model/quad_model.h"
#include "graph_models/quad_model/conversions.h"
#include "graph_models/quad_model/quad_object_id.h"
#include "storage/index/random_access_table/random_access_table.h"

namespace {

std::unique_ptr<QueryExecutor> compile_query(const std::string& query)
{
    auto version_scope = buffer_manager.init_version_readonly();

    static thread_local QueryContext qc;
    QueryContext::set_query_ctx(&qc);
    qc.prepare(*version_scope, std::chrono::seconds(60));

    auto logical_plan = MQL::QueryParser::get_query_plan(query);
    MQL::ExecutorConstructor ctor(MQL::ReturnType::TSV);
    logical_plan->accept_visitor(ctor);
    return std::move(ctor.executor);
}

std::vector<std::vector<std::string>> parse_tsv(const std::string& data)
{
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

std::shared_ptr<VirtualGraph> build_graph_from_rows(
    const std::vector<std::vector<std::string>>& node_rows,
    const std::vector<std::vector<std::string>>& edge_rows
)
{
    auto vg = std::make_shared<VirtualGraph>();

    std::unordered_set<std::string> allowed_nodes;
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
            if (node_id.size() > 1 && node_id[0] == '_' && node_id[1] == 'e')
                continue;

            allowed_nodes.insert(node_id);

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

    if (!edge_rows.empty()) {
        const auto& header = edge_rows.front();
        size_t from_idx = 0;
        size_t to_idx = header.size() >= 3 ? 2 : 1;
        size_t var_idx = 1;
        size_t type_idx = header.size();
        size_t id_idx = header.size();

        for (size_t i = 0; i < header.size(); ++i) {
            const auto& col = header[i];
            if (col == "src_id" || col == "src" || col == "from" || col == "source")
                from_idx = i;
            if (col == "dst_id" || col == "dst" || col == "to" || col == "target")
                to_idx = i;
            if (col == "edge_var" || col == "edge" || col == "r")
                var_idx = i;
            if (col == "rel_type" || col == "type" || col == "label" || col == "t")
                type_idx = i;
            if (col == "rel_id" || col == "id" || col == "edge_id")
                id_idx = i;
        }

        std::unordered_set<std::string> seen_ids;
        std::unordered_set<std::string> seen_keys;

        for (size_t i = 1; i < edge_rows.size(); ++i) {
            const auto& row = edge_rows[i];
            if (row.size() <= std::max({ from_idx, to_idx }))
                continue;

            VirtualGraph::Edge e;
            e.from = row[from_idx];
            e.to = row[to_idx];
            if (row.size() > var_idx)
                e.var = row[var_idx];
            if (row.size() > type_idx)
                e.type = row[type_idx];

            if (e.type.empty() && !e.var.empty()) {
                try {
                    auto edge_oid = QuadObjectId::get_edge(e.var);
                    auto rec = (*quad_model.edge_table)[edge_oid.get_value()];
                    if (rec) {
                        ObjectId type_oid((*rec)[2]);
                        e.type = MQL::Conversions::to_lexical_str(type_oid);
                    }
                } catch (...) {
                    // ignore errors, leave type empty
                }
            }
            if (row.size() > id_idx)
                e.id = row[id_idx];

            if ((e.from.size() > 1 && e.from[0] == '_' && e.from[1] == 'e')
                || (e.to.size() > 1 && e.to[0] == '_' && e.to[1] == 'e'))
                continue;

            if (!allowed_nodes.empty()) {
                if (!allowed_nodes.count(e.from) || !allowed_nodes.count(e.to))
                    continue;
            }

            for (size_t j = 0; j < row.size(); ++j) {
                if (j == from_idx || j == to_idx || j == var_idx || j == type_idx || j == id_idx)
                    continue;
                if (j < header.size())
                    e.properties[header[j]] = row[j];
            }

            if (!e.id.empty()) {
                if (!seen_ids.insert(e.id).second)
                    continue;
            } else {
                std::string key = e.from + "|" + e.to + "|" + e.type;
                for (const auto& [k, v] : e.properties) {
                    key += "|" + k + "=" + v;
                }
                if (!seen_keys.insert(key).second)
                    continue;
            }

            auto ensure_node = [&](const std::string& node_id) {
                auto it = vg->node_index.find(node_id);
                if (it == vg->node_index.end()) {
                    size_t idx = vg->nodes.size();
                    vg->node_index.insert({ node_id, idx });
                    vg->nodes.push_back({ node_id, {} });
                }
            };

            ensure_node(e.from);
            ensure_node(e.to);

            vg->edges.push_back(std::move(e));
        }
    }

    return vg;
}

} // namespace

std::unordered_map<std::string, std::shared_ptr<VirtualGraph>> graphs;

static std::shared_ptr<VirtualGraph> run_project(const std::string& node_query, const std::string& edge_query)
{
    auto node_exec = compile_query(node_query);
    std::stringstream node_ss;
    node_exec->execute(node_ss);
    auto node_rows = parse_tsv(node_ss.str());

    auto edge_exec = compile_query(edge_query);
    std::stringstream edge_ss;
    edge_exec->execute(edge_ss);
    auto edge_rows = parse_tsv(edge_ss.str());

    return build_graph_from_rows(node_rows, edge_rows);
}

std::shared_ptr<VirtualGraph>
    VirtualGraphFactory::project_from_tsv(const std::string& node_tsv, const std::string& edge_tsv)
{
    auto node_rows = parse_tsv(node_tsv);
    auto edge_rows = parse_tsv(edge_tsv);
    return build_graph_from_rows(node_rows, edge_rows);
}

std::shared_ptr<VirtualGraph> VirtualGraphFactory::project(
    const std::string& name,
    const std::string& node_query,
    const std::string& edge_query
)
{
    auto g = run_project(node_query, edge_query);
    graphs[name] = g;
    return g;
}

std::shared_ptr<VirtualGraph> VirtualGraphFactory::get(const std::string& name)
{
    auto it = graphs.find(name);
    if (it != graphs.end()) {
        return it->second;
    }
    return nullptr;
}
