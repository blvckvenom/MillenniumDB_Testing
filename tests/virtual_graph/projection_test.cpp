#include "virtual_graph/virtual_graph.h"
#include <algorithm>
#include <cassert>
#include <iostream>
#include <memory>
#include <sstream>
#include <string>
#include <unordered_map>
#include <unordered_set>

static std::vector<std::vector<std::string>> parse_tsv(const std::string& data)
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

std::shared_ptr<VirtualGraph> load_from_tsv(const std::string& node_tsv, const std::string& edge_tsv)
{
    auto node_rows = parse_tsv(node_tsv);
    auto edge_rows = parse_tsv(edge_tsv);

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
            const auto& node_id = row[id_idx];
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
        size_t type_idx = 1;
        size_t id_idx = header.size();
        for (size_t i = 0; i < header.size(); ++i) {
            if (header[i] == "src_id" || header[i] == "from" || header[i] == "source")
                from_idx = i;
            if (header[i] == "dst_id" || header[i] == "to" || header[i] == "target")
                to_idx = i;
            if (header[i] == "rel_type" || header[i] == "type" || header[i] == "label")
                type_idx = i;
            if (header[i] == "rel_id" || header[i] == "id")
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
            if (row.size() > type_idx)
                e.type = row[type_idx];
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
                if (j == from_idx || j == to_idx || j == type_idx || j == id_idx)
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

            auto ensure_node = [&](const std::string& n) {
                if (!vg->node_index.count(n)) {
                    size_t idx = vg->nodes.size();
                    vg->node_index[n] = idx;
                    vg->nodes.push_back({ n, {} });
                }
            };

            ensure_node(e.from);
            ensure_node(e.to);

            vg->edges.push_back(std::move(e));
        }
    }

    return vg;
}

int main()
{
    const std::string node_tsv = "id\tlabel\n1\tPerson\n2\tCountry\n3\tCompany\n";
    const std::string edge_tsv = "src_id\tdst_id\trel_type\trel_id\n1\t2\tLivesIn\t10\n1\t3\tWorksAt\t11\n";
    auto g = load_from_tsv(node_tsv, edge_tsv);
    assert(g->edges.size() == 2);
    assert(g->edges[0].type == "LivesIn");
    assert(g->edges[1].type == "WorksAt");

    const std::string
        edge_tsv_label = "src_id\tdst_id\tlabel\trel_id\n1\t2\tLivesIn\t20\n1\t3\tWorksAt\t21\n";
    auto g2 = load_from_tsv(node_tsv, edge_tsv_label);
    assert(g2->edges.size() == 2);
    assert(g2->edges[0].type == "LivesIn");
    assert(g2->edges[1].type == "WorksAt");

    const std::string
        edge_tsv_dup = "src_id\tdst_id\trel_type\trel_id\n1\t2\tLivesIn\t100\n1\t2\tLivesIn\t100\n";
    auto g3 = load_from_tsv(node_tsv, edge_tsv_dup);
    assert(g3->edges.size() == 1);
    assert(g3->edges[0].type == "LivesIn");

    std::cout << "All tests passed\n";
    return 0;
}
