#include "virtual_graph/virtual_graph.h"
#include "query/executor/binding_iter/virtual_graph/edge_scan.h"
#include "query/executor/binding.h"
#include "graph_models/object_id.h"
#include <algorithm>
#include <cassert>
#include <iostream>
#include <memory>
#include <sstream>
#include <string>
#include <unordered_map>
#include <unordered_set>

namespace QuadObjectId {
ObjectId get_edge(const std::string& str)
{
    std::string tmp = (str.size() > 2 && str[0] == '_' && str[1] == 'e') ? str.substr(2) : str;
    int number = std::stoi(tmp);
    return ObjectId(static_cast<uint64_t>(number) | ObjectId::MASK_EDGE);
}

ObjectId get_string(const std::string& str)
{
    uint64_t res = 0;
    int shift_size = 8 * (ObjectId::STR_INLINE_BYTES - 1);
    for (uint8_t byte : str) {
        uint64_t byte64 = static_cast<uint64_t>(byte);
        res |= byte64 << shift_size;
        shift_size -= 8;
    }
    return ObjectId(res | ObjectId::MASK_STRING_SIMPLE_INLINED);
}

ObjectId get_named_node(const std::string& str)
{
    return get_string(str);
}

ObjectId get_fixed_node_inside(const std::string& str)
{
    if (str.size() > 1 && str[0] == '_' && str[1] == 'e') {
        return get_edge(str);
    }
    if (str.size() > 1 && str[0] == '_' && str[1] == 'a') {
        std::string tmp = str.substr(2);
        int number = std::stoi(tmp);
        return ObjectId(static_cast<uint64_t>(number) | ObjectId::MASK_ANON_INLINED);
    }
    return get_string(str);
}
} // namespace QuadObjectId

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
            if (col == "rel_type" || col == "type" || col == "label" || col == "t" || col == "__rel_type")
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
            if (row.size() > id_idx)
                e.id = row[id_idx];

            if ((e.from.size() > 1 && e.from[0] == '_' && e.from[1] == 'e') ||
                (e.to.size() > 1 && e.to[0] == '_' && e.to[1] == 'e'))
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
    const std::string node_tsv =
        "id\tlabel\nN1\tPerson\nN2\tPerson\nChile\tCountry\nIMFD\tOrganization\n";

    const std::string edge_tsv =
        "src\tdst\tedge_var\trel_type\trel_id\tsince\n"
        "N1\tChile\t_e1\tLivesIn\t1\t\n"
        "N2\tChile\t_e2\tLivesIn\t2\t\n"
        "N1\tN2\t_e3\tKnows\t3\t\n"
        "N2\tN1\t_e4\tKnows\t4\t\n"
        "N1\tIMFD\t_e5\tWorksAt\t5\t2019-08-30\n";

    auto g = load_from_tsv(node_tsv, edge_tsv);
    assert(g->edges.size() == 5);
    std::vector<std::string> expected = { "LivesIn", "LivesIn", "Knows", "Knows", "WorksAt" };
    for (size_t i = 0; i < expected.size(); ++i) {
        assert(g->edges[i].type == expected[i]);
        assert(!g->edges[i].var.empty());
    }

    // ensure type is stored separately from variable
    for (size_t i = 0; i < g->edges.size(); ++i) {
        if (!g->edges[i].type.empty()) {
            assert(g->edges[i].type != g->edges[i].var);
        }
    }

    const std::string edge_tsv_alias =
        "from\tto\tedge\tlabel\tedge_id\n"
        "N1\tChile\t_e1\tLivesIn\t10\n"
        "N2\tChile\t_e2\tLivesIn\t11\n";
    auto g2 = load_from_tsv(node_tsv, edge_tsv_alias);
    assert(g2->edges.size() == 2);
    assert(g2->edges[0].type == "LivesIn");
    assert(g2->edges[0].var == "_e1");

    const std::string edge_tsv_injected =
        "a\tr\tb\t__rel_type\n"
        "N1\t_e1\tChile\tLivesIn\n"
        "N2\t_e2\tChile\tLivesIn\n";
    auto g3 = load_from_tsv(node_tsv, edge_tsv_injected);
    assert(g3->edges.size() == 2);
    assert(g3->edges[0].type == "LivesIn");
    assert(g3->edges[0].var == "_e1");

    const std::string edge_tsv_back =
        "a\tr\tb\n"
        "N1\t_e1\tChile\n"
        "N2\t_e2\tChile\n";
    auto g4 = load_from_tsv(node_tsv, edge_tsv_back);
    assert(g4->edges.size() == 2);
    assert(g4->edges[0].type.empty());
    assert(g4->edges[0].var == "_e1");

    const std::string edge_tsv_dup =
        "src\tdst\tedge_var\trel_type\trel_id\n"
        "N1\tChile\t_e1\tLivesIn\t100\n"
        "N1\tChile\t_e99\tLivesIn\t100\n";
    auto g5 = load_from_tsv(node_tsv, edge_tsv_dup);
    assert(g5->edges.size() == 1);
    assert(g5->edges[0].var == "_e1");

    // rendering prefers type over var
    {
        std::stringstream ss;
        ss << g->edges[0];
        assert(ss.str() == "LivesIn");
    }

    // A) legacy 3-column project with injected type
    {
        VirtualGraphEdgeScan scan(g3, VarId(0), VarId(1), VarId(2), VarId(3), true);
        Binding b(4);
        scan.begin(b);
        std::vector<ObjectId> e_exp = { QuadObjectId::get_edge("_e1"), QuadObjectId::get_edge("_e2") };
        std::vector<ObjectId> t_exp = { QuadObjectId::get_string("LivesIn"), QuadObjectId::get_string("LivesIn") };
        for (size_t i = 0; i < e_exp.size(); ++i) {
            assert(scan.next());
            assert(b[VarId(2)] == e_exp[i]);
            assert(b[VarId(3)] == t_exp[i]);
        }
        assert(!scan.next());
    }

    // B) explicit rel_type column with multiple types
    {
        VirtualGraphEdgeScan scan(g, VarId(0), VarId(1), VarId(2), VarId(3), true);
        Binding b(4);
        scan.begin(b);
        std::vector<ObjectId> e_exp = {
            QuadObjectId::get_edge("_e1"), QuadObjectId::get_edge("_e2"),
            QuadObjectId::get_edge("_e3"), QuadObjectId::get_edge("_e4"),
            QuadObjectId::get_edge("_e5") };
        std::vector<ObjectId> t_exp = {
            QuadObjectId::get_string("LivesIn"), QuadObjectId::get_string("LivesIn"),
            QuadObjectId::get_string("Knows"),   QuadObjectId::get_string("Knows"),
            QuadObjectId::get_string("WorksAt") };
        for (size_t i = 0; i < e_exp.size(); ++i) {
            assert(scan.next());
            assert(b[VarId(2)] == e_exp[i]);
            assert(b[VarId(3)] == t_exp[i]);
        }
        assert(!scan.next());
    }

    // C) edge pattern without type variable
    {
        VirtualGraphEdgeScan scan(g, VarId(0), VarId(1), VarId(2));
        Binding b(3);
        scan.begin(b);
        size_t count = 0;
        while (scan.next()) {
            assert(b[VarId(2)].get_type() == ObjectId::MASK_EDGE);
            count++;
        }
        assert(count == g->edges.size());
    }

    std::cout << "All tests passed\n";
    return 0;
}

