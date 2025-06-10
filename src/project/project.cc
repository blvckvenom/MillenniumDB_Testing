#include "project.h"

#include <fstream>
#include <sstream>
#include <stdexcept>
#include <filesystem>
#include <cctype>

#include "misc/trim.h"

std::unordered_map<std::string, PropertyGraph> projectedGraphs;

static std::string strip_quotes(const std::string& s) {
    if (s.size() >= 2 && s.front() == '"' && s.back() == '"') {
        return s.substr(1, s.size() - 2);
    }
    return s;
}

static std::vector<std::string> tokenize(const std::string& line) {
    std::vector<std::string> tokens;
    std::string current;
    bool in_quote = false;
    for (char c : line) {
        if (c == '"') {
            in_quote = !in_quote;
            current += c;
        } else if (std::isspace(static_cast<unsigned char>(c)) && !in_quote) {
            if (!current.empty()) {
                tokens.push_back(current);
                current.clear();
            }
        } else {
            current += c;
        }
    }
    if (!current.empty()) {
        tokens.push_back(current);
    }
    return tokens;
}

static void parse_node(PropertyGraph& g, const std::string& line) {
    auto tokens = tokenize(line);
    if (tokens.empty()) return;
    PropertyGraphNode node;
    node.id = tokens[0];
    size_t i = 1;
    for (; i < tokens.size(); i++) {
        const auto& t = tokens[i];
        if (!t.empty() && t[0] == ':' && t.find(':', 1) == std::string::npos) {
            node.labels.push_back(t.substr(1));
        } else {
            break;
        }
    }
    for (; i < tokens.size(); i++) {
        auto pos = tokens[i].find(':');
        if (pos == std::string::npos) continue;
        auto key = tokens[i].substr(0, pos);
        auto val = tokens[i].substr(pos + 1);
        node.properties[key] = val;
    }
    g.nodes[node.id] = std::move(node);
}

static void parse_edge(PropertyGraph& g, const std::string& line) {
    auto arrow_pos = line.find("->");
    if (arrow_pos == std::string::npos) return;
    auto from = std::string(trim_string(line.substr(0, arrow_pos)));
    auto rest = line.substr(arrow_pos + 2);
    auto colon_pos = rest.find(':');
    if (colon_pos == std::string::npos) return;
    auto to = std::string(trim_string(rest.substr(0, colon_pos)));
    rest = rest.substr(colon_pos + 1);
    auto tokens = tokenize(rest);
    if (tokens.empty()) return;
    PropertyGraphEdge edge;
    edge.from = from;
    edge.to = to;
    edge.type = tokens[0];
    for (size_t i = 1; i < tokens.size(); i++) {
        auto pos = tokens[i].find(':');
        if (pos == std::string::npos) continue;
        auto key = tokens[i].substr(0, pos);
        auto val = tokens[i].substr(pos + 1);
        edge.properties[key] = val;
    }
    g.edges.push_back(std::move(edge));
}

std::string project(const std::string& name, const std::string& filePath) {
    std::filesystem::path path(filePath);
    if (path.extension() != ".qm") {
        throw std::runtime_error("project() only accepts .qm files");
    }
    std::ifstream in(filePath);
    if (!in.is_open()) {
        throw std::runtime_error("Could not open file " + filePath);
    }
    PropertyGraph g;
    std::string line;
    std::stringstream serialized;
    while (std::getline(in, line)) {
        auto trimmed_view = trim_string(line);
        std::string trimmed(trimmed_view);
        if (trimmed.empty()) continue;
        serialized << trimmed << "\n";
        if (trimmed.find("->") != std::string::npos) {
            parse_edge(g, trimmed);
        } else {
            parse_node(g, trimmed);
        }
    }
    projectedGraphs[name] = std::move(g);
    return serialized.str();
}
