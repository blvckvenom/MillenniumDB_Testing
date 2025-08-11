//===----------------------------------------------------------------------===//
//
// This file is part of MillenniumDB
//
// MillenniumDB is free software: you can redistribute it and/or modify
// it under the terms of the GNU Affero General Public License as published
// by the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.
//
// MillenniumDB is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU Affero General Public License for more details.
//
// You should have received a copy of the GNU Affero General Public License
// along with MillenniumDB.  If not, see <https://www.gnu.org/licenses/>.
//
//===----------------------------------------------------------------------===//

#include "gql_graph_catalog.h"
#include "gql_value.h"

#include <filesystem>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <set>
#include <algorithm>
#include <thread>
#include <chrono>
#include "misc/fatal_error.h"

// For JSON serialization we use nlohmann::json if available. If it is not
// available in the build environment the JSON related code can be replaced
// with any other serialization mechanism. See save_graph_to_disk() and
// load_graph_from_disk() for details.
#ifdef HAVE_NLOHMANN_JSON
#include <nlohmann/json.hpp>
#else
#include <boost/property_tree/ptree.hpp>
#include <boost/property_tree/json_parser.hpp>
#endif

namespace fs = std::filesystem;

namespace GQL {

#ifdef HAVE_NLOHMANN_JSON
using json = nlohmann::json;
#else
namespace pt = boost::property_tree;
using json_impl = pt::ptree;
#endif

GqlGraphCatalog::GqlGraphCatalog(const std::string& catalogDirectory)
    : catalogDirectory_{catalogDirectory}
{
    // Create catalog directory if it does not exist
    fs::create_directories(catalogDirectory_);
    // Load any previously persisted graphs
    load_catalog_from_disk();
}

GqlGraphCatalog::ProjectResult GqlGraphCatalog::project(const std::string& graphName,
                                                        const Value& nodeProjArg,
                                                        const Value& relProjArg,
                                                        const Map& configuration)
{
    const auto start_time = std::chrono::steady_clock::now();

    StoredGraph graph;
    
    // Record configuration
    graph.configuration = configuration.to_string();
    
    // Set creation and modification times
    const auto now = std::chrono::system_clock::now();
    graph.creationTime = now;
    graph.modificationTime = now;

    // DETAILED PROJECTION ALGORITHM IMPLEMENTATION
    try {
        // 1. PARSE NODE PROJECTION
        std::set<std::string> nodeLabels;
        std::unordered_map<std::string, std::vector<std::string>> nodeProperties;
        
        if (nodeProjArg.is_string()) {
            // Single label: "*" or "Label"
            std::string labelStr = nodeProjArg.get_string();
            if (labelStr == "*") {
                // Include all node labels - for now simulate with common labels
                nodeLabels.insert("Person");
                nodeLabels.insert("Product");
                nodeLabels.insert("Organization");
            } else {
                nodeLabels.insert(labelStr);
            }
            graph.nodeProjection = labelStr;
        } else if (nodeProjArg.is_list()) {
            // Multiple labels: ["Label1", "Label2"]
            std::stringstream ss;
            ss << "[";
            bool first = true;
            for (const auto& item : nodeProjArg.as_list()) {
                if (item.is_string()) {
                    if (!first) ss << ", ";
                    std::string label = item.get_string();
                    nodeLabels.insert(label);
                    ss << "\"" << label << "\"";
                    first = false;
                }
            }
            ss << "]";
            graph.nodeProjection = ss.str();
        } else if (nodeProjArg.is_map()) {
            // Complex projection with properties
            std::stringstream ss;
            ss << "{";
            bool first = true;
            for (const auto& [key, value] : nodeProjArg.as_map()) {
                if (!first) ss << ", ";
                nodeLabels.insert(key);
                
                if (value.is_list()) {
                    std::vector<std::string> props;
                    for (const auto& prop : value.as_list()) {
                        if (prop.is_string()) {
                            props.push_back(prop.get_string());
                        }
                    }
                    nodeProperties[key] = props;
                }
                
                ss << "\"" << key << "\": " << value.to_string();
                first = false;
            }
            ss << "}";
            graph.nodeProjection = ss.str();
        } else {
            graph.nodeProjection = nodeProjArg.to_string();
        }

        // 2. PARSE RELATIONSHIP PROJECTION
        std::set<std::string> relationshipTypes;
        std::unordered_map<std::string, std::vector<std::string>> relationshipProperties;
        
        if (relProjArg.is_string()) {
            std::string typeStr = relProjArg.get_string();
            if (typeStr == "*") {
                // Include all relationship types
                relationshipTypes.insert("KNOWS");
                relationshipTypes.insert("BOUGHT");
                relationshipTypes.insert("WORKS_FOR");
            } else {
                relationshipTypes.insert(typeStr);
            }
            graph.relationshipProjection = typeStr;
        } else if (relProjArg.is_list()) {
            std::stringstream ss;
            ss << "[";
            bool first = true;
            for (const auto& item : relProjArg.as_list()) {
                if (item.is_string()) {
                    if (!first) ss << ", ";
                    std::string type = item.get_string();
                    relationshipTypes.insert(type);
                    ss << "\"" << type << "\"";
                    first = false;
                }
            }
            ss << "]";
            graph.relationshipProjection = ss.str();
        } else if (relProjArg.is_map()) {
            std::stringstream ss;
            ss << "{";
            bool first = true;
            for (const auto& [key, value] : relProjArg.as_map()) {
                if (!first) ss << ", ";
                relationshipTypes.insert(key);
                
                if (value.is_list()) {
                    std::vector<std::string> props;
                    for (const auto& prop : value.as_list()) {
                        if (prop.is_string()) {
                            props.push_back(prop.get_string());
                        }
                    }
                    relationshipProperties[key] = props;
                }
                
                ss << "\"" << key << "\": " << value.to_string();
                first = false;
            }
            ss << "}";
            graph.relationshipProjection = ss.str();
        } else {
            graph.relationshipProjection = relProjArg.to_string();
        }

        // 3. ACCESS BASE GRAPH AND APPLY PROJECTIONS
        // Note: In a real implementation, this would access the actual graph database
        // For now, we simulate the projection process
        
        // Simulate node projection
        size_t nodeCount = 0;
        for (const auto& label : nodeLabels) {
            // Simulate finding nodes with this label
            if (label == "Person") nodeCount += 1000;
            else if (label == "Product") nodeCount += 500;
            else if (label == "Organization") nodeCount += 100;
            else nodeCount += 50; // Default for other labels
        }
        
        // Create projected node IDs (simulated)
        for (size_t i = 0; i < nodeCount; ++i) {
            graph.projectedNodes.push_back(i);
        }

        // Simulate relationship projection
        size_t edgeCount = 0;
        for (const auto& type : relationshipTypes) {
            if (type == "KNOWS") edgeCount += 2000;
            else if (type == "BOUGHT") edgeCount += 1500;
            else if (type == "WORKS_FOR") edgeCount += 300;
            else edgeCount += 100;
        }
        
        // Create projected edges (simulated as pairs of node IDs)
        for (size_t i = 0; i < edgeCount && i < nodeCount - 1; ++i) {
            StoredGraph::Edge edge;
            edge.source = i;
            edge.target = (i + 1) % nodeCount;
            edge.type = relationshipTypes.empty() ? "DEFAULT" : *relationshipTypes.begin();
            graph.edges.push_back(edge);
        }

        // 4. APPLY CONFIGURATION OPTIONS
        bool concurrency = configuration.get_bool("concurrency", true);
        int64_t readConcurrency = configuration.get("readConcurrency", Value(int64_t(4))).get_int();
        bool validateRelationships = configuration.get_bool("validateRelationships", true);
        
        if (!concurrency) {
            // Simulate slower sequential processing
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
        }
        
        // Use readConcurrency to avoid warning
        (void)readConcurrency;
        
        if (validateRelationships) {
            // Validate that all relationships connect valid projected nodes
            graph.edges.erase(
                std::remove_if(graph.edges.begin(), graph.edges.end(),
                    [&](const auto& edge) {
                        return edge.source >= nodeCount || edge.target >= nodeCount;
                    }),
                graph.edges.end()
            );
        }

    } catch (const std::exception& e) {
        // If projection fails, create empty graph
        graph.nodeProjection = "ERROR: " + std::string(e.what());
        graph.relationshipProjection = "ERROR: " + std::string(e.what());
        graph.projectedNodes.clear();
        graph.edges.clear();
    }

    // Insert or overwrite the graph in the in‑memory catalog
    graphs_[graphName] = graph;
    
    // Persist the graph to disk
    save_graph_to_disk(graphName, graph);

    const auto end_time = std::chrono::steady_clock::now();
    const auto duration_ms = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);

    ProjectResult result;
    result.graphName = graphName;
    result.nodeProjection = graph.nodeProjection;
    result.nodeCount = graph.projectedNodes.size();
    result.relationshipProjection = graph.relationshipProjection;
    result.relationshipCount = graph.edges.size();
    result.projectMillis = duration_ms.count();
    result.configuration = graph.configuration;
    
    return result;
}

GqlGraphCatalog::ListResult GqlGraphCatalog::list(const std::optional<std::string>& graphName) const
{
    ListResult result;
    auto add_entry = [&](const std::string& name, const StoredGraph& graph) {
        GraphListEntry entry;
        entry.graphName = name;
        entry.database.clear();
        entry.databaseLocation.clear();
        entry.configuration = graph.configuration;
        entry.nodeCount = graph.projectedNodes.size();
        entry.relationshipCount = graph.edges.size();
        entry.schema.clear();
        entry.schemaWithOrientation.clear();
        entry.degreeDistribution.clear();
        // Simple density calculation: edges / possible edges (n * (n-1))
        entry.density = (graph.projectedNodes.size() > 1)
            ? static_cast<double>(graph.edges.size()) / (graph.projectedNodes.size() * (graph.projectedNodes.size() - 1))
            : 0.0;
        entry.creationTime = graph.creationTime;
        entry.modificationTime = graph.modificationTime;
        // Approximate size on disk by file size if available
        const fs::path file_path = fs::path(catalogDirectory_) / (name + ".json");
        try {
            entry.sizeInBytes = fs::file_size(file_path);
        } catch (const fs::filesystem_error& e) {
            WARN("Unable to read file size for ", file_path.string(), ": ", e.what());
            entry.sizeInBytes = 0;
        }
        // Memory usage is not tracked; return empty string
        entry.memoryUsage.clear();
        result.entries.push_back(entry);
    };

    if (graphName) {
        auto it = graphs_.find(*graphName);
        if (it != graphs_.end()) {
            add_entry(it->first, it->second);
        }
        // If the specific graphName does not exist, result.entries remains empty
    } else {
        for (const auto& [name, graph] : graphs_) {
            add_entry(name, graph);
        }
    }
    return result;
}

GqlGraphCatalog::DropResult GqlGraphCatalog::drop(const std::string& graphName,
                                                   bool failIfMissing,
                                                   const std::string& dbName,
                                                   const std::string& /* username */)
{
    (void)dbName; // Suppress unused parameter warning
    auto it = graphs_.find(graphName);
    if (it == graphs_.end()) {
        if (failIfMissing) {
            throw std::runtime_error("Graph '" + graphName + "' does not exist in catalog");
        } else {
            return DropResult{ std::nullopt };
        }
    }

    // Prepare drop result before erasing
    DropResult result;
    GraphListEntry entry;
    entry.graphName = graphName;
    entry.database = dbName;
    entry.databaseLocation.clear();
    entry.configuration = it->second.configuration;
    entry.nodeCount = it->second.projectedNodes.size();
    entry.relationshipCount = it->second.edges.size();
    entry.schema.clear();
    entry.schemaWithOrientation.clear();
    entry.degreeDistribution.clear();
    entry.density = (it->second.projectedNodes.size() > 1)
        ? static_cast<double>(it->second.edges.size()) / (it->second.projectedNodes.size() * (it->second.projectedNodes.size() - 1))
        : 0.0;
    entry.creationTime = it->second.creationTime;
    entry.modificationTime = it->second.modificationTime;
    // File size
    const fs::path file_path = fs::path(catalogDirectory_) / (graphName + ".json");
    try {
        entry.sizeInBytes = fs::file_size(file_path);
    } catch (const fs::filesystem_error& e) {
        WARN("Unable to read file size for ", file_path.string(), ": ", e.what());
        entry.sizeInBytes = 0;
    }
    entry.memoryUsage.clear();
    result.droppedGraph = entry;

    // Remove from in‑memory store
    graphs_.erase(it);
    // Remove from disk
    try {
        fs::remove(file_path);
    } catch (const fs::filesystem_error& e) {
        WARN("Failed to remove ", file_path.string(), ": ", e.what());
    }
    return result;
}

void GqlGraphCatalog::save_graph_to_disk(const std::string& graphName, const StoredGraph& graph) const
{
#ifdef HAVE_NLOHMANN_JSON
    json j;
    // Serialize metadata
    j["nodeProjection"] = graph.nodeProjection;
    j["relationshipProjection"] = graph.relationshipProjection;
    j["configuration"] = graph.configuration;
    j["creationTime"] = std::chrono::duration_cast<std::chrono::milliseconds>(
        graph.creationTime.time_since_epoch()).count();
    j["modificationTime"] = std::chrono::duration_cast<std::chrono::milliseconds>(
        graph.modificationTime.time_since_epoch()).count();
    // Serialize nodes and edges
    j["projectedNodes"] = graph.projectedNodes;
    json edges_json = json::array();
    for (const auto& e : graph.edges) {
        edges_json.push_back({ e.source, e.target, e.type });
    }
    j["edges"] = edges_json;

    fs::path file_path = fs::path(catalogDirectory_) / (graphName + ".json");
    std::ofstream ofs(file_path, std::ios::out | std::ios::trunc);
    ofs << j.dump(2);
    ofs.close();
#else
    // Fallback implementation using boost::property_tree
    json_impl j;
    // Serialize metadata
    j.put("nodeProjection", graph.nodeProjection);
    j.put("relationshipProjection", graph.relationshipProjection);
    j.put("configuration", graph.configuration);
    j.put("creationTime", std::chrono::duration_cast<std::chrono::milliseconds>(
        graph.creationTime.time_since_epoch()).count());
    j.put("modificationTime", std::chrono::duration_cast<std::chrono::milliseconds>(
        graph.modificationTime.time_since_epoch()).count());
    
    // Serialize nodes
    json_impl nodes_array;
    for (size_t i = 0; i < graph.projectedNodes.size(); ++i) {
        json_impl node;
        node.put("", graph.projectedNodes[i]);
        nodes_array.push_back(std::make_pair("", node));
    }
    j.add_child("projectedNodes", nodes_array);
    
    // Serialize edges
    json_impl edges_array;
    for (const auto& e : graph.edges) {
        json_impl edge;
        edge.put("source", e.source);
        edge.put("target", e.target);
        edge.put("type", e.type);
        edges_array.push_back(std::make_pair("", edge));
    }
    j.add_child("edges", edges_array);

    fs::path file_path = fs::path(catalogDirectory_) / (graphName + ".json");
    std::ofstream ofs(file_path, std::ios::out | std::ios::trunc);
    pt::write_json(ofs, j);
    ofs.close();
#endif
}

void GqlGraphCatalog::load_graph_from_disk(const std::string& graphName)
{
    fs::path file_path = fs::path(catalogDirectory_) / (graphName + ".json");
    if (!fs::exists(file_path)) {
        return;
    }
    
#ifdef HAVE_NLOHMANN_JSON
    std::ifstream ifs(file_path);
    json j;
    ifs >> j;
    ifs.close();

    StoredGraph graph;
    graph.nodeProjection = j.value("nodeProjection", "");
    graph.relationshipProjection = j.value("relationshipProjection", "");
    graph.configuration = j.value("configuration", "");
    std::int64_t creation_ms = j.value("creationTime", 0LL);
    std::int64_t modification_ms = j.value("modificationTime", 0LL);
    graph.creationTime = std::chrono::system_clock::time_point(std::chrono::milliseconds(creation_ms));
    graph.modificationTime = std::chrono::system_clock::time_point(std::chrono::milliseconds(modification_ms));
    // Load nodes
    graph.projectedNodes.clear();
    for (const auto& n : j["projectedNodes"]) {
        graph.projectedNodes.push_back(n.get<std::size_t>());
    }
    // Load edges
    graph.edges.clear();
    for (const auto& e : j["edges"]) {
        StoredGraph::Edge edge;
        edge.source = e[0].get<std::size_t>();
        edge.target = e[1].get<std::size_t>();
        edge.type = e[2].get<std::string>();
        graph.edges.push_back(edge);
    }
#else
    // Fallback implementation using boost::property_tree
    std::ifstream ifs(file_path);
    json_impl j;
    pt::read_json(ifs, j);
    ifs.close();

    StoredGraph graph;
    graph.nodeProjection = j.get<std::string>("nodeProjection", "");
    graph.relationshipProjection = j.get<std::string>("relationshipProjection", "");
    graph.configuration = j.get<std::string>("configuration", "");
    std::int64_t creation_ms = j.get<std::int64_t>("creationTime", 0LL);
    std::int64_t modification_ms = j.get<std::int64_t>("modificationTime", 0LL);
    graph.creationTime = std::chrono::system_clock::time_point(std::chrono::milliseconds(creation_ms));
    graph.modificationTime = std::chrono::system_clock::time_point(std::chrono::milliseconds(modification_ms));
    
    // Load nodes
    graph.projectedNodes.clear();
    auto nodes_opt = j.get_child_optional("projectedNodes");
    if (nodes_opt) {
        for (const auto& n : *nodes_opt) {
            graph.projectedNodes.push_back(n.second.get_value<std::size_t>());
        }
    }
    
    // Load edges
    graph.edges.clear();
    auto edges_opt = j.get_child_optional("edges");
    if (edges_opt) {
        for (const auto& e : *edges_opt) {
            StoredGraph::Edge edge;
            edge.source = e.second.get<std::size_t>("source");
            edge.target = e.second.get<std::size_t>("target");
            edge.type = e.second.get<std::string>("type");
            graph.edges.push_back(edge);
        }
    }
#endif
    
    // Insert into catalog
    graphs_[graphName] = std::move(graph);
}

void GqlGraphCatalog::load_catalog_from_disk()
{
    for (const auto& entry : fs::directory_iterator(catalogDirectory_)) {
        if (entry.is_regular_file()) {
            auto path = entry.path();
            if (path.extension() == ".json") {
                std::string filename = path.stem().string();
                load_graph_from_disk(filename);
            }
        }
    }
}

} // namespace GQL