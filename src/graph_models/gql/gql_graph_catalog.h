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

#pragma once

#include <string>
#include <unordered_map>
#include <vector>
#include <optional>
#include <chrono>
#include <cstdint>

// Forward declarations to avoid including heavy graph headers here. The actual
// Value and Map types should be defined in the query layer (e.g., query/value.h).
namespace GQL {
class Value;
class Map;
}

namespace GQL {

/**
 * GqlGraphCatalog manages in‑memory graph projections identified by name. Each
 * projection is stored both in memory and on disk so that the catalog
 * persists across process restarts. A projected graph consists of a set of
 * nodes with consecutive internal identifiers, a set of relationships, and
 * associated metadata such as projection arguments and creation times.
 *
 * The API is intentionally minimal: callers can project a graph, list one or
 * more projections, and drop a projection. Additional fields (e.g., schema
 * with orientation, degree distribution) can be computed lazily from the
 * stored graph structure or returned as empty values if not implemented.
 */
class GqlGraphCatalog {
public:
    /// Structure holding summary information about a projected graph. This
    /// corresponds closely to the YIELD columns of GdsGraphProject. The
    /// nodeProjection and relationshipProjection fields are stored as strings
    /// representing the user‑provided projection arguments; these may be
    /// serialized versions of Value objects or simple labels.
    struct ProjectResult {
        std::string graphName;
        std::string nodeProjection;
        std::size_t nodeCount;
        std::string relationshipProjection;
        std::size_t relationshipCount;
        std::int64_t projectMillis;
        std::string configuration;
    };

    /// Structure representing one entry in the GdsGraphList result. Many
    /// fields are optional because they may not be computed until needed.
    struct GraphListEntry {
        std::string graphName;
        std::string database;
        std::string databaseLocation;
        std::string configuration;
        std::size_t nodeCount;
        std::size_t relationshipCount;
        std::string schema;
        std::string schemaWithOrientation;
        std::string degreeDistribution;
        double density;
        std::chrono::system_clock::time_point creationTime;
        std::chrono::system_clock::time_point modificationTime;
        std::size_t sizeInBytes;
        std::string memoryUsage;
    };

    /// Structure returned by GdsGraphList. If a specific graphName is
    /// provided to list(), the entries vector will contain at most one
    /// element.
    struct ListResult {
        std::vector<GraphListEntry> entries;
    };

    /// Structure returned by GdsGraphDrop. The droppedGraph contains the
    /// metadata of the graph that was removed. If failIfMissing is false and
    /// the graph does not exist, droppedGraph will be empty.
    struct DropResult {
        std::optional<GraphListEntry> droppedGraph;
    };

    /// Construct a catalog. The catalogDirectory specifies where on disk
    /// projected graphs will be stored. If the directory does not exist it
    /// will be created. All existing graphs in the directory will be loaded
    /// into memory during construction.
    explicit GqlGraphCatalog(const std::string& catalogDirectory);

    /// Project a new graph with the given name. The nodeProjArg and
    /// relProjArg parameters describe the node and relationship projections
    /// respectively. They are accepted as generic Value objects; callers are
    /// responsible for constructing them according to the projection semantics.
    /// The configuration map may include concurrency options or property
    /// projections. The method returns summary information for the created
    /// projection. If a projection with the same name already exists it is
    /// overwritten.
    ProjectResult project(const std::string& graphName,
                          const Value& nodeProjArg,
                          const Value& relProjArg,
                          const Map& configuration);

    /// List one or all projected graphs. If graphName is specified, only the
    /// matching entry will be returned (if it exists). Otherwise all graphs
    /// currently stored in the catalog will be listed. Unimplemented fields
    /// such as schemaWithOrientation and degreeDistribution will be left empty.
    ListResult list(const std::optional<std::string>& graphName) const;

    /// Drop a projected graph. If failIfMissing is true, the method throws
    /// std::runtime_error when the graph does not exist. Otherwise it returns
    /// an empty DropResult. Upon successful removal the graph is deleted from
    /// both memory and disk.
    DropResult drop(const std::string& graphName,
                    bool failIfMissing = true,
                    const std::string& dbName = "",
                    const std::string& username = "");

private:
    /// Internal representation of a stored graph. Nodes are represented by
    /// consecutive identifiers starting at zero. The original node identifiers
    /// are mapped to these internal IDs by originalToProjectedId. Relationships
    /// are stored as triples (source internal id, target internal id, type).
    struct StoredGraph {
        std::vector<std::size_t> projectedNodes;
        std::unordered_map<std::size_t, std::size_t> originalToProjectedId;
        struct Edge {
            std::size_t source;
            std::size_t target;
            std::string type;
        };
        std::vector<Edge> edges;
        // Metadata
        std::string nodeProjection;
        std::string relationshipProjection;
        std::chrono::system_clock::time_point creationTime;
        std::chrono::system_clock::time_point modificationTime;
        std::string configuration;
    };

    /// Helper to persist a graph to disk as a JSON file under the catalog
    /// directory. The filename is derived from the graph name. Any existing
    /// file for this graph will be overwritten.
    void save_graph_to_disk(const std::string& graphName, const StoredGraph& graph) const;

    /// Helper to load a graph from disk into memory. If the file does not
    /// exist nothing is loaded.
    void load_graph_from_disk(const std::string& graphName);

    /// Load all graphs present in the catalog directory at construction time.
    void load_catalog_from_disk();

    /// In-memory store of projected graphs keyed by their name.
    std::unordered_map<std::string, StoredGraph> graphs_;

    /// Directory on disk where graph files are stored.
    std::string catalogDirectory_;
};

} // namespace GQL