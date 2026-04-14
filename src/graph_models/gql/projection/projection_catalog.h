#pragma once

#include <cstdint>
#include <fstream>
#include <ostream>
#include <string>
#include <vector>

namespace GQL {

class ProjectionCatalog {
public:
    static constexpr uint8_t MAJOR_VERSION = 1;
    static constexpr uint8_t MINOR_VERSION = 1;  // Bumped for optional labels/properties support
    static constexpr uint8_t magic_number[] = {0x10, 0x0D, 0xEC, 0xAD, 0xE5, 0xDB};
    static constexpr uint8_t MODEL_ID = 255; // Special ID for projections

    ProjectionCatalog(const std::string& projection_dir);
    ~ProjectionCatalog();

    void print(std::ostream& os) const;
    void save();
    void load();

    // Projection metadata
    std::string projection_name;
    uint64_t creation_timestamp = 0;

    // Graph statistics
    uint64_t node_count = 0;
    uint64_t edge_count = 0;
    uint64_t directed_edge_count = 0;
    uint64_t undirected_edge_count = 0;

    // Feature flags (what's included in this projection)
    bool includes_node_labels = false;
    bool includes_edge_labels = false;
    bool includes_node_properties = false;
    bool includes_edge_properties = false;

    // Legacy flags (backward compatibility)
    bool has_node_properties = false;  // Deprecated: use includes_node_properties
    bool has_edge_properties = false;  // Deprecated: use includes_edge_properties
    bool undirected_relationships = false;

    // Property metadata (which specific properties were included)
    std::vector<std::string> included_node_properties;  // Empty = all properties
    std::vector<std::string> included_edge_properties;  // Empty = all properties

    // Legacy property names (backward compatibility)
    std::vector<std::string> node_property_names;  // Deprecated
    std::vector<std::string> edge_property_names;  // Deprecated

    // Statistics
    uint64_t distinct_node_labels = 0;
    uint64_t distinct_edge_labels = 0;

    // Query information (for debugging/reference)
    std::string original_query;

    // Timing information
    uint64_t projection_millis = 0;

private:
    std::string catalog_path;

    // I/O helpers
    uint8_t read_uint8(std::fstream& file);
    uint64_t read_uint64(std::fstream& file);
    std::string read_string(std::fstream& file);
    std::vector<std::string> read_strvec(std::fstream& file);

    void write_uint8(std::fstream& file, uint8_t value);
    void write_uint64(std::fstream& file, uint64_t value);
    void write_string(std::fstream& file, const std::string& str);
    void write_strvec(std::fstream& file, const std::vector<std::string>& vec);
};

} // namespace GQL
