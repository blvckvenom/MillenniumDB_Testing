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
    static constexpr uint8_t MINOR_VERSION = 0;
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

    // Configuration
    bool has_node_properties = false;
    bool has_edge_properties = false;
    bool undirected_relationships = false;

    // Query information (for debugging/reference)
    std::string original_query;

    // Property metadata
    std::vector<std::string> node_property_names;
    std::vector<std::string> edge_property_names;

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
