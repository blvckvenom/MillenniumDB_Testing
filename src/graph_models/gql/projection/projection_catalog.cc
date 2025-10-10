#include "projection_catalog.h"

#include <cstring>
#include <filesystem>
#include <stdexcept>

namespace fs = std::filesystem;

namespace GQL {

constexpr uint8_t ProjectionCatalog::magic_number[];

ProjectionCatalog::ProjectionCatalog(const std::string& projection_dir)
    : catalog_path(projection_dir + "/catalog.dat")
{
    // Try to load existing catalog
    if (fs::exists(catalog_path)) {
        load();
    }
    // If file doesn't exist, we're creating a new catalog (fields stay at default values)
}

ProjectionCatalog::~ProjectionCatalog() {
    // Destructor - no automatic save, user must call save() explicitly
}

void ProjectionCatalog::load() {
    std::fstream file(catalog_path, std::ios::in | std::ios::binary);
    if (!file.is_open()) {
        throw std::runtime_error("Could not open catalog file: " + catalog_path);
    }

    // Read and verify magic number
    uint8_t buf[6];
    file.read(reinterpret_cast<char*>(buf), sizeof(magic_number));
    if (!file.good() || memcmp(buf, magic_number, sizeof(magic_number)) != 0) {
        throw std::runtime_error("Invalid catalog magic number");
    }

    // Read MDB version (3 bytes)
    read_uint8(file); // major
    read_uint8(file); // minor
    read_uint8(file); // patch

    // Read and verify model ID
    uint8_t model_id = read_uint8(file);
    if (model_id != MODEL_ID) {
        throw std::runtime_error("Invalid model ID in projection catalog");
    }

    // Read and verify catalog version
    uint8_t major_ver = read_uint8(file);
    uint8_t minor_ver = read_uint8(file);
    if (major_ver != MAJOR_VERSION) {
        throw std::runtime_error("Incompatible catalog major version");
    }
    // Minor version differences are acceptable (forward compatible)

    // Read catalog data
    projection_name = read_string(file);
    creation_timestamp = read_uint64(file);

    node_count = read_uint64(file);
    edge_count = read_uint64(file);
    directed_edge_count = read_uint64(file);
    undirected_edge_count = read_uint64(file);

    has_node_properties = read_uint8(file) != 0;
    has_edge_properties = read_uint8(file) != 0;
    undirected_relationships = read_uint8(file) != 0;

    original_query = read_string(file);
    projection_millis = read_uint64(file);

    node_property_names = read_strvec(file);
    edge_property_names = read_strvec(file);

    file.close();
}

void ProjectionCatalog::save() {
    std::fstream file(catalog_path, std::ios::out | std::ios::binary | std::ios::trunc);
    if (!file.is_open()) {
        throw std::runtime_error("Could not create catalog file: " + catalog_path);
    }

    // Write magic number
    for (size_t i = 0; i < sizeof(magic_number); ++i) {
        write_uint8(file, magic_number[i]);
    }

    // Write MDB version (hardcoded for now, could import from System)
    write_uint8(file, 1); // major
    write_uint8(file, 0); // minor
    write_uint8(file, 0); // patch

    // Write model ID and catalog version
    write_uint8(file, MODEL_ID);
    write_uint8(file, MAJOR_VERSION);
    write_uint8(file, MINOR_VERSION);

    // Write catalog data
    write_string(file, projection_name);
    write_uint64(file, creation_timestamp);

    write_uint64(file, node_count);
    write_uint64(file, edge_count);
    write_uint64(file, directed_edge_count);
    write_uint64(file, undirected_edge_count);

    write_uint8(file, has_node_properties ? 1 : 0);
    write_uint8(file, has_edge_properties ? 1 : 0);
    write_uint8(file, undirected_relationships ? 1 : 0);

    write_string(file, original_query);
    write_uint64(file, projection_millis);

    write_strvec(file, node_property_names);
    write_strvec(file, edge_property_names);

    file.close();
}

void ProjectionCatalog::print(std::ostream& os) const {
    os << "Projection: " << projection_name << "\n";
    os << "Created: " << creation_timestamp << "\n";
    os << "Nodes: " << node_count << "\n";
    os << "Edges: " << edge_count
       << " (Directed: " << directed_edge_count
       << ", Undirected: " << undirected_edge_count << ")\n";
    os << "Properties: Nodes=" << (has_node_properties ? "Yes" : "No")
       << ", Edges=" << (has_edge_properties ? "Yes" : "No") << "\n";
    os << "Creation time: " << projection_millis << "ms\n";
}

// I/O helper implementations
uint8_t ProjectionCatalog::read_uint8(std::fstream& file) {
    auto res = static_cast<uint8_t>(file.get());
    if (!file.good()) {
        throw std::runtime_error("Error reading uint8 from catalog");
    }
    return res;
}

uint64_t ProjectionCatalog::read_uint64(std::fstream& file) {
    uint64_t res = 0;
    uint8_t buf[8];
    file.read(reinterpret_cast<char*>(buf), sizeof(buf));

    for (int i = 0, shift = 0; i < 8; ++i, shift += 8) {
        res |= static_cast<uint64_t>(buf[i]) << shift;
    }

    if (!file.good()) {
        throw std::runtime_error("Error reading uint64 from catalog");
    }
    return res;
}

std::string ProjectionCatalog::read_string(std::fstream& file) {
    // Read length as uint32
    uint32_t len = 0;
    uint8_t buf[4];
    file.read(reinterpret_cast<char*>(buf), sizeof(buf));
    for (int i = 0, shift = 0; i < 4; ++i, shift += 8) {
        len |= static_cast<uint32_t>(buf[i]) << shift;
    }

    // Read string data
    char* str_buf = new char[len];
    file.read(str_buf, len);
    std::string res(str_buf, len);
    delete[] str_buf;
    return res;
}

std::vector<std::string> ProjectionCatalog::read_strvec(std::fstream& file) {
    // Read count as uint32
    uint32_t count = 0;
    uint8_t buf[4];
    file.read(reinterpret_cast<char*>(buf), sizeof(buf));
    for (int i = 0, shift = 0; i < 4; ++i, shift += 8) {
        count |= static_cast<uint32_t>(buf[i]) << shift;
    }

    std::vector<std::string> res;
    for (uint32_t i = 0; i < count; ++i) {
        res.push_back(read_string(file));
    }
    return res;
}

void ProjectionCatalog::write_uint8(std::fstream& file, uint8_t value) {
    file.put(static_cast<char>(value));
}

void ProjectionCatalog::write_uint64(std::fstream& file, uint64_t value) {
    uint8_t buf[8];
    for (size_t i = 0, shift = 0; i < sizeof(buf); ++i, shift += 8) {
        buf[i] = (value >> shift) & 0xFF;
    }
    file.write(reinterpret_cast<const char*>(buf), sizeof(buf));
}

void ProjectionCatalog::write_string(std::fstream& file, const std::string& str) {
    // Write length as uint32
    uint32_t len = str.size();
    uint8_t buf[4];
    for (size_t i = 0, shift = 0; i < sizeof(buf); ++i, shift += 8) {
        buf[i] = (len >> shift) & 0xFF;
    }
    file.write(reinterpret_cast<const char*>(buf), sizeof(buf));

    // Write string data
    file.write(str.c_str(), str.size());
}

void ProjectionCatalog::write_strvec(std::fstream& file, const std::vector<std::string>& vec) {
    // Write count as uint32
    uint32_t count = vec.size();
    uint8_t buf[4];
    for (size_t i = 0, shift = 0; i < sizeof(buf); ++i, shift += 8) {
        buf[i] = (count >> shift) & 0xFF;
    }
    file.write(reinterpret_cast<const char*>(buf), sizeof(buf));

    // Write each string
    for (const auto& str : vec) {
        write_string(file, str);
    }
}

} // namespace GQL
