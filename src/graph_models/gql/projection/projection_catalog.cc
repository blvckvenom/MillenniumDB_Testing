#include "projection_catalog.h"

#include <cstring>
#include <filesystem>
#include <stdexcept>

#include "graph_models/gql/projection/index_set.h"
#include "graph_models/gql/projection/native_projection_builder.h"
#include "storage/index/bplus_tree/bpt_leaf_format.h"

namespace fs = std::filesystem;

namespace GQL {

constexpr uint8_t ProjectionCatalog::magic_number[];

namespace {

// Count the number of single-bit ProjectionIndex entries set in the mask
// corresponding to the given IndexSet preset. Used by v1.5 to size the
// leaf_formats byte array both at read time (validation) and at save time
// (defaulting an empty vector to all-BITSET).
//
// Note: the full ProjectionIndex::ALL mask is 14 bits (0x3FFF). Property
// indexes (NODE_KEY_VALUE, KEY_VALUE_NODE, EDGE_KEY_VALUE, KEY_VALUE_EDGE)
// are always counted here because the catalog's index_set byte encodes the
// *preset*, not the runtime-gated property conditionals — matching the
// behavior of project_index_mask_for().
size_t count_materialized_indexes(IndexSet preset) noexcept {
    const auto mask = static_cast<uint32_t>(project_index_mask_for(preset));
    return static_cast<size_t>(__builtin_popcount(mask));
}

}  // namespace

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

    // Read catalog data (common to all v1.x versions)
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

    // Read new fields added in v1.1 (optional labels/properties)
    if (minor_ver >= 1) {
        includes_node_labels = read_uint8(file) != 0;
        includes_edge_labels = read_uint8(file) != 0;
        includes_node_properties = read_uint8(file) != 0;
        includes_edge_properties = read_uint8(file) != 0;

        included_node_properties = read_strvec(file);
        included_edge_properties = read_strvec(file);

        distinct_node_labels = read_uint64(file);
        distinct_edge_labels = read_uint64(file);
    }

    // Read property key mappings
    if (minor_ver >= 3) {
        // v1.3+ format: explicit (key_id, key_name) pairs
        // This correctly preserves key IDs like COUNT_KEY_SYNTHETIC_ID

        // Read node key mappings
        uint32_t node_key_count = read_uint32(file);
        node_keys_str.clear();
        node_keys2id.clear();
        for (uint32_t i = 0; i < node_key_count; ++i) {
            uint64_t key_id = read_uint64(file);
            std::string key_name = read_string(file);
            node_keys2id[key_name] = key_id;
            // Grow vector if needed to accommodate this key_id
            if (key_id >= node_keys_str.size()) {
                node_keys_str.resize(key_id + 1);
            }
            node_keys_str[key_id] = key_name;
        }

        // Read edge key mappings
        uint32_t edge_key_count = read_uint32(file);
        edge_keys_str.clear();
        edge_keys2id.clear();
        for (uint32_t i = 0; i < edge_key_count; ++i) {
            uint64_t key_id = read_uint64(file);
            std::string key_name = read_string(file);
            edge_keys2id[key_name] = key_id;
            // Grow vector if needed to accommodate this key_id
            if (key_id >= edge_keys_str.size()) {
                edge_keys_str.resize(key_id + 1);
            }
            edge_keys_str[key_id] = key_name;
        }
    } else if (minor_ver == 2) {
        // v1.2 format: index-based (DEPRECATED - key IDs may be incorrect)
        // Projections with aggregation properties like _count must be recreated
        node_keys_str = read_strvec(file);
        node_keys2id.clear();
        for (size_t i = 0; i < node_keys_str.size(); ++i) {
            node_keys2id[node_keys_str[i]] = i;
        }

        edge_keys_str = read_strvec(file);
        edge_keys2id.clear();
        for (size_t i = 0; i < edge_keys_str.size(); ++i) {
            edge_keys2id[edge_keys_str[i]] = i;
        }
    }

    // v1.4 field: IndexSet preset byte. For v1.3 and earlier catalogs we
    // default to IndexSet::ALL (full materialization, the behavior shipped
    // before Spec #3). Writing this AFTER the key mappings keeps the v1.4
    // format a strict append to v1.3.
    if (minor_ver >= 4) {
        uint8_t raw = read_uint8(file);
        // We intentionally do NOT validate the byte here — project_index_mask_for()
        // in index_set.h handles the "out-of-range" case with an assert in debug
        // and IndexSet::NONE mask in release, which is the documented drift path.
        index_set = static_cast<IndexSet>(raw);
    } else {
        index_set = IndexSet::ALL;
    }

    // v1.5 field: per-index leaf_format byte array. One byte per materialized
    // index, in canonical ProjectionIndex enum order. For v1.4 and earlier
    // catalogs, populate with all BITSET (1) so pre-Spec-#5 projections read
    // as redundant-bitset-encoded (preserving historical behavior).
    leaf_formats.clear();
    if (minor_ver >= 5) {
        const uint8_t num_format_bytes = read_uint8(file);
        const size_t expected = count_materialized_indexes(index_set);
        if (static_cast<size_t>(num_format_bytes) != expected) {
            throw std::runtime_error(
                "Catalog v1.5: leaf_format array length mismatch (got "
                + std::to_string(num_format_bytes)
                + ", expected " + std::to_string(expected) + ")");
        }
        leaf_formats.reserve(num_format_bytes);
        for (uint8_t i = 0; i < num_format_bytes; ++i) {
            const uint8_t fmt = read_uint8(file);
            if (fmt != static_cast<uint8_t>(BPT::LeafFormat::BITSET) &&
                fmt != static_cast<uint8_t>(BPT::LeafFormat::DELTA_VARINT)) {
                throw std::runtime_error(
                    "Catalog v1.5: invalid leaf_format byte "
                    + std::to_string(fmt) + " at index " + std::to_string(i)
                    + " (expected 1=BITSET or 2=DELTA_VARINT)");
            }
            leaf_formats.push_back(fmt);
        }
    } else {
        // Pre-v1.5 catalog: synthesize an all-BITSET array whose length
        // matches the current materialized-index count. This keeps the
        // in-memory shape consistent for any consumer that iterates
        // leaf_formats without caring whether the on-disk file had the
        // section or not.
        const size_t n = count_materialized_indexes(index_set);
        leaf_formats.assign(n, static_cast<uint8_t>(BPT::LeafFormat::BITSET));
    }

    // v1.6 field: per-projection graphStorage byte. For MINOR < 6 we leave
    // graph_storage at its default (1 = BTREE), preserving the byte-for-byte
    // behavior of every pre-Spec-#8 projection. For MINOR >= 6 the byte must
    // be 1 (BTREE) or 2 (CSR_HYBRID); any other value is an on-disk
    // corruption signal and raises.
    if (minor_ver >= 6) {
        graph_storage = read_uint8(file);
        if (graph_storage != 1 && graph_storage != 2) {
            throw std::runtime_error(
                "Catalog v1.6: invalid graph_storage byte "
                + std::to_string(graph_storage)
                + " (expected 1=BTREE or 2=CSR_HYBRID)");
        }
    }
    // v1.5 and earlier: graph_storage stays at default (1 = BTREE)

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

    // Write catalog data (v1.0 fields)
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

    // Write v1.1 fields (optional labels/properties)
    write_uint8(file, includes_node_labels ? 1 : 0);
    write_uint8(file, includes_edge_labels ? 1 : 0);
    write_uint8(file, includes_node_properties ? 1 : 0);
    write_uint8(file, includes_edge_properties ? 1 : 0);

    write_strvec(file, included_node_properties);
    write_strvec(file, included_edge_properties);

    write_uint64(file, distinct_node_labels);
    write_uint64(file, distinct_edge_labels);

    // Write v1.3 fields (property key mappings with explicit IDs)
    // Write node key mappings as (key_id, key_name) pairs
    write_uint32(file, static_cast<uint32_t>(node_keys2id.size()));
    for (const auto& [name, id] : node_keys2id) {
        write_uint64(file, id);
        write_string(file, name);
    }

    // Write edge key mappings as (key_id, key_name) pairs
    write_uint32(file, static_cast<uint32_t>(edge_keys2id.size()));
    for (const auto& [name, id] : edge_keys2id) {
        write_uint64(file, id);
        write_string(file, name);
    }

    // Write v1.4 field: IndexSet preset byte. Must stay after v1.3 key
    // mappings so v1.3 readers don't attempt to interpret this byte as a
    // uint32 count. v1.3 readers stop reading after the edge key mappings
    // loop, which is why appending here preserves read-side compatibility.
    write_uint8(file, static_cast<uint8_t>(index_set));

    // Write v1.5 field: per-index leaf_format byte array. Length-prefixed
    // (uint8_t) because the full ProjectionIndex enum has 14 single-bit
    // values, well within 0-255. If leaf_formats hasn't been populated by
    // the caller (e.g. a freshly-constructed catalog that only set index_set
    // and the basic metadata), default every slot to BITSET so on-disk and
    // on-read invariants hold. Same defaulting rule is applied by the
    // v1.4-catalog read path so both code paths yield the same in-memory
    // state for pre-Spec-#5 workloads.
    if (leaf_formats.empty()) {
        const size_t n = count_materialized_indexes(index_set);
        leaf_formats.assign(n, static_cast<uint8_t>(BPT::LeafFormat::BITSET));
    }
    // Guard against an accidental out-of-range size. The v1.5 length prefix
    // is a single byte, so we must keep leaf_formats.size() <= 255. The
    // active ProjectionIndex enum tops out at 14 bits, but this guard
    // insulates the write path from any future enum expansion that would
    // silently truncate the count.
    if (leaf_formats.size() > 255) {
        throw std::runtime_error(
            "Catalog v1.5: leaf_formats size exceeds uint8_t limit ("
            + std::to_string(leaf_formats.size()) + " > 255)");
    }
    write_uint8(file, static_cast<uint8_t>(leaf_formats.size()));
    for (uint8_t fmt : leaf_formats) {
        write_uint8(file, fmt);
    }

    // Write v1.6 field: per-projection graphStorage byte. Appended after the
    // v1.5 leaf_formats array so v1.5 readers stop cleanly after the format
    // bytes. The MINOR_VERSION header (written above) is now 6, so any
    // future reader selecting on minor_ver >= 6 will consume this byte.
    // Values are fixed to 1=BTREE (default) or 2=CSR_HYBRID — T8.8 will
    // plumb the value from the graph_project config; T8.6 dispatches on it.
    write_uint8(file, graph_storage);

    // Ensure data is flushed to OS buffer before close
    file.flush();
    file.close();
}

void ProjectionCatalog::print(std::ostream& os) const {
    os << "Projection: " << projection_name << "\n";
    os << "Created: " << creation_timestamp << "\n";
    os << "Nodes: " << node_count << "\n";
    os << "Edges: " << edge_count
       << " (Directed: " << directed_edge_count
       << ", Undirected: " << undirected_edge_count << ")\n";

    // Show v1.1 features if available
    os << "Features:\n";
    os << "  Node Labels: " << (includes_node_labels ? "Yes" : "No");
    if (includes_node_labels && distinct_node_labels > 0) {
        os << " (" << distinct_node_labels << " distinct)";
    }
    os << "\n";

    os << "  Edge Labels: " << (includes_edge_labels ? "Yes" : "No");
    if (includes_edge_labels && distinct_edge_labels > 0) {
        os << " (" << distinct_edge_labels << " distinct)";
    }
    os << "\n";

    os << "  Node Properties: " << (includes_node_properties ? "Yes" : "No");
    if (includes_node_properties && !included_node_properties.empty()) {
        os << " [";
        for (size_t i = 0; i < included_node_properties.size(); ++i) {
            if (i > 0) os << ", ";
            os << included_node_properties[i];
        }
        os << "]";
    }
    os << "\n";

    os << "  Edge Properties: " << (includes_edge_properties ? "Yes" : "No");
    if (includes_edge_properties && !included_edge_properties.empty()) {
        os << " [";
        for (size_t i = 0; i < included_edge_properties.size(); ++i) {
            if (i > 0) os << ", ";
            os << included_edge_properties[i];
        }
        os << "]";
    }
    os << "\n";

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

uint32_t ProjectionCatalog::read_uint32(std::fstream& file) {
    uint32_t res = 0;
    uint8_t buf[4];
    file.read(reinterpret_cast<char*>(buf), sizeof(buf));

    for (int i = 0, shift = 0; i < 4; ++i, shift += 8) {
        res |= static_cast<uint32_t>(buf[i]) << shift;
    }

    if (!file.good()) {
        throw std::runtime_error("Error reading uint32 from catalog");
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

void ProjectionCatalog::write_uint32(std::fstream& file, uint32_t value) {
    uint8_t buf[4];
    for (size_t i = 0, shift = 0; i < sizeof(buf); ++i, shift += 8) {
        buf[i] = (value >> shift) & 0xFF;
    }
    file.write(reinterpret_cast<const char*>(buf), sizeof(buf));
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

void ProjectionCatalog::add_node_key(const std::string& key_name, uint64_t key_id) {
    if (node_keys2id.find(key_name) != node_keys2id.end()) {
        return;  // Already registered
    }
    node_keys2id[key_name] = key_id;
    // Ensure node_keys_str has space for the ID
    if (key_id >= node_keys_str.size()) {
        node_keys_str.resize(key_id + 1);
    }
    node_keys_str[key_id] = key_name;
}

void ProjectionCatalog::add_edge_key(const std::string& key_name, uint64_t key_id) {
    if (edge_keys2id.find(key_name) != edge_keys2id.end()) {
        return;  // Already registered
    }
    edge_keys2id[key_name] = key_id;
    // Ensure edge_keys_str has space for the ID
    if (key_id >= edge_keys_str.size()) {
        edge_keys_str.resize(key_id + 1);
    }
    edge_keys_str[key_id] = key_name;
}

uint64_t ProjectionCatalog::get_node_key_id(const std::string& key_name) const {
    auto it = node_keys2id.find(key_name);
    if (it != node_keys2id.end()) {
        return it->second;
    }
    return 0;  // Not found
}

uint64_t ProjectionCatalog::get_edge_key_id(const std::string& key_name) const {
    auto it = edge_keys2id.find(key_name);
    if (it != edge_keys2id.end()) {
        return it->second;
    }
    return 0;  // Not found
}

} // namespace GQL
