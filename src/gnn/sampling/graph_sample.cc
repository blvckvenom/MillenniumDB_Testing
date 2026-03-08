#include "gnn/sampling/graph_sample.h"

#include <cstring>
#include <stdexcept>

namespace mdb::gnn {

// =============================================================================
// Helper Functions for Binary I/O
// =============================================================================

namespace {

template<typename T>
void write_value(std::ostream& out, const T& value) {
    out.write(reinterpret_cast<const char*>(&value), sizeof(T));
    if (!out) {
        throw std::runtime_error("GraphSample::serialize: write failed");
    }
}

template<typename T>
T read_value(std::istream& in) {
    T value;
    in.read(reinterpret_cast<char*>(&value), sizeof(T));
    if (!in) {
        throw std::runtime_error("GraphSample::deserialize: read failed");
    }
    return value;
}

void write_object_id_vector(std::ostream& out, const std::vector<ObjectId>& vec) {
    write_value(out, static_cast<uint64_t>(vec.size()));
    for (const auto& oid : vec) {
        write_value(out, oid.id);
    }
}

std::vector<ObjectId> read_object_id_vector(std::istream& in) {
    uint64_t size = read_value<uint64_t>(in);
    std::vector<ObjectId> vec;
    vec.reserve(size);
    for (uint64_t i = 0; i < size; ++i) {
        vec.emplace_back(read_value<uint64_t>(in));
    }
    return vec;
}

void write_int32_vector(std::ostream& out, const std::vector<int32_t>& vec) {
    write_value(out, static_cast<uint64_t>(vec.size()));
    for (int32_t val : vec) {
        write_value(out, val);
    }
}

std::vector<int32_t> read_int32_vector(std::istream& in) {
    uint64_t size = read_value<uint64_t>(in);
    std::vector<int32_t> vec;
    vec.reserve(size);
    for (uint64_t i = 0; i < size; ++i) {
        vec.push_back(read_value<int32_t>(in));
    }
    return vec;
}

} // anonymous namespace

// =============================================================================
// GraphSample Serialization
// =============================================================================

void GraphSample::serialize(std::ostream& out) const {
    // Header
    write_value(out, MAGIC);
    write_value(out, VERSION);

    // Identification (v2: no epoch field)
    write_value(out, batch_id);
    write_value(out, static_cast<uint8_t>(split));

    // Nodes per layer
    write_value(out, static_cast<uint64_t>(nodes_per_layer.size()));
    for (const auto& layer : nodes_per_layer) {
        write_object_id_vector(out, layer);
    }

    // Edges per layer
    write_value(out, static_cast<uint64_t>(edges_per_layer.size()));
    for (const auto& edges : edges_per_layer) {
        write_int32_vector(out, edges.src_indices);
        write_int32_vector(out, edges.dst_indices);
        write_object_id_vector(out, edges.edge_ids);
    }

    // All unique nodes
    write_object_id_vector(out, all_unique_nodes);
}

GraphSample GraphSample::deserialize(std::istream& in) {
    // Header validation
    uint32_t magic = read_value<uint32_t>(in);
    if (magic != MAGIC) {
        throw std::runtime_error(
            "GraphSample::deserialize: invalid magic number (expected 0x" +
            std::to_string(MAGIC) + ", got 0x" + std::to_string(magic) + ")"
        );
    }

    uint32_t version = read_value<uint32_t>(in);
    if (version != VERSION && version != 1) {
        throw std::runtime_error(
            "GraphSample::deserialize: unsupported version " + std::to_string(version)
        );
    }

    GraphSample sample;

    // Identification
    sample.batch_id = read_value<uint64_t>(in);

    // v1 compatibility: read and discard legacy epoch field
    if (version == 1) {
        [[maybe_unused]] uint64_t legacy_epoch = read_value<uint64_t>(in);
    }

    sample.split = static_cast<SplitType>(read_value<uint8_t>(in));

    // Nodes per layer
    uint64_t num_node_layers = read_value<uint64_t>(in);
    sample.nodes_per_layer.reserve(num_node_layers);
    for (uint64_t i = 0; i < num_node_layers; ++i) {
        sample.nodes_per_layer.push_back(read_object_id_vector(in));
    }

    // Edges per layer
    uint64_t num_edge_layers = read_value<uint64_t>(in);
    sample.edges_per_layer.reserve(num_edge_layers);
    for (uint64_t i = 0; i < num_edge_layers; ++i) {
        LayerEdges edges;
        edges.src_indices = read_int32_vector(in);
        edges.dst_indices = read_int32_vector(in);
        edges.edge_ids = read_object_id_vector(in);
        sample.edges_per_layer.push_back(std::move(edges));
    }

    // All unique nodes
    sample.all_unique_nodes = read_object_id_vector(in);

    return sample;
}

SplitType GraphSample::read_split(std::istream& in) {
    // Header validation
    uint32_t magic = read_value<uint32_t>(in);
    if (magic != MAGIC) {
        throw std::runtime_error(
            "GraphSample::read_split: invalid magic number (expected 0x" +
            std::to_string(MAGIC) + ", got 0x" + std::to_string(magic) + ")"
        );
    }

    uint32_t version = read_value<uint32_t>(in);
    if (version != VERSION && version != 1) {
        throw std::runtime_error(
            "GraphSample::read_split: unsupported version " + std::to_string(version)
        );
    }

    // Skip batch_id (8 bytes)
    read_value<uint64_t>(in);

    // v1 compatibility: skip legacy epoch field (8 bytes)
    if (version == 1) {
        read_value<uint64_t>(in);
    }

    // Read and return the split field
    return static_cast<SplitType>(read_value<uint8_t>(in));
}

} // namespace mdb::gnn
