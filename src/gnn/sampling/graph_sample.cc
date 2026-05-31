#include "gnn/sampling/graph_sample.h"

#include <cstring>
#include <stdexcept>
#include <type_traits>

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

// -----------------------------------------------------------------------------
// v2 (legacy) per-element helpers — preserved for backward compatibility
// when reading v1/v2 samples written before Round 2A (2026-05-15).
// -----------------------------------------------------------------------------

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

// -----------------------------------------------------------------------------
// v3 bulk helpers — Round 2A hot-path optimization (2026-05-15)
//
// Format: uint64_t size + (size * sizeof(T)) raw bytes.
// Eliminates per-element istream::read/ostream::write overhead (vtable dispatch
// + sentry construction + state checks) for millions of elements per batch on
// papers100M-scale samples (~6M nodes/batch + edges).
//
// On-disk byte layout is identical to the v2 element-by-element format for
// fixed-width trivially-copyable types (ObjectId == uint64_t, int32_t), so v3
// pages could in principle be read by v2 code — but the version field
// distinguishes them so that v2 readers fail with "unsupported version" rather
// than silently misinterpreting future format changes.
// -----------------------------------------------------------------------------

template <typename T>
void write_bulk_vector(std::ostream& out, const std::vector<T>& vec) {
    static_assert(std::is_trivially_copyable_v<T>,
                  "write_bulk_vector requires trivially copyable T");
    uint64_t size = static_cast<uint64_t>(vec.size());
    out.write(reinterpret_cast<const char*>(&size), sizeof(size));
    if (!out) {
        throw std::runtime_error("write_bulk_vector: size write failed");
    }
    if (size > 0) {
        out.write(reinterpret_cast<const char*>(vec.data()),
                  static_cast<std::streamsize>(size * sizeof(T)));
        if (!out) {
            throw std::runtime_error("write_bulk_vector: data write failed");
        }
    }
}

template <typename T>
std::vector<T> read_bulk_vector(std::istream& in) {
    static_assert(std::is_trivially_copyable_v<T>,
                  "read_bulk_vector requires trivially copyable T");
    uint64_t size = 0;
    in.read(reinterpret_cast<char*>(&size), sizeof(size));
    if (!in) {
        throw std::runtime_error("read_bulk_vector: size read failed");
    }
    std::vector<T> vec(size);
    if (size > 0) {
        in.read(reinterpret_cast<char*>(vec.data()),
                static_cast<std::streamsize>(size * sizeof(T)));
        if (!in) {
            throw std::runtime_error("read_bulk_vector: data read failed");
        }
    }
    return vec;
}

// ObjectId is a class wrapping a single uint64_t; assert trivial copyability
// (also asserted at the bottom of object_id.h) so the bulk read/write is safe.
static_assert(sizeof(ObjectId) == sizeof(uint64_t),
              "ObjectId must be 8 bytes for bulk I/O");
static_assert(std::is_trivially_copyable_v<ObjectId>,
              "ObjectId must be trivially copyable for bulk I/O");

void write_object_id_vector_bulk(std::ostream& out, const std::vector<ObjectId>& vec) {
    write_bulk_vector(out, vec);
}

std::vector<ObjectId> read_object_id_vector_bulk(std::istream& in) {
    return read_bulk_vector<ObjectId>(in);
}

} // anonymous namespace

// =============================================================================
// GraphSample Serialization
// =============================================================================

void GraphSample::serialize(std::ostream& out) const {
    // Header — always written at the current VERSION (v3 bulk format).
    write_value(out, MAGIC);
    write_value(out, VERSION);

    // Identification (v2/v3: no epoch field)
    write_value(out, batch_id);
    write_value(out, static_cast<uint8_t>(split));

    // Nodes per layer — bulk write (Round 2A, 2026-05-15)
    write_value(out, static_cast<uint64_t>(nodes_per_layer.size()));
    for (const auto& layer : nodes_per_layer) {
        write_object_id_vector_bulk(out, layer);
    }

    // Edges per layer — bulk write
    write_value(out, static_cast<uint64_t>(edges_per_layer.size()));
    for (const auto& edges : edges_per_layer) {
        write_bulk_vector(out, edges.src_indices);     // int32_t
        write_bulk_vector(out, edges.dst_indices);     // int32_t
        write_object_id_vector_bulk(out, edges.edge_ids);
    }

    // All unique nodes
    write_object_id_vector_bulk(out, all_unique_nodes);
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
    if (version != VERSION_V2 && version != VERSION_V3 && version != 1) {
        throw std::runtime_error(
            "GraphSample::deserialize: unsupported version " + std::to_string(version)
        );
    }
    // v3 introduced bulk binary I/O. v1/v2 used element-by-element I/O.
    // On-disk byte layout is identical for fixed-width trivially-copyable types
    // (uint64_t / int32_t), so the only difference is throughput.
    const bool use_bulk = (version == VERSION_V3);

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
        sample.nodes_per_layer.push_back(
            use_bulk ? read_object_id_vector_bulk(in) : read_object_id_vector(in)
        );
    }

    // Edges per layer
    uint64_t num_edge_layers = read_value<uint64_t>(in);
    sample.edges_per_layer.reserve(num_edge_layers);
    for (uint64_t i = 0; i < num_edge_layers; ++i) {
        LayerEdges edges;
        edges.src_indices = use_bulk ? read_bulk_vector<int32_t>(in)
                                     : read_int32_vector(in);
        edges.dst_indices = use_bulk ? read_bulk_vector<int32_t>(in)
                                     : read_int32_vector(in);
        edges.edge_ids = use_bulk ? read_object_id_vector_bulk(in)
                                  : read_object_id_vector(in);
        sample.edges_per_layer.push_back(std::move(edges));
    }

    // All unique nodes
    sample.all_unique_nodes = use_bulk ? read_object_id_vector_bulk(in)
                                       : read_object_id_vector(in);

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
    if (version != VERSION_V2 && version != VERSION_V3 && version != 1) {
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
