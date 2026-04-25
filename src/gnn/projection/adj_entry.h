#pragma once

#include <cstdint>

namespace mdb::gnn {

/**
 * @brief Adjacency entry for cached topology — node id + edge id.
 *
 * Layout matches the existing private `TopologyAccessor::AdjEntry` and
 * `EmbeddingWriter::AdjEntry` structs (commits `25a663ba`, `6521cc21`)
 * but is hoisted into a public header so Spec #13 Phase 2's
 * `L1HashCache`, `L2CompactCsr`, and `FourLevelTopologyStore::Neighbors`
 * can share a single canonical type without taking a dependency on the
 * full `topology_accessor.h` translation unit.
 *
 * The two `uint64_t` payloads are the **raw** ObjectId bit patterns
 * (i.e. without the 8-bit type tag stripped, just as
 * `TopologyAccessor::Impl` stores them in its hash cache). Callers
 * that want a typed `ObjectId` must rebuild it with `ObjectId(node_id)`
 * the same way the BPT path does today.
 *
 * Phase 3 (T13.8) will deduplicate the older private copies in
 * `topology_accessor.cc` and `embedding_writer.cc` against this header.
 * Phase 2 keeps both forms alive to stay surgical.
 */
struct AdjEntry {
    uint64_t node_id;
    uint64_t edge_id;
};

}  // namespace mdb::gnn
