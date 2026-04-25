#pragma once

// topology_snapshot_from_leaf.h — build CSR topology sidecars from a
// just-written BPT .leaf file via mmap, instead of re-walking the BPT
// through its iterator (the original post-hoc path at
// native_projection_builder.cc:2453-2526).
//
// Motivation: the post-hoc path for building the topology CSR sidecars
// (topology_fwd.csr / topology_rev.csr via mmap) scanned each edge index
// twice (pass 1 = degree histogram, pass 2 = edge stream) through the
// full BPT iterator. On ogbn-products this doubled the build time
// (214 s -> 454 s, +112 %). Since the sorted records have already landed
// sequentially in the `.leaf` file, a page-aligned mmap read is
// ~2-4 x faster than navigating the B+Tree directory — we skip every
// buffer-pool fetch, every node-level branch, every page pin/unpin.
//
// Contract:
//   - Produces **byte-identical** `topology_{fwd,rev}.csr` output vs the
//     BPT-iterator path. The record stream seen by
//     TopologySnapshotWriter::append_edge is the same sequence of
//     (src_idx, dst_full, edge_id_full) triples the iterator emitted,
//     in the same order — sorted by key[0] asc (the BPT contract) and
//     within a key[0] the same tie-breaking order.
//   - `.leaf` must have been written by `BPTLeafWriter<3>::process_block`
//     (the only writer used by the projection pipeline today). The page
//     layout is fixed at:
//       [0]  uint32_t value_count
//       [4]  uint32_t next_leaf         (0 on last page; else page_number)
//       [8]  3 bytes   bitset           (= 0x000000 — no compression is
//                                         emitted by `build_index_streaming`)
//       [11] records[] packed uint64_t  (3 × uint64 per record = 24 B)
//       [tail] zero-padding to Page::SIZE = 4096 B
//   - `make_empty()` writes one fully-zeroed page (value_count == 0); this
//     helper treats that as "no edges" and emits an empty-CSR.
//   - Atomic writer commit semantics are provided by TopologySnapshotWriter
//     (unchanged); this helper only feeds it and calls finalize().
//
// Why two passes over mmap instead of one:
//   TopologySnapshotWriter::operator() requires the per-node degree
//   histogram **at construction time** (used to build ROW_PTR up front).
//   A single pass would need an in-memory stash of all edge records —
//   O(M) RAM, which at papers100M's 1.6 B edges would eat ~38 GB.
//   Instead we mmap and walk twice; both passes share the kernel page
//   cache, so the second pass is effectively free after the first
//   populates it.

#include <cstdint>
#include <filesystem>

#include "graph_models/gql/projection/topology_snapshot_writer.h"

namespace GQL::Projection {

/// Build one direction of the CSR sidecar by reading the corresponding
/// B+Tree `.leaf` file via mmap.
///
/// @param projection_dir   Directory containing the .leaf and where the
///                         `.csr` will be written.
/// @param dir              FORWARD emits `topology_fwd.csr` from
///                         `from_to_edge.leaf`; REVERSE emits
///                         `topology_rev.csr` from `to_from_edge.leaf`.
/// @param num_nodes        Node count as reported by the projection
///                         catalog (sizes ROW_PTR to N+1).
/// @param include_edge_ids When true, emits the EDGE_IDS section;
///                         matches the legacy `include_edge_ids=true`
///                         choice of `build_one_topology_snapshot_`.
///
/// Throws `std::runtime_error` on I/O failure. The writer's atomic
/// rename contract is preserved: on throw, the `.csr.tmp` is cleaned up
/// by the writer's destructor.
void build_topology_snapshot_from_leaf(
    const std::filesystem::path&   projection_dir,
    TopologySnapshotWriter::Direction dir,
    uint64_t                        num_nodes,
    bool                            include_edge_ids = true);

}  // namespace GQL::Projection
