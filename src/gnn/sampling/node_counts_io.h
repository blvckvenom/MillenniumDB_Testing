#pragma once

// node_counts_io — read/write helpers for `<projection_dir>/node_counts.bin`.
//
// `node_counts.bin` is the warm-start artifact consumed by
// `TopologyFrequencyProfiler::compute_from_node_counts_` (the frequency-profiling
// phase of the Four-Level Topology Store: L1 RAM hash / L2 compact uint32 CSR /
// L3 mmap sidecar / L4 direct B+Tree) and produced by:
//   (a) `OfflineSamplingEngine::run()` at the END of a sample build
//       (accumulates real access counts during sampling), and
//   (b) `TopologyWalkProfiler` as a cheap cold-start profiler (random walks over
//       the mmap-backed topology CSR sidecar files topology_{fwd,rev}.csr that
//       provide O(1) neighbor slices) when no `node_counts.bin` exists yet.
//
// Both (a) and (b) write the SAME on-disk format so the reader path in
// `TopologyFrequencyProfiler::compute_from_node_counts_` doesn't care
// which producer wrote it.
//
// Format (written by `persist()` below):
//
//   Offset  Size              Field
//   0       8B                Magic "NODECNT0"
//   8       8B  uint64_t      num_nodes
//   16      8B  uint64_t      direction_bitmask  (1=NATURAL, 2=REVERSE, 3=UNDIRECTED)
//   24      num_nodes×8B      counts[num_nodes] (uint64_t each)
//
// Atomic write: temp file → fsync → rename → fsync(parent dir). Mirrors
// `src/gnn/output/model_checkpoint.cc::save_full` so a crash mid-write
// never leaves a corrupted `node_counts.bin`.

#include <cstdint>
#include <filesystem>
#include <vector>

#include "gnn/projection/edge_orientation.h"

namespace mdb::gnn::node_counts_io {

/// Persist a per-node count vector to `<projection_dir>/node_counts.bin`
/// using the format documented above. No-op if `projection_dir` is empty
/// or `counts` is empty. Warnings on I/O failure go to std::cerr; no
/// exception is thrown — call sites are expected to be best-effort.
///
/// The `orientation` parameter selects the direction_bitmask written
/// to disk. The reader (`TopologyFrequencyProfiler`) treats this as
/// informational and accepts cross-direction counts (popularity is
/// direction-agnostic), so callers MAY normalize to UNDIRECTED.
void persist(const std::filesystem::path&  projection_dir,
             const std::vector<uint64_t>&  counts,
             EdgeOrientation               orientation);

}  // namespace mdb::gnn::node_counts_io
