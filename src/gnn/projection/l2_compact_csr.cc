#include "gnn/projection/l2_compact_csr.h"

#include <cstdint>
#include <cstddef>
#include <limits>
#include <stdexcept>
#include <string>

#include "gnn/projection/topology_frequency_profiler.h"

namespace mdb::gnn {

L2CompactCsr::L2CompactCsr(std::size_t num_l2_nodes_hint) {
    if (num_l2_nodes_hint > 0) {
        row_ptr_.reserve(num_l2_nodes_hint + 1);
        node_to_l2_idx_.reserve(num_l2_nodes_hint);
        degrees_.reserve(num_l2_nodes_hint);
    }
}

void L2CompactCsr::add_node(uint64_t                     src_node_id,
                            const std::vector<AdjEntry>& neighbors)
{
    if (frozen_) {
        throw std::logic_error(
            "L2CompactCsr::add_node called after freeze() — cache is immutable");
    }

    // Reject duplicates. The orchestrator must dedupe before adding;
    // this is a safety net rather than the primary line of defence.
    if (node_to_l2_idx_.count(src_node_id) != 0) {
        throw std::invalid_argument(
            "L2CompactCsr::add_node duplicate src_node_id: "
            + std::to_string(src_node_id));
    }

    const uint32_t l2_idx = static_cast<uint32_t>(node_to_l2_idx_.size());
    node_to_l2_idx_.emplace(src_node_id, l2_idx);
    degrees_.push_back(static_cast<uint32_t>(neighbors.size()));

    // Append destination row indexes to the flat col_idx_ array. The
    // ObjectId payload occupies the lower 56 bits; we drop only the
    // 8-bit type tag here. uint32 truncation safety is enforced at
    // freeze() time.
    //
    // NOTE (2026-05-21 fix): the previous version called
    //   col_idx_.reserve(col_idx_.size() + neighbors.size());
    // before each push_back loop. That defeats vector's geometric
    // growth strategy: every reserve(N+k) with N+k > capacity forces
    // a fresh allocation + copy of all N existing elements, making
    // add_node O(N) and the populate phase O(E_l2²) overall. On
    // papers100M (~210M tier-2 edges) this caused a >19 h hang in
    // populate_via_sidecar. Plain push_back amortizes to O(1) via
    // vector's exponential capacity growth, restoring linear total.
    for (const auto& nb : neighbors) {
        col_idx_.push_back(static_cast<uint32_t>(nb.node_id));
    }
}

void L2CompactCsr::freeze() {
    if (frozen_) return;

    // Validate the uint32 col_idx invariant. Each col_idx_ entry is a
    // truncated ObjectId payload — for projections with > 4 billion
    // nodes this would silently alias. Fail loudly instead.
    if (col_idx_.size() > std::numeric_limits<uint32_t>::max()) {
        throw std::overflow_error(
            "L2CompactCsr::freeze edge count exceeds uint32 — "
            "graph too large for L2 layout");
    }

    // Build row_ptr_ prefix sum: row_ptr_[i] = sum of degrees of L2
    // rows 0..i-1; row_ptr_[N] = total edges.
    const std::size_t n = degrees_.size();
    row_ptr_.assign(n + 1, 0);
    for (std::size_t i = 0; i < n; ++i) {
        row_ptr_[i + 1] = row_ptr_[i] + degrees_[i];
    }

    frozen_ = true;
}

L2CompactCsr::ColIdxSpan L2CompactCsr::get(uint64_t src_node_id) const {
    // Symmetric to add_node()'s post-freeze throw: a pre-freeze get()
    // would silently miss every src (row_ptr_ is empty until freeze
    // builds the prefix sum), masking orchestrator bugs. Fail loud
    // instead — matches the project's "fail loud" discipline.
    if (!frozen_) {
        throw std::logic_error(
            "L2CompactCsr::get called before freeze() — "
            "call freeze() to lock the structure first");
    }
    auto it = node_to_l2_idx_.find(src_node_id);
    if (it == node_to_l2_idx_.end()) {
        return ColIdxSpan{ nullptr, 0 };
    }
    const uint32_t l2_idx = it->second;
    const uint64_t start = row_ptr_[l2_idx];
    const uint64_t end   = row_ptr_[static_cast<std::size_t>(l2_idx) + 1];
    return ColIdxSpan{ col_idx_.data() + start,
                       static_cast<std::size_t>(end - start) };
}

std::size_t L2CompactCsr::total_bytes() const {
    // Per Spec #13 Phase 1 contract:
    //   bytes(node) = kL2NodeFixedOverhead + kL2PerEdgeBytes * degree
    // (Upper bound — see header note on the edge_ids decision.)
    std::size_t bytes = 0;
    for (const auto& deg : degrees_) {
        bytes += kL2NodeFixedOverhead;
        bytes += kL2PerEdgeBytes * static_cast<std::size_t>(deg);
    }
    return bytes;
}

}  // namespace mdb::gnn
