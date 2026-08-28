#include "gnn/sampling/parallel_cursor_merger.h"

#include "gnn/sampling/seekable_edge_iter.h"

namespace mdb::gnn {

struct ParallelCursorMerger::Impl {
    SeekableEdgeIter out_iter;  ///< Iterator over from_to_index (outgoing edges)
    SeekableEdgeIter in_iter;   ///< Iterator over to_from_index (incoming edges)
    bool* interruption;

    // Statistics
    uint64_t total_seeks = 0;
    uint64_t total_edges = 0;

    // Deduplication set (reused across calls to avoid reallocation)
    std::unordered_set<MergedEdge, MergedEdgeHash> dedup_set;

    Impl(BPlusTree<3>& from_to_index, BPlusTree<3>& to_from_index, bool* interruption_)
        : out_iter(from_to_index, interruption_)
        , in_iter(to_from_index, interruption_)
        , interruption(interruption_)
    {}

    /**
     * @brief Collect all outgoing edges from node.
     *
     * Seeks to node in from_to_index and collects edges.
     *
     * @param node_id Source node
     * @param result Vector to append edges to
     * @param limit Maximum edges (0 = unlimited)
     */
    void collect_outgoing(uint64_t node_id, std::vector<MergedEdge>& result, size_t limit) {
        if (!out_iter.seek_from(node_id)) {
            return;  // Index exhausted
        }

        while (!out_iter.exhausted()) {
            EdgeRecord edge = out_iter.current();

            if (edge.from_id != node_id) {
                break;  // Moved to different source
            }

            MergedEdge merged{edge.to_id, edge.edge_id, true};

            // Check for duplicates
            if (dedup_set.insert(merged).second) {
                result.push_back(merged);
                total_edges++;

                if (limit > 0 && result.size() >= limit) {
                    return;  // Limit reached
                }
            }

            if (!out_iter.next_from_current()) {
                break;
            }
        }
    }

    /**
     * @brief Collect all incoming edges to node.
     *
     * Seeks to node in to_from_index and collects edges.
     * Note: to_from_index has key [to_id, from_id, edge_id], so we seek by to_id.
     *
     * @param node_id Target node (seeking as "to" in to_from_index)
     * @param result Vector to append edges to
     * @param limit Maximum edges (0 = unlimited)
     */
    void collect_incoming(uint64_t node_id, std::vector<MergedEdge>& result, size_t limit) {
        if (!in_iter.seek_from(node_id)) {
            return;  // Index exhausted
        }

        while (!in_iter.exhausted()) {
            EdgeRecord edge = in_iter.current();

            // In to_from_index: from_id is actually the "to" node (seeked key)
            // and to_id is the source neighbor
            if (edge.from_id != node_id) {
                break;  // Moved to different target
            }

            // edge.to_id is the actual neighbor (source of incoming edge)
            MergedEdge merged{edge.to_id, edge.edge_id, false};

            // Check for duplicates
            if (dedup_set.insert(merged).second) {
                result.push_back(merged);
                total_edges++;

                if (limit > 0 && result.size() >= limit) {
                    return;  // Limit reached
                }
            }

            if (!in_iter.next_from_current()) {
                break;
            }
        }
    }

    std::vector<MergedEdge> get_edges_for_node(uint64_t node_id, size_t limit) {
        if (interruption && *interruption) {
            return {};
        }

        std::vector<MergedEdge> result;
        dedup_set.clear();

        // Collect from both directions
        // Process outgoing first, then incoming (order doesn't affect correctness)
        collect_outgoing(node_id, result, limit);

        // Only continue if we haven't hit the limit
        if (limit == 0 || result.size() < limit) {
            collect_incoming(node_id, result, limit);
        }

        // Update seek statistics
        total_seeks += 2;  // One seek per direction

        return result;
    }
};

// =============================================================================
// Public Interface
// =============================================================================

ParallelCursorMerger::ParallelCursorMerger(
    BPlusTree<3>& from_to_index,
    BPlusTree<3>& to_from_index,
    bool* interruption
)
    : impl_(std::make_unique<Impl>(from_to_index, to_from_index, interruption))
{}

ParallelCursorMerger::~ParallelCursorMerger() = default;

ParallelCursorMerger::ParallelCursorMerger(ParallelCursorMerger&&) noexcept = default;
ParallelCursorMerger& ParallelCursorMerger::operator=(ParallelCursorMerger&&) noexcept = default;

std::vector<MergedEdge> ParallelCursorMerger::get_all_edges(uint64_t node_id) {
    return impl_->get_edges_for_node(node_id, 0);  // 0 = no limit
}

std::vector<MergedEdge> ParallelCursorMerger::get_all_edges(uint64_t node_id, size_t limit) {
    return impl_->get_edges_for_node(node_id, limit);
}

std::pair<uint64_t, uint64_t> ParallelCursorMerger::get_statistics() const {
    return {impl_->total_seeks, impl_->total_edges};
}

void ParallelCursorMerger::reset_statistics() {
    impl_->total_seeks = 0;
    impl_->total_edges = 0;
}

} // namespace mdb::gnn
