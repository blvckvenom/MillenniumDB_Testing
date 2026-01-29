#include "native_scanner.h"

#include <stdexcept>

#include "storage/index/bplus_tree/bplus_tree.h"
#include "storage/index/record.h"

namespace GQL {

NativeScanner::NativeScanner(
    BPlusTree<2>* label_node_idx,
    BPlusTree<2>* label_edge_idx,
    BPlusTree<3>* from_to_edge_idx,
    BPlusTree<3>* edge_from_to_idx,
    BPlusTree<3>* n1_n2_edge_idx,
    BPlusTree<3>* edge_n1_n2_idx
)
    : label_node_index(label_node_idx)
    , label_edge_index(label_edge_idx)
    , from_to_edge_index(from_to_edge_idx)
    , edge_from_to_index(edge_from_to_idx)
    , n1_n2_edge_index(n1_n2_edge_idx)
    , edge_n1_n2_index(edge_n1_n2_idx)
{
    if (!label_node_index || !label_edge_index || !from_to_edge_index || !n1_n2_edge_index) {
        throw std::runtime_error("NativeScanner: null index pointer provided");
    }
    // edge_from_to_index and edge_n1_n2_index are optional, can be nullptr
}

NativeScanner::~NativeScanner() {
    // Non-owning pointers, no cleanup needed
}

uint64_t NativeScanner::scan_label_node(
    ObjectId label_id,
    std::function<void(ObjectId)> callback
) {
    // The label_node B+Tree stores full ObjectIds WITH type masks
    // We need to use the full label_id as-is
    uint64_t search_label_id = label_id.id;

    // Debug logging removed for performance

    // Define range: all records where first key = search_label_id
    Record<2> min_record;
    min_record[0] = search_label_id;
    min_record[1] = 0;

    Record<2> max_record;
    max_record[0] = search_label_id;
    max_record[1] = UINT64_MAX;

    // Create range iterator with interruption support
    bool interruption_requested = false;
    auto iter = label_node_index->get_range(&interruption_requested, min_record, max_record);

    // Iterate over matching records
    uint64_t count = 0;
    const Record<2>* record;
    while ((record = iter.next()) != nullptr) {
        // Record format: {label_id, node_id}
        // Extract node_id from second field
        ObjectId node_id((*record)[1]);
        callback(node_id);
        count++;
    }

    return count;
}

uint64_t NativeScanner::scan_label_edge(
    ObjectId type_id,
    std::function<void(ObjectId)> callback
) {
    // The label_edge B+Tree stores full ObjectIds WITH type masks
    uint64_t search_type_id = type_id.id;

    // Define range: all records where first key = search_type_id
    Record<2> min_record;
    min_record[0] = search_type_id;
    min_record[1] = 0;

    Record<2> max_record;
    max_record[0] = search_type_id;
    max_record[1] = UINT64_MAX;

    // Create range iterator with interruption support
    bool interruption_requested = false;
    auto iter = label_edge_index->get_range(&interruption_requested, min_record, max_record);

    // Iterate over matching records
    uint64_t count = 0;
    const Record<2>* record;
    while ((record = iter.next()) != nullptr) {
        // Record format: {type_id, edge_id}
        // Extract edge_id from second field
        ObjectId edge_id((*record)[1]);
        callback(edge_id);
        count++;
    }

    return count;
}

uint64_t NativeScanner::scan_label_edge_with_endpoints(
    ObjectId type_id,
    std::function<void(ObjectId, ObjectId, ObjectId)> callback
) {
    // Scan label_edge index to get all edges with this type
    uint64_t search_type_id = type_id.id;

    // Define range: all records where first key = search_type_id
    Record<2> min_record;
    min_record[0] = search_type_id;
    min_record[1] = 0;

    Record<2> max_record;
    max_record[0] = search_type_id;
    max_record[1] = UINT64_MAX;

    // Create range iterator with interruption support
    bool interruption_requested = false;
    auto iter = label_edge_index->get_range(&interruption_requested, min_record, max_record);

    // Iterate over matching records
    uint64_t count = 0;
    const Record<2>* record;
    while ((record = iter.next()) != nullptr) {
        // Record format: {type_id, edge_id}
        ObjectId edge_id((*record)[1]);

        // Detect edge type by examining the ObjectId mask
        uint64_t edge_type = edge_id.id & ObjectId::SUB_TYPE_MASK;
        bool is_undirected = (edge_type == ObjectId::MASK_UNDIRECTED_EDGE);

        ObjectId from_node, to_node;

        if (is_undirected) {
            // ===== UNDIRECTED EDGE (0xe4) =====
            // Fast path: Use edge_n1_n2 index if available (O(log n) lookup)
            if (edge_n1_n2_index) {
                Record<3> min_rec;
                min_rec[0] = edge_id.id;
                min_rec[1] = 0;
                min_rec[2] = 0;

                Record<3> max_rec;
                max_rec[0] = edge_id.id;
                max_rec[1] = UINT64_MAX;
                max_rec[2] = UINT64_MAX;

                bool interrupt = false;
                auto endpoint_iter = edge_n1_n2_index->get_range(&interrupt, min_rec, max_rec);

                const Record<3>* endpoint_rec = endpoint_iter.next();
                if (endpoint_rec != nullptr) {
                    from_node = ObjectId((*endpoint_rec)[1]);
                    to_node = ObjectId((*endpoint_rec)[2]);
                } else {
                    // Edge not found in index - skip
                    continue;
                }
            } else {
                // Slow path: Scan n1_n2_edge index (should not happen in practice)
                Record<3> min_rec;
                min_rec[0] = 0;
                min_rec[1] = 0;
                min_rec[2] = 0;

                Record<3> max_rec;
                max_rec[0] = UINT64_MAX;
                max_rec[1] = UINT64_MAX;
                max_rec[2] = UINT64_MAX;

                bool interrupt = false;
                auto endpoint_iter = n1_n2_edge_index->get_range(&interrupt, min_rec, max_rec);

                bool found = false;
                const Record<3>* endpoint_rec;
                while ((endpoint_rec = endpoint_iter.next()) != nullptr) {
                    if ((*endpoint_rec)[2] == edge_id.id) {
                        from_node = ObjectId((*endpoint_rec)[0]);
                        to_node = ObjectId((*endpoint_rec)[1]);
                        found = true;
                        break;
                    }
                }
                if (!found) {
                    // Edge not found in index - skip
                    continue;
                }
            }
        } else {
            // ===== DIRECTED EDGE (0xe0) =====
            // Fast path: Use edge_from_to index if available (O(log n) lookup)
            if (edge_from_to_index) {
                Record<3> min_rec;
                min_rec[0] = edge_id.id;
                min_rec[1] = 0;
                min_rec[2] = 0;

                Record<3> max_rec;
                max_rec[0] = edge_id.id;
                max_rec[1] = UINT64_MAX;
                max_rec[2] = UINT64_MAX;

                bool interrupt = false;
                auto endpoint_iter = edge_from_to_index->get_range(&interrupt, min_rec, max_rec);

                const Record<3>* endpoint_rec = endpoint_iter.next();
                if (endpoint_rec != nullptr) {
                    from_node = ObjectId((*endpoint_rec)[1]);
                    to_node = ObjectId((*endpoint_rec)[2]);
                } else {
                    // Edge not found in index - skip
                    continue;
                }
            } else {
                // Slow path: Scan from_to_edge index (should not happen in practice)
                Record<3> min_rec;
                min_rec[0] = 0;
                min_rec[1] = 0;
                min_rec[2] = 0;

                Record<3> max_rec;
                max_rec[0] = UINT64_MAX;
                max_rec[1] = UINT64_MAX;
                max_rec[2] = UINT64_MAX;

                bool interrupt = false;
                auto endpoint_iter = from_to_edge_index->get_range(&interrupt, min_rec, max_rec);

                bool found = false;
                const Record<3>* endpoint_rec;
                while ((endpoint_rec = endpoint_iter.next()) != nullptr) {
                    if ((*endpoint_rec)[2] == edge_id.id) {
                        from_node = ObjectId((*endpoint_rec)[0]);
                        to_node = ObjectId((*endpoint_rec)[1]);
                        found = true;
                        break;
                    }
                }
                if (!found) {
                    // Edge not found in index - skip
                    continue;
                }
            }
        }

        // Call callback with (edge_id, from_node, to_node)
        callback(edge_id, from_node, to_node);
        count++;
    }

    return count;
}

std::pair<ObjectId, ObjectId> NativeScanner::get_edge_endpoints(ObjectId edge_id) {
    // Detect edge type by examining the ObjectId mask
    uint64_t edge_type = edge_id.id & ObjectId::SUB_TYPE_MASK;
    bool is_undirected = (edge_type == ObjectId::MASK_UNDIRECTED_EDGE);

    if (is_undirected) {
        // ===== UNDIRECTED EDGE (0xe4) =====
        // Fast path: Use edge_n1_n2 index if available (O(log n) lookup)
        // Index structure: {edge_id, n1, n2}
        if (edge_n1_n2_index) {

            Record<3> min_record;
            min_record[0] = edge_id.id;
            min_record[1] = 0;
            min_record[2] = 0;

            Record<3> max_record;
            max_record[0] = edge_id.id;
            max_record[1] = UINT64_MAX;
            max_record[2] = UINT64_MAX;

            bool interruption_requested = false;
            auto iter = edge_n1_n2_index->get_range(&interruption_requested, min_record, max_record);

            const Record<3>* record = iter.next();
            if (record != nullptr) {
                // Record format: {edge_id, n1, n2}
                ObjectId n1((*record)[1]);
                ObjectId n2((*record)[2]);
                return {n1, n2};
            }
            // Fall through to slow path if not found
        }

        // Slow path: Scan n1_n2_edge index (O(E) worst case)
        // Index structure: {n1, n2, edge_id}
        Record<3> min_record;
        min_record[0] = 0;
        min_record[1] = 0;
        min_record[2] = 0;

        Record<3> max_record;
        max_record[0] = UINT64_MAX;
        max_record[1] = UINT64_MAX;
        max_record[2] = UINT64_MAX;

        bool interruption_requested = false;
        auto iter = n1_n2_edge_index->get_range(&interruption_requested, min_record, max_record);

        // Linear scan to find edge
        const Record<3>* record;
        while ((record = iter.next()) != nullptr) {
            if ((*record)[2] == edge_id.id) {
                // Found it! Extract endpoints
                ObjectId n1((*record)[0]);
                ObjectId n2((*record)[1]);
                return {n1, n2};
            }
        }

        throw std::runtime_error(
            "NativeScanner::get_edge_endpoints: Undirected edge not found: " + std::to_string(edge_id.id)
        );

    } else {
        // ===== DIRECTED EDGE (0xe0) =====
        // Fast path: Use edge_from_to index if available (O(log n) lookup)
        // Index structure: {edge_id, from, to}
        if (edge_from_to_index) {

            Record<3> min_record;
            min_record[0] = edge_id.id;
            min_record[1] = 0;
            min_record[2] = 0;

            Record<3> max_record;
            max_record[0] = edge_id.id;
            max_record[1] = UINT64_MAX;
            max_record[2] = UINT64_MAX;

            bool interruption_requested = false;
            auto iter = edge_from_to_index->get_range(&interruption_requested, min_record, max_record);

            const Record<3>* record = iter.next();
            if (record != nullptr) {
                // Record format: {edge_id, from, to}
                ObjectId from_node((*record)[1]);
                ObjectId to_node((*record)[2]);
                return {from_node, to_node};
            }
            // Fall through to slow path if not found
        }

        // Slow path: Scan from_to_edge index (O(E) worst case)
        // Index structure: {from, to, edge_id}
        Record<3> min_record;
        min_record[0] = 0;
        min_record[1] = 0;
        min_record[2] = 0;

        Record<3> max_record;
        max_record[0] = UINT64_MAX;
        max_record[1] = UINT64_MAX;
        max_record[2] = UINT64_MAX;

        bool interruption_requested = false;
        auto iter = from_to_edge_index->get_range(&interruption_requested, min_record, max_record);

        // Linear scan to find edge
        const Record<3>* record;
        while ((record = iter.next()) != nullptr) {
            if ((*record)[2] == edge_id.id) {
                // Found it! Extract endpoints
                ObjectId from_node((*record)[0]);
                ObjectId to_node((*record)[1]);
                return {from_node, to_node};
            }
        }

        throw std::runtime_error(
            "NativeScanner::get_edge_endpoints: Directed edge not found: " + std::to_string(edge_id.id)
        );
    }
}

uint64_t NativeScanner::count_edges_by_type(ObjectId type_id) {
    // Quick count of edges with this type by scanning label_edge index
    uint64_t search_type_id = type_id.id;

    // Define range: all records where first key = search_type_id
    Record<2> min_record;
    min_record[0] = search_type_id;
    min_record[1] = 0;

    Record<2> max_record;
    max_record[0] = search_type_id;
    max_record[1] = UINT64_MAX;

    // Create range iterator with interruption support
    bool interruption_requested = false;
    auto iter = label_edge_index->get_range(&interruption_requested, min_record, max_record);

    // Count matching records
    uint64_t count = 0;
    while (iter.next() != nullptr) {
        count++;
    }

    return count;
}

} // namespace GQL
