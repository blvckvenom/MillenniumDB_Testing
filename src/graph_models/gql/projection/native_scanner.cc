#include "native_scanner.h"

#include <iostream>
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

    std::cerr << "[NativeScanner] scan_label_node: Scanning for label_id=0x" << std::hex << search_label_id << std::dec << std::endl;

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

    std::cerr << "[NativeScanner] scan_label_node: Range iterator created" << std::endl;

    // Iterate over matching records
    uint64_t count = 0;
    const Record<2>* record;
    while ((record = iter.next()) != nullptr) {
        // Record format: {label_id, node_id}
        // Extract node_id from second field
        ObjectId node_id((*record)[1]);
        std::cerr << "[NativeScanner] scan_label_node: Found node_id=0x" << std::hex << node_id.id << std::dec << std::endl;
        callback(node_id);
        count++;
    }

    std::cerr << "[NativeScanner] scan_label_node: Total nodes found: " << count << std::endl;
    return count;
}

uint64_t NativeScanner::scan_label_edge(
    ObjectId type_id,
    std::function<void(ObjectId)> callback
) {
    // The label_edge B+Tree stores full ObjectIds WITH type masks
    uint64_t search_type_id = type_id.id;

    std::cerr << "[NativeScanner] scan_label_edge: Scanning for type_id=0x" << std::hex << search_type_id << std::dec << std::endl;

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
        if (count < 5) {  // Only log first 5 to avoid spam
            std::cerr << "[NativeScanner] scan_label_edge: Found edge_id=0x" << std::hex << edge_id.id << std::dec << std::endl;
        }
        callback(edge_id);
        count++;
    }

    std::cerr << "[NativeScanner] scan_label_edge: Total edges found: " << count << std::endl;
    return count;
}

uint64_t NativeScanner::scan_label_edge_with_endpoints(
    ObjectId type_id,
    std::function<void(ObjectId, ObjectId, ObjectId)> callback
) {
    // Scan label_edge index to get all edges with this type
    uint64_t search_type_id = type_id.id;

    std::cerr << "[NativeScanner] scan_label_edge_with_endpoints: Scanning for type_id=0x"
              << std::hex << search_type_id << std::dec << std::endl;

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
                    // Edge not found - skip
                    std::cerr << "[NativeScanner] WARNING: Undirected edge 0x" << std::hex
                              << edge_id.id << " not found in edge_n1_n2" << std::dec << std::endl;
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
                    std::cerr << "[NativeScanner] WARNING: Undirected edge 0x" << std::hex
                              << edge_id.id << " not found" << std::dec << std::endl;
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
                    // Edge not found - skip
                    std::cerr << "[NativeScanner] WARNING: Directed edge 0x" << std::hex
                              << edge_id.id << " not found in edge_from_to" << std::dec << std::endl;
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
                    std::cerr << "[NativeScanner] WARNING: Directed edge 0x" << std::hex
                              << edge_id.id << " not found" << std::dec << std::endl;
                    continue;
                }
            }
        }

        // Call callback with (edge_id, from_node, to_node)
        callback(edge_id, from_node, to_node);
        count++;
    }

    std::cerr << "[NativeScanner] scan_label_edge_with_endpoints: Total edges found: " << count << std::endl;
    return count;
}

std::pair<ObjectId, ObjectId> NativeScanner::get_edge_endpoints(ObjectId edge_id) {
    std::cerr << "[NativeScanner] get_edge_endpoints: Looking up edge_id=0x" << std::hex << edge_id.id << std::dec << std::endl;

    // Detect edge type by examining the ObjectId mask
    uint64_t edge_type = edge_id.id & ObjectId::SUB_TYPE_MASK;
    bool is_undirected = (edge_type == ObjectId::MASK_UNDIRECTED_EDGE);

    std::cerr << "[NativeScanner] Edge type mask: 0x" << std::hex << edge_type << std::dec
              << (is_undirected ? " (UNDIRECTED)" : " (DIRECTED)") << std::endl;

    if (is_undirected) {
        // ===== UNDIRECTED EDGE (0xe4) =====
        // Fast path: Use edge_n1_n2 index if available (O(log n) lookup)
        // Index structure: {edge_id, n1, n2}
        if (edge_n1_n2_index) {
            std::cerr << "[NativeScanner] Using fast path (edge_n1_n2 index)" << std::endl;

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
                std::cerr << "[NativeScanner] Found endpoints: n1=0x" << std::hex << n1.id
                          << " n2=0x" << n2.id << std::dec << std::endl;
                return {n1, n2};
            }

            std::cerr << "[NativeScanner] Edge not found in edge_n1_n2 index, falling back to slow path" << std::endl;
        } else {
            std::cerr << "[NativeScanner] Using slow path (scanning n1_n2_edge)" << std::endl;
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
                std::cerr << "[NativeScanner] Found in slow path: n1=0x" << std::hex << n1.id
                          << " n2=0x" << n2.id << std::dec << std::endl;
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
            std::cerr << "[NativeScanner] Using fast path (edge_from_to index)" << std::endl;

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
                std::cerr << "[NativeScanner] Found endpoints: from=0x" << std::hex << from_node.id
                          << " to=0x" << to_node.id << std::dec << std::endl;
                return {from_node, to_node};
            }

            std::cerr << "[NativeScanner] Edge not found in edge_from_to index, falling back to slow path" << std::endl;
        } else {
            std::cerr << "[NativeScanner] Using slow path (scanning from_to_edge)" << std::endl;
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
                std::cerr << "[NativeScanner] Found in slow path: from=0x" << std::hex << from_node.id
                          << " to=0x" << to_node.id << std::dec << std::endl;
                return {from_node, to_node};
            }
        }

        throw std::runtime_error(
            "NativeScanner::get_edge_endpoints: Directed edge not found: " + std::to_string(edge_id.id)
        );
    }
}

} // namespace GQL
