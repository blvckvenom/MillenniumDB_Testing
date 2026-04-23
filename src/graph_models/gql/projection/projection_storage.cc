#include "projection_storage.h"

#include <algorithm>
#include <bitset>
#include <chrono>
#include <cstring>
#include <filesystem>
#include <stdexcept>

// Parallel execution for std::sort (requires TBB on GCC/Clang)
#ifdef HAS_TBB
#include <execution>
#endif

#include "external_record_sort.h"
#include "graph_models/gql/projection/index_set.h"
#include "graph_models/gql/projection/native_projection_builder.h"
#include "graph_models/gql/projection/sorter_dispatch.h"
#include "projection_catalog.h"
#include "storage/index/bplus_tree/bplus_tree.h"
#include "storage/index/bplus_tree/bpt_mem_import.h"
#include "storage/index/record.h"
#include "storage/page/page.h"

namespace GQL {

// Helper function to initialize an empty BPlusTree
template<std::size_t N>
static void init_empty_bptree(const std::string& base_name) {
    BPTLeafWriter<N> leaf_writer(base_name + ".leaf");
    leaf_writer.make_empty();

    BPTDirWriter<N> dir_writer(base_name + ".dir");
    // dir_writer automatically creates a root page when destroyed
}

ProjectionStorage::ProjectionStorage(const std::string& projection_dir_, const std::string& db_folder)
    : projection_dir(projection_dir_)
{
    // Calculate relative path from db_folder
    // E.g., if projection_dir = "test_db/projections/test_projection" and db_folder = "test_db"
    // then rel_dir = "projections/test_projection"
    if (projection_dir.find(db_folder) == 0) {
        rel_dir = projection_dir.substr(db_folder.length());
        // Remove leading slash if present
        if (!rel_dir.empty() && rel_dir[0] == '/') {
            rel_dir = rel_dir.substr(1);
        }
    } else {
        // Fallback: use full path
        rel_dir = projection_dir;
    }

    // Extract projection name from path (last component)
    size_t last_slash = projection_dir.find_last_of("/\\");
    if (last_slash != std::string::npos) {
        projection_name = projection_dir.substr(last_slash + 1);
    } else {
        projection_name = projection_dir;
    }

    // Pre-allocate for better performance
    collected_nodes_.reserve(INITIAL_CAPACITY);
    node_batch.reserve(BATCH_SIZE);
    edge_batch.reserve(BATCH_SIZE);

    // Initialize Bloom filter for memory-efficient edge deduplication
    // Using default expected edges with 1% false positive rate
    edge_bloom_filter_ = std::make_unique<BloomFilter>(DEFAULT_EXPECTED_EDGES, BLOOM_FILTER_FPR);

    // Initialize streaming record buffers (memory-bounded, spill to disk)
    initialize_streaming_buffers();
}

ProjectionStorage::ProjectionStorage(const std::string& projection_dir_,
                                     const std::string& db_folder,
                                     const std::string& projection_name_)
    : projection_dir(projection_dir_), projection_name(projection_name_)
{
    // Calculate relative path from db_folder
    if (projection_dir.find(db_folder) == 0) {
        rel_dir = projection_dir.substr(db_folder.length());
        if (!rel_dir.empty() && rel_dir[0] == '/') {
            rel_dir = rel_dir.substr(1);
        }
    } else {
        rel_dir = projection_dir;
    }

    // Pre-allocate for better performance
    collected_nodes_.reserve(INITIAL_CAPACITY);
    node_batch.reserve(BATCH_SIZE);
    edge_batch.reserve(BATCH_SIZE);

    // Initialize Bloom filter for memory-efficient edge deduplication
    edge_bloom_filter_ = std::make_unique<BloomFilter>(DEFAULT_EXPECTED_EDGES, BLOOM_FILTER_FPR);

    // Initialize streaming record buffers (memory-bounded, spill to disk)
    initialize_streaming_buffers();
}

ProjectionStorage::ProjectionStorage(const std::string& projection_dir_,
                                     const std::string& db_folder,
                                     const std::string& projection_name_,
                                     const Features& features_)
    : projection_dir(projection_dir_),
      projection_name(projection_name_),
      features(features_)
{
    // Calculate relative path from db_folder
    if (projection_dir.find(db_folder) == 0) {
        rel_dir = projection_dir.substr(db_folder.length());
        if (!rel_dir.empty() && rel_dir[0] == '/') {
            rel_dir = rel_dir.substr(1);
        }
    } else {
        rel_dir = projection_dir;
    }

    // Pre-allocate for better performance
    collected_nodes_.reserve(INITIAL_CAPACITY);
    node_batch.reserve(BATCH_SIZE);
    edge_batch.reserve(BATCH_SIZE);

    // Initialize Bloom filter for memory-efficient edge deduplication
    edge_bloom_filter_ = std::make_unique<BloomFilter>(DEFAULT_EXPECTED_EDGES, BLOOM_FILTER_FPR);

    // Initialize streaming record buffers (memory-bounded, spill to disk)
    initialize_streaming_buffers();
}

ProjectionStorage::~ProjectionStorage() {
    // Destructors are implicitly noexcept — a throw from flush() during
    // unwinding (e.g. builder aborted after an error and the projection
    // directory was rolled back) would call std::terminate. Swallow any
    // flush failure here; the caller's rollback handles cleanup.
    try {
        flush();
    } catch (...) {
        // Best-effort cleanup.
    }
}

void ProjectionStorage::init() {
    // BULK IMPORT MODE: Don't create B+tree files here.
    // B+tree indexes will be created during flush() via build_all_indexes_bulk().
    // This completely bypasses the buffer pool for large projections.
    //
    // The directory should already exist (created by NativeProjectionBuilder).
    // We just need to ensure it exists for safety.
    std::filesystem::create_directories(projection_dir);
}

void ProjectionStorage::open() {
    // Open existing BPlusTree objects (using relative paths for file_manager)
    // Do NOT call init_empty_bptree - the files already exist!

    // Load catalog to restore statistics and metadata
    std::filesystem::path proj_path(projection_dir);
    if (std::filesystem::exists(proj_path / "catalog.dat")) {
        ProjectionCatalog catalog(projection_dir);
        catalog.load();

        // Restore statistics from catalog
        node_count = catalog.node_count;
        edge_count = catalog.edge_count;
        directed_edge_count = catalog.directed_edge_count;
        undirected_edge_count = catalog.undirected_edge_count;

        // Restore IndexSet preset so the query-layer diagnostic (T3.9) can
        // name the preset this projection was built under when a missing
        // index is accessed. Older catalogs (pre-v1.4) default to ALL via
        // ProjectionCatalog::load(), so this is safe for all versions.
        requested_index_set = catalog.index_set;
    }

    // Open required indexes (always present)
    nodes_index = std::make_unique<BPlusTree<1>>(rel_dir + "/nodes");
    from_to_edge_index = std::make_unique<BPlusTree<3>>(rel_dir + "/from_to_edge");
    to_from_edge_index = std::make_unique<BPlusTree<3>>(rel_dir + "/to_from_edge");
    edge_direction_index = std::make_unique<BPlusTree<2>>(rel_dir + "/edge_direction");

    // Open edge-first indexes if they exist (added in Phase 13)
    std::filesystem::path proj_path_check(projection_dir);
    if (std::filesystem::exists(proj_path_check / "edge_from_to.leaf")) {
        edge_from_to_index = std::make_unique<BPlusTree<3>>(rel_dir + "/edge_from_to");
    }
    if (std::filesystem::exists(proj_path_check / "edge_n1_n2.leaf")) {
        edge_n1_n2_index = std::make_unique<BPlusTree<3>>(rel_dir + "/edge_n1_n2");
    }

    // Open optional label indexes if they exist
    if (std::filesystem::exists(proj_path / "node_label.leaf")) {
        node_label_index = std::make_unique<BPlusTree<2>>(rel_dir + "/node_label");
        features.include_node_labels = true;
        // Also open auxiliary index if it exists
        if (std::filesystem::exists(proj_path / "label_node.leaf")) {
            label_node_index = std::make_unique<BPlusTree<2>>(rel_dir + "/label_node");
        }
    }

    if (std::filesystem::exists(proj_path / "edge_label.leaf")) {
        edge_label_index = std::make_unique<BPlusTree<2>>(rel_dir + "/edge_label");
        features.include_edge_labels = true;
        // Also open auxiliary index if it exists
        if (std::filesystem::exists(proj_path / "label_edge.leaf")) {
            label_edge_index = std::make_unique<BPlusTree<2>>(rel_dir + "/label_edge");
        }
    }

    // Open optional property indexes if they exist
    if (std::filesystem::exists(proj_path / "node_key_value.leaf")) {
        node_key_value_index = std::make_unique<BPlusTree<3>>(rel_dir + "/node_key_value");
        features.include_node_properties = true;
        // Also open auxiliary index if it exists
        if (std::filesystem::exists(proj_path / "key_value_node.leaf")) {
            key_value_node_index = std::make_unique<BPlusTree<3>>(rel_dir + "/key_value_node");
        }
    }

    // Edge property indexes
    if (std::filesystem::exists(proj_path / "edge_key_value.leaf")) {
        edge_key_value_index = std::make_unique<BPlusTree<3>>(rel_dir + "/edge_key_value");
        features.include_edge_properties = true;
        // Also open auxiliary index if it exists
        if (std::filesystem::exists(proj_path / "key_value_edge.leaf")) {
            key_value_edge_index = std::make_unique<BPlusTree<3>>(rel_dir + "/key_value_edge");
        }
    }
}

void ProjectionStorage::ensure_node_property_indexes() {
    std::filesystem::path proj_path(projection_dir);

    if (!node_key_value_index) {
        std::string base = rel_dir + "/node_key_value";
        { BPTLeafWriter<3> lw(base + ".leaf"); lw.make_empty(); }
        { BPTDirWriter<3>  dw(base + ".dir"); }
        node_key_value_index = std::make_unique<BPlusTree<3>>(base);
        features.include_node_properties = true;
    }
    if (!key_value_node_index) {
        std::string base = rel_dir + "/key_value_node";
        { BPTLeafWriter<3> lw(base + ".leaf"); lw.make_empty(); }
        { BPTDirWriter<3>  dw(base + ".dir"); }
        key_value_node_index = std::make_unique<BPlusTree<3>>(base);
    }
}

void ProjectionStorage::add_node(const ProjectedNode& node) {
    // Append unconditionally. finalize_node_scan() collapses duplicates with
    // std::sort + std::unique once the node scan completes, trading a 48 B/entry
    // hash set (previous impl) for an 8 B/entry sorted vector. Callers that
    // truly require per-insert dedup semantics must use has_node() before
    // add_node(); existing flows (single- and multi-label scan) tolerate
    // duplicates because node scan callbacks are the only producer.
    collected_nodes_.push_back(node.node_id.id);
    collected_nodes_sorted_ = false;  // appended unsorted → invalidates invariant

    node_batch.push_back(node);

    // Flush batch if it reaches threshold
    if (node_batch.size() >= BATCH_SIZE) {
        flush_node_batch();
    }
}

void ProjectionStorage::add_edge(const ProjectedEdge& edge, bool skip_bloom_check) {
    // MEMORY-EFFICIENT DEDUPLICATION: Use Bloom filter instead of hash set.
    // This reduces memory from O(n) to O(1) with configurable false positive rate.
    //
    // False positives are acceptable because:
    // 1. They only cause a small percentage of legitimate edges to be skipped
    // 2. Final correctness is guaranteed by std::unique() during bulk index build
    //
    // Memory savings: ~4 GB for 123M edges (hash set) → ~128 MB (Bloom filter)
    //
    // NOTE: skip_bloom_check=true bypasses the Bloom filter for paths that already
    // guarantee uniqueness (e.g., streaming aggregation which deduplicates via sorting).
    // This eliminates false positives for those paths, achieving 100% accuracy.

    if (!skip_bloom_check) {
        // Check Bloom filter for probable duplicates
        if (edge_bloom_filter_->probably_contains_edge(edge.from_node.id, edge.to_node.id, edge.edge_id.id)) {
            return;  // Probably already seen (may be false positive, but std::unique handles it)
        }

        // Add to Bloom filter for future duplicate detection
        edge_bloom_filter_->add_edge(edge.from_node.id, edge.to_node.id, edge.edge_id.id);
    }

    edge_batch.push_back(edge);

    // Flush batch to collection vectors if threshold reached
    if (edge_batch.size() >= BATCH_SIZE) {
        flush_edge_batch();
        // NOTE: Don't clear Bloom filter - we need to track ALL edges for duplicate detection
    }
}

void ProjectionStorage::add_node_label(ObjectId node_id, ObjectId label_id) {
    // Only collect if label indexing is enabled
    if (!features.include_node_labels) {
        return;
    }

    // STREAMING BUFFER: Collect records with automatic disk spill
    // Primary index: {node_id, label_id}
    Record<2> node_label_record;
    node_label_record[0] = node_id.id;
    node_label_record[1] = label_id.id;
    node_label_records_buffer_->push_back(node_label_record);

    // Auxiliary index: {label_id, node_id} (for efficient label->nodes queries)
    Record<2> label_node_record;
    label_node_record[0] = label_id.id;
    label_node_record[1] = node_id.id;
    label_node_records_buffer_->push_back(label_node_record);
}

void ProjectionStorage::add_edge_label(ObjectId edge_id, ObjectId label_id) {
    // Only collect if label indexing is enabled
    if (!features.include_edge_labels) {
        return;
    }

    // STREAMING BUFFER: Collect records with automatic disk spill
    // Primary index: {edge_id, label_id}
    Record<2> edge_label_record;
    edge_label_record[0] = edge_id.id;
    edge_label_record[1] = label_id.id;
    edge_label_records_buffer_->push_back(edge_label_record);

    // Auxiliary index: {label_id, edge_id} (for efficient label->edges queries)
    Record<2> label_edge_record;
    label_edge_record[0] = label_id.id;
    label_edge_record[1] = edge_id.id;
    label_edge_records_buffer_->push_back(label_edge_record);
}

void ProjectionStorage::add_node_property(ObjectId node_id, ObjectId key_id, ObjectId value_id) {
    // Only collect if property indexing is enabled
    if (!features.include_node_properties) {
        return;
    }

    // STREAMING BUFFER: Collect records with automatic disk spill
    // Primary index: {node_id, key_id, value_id}
    Record<3> node_prop_record;
    node_prop_record[0] = node_id.id;
    node_prop_record[1] = key_id.id;
    node_prop_record[2] = value_id.id;
    node_key_value_records_buffer_->push_back(node_prop_record);

    // Auxiliary index: {key_id, value_id, node_id} (for efficient property->nodes queries)
    Record<3> key_value_node_record;
    key_value_node_record[0] = key_id.id;
    key_value_node_record[1] = value_id.id;
    key_value_node_record[2] = node_id.id;
    key_value_node_records_buffer_->push_back(key_value_node_record);
}

void ProjectionStorage::add_edge_property(ObjectId edge_id, ObjectId key_id, ObjectId value_id) {
    // Only collect if property indexing is enabled
    if (!features.include_edge_properties) {
        return;
    }

    // STREAMING BUFFER: Collect records with automatic disk spill
    // Primary index: {edge_id, key_id, value_id}
    Record<3> edge_prop_record;
    edge_prop_record[0] = edge_id.id;
    edge_prop_record[1] = key_id.id;
    edge_prop_record[2] = value_id.id;
    edge_key_value_records_buffer_->push_back(edge_prop_record);

    // Auxiliary index: {key_id, value_id, edge_id} (for efficient property->edges queries)
    Record<3> key_value_edge_record;
    key_value_edge_record[0] = key_id.id;
    key_value_edge_record[1] = value_id.id;
    key_value_edge_record[2] = edge_id.id;
    key_value_edge_records_buffer_->push_back(key_value_edge_record);
}

bool ProjectionStorage::has_node(ObjectId node_id) const {
    // In-memory scan-phase check against the sorted-vector dedup tracker.
    // Expected path: finalize_node_scan() was called at scan→scan transition,
    // so we can use O(log N) binary search. The pre-finalize fallback is
    // defensive (e.g. unit tests that query mid-scan) and correctness-
    // preserving.
    if (collected_nodes_sorted_) {
        if (std::binary_search(collected_nodes_.begin(), collected_nodes_.end(),
                               node_id.id)) {
            return true;
        }
    } else if (std::find(collected_nodes_.begin(), collected_nodes_.end(),
                         node_id.id) != collected_nodes_.end()) {
        return true;
    }

    // Fall back to B+tree check if index exists (for query phase after flush)
    if (!nodes_index) {
        return false;
    }

    Record<1> search_record;
    search_record[0] = node_id.id;

    // Use range query to check existence
    bool interruption_requested = false;
    auto iter = nodes_index->get_range(&interruption_requested, search_record, search_record);
    return iter.next() != nullptr;
}

bool ProjectionStorage::has_edge(ObjectId from, ObjectId to) const {
    if (!from_to_edge_index) {
        return false;
    }

    Record<3> min_record;
    min_record[0] = from.id;
    min_record[1] = to.id;
    min_record[2] = 0;

    Record<3> max_record;
    max_record[0] = from.id;
    max_record[1] = to.id;
    max_record[2] = UINT64_MAX;

    bool interruption_requested = false;
    auto iter = from_to_edge_index->get_range(&interruption_requested, min_record, max_record);
    return iter.next() != nullptr;
}

void ProjectionStorage::drain_pending_batches() {
    flush_node_batch();
    flush_edge_batch();
}

void ProjectionStorage::flush() {
    // Flush any pending batched writes to streaming buffers
    flush_node_batch();
    flush_edge_batch();

    // STREAMING BULK IMPORT: Build all B+tree indexes at once using direct file writes.
    // This completely bypasses the buffer pool, enabling projections of arbitrary size.
    // Only build if we have records to write (avoid overwriting existing indexes on re-flush).
    bool has_records = (node_records_buffer_ && node_records_buffer_->size() > 0) ||
                       (from_to_records_buffer_ && from_to_records_buffer_->size() > 0) ||
                       (to_from_records_buffer_ && to_from_records_buffer_->size() > 0);

    if (has_records) {
        build_all_indexes_bulk();
    }

    // Save catalog with projection metadata
    save_catalog();
}

void ProjectionStorage::flush_node_batch() {
    if (node_batch.empty()) {
        return;
    }

    // STREAMING BUFFER: Collect records with automatic disk spill.
    // NO B+tree insertions here - all indexes are built in bulk during finalize().
    // Memory bounded: spills to disk when threshold exceeded.

    for (const auto& node : node_batch) {
        uint64_t node_id_val = node.node_id.id;

        // Collect node record for bulk building
        Record<1> node_record;
        node_record[0] = node_id_val;
        node_records_buffer_->push_back(node_record);
        node_count++;

        // Note: Node properties embedded in ProjectedNode are handled via
        // add_node_property() calls from NativeProjectionBuilder, not here.
        // This avoids double-counting and ensures proper key_id encoding.
    }

    node_batch.clear();
}

void ProjectionStorage::flush_edge_batch() {
    if (edge_batch.empty()) {
        return;
    }

    // STREAMING BUFFER: Collect records with automatic disk spill.
    // NO B+tree insertions here - all indexes are built in bulk during finalize().
    // Memory bounded: spills to disk when threshold exceeded.
    //
    // When serial_write_mask_ is non-zero (SERIAL scan mode, set by
    // begin_serial_edge_pass_), only write to the buffer(s) whose
    // ProjectionIndex bit is set.  This bounds peak scratch disk to
    // O(max single-pass spill) instead of O(5 × full-edge-set) on
    // large datasets (papers100M disk fix — see begin_serial_edge_pass_).
    const bool serial_mode = (serial_write_mask_ != 0);
    const uint32_t mask    = serial_write_mask_;

    for (const auto& edge : edge_batch) {
        uint64_t from_id = edge.from_node.id;
        uint64_t to_id = edge.to_node.id;
        uint64_t edge_id = edge.edge_id.id;

        // Collect from->to record
        if (!serial_mode || (mask & static_cast<uint32_t>(ProjectionIndex::FROM_TO_EDGE))) {
            Record<3> from_to_record;
            from_to_record[0] = from_id;
            from_to_record[1] = to_id;
            from_to_record[2] = edge_id;
            from_to_records_buffer_->push_back(from_to_record);
        }

        // Collect to->from record
        if (!serial_mode || (mask & static_cast<uint32_t>(ProjectionIndex::TO_FROM_EDGE))) {
            Record<3> to_from_record;
            to_from_record[0] = to_id;
            to_from_record[1] = from_id;
            to_from_record[2] = edge_id;
            to_from_records_buffer_->push_back(to_from_record);
        }

        // Collect direction record
        if (!serial_mode || (mask & static_cast<uint32_t>(ProjectionIndex::EDGE_DIRECTION))) {
            Record<2> direction_record;
            direction_record[0] = edge_id;
            direction_record[1] = edge.is_directed ? 1 : 0;
            direction_records_buffer_->push_back(direction_record);
        }

        // Collect edge-first records (for edge-bound query patterns)
        if (!serial_mode || (mask & (static_cast<uint32_t>(ProjectionIndex::EDGE_FROM_TO)
                                    | static_cast<uint32_t>(ProjectionIndex::EDGE_N1_N2)))) {
            Record<3> edge_first_record;
            edge_first_record[0] = edge_id;
            edge_first_record[1] = from_id;
            edge_first_record[2] = to_id;
            if (edge.is_directed) {
                if (!serial_mode || (mask & static_cast<uint32_t>(ProjectionIndex::EDGE_FROM_TO))) {
                    edge_from_to_records_buffer_->push_back(edge_first_record);
                }
            } else {
                if (!serial_mode || (mask & static_cast<uint32_t>(ProjectionIndex::EDGE_N1_N2))) {
                    edge_n1_n2_records_buffer_->push_back(edge_first_record);
                }
            }
        }

        // Update counts once per unique edge processed.
        // In SERIAL mode each core edge pass re-scans the same edges;
        // only count on the canonical FROM_TO_EDGE pass (the first pass
        // in the edge phase) to avoid multiplying edge_count by 5.
        // In CLASSIC mode (serial_mode=false) always count.
        const bool count_this_pass =
            !serial_mode ||
            (mask & static_cast<uint32_t>(ProjectionIndex::FROM_TO_EDGE));
        if (count_this_pass) {
            edge_count++;
            if (edge.is_directed) {
                directed_edge_count++;
            } else {
                undirected_edge_count++;
            }
        }
    }

    edge_batch.clear();
}

void ProjectionStorage::resize_bloom_filter(size_t expected_edges, double fpr) {
    // Only resize if new size is larger than current default
    // This prevents unnecessary memory allocation for small projections
    if (expected_edges <= DEFAULT_EXPECTED_EDGES) {
        return;  // Default 10M is sufficient
    }

    // Create new larger Bloom filter to handle the expected edge count
    // Memory scales linearly: ~10 bits per edge for 1% FPR
    // Example: 61M edges → ~76 MB, 100M edges → ~125 MB
    edge_bloom_filter_ = std::make_unique<BloomFilter>(expected_edges, fpr);
}

void ProjectionStorage::finalize_node_scan() {
    if (collected_nodes_sorted_) {
        return;  // Idempotent: second call is a no-op.
    }
    std::sort(collected_nodes_.begin(), collected_nodes_.end());
    collected_nodes_.erase(std::unique(collected_nodes_.begin(),
                                       collected_nodes_.end()),
                           collected_nodes_.end());
    // Release any excess capacity from multi-label over-collection.
    collected_nodes_.shrink_to_fit();
    collected_nodes_sorted_ = true;
}

std::vector<ObjectId> ProjectionStorage::get_all_node_ids() const {
    std::vector<ObjectId> node_ids;

    if (!nodes_index) {
        return node_ids;
    }

    // Scan all nodes in the B+tree
    Record<1> min_record;
    min_record[0] = 0;

    Record<1> max_record;
    max_record[0] = UINT64_MAX;

    bool interruption_requested = false;
    auto iter = nodes_index->get_range(&interruption_requested, min_record, max_record);

    const Record<1>* record;
    while ((record = iter.next()) != nullptr) {
        node_ids.push_back(ObjectId((*record)[0]));
    }

    return node_ids;
}

std::vector<std::tuple<ObjectId, ObjectId, ObjectId, bool>> ProjectionStorage::get_all_edges_info() const {
    std::vector<std::tuple<ObjectId, ObjectId, ObjectId, bool>> edges;

    if (!from_to_edge_index || !edge_direction_index) {
        return edges;
    }

    // Scan all edges in the from_to_edge B+tree
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

    const Record<3>* record;
    while ((record = iter.next()) != nullptr) {
        ObjectId from_node((*record)[0]);
        ObjectId to_node((*record)[1]);
        ObjectId edge_id((*record)[2]);

        // Look up direction for this edge
        Record<2> dir_min;
        dir_min[0] = edge_id.id;
        dir_min[1] = 0;

        Record<2> dir_max;
        dir_max[0] = edge_id.id;
        dir_max[1] = 1;

        bool interruption = false;
        auto dir_iter = edge_direction_index->get_range(&interruption, dir_min, dir_max);
        const Record<2>* dir_record = dir_iter.next();

        bool is_directed = true;
        if (dir_record != nullptr) {
            is_directed = ((*dir_record)[1] == 1);
        }

        edges.push_back(std::make_tuple(from_node, to_node, edge_id, is_directed));
    }

    return edges;
}

std::optional<ObjectId> ProjectionStorage::get_node_property(ObjectId node_id, ObjectId key_id) const {
    // Check if node properties index exists
    if (!node_key_value_index) {
        return std::nullopt;
    }

    // Search for (node_id, key_id, MIN) to (node_id, key_id, MAX)
    Record<3> min_record;
    min_record[0] = node_id.id;
    min_record[1] = key_id.id;
    min_record[2] = 0;

    Record<3> max_record;
    max_record[0] = node_id.id;
    max_record[1] = key_id.id;
    max_record[2] = UINT64_MAX;

    bool interruption_requested = false;
    auto iter = node_key_value_index->get_range(&interruption_requested, min_record, max_record);

    const Record<3>* record = iter.next();
    if (record == nullptr) {
        return std::nullopt;
    }

    // Return the value (third element)
    return ObjectId((*record)[2]);
}

std::optional<ObjectId> ProjectionStorage::get_edge_property(ObjectId edge_id, ObjectId key_id) const {
    // Check if edge properties index exists
    if (!edge_key_value_index) {
        return std::nullopt;
    }

    // Search for (edge_id, key_id, MIN) to (edge_id, key_id, MAX)
    Record<3> min_record;
    min_record[0] = edge_id.id;
    min_record[1] = key_id.id;
    min_record[2] = 0;

    Record<3> max_record;
    max_record[0] = edge_id.id;
    max_record[1] = key_id.id;
    max_record[2] = UINT64_MAX;

    bool interruption_requested = false;
    auto iter = edge_key_value_index->get_range(&interruption_requested, min_record, max_record);

    const Record<3>* record = iter.next();
    if (record == nullptr) {
        return std::nullopt;
    }

    // Return the value (third element)
    return ObjectId((*record)[2]);
}

void ProjectionStorage::register_node_key(const std::string& key_name, uint64_t key_id) {
    if (node_keys2id_.find(key_name) != node_keys2id_.end()) {
        return;  // Already registered
    }
    node_keys2id_[key_name] = key_id;
    // Ensure node_keys_str_ has space for the ID
    if (key_id >= node_keys_str_.size()) {
        node_keys_str_.resize(key_id + 1);
    }
    node_keys_str_[key_id] = key_name;
}

void ProjectionStorage::register_edge_key(const std::string& key_name, uint64_t key_id) {
    if (edge_keys2id_.find(key_name) != edge_keys2id_.end()) {
        return;  // Already registered
    }
    edge_keys2id_[key_name] = key_id;
    // Ensure edge_keys_str_ has space for the ID
    if (key_id >= edge_keys_str_.size()) {
        edge_keys_str_.resize(key_id + 1);
    }
    edge_keys_str_[key_id] = key_name;
}

void ProjectionStorage::save_catalog() {
    // Don't create catalog if projection name is empty (e.g., when opening existing projection)
    if (projection_name.empty()) {
        return;
    }

    ProjectionCatalog catalog(projection_dir);

    // Set projection metadata
    catalog.projection_name = projection_name;
    catalog.creation_timestamp = std::chrono::duration_cast<std::chrono::seconds>(
        std::chrono::system_clock::now().time_since_epoch()).count();

    // Set statistics
    catalog.node_count = node_count;
    catalog.edge_count = edge_count;
    catalog.directed_edge_count = directed_edge_count;
    catalog.undirected_edge_count = undirected_edge_count;

    // Set legacy configuration flags (v1.0 compatibility)
    catalog.has_node_properties = (node_properties_index != nullptr);
    catalog.has_edge_properties = (edge_properties_index != nullptr);
    catalog.undirected_relationships = (undirected_edge_count > 0);

    // Set v1.1 feature flags (optional indexes)
    catalog.includes_node_labels = features.include_node_labels;
    catalog.includes_edge_labels = features.include_edge_labels;
    catalog.includes_node_properties = features.include_node_properties;
    catalog.includes_edge_properties = features.include_edge_properties;

    // Persist the property name lists the caller asked for, so inspect-projection
    // can show them after the fact. Empty = "all available" (builder convention).
    catalog.included_node_properties = requested_node_properties;
    catalog.included_edge_properties = requested_edge_properties;

    // Set v1.2 key mappings (for projection-specific properties like _count)
    for (const auto& [name, id] : node_keys2id_) {
        catalog.add_node_key(name, id);
    }
    for (const auto& [name, id] : edge_keys2id_) {
        catalog.add_edge_key(name, id);
    }

    // Persist the v1.4 IndexSet preset. The default ordinal (0 = ALL) is set
    // both here and on the catalog field, so a builder that never sets the
    // preset still produces a well-formed v1.4 catalog equivalent to the
    // pre-Spec #3 "everything materialized" behavior.
    catalog.index_set = requested_index_set;

    // Save to disk
    catalog.save();
}

bool ProjectionStorage::edge_exists_in_btree(ObjectId from, ObjectId to, ObjectId edge_id) const {
    if (!from_to_edge_index) {
        return false;
    }

    // Exact match lookup: (from, to, edge_id)
    Record<3> search_record;
    search_record[0] = from.id;
    search_record[1] = to.id;
    search_record[2] = edge_id.id;

    bool interruption_requested = false;
    auto iter = from_to_edge_index->get_range(&interruption_requested, search_record, search_record);
    return iter.next() != nullptr;
}

// ============================================================================
// BULK IMPORT IMPLEMENTATION
// ============================================================================
//
// These methods build B+tree indexes using direct file writes, completely
// bypassing the buffer pool. This enables building indexes for arbitrarily
// large datasets without memory pressure issues.
//
// Pattern: Sort records → Write leaf pages → Build directory
// ============================================================================

template<std::size_t N>
size_t ProjectionStorage::build_index_bulk(std::vector<Record<N>>& records, const std::string& base_path) {
    BPTLeafWriter<N> leaf_writer(base_path + ".leaf");
    BPTDirWriter<N> dir_writer(base_path + ".dir");

    // Handle empty index case
    if (records.empty()) {
        leaf_writer.make_empty();
        return 0;
    }

    // Sort records for B+tree ordering
#ifdef HAS_TBB
    std::sort(std::execution::par_unseq, records.begin(), records.end());
#else
    std::sort(records.begin(), records.end());
#endif

    // Remove duplicates (stable, maintains sorted order)
    auto last = std::unique(records.begin(), records.end());
    records.erase(last, records.end());

    // Capture count after deduplication (for accurate statistics)
    size_t unique_count = records.size();

    // Calculate leaf page capacity
    // IMPORTANT: BPTLeafWriter::max_records formula doesn't account for the N-byte bitset!
    // Formula (Page::SIZE - 8) / (8*N) = 511 for N=1, but only 510 records actually fit.
    // Correct formula: (Page::SIZE - header - bitset) / record_size
    constexpr size_t max_records_per_leaf = (Page::SIZE - 2*sizeof(uint32_t) - N) / (sizeof(uint64_t) * N);

    // Allocate buffer for formatted leaf data
    // Format: [bitset (N bytes)] + [records (size * N * 8 bytes)]
    // With no compression (bitset=0), we need N + records * N * 8 bytes max
    constexpr size_t max_buffer_size = N + max_records_per_leaf * sizeof(uint64_t) * N;
    auto leaf_buffer = std::make_unique<char[]>(max_buffer_size);

    // Write records to leaf pages
    size_t total_records = records.size();
    size_t leaf_page_number = 0;
    size_t records_written = 0;

    std::bitset<N * 8> no_compression;  // All zeros = no compression

    while (records_written < total_records) {
        // Calculate records for this leaf page
        size_t records_in_page = std::min(max_records_per_leaf, total_records - records_written);

        // Get pointer to first record of this leaf
        Record<N>* page_start = &records[records_written];

        // Determine next leaf page number (0 for last page)
        uint32_t next_page = (records_written + records_in_page < total_records)
                                 ? static_cast<uint32_t>(leaf_page_number + 1)
                                 : 0;

        // Build directory entry (skip first leaf - B+tree convention)
        if (leaf_page_number > 0) {
            dir_writer.bulk_insert(page_start, 0, static_cast<int32_t>(leaf_page_number));
        }

        // Format data for process_block():
        // [bitset as N bytes] + [all record bytes]
        // With no compression, bitset=0 so just N zero bytes followed by raw records
        size_t buffer_pos = 0;

        // Write bitset (N zero bytes for no compression)
        unsigned long bits_ul = no_compression.to_ulong();
        std::memcpy(leaf_buffer.get(), &bits_ul, N);
        buffer_pos += N;

        // Write all record bytes (no compression = full records)
        std::memcpy(leaf_buffer.get() + buffer_pos,
                    reinterpret_cast<char*>(page_start),
                    records_in_page * sizeof(uint64_t) * N);

        // Write leaf page
        leaf_writer.process_block(
            leaf_buffer.get(),
            static_cast<uint32_t>(records_in_page),
            no_compression,
            next_page
        );

        records_written += records_in_page;
        leaf_page_number++;
    }

    // Writers automatically flush to disk when destroyed
    return unique_count;
}

// ============================================================================
// STREAMING BULK IMPORT IMPLEMENTATION
// ============================================================================
//
// Memory-bounded index building using external merge sort.
// Streams sorted records without loading all data into RAM.
//
// Memory model: O(buffer_size) regardless of dataset size
// ============================================================================

template<std::size_t N>
size_t ProjectionStorage::build_index_streaming(ExternalRecordSort<N>& sorter, const std::string& base_path) {
    BPTLeafWriter<N> leaf_writer(base_path + ".leaf");
    BPTDirWriter<N> dir_writer(base_path + ".dir");

    // Handle empty index case
    if (sorter.total_records() == 0) {
        leaf_writer.make_empty();
        return 0;
    }

    // Calculate leaf page capacity (same formula as build_index_bulk)
    constexpr size_t max_records_per_leaf = (Page::SIZE - 2*sizeof(uint32_t) - N) / (sizeof(uint64_t) * N);

    // Allocate buffer for formatted leaf data
    constexpr size_t max_buffer_size = N + max_records_per_leaf * sizeof(uint64_t) * N;
    auto leaf_buffer = std::make_unique<char[]>(max_buffer_size);

    // Streaming state
    std::vector<Record<N>> page_records;
    page_records.reserve(max_records_per_leaf);

    Record<N> prev_record;
    bool has_prev = false;
    size_t unique_count = 0;
    size_t leaf_page_number = 0;

    std::bitset<N * 8> no_compression;  // All zeros = no compression

    // Lambda to write a leaf page
    auto write_leaf_page = [&](bool is_last_page) {
        if (page_records.empty()) return;

        // Determine next leaf page number (0 for last page)
        uint32_t next_page = is_last_page ? 0 : static_cast<uint32_t>(leaf_page_number + 1);

        // Build directory entry (skip first leaf - B+tree convention)
        if (leaf_page_number > 0) {
            dir_writer.bulk_insert(&page_records[0], 0, static_cast<int32_t>(leaf_page_number));
        }

        // Format data for process_block():
        // [bitset as N bytes] + [all record bytes]
        size_t buffer_pos = 0;

        // Write bitset (N zero bytes for no compression)
        unsigned long bits_ul = no_compression.to_ulong();
        std::memcpy(leaf_buffer.get(), &bits_ul, N);
        buffer_pos += N;

        // Write all record bytes
        std::memcpy(leaf_buffer.get() + buffer_pos,
                    reinterpret_cast<char*>(page_records.data()),
                    page_records.size() * sizeof(uint64_t) * N);

        // Write leaf page
        leaf_writer.process_block(
            leaf_buffer.get(),
            static_cast<uint32_t>(page_records.size()),
            no_compression,
            next_page
        );

        leaf_page_number++;
        page_records.clear();
    };

    // Stream sorted records with inline deduplication
    sorter.stream_sorted([&](const Record<N>& record) {
        // Deduplication: skip if same as previous
        if (has_prev && record == prev_record) {
            return;
        }
        prev_record = record;
        has_prev = true;
        unique_count++;

        page_records.push_back(record);

        // Write page when full
        if (page_records.size() >= max_records_per_leaf) {
            write_leaf_page(false);  // Not last page
        }
    });

    // Write final partial page
    write_leaf_page(true);  // Is last page

    return unique_count;
}

// Explicit template instantiations for streaming build
template size_t ProjectionStorage::build_index_streaming<1>(ExternalRecordSort<1>&, const std::string&);
template size_t ProjectionStorage::build_index_streaming<2>(ExternalRecordSort<2>&, const std::string&);
template size_t ProjectionStorage::build_index_streaming<3>(ExternalRecordSort<3>&, const std::string&);

// =========================================================================
// Per-index build methods (extracted from build_all_indexes_bulk).
//
// Each method wraps exactly one GQL::sort_and_build_index<N>() call,
// using the matching StreamingRecordBuffer and a per-projection
// sort_tmp/ scratch directory. Optional indexes early-return when
// their feature flag is false or the backing buffer is null.
//
// Callers: build_all_indexes_bulk() in CLASSIC mode, and
// build_one_index(ProjectionIndex) in SERIALIZED mode (Task 5).
// =========================================================================

void ProjectionStorage::build_nodes_index_() {
    // 1. Nodes index (Record<1>: node_id)
    std::string sort_temp_dir = projection_dir + "/sort_tmp";
    std::filesystem::create_directories(sort_temp_dir);
    node_count = GQL::sort_and_build_index<1>(
        *node_records_buffer_,
        projection_dir + "/nodes",
        /*estimated_count=*/0,  // Unused by CLASSIC; RADIX wires this in Task 12.
        [this](ExternalRecordSort<1>& sorter, const std::string& path) {
            return build_index_streaming<1>(sorter, path);
        },
        sort_temp_dir
    );
}

void ProjectionStorage::build_from_to_edge_index_() {
    // 2. From→To edge index (Record<3>: from, to, edge_id)
    std::string sort_temp_dir = projection_dir + "/sort_tmp";
    std::filesystem::create_directories(sort_temp_dir);
    GQL::sort_and_build_index<3>(
        *from_to_records_buffer_,
        projection_dir + "/from_to_edge",
        /*estimated_count=*/0,  // Unused by CLASSIC; RADIX wires this in Task 12.
        [this](ExternalRecordSort<3>& sorter, const std::string& path) {
            return build_index_streaming<3>(sorter, path);
        },
        sort_temp_dir
    );
}

void ProjectionStorage::build_to_from_edge_index_() {
    // 3. To→From edge index (Record<3>: to, from, edge_id)
    std::string sort_temp_dir = projection_dir + "/sort_tmp";
    std::filesystem::create_directories(sort_temp_dir);
    GQL::sort_and_build_index<3>(
        *to_from_records_buffer_,
        projection_dir + "/to_from_edge",
        /*estimated_count=*/0,  // Unused by CLASSIC; RADIX wires this in Task 12.
        [this](ExternalRecordSort<3>& sorter, const std::string& path) {
            return build_index_streaming<3>(sorter, path);
        },
        sort_temp_dir
    );
}

void ProjectionStorage::build_edge_direction_index_() {
    // 4. Edge direction index (Record<2>: edge_id, is_directed)
    std::string sort_temp_dir = projection_dir + "/sort_tmp";
    std::filesystem::create_directories(sort_temp_dir);
    GQL::sort_and_build_index<2>(
        *direction_records_buffer_,
        projection_dir + "/edge_direction",
        /*estimated_count=*/0,  // Unused by CLASSIC; RADIX wires this in Task 12.
        [this](ExternalRecordSort<2>& sorter, const std::string& path) {
            return build_index_streaming<2>(sorter, path);
        },
        sort_temp_dir
    );
}

void ProjectionStorage::build_edge_from_to_index_() {
    // 5. Edge→From→To index for directed edges (Record<3>: edge_id, from, to)
    std::string sort_temp_dir = projection_dir + "/sort_tmp";
    std::filesystem::create_directories(sort_temp_dir);
    GQL::sort_and_build_index<3>(
        *edge_from_to_records_buffer_,
        projection_dir + "/edge_from_to",
        /*estimated_count=*/0,  // Unused by CLASSIC; RADIX wires this in Task 12.
        [this](ExternalRecordSort<3>& sorter, const std::string& path) {
            return build_index_streaming<3>(sorter, path);
        },
        sort_temp_dir
    );
}

void ProjectionStorage::build_edge_n1_n2_index_() {
    // 6. Edge→N1→N2 index for undirected edges (Record<3>: edge_id, n1, n2)
    std::string sort_temp_dir = projection_dir + "/sort_tmp";
    std::filesystem::create_directories(sort_temp_dir);
    GQL::sort_and_build_index<3>(
        *edge_n1_n2_records_buffer_,
        projection_dir + "/edge_n1_n2",
        /*estimated_count=*/0,  // Unused by CLASSIC; RADIX wires this in Task 12.
        [this](ExternalRecordSort<3>& sorter, const std::string& path) {
            return build_index_streaming<3>(sorter, path);
        },
        sort_temp_dir
    );
}

void ProjectionStorage::build_node_label_index_() {
    // Node→Label index (Record<2>: node_id, label_id)
    if (!features.include_node_labels || !node_label_records_buffer_) return;
    std::string sort_temp_dir = projection_dir + "/sort_tmp";
    std::filesystem::create_directories(sort_temp_dir);
    GQL::sort_and_build_index<2>(
        *node_label_records_buffer_,
        projection_dir + "/node_label",
        /*estimated_count=*/0,  // Unused by CLASSIC; RADIX wires this in Task 12.
        [this](ExternalRecordSort<2>& sorter, const std::string& path) {
            return build_index_streaming<2>(sorter, path);
        },
        sort_temp_dir
    );
}

void ProjectionStorage::build_label_node_index_() {
    // Label→Node index (Record<2>: label_id, node_id)
    if (!features.include_node_labels || !label_node_records_buffer_) return;
    std::string sort_temp_dir = projection_dir + "/sort_tmp";
    std::filesystem::create_directories(sort_temp_dir);
    GQL::sort_and_build_index<2>(
        *label_node_records_buffer_,
        projection_dir + "/label_node",
        /*estimated_count=*/0,  // Unused by CLASSIC; RADIX wires this in Task 12.
        [this](ExternalRecordSort<2>& sorter, const std::string& path) {
            return build_index_streaming<2>(sorter, path);
        },
        sort_temp_dir
    );
}

void ProjectionStorage::build_edge_label_index_() {
    // Edge→Label index (Record<2>: edge_id, label_id)
    if (!features.include_edge_labels || !edge_label_records_buffer_) return;
    std::string sort_temp_dir = projection_dir + "/sort_tmp";
    std::filesystem::create_directories(sort_temp_dir);
    GQL::sort_and_build_index<2>(
        *edge_label_records_buffer_,
        projection_dir + "/edge_label",
        /*estimated_count=*/0,  // Unused by CLASSIC; RADIX wires this in Task 12.
        [this](ExternalRecordSort<2>& sorter, const std::string& path) {
            return build_index_streaming<2>(sorter, path);
        },
        sort_temp_dir
    );
}

void ProjectionStorage::build_label_edge_index_() {
    // Label→Edge index (Record<2>: label_id, edge_id)
    if (!features.include_edge_labels || !label_edge_records_buffer_) return;
    std::string sort_temp_dir = projection_dir + "/sort_tmp";
    std::filesystem::create_directories(sort_temp_dir);
    GQL::sort_and_build_index<2>(
        *label_edge_records_buffer_,
        projection_dir + "/label_edge",
        /*estimated_count=*/0,  // Unused by CLASSIC; RADIX wires this in Task 12.
        [this](ExternalRecordSort<2>& sorter, const std::string& path) {
            return build_index_streaming<2>(sorter, path);
        },
        sort_temp_dir
    );
}

void ProjectionStorage::build_node_key_value_index_() {
    // Node→Key→Value index (Record<3>: node_id, key_id, value_id)
    if (!features.include_node_properties || !node_key_value_records_buffer_) return;
    std::string sort_temp_dir = projection_dir + "/sort_tmp";
    std::filesystem::create_directories(sort_temp_dir);
    GQL::sort_and_build_index<3>(
        *node_key_value_records_buffer_,
        projection_dir + "/node_key_value",
        /*estimated_count=*/0,  // Unused by CLASSIC; RADIX wires this in Task 12.
        [this](ExternalRecordSort<3>& sorter, const std::string& path) {
            return build_index_streaming<3>(sorter, path);
        },
        sort_temp_dir
    );
}

void ProjectionStorage::build_key_value_node_index_() {
    // Key→Value→Node index (Record<3>: key_id, value_id, node_id)
    if (!features.include_node_properties || !key_value_node_records_buffer_) return;
    std::string sort_temp_dir = projection_dir + "/sort_tmp";
    std::filesystem::create_directories(sort_temp_dir);
    GQL::sort_and_build_index<3>(
        *key_value_node_records_buffer_,
        projection_dir + "/key_value_node",
        /*estimated_count=*/0,  // Unused by CLASSIC; RADIX wires this in Task 12.
        [this](ExternalRecordSort<3>& sorter, const std::string& path) {
            return build_index_streaming<3>(sorter, path);
        },
        sort_temp_dir
    );
}

void ProjectionStorage::build_edge_key_value_index_() {
    // Edge→Key→Value index (Record<3>: edge_id, key_id, value_id)
    if (!features.include_edge_properties || !edge_key_value_records_buffer_) return;
    std::string sort_temp_dir = projection_dir + "/sort_tmp";
    std::filesystem::create_directories(sort_temp_dir);
    GQL::sort_and_build_index<3>(
        *edge_key_value_records_buffer_,
        projection_dir + "/edge_key_value",
        /*estimated_count=*/0,  // Unused by CLASSIC; RADIX wires this in Task 12.
        [this](ExternalRecordSort<3>& sorter, const std::string& path) {
            return build_index_streaming<3>(sorter, path);
        },
        sort_temp_dir
    );
}

void ProjectionStorage::build_key_value_edge_index_() {
    // Key→Value→Edge index (Record<3>: key_id, value_id, edge_id)
    if (!features.include_edge_properties || !key_value_edge_records_buffer_) return;
    std::string sort_temp_dir = projection_dir + "/sort_tmp";
    std::filesystem::create_directories(sort_temp_dir);
    GQL::sort_and_build_index<3>(
        *key_value_edge_records_buffer_,
        projection_dir + "/key_value_edge",
        /*estimated_count=*/0,  // Unused by CLASSIC; RADIX wires this in Task 12.
        [this](ExternalRecordSort<3>& sorter, const std::string& path) {
            return build_index_streaming<3>(sorter, path);
        },
        sort_temp_dir
    );
}

void ProjectionStorage::reset_sort_scratch_() {
    std::error_code ec;
    std::filesystem::remove_all(projection_dir + "/sort_tmp", ec);
    // Best-effort cleanup; recreate empty dir for the next pass.
    std::filesystem::create_directories(projection_dir + "/sort_tmp");
    // NOTE: .radix_scratch (Spec #1) is managed by RadixPartitionSort's
    // destructor; we don't need to touch it here.
}

void ProjectionStorage::begin_serial_edge_pass_(ProjectionIndex which) {
    // -----------------------------------------------------------------------
    // Disk-bound fix for papers100M Run 7 (SERIAL+RADIX ENOSPC).
    //
    // Root cause: flush_edge_batch() distributes every edge to ALL 5 edge
    // streaming buffers simultaneously. In the old design, pass 1
    // (FROM_TO_EDGE) filled all 5 buffers in one scan, with passes 2-5
    // each consuming their pre-filled buffer. On papers100M (1.6B edges ×
    // 24 bytes × 4 non-target buffers ≈ 153 GB of spill files) the disk
    // filled before RadixPartitionSort could write partition files for the
    // first edge-index pass, producing the "short write to part_7.bin"
    // ENOSPC error.
    //
    // Fix (per-pass mask):
    //   serial_write_mask_ restricts flush_edge_batch() to only write to
    //   the buffer(s) corresponding to the current target index. Every pass
    //   independently re-scans the source edges for its own buffer only.
    //   The edge bloom filter is also cleared so the fresh per-pass scan
    //   emits all edges regardless of what prior passes added (the
    //   EdgeFilter from Phase B handles structural dedup; the bloom is
    //   redundant in SERIAL mode).
    //
    //   Non-target buffers that may have accumulated spill data from the
    //   immediately preceding pass are explicitly cleared here to reclaim
    //   disk.  (The target buffer itself is cleared by sort_and_build_index
    //   via input_stream.clear() after the B+Tree is written.)
    //
    // Bounded disk: O(max single-pass spill) at all times.
    // Correctness:  each pass has a fresh bloom + fresh target buffer.
    // -----------------------------------------------------------------------

    // Clear any residual spill files from non-target buffers.
    auto clear_if = [](auto& buf) { if (buf) buf->clear(); };
    clear_if(from_to_records_buffer_);
    clear_if(to_from_records_buffer_);
    clear_if(direction_records_buffer_);
    clear_if(edge_from_to_records_buffer_);
    clear_if(edge_n1_n2_records_buffer_);
    clear_if(edge_label_records_buffer_);
    clear_if(label_edge_records_buffer_);
    clear_if(edge_key_value_records_buffer_);
    clear_if(key_value_edge_records_buffer_);

    // Reset bloom so each pass sees all edges fresh.
    if (edge_bloom_filter_) edge_bloom_filter_->clear();

    // Arm the write mask: flush_edge_batch() will skip all other buffers.
    serial_write_mask_ = static_cast<uint32_t>(which);
}

void ProjectionStorage::end_serial_edge_pass_() {
    // Disarm the write mask so that any post-serialized code (e.g., classic
    // path, label/property passes) uses the full all-buffer write path.
    serial_write_mask_ = 0;
}

void ProjectionStorage::build_all_indexes_bulk() {
    // =========================================================================
    // STREAMING BULK IMPORT: Build indexes with bounded memory
    // =========================================================================
    //
    // Uses ExternalRecordSort for memory-bounded external merge sort.
    // Each index is built by streaming sorted records directly to B+tree
    // leaf pages, with inline deduplication.
    //
    // Memory model: O(adaptive buffer) = O(MemAvailable * 0.75),
    //               with a 256 MB floor. See src/misc/available_ram.h.
    //               ~adaptive MB for sort + one page for B+tree writing
    // =========================================================================

    // Create temp directory for sorted runs (shared by all 14 builds)
    std::string sort_temp_dir = projection_dir + "/sort_tmp";
    std::filesystem::create_directories(sort_temp_dir);

    // Spec #3 T3.8: gate topology/label index materialization on the active
    // IndexSet preset mask. The 10 gated bits are: NODES, NODE_LABEL,
    // LABEL_NODE, FROM_TO_EDGE, TO_FROM_EDGE, EDGE_DIRECTION, EDGE_FROM_TO,
    // EDGE_N1_N2, EDGE_LABEL, LABEL_EDGE. Property indexes (NODE_KEY_VALUE,
    // KEY_VALUE_NODE, EDGE_KEY_VALUE, KEY_VALUE_EDGE) are NOT gated here —
    // they remain conditional on features.include_*_properties inside their
    // respective build_*_index_() helpers (Spec #3 §3.4). The mask is
    // computed once per build since requested_index_set is immutable after
    // the builder hands off to flush().
    const ProjectionIndex active_mask = project_index_mask_for(requested_index_set);

    // PHASE 1: Required topology indexes — now gated by IndexSet.
    // GNN_MINIMAL keeps NODES + FROM_TO_EDGE + TO_FROM_EDGE; the 3
    // edge-lookup indexes (EDGE_DIRECTION / EDGE_FROM_TO / EDGE_N1_N2)
    // are elided for k-hop-sampling-only workloads.
    if (has_flag(active_mask, ProjectionIndex::NODES)) {
        build_nodes_index_();
    }
    if (has_flag(active_mask, ProjectionIndex::FROM_TO_EDGE)) {
        build_from_to_edge_index_();
    }
    if (has_flag(active_mask, ProjectionIndex::TO_FROM_EDGE)) {
        build_to_from_edge_index_();
    }
    if (has_flag(active_mask, ProjectionIndex::EDGE_DIRECTION)) {
        build_edge_direction_index_();
    }
    if (has_flag(active_mask, ProjectionIndex::EDGE_FROM_TO)) {
        build_edge_from_to_index_();
    }
    if (has_flag(active_mask, ProjectionIndex::EDGE_N1_N2)) {
        build_edge_n1_n2_index_();
    }

    // PHASE 2: Optional label indexes — gated by both IndexSet and the
    // features.include_*_labels flag. The features flag is still consulted
    // inside each build_*_label_index_() helper, so when include_label_indexes
    // is false the helpers short-circuit even if IndexSet would have kept
    // them; conversely, GNN_MINIMAL omits all four label bits regardless of
    // the features flag.
    if (has_flag(active_mask, ProjectionIndex::NODE_LABEL)) {
        build_node_label_index_();
    }
    if (has_flag(active_mask, ProjectionIndex::LABEL_NODE)) {
        build_label_node_index_();
    }
    if (has_flag(active_mask, ProjectionIndex::EDGE_LABEL)) {
        build_edge_label_index_();
    }
    if (has_flag(active_mask, ProjectionIndex::LABEL_EDGE)) {
        build_label_edge_index_();
    }

    // PHASE 3: Optional property indexes — NOT gated by IndexSet (Spec #3
    // §3.4). Property-config gates via features.include_*_properties remain
    // the sole controller inside each helper.
    build_node_key_value_index_();
    build_key_value_node_index_();
    build_edge_key_value_index_();
    build_key_value_edge_index_();

    // PHASE 4: Cleanup + open indexes for reading.
    // Delegated to open_all_bplustree_readers_() so the SERIALIZED path
    // (finalize_serialized_) can call the same Phase 4 after its piecemeal
    // build passes without duplicating the reader-open logic (Spec #2, Task 11).
    open_all_bplustree_readers_();
}

// Spec #2, Task 11 — Phase 4 extracted from build_all_indexes_bulk().
//
// Opens all B+Tree index readers after the .leaf/.dir files have been built.
// Removes the sort_tmp scratch directory as a final cleanup step.
//
// Called by:
//   - build_all_indexes_bulk() (CLASSIC path) after all 14 sort+build calls.
//   - NativeProjectionBuilder::finalize_serialized_() (SERIALIZED path) after
//     the last Phase C build_one_index() call, before save_catalog() runs.
void ProjectionStorage::open_all_bplustree_readers_() {
    // Remove sort_tmp scratch directory (created by build_*_index_ helpers
    // and also by reset_sort_scratch_ between serialized passes).
    // best-effort: if already removed, remove_all is a no-op.
    std::filesystem::remove_all(projection_dir + "/sort_tmp");

    // Spec #3 T3.8: open readers only for indexes whose bit is in the active
    // IndexSet preset. FileManager::get_file_id() opens with O_CREAT, so
    // unconditional construction of a BPlusTree for a skipped index would
    // produce a 0-byte .leaf/.dir on disk — defeating the file-count /
    // disk-footprint contract of GNN_MINIMAL. The query layer (T3.9, out of
    // scope here) will diagnose attempts to use an elided index.
    const ProjectionIndex active_mask = project_index_mask_for(requested_index_set);

    // Open topology indexes (previously unconditionally opened).
    if (has_flag(active_mask, ProjectionIndex::NODES)) {
        nodes_index = std::make_unique<BPlusTree<1>>(rel_dir + "/nodes");
    }
    if (has_flag(active_mask, ProjectionIndex::FROM_TO_EDGE)) {
        from_to_edge_index = std::make_unique<BPlusTree<3>>(rel_dir + "/from_to_edge");
    }
    if (has_flag(active_mask, ProjectionIndex::TO_FROM_EDGE)) {
        to_from_edge_index = std::make_unique<BPlusTree<3>>(rel_dir + "/to_from_edge");
    }
    if (has_flag(active_mask, ProjectionIndex::EDGE_DIRECTION)) {
        edge_direction_index = std::make_unique<BPlusTree<2>>(rel_dir + "/edge_direction");
    }
    if (has_flag(active_mask, ProjectionIndex::EDGE_FROM_TO)) {
        edge_from_to_index = std::make_unique<BPlusTree<3>>(rel_dir + "/edge_from_to");
    }
    if (has_flag(active_mask, ProjectionIndex::EDGE_N1_N2)) {
        edge_n1_n2_index = std::make_unique<BPlusTree<3>>(rel_dir + "/edge_n1_n2");
    }

    // Open optional label indexes — dual gate: features.include_*_labels
    // AND the IndexSet bit. features flag still participates so legacy
    // include_label_indexes=false users retain their disk-saving behavior.
    if (features.include_node_labels) {
        if (has_flag(active_mask, ProjectionIndex::NODE_LABEL)) {
            node_label_index = std::make_unique<BPlusTree<2>>(rel_dir + "/node_label");
        }
        if (has_flag(active_mask, ProjectionIndex::LABEL_NODE)) {
            label_node_index = std::make_unique<BPlusTree<2>>(rel_dir + "/label_node");
        }
    }
    if (features.include_edge_labels) {
        if (has_flag(active_mask, ProjectionIndex::EDGE_LABEL)) {
            edge_label_index = std::make_unique<BPlusTree<2>>(rel_dir + "/edge_label");
        }
        if (has_flag(active_mask, ProjectionIndex::LABEL_EDGE)) {
            label_edge_index = std::make_unique<BPlusTree<2>>(rel_dir + "/label_edge");
        }
    }

    // Open optional property indexes — NOT gated by IndexSet (Spec #3 §3.4).
    if (features.include_node_properties) {
        node_key_value_index = std::make_unique<BPlusTree<3>>(rel_dir + "/node_key_value");
        key_value_node_index = std::make_unique<BPlusTree<3>>(rel_dir + "/key_value_node");
    }
    if (features.include_edge_properties) {
        edge_key_value_index = std::make_unique<BPlusTree<3>>(rel_dir + "/edge_key_value");
        key_value_edge_index = std::make_unique<BPlusTree<3>>(rel_dir + "/key_value_edge");
    }
}

// Dispatcher for the serialized scan pipeline (Spec #2, Task 5).
// Routes a single-bit ProjectionIndex value to the corresponding
// private build_<name>_index_() helper. The default: clause catches
// NONE, ALL_NODE, ALL_EDGE, ALL, and any unknown bit pattern, enforcing
// single-bit-only semantics. Callers (Task 10's finalize_serialized_)
// iterate over single-bit enumerators via enabled_indexes_().
void ProjectionStorage::build_one_index(ProjectionIndex which) {
    switch (which) {
        case ProjectionIndex::NODES:           build_nodes_index_();          break;
        case ProjectionIndex::NODE_LABEL:      build_node_label_index_();     break;
        case ProjectionIndex::LABEL_NODE:      build_label_node_index_();     break;
        case ProjectionIndex::NODE_KEY_VALUE:  build_node_key_value_index_(); break;
        case ProjectionIndex::KEY_VALUE_NODE:  build_key_value_node_index_(); break;
        case ProjectionIndex::FROM_TO_EDGE:    build_from_to_edge_index_();   break;
        case ProjectionIndex::TO_FROM_EDGE:    build_to_from_edge_index_();   break;
        case ProjectionIndex::EDGE_DIRECTION:  build_edge_direction_index_(); break;
        case ProjectionIndex::EDGE_FROM_TO:    build_edge_from_to_index_();   break;
        case ProjectionIndex::EDGE_N1_N2:      build_edge_n1_n2_index_();     break;
        case ProjectionIndex::EDGE_LABEL:      build_edge_label_index_();     break;
        case ProjectionIndex::LABEL_EDGE:      build_label_edge_index_();     break;
        case ProjectionIndex::EDGE_KEY_VALUE:  build_edge_key_value_index_(); break;
        case ProjectionIndex::KEY_VALUE_EDGE:  build_key_value_edge_index_(); break;
        default:
            throw std::invalid_argument(
                "build_one_index: must be a single-bit ProjectionIndex value (got 0x" +
                std::to_string(static_cast<uint32_t>(which)) + ")");
    }
}

// Explicit template instantiations for the record sizes used
template size_t ProjectionStorage::build_index_bulk<1>(std::vector<Record<1>>&, const std::string&);
template size_t ProjectionStorage::build_index_bulk<2>(std::vector<Record<2>>&, const std::string&);
template size_t ProjectionStorage::build_index_bulk<3>(std::vector<Record<3>>&, const std::string&);

void ProjectionStorage::initialize_streaming_buffers() {
    // =========================================================================
    // Initialize streaming record buffers for memory-bounded bulk import
    // =========================================================================
    //
    // Each buffer has a configurable memory threshold (default 64 MB).
    // When the threshold is exceeded, records are spilled to temporary files.
    // This enables building projections of arbitrary size with bounded memory.
    //
    // Total memory budget: ~512 MB for all buffers (vs 12-14 GB for vectors)
    //
    // Spill path override:
    //   If the env var MDB_PROJECTION_SPILL_DIR is set, spill files are
    //   written under <env>/<projection_name>/tmp_* instead of inside the
    //   projection directory. This lets operators redirect spill I/O to a
    //   different volume (HDD archive, tmpfs, separate SSD) when the DB
    //   volume is disk-constrained, without affecting where the final B+Tree
    //   indexes and catalog live. Spills are ephemeral so no persistent data
    //   leaves the projection dir. See analysis doc §3.C.
    // =========================================================================

    std::string spill_base = projection_dir;
    if (const char* env_dir = std::getenv("MDB_PROJECTION_SPILL_DIR")) {
        if (env_dir[0] != '\0') {
            // Derive a subdir named after the projection so concurrent
            // projection builds don't collide. Extracting the trailing path
            // component of projection_dir (e.g. "papers100M_proj" from
            // "data/dbs/gql/papers100M/projections/papers100M_proj").
            std::filesystem::path pd(projection_dir);
            std::string subname = pd.filename().string();
            if (subname.empty()) {
                subname = "projection";
            }
            spill_base = std::string(env_dir) + "/" + subname;
            std::error_code ec;
            std::filesystem::create_directories(spill_base, ec);
            if (ec) {
                throw std::runtime_error(
                    "MDB_PROJECTION_SPILL_DIR='" + std::string(env_dir) +
                    "': cannot create spill directory '" + spill_base +
                    "': " + ec.message());
            }
        }
    }

    // Required index buffers (always created)
    node_records_buffer_ = std::make_unique<StreamingRecordBuffer<1>>(
        spill_base + "/tmp_nodes", STREAMING_BUFFER_THRESHOLD);

    from_to_records_buffer_ = std::make_unique<StreamingRecordBuffer<3>>(
        spill_base + "/tmp_from_to", STREAMING_BUFFER_THRESHOLD);

    to_from_records_buffer_ = std::make_unique<StreamingRecordBuffer<3>>(
        spill_base + "/tmp_to_from", STREAMING_BUFFER_THRESHOLD);

    direction_records_buffer_ = std::make_unique<StreamingRecordBuffer<2>>(
        spill_base + "/tmp_direction", STREAMING_BUFFER_THRESHOLD);

    edge_from_to_records_buffer_ = std::make_unique<StreamingRecordBuffer<3>>(
        spill_base + "/tmp_edge_from_to", STREAMING_BUFFER_THRESHOLD);

    edge_n1_n2_records_buffer_ = std::make_unique<StreamingRecordBuffer<3>>(
        spill_base + "/tmp_edge_n1_n2", STREAMING_BUFFER_THRESHOLD);

    // Optional label index buffers (always created for simplicity, but may remain empty)
    node_label_records_buffer_ = std::make_unique<StreamingRecordBuffer<2>>(
        spill_base + "/tmp_node_label", STREAMING_BUFFER_THRESHOLD);

    label_node_records_buffer_ = std::make_unique<StreamingRecordBuffer<2>>(
        spill_base + "/tmp_label_node", STREAMING_BUFFER_THRESHOLD);

    edge_label_records_buffer_ = std::make_unique<StreamingRecordBuffer<2>>(
        spill_base + "/tmp_edge_label", STREAMING_BUFFER_THRESHOLD);

    label_edge_records_buffer_ = std::make_unique<StreamingRecordBuffer<2>>(
        spill_base + "/tmp_label_edge", STREAMING_BUFFER_THRESHOLD);

    // Optional property index buffers (always created for simplicity, but may remain empty)
    node_key_value_records_buffer_ = std::make_unique<StreamingRecordBuffer<3>>(
        spill_base + "/tmp_node_kv", STREAMING_BUFFER_THRESHOLD);

    key_value_node_records_buffer_ = std::make_unique<StreamingRecordBuffer<3>>(
        spill_base + "/tmp_kv_node", STREAMING_BUFFER_THRESHOLD);

    edge_key_value_records_buffer_ = std::make_unique<StreamingRecordBuffer<3>>(
        spill_base + "/tmp_edge_kv", STREAMING_BUFFER_THRESHOLD);

    key_value_edge_records_buffer_ = std::make_unique<StreamingRecordBuffer<3>>(
        spill_base + "/tmp_kv_edge", STREAMING_BUFFER_THRESHOLD);
}

} // namespace GQL
