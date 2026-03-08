#include "hnsw_index.h"

#include <cstring>
#include <filesystem>
#include <memory>

#include "graph_models/common/datatypes/tensor/operations/avx.h"
#include "misc/logger.h"
#include "graph_models/quad_model/quad_model.h"
#include "storage/index/hnsw/hnsw_prefetch.h"
#include "storage/index/hnsw/hnsw_thread_pool.h"
#include "graph_models/quad_model/quad_object_id.h"
#include "graph_models/rdf_model/conversions.h"
#include "graph_models/rdf_model/rdf_model.h"
#include "query/optimizer/quad_model/plan/property_plan.h"
#include "query/optimizer/rdf_model/plan/triple_plan.h"
#include "query/query_context.h"
#include "storage/index/hnsw/hnsw_heap.h"
#include "storage/index/hnsw/hnsw_index_manager.h"
#include "storage/index/hnsw/hnsw_visited_set.h"
#include "system/file_manager.h"

namespace fs = std::filesystem;

namespace HNSW {
std::unique_ptr<HNSWIndex> HNSWIndex::create(
    const std::string& hnsw_index_name,
    uint64_t dimensions,
    uint64_t max_neighbors,
    uint64_t n_candidates_insertion,
    MetricFuncType metric_func
)
{
    const auto relative_index_path = fs::path(HNSWIndexManager::HNSW_INDEX_DIR) / hnsw_index_name;
    const auto absolute_index_path = file_manager.get_file_path(relative_index_path);

    // create_directories returns false when directory already exists, not just on failure
    std::error_code ec;
    fs::create_directories(absolute_index_path, ec);
    if (ec && ec != std::errc::file_exists) {
        throw std::runtime_error("Could not create directories: " + absolute_index_path + " (" + ec.message() + ")");
    }
    // Verify the directory actually exists now
    if (!fs::exists(absolute_index_path)) {
        throw std::runtime_error("Could not create directories: " + absolute_index_path);
    }

    HNSWIndexParams params {};
    params.entry_point_id = 0;
    params.dimensions = dimensions;
    params.layers = 1;
    params.M = max_neighbors;
    params.ef_construction = n_candidates_insertion;

    return std::make_unique<HNSWIndex>(params, metric_func);
}

std::unique_ptr<HNSWIndex> HNSWIndex::load(const std::string& hnsw_index_name, MetricFuncType metric_func)
{
    const auto relative_index_path = fs::path(HNSWIndexManager::HNSW_INDEX_DIR) / hnsw_index_name;
    const auto absolute_index_path = file_manager.get_file_path(relative_index_path);
    const auto absolute_data_file_path = fs::path(absolute_index_path) / HNSWIndex::DATA_FILENAME;

    std::fstream ifs(absolute_data_file_path, std::ios::in | std::ios::binary);

    // load params
    HNSWIndex::HNSWIndexParams params;
    ifs.read(reinterpret_cast<char*>(&params), sizeof(HNSWIndex::HNSWIndexParams));
    if (!ifs.good()) {
        throw std::runtime_error("Could not read index file");
    }

    uint64_t num_nodes;
    ifs.read(reinterpret_cast<char*>(&num_nodes), sizeof(uint64_t));
    if (!ifs.good()) {
        throw std::runtime_error("Could not read index file");
    }

    std::vector<HNSWNode> node_storage;
    std::vector<std::vector<HNSWEntryFlatMap>> node_neighbors;

    node_storage.reserve(num_nodes);
    node_neighbors.reserve(num_nodes);

    for (uint32_t node_id = 0; node_id < num_nodes; ++node_id) {
        // read node
        HNSWNode node {};
        ifs.read(reinterpret_cast<char*>(&node), sizeof(HNSWNode));
        if (!ifs.good()) {
            throw std::runtime_error("Could not read index file");
        }
        node_storage.emplace_back(std::move(node));

        // read neighbors
        uint64_t num_layers;
        ifs.read(reinterpret_cast<char*>(&num_layers), sizeof(uint64_t));
        if (!ifs.good()) {
            throw std::runtime_error("Could not read index file");
        }

        node_neighbors.emplace_back(std::vector<HNSWEntryFlatMap>(num_layers));

        for (uint64_t layer_i = 0; layer_i < num_layers; ++layer_i) {
            uint64_t num_neighbors_at_layer_i;
            ifs.read(reinterpret_cast<char*>(&num_neighbors_at_layer_i), sizeof(uint64_t));
            if (!ifs.good()) {
                throw std::runtime_error("Could not read index file");
            }

            Entry entry_buf;
            for (uint64_t neighbor_i = 0; neighbor_i < num_neighbors_at_layer_i; ++neighbor_i) {
                ifs.read(reinterpret_cast<char*>(&entry_buf), sizeof(entry_buf));
                if (!ifs.good()) {
                    throw std::runtime_error("Could not read index file");
                }
                node_neighbors[node_id][layer_i].emplace(entry_buf);
            }
        }
    }

    // read tombstones
    using block_type = boost::dynamic_bitset<>::block_type;
    boost::dynamic_bitset<> tombstone_bitset(node_storage.size());

    std::vector<block_type> tombstone_blocks(tombstone_bitset.num_blocks());
    ifs.read(reinterpret_cast<char*>(tombstone_blocks.data()), sizeof(block_type) * tombstone_blocks.size());
    if (!ifs.good()) {
        throw std::runtime_error("Could not read index file");
    }
    boost::from_block_range(tombstone_blocks.begin(), tombstone_blocks.end(), tombstone_bitset);

    // read raw embeddings flag and data (for GNN indexes)
    bool uses_raw = false;
    std::vector<float> raw_embeddings;
    std::vector<float> raw_embedding_norms;

    // Try to read the raw embeddings flag (may not exist in older index files)
    ifs.read(reinterpret_cast<char*>(&uses_raw), sizeof(bool));
    if (ifs.good() && uses_raw) {
        uint64_t raw_size = 0;
        ifs.read(reinterpret_cast<char*>(&raw_size), sizeof(uint64_t));
        if (ifs.good() && raw_size > 0) {
            raw_embeddings.resize(raw_size);
            ifs.read(reinterpret_cast<char*>(raw_embeddings.data()), sizeof(float) * raw_size);
            if (!ifs.good()) {
                throw std::runtime_error("Could not read raw embeddings from index file");
            }
        }

        // Try to read pre-computed norms (may not exist in older index files)
        uint64_t norms_size = 0;
        ifs.read(reinterpret_cast<char*>(&norms_size), sizeof(uint64_t));
        if (ifs.good() && norms_size > 0) {
            raw_embedding_norms.resize(norms_size);
            ifs.read(reinterpret_cast<char*>(raw_embedding_norms.data()), sizeof(float) * norms_size);
            // Don't throw if norms can't be read - they can be recomputed
        }
    }

    auto index = std::make_unique<HNSWIndex>(
        params,
        metric_func,
        std::move(node_storage),
        std::move(node_neighbors),
        std::move(tombstone_bitset)
    );

    // Set raw embeddings if present
    if (uses_raw && !raw_embeddings.empty()) {
        index->uses_raw_embeddings_ = true;
        index->raw_embeddings_ = std::move(raw_embeddings);

        // Set norms if present, otherwise recompute them
        if (!raw_embedding_norms.empty()) {
            index->raw_embedding_norms_ = std::move(raw_embedding_norms);
        } else {
            // Recompute norms for backwards compatibility with older index files
            index->compute_embedding_norms();
        }
    }

    return index;
}

HNSWHeap HNSWIndex::query(
    const tensor::Tensor<float>& query_tensor,
    uint64_t num_neighbors,
    uint64_t num_candidates,
    HNSWVisitedSet* visited_set,
    HNSWHeap* discarded_heap_ptr
)
{
    const uint64_t top_layer = params.layers - 1;

    // Initialize with entry point
    const auto entry_tensor = Common::Conversions::to_tensor<float>(
        node_storage[params.entry_point_id].tensor_oid
    );
    const float entry_dist = metric_func(query_tensor, entry_tensor);
    std::vector<Entry> entry_points;
    entry_points.emplace_back(entry_dist, params.entry_point_id);

    // find the best set entry points for layer 0
    for (uint64_t current_layer = top_layer; current_layer > 0; --current_layer) {
        const auto current_layer_nns = search_at_layer<true>(
            query_tensor,
            entry_points,
            1,
            current_layer,
            nullptr,
            nullptr
        );
        assert(current_layer_nns.size() == 1);
        entry_points[0] = current_layer_nns.get_min();
    }

    auto nearest_neighbors = search_at_layer<true>(
        query_tensor,
        entry_points,
        num_candidates,
        0,
        visited_set,
        discarded_heap_ptr
    );

    if (discarded_heap_ptr) {
        // modify inplace and store discarded
        while (nearest_neighbors.size() > num_neighbors) {
            discarded_heap_ptr->push(nearest_neighbors.get_max());
            nearest_neighbors.pop_max();
        }
        return nearest_neighbors;
    }

    const bool is_inplace_cheaper = nearest_neighbors.size() < 2 * num_neighbors;
    if (is_inplace_cheaper) {
        // modify inplace
        while (nearest_neighbors.size() > num_neighbors) {
            nearest_neighbors.pop_max();
        }
        return nearest_neighbors;
    }

    // create a new heap
    HNSWHeap res(num_neighbors);
    while (res.size() < num_neighbors) {
        res.push(nearest_neighbors.get_min());
        nearest_neighbors.pop_min();
    }
    return res;
}

std::unique_ptr<HNSWQueryIterator> HNSWIndex::query_iterator(
    const bool* interruption_requested,
    tensor::Tensor<float>&& query_tensor,
    uint64_t num_neighbors,
    uint64_t num_candidates
)
{
    return std::make_unique<HNSWQueryIterator>(
        interruption_requested,
        this,
        std::move(query_tensor),
        num_neighbors,
        num_candidates
    );
}

HNSWHeap HNSWIndex::resume_query(
    const tensor::Tensor<float>& query_tensor,
    uint64_t batch_size,
    const std::vector<Entry>& entry_points,
    HNSWVisitedSet* visited_set_ptr,
    HNSWHeap* discarded_heap_ptr
)
{
    return search_at_layer<true>(
        query_tensor,
        entry_points,
        batch_size,
        0,
        visited_set_ptr,
        discarded_heap_ptr
    );
}

uint_fast32_t HNSWIndex::index_predicate(const std::string& predicate)
{
    const auto subject_var = get_query_ctx().get_internal_var();
    const auto predicate_val = SPARQL::Conversions::pack_iri(predicate);
    const auto object_var = get_query_ctx().get_internal_var();

    const auto triple_plan = SPARQL::TriplePlan(subject_var, predicate_val, object_var);
    auto triple_plan_iter = triple_plan.get_binding_iter();

    Binding binding(get_query_ctx().get_var_size());
    triple_plan_iter->begin(binding);

    const std::size_t num_expected_insertions = rdf_model.catalog.get_predicate_count(predicate_val.id);

    node_storage.reserve(num_expected_insertions);
    node_neighbors_at_layer.reserve(num_expected_insertions);

    uint_fast32_t total_inserted_elements = 0;
    while (triple_plan_iter->next()) {
        const auto object_oid = binding[object_var];
        const auto subject_oid = binding[subject_var];

        if (index_single<false>(subject_oid, object_oid)) {
            ++total_inserted_elements;
        }
    }

    return total_inserted_elements;
}

uint_fast32_t HNSWIndex::index_property(const std::string& key)
{
    const auto object_var = get_query_ctx().get_internal_var();
    const auto key_val = QuadObjectId::get_string(key);
    const auto value_var = get_query_ctx().get_internal_var();

    const auto property_plan = PropertyPlan(object_var, key_val, value_var);

    auto property_plan_iter = property_plan.get_binding_iter();

    Binding binding(get_query_ctx().get_var_size());
    property_plan_iter->begin(binding);

    auto it = quad_model.catalog.key2total_count.find(key_val.id);
    std::size_t num_expected_insertions = 0;
    if (it != quad_model.catalog.key2total_count.end()) {
        num_expected_insertions = it->second;
    }

    node_storage.reserve(num_expected_insertions);
    node_neighbors_at_layer.reserve(num_expected_insertions);

    uint_fast32_t total_inserted_elements { 0 };
    while (property_plan_iter->next()) {
        const auto object_oid = binding[object_var];
        const auto value_oid = binding[value_var];

        if (index_single<false>(object_oid, value_oid)) {
            ++total_inserted_elements;
        }
    }

    return total_inserted_elements;
}

template<bool CheckTombstones>
bool HNSWIndex::index_single(ObjectId ref_object_id, ObjectId tensor_object_id)
{
    const auto gen_t = RDF_OID::get_generic_type(tensor_object_id);
    if (gen_t != RDF_OID::GenericType::TENSOR) {
        // Object is not a tensor
        return false;
    }

    if constexpr (CheckTombstones) {
        HNSWNode hnsw_node(ref_object_id, tensor_object_id);

        const auto it = hnsw_node2node_id.find(hnsw_node);
        if (it != hnsw_node2node_id.end()) {
            // entry is already in graph, unset its tombstone
            assert(node_tombstones.test(it->second) && "inserted an already set triple. Something went wrong");
            node_tombstones.set(it->second, false);
            has_changes = true;
            return true;
        }
    }

    const auto query_tensor = Common::Conversions::to_tensor<float>(tensor_object_id);
    if (query_tensor.size() != params.dimensions) {
        // Tensor dimension does not match
        return false;
    }

    // Create new node at a random layer
    const uint64_t node_top_layer = get_random_layer();
    const uint64_t node_id = create_new_node(ref_object_id, tensor_object_id, node_top_layer);

    // Initialize with entry point
    const auto entry_tensor = Common::Conversions::to_tensor<float>(
        node_storage[params.entry_point_id].tensor_oid
    );
    const auto entry_dist = metric_func(query_tensor, entry_tensor);
    std::vector<Entry> entry_points;
    entry_points.emplace_back(entry_dist, params.entry_point_id);

    // Update the best entry point for layers in range [top_layer, node_top_layer)
    const uint64_t top_layer = params.layers - 1;
    for (uint64_t current_layer = top_layer; current_layer > node_top_layer; --current_layer) {
        const auto current_layer_nns = search_at_layer<CheckTombstones>(
            query_tensor,
            entry_points,
            1,
            current_layer,
            nullptr,
            nullptr
        );
        assert(current_layer_nns.size() == 1);
        entry_points[0] = current_layer_nns.get_min();
    }

    // Insert in node's layer and the layers below
    for (int64_t current_layer = std::min(node_top_layer, top_layer); current_layer >= 0; --current_layer) {
        auto current_layer_nns = search_at_layer<CheckTombstones>(
            query_tensor,
            entry_points,
            params.ef_construction,
            current_layer,
            nullptr,
            nullptr
        );
        const uint64_t max_neighbors = current_layer == 0 ? M0 : Mi;

        auto current_layer_top_k = current_layer_nns.extract_n_min(max_neighbors);

        // Mutually connect node with its new neighbors
        set_neighbors_at_layer(node_id, current_layer, current_layer_top_k);

        // Update entry points
        entry_points = std::move(current_layer_top_k);
    }

    if (node_top_layer > top_layer) {
        params.entry_point_id = node_id;
        params.layers = node_top_layer + 1;
    };

    has_changes = true;

    return true;
}

bool HNSWIndex::remove_single(ObjectId ref_object_id, ObjectId tensor_object_id)
{
    HNSWNode hnsw_node(ref_object_id, tensor_object_id);

    const auto it = hnsw_node2node_id.find(hnsw_node);
    assert(it != hnsw_node2node_id.end());

    node_tombstones.set(it->second, true);

    has_changes = true;

    return true;
}

HNSWIndex::HNSWIndex(HNSWIndexParams params_, MetricFuncType metric_func_) :
    metric_func { metric_func_ },
    params { params_ }
{
    init_constants();
}

HNSWIndex::HNSWIndex(
    HNSWIndexParams params_,
    MetricFuncType metric_func_,
    std::vector<HNSWNode>&& node_storage_,
    std::vector<std::vector<HNSWEntryFlatMap>>&& node_neighbors_at_layer_,
    boost::dynamic_bitset<>&& node_tombstones_
) :
    metric_func { metric_func_ },
    params { params_ },
    node_storage { std::move(node_storage_) },
    node_neighbors_at_layer { std::move(node_neighbors_at_layer_) },
    node_tombstones { node_tombstones_ }
{
    init_constants();

    for (std::size_t i = 0; i < node_storage.size(); ++i) {
        hnsw_node2node_id.emplace(node_storage[i], i);
    }
}

template<bool CheckTombstones>
HNSWHeap HNSWIndex::search_at_layer(
    const tensor::Tensor<float>& query_tensor,
    const std::vector<Entry>& entry_points,
    uint64_t num_neighbors,
    uint64_t layer,
    HNSWVisitedSet* visited_set_ptr,
    HNSWHeap* discarded_heap_ptr
)
{
    assert(!entry_points.empty());
    assert(num_neighbors > 0);
    assert(entry_points.size() <= num_neighbors);

    HNSWVisitedSet visited_set_;
    HNSWHeap candidates_heap(num_neighbors);
    HNSWHeap top_k_heap(num_neighbors);

    if (visited_set_ptr != nullptr) {
        for (std::size_t i = 0; i < entry_points.size(); ++i) {
            candidates_heap.emplace(entry_points[i]);
            top_k_heap.emplace(entry_points[i]);
            visited_set_ptr->emplace(entry_points[i].node_id);
        }
    } else {
        visited_set_.reserve(node_storage.size());
        // no visited set was provided, initialize a new one
        for (std::size_t i = 0; i < entry_points.size(); ++i) {
            visited_set_.emplace(entry_points[i].node_id);
            candidates_heap.emplace(entry_points[i]);
            top_k_heap.emplace(entry_points[i]);
        }

        visited_set_ptr = &visited_set_;
    }

    while (!candidates_heap.empty()) {
        // get best candidate and worst result
        const Entry candidate_entry = candidates_heap.get_min();
        if (candidate_entry.distance > top_k_heap.get_max().distance) {
            // no further result improvement
            break;
        }

        candidates_heap.pop_min();

        // explore candidate's neighborhood
        for (const Entry& candidate_neighbor_entry : node_neighbors_at_layer[candidate_entry.node_id][layer])
        {
            if (visited_set_ptr->contains(candidate_neighbor_entry.node_id)) {
                continue;
            }
            visited_set_ptr->emplace(candidate_neighbor_entry.node_id);

            const auto candidate_neighbor_tensor = Common::Conversions::to_tensor<float>(
                node_storage[candidate_neighbor_entry.node_id].tensor_oid
            );
            const float candidate_neighbor_dist = metric_func(query_tensor, candidate_neighbor_tensor);

            if constexpr (CheckTombstones) {
                if (node_tombstones.test(candidate_neighbor_entry.node_id)) {
                    // candidate_neighbor_entry.node_id is marked as deleted,
                    // it can be explored but not added to the top_k_heap
                    candidates_heap.emplace(candidate_neighbor_dist, candidate_neighbor_entry.node_id);
                    continue;
                }
            }

            if (top_k_heap.size() < num_neighbors) {
                // always add, not enough results
                candidates_heap.emplace(candidate_neighbor_dist, candidate_neighbor_entry.node_id);
                top_k_heap.emplace(candidate_neighbor_dist, candidate_neighbor_entry.node_id);
                continue;
            }

            if (candidate_neighbor_dist < top_k_heap.get_max().distance) {
                // the candidate improves the result, furthest result must be removed
                if (discarded_heap_ptr != nullptr) {
                    discarded_heap_ptr->emplace(top_k_heap.get_max());
                }
                candidates_heap.emplace(candidate_neighbor_dist, candidate_neighbor_entry.node_id);
                top_k_heap.pop_max();
                top_k_heap.emplace(candidate_neighbor_dist, candidate_neighbor_entry.node_id);
                continue;
            }

            if (discarded_heap_ptr != nullptr) {
                // the candidate does not improve the result
                discarded_heap_ptr->emplace(candidate_neighbor_dist, candidate_neighbor_entry.node_id);
            }
        }
    }

    return top_k_heap;
}

void HNSWIndex::set_neighbors_at_layer(
    uint32_t node_id,
    uint64_t layer,
    const std::vector<Entry>& new_neighbors
)
{
    const auto max_neighbors = layer == 0 ? M0 : Mi;
    assert(new_neighbors.size() <= max_neighbors);

    node_neighbors_at_layer[node_id][layer].reserve(new_neighbors.size());

    for (const Entry& new_neighbor_entry : new_neighbors) {
        // node -> neighbor
        node_neighbors_at_layer[node_id][layer].emplace(new_neighbor_entry);

        // Bounds check: ensure neighbor node exists and has this layer
        if (new_neighbor_entry.node_id >= node_neighbors_at_layer.size()) {
            continue;  // Neighbor doesn't exist yet
        }
        auto& neighbor_layers = node_neighbors_at_layer[new_neighbor_entry.node_id];
        if (layer >= neighbor_layers.size()) {
            continue;  // Neighbor doesn't have this layer
        }

        // neighbor -> node
        auto& new_neighbor_neighbors_at_layer = neighbor_layers[layer];
        if (new_neighbor_neighbors_at_layer.size() < max_neighbors) {
            // max_neighbors not reached, just add the connection
            new_neighbor_neighbors_at_layer.emplace(new_neighbor_entry.distance, node_id);
        } else {
            // replace if the connection is better than the furthest one
            const auto furthest_it = --new_neighbor_neighbors_at_layer.end();
            if (new_neighbor_entry.distance < furthest_it->distance) {
                new_neighbor_neighbors_at_layer.erase(furthest_it);
                new_neighbor_neighbors_at_layer.emplace(new_neighbor_entry.distance, node_id);
            }
        }
    }
}

// ==================== Raw Embeddings Support (for GNN) ====================

void HNSWIndex::compute_embedding_norms()
{
    if (!uses_raw_embeddings_ || raw_embeddings_.empty()) {
        return;
    }

    const uint64_t num_nodes = raw_embeddings_.size() / params.dimensions;
    raw_embedding_norms_.resize(num_nodes);

    for (uint64_t i = 0; i < num_nodes; ++i) {
        const float* emb = get_raw_embedding(i);
        if (emb == nullptr) {
            raw_embedding_norms_[i] = 0.0f;  // Safe default for missing embeddings
            continue;
        }
        raw_embedding_norms_[i] = tensor::avx::computation<float>::norm_squared(
            emb, params.dimensions
        );
    }
}

uint_fast32_t HNSWIndex::index_from_raw_embeddings(
    const float* embeddings,
    uint64_t num_nodes,
    uint64_t dim
)
{
    if (dim != params.dimensions) {
        throw std::runtime_error(
            "Embedding dimension mismatch: expected " + std::to_string(params.dimensions) +
            ", got " + std::to_string(dim)
        );
    }

    // Switch to raw embeddings mode
    uses_raw_embeddings_ = true;

    // Copy embeddings into internal storage
    raw_embeddings_.resize(num_nodes * dim);
    std::memcpy(raw_embeddings_.data(), embeddings, num_nodes * dim * sizeof(float));

    // Reserve space for nodes
    node_storage.reserve(num_nodes);
    node_neighbors_at_layer.reserve(num_nodes);

    uint_fast32_t total_indexed = 0;

    for (uint64_t node_id = 0; node_id < num_nodes; ++node_id) {
        const float* node_embedding = embeddings + (node_id * dim);

        // Create ObjectId for the node (GQL node encoding)
        ObjectId node_oid(ObjectId::MASK_NODE | node_id);
        // Use a special marker for raw embeddings (node_id encoded in tensor_oid)
        ObjectId tensor_marker(ObjectId::MASK_TENSOR_FLOAT_TMP | node_id);

        // Create new node at a random layer
        const uint64_t node_top_layer = get_random_layer();
        const uint64_t internal_node_id = create_new_node(node_oid, tensor_marker, node_top_layer);

        if (internal_node_id == 0) {
            // First node - just update params
            params.entry_point_id = 0;
            params.layers = node_top_layer + 1;
            ++total_indexed;
            continue;
        }

        // Get entry point embedding
        const float* entry_embedding = get_raw_embedding(params.entry_point_id);
        if (entry_embedding == nullptr) {
            logger.error() << "index_from_raw_embeddings: entry_embedding is NULL!";
            continue;
        }

        // Compute distance to entry point
        tensor::Tensor<float> query_tensor(node_embedding, node_embedding + dim);
        tensor::Tensor<float> entry_tensor(entry_embedding, entry_embedding + dim);
        const float entry_dist = metric_func(query_tensor, entry_tensor);

        std::vector<Entry> entry_points;
        entry_points.emplace_back(entry_dist, params.entry_point_id);

        // Update the best entry point for layers in range [top_layer, node_top_layer)
        const uint64_t top_layer = params.layers - 1;
        for (uint64_t current_layer = top_layer; current_layer > node_top_layer; --current_layer) {
            auto current_layer_nns = search_at_layer_raw<false>(
                node_embedding,
                entry_points,
                1,
                current_layer,
                nullptr,
                nullptr
            );
            assert(current_layer_nns.size() == 1);
            entry_points[0] = current_layer_nns.get_min();
        }

        // Insert in node's layer and the layers below
        for (int64_t current_layer = std::min(node_top_layer, top_layer); current_layer >= 0; --current_layer) {
            auto current_layer_nns = search_at_layer_raw<false>(
                node_embedding,
                entry_points,
                params.ef_construction,
                current_layer,
                nullptr,
                nullptr
            );
            const uint64_t max_neighbors = current_layer == 0 ? M0 : Mi;

            auto current_layer_top_k = current_layer_nns.extract_n_min(max_neighbors);

            // Mutually connect node with its new neighbors
            set_neighbors_at_layer(internal_node_id, current_layer, current_layer_top_k);

            // Update entry points
            entry_points = std::move(current_layer_top_k);
        }

        if (node_top_layer > top_layer) {
            params.entry_point_id = internal_node_id;
            params.layers = node_top_layer + 1;
        }

        ++total_indexed;
    }

    has_changes = true;

    // Compute norms for optimized search
    compute_embedding_norms();

    return total_indexed;
}

uint_fast32_t HNSWIndex::index_from_raw_embeddings_parallel(
    const float* embeddings,
    uint64_t num_nodes,
    uint64_t dim,
    size_t num_threads
)
{
    if (dim != params.dimensions) {
        throw std::runtime_error(
            "Embedding dimension mismatch: expected " + std::to_string(params.dimensions) +
            ", got " + std::to_string(dim)
        );
    }

    // Fall back to sequential for small datasets or single thread
    if (num_threads <= 1 || num_nodes < 1000) {
        return index_from_raw_embeddings(embeddings, num_nodes, dim);
    }

    // ========== Phase 1: Setup (single-threaded) ==========
    uses_raw_embeddings_ = true;

    // Copy embeddings into internal storage
    raw_embeddings_.resize(num_nodes * dim);
    std::memcpy(raw_embeddings_.data(), embeddings, num_nodes * dim * sizeof(float));

    // Pre-allocate storage
    node_storage.reserve(num_nodes);
    node_neighbors_at_layer.reserve(num_nodes);
    raw_embedding_norms_.resize(num_nodes);

    // Initialize per-node locks for fine-grained synchronization
    node_locks_ = std::make_unique<std::mutex[]>(num_nodes);

    ThreadPool pool(num_threads);

    // ========== Phase 2: Compute norms (parallel) ==========
    pool.parallel_for(0, num_nodes, [this, dim](size_t i) {
        const float* emb = get_raw_embedding(static_cast<uint32_t>(i));
        if (emb != nullptr) {
            raw_embedding_norms_[i] = tensor::avx::computation<float>::norm_squared(emb, dim);
        } else {
            raw_embedding_norms_[i] = 0.0f;  // Safe default for invalid embeddings
        }
    });

    // ========== Phase 3: Generate layer assignments (sequential, deterministic) ==========
    std::vector<uint64_t> node_layers(num_nodes);
    for (uint64_t i = 0; i < num_nodes; ++i) {
        node_layers[i] = get_random_layer();
    }

    // ========== Phase 4: Insert first node (establishes entry point) ==========
    {
        ObjectId node_oid(ObjectId::MASK_NODE | 0);
        ObjectId tensor_marker(ObjectId::MASK_TENSOR_FLOAT_TMP | 0);
        create_new_node(node_oid, tensor_marker, node_layers[0]);
        params.entry_point_id = 0;
        params.layers = node_layers[0] + 1;
    }

    // ========== Phase 5: Batch parallel insertion ==========
    const size_t batch_size = std::max(size_t(500), num_threads * 16);

    for (uint64_t batch_start = 1; batch_start < num_nodes; batch_start += batch_size) {
        const uint64_t batch_end = std::min(batch_start + batch_size, num_nodes);
        const size_t batch_count = batch_end - batch_start;

        // Structure to hold search results for each node in batch
        struct NodeSearchResult {
            std::vector<std::vector<Entry>> neighbors_by_layer;
        };
        std::vector<NodeSearchResult> batch_results(batch_count);

        // Phase 5a: Create nodes (sequential - modifies node_storage)
        for (uint64_t node_id = batch_start; node_id < batch_end; ++node_id) {
            ObjectId node_oid(ObjectId::MASK_NODE | node_id);
            ObjectId tensor_marker(ObjectId::MASK_TENSOR_FLOAT_TMP | node_id);
            create_new_node(node_oid, tensor_marker, node_layers[node_id]);
        }

        // Phase 5b: Parallel neighbor search (read-only graph access)
        pool.parallel_for(batch_start, batch_end, [&, this](size_t node_id) {
            const float* node_embedding = get_raw_embedding(static_cast<uint32_t>(node_id));
            if (node_embedding == nullptr) {
                return;  // Skip this node if embedding not accessible
            }
            const uint64_t node_top_layer = node_layers[node_id];

            // Get entry point info (under shared lock)
            uint64_t current_entry_point;
            uint64_t current_top_layer;
            {
                std::shared_lock<std::shared_mutex> lock(graph_mutex_);
                current_entry_point = params.entry_point_id;
                current_top_layer = params.layers - 1;
            }

            // Compute distance to entry point (with bounds checks)
            const float* entry_embedding = get_raw_embedding(static_cast<uint32_t>(current_entry_point));
            if (entry_embedding == nullptr) {
                return;  // Skip if entry point embedding not accessible
            }
            float entry_dist;
            if (node_id < raw_embedding_norms_.size() &&
                current_entry_point < raw_embedding_norms_.size()) {
                entry_dist = tensor::avx::computation<float>::cosine_distance_with_norms(
                    node_embedding, raw_embedding_norms_[node_id],
                    entry_embedding, raw_embedding_norms_[current_entry_point],
                    dim
                );
            } else {
                // Fallback: compute using standard metric
                tensor::Tensor<float> query_tensor(node_embedding, node_embedding + dim);
                tensor::Tensor<float> entry_tensor(entry_embedding, entry_embedding + dim);
                entry_dist = metric_func(query_tensor, entry_tensor);
            }

            std::vector<Entry> entry_points;
            entry_points.emplace_back(entry_dist, current_entry_point);

            // Navigate to node's layer
            {
                std::shared_lock<std::shared_mutex> lock(graph_mutex_);
                for (uint64_t layer = current_top_layer; layer > node_top_layer; --layer) {
                    auto layer_nns = search_at_layer_raw<false>(
                        node_embedding, entry_points, 1, layer, nullptr, nullptr
                    );
                    if (!layer_nns.empty()) {
                        entry_points[0] = layer_nns.get_min();
                    }
                }
            }

            // Search for neighbors at each layer
            auto& result = batch_results[node_id - batch_start];
            result.neighbors_by_layer.resize(node_top_layer + 1);

            {
                std::shared_lock<std::shared_mutex> lock(graph_mutex_);
                for (int64_t layer = std::min(node_top_layer, current_top_layer); layer >= 0; --layer) {
                    auto layer_nns = search_at_layer_raw<false>(
                        node_embedding, entry_points, params.ef_construction, layer, nullptr, nullptr
                    );
                    const uint64_t max_neighbors = layer == 0 ? M0 : Mi;
                    result.neighbors_by_layer[layer] = layer_nns.extract_n_min(max_neighbors);
                    entry_points = result.neighbors_by_layer[layer];
                }
            }
        });

        // Phase 5c: Connect nodes (sequential writes with fine-grained locking)
        {
            std::unique_lock<std::shared_mutex> lock(graph_mutex_);

            for (uint64_t node_id = batch_start; node_id < batch_end; ++node_id) {
                const auto& result = batch_results[node_id - batch_start];
                const uint64_t node_top_layer = node_layers[node_id];

                // Connect at each layer
                for (uint64_t layer = 0; layer <= node_top_layer && layer < result.neighbors_by_layer.size(); ++layer) {
                    const auto& layer_neighbors = result.neighbors_by_layer[layer];
                    const uint64_t max_neighbors = layer == 0 ? M0 : Mi;

                    for (const Entry& neighbor : layer_neighbors) {
                        // Forward edge: node -> neighbor (bounds check)
                        if (node_id < node_neighbors_at_layer.size() &&
                            layer < node_neighbors_at_layer[node_id].size()) {
                            node_neighbors_at_layer[node_id][layer].emplace(neighbor);
                        }

                        // Reverse edge: neighbor -> node (with per-node lock)
                        // Check that neighbor exists and has this layer
                        if (neighbor.node_id < node_neighbors_at_layer.size() &&
                            layer < node_neighbors_at_layer[neighbor.node_id].size()) {
                            std::lock_guard<std::mutex> node_lock(node_locks_[neighbor.node_id]);
                            auto& neighbor_edges = node_neighbors_at_layer[neighbor.node_id][layer];

                            if (neighbor_edges.size() < max_neighbors) {
                                neighbor_edges.emplace(neighbor.distance, node_id);
                            } else {
                                auto furthest = --neighbor_edges.end();
                                if (neighbor.distance < furthest->distance) {
                                    neighbor_edges.erase(furthest);
                                    neighbor_edges.emplace(neighbor.distance, node_id);
                                }
                            }
                        }
                    }
                }

                // Update entry point if this node has higher layer
                if (node_top_layer >= params.layers) {
                    params.entry_point_id = node_id;
                    params.layers = node_top_layer + 1;
                }
            }
        }

        nodes_inserted_ = batch_end;
    }

    has_changes = true;
    return num_nodes;
}

HNSWHeap HNSWIndex::query_raw(
    const float* query_embedding,
    uint64_t num_neighbors,
    uint64_t num_candidates
)
{
    if (!uses_raw_embeddings_) {
        throw std::runtime_error("query_raw() called on index not using raw embeddings mode");
    }

    if (node_storage.empty()) {
        return HNSWHeap(num_neighbors);
    }

    const uint64_t top_layer = params.layers - 1;

    // Initialize with entry point
    const float* entry_embedding = get_raw_embedding(params.entry_point_id);
    if (entry_embedding == nullptr) {
        // Entry point has no valid embedding — cannot search
        return HNSWHeap(num_neighbors);
    }
    tensor::Tensor<float> query_tensor(query_embedding, query_embedding + params.dimensions);
    tensor::Tensor<float> entry_tensor(entry_embedding, entry_embedding + params.dimensions);
    const float entry_dist = metric_func(query_tensor, entry_tensor);

    std::vector<Entry> entry_points;
    entry_points.emplace_back(entry_dist, params.entry_point_id);

    // Find the best entry points for layer 0
    for (uint64_t current_layer = top_layer; current_layer > 0; --current_layer) {
        auto current_layer_nns = search_at_layer_raw<true>(
            query_embedding,
            entry_points,
            1,
            current_layer,
            nullptr,
            nullptr
        );
        assert(current_layer_nns.size() == 1);
        entry_points[0] = current_layer_nns.get_min();
    }

    auto nearest_neighbors = search_at_layer_raw<true>(
        query_embedding,
        entry_points,
        num_candidates,
        0,
        nullptr,
        nullptr
    );

    // Trim to requested number of neighbors
    while (nearest_neighbors.size() > num_neighbors) {
        nearest_neighbors.pop_max();
    }

    return nearest_neighbors;
}

template<bool CheckTombstones>
HNSWHeap HNSWIndex::search_at_layer_raw(
    const float* query_embedding,
    const std::vector<Entry>& entry_points,
    uint64_t num_neighbors,
    uint64_t layer,
    HNSWVisitedSet* visited_set_ptr,
    HNSWHeap* discarded_heap_ptr
)
{
    assert(!entry_points.empty());
    assert(num_neighbors > 0);
    assert(entry_points.size() <= num_neighbors);

    HNSWVisitedSet visited_set_;
    HNSWHeap candidates_heap(num_neighbors);
    HNSWHeap top_k_heap(num_neighbors);

    if (visited_set_ptr != nullptr) {
        for (std::size_t i = 0; i < entry_points.size(); ++i) {
            candidates_heap.emplace(entry_points[i]);
            top_k_heap.emplace(entry_points[i]);
            visited_set_ptr->emplace(entry_points[i].node_id);
        }
    } else {
        visited_set_.reserve(node_storage.size());
        for (std::size_t i = 0; i < entry_points.size(); ++i) {
            visited_set_.emplace(entry_points[i].node_id);
            candidates_heap.emplace(entry_points[i]);
            top_k_heap.emplace(entry_points[i]);
        }
        visited_set_ptr = &visited_set_;
    }

    // Pre-compute query norm ONCE (optimization: avoids redundant computation)
    const float query_norm_sq = tensor::avx::computation<float>::norm_squared(
        query_embedding, params.dimensions
    );

    // Check if we have pre-computed norms available
    const bool use_optimized_distance = !raw_embedding_norms_.empty();

    while (!candidates_heap.empty()) {
        const Entry candidate_entry = candidates_heap.get_min();
        if (candidate_entry.distance > top_k_heap.get_max().distance) {
            break;
        }

        candidates_heap.pop_min();

        // Bounds check: ensure candidate node exists and has this layer
        if (candidate_entry.node_id >= node_neighbors_at_layer.size()) {
            continue;  // Skip invalid node reference
        }
        const auto& candidate_layers = node_neighbors_at_layer[candidate_entry.node_id];
        if (layer >= candidate_layers.size()) {
            continue;  // Node doesn't have this layer
        }
        const auto& neighbors = candidate_layers[layer];

        // Prefetch first few unvisited neighbors (hide memory latency)
        size_t prefetch_count = 0;
        const size_t num_nodes = raw_embeddings_.size() / params.dimensions;
        for (const auto& neighbor_entry : neighbors) {
            if (prefetch_count >= 3) break;  // Prefetch up to 3 ahead
            if (!visited_set_ptr->contains(neighbor_entry.node_id) &&
                neighbor_entry.node_id < num_nodes) {
                const float* prefetch_emb = get_raw_embedding(neighbor_entry.node_id);
                if (prefetch_emb == nullptr) continue;
                prefetch_embedding(prefetch_emb, params.dimensions);
                if (use_optimized_distance &&
                    neighbor_entry.node_id < raw_embedding_norms_.size()) {
                    prefetch_norm(raw_embedding_norms_.data(), neighbor_entry.node_id);
                }
                ++prefetch_count;
            }
        }

        // Explore candidate's neighborhood
        for (const Entry& candidate_neighbor_entry : neighbors) {
            if (visited_set_ptr->contains(candidate_neighbor_entry.node_id)) {
                continue;
            }
            // Bounds check for embedding access
            if (candidate_neighbor_entry.node_id >= num_nodes) {
                continue;  // Skip invalid node reference
            }

            // Get neighbor embedding with null check
            const float* neighbor_embedding = get_raw_embedding(candidate_neighbor_entry.node_id);
            if (neighbor_embedding == nullptr) {
                continue;  // Skip if embedding not accessible
            }

            visited_set_ptr->emplace(candidate_neighbor_entry.node_id);

            // Compute distance using optimized path if norms available
            float candidate_neighbor_dist;

            if (use_optimized_distance &&
                candidate_neighbor_entry.node_id < raw_embedding_norms_.size()) {
                // Use pre-computed norms (avoids redundant norm calculation)
                const float neighbor_norm_sq = raw_embedding_norms_[candidate_neighbor_entry.node_id];
                candidate_neighbor_dist = tensor::avx::computation<float>::cosine_distance_with_norms(
                    query_embedding, query_norm_sq,
                    neighbor_embedding, neighbor_norm_sq,
                    params.dimensions
                );
            } else {
                // Fallback to standard metric function
                tensor::Tensor<float> query_tensor(query_embedding, query_embedding + params.dimensions);
                tensor::Tensor<float> neighbor_tensor(neighbor_embedding, neighbor_embedding + params.dimensions);
                candidate_neighbor_dist = metric_func(query_tensor, neighbor_tensor);
            }

            if constexpr (CheckTombstones) {
                if (candidate_neighbor_entry.node_id < node_tombstones.size() &&
                    node_tombstones.test(candidate_neighbor_entry.node_id)) {
                    candidates_heap.emplace(candidate_neighbor_dist, candidate_neighbor_entry.node_id);
                    continue;
                }
            }

            if (top_k_heap.size() < num_neighbors) {
                candidates_heap.emplace(candidate_neighbor_dist, candidate_neighbor_entry.node_id);
                top_k_heap.emplace(candidate_neighbor_dist, candidate_neighbor_entry.node_id);
                continue;
            }

            if (candidate_neighbor_dist < top_k_heap.get_max().distance) {
                if (discarded_heap_ptr != nullptr) {
                    discarded_heap_ptr->emplace(top_k_heap.get_max());
                }
                candidates_heap.emplace(candidate_neighbor_dist, candidate_neighbor_entry.node_id);
                top_k_heap.pop_max();
                top_k_heap.emplace(candidate_neighbor_dist, candidate_neighbor_entry.node_id);
                continue;
            }

            if (discarded_heap_ptr != nullptr) {
                discarded_heap_ptr->emplace(candidate_neighbor_dist, candidate_neighbor_entry.node_id);
            }
        }
    }

    return top_k_heap;
}

} // namespace HNSW

template bool HNSW::HNSWIndex::index_single<false>(ObjectId, ObjectId);
template bool HNSW::HNSWIndex::index_single<true>(ObjectId, ObjectId);

// Explicit template instantiations for search_at_layer_raw
template HNSW::HNSWHeap HNSW::HNSWIndex::search_at_layer_raw<false>(const float*, const std::vector<HNSW::Entry>&, uint64_t, uint64_t, HNSW::HNSWVisitedSet*, HNSW::HNSWHeap*);
template HNSW::HNSWHeap HNSW::HNSWIndex::search_at_layer_raw<true>(const float*, const std::vector<HNSW::Entry>&, uint64_t, uint64_t, HNSW::HNSWVisitedSet*, HNSW::HNSWHeap*);
