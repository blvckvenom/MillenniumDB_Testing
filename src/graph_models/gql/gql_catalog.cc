#include "gql_catalog.h"

#include <algorithm>

#include "gnn/common/validate_name.h"
#include "query/exceptions.h"

GQLCatalog::GQLCatalog(const std::string& filename) :
    Catalog(filename)
{
    if (!is_empty()) {
        auto diff_minor_version = check_version("GQL", MODEL_ID, MAJOR_VERSION, MINOR_VERSION);

        // Read core fields (present in all versions)
        nodes_count = read_uint64();
        directed_edges_count = read_uint64();
        undirected_edges_count = read_uint64();
        node_labels_count = read_uint64();
        edge_labels_count = read_uint64();
        node_properties_count = read_uint64();
        edge_properties_count = read_uint64();
        equal_directed_edges_count = read_uint64();
        equal_undirected_edges_count = read_uint64();

        node_label2total_count = read_map();
        edge_label2total_count = read_map();
        node_key2total_count = read_map();
        edge_key2total_count = read_map();

        node_labels_str = read_strvec();
        edge_labels_str = read_strvec();

        node_labels2id = convert_strvec_to_map(node_labels_str);
        edge_labels2id = convert_strvec_to_map(edge_labels_str);

        node_keys_str = read_strvec();
        edge_keys_str = read_strvec();

        node_keys2id = convert_strvec_to_map(node_keys_str);
        edge_keys2id = convert_strvec_to_map(edge_keys_str);

        // --- GNN feature registry + HNSW metadata (version-dependent) ---
        // Compute the actual on-disk version for clarity and forward-compatibility.
        // Using on_disk_version (not diff_minor_version) means future version bumps
        // don't silently reassign existing case labels.
        auto on_disk_version = static_cast<uint8_t>(MINOR_VERSION - diff_minor_version);

#ifdef ENABLE_GNN
        hnsw_index_manager.init();
#endif

        switch (on_disk_version) {
            case 3: {
                // Current format: feature name registry + HNSW index metadata
                gnn_feature_names = read_strvec();

                const auto hnsw_count = read_uint64();
#ifdef ENABLE_GNN
                for (uint_fast32_t i = 0; i < hnsw_count; ++i) {
                    const auto name = read_string();
                    HNSW::HNSWIndexManager::HNSWIndexMetadata metadata;
                    metadata.metric_type = static_cast<HNSW::MetricType>(read_uint8());
                    metadata.predicate = read_string();
                    hnsw_index_manager.load_hnsw_index(name, metadata);
                }
#else
                // Skip HNSW metadata when GNN is disabled
                for (uint_fast32_t i = 0; i < hnsw_count; ++i) {
                    read_string();  // name
                    read_uint8();   // metric_type
                    read_string();  // predicate
                }
#endif
                break;
            }
            case 2:
                // v2 → v3 migration: feature names only, no HNSW metadata
                gnn_feature_names = read_strvec();
                has_changes = true;
                break;
            case 1:
                // v1 → v3 migration: read and discard old 3-field format.
                // We do NOT register "node_features" because the old data lives
                // in gnn_tensors/ (legacy FileGnnTensorStore shard format from v1). New data uses gnn_features/ (FeatureMatrix format).
                read_uint64(); // discard old has_gnn_tensors flag
                read_uint64(); // discard gnn_tensor_num_rows
                read_uint64(); // discard gnn_tensor_num_cols
                has_changes = true;
                break;
            case 0:
                // v0 → v3 migration: no GNN fields existed, nothing to read
                has_changes = true;
                break;
            default:
                throw std::runtime_error(
                    "GQLCatalog: unexpected on-disk minor version " +
                    std::to_string(on_disk_version));
        }
    } else {
        has_changes = true;
#ifdef ENABLE_GNN
        hnsw_index_manager.init();
#endif
    }
}

GQLCatalog::~GQLCatalog()
{
#ifdef ENABLE_GNN
    if (has_changes || hnsw_index_manager.has_changes()) {
#else
    if (has_changes) {
#endif
        save();
    }
}

void GQLCatalog::print(std::ostream& os)
{
    os << "-------------------------------------\n";
    os << "GQL Catalog:\n";
    os << "  Nodes:                " << nodes_count << "\n";
    os << "  Edges:                " << undirected_edges_count + directed_edges_count << "\n";
    os << "     Directed:          " << directed_edges_count << "\n";
    os << "     Undirected:        " << undirected_edges_count << "\n";
    os << "  Node Labels:          " << node_labels_count << "\n";
    os << "  Edge Labels:          " << edge_labels_count << "\n";
    os << "  Node Properties:      " << node_properties_count << "\n";
    os << "  Edge Properties:      " << edge_properties_count << "\n";
    os << "  Distinct Node Labels: " << node_label2total_count.size() << "\n";
    os << "  Distinct Edge Labels: " << edge_label2total_count.size() << "\n";
    os << "  Distinct Node Keys:   " << node_key2total_count.size() << "\n";
    os << "  Distinct Edge Keys:   " << edge_key2total_count.size() << "\n";
    if (has_gnn_features()) {
        os << "  GNN Features:         " << gnn_feature_names.size() << " registered\n";
        for (const auto& name : gnn_feature_names) {
            os << "    - " << name << "\n";
        }
    }
#ifdef ENABLE_GNN
    if (hnsw_index_manager.num_hnsw_indexes() > 0) {
        os << "  HNSW Indexes:         " << hnsw_index_manager.num_hnsw_indexes() << "\n";
    }
#endif
    os << "-------------------------------------\n";
}

void GQLCatalog::save()
{
    start_write(MODEL_ID, MAJOR_VERSION, MINOR_VERSION);

    write_uint64(nodes_count);
    write_uint64(directed_edges_count);
    write_uint64(undirected_edges_count);
    write_uint64(node_labels_count);
    write_uint64(edge_labels_count);

    write_uint64(node_properties_count);
    write_uint64(edge_properties_count);
    write_uint64(equal_directed_edges_count);
    write_uint64(equal_undirected_edges_count);

    write_map(node_label2total_count);
    write_map(edge_label2total_count);
    write_map(node_key2total_count);
    write_map(edge_key2total_count);

    write_strvec(node_labels_str);
    write_strvec(edge_labels_str);

    write_strvec(node_keys_str);
    write_strvec(edge_keys_str);

    // GNN feature registry
    write_strvec(gnn_feature_names);

    // HNSW index metadata (minor version 3)
#ifdef ENABLE_GNN
    const auto hnsw_metadata = hnsw_index_manager.get_name2metadata();
    write_uint64(hnsw_metadata.size());
    for (const auto& [name, metadata] : hnsw_metadata) {
        write_string(name);
        write_uint8(static_cast<uint8_t>(metadata.metric_type));
        write_string(metadata.predicate);
    }
#else
    write_uint64(0);
#endif
}

bool GQLCatalog::register_gnn_feature(const std::string& name) {
    mdb::gnn::validate_safe_name(name, "feature name");
    auto it = std::find(gnn_feature_names.begin(), gnn_feature_names.end(), name);
    if (it != gnn_feature_names.end()) {
        return false;  // already registered
    }
    gnn_feature_names.push_back(name);
    has_changes = true;
    return true;
}
