#include "gql_catalog.h"

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

        // --- GNN feature registry (version-dependent) ---
        if (diff_minor_version == 0) {
            // v2: feature name registry
            gnn_feature_names = read_strvec();
        } else if (diff_minor_version == 1) {
            // v1 → v2 migration: read old 3-field format, convert
            bool old_has_tensors = read_uint64() != 0;
            read_uint64(); // discard gnn_tensor_num_rows (now in .fmat header)
            read_uint64(); // discard gnn_tensor_num_cols (now in .fmat header)
            if (old_has_tensors) {
                gnn_feature_names.push_back("node_features");
            }
            has_changes = true; // trigger re-save in v2 format
        } else if (diff_minor_version == 2) {
            // v0 → v2 migration: no GNN fields existed, nothing to read
            has_changes = true;
        }

#ifdef ENABLE_GNN
        // Initialize HNSW index manager (loads existing indexes from disk)
        hnsw_index_manager.init();
#endif
    } else {
        has_changes = true;
#ifdef ENABLE_GNN
        // Initialize HNSW index manager for new database
        hnsw_index_manager.init();
#endif
    }
}

GQLCatalog::~GQLCatalog()
{
    if (has_changes) {
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

    // GNN feature registry (minor version 2)
    write_strvec(gnn_feature_names);
}
