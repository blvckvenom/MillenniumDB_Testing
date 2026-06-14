#include "query/procedure/builtin/gnn_create_structural_features_procedure.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <limits>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "gnn/storage/feature_matrix.h"
#include "gnn/storage/gnn_dtype.h"
#include "gnn/storage/row_mapping.h"
#include "graph_models/gql/gql_model.h"
#include "graph_models/gql/projection/native_scanner.h"
#include "graph_models/object_id.h"
#include "query/procedure/builtin/gnn_procedure_utils.h"
#include "query/procedure/procedure_context.h"

namespace fs = std::filesystem;

namespace GQL::Procedures {

namespace {

constexpr uint64_t FEATURE_DIM = 6;

enum class NormalizeMode {
    NONE,
    ZSCORE,
};

struct NodeStats {
    uint64_t out_degree = 0;
    uint64_t in_degree = 0;
    std::unordered_set<uint64_t> neighbors;
};

NormalizeMode parse_normalize_mode(ProcedureContext& ctx) {
    if (ctx.arguments.size() < 4) {
        return NormalizeMode::NONE;
    }

    DictOptions opts(ctx.get_argument(3));
    auto normalize = opts.get_string("normalize").value_or("none");
    if (normalize == "none") {
        return NormalizeMode::NONE;
    }
    if (normalize == "zscore") {
        return NormalizeMode::ZSCORE;
    }
    throw std::runtime_error(
        "Invalid normalize option for gnn_create_structural_features: '" +
        normalize + "'. Expected 'none' or 'zscore'.");
}

std::string parse_required_string(
    ProcedureContext& ctx,
    size_t index,
    const char* param_name)
{
    std::string value;
    try {
        value = ctx.get_string_argument(index);
    } catch (const std::exception& e) {
        throw std::runtime_error(
            std::string("Invalid ") + param_name + " parameter: " + e.what());
    }
    if (value.empty()) {
        throw std::runtime_error(std::string(param_name) + " cannot be empty.");
    }
    return value;
}

void zscore_normalize(std::vector<float>& features, uint64_t rows) {
    if (rows == 0) {
        return;
    }

    for (uint64_t col = 0; col < FEATURE_DIM; ++col) {
        double sum = 0.0;
        for (uint64_t row = 0; row < rows; ++row) {
            sum += static_cast<double>(features[row * FEATURE_DIM + col]);
        }
        const double mean = sum / static_cast<double>(rows);

        double sq_sum = 0.0;
        for (uint64_t row = 0; row < rows; ++row) {
            const double diff =
                static_cast<double>(features[row * FEATURE_DIM + col]) - mean;
            sq_sum += diff * diff;
        }
        const double stddev = std::sqrt(sq_sum / static_cast<double>(rows));
        if (stddev <= std::numeric_limits<double>::epsilon()) {
            for (uint64_t row = 0; row < rows; ++row) {
                features[row * FEATURE_DIM + col] = 0.0f;
            }
            continue;
        }

        for (uint64_t row = 0; row < rows; ++row) {
            float& value = features[row * FEATURE_DIM + col];
            value = static_cast<float>((static_cast<double>(value) - mean) / stddev);
        }
    }
}

} // namespace

void GnnCreateStructuralFeaturesProcedure::execute(ProcedureContext& ctx) {
    using namespace mdb::gnn;

    if (ctx.arguments.size() < 3 || ctx.arguments.size() > 4) {
        throw std::runtime_error(
            "gnn_create_structural_features requires 3-4 arguments.\n\n"
            "Usage: CALL gnn_create_structural_features(featureName, nodeLabel, edgeType [, options])\n"
            "Example: CALL gnn_create_structural_features('cora_structural_native', 'Paper', 'CITES', {normalize:'zscore'})");
    }

    std::string feature_name = parse_required_string(ctx, 0, "featureName");
    std::string node_label = parse_required_string(ctx, 1, "nodeLabel");
    std::string edge_type = parse_required_string(ctx, 2, "edgeType");
    validate_safe_name(feature_name, "featureName");

    const NormalizeMode normalize_mode = parse_normalize_mode(ctx);
    const bool normalized = normalize_mode == NormalizeMode::ZSCORE;

    auto label_it = gql_model.catalog.node_labels2id.find(node_label);
    if (label_it == gql_model.catalog.node_labels2id.end()) {
        throw std::runtime_error(
            format_not_found_error(
                "node label", node_label, gql_model.catalog.node_labels_str));
    }
    auto edge_it = gql_model.catalog.edge_labels2id.find(edge_type);
    if (edge_it == gql_model.catalog.edge_labels2id.end()) {
        throw std::runtime_error(
            format_not_found_error(
                "edge type", edge_type, gql_model.catalog.edge_labels_str));
    }

    GQL::NativeScanner scanner(
        &gql_model.get_label_node(),
        &gql_model.get_label_edge(),
        &gql_model.get_from_to_edge(),
        &gql_model.get_edge_from_to(),
        &gql_model.get_n1_n2_edge(),
        &gql_model.get_edge_n1_n2());

    const ObjectId label_id(label_it->second | ObjectId::MASK_NODE_LABEL);
    const ObjectId type_id(edge_it->second | ObjectId::MASK_EDGE_LABEL);

    std::vector<ObjectId> row_oids;
    scanner.scan_label_node(label_id, [&](ObjectId node_id) {
        row_oids.push_back(node_id);
    });
    std::sort(row_oids.begin(), row_oids.end(), [](ObjectId lhs, ObjectId rhs) {
        return lhs.id < rhs.id;
    });
    row_oids.erase(
        std::unique(row_oids.begin(), row_oids.end(),
            [](ObjectId lhs, ObjectId rhs) { return lhs.id == rhs.id; }),
        row_oids.end());
    if (row_oids.empty()) {
        throw std::runtime_error(
            "gnn_create_structural_features: nodeLabel '" + node_label +
            "' matched no nodes; cannot create an empty FeatureMatrix.");
    }

    std::unordered_map<uint64_t, uint64_t> row_by_oid;
    row_by_oid.reserve(row_oids.size());
    for (uint64_t row = 0; row < row_oids.size(); ++row) {
        row_by_oid.emplace(row_oids[row].id, row);
    }

    std::vector<NodeStats> stats(row_oids.size());

    scanner.scan_label_edge_with_endpoints(
        type_id,
        [&](ObjectId edge_id, ObjectId from_node, ObjectId to_node) {
            auto from_it = row_by_oid.find(from_node.id);
            auto to_it = row_by_oid.find(to_node.id);
            if (from_it == row_by_oid.end() || to_it == row_by_oid.end()) {
                return;
            }

            const bool undirected =
                (edge_id.id & ObjectId::SUB_TYPE_MASK) == ObjectId::MASK_UNDIRECTED_EDGE;
            const uint64_t from_row = from_it->second;
            const uint64_t to_row = to_it->second;

            stats[from_row].out_degree++;
            stats[to_row].in_degree++;
            stats[from_row].neighbors.insert(to_node.id);
            stats[to_row].neighbors.insert(from_node.id);

            if (undirected && from_row != to_row) {
                stats[to_row].out_degree++;
                stats[from_row].in_degree++;
            }
        });

    std::vector<float> features(row_oids.size() * FEATURE_DIM, 0.0f);
    for (uint64_t row = 0; row < row_oids.size(); ++row) {
        std::unordered_set<uint64_t> two_hop;
        for (uint64_t neighbor_oid : stats[row].neighbors) {
            auto neighbor_row_it = row_by_oid.find(neighbor_oid);
            if (neighbor_row_it == row_by_oid.end()) {
                continue;
            }
            for (uint64_t two_hop_oid : stats[neighbor_row_it->second].neighbors) {
                if (two_hop_oid != row_oids[row].id) {
                    two_hop.insert(two_hop_oid);
                }
            }
        }

        const float one_hop_count =
            static_cast<float>(stats[row].neighbors.size());
        const float two_hop_count = static_cast<float>(two_hop.size());

        float* dst = &features[row * FEATURE_DIM];
        dst[0] = static_cast<float>(stats[row].out_degree);
        dst[1] = static_cast<float>(stats[row].in_degree);
        dst[2] = static_cast<float>(stats[row].out_degree + stats[row].in_degree);
        dst[3] = one_hop_count;
        dst[4] = two_hop_count;
        dst[5] = one_hop_count > 0.0f ? two_hop_count / one_hop_count : 0.0f;
    }

    if (normalized) {
        zscore_normalize(features, row_oids.size());
    }

    const fs::path feature_dir = fs::path(get_db_folder()) / "gnn_features";
    fs::create_directories(feature_dir);
    const fs::path fmat_path = feature_dir / (feature_name + ".fmat");
    const fs::path rmap_path = feature_dir / (feature_name + ".rmap");

    FeatureMatrix::create(
        fmat_path,
        static_cast<uint64_t>(row_oids.size()),
        FEATURE_DIM,
        GnnDtype::FLOAT32,
        features.data());
    RowMapping::create(rmap_path, row_oids);
    gql_model.catalog.register_gnn_feature(feature_name);

    ctx.yield("featureName", ctx.create_string(feature_name));
    ctx.yield("nodeCount",   ctx.create_int(static_cast<int64_t>(row_oids.size())));
    ctx.yield("featureDim",  ctx.create_int(static_cast<int64_t>(FEATURE_DIM)));
    ctx.yield("fmatPath",    ctx.create_string(fmat_path.string()));
    ctx.yield("rmapPath",    ctx.create_string(rmap_path.string()));
    ctx.yield("normalized",  ctx.create_bool(normalized));
    ctx.yield_row();
}

} // namespace GQL::Procedures
