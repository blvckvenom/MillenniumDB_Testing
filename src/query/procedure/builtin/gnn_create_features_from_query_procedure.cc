#include "query/procedure/builtin/gnn_create_features_from_query_procedure.h"

#include <algorithm>
#include <cerrno>
#include <cmath>
#include <cstdlib>
#include <cctype>
#include <filesystem>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_set>
#include <utility>
#include <vector>

#include "gnn/storage/feature_matrix.h"
#include "gnn/storage/gnn_dtype.h"
#include "gnn/storage/row_mapping.h"
#include "graph_models/gql/gql_model.h"
#include "graph_models/object_id.h"
#include "query/optimizer/property_graph_model/executor_constructor.h"
#include "query/parser/gql_query_parser.h"
#include "query/procedure/builtin/gnn_procedure_utils.h"
#include "query/procedure/procedure_context.h"

namespace fs = std::filesystem;

namespace GQL::Procedures {

namespace {

enum class NormalizeMode {
    NONE,
    ZSCORE,
};

struct ParsedOptions {
    NormalizeMode normalize = NormalizeMode::NONE;
    std::string append_to_feature;
};

ParsedOptions parse_options(ProcedureContext& ctx) {
    ParsedOptions parsed;
    if (ctx.arguments.size() < 3) {
        return parsed;
    }

    DictOptions opts(ctx.get_argument(2));
    auto normalize = opts.get_string("normalize").value_or("none");
    if (normalize == "none") {
        parsed.normalize = NormalizeMode::NONE;
    } else if (normalize == "zscore") {
        parsed.normalize = NormalizeMode::ZSCORE;
    } else {
        throw std::runtime_error(
            "Invalid normalize option for gnn_create_features_from_query: '" +
            normalize + "'. Expected 'none' or 'zscore'.");
    }

    auto append_to_feature = opts.get_string("appendToFeature");
    if (append_to_feature) {
        if (append_to_feature->empty()) {
            throw std::runtime_error("appendToFeature cannot be empty.");
        }
        parsed.append_to_feature = *append_to_feature;
    }
    return parsed;
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

std::vector<std::string> split_tsv_line(const std::string& line) {
    std::vector<std::string> cols;
    size_t start = 0;
    while (true) {
        size_t tab = line.find('\t', start);
        if (tab == std::string::npos) {
            cols.emplace_back(line.substr(start));
            break;
        }
        cols.emplace_back(line.substr(start, tab - start));
        start = tab + 1;
    }
    if (!cols.empty() && !cols.back().empty() && cols.back().back() == '\r') {
        cols.back().pop_back();
    }
    return cols;
}

bool is_unsigned_decimal(const std::string& token) {
    return !token.empty()
        && std::all_of(token.begin(), token.end(), [](unsigned char c) {
               return std::isdigit(c) != 0;
           });
}

uint64_t parse_node_id(const std::string& token, uint64_t row_number) {
    if (!is_unsigned_decimal(token)) {
        throw std::runtime_error(
            "gnn_create_features_from_query: row " + std::to_string(row_number) +
            " has invalid node_id '" + token +
            "'; expected an unsigned integer from n._id.");
    }

    errno = 0;
    char* end = nullptr;
    unsigned long long parsed = std::strtoull(token.c_str(), &end, 10);
    if (errno == ERANGE || end == token.c_str() || *end != '\0') {
        throw std::runtime_error(
            "gnn_create_features_from_query: row " + std::to_string(row_number) +
            " has node_id outside the uint64 range: '" + token + "'.");
    }
    if (parsed > ObjectId::VALUE_MASK) {
        throw std::runtime_error(
            "gnn_create_features_from_query: row " + std::to_string(row_number) +
            " has node_id outside the 56-bit ObjectId payload range: '" +
            token + "'.");
    }
    return static_cast<uint64_t>(parsed);
}

float parse_feature_value(
    const std::string& token,
    uint64_t row_number,
    const std::string& column_name)
{
    if (token.empty() || token == "NULL") {
        throw std::runtime_error(
            "gnn_create_features_from_query: row " + std::to_string(row_number) +
            ", column '" + column_name + "' is empty or NULL; expected a float.");
    }
    if (token.find('"') != std::string::npos || token.find('[') != std::string::npos
        || token.find('{') != std::string::npos || token.find('<') != std::string::npos)
    {
        throw std::runtime_error(
            "gnn_create_features_from_query: row " + std::to_string(row_number) +
            ", column '" + column_name +
            "' contains a quoted or complex value ('" + token +
            "'); expected a scalar float.");
    }

    errno = 0;
    char* end = nullptr;
    float value = std::strtof(token.c_str(), &end);
    if (errno == ERANGE || end == token.c_str() || *end != '\0'
        || !std::isfinite(value))
    {
        throw std::runtime_error(
            "gnn_create_features_from_query: row " + std::to_string(row_number) +
            ", column '" + column_name + "' has non-numeric feature value '" +
            token + "'.");
    }
    return value;
}

void zscore_normalize(std::vector<float>& features, uint64_t rows, uint64_t cols) {
    if (rows == 0 || cols == 0) {
        return;
    }

    for (uint64_t col = 0; col < cols; ++col) {
        double sum = 0.0;
        for (uint64_t row = 0; row < rows; ++row) {
            sum += static_cast<double>(features[row * cols + col]);
        }
        const double mean = sum / static_cast<double>(rows);

        double sq_sum = 0.0;
        for (uint64_t row = 0; row < rows; ++row) {
            const double diff = static_cast<double>(features[row * cols + col]) - mean;
            sq_sum += diff * diff;
        }

        const double stddev = std::sqrt(sq_sum / static_cast<double>(rows));
        if (stddev <= std::numeric_limits<double>::epsilon()) {
            for (uint64_t row = 0; row < rows; ++row) {
                features[row * cols + col] = 0.0f;
            }
            continue;
        }

        for (uint64_t row = 0; row < rows; ++row) {
            float& value = features[row * cols + col];
            value = static_cast<float>((static_cast<double>(value) - mean) / stddev);
        }
    }
}

std::string execute_gql_query_as_tsv(const std::string& query) {
    auto logical_plan = GQL::QueryParser::get_query_plan(query);

    auto query_optimizer = GQL::ExecutorConstructor(GQL::ReturnType::TSV);
    logical_plan->accept_visitor(query_optimizer);

    auto physical_plan = std::move(query_optimizer.executor);
    if (!physical_plan) {
        throw std::runtime_error(
            "gnn_create_features_from_query: query did not produce an executable RETURN plan.");
    }

    std::stringstream tsv;
    physical_plan->execute(tsv);
    return tsv.str();
}

} // namespace

void GnnCreateFeaturesFromQueryProcedure::execute(ProcedureContext& ctx) {
    using namespace mdb::gnn;

    if (ctx.arguments.size() < 2 || ctx.arguments.size() > 3) {
        throw std::runtime_error(
            "gnn_create_features_from_query requires 2-3 arguments.\n\n"
            "Usage: CALL gnn_create_features_from_query(featureName, query [, options])\n"
            "Example: CALL gnn_create_features_from_query('query_features', 'MATCH (n:Paper) RETURN n._id AS node_id, toFloat(n.score) AS x_score', {normalize:'zscore'})");
    }

    std::string feature_name = parse_required_string(ctx, 0, "featureName");
    std::string query = parse_required_string(ctx, 1, "query");
    validate_safe_name(feature_name, "featureName");

    const ParsedOptions options = parse_options(ctx);
    const bool normalized = options.normalize == NormalizeMode::ZSCORE;
    if (!options.append_to_feature.empty()) {
        validate_safe_name(options.append_to_feature, "appendToFeature");
        if (options.append_to_feature == feature_name) {
            throw std::runtime_error(
                "appendToFeature must not equal featureName ('" + feature_name + "').");
        }
    }

    const std::string tsv = execute_gql_query_as_tsv(query);

    std::stringstream input(tsv);
    std::string header_line;
    if (!std::getline(input, header_line)) {
        throw std::runtime_error(
            "gnn_create_features_from_query: query TSV output did not contain a header.");
    }

    auto header = split_tsv_line(header_line);
    if (header.empty()) {
        throw std::runtime_error(
            "gnn_create_features_from_query: query TSV output header is empty.");
    }
    if (header[0] != "node_id") {
        throw std::runtime_error(
            "gnn_create_features_from_query: first query result column must be exactly "
            "'node_id'; got '" + header[0] +
            "'. Return n._id AS node_id, not n AS node_id.");
    }
    if (header.size() < 2) {
        throw std::runtime_error(
            "gnn_create_features_from_query: query must return at least one feature "
            "column after node_id.");
    }

    const uint64_t feature_dim = static_cast<uint64_t>(header.size() - 1);
    std::vector<ObjectId> row_oids;
    std::vector<float> features;
    std::unordered_set<uint64_t> seen_node_ids;

    std::string line;
    uint64_t row_number = 1;
    while (std::getline(input, line)) {
        if (!line.empty() && line.back() == '\r') {
            line.pop_back();
        }
        ++row_number;
        auto cols = split_tsv_line(line);
        if (cols.size() != header.size()) {
            throw std::runtime_error(
                "gnn_create_features_from_query: row " + std::to_string(row_number) +
                " has " + std::to_string(cols.size()) + " columns; expected " +
                std::to_string(header.size()) + ".");
        }

        uint64_t node_id = parse_node_id(cols[0], row_number);
        if (!seen_node_ids.insert(node_id).second) {
            throw std::runtime_error(
                "gnn_create_features_from_query: duplicate node_id " +
                std::to_string(node_id) + " in query result.");
        }

        row_oids.emplace_back(ObjectId(ObjectId::MASK_NODE | node_id));
        for (uint64_t col = 1; col < header.size(); ++col) {
            features.push_back(parse_feature_value(cols[col], row_number, header[col]));
        }
    }

    if (row_oids.empty()) {
        throw std::runtime_error(
            "gnn_create_features_from_query: query returned no data rows.");
    }

    if (normalized) {
        zscore_normalize(features, static_cast<uint64_t>(row_oids.size()), feature_dim);
    }

    const fs::path feature_dir = fs::path(get_db_folder()) / "gnn_features";
    fs::create_directories(feature_dir);
    const fs::path fmat_path = feature_dir / (feature_name + ".fmat");
    const fs::path rmap_path = feature_dir / (feature_name + ".rmap");

    uint64_t base_feature_dim = 0;
    uint64_t output_feature_dim = feature_dim;
    std::vector<float> output_features;

    if (!options.append_to_feature.empty()) {
        const auto& names = gql_model.catalog.gnn_feature_names;
        if (std::find(names.begin(), names.end(), options.append_to_feature) == names.end()) {
            throw std::runtime_error(
                format_not_found_error("feature", options.append_to_feature, names));
        }

        const fs::path base_fmat_path = feature_dir / (options.append_to_feature + ".fmat");
        const fs::path base_rmap_path = feature_dir / (options.append_to_feature + ".rmap");
        if (!fs::exists(base_fmat_path)) {
            throw std::runtime_error(
                "appendToFeature base FeatureMatrix not found at: " +
                base_fmat_path.string());
        }
        if (!fs::exists(base_rmap_path)) {
            throw std::runtime_error(
                "appendToFeature base RowMapping not found at: " +
                base_rmap_path.string());
        }

        auto base_fm = FeatureMatrix::open(base_fmat_path);
        auto base_rm = RowMapping::open(base_rmap_path);
        if (base_fm.dtype() != GnnDtype::FLOAT32) {
            throw std::runtime_error(
                "appendToFeature base FeatureMatrix '" + options.append_to_feature +
                "' must be FLOAT32, got " + dtype_name(base_fm.dtype()) + ".");
        }
        if (base_fm.num_rows() != base_rm.size()) {
            throw std::runtime_error(
                "appendToFeature base feature '" + options.append_to_feature +
                "' has mismatched .fmat rows (" +
                std::to_string(base_fm.num_rows()) + ") and .rmap rows (" +
                std::to_string(base_rm.size()) + ").");
        }

        base_feature_dim = base_fm.num_cols();
        output_feature_dim = base_feature_dim + feature_dim;
        output_features.resize(row_oids.size() * output_feature_dim);

        for (uint64_t row = 0; row < row_oids.size(); ++row) {
            auto base_row = base_rm.find(row_oids[row]);
            if (!base_row) {
                throw std::runtime_error(
                    "appendToFeature base row mapping for '" +
                    options.append_to_feature + "' does not contain node ObjectId " +
                    std::to_string(row_oids[row].id) + ".");
            }

            const float* base_src = base_fm.row_as<float>(*base_row);
            float* dst = &output_features[row * output_feature_dim];
            std::copy(base_src, base_src + base_feature_dim, dst);
            std::copy(
                &features[row * feature_dim],
                &features[(row + 1) * feature_dim],
                dst + base_feature_dim);
        }
    } else {
        output_features = std::move(features);
    }

    FeatureMatrix::create(
        fmat_path,
        static_cast<uint64_t>(row_oids.size()),
        output_feature_dim,
        GnnDtype::FLOAT32,
        output_features.data());
    RowMapping::create(rmap_path, row_oids);
    gql_model.catalog.register_gnn_feature(feature_name);

    ctx.yield("featureName",       ctx.create_string(feature_name));
    ctx.yield("nodeCount",         ctx.create_int(static_cast<int64_t>(row_oids.size())));
    ctx.yield("featureDim",        ctx.create_int(static_cast<int64_t>(output_feature_dim)));
    ctx.yield("baseFeatureDim",    ctx.create_int(static_cast<int64_t>(base_feature_dim)));
    ctx.yield("queryFeatureDim",   ctx.create_int(static_cast<int64_t>(feature_dim)));
    ctx.yield("fmatPath",          ctx.create_string(fmat_path.string()));
    ctx.yield("rmapPath",          ctx.create_string(rmap_path.string()));
    ctx.yield("normalized",        ctx.create_bool(normalized));
    ctx.yield("appendedToFeature", ctx.create_string(options.append_to_feature));
    ctx.yield_row();
}

} // namespace GQL::Procedures
