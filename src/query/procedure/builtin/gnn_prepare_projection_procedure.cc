#include "query/procedure/builtin/gnn_prepare_projection_procedure.h"

#include <algorithm>
#include <cerrno>
#include <cctype>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <limits>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_set>
#include <vector>

#include "gnn/projection/gnn_meta.h"
#include "gnn/storage/feature_matrix.h"
#include "gnn/storage/row_mapping.h"
#include "graph_models/object_id.h"
#include "query/optimizer/property_graph_model/executor_constructor.h"
#include "query/parser/gql_query_parser.h"
#include "query/procedure/builtin/gnn_procedure_utils.h"
#include "query/procedure/procedure_context.h"

namespace fs = std::filesystem;

namespace GQL::Procedures {

namespace {

struct ParsedOptions {
    std::string include_features;
    std::string label_property;
    std::string split_property;
};

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

ParsedOptions parse_options(ProcedureContext& ctx) {
    DictOptions opts(ctx.get_argument(1));
    ParsedOptions parsed;

    auto include_features = opts.get_string("includeFeatures");
    if (!include_features || include_features->empty()) {
        throw std::runtime_error(
            "gnn_prepare_projection: options.includeFeatures must be a non-empty string.");
    }
    parsed.include_features = *include_features;
    parsed.label_property = opts.get_string("labelProperty").value_or("");
    parsed.split_property = opts.get_string("splitProperty").value_or("");
    return parsed;
}

bool is_identifier_start(unsigned char c) {
    return std::isalpha(c) != 0 || c == '_';
}

bool is_identifier_continue(unsigned char c) {
    return std::isalnum(c) != 0 || c == '_';
}

void validate_gql_identifier(const std::string& name, const std::string& param_name) {
    validate_safe_name(name, param_name);
    if (!is_identifier_start(static_cast<unsigned char>(name.front()))) {
        throw std::runtime_error(
            param_name + " must start with a letter or underscore: '" + name + "'.");
    }
    for (char c : name) {
        if (!is_identifier_continue(static_cast<unsigned char>(c))) {
            throw std::runtime_error(
                param_name + " must contain only letters, digits, and underscores: '" +
                name + "'.");
        }
    }
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
            "gnn_prepare_projection: row " + std::to_string(row_number) +
            " has invalid node_id '" + token +
            "'; expected an unsigned integer from n._id.");
    }

    errno = 0;
    char* end = nullptr;
    unsigned long long parsed = std::strtoull(token.c_str(), &end, 10);
    if (errno == ERANGE || end == token.c_str() || *end != '\0') {
        throw std::runtime_error(
            "gnn_prepare_projection: row " + std::to_string(row_number) +
            " has node_id outside the uint64 range: '" + token + "'.");
    }
    if (parsed > ObjectId::VALUE_MASK) {
        throw std::runtime_error(
            "gnn_prepare_projection: row " + std::to_string(row_number) +
            " has node_id outside the 56-bit ObjectId payload range: '" +
            token + "'.");
    }
    return static_cast<uint64_t>(parsed);
}

std::optional<int64_t> parse_optional_label(const std::string& token) {
    if (token.empty() || token == "NULL") {
        return std::nullopt;
    }

    errno = 0;
    char* end = nullptr;
    long long parsed = std::strtoll(token.c_str(), &end, 10);
    if (errno == ERANGE || end == token.c_str() || *end != '\0') {
        return std::nullopt;
    }
    if (parsed < std::numeric_limits<int64_t>::min()
        || parsed > std::numeric_limits<int64_t>::max())
    {
        return std::nullopt;
    }
    return static_cast<int64_t>(parsed);
}

std::optional<uint8_t> parse_optional_split(const std::string& token) {
    std::string value = token;
    if (value.size() >= 2 && value.front() == '"' && value.back() == '"') {
        value = value.substr(1, value.size() - 2);
    }

    if (value == "train") {
        return static_cast<uint8_t>(0);
    }
    if (value == "val" || value == "valid" || value == "validation") {
        return static_cast<uint8_t>(1);
    }
    if (value == "test") {
        return static_cast<uint8_t>(2);
    }
    return std::nullopt;
}

void write_labels_file(
    const fs::path& path,
    const std::vector<int64_t>& labels,
    uint64_t num_classes)
{
    std::ofstream lf(path, std::ios::binary | std::ios::trunc);
    if (!lf.is_open()) {
        throw std::runtime_error("Cannot open " + path.string() + " for writing");
    }
    const uint8_t label_magic[8] = {'G','N','N','L','\0','\0','\0','\0'};
    uint32_t label_version = 1;
    uint32_t label_reserved = 0;
    uint64_t label_num_nodes = labels.size();
    uint64_t label_num_classes = num_classes;
    lf.write(reinterpret_cast<const char*>(label_magic), 8);
    lf.write(reinterpret_cast<const char*>(&label_version), 4);
    lf.write(reinterpret_cast<const char*>(&label_reserved), 4);
    lf.write(reinterpret_cast<const char*>(&label_num_nodes), 8);
    lf.write(reinterpret_cast<const char*>(&label_num_classes), 8);
    lf.write(
        reinterpret_cast<const char*>(labels.data()),
        static_cast<std::streamsize>(labels.size() * sizeof(int64_t)));
    if (!lf) {
        throw std::runtime_error("I/O error writing " + path.string());
    }
}

void write_splits_file(const fs::path& path, const std::vector<uint8_t>& splits) {
    std::ofstream sf(path, std::ios::binary | std::ios::trunc);
    if (!sf.is_open()) {
        throw std::runtime_error("Cannot open " + path.string() + " for writing");
    }
    const uint8_t split_magic[8] = {'G','N','N','S','\0','\0','\0','\0'};
    uint32_t split_version = 1;
    uint32_t split_reserved = 0;
    uint64_t split_num_nodes = splits.size();
    sf.write(reinterpret_cast<const char*>(split_magic), 8);
    sf.write(reinterpret_cast<const char*>(&split_version), 4);
    sf.write(reinterpret_cast<const char*>(&split_reserved), 4);
    sf.write(reinterpret_cast<const char*>(&split_num_nodes), 8);
    sf.write(
        reinterpret_cast<const char*>(splits.data()),
        static_cast<std::streamsize>(splits.size() * sizeof(uint8_t)));
    if (!sf) {
        throw std::runtime_error("I/O error writing " + path.string());
    }
}

std::string execute_gql_query_as_tsv(const std::string& query) {
    auto logical_plan = GQL::QueryParser::get_query_plan(query);

    auto query_optimizer = GQL::ExecutorConstructor(GQL::ReturnType::TSV);
    logical_plan->accept_visitor(query_optimizer);

    auto physical_plan = std::move(query_optimizer.executor);
    if (!physical_plan) {
        throw std::runtime_error(
            "gnn_prepare_projection: internal query did not produce an executable RETURN plan.");
    }

    std::stringstream tsv;
    physical_plan->execute(tsv);
    return tsv.str();
}

std::string build_prepare_query(
    const std::string& projection_name,
    const std::string& label_property,
    const std::string& split_property)
{
    std::ostringstream query;
    query << "USE " << projection_name
          << " MATCH (n) RETURN n._id AS node_id";
    if (!label_property.empty()) {
        query << ", n." << label_property << " AS gnn_label";
    }
    if (!split_property.empty()) {
        query << ", n." << split_property << " AS gnn_split";
    }
    return query.str();
}

} // namespace

void GnnPrepareProjectionProcedure::execute(ProcedureContext& ctx) {
    using namespace mdb::gnn;

    if (ctx.arguments.size() != 2) {
        throw std::runtime_error(
            "gnn_prepare_projection requires 2 arguments.\n\n"
            "Usage: CALL gnn_prepare_projection(projectionName, options)\n"
            "Example: CALL gnn_prepare_projection('cora_project_query', {includeFeatures:'node_features', labelProperty:'label', splitProperty:'split'})");
    }

    const std::string projection_name = parse_required_string(ctx, 0, "projectionName");
    const ParsedOptions options = parse_options(ctx);

    validate_gql_identifier(projection_name, "projectionName");
    validate_safe_name(options.include_features, "includeFeatures");
    if (!options.label_property.empty()) {
        validate_gql_identifier(options.label_property, "labelProperty");
    }
    if (!options.split_property.empty()) {
        validate_gql_identifier(options.split_property, "splitProperty");
    }

    const fs::path db_dir = fs::path(get_db_folder());
    const fs::path proj_dir = db_dir / "projections" / projection_name;
    if (!fs::is_directory(proj_dir)) {
        throw std::runtime_error(
            "gnn_prepare_projection: projection directory does not exist: " +
            proj_dir.string());
    }

    const fs::path feature_dir = db_dir / "gnn_features";
    const fs::path fmat_path = feature_dir / (options.include_features + ".fmat");
    const fs::path rmap_path = feature_dir / (options.include_features + ".rmap");
    if (!fs::exists(fmat_path)) {
        throw std::runtime_error(
            "gnn_prepare_projection: FeatureMatrix not found at: " +
            fmat_path.string());
    }
    if (!fs::exists(rmap_path)) {
        throw std::runtime_error(
            "gnn_prepare_projection: RowMapping not found at: " +
            rmap_path.string());
    }

    auto fm = FeatureMatrix::open(fmat_path);
    auto rm = RowMapping::open(rmap_path);
    const uint64_t node_count = rm.size();
    const uint64_t feature_dim = fm.num_cols();
    if (feature_dim > std::numeric_limits<uint32_t>::max()) {
        throw std::runtime_error(
            "gnn_prepare_projection: feature dimension " +
            std::to_string(feature_dim) +
            " exceeds gnn_meta.bin uint32 feature_dim limit.");
    }
    if (fm.num_rows() != node_count) {
        throw std::runtime_error(
            "gnn_prepare_projection: feature '" + options.include_features +
            "' has mismatched .fmat rows (" + std::to_string(fm.num_rows()) +
            ") and .rmap rows (" + std::to_string(node_count) + ").");
    }

    std::vector<int64_t> labels_buffer(node_count, -1);
    std::vector<uint8_t> splits_buffer(node_count, 255);
    std::unordered_set<int64_t> unique_classes;
    bool any_split = false;

    const std::string query = build_prepare_query(
        projection_name,
        options.label_property,
        options.split_property);
    const std::string tsv = execute_gql_query_as_tsv(query);

    std::stringstream input(tsv);
    std::string header_line;
    if (!std::getline(input, header_line)) {
        throw std::runtime_error(
            "gnn_prepare_projection: internal query TSV output did not contain a header.");
    }

    auto header = split_tsv_line(header_line);
    const size_t expected_cols = 1
        + (options.label_property.empty() ? 0 : 1)
        + (options.split_property.empty() ? 0 : 1);
    if (header.size() != expected_cols || header[0] != "node_id") {
        throw std::runtime_error(
            "gnn_prepare_projection: unexpected internal query TSV header.");
    }

    size_t label_col = std::numeric_limits<size_t>::max();
    size_t split_col = std::numeric_limits<size_t>::max();
    size_t next_col = 1;
    if (!options.label_property.empty()) {
        if (header[next_col] != "gnn_label") {
            throw std::runtime_error(
                "gnn_prepare_projection: expected internal TSV column 'gnn_label'.");
        }
        label_col = next_col++;
    }
    if (!options.split_property.empty()) {
        if (header[next_col] != "gnn_split") {
            throw std::runtime_error(
                "gnn_prepare_projection: expected internal TSV column 'gnn_split'.");
        }
        split_col = next_col++;
    }

    std::string line;
    uint64_t row_number = 1;
    while (std::getline(input, line)) {
        if (!line.empty() && line.back() == '\r') {
            line.pop_back();
        }
        ++row_number;
        auto cols = split_tsv_line(line);
        if (cols.size() != expected_cols) {
            throw std::runtime_error(
                "gnn_prepare_projection: row " + std::to_string(row_number) +
                " has " + std::to_string(cols.size()) + " columns; expected " +
                std::to_string(expected_cols) + ".");
        }

        const uint64_t node_id = parse_node_id(cols[0], row_number);
        auto row_opt = rm.find(ObjectId(ObjectId::MASK_NODE | node_id));
        if (!row_opt) {
            continue;
        }
        const uint64_t row = *row_opt;

        if (!options.label_property.empty()) {
            auto label = parse_optional_label(cols[label_col]);
            if (label) {
                labels_buffer[row] = *label;
                unique_classes.insert(*label);
            }
        }

        if (!options.split_property.empty()) {
            auto split = parse_optional_split(cols[split_col]);
            if (split) {
                splits_buffer[row] = *split;
                any_split = true;
            }
        }
    }

    if (!options.label_property.empty() && unique_classes.empty()) {
        throw std::runtime_error(
            "gnn_prepare_projection: labelProperty='" + options.label_property +
            "' was set but no node produced an integer label (every value was "
            "missing or non-integer). Check the property name and that labels "
            "are stored as integers.");
    }
    if (!options.split_property.empty() && !any_split) {
        throw std::runtime_error(
            "gnn_prepare_projection: splitProperty='" + options.split_property +
            "' was set but no node had a recognized split value (expected one "
            "of: train, val, valid, validation, test). Check the property name "
            "and its values.");
    }

    const fs::path meta_path = proj_dir / "gnn_meta.bin";
    const fs::path labels_path = proj_dir / "labels.bin";
    const fs::path splits_path = proj_dir / "splits.bin";

    GnnMeta meta;
    meta.feature_name = options.include_features;
    meta.feature_dim = static_cast<uint32_t>(feature_dim);
    meta.num_nodes = node_count;
    meta.num_classes = unique_classes.size();
    meta.has_labels = !options.label_property.empty();
    meta.has_splits = !options.split_property.empty();
    meta.write(meta_path);

    if (meta.has_labels) {
        write_labels_file(labels_path, labels_buffer, unique_classes.size());
    }
    if (meta.has_splits) {
        write_splits_file(splits_path, splits_buffer);
    }

    ctx.yield("projectionName", ctx.create_string(projection_name));
    ctx.yield("featureName", ctx.create_string(options.include_features));
    ctx.yield("nodeCount", ctx.create_int(static_cast<int64_t>(node_count)));
    ctx.yield("featureDim", ctx.create_int(static_cast<int64_t>(feature_dim)));
    ctx.yield("numClasses", ctx.create_int(static_cast<int64_t>(unique_classes.size())));
    ctx.yield("hasLabels", ctx.create_bool(meta.has_labels));
    ctx.yield("hasSplits", ctx.create_bool(meta.has_splits));
    ctx.yield("labelsPath", ctx.create_string(meta.has_labels ? labels_path.string() : ""));
    ctx.yield("splitsPath", ctx.create_string(meta.has_splits ? splits_path.string() : ""));
    ctx.yield("metaPath", ctx.create_string(meta_path.string()));
    ctx.yield_row();
}

} // namespace GQL::Procedures
