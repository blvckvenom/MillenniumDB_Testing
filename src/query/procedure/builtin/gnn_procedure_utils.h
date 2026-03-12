#pragma once

#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

#include "gnn/common/validate_name.h"
#include "graph_models/common/conversions.h"
#include "graph_models/gql/conversions.h"
#include "graph_models/gql/gql_object_id.h"
#include "storage/dictionary/dictionary.h"
#include "system/file_manager.h"

namespace GQL::Procedures {

/// Validate that a name is safe for use in filesystem paths.
/// Delegates to the shared mdb::gnn::validate_safe_name utility.
/// Used for user-supplied tensor keys and index names.
inline void validate_safe_name(const std::string& name, const std::string& param_name) {
    mdb::gnn::validate_safe_name(name, param_name);
}

/// Get the database folder path with trailing slash stripped.
/// Used by all GNN procedures that access the filesystem.
inline std::string get_db_folder() {
    std::string db_folder = file_manager.get_file_path("");
    if (!db_folder.empty() && db_folder.back() == '/') {
        db_folder.pop_back();
    }
    return db_folder;
}

/// Format a "not found, available: [list]" error message.
inline std::string format_not_found_error(
    const std::string& item_type,
    const std::string& name,
    const std::vector<std::string>& available,
    const std::string& create_hint = ""
) {
    std::string msg = item_type + " '" + name + "' not found.\n\n";
    if (available.empty()) {
        msg += "No " + item_type + "s exist.";
        if (!create_hint.empty()) {
            msg += " Create one first with:\n  " + create_hint;
        }
    } else {
        msg += "Available " + item_type + "s: [";
        for (size_t i = 0; i < available.size(); i++) {
            if (i > 0) msg += ", ";
            msg += "'" + available[i] + "'";
        }
        msg += "]";
    }
    return msg;
}

/// Helper for parsing dictionary/map arguments passed to GNN procedures.
///
/// Wraps the boilerplate of validating the argument type, unpacking the
/// dictionary, and providing typed value lookups by key.
///
/// Usage:
///   DictOptions opts(ctx.get_argument(arg_index));
///   if (auto v = opts.get_string("metric")) { ... }
///   if (auto v = opts.get_int("M"))          { ... }
///   if (auto v = opts.get_double("ratio"))   { ... }
class DictOptions {
public:
    /// Construct from an ObjectId that should be a DICTIONARY.
    /// Throws std::runtime_error if the argument is not a dictionary.
    explicit DictOptions(ObjectId arg) {
        auto type = GQL_OID::get_type(arg);
        if (type != GQL_OID::Type::DICTIONARY) {
            throw std::runtime_error(
                "options must be a MAP/DICTIONARY, got type: " +
                std::to_string(static_cast<int>(type))
            );
        }

        dict_ = Common::Conversions::unpack_dictionary(arg);
        dict_obj_ = dynamic_cast<DictionaryObject*>(dict_->dictionary.get());
        if (!dict_obj_) {
            throw std::runtime_error("Failed to parse options map");
        }
    }

    /// Look up a key in the dictionary and return its ObjectId, or nullopt if absent.
    std::optional<ObjectId> get_value(const std::string& key) const {
        for (const auto& [key_oid, val_item] : dict_obj_->keys) {
            std::string key_str = Conversions::unpack_string(key_oid);
            if (key_str == key) {
                auto lit = dynamic_cast<DictionaryLiteral*>(val_item.get());
                if (!lit) {
                    throw std::runtime_error(
                        "Option '" + key + "' has an unsupported value type "
                        "(expected a literal, got a nested structure)");
                }
                return lit->object_id;
            }
        }
        return std::nullopt;
    }

    /// Look up a string value by key. Returns nullopt if key is absent.
    /// Throws if the value exists but is not a string type.
    std::optional<std::string> get_string(const std::string& key) const {
        auto value = get_value(key);
        if (!value) return std::nullopt;

        auto vtype = GQL_OID::get_type(*value);
        if (vtype == GQL_OID::Type::STRING_SIMPLE_INLINE ||
            vtype == GQL_OID::Type::STRING_SIMPLE_EXTERN ||
            vtype == GQL_OID::Type::STRING_SIMPLE_TMP)
        {
            return Conversions::unpack_string(*value);
        }
        throw std::runtime_error(key + " must be a string");
    }

    /// Look up an integer value by key. Returns nullopt if key is absent.
    /// Throws if the value exists but is not an integer type.
    std::optional<int64_t> get_int(const std::string& key) const {
        auto value = get_value(key);
        if (!value) return std::nullopt;

        auto vtype = GQL_OID::get_type(*value);
        if (vtype == GQL_OID::Type::INT56_INLINE ||
            vtype == GQL_OID::Type::INT64_EXTERN ||
            vtype == GQL_OID::Type::INT64_TMP)
        {
            return Conversions::unpack_int(*value);
        }
        throw std::runtime_error(key + " must be an integer");
    }

    /// Look up a numeric value by key and return it as double.
    /// Accepts int, float, double, and decimal types. Returns nullopt if key is absent.
    /// Throws if the value exists but is not a numeric type.
    std::optional<double> get_double(const std::string& key) const {
        auto value = get_value(key);
        if (!value) return std::nullopt;

        auto vtype = GQL_OID::get_type(*value);
        if (vtype == GQL_OID::Type::INT56_INLINE ||
            vtype == GQL_OID::Type::INT64_EXTERN ||
            vtype == GQL_OID::Type::INT64_TMP)
        {
            return static_cast<double>(Conversions::unpack_int(*value));
        }
        if (vtype == GQL_OID::Type::FLOAT32) {
            return static_cast<double>(Conversions::unpack_float(*value));
        }
        if (vtype == GQL_OID::Type::DOUBLE64_EXTERN ||
            vtype == GQL_OID::Type::DOUBLE64_TMP)
        {
            return Conversions::unpack_double(*value);
        }
        if (vtype == GQL_OID::Type::DECIMAL_INLINE ||
            vtype == GQL_OID::Type::DECIMAL_EXTERN ||
            vtype == GQL_OID::Type::DECIMAL_TMP)
        {
            return Common::Conversions::unpack_decimal(*value).to_double();
        }
        throw std::runtime_error(key + " must be a numeric value");
    }

private:
    std::unique_ptr<Dictionary> dict_;
    DictionaryObject* dict_obj_ = nullptr;  // non-owning; owned by dict_
};

} // namespace GQL::Procedures
