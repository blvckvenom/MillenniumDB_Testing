//===----------------------------------------------------------------------===//
//
// This file is part of MillenniumDB
//
// MillenniumDB is free software: you can redistribute it and/or modify
// it under the terms of the GNU Affero General Public License as published
// by the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.
//
// MillenniumDB is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU Affero General Public License for more details.
//
// You should have received a copy of the GNU Affero General Public License
// along with MillenniumDB.  If not, see <https://www.gnu.org/licenses/>.
//
//===----------------------------------------------------------------------===//

#pragma once

#include <string>
#include <vector>
#include <unordered_map>
#include <variant>
#include <sstream>

namespace GQL {

/**
 * Represents a value in the GQL type system.
 * Can hold strings, numbers, booleans, lists, or maps.
 */
class Value {
public:
    using ValueList = std::vector<Value>;
    using ValueMap = std::unordered_map<std::string, Value>;
    using ValueVariant = std::variant<std::string, int64_t, double, bool, ValueList, ValueMap>;

private:
    ValueVariant value_;

public:
    // Constructors
    Value() : value_(std::string{}) {}
    Value(const std::string& str) : value_(str) {}
    Value(const char* str) : value_(std::string(str)) {}
    Value(int64_t num) : value_(num) {}
    Value(double num) : value_(num) {}
    Value(bool b) : value_(b) {}
    Value(const ValueList& list) : value_(list) {}
    Value(const ValueMap& map) : value_(map) {}

    // Type checking methods
    bool is_string() const { return std::holds_alternative<std::string>(value_); }
    bool is_int() const { return std::holds_alternative<int64_t>(value_); }
    bool is_double() const { return std::holds_alternative<double>(value_); }
    bool is_bool() const { return std::holds_alternative<bool>(value_); }
    bool is_list() const { return std::holds_alternative<ValueList>(value_); }
    bool is_map() const { return std::holds_alternative<ValueMap>(value_); }

    // Getter methods
    const std::string& get_string() const { return std::get<std::string>(value_); }
    int64_t get_int() const { return std::get<int64_t>(value_); }
    double get_double() const { return std::get<double>(value_); }
    bool get_bool() const { return std::get<bool>(value_); }
    const ValueList& as_list() const { return std::get<ValueList>(value_); }
    const ValueMap& as_map() const { return std::get<ValueMap>(value_); }

    // Convert to string representation
    std::string to_string() const {
        std::stringstream ss;
        std::visit([&ss](const auto& v) {
            using T = std::decay_t<decltype(v)>;
            if constexpr (std::is_same_v<T, std::string>) {
                ss << "\"" << v << "\"";
            } else if constexpr (std::is_same_v<T, int64_t> || std::is_same_v<T, double>) {
                ss << v;
            } else if constexpr (std::is_same_v<T, bool>) {
                ss << (v ? "true" : "false");
            } else if constexpr (std::is_same_v<T, ValueList>) {
                ss << "[";
                for (size_t i = 0; i < v.size(); ++i) {
                    if (i > 0) ss << ", ";
                    ss << v[i].to_string();
                }
                ss << "]";
            } else if constexpr (std::is_same_v<T, ValueMap>) {
                ss << "{";
                bool first = true;
                for (const auto& [key, val] : v) {
                    if (!first) ss << ", ";
                    ss << "\"" << key << "\": " << val.to_string();
                    first = false;
                }
                ss << "}";
            }
        }, value_);
        return ss.str();
    }
};

/**
 * Represents a configuration map for GQL operations.
 */
class Map {
private:
    std::unordered_map<std::string, Value> data_;

public:
    Map() = default;
    Map(const std::unordered_map<std::string, Value>& data) : data_(data) {}

    // Set a value
    void set(const std::string& key, const Value& value) {
        data_[key] = value;
    }

    // Get a value with default
    Value get(const std::string& key, const Value& defaultValue = Value()) const {
        auto it = data_.find(key);
        return (it != data_.end()) ? it->second : defaultValue;
    }

    // Get boolean with default
    bool get_bool(const std::string& key, bool defaultValue = false) const {
        auto it = data_.find(key);
        if (it != data_.end() && it->second.is_bool()) {
            return it->second.get_bool();
        }
        return defaultValue;
    }

    // Get string with default
    std::string get_string(const std::string& key, const std::string& defaultValue = "") const {
        auto it = data_.find(key);
        if (it != data_.end() && it->second.is_string()) {
            return it->second.get_string();
        }
        return defaultValue;
    }

    // Check if key exists
    bool has(const std::string& key) const {
        return data_.find(key) != data_.end();
    }

    // Convert to string representation
    std::string to_string() const {
        std::stringstream ss;
        ss << "{";
        bool first = true;
        for (const auto& [key, value] : data_) {
            if (!first) ss << ", ";
            ss << "\"" << key << "\": " << value.to_string();
            first = false;
        }
        ss << "}";
        return ss.str();
    }

    // Iterators
    auto begin() const { return data_.begin(); }
    auto end() const { return data_.end(); }
    size_t size() const { return data_.size(); }
    bool empty() const { return data_.empty(); }
};

} // namespace GQL
