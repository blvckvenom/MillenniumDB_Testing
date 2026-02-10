#pragma once

#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

#include "graph_models/gql/conversions.h"
#include "graph_models/gql/gql_object_id.h"
#include "graph_models/object_id.h"
#include "query/executor/binding.h"

namespace GQL {

/**
 * Execution context for stored procedures.
 * Provides access to input arguments and allows yielding output values.
 */
class ProcedureContext {
public:
    /**
     * Input arguments passed to the procedure.
     * Set by the CallProcedure iterator before execution.
     */
    std::vector<ObjectId> arguments;

    /**
     * Constructs a procedure context.
     *
     * @param binding The current query binding for variable resolution.
     */
    ProcedureContext(Binding& binding) :
        binding(binding)
    { }

    // ==================== Argument Accessors ====================

    /**
     * Gets the argument at the specified index.
     *
     * @param index The argument index (0-based).
     * @return The ObjectId of the argument.
     * @throws std::out_of_range if index is out of bounds.
     */
    ObjectId get_argument(size_t index) const
    {
        if (index >= arguments.size()) {
            throw std::out_of_range(
                "Argument index " + std::to_string(index) + " out of range (have " +
                std::to_string(arguments.size()) + " arguments)"
            );
        }
        return arguments[index];
    }

    /**
     * Gets a string argument.
     *
     * @param index The argument index.
     * @return The unpacked string value.
     * @throws std::runtime_error if the argument is not a string.
     */
    std::string get_string_argument(size_t index) const
    {
        ObjectId oid = get_argument(index);
        auto type = GQL_OID::get_type(oid);

        if (type == GQL_OID::Type::STRING_SIMPLE_INLINE ||
            type == GQL_OID::Type::STRING_SIMPLE_EXTERN ||
            type == GQL_OID::Type::STRING_SIMPLE_TMP)
        {
            return Conversions::unpack_string(oid);
        }

        throw std::runtime_error(
            "Argument " + std::to_string(index) + " is not a string (type: " +
            std::to_string(static_cast<int>(type)) + ")"
        );
    }

    /**
     * Gets an integer argument.
     *
     * @param index The argument index.
     * @return The unpacked integer value.
     * @throws std::runtime_error if the argument is not an integer.
     */
    int64_t get_int_argument(size_t index) const
    {
        ObjectId oid = get_argument(index);
        auto type = GQL_OID::get_type(oid);

        if (type == GQL_OID::Type::INT56_INLINE ||
            type == GQL_OID::Type::INT64_EXTERN ||
            type == GQL_OID::Type::INT64_TMP) {
            return Conversions::unpack_int(oid);
        }

        throw std::runtime_error(
            "Argument " + std::to_string(index) + " is not an integer (type: " +
            std::to_string(static_cast<int>(type)) + ")"
        );
    }

    /**
     * Gets a float argument.
     *
     * @param index The argument index.
     * @return The unpacked float value.
     * @throws std::runtime_error if the argument is not a float.
     */
    float get_float_argument(size_t index) const
    {
        ObjectId oid = get_argument(index);
        auto type = GQL_OID::get_type(oid);

        if (type == GQL_OID::Type::FLOAT32) {
            return Conversions::unpack_float(oid);
        }

        throw std::runtime_error(
            "Argument " + std::to_string(index) + " is not a float (type: " +
            std::to_string(static_cast<int>(type)) + ")"
        );
    }

    /**
     * Gets a double argument.
     *
     * @param index The argument index.
     * @return The unpacked double value.
     * @throws std::runtime_error if the argument is not a double.
     */
    double get_double_argument(size_t index) const
    {
        ObjectId oid = get_argument(index);
        auto type = GQL_OID::get_type(oid);

        if (type == GQL_OID::Type::DOUBLE64_EXTERN || type == GQL_OID::Type::DOUBLE64_TMP) {
            return Conversions::unpack_double(oid);
        }

        throw std::runtime_error(
            "Argument " + std::to_string(index) + " is not a double (type: " +
            std::to_string(static_cast<int>(type)) + ")"
        );
    }

    /**
     * Gets a boolean argument.
     *
     * @param index The argument index.
     * @return The unpacked boolean value.
     * @throws std::runtime_error if the argument is not a boolean.
     */
    bool get_bool_argument(size_t index) const
    {
        ObjectId oid = get_argument(index);
        auto type = GQL_OID::get_type(oid);

        if (type == GQL_OID::Type::BOOL) {
            return Conversions::unpack_bool(oid);
        }

        throw std::runtime_error(
            "Argument " + std::to_string(index) + " is not a boolean (type: " +
            std::to_string(static_cast<int>(type)) + ")"
        );
    }

    // ==================== Yield Methods ====================

    /**
     * Yields a value for the specified field in the current result row.
     *
     * @param field The yield field name.
     * @param value The value to yield.
     */
    void yield(const std::string& field, ObjectId value)
    {
        current_row[field] = value;
    }

    /**
     * Completes the current result row and starts a new one.
     * After calling this, subsequent yield() calls will populate a new row.
     */
    void yield_row()
    {
        result_rows.push_back(std::move(current_row));
        current_row.clear();
    }

    // ==================== Context Access ====================

    /**
     * Gets the current binding for variable resolution.
     *
     * @return Reference to the binding.
     */
    Binding& get_binding()
    {
        return binding;
    }

    /**
     * Gets all result rows produced by the procedure.
     *
     * @return Vector of result rows (field name → value maps).
     */
    const std::vector<std::unordered_map<std::string, ObjectId>>& get_result_rows() const
    {
        return result_rows;
    }

    // ==================== Helper Methods for Creating ObjectIds ====================

    /**
     * Creates a string ObjectId from a C++ string.
     *
     * @param str The string value.
     * @return An ObjectId representing the string.
     */
    ObjectId create_string(const std::string& str)
    {
        return Conversions::pack_string_simple(str);
    }

    /**
     * Creates an integer ObjectId from an int64_t value.
     *
     * @param value The integer value.
     * @return An ObjectId representing the integer.
     */
    ObjectId create_int(int64_t value)
    {
        return Conversions::pack_int(value);
    }

    /**
     * Creates a float ObjectId from a float value.
     *
     * @param value The float value.
     * @return An ObjectId representing the float.
     */
    ObjectId create_float(float value)
    {
        return Conversions::pack_float(value);
    }

    /**
     * Creates a double ObjectId from a double value.
     *
     * @param value The double value.
     * @return An ObjectId representing the double.
     */
    ObjectId create_double(double value)
    {
        return Conversions::pack_double(value);
    }

    /**
     * Creates a boolean ObjectId from a bool value.
     *
     * @param value The boolean value.
     * @return An ObjectId representing the boolean.
     */
    ObjectId create_bool(bool value)
    {
        return Conversions::pack_bool(value);
    }

private:
    Binding& binding;

    // Result row being built
    std::unordered_map<std::string, ObjectId> current_row;

    // Completed result rows
    std::vector<std::unordered_map<std::string, ObjectId>> result_rows;
};

} // namespace GQL
