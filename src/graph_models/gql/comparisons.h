#pragma once

#include "graph_models/object_id.h"

namespace GQL {

/**
 * @brief GQL-compliant comparison operations for ObjectId values.
 *
 * Implements comparison semantics according to ISO/IEC 39075:2024.
 * Supports two comparison modes:
 *
 * ## Comparison Modes
 *
 * - **Normal**: Allows comparison across types with type coercion.
 *   Non-comparable types return 0 (equal) for ordering purposes.
 *
 * - **Strict**: Requires same-type comparison. Sets error flag if
 *   types are incompatible (used for equality comparisons in WHERE).
 *
 * ## NULL Handling
 *
 * - **null_first**: NULLs sort before all other values (ASC NULLS FIRST)
 * - **null_last**: NULLs sort after all other values (ASC NULLS LAST)
 *
 * ## Return Values
 *
 * | Return | Meaning |
 * |--------|---------|
 * | < 0 | lhs < rhs |
 * | = 0 | lhs = rhs |
 * | > 0 | lhs > rhs |
 *
 * @see GQL_OID::get_generic_type for type classification
 */
class Comparisons {
public:
    /**
     * @brief Compares two ObjectIds with NULLs sorting first.
     * @param lhs Left-hand side ObjectId
     * @param rhs Right-hand side ObjectId
     * @return Negative if lhs < rhs, positive if lhs > rhs, zero if equal
     */
    static int64_t compare_null_first(ObjectId lhs, ObjectId rhs)
    {
        return _compare<Mode::Normal, true>(lhs, rhs);
    }

    /**
     * @brief Strict comparison with NULLs first (sets error on type mismatch).
     * @param lhs Left-hand side ObjectId
     * @param rhs Right-hand side ObjectId
     * @param[out] error Set to true if types are incompatible
     * @return Comparison result, or 0 if error
     */
    static int64_t strict_compare_null_first(ObjectId lhs, ObjectId rhs, bool* error)
    {
        return _compare<Mode::Strict, true>(lhs, rhs, error);
    }

    /**
     * @brief Compares two ObjectIds with NULLs sorting last.
     * @param lhs Left-hand side ObjectId
     * @param rhs Right-hand side ObjectId
     * @return Negative if lhs < rhs, positive if lhs > rhs, zero if equal
     */
    static int64_t compare_null_last(ObjectId lhs, ObjectId rhs)
    {
        return _compare<Mode::Normal, false>(lhs, rhs);
    }

    /**
     * @brief Strict comparison with NULLs last (sets error on type mismatch).
     * @param lhs Left-hand side ObjectId
     * @param rhs Right-hand side ObjectId
     * @param[out] error Set to true if types are incompatible
     * @return Comparison result, or 0 if error
     */
    static int64_t strict_compare_null_last(ObjectId lhs, ObjectId rhs, bool* error)
    {
        return _compare<Mode::Strict, false>(lhs, rhs, error);
    }

private:
    /// Comparison mode selector
    enum class Mode {
        Normal,  ///< Allow cross-type comparison with coercion
        Strict   ///< Require same types, report error otherwise
    };

    /// @brief Internal templated comparison implementation
    template<Mode mode, bool null_first>
    static int64_t _compare(ObjectId lhs, ObjectId rhs, bool* error = nullptr);
};
} // namespace GQL
