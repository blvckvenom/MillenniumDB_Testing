#pragma once

#include <cassert>

#include "graph_models/object_id.h"

/**
 * @brief GQL-specific ObjectId type system.
 *
 * This namespace provides type classification utilities for GQL (Graph Query Language)
 * ObjectIds. The type system follows a 3-level hierarchy:
 *
 * 1. **GenericType**: High-level type categories (STRING, NUMERIC, NODE, EDGE, etc.)
 * 2. **GenericSubType**: Refined subtypes (INTEGER vs FLOAT vs DOUBLE within NUMERIC)
 * 3. **Type**: Complete type including storage mode (INLINE, EXTERN, TMP)
 *
 * ## ObjectId Layout (64 bits)
 *
 * ```
 * [8-bit type prefix][56-bit value payload]
 *  ├─ 4 bits: Generic type
 *  ├─ 2 bits: Subtype
 *  └─ 2 bits: Storage modifier (inline=0b00, extern=0b01, tmp=0b10)
 * ```
 *
 * ## GQL-Specific Type Prefixes
 *
 * | Prefix | Type | Description |
 * |--------|------|-------------|
 * | 0xC0 | NODE | Graph node |
 * | 0xE0 | DIRECTED_EDGE | Directed relationship |
 * | 0xE4 | UNDIRECTED_EDGE | Undirected relationship |
 * | 0xC4 | NODE_LABEL | Node label identifier |
 * | 0xC8 | EDGE_LABEL | Edge label identifier |
 * | 0xCC | NODE_KEY | Node property key |
 * | 0xD0 | EDGE_KEY | Edge property key |
 *
 * @see ObjectId for base type mask definitions
 * @see src/graph_models/object_id.h for complete mask reference
 */
namespace GQL_OID {

/**
 * @brief High-level type categories for GQL values.
 *
 * Represents the most general classification of GQL types.
 * Used for type checking in expressions and comparisons.
 *
 * @note These map to GQL standard value types (ISO/IEC 39075:2024 §4.4)
 */
enum class GenericType {
        NULL_ID = 0x00,
        STRING,
        NUMERIC,
        DATE,
        BOOL,
        PATH,
        NODE,
        EDGE,
        LABEL,
        KEY,
        LIST,
        DICTIONARY,
        TENSOR,
        // GEOMETRY, ?
    };

    /**
     * @brief Refined subtypes within each GenericType category.
     *
     * Provides finer granularity than GenericType while still abstracting
     * away storage details. For example, INTEGER, FLOAT, DOUBLE, DECIMAL
     * are all NUMERIC at the generic level but distinct subtypes.
     *
     * Used for type coercion and comparison semantics.
     */
    enum class GenericSubType {
        NULL_ID = 0x00,
        // URI,
        STRING_SIMPLE,
        INTEGER,
        FLOAT,
        DOUBLE,
        DECIMAL,
        DATE,
        BOOL,
        PATH,
        NODE,
        EDGE,
        LABEL,
        KEY,
        LIST,
        DICTIONARY,
        TENSOR_FLOAT,
        TENSOR_DOUBLE,
        // POINT, ?
    };

    /**
     * @brief Complete type enumeration including storage mode.
     *
     * The most specific type classification, distinguishing between:
     * - **INLINE**: Value stored directly in the 56-bit payload (small values)
     * - **EXTERN**: Value stored in external dictionary, payload contains reference
     * - **TMP**: Temporary value created during query execution
     *
     * ## Storage Mode Selection
     *
     * | Type | Inline Threshold | External Storage |
     * |------|------------------|------------------|
     * | STRING | ≤7 bytes | String dictionary |
     * | INT | 56-bit range | 64-bit external |
     * | DECIMAL | Small precision | External decimal |
     * | DOUBLE | N/A | Always external |
     *
     * @see get_type() to extract Type from ObjectId
     */
    enum class Type {
        NULL_ID = 0x00,
        // URI,
        STRING_SIMPLE_INLINE,
        STRING_SIMPLE_EXTERN,
        STRING_SIMPLE_TMP,
        INT56_INLINE,
        INT64_EXTERN,
        INT64_TMP,
        FLOAT32,
        DOUBLE64_EXTERN,
        DOUBLE64_TMP,
        DECIMAL_INLINE,
        DECIMAL_EXTERN,
        DECIMAL_TMP,
        DATE,
        TIME,
        DATETIME,
        DATETIMESTAMP,
        BOOL,
        PATH,
        NODE,
        DIRECTED_EDGE,
        UNDIRECTED_EDGE,
        NODE_LABEL,
        EDGE_LABEL,
        NODE_KEY,
        EDGE_KEY,
        LIST,
        DICTIONARY,
        TENSOR_FLOAT_INLINED,
        TENSOR_FLOAT_EXTERN,
        TENSOR_FLOAT_TMP,
        TENSOR_DOUBLE_INLINED,
        TENSOR_DOUBLE_EXTERN,
        TENSOR_DOUBLE_TMP,
        // POINT, ?
    };

    /// Maximum bytes for inline string storage (longer strings use external dictionary)
    static constexpr int MAX_INLINE_LEN_STRING = 7;

    /**
     * @brief Extracts the complete Type from an ObjectId.
     *
     * Examines the 8-bit type prefix and returns the corresponding Type enum value.
     * This is the primary type introspection function for GQL values.
     *
     * @param oid The ObjectId to examine
     * @return The Type of the ObjectId
     *
     * @note Performance: O(1) - single switch on masked value
     *
     * Example:
     * @code
     * ObjectId node_id = ...; // Some node
     * if (GQL_OID::get_type(node_id) == GQL_OID::Type::NODE) {
     *     // Handle node
     * }
     * @endcode
     */
    inline constexpr Type get_type(ObjectId oid) {
        switch (oid.id >> 56) {
        case (ObjectId::MASK_NULL >> 56):
            return Type::NULL_ID;

        case (ObjectId::MASK_NODE >> 56):
            return Type::NODE;
        case (ObjectId::MASK_DIRECTED_EDGE >> 56):
            return Type::DIRECTED_EDGE;
        case (ObjectId::MASK_UNDIRECTED_EDGE >> 56):
            return Type::UNDIRECTED_EDGE;
        case (ObjectId::MASK_NODE_LABEL >> 56):
            return Type::NODE_LABEL;
        case (ObjectId::MASK_EDGE_LABEL >> 56):
            return Type::EDGE_LABEL;
        case (ObjectId::MASK_NODE_KEY >> 56):
            return Type::NODE_KEY;
        case (ObjectId::MASK_EDGE_KEY >> 56):
            return Type::EDGE_KEY;
        case (ObjectId::MASK_LIST >> 56):
        case (ObjectId::MASK_LIST_EXTERN >> 56):
        case (ObjectId::MASK_LIST_TMP >> 56):
            return Type::LIST;

        case (ObjectId::MASK_STRING_SIMPLE_INLINED >> 56):
            return Type::STRING_SIMPLE_INLINE;
        case (ObjectId::MASK_STRING_SIMPLE_EXTERN >> 56):
            return Type::STRING_SIMPLE_EXTERN;
        case (ObjectId::MASK_STRING_SIMPLE_TMP >> 56):
            return Type::STRING_SIMPLE_TMP;

        case (ObjectId::MASK_FLOAT >> 56):
            return Type::FLOAT32;
        case (ObjectId::MASK_NEGATIVE_INT >> 56):
            return Type::INT56_INLINE;
        case (ObjectId::MASK_POSITIVE_INT >> 56):
            return Type::INT56_INLINE;
        case (ObjectId::MASK_DECIMAL_INLINED >> 56):
            return Type::DECIMAL_INLINE;
        case (ObjectId::MASK_DECIMAL_EXTERN >> 56):
            return Type::DECIMAL_EXTERN;
        case (ObjectId::MASK_DECIMAL_TMP >> 56):
            return Type::DECIMAL_TMP;
        case (ObjectId::MASK_DOUBLE_EXTERN >> 56):
            return Type::DOUBLE64_EXTERN;
        case (ObjectId::MASK_DOUBLE_TMP >> 56):
            return Type::DOUBLE64_TMP;

        case (ObjectId::MASK_BOOL >> 56):
            return Type::BOOL;

        case (ObjectId::MASK_GQL_PATH >> 56):
            return Type::PATH;
        case (ObjectId::MASK_DICTIONARY >> 56):
        case (ObjectId::MASK_DICTIONARY_EXTERN >> 56):
        case (ObjectId::MASK_DICTIONARY_TMP >> 56):
            return Type::DICTIONARY;

        case (ObjectId::MASK_DT_DATE >> 56):
            return Type::DATE;
        case (ObjectId::MASK_DT_TIME >> 56):
            return Type::TIME;
        case (ObjectId::MASK_DT_DATETIME >> 56):
            return Type::DATETIME;
        case (ObjectId::MASK_DT_DATETIMESTAMP >> 56):
            return Type::DATETIMESTAMP;

        case ObjectId::MASK_TENSOR_FLOAT_INLINED >> 56:
            return Type::TENSOR_FLOAT_INLINED;
        case ObjectId::MASK_TENSOR_FLOAT_EXTERN >> 56:
            return Type::TENSOR_FLOAT_EXTERN;
        case ObjectId::MASK_TENSOR_FLOAT_TMP >> 56:
            return Type::TENSOR_FLOAT_TMP;
        case ObjectId::MASK_TENSOR_DOUBLE_INLINED >> 56:
            return Type::TENSOR_DOUBLE_INLINED;
        case ObjectId::MASK_TENSOR_DOUBLE_EXTERN >> 56:
            return Type::TENSOR_DOUBLE_EXTERN;
        case ObjectId::MASK_TENSOR_DOUBLE_TMP >> 56:
            return Type::TENSOR_DOUBLE_TMP;

        default:
            assert(false);
            return Type::NULL_ID;
        }
        // return static_cast<Type>(oid.id);
    }

    /**
     * @brief Gets the GenericSubType for an ObjectId.
     *
     * Maps the complete Type to its parent subtype category.
     * Useful for type coercion decisions (e.g., can INTEGER be compared with FLOAT?).
     *
     * @param oid The ObjectId to examine
     * @return The GenericSubType category
     *
     * @note All storage modes (INLINE/EXTERN/TMP) map to the same subtype
     */
    inline GenericSubType get_generic_sub_type(ObjectId oid) {
        switch (get_type(oid)) {
            case Type::NULL_ID:
                return GenericSubType::NULL_ID;
            case Type::NODE:
                return GenericSubType::NODE;
            case Type::DIRECTED_EDGE:
                return GenericSubType::EDGE;
            case Type::UNDIRECTED_EDGE:
                return GenericSubType::EDGE;
            case Type::NODE_LABEL:
                return GenericSubType::LABEL;
            case Type::EDGE_LABEL:
                return GenericSubType::LABEL;
            case Type::NODE_KEY:
                return GenericSubType::KEY;
            case Type::EDGE_KEY:
                return GenericSubType::KEY;
            case Type::LIST:
                return GenericSubType::LIST;
            case Type::STRING_SIMPLE_INLINE:
            case Type::STRING_SIMPLE_EXTERN:
            case Type::STRING_SIMPLE_TMP:
                return GenericSubType::STRING_SIMPLE;
            case Type::INT56_INLINE:
            case Type::INT64_EXTERN:
            case Type::INT64_TMP:
                return GenericSubType::INTEGER;
            case Type::FLOAT32:
                return GenericSubType::FLOAT;
            case Type::DOUBLE64_EXTERN:
            case Type::DOUBLE64_TMP:
                return GenericSubType::DOUBLE;
            case Type::DECIMAL_INLINE:
            case Type::DECIMAL_EXTERN:
            case Type::DECIMAL_TMP:
                return GenericSubType::DECIMAL;
            case Type::BOOL:
                return GenericSubType::BOOL;
            case Type::DATE:
            case Type::TIME:
            case Type::DATETIME:
            case Type::DATETIMESTAMP:
                return GenericSubType::DATE;
            case Type::PATH:
                return GenericSubType::PATH;
            case Type::DICTIONARY:
                return GenericSubType::DICTIONARY;
            case Type::TENSOR_FLOAT_INLINED:
            case Type::TENSOR_FLOAT_EXTERN:
            case Type::TENSOR_FLOAT_TMP:
                return GenericSubType::TENSOR_FLOAT;
            case Type::TENSOR_DOUBLE_INLINED:
            case Type::TENSOR_DOUBLE_EXTERN:
            case Type::TENSOR_DOUBLE_TMP:
                return GenericSubType::TENSOR_DOUBLE;
            }
        assert(false);
        return GenericSubType::NULL_ID;
    }

    /**
     * @brief Gets the highest-level GenericType for an ObjectId.
     *
     * Returns the broadest type category. Used for:
     * - Type compatibility checking in expressions
     * - Determining comparison semantics (numeric vs string comparison)
     * - Query result type inference
     *
     * @param oid The ObjectId to examine
     * @return The GenericType category
     *
     * @note All numeric types (INT, FLOAT, DOUBLE, DECIMAL) return GenericType::NUMERIC
     * @note Both DIRECTED_EDGE and UNDIRECTED_EDGE return GenericType::EDGE
     */
    inline GenericType get_generic_type(ObjectId oid) {
        switch (get_type(oid)) {
            case Type::NULL_ID:
                return GenericType::NULL_ID;
            case Type::NODE:
                return GenericType::NODE;
            case Type::DIRECTED_EDGE:
                return GenericType::EDGE;
            case Type::UNDIRECTED_EDGE:
                return GenericType::EDGE;
            case Type::NODE_LABEL:
                return GenericType::LABEL;
            case Type::EDGE_LABEL:
                return GenericType::LABEL;
            case Type::NODE_KEY:
                return GenericType::KEY;
            case Type::EDGE_KEY:
                return GenericType::KEY;
            case Type::LIST:
                return GenericType::LIST;
            case Type::STRING_SIMPLE_INLINE:
            case Type::STRING_SIMPLE_EXTERN:
            case Type::STRING_SIMPLE_TMP:
                return GenericType::STRING;
            case Type::INT56_INLINE:
            case Type::INT64_EXTERN:
            case Type::INT64_TMP:
            case Type::FLOAT32:
            case Type::DOUBLE64_EXTERN:
            case Type::DOUBLE64_TMP:
            case Type::DECIMAL_INLINE:
            case Type::DECIMAL_EXTERN:
            case Type::DECIMAL_TMP:
                return GenericType::NUMERIC;
            case Type::BOOL:
                return GenericType::BOOL;
            case Type::DATE:
            case Type::TIME:
            case Type::DATETIME:
            case Type::DATETIMESTAMP:
                return GenericType::DATE;
            case Type::PATH:
                return GenericType::PATH;
            case Type::DICTIONARY:
                return GenericType::DICTIONARY;
            case Type::TENSOR_FLOAT_INLINED:
            case Type::TENSOR_FLOAT_EXTERN:
            case Type::TENSOR_FLOAT_TMP:
            case Type::TENSOR_DOUBLE_INLINED:
            case Type::TENSOR_DOUBLE_EXTERN:
            case Type::TENSOR_DOUBLE_TMP:
                return GenericType::TENSOR;
            default:
                assert(false);
                return GenericType::NULL_ID;
        }
    }
}
