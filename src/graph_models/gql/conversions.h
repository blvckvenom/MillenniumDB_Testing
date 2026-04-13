#pragma once

#include <cstdint>
#include <string>

#include "graph_models/common/conversions.h" // IWYU pragma: export
#include "graph_models/gql/gql_object_id.h" // IWYU pragma: export

/**
 * @brief GQL-specific conversion utilities for ObjectId packing/unpacking.
 *
 * Provides functions to convert between C++ types (strings, vectors) and
 * ObjectId representations. Handles both inline storage (small values) and
 * external storage (dictionary references).
 *
 * ## Storage Modes
 *
 * - **Inline**: Values ≤7 bytes stored directly in ObjectId payload
 * - **External**: Larger values stored in dictionary, ObjectId contains reference
 * - **Temporary**: Query-scoped values that don't persist
 *
 * ## Common Operations
 *
 * | Operation | Function |
 * |-----------|----------|
 * | String → ObjectId | pack_string_simple() |
 * | ObjectId → String | unpack_string() |
 * | Label → ObjectId | pack_node_label(), pack_edge_label() |
 * | Property key → ObjectId | pack_node_property(), pack_edge_property() |
 * | List → ObjectId | pack_list() |
 * | ObjectId → List | unpack_list() |
 *
 * @see Common::Conversions for shared conversion utilities
 * @see GQL_OID for type system definitions
 */
namespace GQL { namespace Conversions {

using namespace Common::Conversions;

/// Maximum temporary ID before overflow
static constexpr uint64_t LAST_TMP_ID = ObjectId::MASK_LITERAL_TAG;

/// Bit shift for temporary ID encoding
static constexpr uint64_t TMP_SHIFT = 44;

/// @name String Conversions
/// @{

/**
 * @brief Extracts string value from an ObjectId.
 * @param oid String ObjectId (inline or external)
 * @return The string value
 */
std::string unpack_string(ObjectId oid);

/**
 * @brief Prints string value to output stream.
 * @param oid String ObjectId
 * @param os Output stream
 */
void print_string(ObjectId oid, std::ostream&);

/**
 * @brief Prints string value to character buffer.
 * @param oid String ObjectId
 * @param[out] out Output buffer (must be large enough)
 * @return Number of characters written
 */
size_t print_string(ObjectId oid, char* out);

/**
 * @brief Creates ObjectId from simple string.
 * @param str Input string
 * @return String ObjectId (inline if ≤7 bytes, external otherwise)
 */
ObjectId pack_string_simple(const std::string& str);
/// @}

/// @name Path Conversions
/// @{

/// @brief Prints a node in path format
void print_path_node(std::ostream& os, ObjectId node_id);

/// @brief Prints an edge in path format (with direction indicator)
void print_path_edge(std::ostream& os, ObjectId edge_id, bool inverse);

/// @brief Prints complete path to output stream
void print_path(std::ostream& os, ObjectId oid);

/// @brief Packs a sequence of ObjectIds into a path ObjectId
ObjectId pack_path(const std::vector<ObjectId>& oid_list);

/// @brief Unpacks path ObjectId into vector of node/edge ObjectIds
void unpack_path(ObjectId oid, std::vector<ObjectId>& out);
/// @}

/// @name Label Conversions
/// @{

/**
 * @brief Creates edge label ObjectId from string.
 * @param label Label name (e.g., "KNOWS", "LIKES")
 * @return Edge label ObjectId
 */
ObjectId pack_edge_label(const std::string& label);

/**
 * @brief Creates node label ObjectId from string.
 * @param label Label name (e.g., "Person", "Movie")
 * @return Node label ObjectId
 */
ObjectId pack_node_label(const std::string& label);
/// @}

/// @name Property Key Conversions
/// @{

/// @brief Creates node property key ObjectId from string
ObjectId pack_node_property(const std::string& key);

/// @brief Creates edge property key ObjectId from string
ObjectId pack_edge_property(const std::string& key);
/// @}

/// @name List Conversions
/// @{

/// Mask for extracting file ID from list ObjectId
constexpr uint64_t LIST_FILE_ID_MASK = 0x00FF'FF00'0000'0000UL;

/// Mask for extracting offset from list ObjectId
constexpr uint64_t LIST_OFFSET_MASK = 0x0000'00FF'FFFF'FFFFUL;

/// @brief Packs vector of ObjectIds into a list ObjectId
ObjectId pack_list(const std::vector<ObjectId>& list);

/// @brief Unpacks list ObjectId into vector (output parameter version)
void unpack_list(ObjectId list_id, std::vector<ObjectId>& out);

/// @brief Unpacks list ObjectId into vector (return value version)
std::vector<ObjectId> unpack_list(ObjectId list_id);
/// @}

/// @name Type Conversions
/// @{

/**
 * @brief Converts any ObjectId to boolean according to GQL truthiness rules.
 * @param oid Input ObjectId
 * @return Boolean ObjectId (true or false)
 */
ObjectId to_boolean(ObjectId oid);

/**
 * @brief Gets lexical (string) representation of any ObjectId value.
 * @param oid Input ObjectId
 * @return Human-readable string representation
 */
std::string to_lexical_str(ObjectId oid);
/// @}

/// @name Debugging
/// @{

/// @brief Prints ObjectId in debug format (type + value)
std::ostream& debug_print(std::ostream& os, ObjectId oid);
/// @}

}} // namespace GQL::Conversions
