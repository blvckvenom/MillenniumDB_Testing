// src/gnn/storage/addr_table_reader.h
#pragma once

#include "gnn/storage/addr_table.h"

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <stdexcept>
#include <vector>

namespace mdb::gnn {

/// Thrown when the on-disk addr_table file is structurally valid but its
/// meta_sha256_head does not match the runtime-expected value. Distinct
/// type so callers (FourLevelStore::load_batch_features) can catch this
/// and fall back to the legacy classification path without confusing it
/// with a real corruption alarm (which still throws std::runtime_error).
class AddrTableStaleException : public std::runtime_error {
public:
    using std::runtime_error::runtime_error;
};

class AddrTableReader {
public:
    /// Span-like accessor over a (ptr, size) pair, owned by Result::data.
    template <typename T>
    struct ConstView {
        const T*  data;
        size_t    n;
        const T& operator[](size_t i) const { return data[i]; }
        size_t size() const { return n; }
    };

    /// Result of opening and validating an addr_table file. The 9 ConstView
    /// fields point into `data` (the owned buffer). Lifetime: data owns
    /// the storage; the views are invalidated when Result is destroyed.
    struct Result {
        AddrTableHeader              header{};
        std::vector<unsigned char>   data;
        ConstView<uint32_t> l1_positions   {nullptr, 0};
        ConstView<uint32_t> l1_indices     {nullptr, 0};
        ConstView<uint32_t> l2_positions   {nullptr, 0};
        ConstView<uint32_t> l2_indices     {nullptr, 0};
        ConstView<uint32_t> l3_positions   {nullptr, 0};
        ConstView<uint64_t> l3_row_idxs    {nullptr, 0};
        ConstView<uint32_t> l4_positions   {nullptr, 0};
        ConstView<uint32_t> l4_indices     {nullptr, 0};
        ConstView<uint32_t> zero_positions {nullptr, 0};
    };

    /// Open + validate an addr_table file. Reads the whole file (max ~few MB
    /// per batch). Throws:
    ///   - std::runtime_error on magic/version/size invariant violation
    ///   - AddrTableStaleException if expected_meta_sha_head != 0 and
    ///     the file's meta_sha256_head does not match it
    /// expected_meta_sha_head == 0 disables the staleness check.
    static Result open(const std::filesystem::path& path,
                       uint64_t expected_meta_sha_head);
};

} // namespace mdb::gnn
