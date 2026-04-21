// src/graph_models/gql/projection/partition_file.h
#pragma once

#include <cstddef>
#include <cstdio>
#include <memory>
#include <string>
#include <vector>

#include "storage/index/record.h"

namespace GQL {

/**
 * @brief Append-mode writer for a single partition's records + reader.
 *        Buffered at 4 MB; flushes on reaching buffer or destructor.
 *        No concurrent writers on the same file; OK across threads for
 *        distinct PartitionFile instances.
 */
template<std::size_t N>
class PartitionFile {
public:
    static constexpr std::size_t DEFAULT_BUFFER_BYTES = 4ULL * 1024 * 1024;

    PartitionFile(const std::string& path,
                  std::size_t buffer_bytes = DEFAULT_BUFFER_BYTES);
    ~PartitionFile();

    void append(const Record<N>& r);
    void flush();

    std::size_t record_count() const { return record_count_; }
    std::size_t bytes_written() const { return bytes_written_; }
    const std::string& path() const { return path_; }

    class Reader {
    public:
        explicit Reader(const std::string& path);
        ~Reader();

        /// @return true if a record was read, false on EOF.
        bool next(Record<N>& out);
        bool eof() const { return eof_; }

    private:
        std::FILE* fp_ = nullptr;
        bool eof_ = false;
    };

private:
    std::string   path_;
    std::FILE*    fp_ = nullptr;
    std::size_t   buffer_capacity_;
    std::vector<Record<N>> buffer_;
    std::size_t   record_count_ = 0;
    std::size_t   bytes_written_ = 0;
};

extern template class PartitionFile<1>;
extern template class PartitionFile<2>;
extern template class PartitionFile<3>;

}  // namespace GQL
