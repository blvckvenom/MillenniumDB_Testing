// src/graph_models/gql/projection/partition_file.cc
#include "graph_models/gql/projection/partition_file.h"

#include <cerrno>
#include <cstring>
#include <stdexcept>

namespace GQL {

template<std::size_t N>
PartitionFile<N>::PartitionFile(const std::string& path, std::size_t buffer_bytes)
    : path_(path),
      buffer_capacity_(buffer_bytes / sizeof(Record<N>))
{
    fp_ = std::fopen(path_.c_str(), "wb");
    if (!fp_) {
        throw std::runtime_error("PartitionFile: cannot open " + path_);
    }
    buffer_.reserve(buffer_capacity_);
}

template<std::size_t N>
PartitionFile<N>::~PartitionFile() {
    try { flush(); } catch (...) {}
    if (fp_) std::fclose(fp_);
}

template<std::size_t N>
void PartitionFile<N>::append(const Record<N>& r) {
    buffer_.push_back(r);
    record_count_++;
    if (buffer_.size() >= buffer_capacity_) {
        flush();
    }
}

template<std::size_t N>
void PartitionFile<N>::flush() {
    if (!buffer_.empty()) {
        std::size_t n = std::fwrite(buffer_.data(), sizeof(Record<N>),
                                     buffer_.size(), fp_);
        if (n != buffer_.size()) {
            throw std::runtime_error("PartitionFile: short write to " + path_);
        }
        bytes_written_ += n * sizeof(Record<N>);
        buffer_.clear();
    }
    // Push libc stdio buffer to the kernel so callers that read the file
    // while the writer is still alive (e.g. ParallelScanPartitioner's
    // collect_merged_partition_paths) see the full contents.
    if (fp_) std::fflush(fp_);
}

template<std::size_t N>
PartitionFile<N>::Reader::Reader(const std::string& path)
    : path_(path)
{
    fp_ = std::fopen(path.c_str(), "rb");
    if (!fp_) {
        eof_ = true;  // absent file == empty partition
    }
}

template<std::size_t N>
PartitionFile<N>::Reader::~Reader() {
    if (fp_) std::fclose(fp_);
}

template<std::size_t N>
bool PartitionFile<N>::Reader::next(Record<N>& out) {
    if (eof_ || !fp_) return false;
    std::size_t n = std::fread(&out, sizeof(Record<N>), 1, fp_);
    if (n != 1) {
        // Distinguish a real read failure from a clean EOF — treating an
        // I/O error as EOF would silently drop the tail of the partition.
        if (std::ferror(fp_)) {
            throw std::runtime_error(
                "PartitionFile: read error on " + path_ + ": "
                + std::strerror(errno));
        }
        eof_ = true;
        return false;
    }
    return true;
}

template<std::size_t N>
std::size_t PartitionFile<N>::Reader::read_batch(
    Record<N>* out, std::size_t max_records)
{
    if (eof_ || !fp_ || max_records == 0) return 0;
    std::size_t n = std::fread(out, sizeof(Record<N>), max_records, fp_);
    if (n < max_records) {
        // Distinguish a real read failure from a clean EOF — treating an
        // I/O error as EOF would silently drop the tail of the partition.
        if (std::ferror(fp_)) {
            throw std::runtime_error(
                "PartitionFile: read error on " + path_ + ": "
                + std::strerror(errno));
        }
        eof_ = true;  // hit EOF
    }
    return n;
}

template class PartitionFile<1>;
template class PartitionFile<2>;
template class PartitionFile<3>;

}  // namespace GQL
