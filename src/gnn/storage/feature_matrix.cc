#include "gnn/storage/feature_matrix.h"

#include <stdexcept>

namespace mdb::gnn {

FeatureMatrix::FeatureMatrix(FeatureMatrix&& other) noexcept
    : header_(other.header_),
      path_(std::move(other.path_)),
      mmap_ptr_(other.mmap_ptr_),
      mmap_size_(other.mmap_size_)
{
    other.mmap_ptr_  = nullptr;
    other.mmap_size_ = 0;
}

FeatureMatrix& FeatureMatrix::operator=(FeatureMatrix&& other) noexcept {
    if (this != &other) {
        // TODO: munmap current if needed
        header_    = other.header_;
        path_      = std::move(other.path_);
        mmap_ptr_  = other.mmap_ptr_;
        mmap_size_ = other.mmap_size_;
        other.mmap_ptr_  = nullptr;
        other.mmap_size_ = 0;
    }
    return *this;
}

FeatureMatrix::~FeatureMatrix() {
    // TODO: munmap
}

const void* FeatureMatrix::data_ptr() const {
    return static_cast<const char*>(mmap_ptr_) + FeatureMatrixHeader::SIZE;
}

FeatureMatrix FeatureMatrix::create(
    const fs::path& /*path*/, uint64_t /*num_rows*/, uint64_t /*num_cols*/,
    GnnDtype /*dtype*/, const void* /*data*/)
{
    throw std::runtime_error("FeatureMatrix::create not implemented yet");
}

FeatureMatrix FeatureMatrix::create_streaming(
    const fs::path& /*path*/, uint64_t /*num_rows*/, uint64_t /*num_cols*/,
    GnnDtype /*dtype*/, RowWriter /*writer*/)
{
    throw std::runtime_error("FeatureMatrix::create_streaming not implemented yet");
}

FeatureMatrix FeatureMatrix::open(const fs::path& /*path*/) {
    throw std::runtime_error("FeatureMatrix::open not implemented yet");
}

const void* FeatureMatrix::row(uint64_t /*row_id*/) const {
    throw std::runtime_error("FeatureMatrix::row not implemented yet");
}

void FeatureMatrix::scan(RowCallback /*callback*/) const {
    throw std::runtime_error("FeatureMatrix::scan not implemented yet");
}

void FeatureMatrix::extract_rows(
    const std::vector<uint64_t>& /*row_ids*/, void* /*out*/) const
{
    throw std::runtime_error("FeatureMatrix::extract_rows not implemented yet");
}

FeatureMatrix FeatureMatrix::create_reordered(
    const FeatureMatrix& /*source*/,
    const std::vector<uint64_t>& /*permutation*/,
    const fs::path& /*output_path*/)
{
    throw std::runtime_error("FeatureMatrix::create_reordered not implemented yet");
}

} // namespace mdb::gnn
