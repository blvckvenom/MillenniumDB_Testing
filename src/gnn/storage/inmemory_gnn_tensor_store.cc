#include "inmemory_gnn_tensor_store.h"

#include <algorithm>
#include <cstring>

namespace mdb::gnn {

bool InMemoryGnnTensorStore::store(const std::string& key,
                                   const void* data,
                                   const std::vector<int64_t>& shape,
                                   GnnDtype dtype) {
    if (data == nullptr || shape.empty()) {
        return false;
    }

    // Validate shape (no zero or negative dimensions)
    for (auto dim : shape) {
        if (dim <= 0) {
            return false;
        }
    }

    // Compute metadata
    GnnTensorMetadata metadata;
    metadata.shape = shape;
    metadata.dtype = dtype;
    metadata.byte_size = metadata.compute_byte_size();

    // Create storage entry with copied data
    StorageEntry entry;
    entry.metadata = std::move(metadata);
    entry.data.resize(entry.metadata.byte_size);
    std::memcpy(entry.data.data(), data, entry.metadata.byte_size);

    // Store with exclusive lock
    std::unique_lock lock(mutex_);

    // Update total size (subtract old entry if exists)
    auto it = storage_.find(key);
    if (it != storage_.end()) {
        total_bytes_ -= it->second.metadata.byte_size;
    }

    total_bytes_ += entry.metadata.byte_size;
    storage_[key] = std::move(entry);

    return true;
}

GnnTensorView InMemoryGnnTensorStore::load(const std::string& key) const {
    std::shared_lock lock(mutex_);

    auto it = storage_.find(key);
    if (it == storage_.end()) {
        return GnnTensorView();  // Invalid view
    }

    return GnnTensorView(it->second.data.data(), it->second.metadata);
}

bool InMemoryGnnTensorStore::remove(const std::string& key) {
    std::unique_lock lock(mutex_);

    auto it = storage_.find(key);
    if (it == storage_.end()) {
        return false;
    }

    total_bytes_ -= it->second.metadata.byte_size;
    storage_.erase(it);
    return true;
}

bool InMemoryGnnTensorStore::exists(const std::string& key) const {
    std::shared_lock lock(mutex_);
    return storage_.find(key) != storage_.end();
}

std::optional<GnnTensorMetadata> InMemoryGnnTensorStore::get_metadata(const std::string& key) const {
    std::shared_lock lock(mutex_);

    auto it = storage_.find(key);
    if (it == storage_.end()) {
        return std::nullopt;
    }

    return it->second.metadata;
}

std::vector<std::string> InMemoryGnnTensorStore::list_keys() const {
    std::shared_lock lock(mutex_);

    std::vector<std::string> keys;
    keys.reserve(storage_.size());
    for (const auto& [key, _] : storage_) {
        keys.push_back(key);
    }
    return keys;
}

size_t InMemoryGnnTensorStore::total_size() const {
    std::shared_lock lock(mutex_);
    return total_bytes_;
}

size_t InMemoryGnnTensorStore::count() const {
    std::shared_lock lock(mutex_);
    return storage_.size();
}

void InMemoryGnnTensorStore::clear() {
    std::unique_lock lock(mutex_);
    storage_.clear();
    total_bytes_ = 0;
}

} // namespace mdb::gnn
