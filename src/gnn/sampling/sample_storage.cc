#include "gnn/sampling/sample_storage.h"

#include <algorithm>
#include <fstream>
#include <list>
#include <mutex>
#include <sstream>
#include <stdexcept>
#include <streambuf>
#include <unordered_map>

#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#include "gnn/common/page_cache_hint.h"  // Fix #22
#include "gnn/sampling/sample_fingerprint.h"  // STEP 8 content fingerprint
#include "gnn/storage/row_mapping.h"
#include "misc/logger.h"

namespace mdb::gnn {

// =============================================================================
// File Format Constants
// =============================================================================

namespace {

constexpr const char* SAMPLES_DIR = "samples";
constexpr const char* BATCH_DATA_FILE = "batches.dat";
constexpr const char* BATCH_INDEX_FILE = "batches.idx";
constexpr const char* FREQUENCY_FILE = "frequency.dat";

constexpr uint32_t BATCH_MAGIC = 0x48435442;  // "BTCH"
constexpr uint32_t INDEX_MAGIC = 0x58444E49;  // "INDX"
constexpr uint32_t FREQ_MAGIC = 0x51455246;   // "FREQ"

} // anonymous namespace

// =============================================================================
// Implementation
// =============================================================================

// Estimate the in-RAM footprint of a deserialized GraphSample (for the
// read-mode sample cache budget). Counts the dynamic vector payloads plus
// per-vector overhead; close enough to bound the cache to a byte budget.
inline size_t estimate_sample_bytes(const GraphSample& s) {
    size_t b = sizeof(GraphSample);
    for (const auto& layer : s.nodes_per_layer) {
        b += sizeof(layer) + layer.size() * sizeof(ObjectId);
    }
    for (const auto& e : s.edges_per_layer) {
        b += sizeof(e)
           + e.src_indices.size() * sizeof(int32_t)
           + e.dst_indices.size() * sizeof(int32_t)
           + e.edge_ids.size()    * sizeof(ObjectId);
    }
    b += s.all_unique_nodes.size() * sizeof(ObjectId);
    return b;
}

// Fix #19 (2026-05-13): membuf wrapping an mmap'd region so we can drive
// GraphSample::deserialize(istream&) without copying batches.dat into a
// user-space buffer. Eliminates ~87 GB of memcpy on papers100M MinHash.
class MmapStreamBuf : public std::streambuf {
public:
    MmapStreamBuf(const char* data, size_t size) {
        char* begin = const_cast<char*>(data);
        setg(begin, begin, begin + size);
    }
    MmapStreamBuf(const MmapStreamBuf&) = delete;
    MmapStreamBuf& operator=(const MmapStreamBuf&) = delete;

protected:
    // Seekoff/seekpos needed because GraphSample::deserialize sometimes
    // calls .seekg(); the istream wrapper delegates to this streambuf.
    pos_type seekoff(off_type off, std::ios_base::seekdir way,
                     std::ios_base::openmode which) override {
        if (!(which & std::ios_base::in)) return pos_type(-1);
        char* base = eback();
        char* end  = egptr();
        char* cur  = gptr();
        char* target;
        switch (way) {
            case std::ios_base::beg: target = base + off; break;
            case std::ios_base::cur: target = cur + off; break;
            case std::ios_base::end: target = end + off; break;
            default: return pos_type(-1);
        }
        if (target < base || target > end) return pos_type(-1);
        setg(base, target, end);
        return pos_type(target - base);
    }
    pos_type seekpos(pos_type pos, std::ios_base::openmode which) override {
        return seekoff(off_type(pos), std::ios_base::beg, which);
    }
};

class MmapIStream : public std::istream {
public:
    MmapIStream(const char* data, size_t size)
        : std::istream(&buf_), buf_(data, size) {}

private:
    MmapStreamBuf buf_;
};

struct SampleStorage::Impl {
    std::filesystem::path storage_path;
    SampleCatalog catalog;
    bool write_mode = false;
    bool finalized = false;

    // Write mode streams
    std::unique_ptr<std::ofstream> batch_data_stream;
    std::unique_ptr<std::ofstream> batch_index_stream;

    // Fix #19: read-mode mmap of batches.dat. Populated by init_read_mode
    // when the file is accessible; falls back to per-call ifstream if
    // mmap fails (e.g. tmpfs, very large file > address space).
    void*  data_mmap_ptr_  = nullptr;
    size_t data_mmap_size_ = 0;

    // Index: batch_id -> (file_offset, data_size)
    std::vector<std::pair<uint64_t, uint64_t>> batch_index;

    // Frequency tracking — sparse fallback (when no RowMapping)
    std::unordered_map<uint64_t, uint64_t> node_frequencies;

    // Dense frequency tracking (when RowMapping available during write)
    const RowMapping* row_mapping_ = nullptr;  // non-owning, optional
    std::vector<uint64_t> node_freq_dense;     // indexed by row_index
    std::vector<bool> node_seen;               // bitset indexed by row_index
    uint64_t dense_unique_count = 0;           // count of unique nodes (bitset path)

    // Dense frequency cache (for read-mode loading of v2 files)
    std::vector<uint64_t> dense_freq_cache_;
    int freq_version_ = 0;  // 0=not loaded, 1=v1, 2=v2

    /// Index: split_type -> batch IDs (populated on load/write)
    std::unordered_map<int, std::vector<uint64_t>> split_index;

    // Statistics tracking
    uint64_t total_batches_written = 0;
    uint64_t train_batches_written = 0;
    uint64_t validation_batches_written = 0;
    uint64_t test_batches_written = 0;
    std::unordered_set<uint64_t> unique_nodes_seen;  // sparse fallback
    uint64_t total_edges_written = 0;

    // STEP 8: order-independent XOR fold of per-batch content hashes. Commutative
    // so it is invariant to worker write-completion order (numWorkers>=2); each
    // batch_id contributes exactly once via write_sample_impl.
    uint64_t content_fp_accumulator_ = 0;

    // Deserialized-sample LRU cache (read-mode hot path). 0 budget = disabled.
    // Guarded by cache_mu_; the disk read + deserialize itself happens OUTSIDE
    // the lock so concurrent readers don't serialize on I/O.
    struct CacheEntry {
        GraphSample sample;
        std::list<uint64_t>::iterator lru_it;
        size_t bytes;
    };
    mutable std::mutex cache_mu_;
    size_t cache_budget_ = 0;
    size_t cache_bytes_  = 0;
    // When set, read_sample_impl deserializes WITHOUT the per-layer edge_ids
    // blocks (seeks past them) — ~halves the per-batch sample read I/O. Enabled
    // by the training path, which never consumes sample edge_ids.
    bool   skip_edge_ids_ = false;
    // When set, read_sample_impl deserializes WITHOUT the per-layer src/dst edge
    // index blocks (seeks past them) — used when baked blocks supply the edge
    // structure at train time. Enabled by the training path in that case.
    bool   skip_edges_ = false;
    std::unordered_map<uint64_t, CacheEntry> sample_cache_;
    std::list<uint64_t> cache_lru_;  // front = most recently used
    uint64_t cache_hits_ = 0;
    uint64_t cache_misses_ = 0;
    uint64_t cache_evictions_ = 0;

    Impl() = default;

    // =========================================================================
    // Write Mode Methods
    // =========================================================================

    void init_write_mode(const std::filesystem::path& path, const SamplingConfig& config) {
        storage_path = path;
        write_mode = true;
        finalized = false;
        catalog = SampleCatalog(config);

        // Create directory
        std::filesystem::create_directories(storage_path);

        // Open data file
        auto data_path = storage_path / BATCH_DATA_FILE;
        batch_data_stream = std::make_unique<std::ofstream>(data_path, std::ios::binary);
        if (!*batch_data_stream) {
            throw std::runtime_error("Failed to create batch data file: " + data_path.string());
        }

        // Write data file header
        write_value(*batch_data_stream, BATCH_MAGIC);
        write_value(*batch_data_stream, static_cast<uint32_t>(1));  // Version
    }

    void write_sample_impl(const GraphSample& sample) {
        if (!write_mode || finalized) {
            throw std::runtime_error("Storage not in write mode");
        }

        // Record offset before writing
        uint64_t offset = batch_data_stream->tellp();

        // Serialize sample to an in-memory buffer first.
        // If serialize() throws, no side-effects have occurred.
        std::ostringstream buffer(std::ios::binary);
        sample.serialize(buffer);
        std::string data = buffer.str();

        // Write to file. If write fails, we must not update any statistics.
        batch_data_stream->write(data.data(), data.size());
        if (!batch_data_stream->good()) {
            // Attempt to seek back to the original offset so the data file
            // remains consistent for any subsequent writes after recovery.
            batch_data_stream->clear();
            batch_data_stream->seekp(offset);
            throw std::runtime_error("Failed to write batch data for batch_id "
                                     + std::to_string(sample.batch_id));
        }

        // --- Point of no return ------------------------------------------------
        // The write succeeded. All mutations below update in-memory statistics
        // and index structures. These are non-throwing (STL container operations
        // with pre-reserved or small allocations), so we will not leave the
        // storage in an inconsistent state.
        // -----------------------------------------------------------------------

        // Update index
        if (sample.batch_id >= batch_index.size()) {
            batch_index.resize(sample.batch_id + 1, {0, 0});
        }
        batch_index[sample.batch_id] = {offset, data.size()};

        // Update statistics
        total_batches_written++;
        switch (sample.split) {
            case SplitType::TRAIN:
                train_batches_written++;
                break;
            case SplitType::VALIDATION:
                validation_batches_written++;
                break;
            case SplitType::TEST:
                test_batches_written++;
                break;
        }

        // Track unique nodes and frequencies
        for (const auto& node : sample.all_unique_nodes) {
            if (row_mapping_) {
                // Dense path: O(log N) lookup per node, O(1) vector access
                auto row = row_mapping_->find(node);
                if (row.has_value() && *row < node_freq_dense.size()) {
                    node_freq_dense[*row]++;
                    if (!node_seen[*row]) {
                        node_seen[*row] = true;
                        dense_unique_count++;
                    }
                }
                // Nodes not in RowMapping are silently skipped — they are
                // outside the projected graph and have no feature row.
            } else {
                // Sparse fallback: hash-map based tracking
                unique_nodes_seen.insert(node.id);
                node_frequencies[node.id]++;
            }
        }

        // Track edges
        total_edges_written += sample.total_edges();

        // Update split index
        split_index[static_cast<int>(sample.split)].push_back(sample.batch_id);

        // STEP 8: fold this batch's layout-independent content hash into the
        // sample fingerprint. XOR is commutative so the result is invariant to
        // worker write-completion order; the per-batch hash is keyed by batch_id
        // so distinct batches do not cancel.
        content_fp_accumulator_ ^= compute_batch_content_hash(sample);
    }

    void finalize_impl() {
        if (!write_mode || finalized) {
            return;
        }

        // Close data stream
        batch_data_stream.reset();

        // Write index file
        auto index_path = storage_path / BATCH_INDEX_FILE;
        std::ofstream index_out(index_path, std::ios::binary);
        if (!index_out) {
            throw std::runtime_error("Failed to create index file: " + index_path.string());
        }

        write_value(index_out, INDEX_MAGIC);
        write_value(index_out, static_cast<uint32_t>(1));  // Version
        write_value(index_out, static_cast<uint64_t>(batch_index.size()));

        for (const auto& [offset, size] : batch_index) {
            write_value(index_out, offset);
            write_value(index_out, size);
        }

        // Write frequency file
        auto freq_path = storage_path / FREQUENCY_FILE;
        std::ofstream freq_out(freq_path, std::ios::binary);
        if (!freq_out) {
            throw std::runtime_error("Failed to create frequency file: " + freq_path.string());
        }

        if (row_mapping_) {
            // Dense format (v2): N consecutive uint64 counts indexed by row_index
            write_value(freq_out, FREQ_MAGIC);
            write_value(freq_out, static_cast<uint32_t>(2));  // Version 2
            uint64_t count = node_freq_dense.size();
            write_value(freq_out, count);
            freq_out.write(reinterpret_cast<const char*>(node_freq_dense.data()),
                           count * sizeof(uint64_t));
        } else {
            // Sparse format (v1): [oid, count] pairs
            write_value(freq_out, FREQ_MAGIC);
            write_value(freq_out, static_cast<uint32_t>(1));  // Version 1
            write_value(freq_out, static_cast<uint64_t>(node_frequencies.size()));

            for (const auto& [node_id, count] : node_frequencies) {
                write_value(freq_out, node_id);
                write_value(freq_out, count);
            }
        }

        // Update and save catalog
        catalog.total_batches = total_batches_written;
        catalog.train_batches = train_batches_written;
        catalog.validation_batches = validation_batches_written;
        catalog.test_batches = test_batches_written;
        catalog.unique_nodes = row_mapping_ ? dense_unique_count : unique_nodes_seen.size();
        catalog.total_edges = total_edges_written;

        // STEP 8: persist the content fingerprint. 0 is reserved for
        // "absent/UNKNOWN", so map a (vanishingly rare) all-XOR-cancelled 0 to 1.
        catalog.sample_content_fp =
            (content_fp_accumulator_ == 0) ? 1 : content_fp_accumulator_;

        catalog.save(storage_path);

        finalized = true;
        write_mode = false;
    }

    // =========================================================================
    // Read Mode Methods
    // =========================================================================

    void init_read_mode(const std::filesystem::path& path) {
        storage_path = path;
        write_mode = false;
        finalized = true;

        // Load catalog
        catalog = SampleCatalog::load(storage_path);

        // Load index
        auto index_path = storage_path / BATCH_INDEX_FILE;
        std::ifstream index_in(index_path, std::ios::binary);
        if (!index_in) {
            throw std::runtime_error("Failed to open index file: " + index_path.string());
        }

        uint32_t magic = read_value<uint32_t>(index_in);
        if (magic != INDEX_MAGIC) {
            throw std::runtime_error("Invalid index file magic");
        }

        uint32_t version = read_value<uint32_t>(index_in);
        if (version != 1) {
            throw std::runtime_error("Unsupported index version: " + std::to_string(version));
        }

        uint64_t num_entries = read_value<uint64_t>(index_in);
        batch_index.resize(num_entries);

        for (uint64_t i = 0; i < num_entries; ++i) {
            uint64_t offset = read_value<uint64_t>(index_in);
            uint64_t size = read_value<uint64_t>(index_in);
            batch_index[i] = {offset, size};
        }

        // Build split index by reading only the split field from each batch header
        // instead of deserializing the entire GraphSample
        auto data_path = storage_path / BATCH_DATA_FILE;
        std::ifstream data_in(data_path, std::ios::binary);
        if (data_in) {
            for (uint64_t i = 0; i < batch_index.size(); ++i) {
                if (batch_index[i].second == 0) continue;
                data_in.seekg(batch_index[i].first);
                SplitType split = GraphSample::read_split(data_in);
                split_index[static_cast<int>(split)].push_back(i);
            }
        }

        // Fix #19: mmap batches.dat for zero-copy read_sample. OPT-IN
        // via env var MDB_GNN_MMAP_BATCHES_DAT=1 because empirical
        // validation on papers100M (30 GB RAM) showed the mmap'd 87 GB
        // region competes with reordered.fmat + cache files for page
        // cache, pushing L4 packed_slim throughput from 5.5 → 3.6
        // batches/s (+121 s wall-clock penalty) despite saving ~6 s
        // on MinHash compute. Net loss on memory-constrained systems.
        // Enable when the host has >> file size in RAM.
        bool enable_mmap = false;
        if (const char* env = std::getenv("MDB_GNN_MMAP_BATCHES_DAT")) {
            std::string s(env);
            if (s == "1" || s == "true" || s == "yes") enable_mmap = true;
        }
        if (enable_mmap) {
            struct stat st{};
            if (::stat(data_path.c_str(), &st) == 0 && st.st_size > 0) {
                int fd = ::open(data_path.c_str(), O_RDONLY);
                if (fd >= 0) {
                    void* p = ::mmap(nullptr, static_cast<size_t>(st.st_size),
                                     PROT_READ, MAP_PRIVATE, fd, 0);
                    ::close(fd);
                    if (p != MAP_FAILED) {
                        data_mmap_ptr_  = p;
                        data_mmap_size_ = static_cast<size_t>(st.st_size);
                    }
                }
            }
        }
    }

    ~Impl() {
        if (data_mmap_ptr_ != nullptr) {
            ::munmap(data_mmap_ptr_, data_mmap_size_);
            data_mmap_ptr_ = nullptr;
        }
    }

    GraphSample read_sample_impl(uint64_t batch_id) {
        if (write_mode && !finalized) {
            throw std::runtime_error("Cannot read from storage in write mode");
        }

        if (batch_id >= batch_index.size()) {
            throw std::runtime_error("Batch ID out of range: " + std::to_string(batch_id));
        }

        auto [offset, size] = batch_index[batch_id];
        if (size == 0) {
            throw std::runtime_error("Batch ID not found: " + std::to_string(batch_id));
        }

        // Cache fast path: serve a previously-read batch from the in-RAM LRU of
        // deserialized samples (skips disk read + deserialize). The disk read
        // below runs OUTSIDE the lock so concurrent readers don't serialize.
        if (cache_budget_ > 0) {
            std::lock_guard<std::mutex> lk(cache_mu_);
            auto it = sample_cache_.find(batch_id);
            if (it != sample_cache_.end()) {
                ++cache_hits_;
                cache_lru_.splice(cache_lru_.begin(), cache_lru_, it->second.lru_it);
                return it->second.sample;  // copy
            }
            ++cache_misses_;
        }

        GraphSample sample;
        // Fix #19: prefer the mmap'd region (zero-copy parse) when
        // init_read_mode succeeded in mmap'ing batches.dat. Fall back to
        // the legacy ifstream open per call if mmap was unavailable.
        if (data_mmap_ptr_ != nullptr &&
            offset + size <= data_mmap_size_)
        {
            const char* base = static_cast<const char*>(data_mmap_ptr_);
            MmapIStream stream(base + offset, size);
            sample = GraphSample::deserialize(stream, skip_edge_ids_, skip_edges_);
            // Fix #22: release this region of batches.dat from the page cache.
            // The MmapStreamBuf wraps a const-pointer view; the underlying memory
            // is read-only, so the cast away const is safe (madvise doesn't write).
            madvise_dontneed(const_cast<char*>(base + offset), size);
        } else {
            // Legacy path — fresh ifstream per call.
            auto data_path = storage_path / BATCH_DATA_FILE;
            std::ifstream data_in(data_path, std::ios::binary);
            if (!data_in) {
                throw std::runtime_error("Failed to open batch data file");
            }
            data_in.seekg(offset);
            sample = GraphSample::deserialize(data_in, skip_edge_ids_, skip_edges_);

            // Fix #22: tell the kernel we're done with [offset, offset+size).
            // ifstream doesn't expose its underlying fd, so we open a brief
            // read-only fd just to issue the hint. Adds one open+close per
            // batch but avoids 87 GB of stale batches.dat pages in the cache
            // on papers100M-scale runs.
            int hint_fd = ::open(data_path.c_str(), O_RDONLY);
            if (hint_fd >= 0) {
                fadvise_dontneed(hint_fd, static_cast<off_t>(offset),
                                 static_cast<off_t>(size));
                ::close(hint_fd);
            }
        }

        // Cache insert (with LRU eviction to stay under budget). We keep our own
        // deserialized copy in RAM, so the Fix #22 DONTNEED above is fine — the
        // page cache is freed while our cache retains the hot working set.
        if (cache_budget_ > 0) {
            std::lock_guard<std::mutex> lk(cache_mu_);
            if (sample_cache_.find(batch_id) == sample_cache_.end()) {
                size_t sz = estimate_sample_bytes(sample);
                while (cache_bytes_ + sz > cache_budget_ && !cache_lru_.empty()) {
                    uint64_t victim = cache_lru_.back();
                    cache_lru_.pop_back();
                    auto vit = sample_cache_.find(victim);
                    if (vit != sample_cache_.end()) {
                        cache_bytes_ -= vit->second.bytes;
                        sample_cache_.erase(vit);
                        ++cache_evictions_;
                    }
                }
                if (sz <= cache_budget_) {
                    cache_lru_.push_front(batch_id);
                    sample_cache_.emplace(
                        batch_id, CacheEntry{sample, cache_lru_.begin(), sz});
                    cache_bytes_ += sz;
                }
            }
        }
        return sample;
    }

    void load_frequencies_if_needed() {
        if (freq_version_ != 0) {
            return;  // Already loaded
        }

        // Check in-memory data from write mode before trying disk
        if (!node_frequencies.empty()) {
            // Sparse path was used during write
            freq_version_ = 1;
            return;
        }
        if (!node_freq_dense.empty()) {
            // Dense path was used during write — copy to cache
            dense_freq_cache_ = node_freq_dense;
            freq_version_ = 2;
            return;
        }
        if (!dense_freq_cache_.empty()) {
            freq_version_ = 2;
            return;
        }

        auto freq_path = storage_path / FREQUENCY_FILE;
        std::ifstream freq_in(freq_path, std::ios::binary);
        if (!freq_in) {
            return;
        }

        uint32_t magic = read_value<uint32_t>(freq_in);
        if (magic != FREQ_MAGIC) {
            return;
        }

        uint32_t version = read_value<uint32_t>(freq_in);
        uint64_t num_entries = read_value<uint64_t>(freq_in);

        if (version == 2) {
            // Dense format: N consecutive uint64 counts
            dense_freq_cache_.resize(num_entries);
            freq_in.read(reinterpret_cast<char*>(dense_freq_cache_.data()),
                         num_entries * sizeof(uint64_t));
            freq_version_ = 2;
        } else {
            // v1 sparse format: [oid, count] pairs
            for (uint64_t i = 0; i < num_entries; ++i) {
                uint64_t node_id = read_value<uint64_t>(freq_in);
                uint64_t count = read_value<uint64_t>(freq_in);
                node_frequencies[node_id] = count;
            }
            freq_version_ = 1;
        }
    }

    /// Backward-compatible: returns hash map of oid->count.
    /// For v2 files this requires a RowMapping to convert back to OIDs,
    /// but the existing callers (get_node_frequencies, get_top_frequent_nodes)
    /// don't pass one, so v2 files convert using the RowMapping pointer
    /// only if one was provided at write time (still in memory).
    std::unordered_map<uint64_t, uint64_t> load_frequencies() {
        load_frequencies_if_needed();

        if (freq_version_ == 1) {
            return node_frequencies;
        }

        if (freq_version_ == 2 && row_mapping_) {
            // Convert dense data back to sparse map using the RowMapping.
            // Pick whichever dense vector is populated: dense_freq_cache_
            // (loaded from disk) or node_freq_dense (from write mode).
            const auto& dense = !dense_freq_cache_.empty()
                ? dense_freq_cache_
                : node_freq_dense;
            std::unordered_map<uint64_t, uint64_t> result;
            for (uint64_t i = 0; i < dense.size(); ++i) {
                if (dense[i] > 0) {
                    ObjectId oid = row_mapping_->get(i);
                    result[oid.id] = dense[i];
                }
            }
            return result;
        }

        // freq_version_==2 without RowMapping: cannot convert dense→sparse.
        // Callers needing dense format should use get_dense_frequencies().
        return {};
    }

    // =========================================================================
    // Binary I/O Helpers
    // =========================================================================

    template<typename T>
    static void write_value(std::ostream& out, const T& value) {
        out.write(reinterpret_cast<const char*>(&value), sizeof(T));
    }

    template<typename T>
    static T read_value(std::istream& in) {
        T value;
        in.read(reinterpret_cast<char*>(&value), sizeof(T));
        return value;
    }
};

// =============================================================================
// SampleStorage Public Interface
// =============================================================================

SampleStorage::SampleStorage() : impl_(std::make_unique<Impl>()) {}

SampleStorage::~SampleStorage() {
    // Ensure finalization on destruction
    if (impl_ && impl_->write_mode && !impl_->finalized) {
        try {
            impl_->finalize_impl();
        } catch (const std::exception& e) {
            logger.error() << "SampleStorage finalization failed in destructor: " << e.what();
        } catch (...) {
            logger.error() << "SampleStorage finalization failed with unknown exception";
        }
    }
}

SampleStorage::SampleStorage(SampleStorage&&) noexcept = default;
SampleStorage& SampleStorage::operator=(SampleStorage&&) noexcept = default;

SampleStorage SampleStorage::create(
    const std::filesystem::path& db_folder,
    const SamplingConfig& config
) {
    auto path = get_storage_path(db_folder, config.sample_name);

    if (std::filesystem::exists(path)) {
        throw std::runtime_error("Sample storage already exists: " + path.string());
    }

    SampleStorage storage;
    storage.impl_->init_write_mode(path, config);
    return storage;
}

SampleStorage SampleStorage::create(
    const std::filesystem::path& db_folder,
    const SamplingConfig& config,
    const RowMapping& row_mapping
) {
    auto path = get_storage_path(db_folder, config.sample_name);

    if (std::filesystem::exists(path)) {
        throw std::runtime_error("Sample storage already exists: " + path.string());
    }

    SampleStorage storage;
    storage.impl_->row_mapping_ = &row_mapping;
    storage.impl_->init_write_mode(path, config);

    // Initialize dense tracking structures
    uint64_t N = row_mapping.size();
    storage.impl_->node_freq_dense.assign(N, 0);
    storage.impl_->node_seen.assign(N, false);
    storage.impl_->dense_unique_count = 0;

    return storage;
}

SampleStorage SampleStorage::open(const std::filesystem::path& storage_path) {
    if (!std::filesystem::exists(storage_path)) {
        throw std::runtime_error("Sample storage not found: " + storage_path.string());
    }

    SampleStorage storage;
    storage.impl_->init_read_mode(storage_path);
    return storage;
}

bool SampleStorage::exists(
    const std::filesystem::path& db_folder,
    const std::string& sample_name
) {
    return SampleCatalog::exists(get_storage_path(db_folder, sample_name));
}

std::filesystem::path SampleStorage::get_storage_path(
    const std::filesystem::path& db_folder,
    const std::string& sample_name
) {
    return db_folder / SAMPLES_DIR / sample_name;
}

void SampleStorage::write_sample(const GraphSample& sample) {
    impl_->write_sample_impl(sample);
}

void SampleStorage::finalize() {
    impl_->finalize_impl();
}

GraphSample SampleStorage::read_sample(uint64_t batch_id) {
    return impl_->read_sample_impl(batch_id);
}

void SampleStorage::set_skip_edge_ids_on_read(bool enable) {
    impl_->skip_edge_ids_ = enable;
}

void SampleStorage::set_skip_edges_on_read(bool enable) {
    impl_->skip_edges_ = enable;
}

void SampleStorage::set_sample_cache_budget_bytes(size_t budget_bytes) {
    std::lock_guard<std::mutex> lk(impl_->cache_mu_);
    impl_->cache_budget_ = budget_bytes;
    if (budget_bytes == 0) {
        // Disabling: drop everything so we don't pin RAM.
        impl_->sample_cache_.clear();
        impl_->cache_lru_.clear();
        impl_->cache_bytes_ = 0;
    }
}

SampleStorage::SampleCacheStats SampleStorage::sample_cache_stats() const {
    std::lock_guard<std::mutex> lk(impl_->cache_mu_);
    SampleCacheStats s;
    s.hits      = impl_->cache_hits_;
    s.misses    = impl_->cache_misses_;
    s.evictions = impl_->cache_evictions_;
    s.bytes     = impl_->cache_bytes_;
    s.budget    = impl_->cache_budget_;
    s.entries   = impl_->sample_cache_.size();
    return s;
}

std::vector<GraphSample> SampleStorage::read_samples(const std::vector<uint64_t>& batch_ids) {
    std::vector<GraphSample> samples;
    samples.reserve(batch_ids.size());

    for (uint64_t id : batch_ids) {
        samples.push_back(read_sample(id));
    }

    return samples;
}

std::vector<uint64_t> SampleStorage::get_batch_ids(SplitType split) {
    auto it = impl_->split_index.find(static_cast<int>(split));
    if (it != impl_->split_index.end()) {
        return it->second;
    }
    return {};
}

std::vector<uint64_t> SampleStorage::get_all_batch_ids() {
    std::vector<uint64_t> ids;

    for (uint64_t i = 0; i < impl_->batch_index.size(); ++i) {
        if (impl_->batch_index[i].second > 0) {
            ids.push_back(i);
        }
    }

    return ids;
}

const SampleCatalog& SampleStorage::get_catalog() const {
    return impl_->catalog;
}

const std::filesystem::path& SampleStorage::get_path() const {
    return impl_->storage_path;
}

bool SampleStorage::is_write_mode() const {
    return impl_->write_mode;
}

std::unordered_map<uint64_t, uint64_t> SampleStorage::get_node_frequencies() {
    return impl_->load_frequencies();
}

std::vector<uint64_t> SampleStorage::get_dense_frequencies(const RowMapping& rm) {
    impl_->load_frequencies_if_needed();

    if (impl_->freq_version_ == 2) {
        // Already dense — return directly
        return impl_->dense_freq_cache_;
    }

    if (impl_->freq_version_ == 1) {
        // Convert v1 (oid->count map) to dense vector via RowMapping
        uint64_t N = rm.size();
        std::vector<uint64_t> dense(N, 0);
        for (const auto& [oid_id, count] : impl_->node_frequencies) {
            auto row = rm.find(ObjectId(oid_id));
            if (row.has_value() && *row < N) {
                dense[*row] = count;
            }
        }
        return dense;
    }

    // No frequency data loaded
    return {};
}

std::vector<std::pair<ObjectId, uint64_t>> SampleStorage::get_top_frequent_nodes(size_t k) {
    auto freqs = get_node_frequencies();

    std::vector<std::pair<ObjectId, uint64_t>> sorted;
    sorted.reserve(freqs.size());

    for (const auto& [node_id, count] : freqs) {
        sorted.emplace_back(ObjectId(node_id), count);
    }

    // Partial sort for top-k
    if (k < sorted.size()) {
        std::partial_sort(sorted.begin(), sorted.begin() + k, sorted.end(),
                         [](const auto& a, const auto& b) { return a.second > b.second; });
        sorted.resize(k);
    } else {
        std::sort(sorted.begin(), sorted.end(),
                 [](const auto& a, const auto& b) { return a.second > b.second; });
    }

    return sorted;
}

} // namespace mdb::gnn
