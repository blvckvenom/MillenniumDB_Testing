#include "gnn/sampling/sample_storage.h"

#include <algorithm>
#include <fstream>
#include <stdexcept>
#include <unordered_map>

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

struct SampleStorage::Impl {
    std::filesystem::path storage_path;
    SampleCatalog catalog;
    bool write_mode = false;
    bool finalized = false;

    // Write mode streams
    std::unique_ptr<std::ofstream> batch_data_stream;
    std::unique_ptr<std::ofstream> batch_index_stream;

    // Index: batch_id -> (file_offset, data_size)
    std::vector<std::pair<uint64_t, uint64_t>> batch_index;

    // Frequency tracking
    std::unordered_map<uint64_t, uint64_t> node_frequencies;

    /// Index: split_type -> batch IDs (populated on load/write)
    std::unordered_map<int, std::vector<uint64_t>> split_index;

    // Statistics tracking
    uint64_t total_batches_written = 0;
    uint64_t train_batches_written = 0;
    uint64_t validation_batches_written = 0;
    uint64_t test_batches_written = 0;
    std::unordered_set<uint64_t> unique_nodes_seen;
    uint64_t total_edges_written = 0;

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

        // Serialize sample to buffer
        std::ostringstream buffer(std::ios::binary);
        sample.serialize(buffer);
        std::string data = buffer.str();

        // Write to file
        batch_data_stream->write(data.data(), data.size());

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
            unique_nodes_seen.insert(node.id);
            node_frequencies[node.id]++;
        }

        // Track edges
        total_edges_written += sample.total_edges();

        // Update split index
        split_index[static_cast<int>(sample.split)].push_back(sample.batch_id);
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

        write_value(freq_out, FREQ_MAGIC);
        write_value(freq_out, static_cast<uint32_t>(1));  // Version
        write_value(freq_out, static_cast<uint64_t>(node_frequencies.size()));

        for (const auto& [node_id, count] : node_frequencies) {
            write_value(freq_out, node_id);
            write_value(freq_out, count);
        }

        // Update and save catalog
        catalog.total_batches = total_batches_written;
        catalog.train_batches = train_batches_written;
        catalog.validation_batches = validation_batches_written;
        catalog.test_batches = test_batches_written;
        catalog.unique_nodes = unique_nodes_seen.size();
        catalog.total_edges = total_edges_written;

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

        // Build split index from stored data
        auto data_path = storage_path / BATCH_DATA_FILE;
        std::ifstream data_in(data_path, std::ios::binary);
        if (data_in) {
            for (uint64_t i = 0; i < batch_index.size(); ++i) {
                if (batch_index[i].second == 0) continue;
                data_in.seekg(batch_index[i].first);
                GraphSample sample = GraphSample::deserialize(data_in);
                split_index[static_cast<int>(sample.split)].push_back(i);
            }
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

        // Open data file and seek to offset
        auto data_path = storage_path / BATCH_DATA_FILE;
        std::ifstream data_in(data_path, std::ios::binary);
        if (!data_in) {
            throw std::runtime_error("Failed to open batch data file");
        }

        data_in.seekg(offset);

        // Read and deserialize
        return GraphSample::deserialize(data_in);
    }

    std::unordered_map<uint64_t, uint64_t> load_frequencies() {
        if (!node_frequencies.empty()) {
            return node_frequencies;
        }

        auto freq_path = storage_path / FREQUENCY_FILE;
        std::ifstream freq_in(freq_path, std::ios::binary);
        if (!freq_in) {
            return {};
        }

        uint32_t magic = read_value<uint32_t>(freq_in);
        if (magic != FREQ_MAGIC) {
            return {};
        }

        read_value<uint32_t>(freq_in);  // Skip version
        uint64_t num_entries = read_value<uint64_t>(freq_in);

        for (uint64_t i = 0; i < num_entries; ++i) {
            uint64_t node_id = read_value<uint64_t>(freq_in);
            uint64_t count = read_value<uint64_t>(freq_in);
            node_frequencies[node_id] = count;
        }

        return node_frequencies;
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
        } catch (...) {
            // Ignore errors in destructor
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
