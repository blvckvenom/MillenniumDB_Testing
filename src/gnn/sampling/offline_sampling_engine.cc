#include "gnn/sampling/offline_sampling_engine.h"

#include <fcntl.h>     // open
#include <unistd.h>    // fsync, close

#include <atomic>
#include <cerrno>
#include <chrono>
#include <cstdio>      // rename
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <mutex>
#include <stdexcept>
#include <system_error>
#include <thread>

#include "gnn/projection/gnn_meta.h"
#include "gnn/projection/topology_accessor.h"
#include "gnn/sampling/basic_khop_sampler.h"
#include "gnn/sampling/sample_storage.h"
#include "gnn/sampling/seed_selector.h"
#include "gnn/sampling/sorted_batch_sampler.h"
#include "gnn/storage/row_mapping.h"
#include "graph_models/gql/projection/projection_storage.h"
#include "query/query_context.h"

namespace mdb::gnn {

namespace {

// Spec #13 Phase 5 — `node_counts.bin` writer.
//
// Format (mirrors TopologyFrequencyProfiler::compute_from_node_counts_):
//   [8B magic "NODECNT0"]
//   [uint64_t num_nodes]
//   [uint64_t direction_bitmask]   (1=NATURAL, 2=REVERSE, 3=UNDIRECTED)
//   [num_nodes × uint64_t counts]
//
// Atomic write: temp file → fsync → rename → fsync(parent dir).
// Mirrors src/gnn/output/model_checkpoint.cc::save_full's idiom so a
// process crash mid-write never leaves a corrupted node_counts.bin.
constexpr uint8_t kNodeCountsMagic[8] = {'N','O','D','E','C','N','T','0'};

void fsync_directory_(const std::filesystem::path& dir) {
    int fd = ::open(dir.c_str(), O_RDONLY);
    if (fd < 0) {
        std::cerr << "[OfflineSamplingEngine] WARNING: cannot open dir for "
                  << "fsync " << dir.string() << " (errno=" << errno
                  << "); node_counts.bin may not be durable.\n";
        return;
    }
    if (::fsync(fd) != 0) {
        std::cerr << "[OfflineSamplingEngine] WARNING: fsync(dir) failed "
                  << "for " << dir.string() << " (errno=" << errno
                  << ").\n";
    }
    ::close(fd);
}

void persist_node_counts_(
    const std::filesystem::path&        projection_dir,
    const std::vector<uint64_t>&        counts,
    EdgeOrientation                     orientation)
{
    if (projection_dir.empty()) return;
    if (counts.empty()) {
        // No tally accumulated (sample_neighbors_uniform was never
        // entered — degenerate run). Nothing to persist.
        return;
    }

    std::error_code ec;
    std::filesystem::create_directories(projection_dir, ec);
    auto target = projection_dir / "node_counts.bin";
    auto tmp    = projection_dir / "node_counts.bin.tmp";

    {
        std::ofstream f(tmp, std::ios::binary | std::ios::trunc);
        if (!f) {
            std::cerr << "[OfflineSamplingEngine] WARNING: cannot open "
                      << tmp.string() << " for write (errno="
                      << errno << "); node_counts.bin not persisted.\n";
            return;
        }

        const uint64_t num_nodes = static_cast<uint64_t>(counts.size());
        uint64_t direction_bitmask = 0;
        switch (orientation) {
            case EdgeOrientation::NATURAL:    direction_bitmask = 1; break;
            case EdgeOrientation::REVERSE:    direction_bitmask = 2; break;
            case EdgeOrientation::UNDIRECTED: direction_bitmask = 3; break;
        }

        f.write(reinterpret_cast<const char*>(kNodeCountsMagic), 8);
        f.write(reinterpret_cast<const char*>(&num_nodes),         sizeof(num_nodes));
        f.write(reinterpret_cast<const char*>(&direction_bitmask), sizeof(direction_bitmask));
        f.write(reinterpret_cast<const char*>(counts.data()),
                static_cast<std::streamsize>(num_nodes * sizeof(uint64_t)));
        if (!f) {
            std::cerr << "[OfflineSamplingEngine] WARNING: I/O error "
                      << "writing " << tmp.string()
                      << "; node_counts.bin not persisted.\n";
            std::filesystem::remove(tmp, ec);
            return;
        }
        f.flush();
        // Best-effort fsync of the temp file before rename. ofstream's
        // underlying FILE* doesn't expose its fd portably, so we close
        // here (RAII) and trust the OS — the directory fsync below
        // commits the rename, which is the durability contract that
        // matters for this restart-correctness use case.
    }

    if (std::rename(tmp.c_str(), target.c_str()) != 0) {
        std::cerr << "[OfflineSamplingEngine] WARNING: rename "
                  << tmp.string() << " -> " << target.string()
                  << " failed (errno=" << errno
                  << "); node_counts.bin not persisted.\n";
        std::filesystem::remove(tmp, ec);
        return;
    }
    fsync_directory_(projection_dir);

    uint64_t logged_bitmask = 0;
    switch (orientation) {
        case EdgeOrientation::NATURAL:    logged_bitmask = 1; break;
        case EdgeOrientation::REVERSE:    logged_bitmask = 2; break;
        case EdgeOrientation::UNDIRECTED: logged_bitmask = 3; break;
    }
    std::cerr << "[OfflineSamplingEngine] Persisted "
              << target.string() << " (" << counts.size()
              << " nodes, direction_bitmask=" << logged_bitmask
              << "). Next gnn_offline_sample run will warm-start the "
              << "Four-Level Topology Store.\n";
}

}  // namespace

// =============================================================================
// Implementation
// =============================================================================

struct OfflineSamplingEngine::Impl {
    GQL::ProjectionStorage& storage;
    SamplingConfig config;
    std::filesystem::path db_folder;

    // Components
    std::unique_ptr<SeedSelector> seed_selector;
    std::unique_ptr<BasicKHopSampler> khop_sampler;
    std::unique_ptr<RowMapping> row_mapping;  ///< Loaded lazily when use_predefined_splits=true

    // Callbacks and settings
    ProgressCallback progress_callback;
    uint64_t progress_interval = 10;
    bool use_optimized_sampling = true;

    // State
    std::atomic<bool> cancel_requested{false};

    Impl(GQL::ProjectionStorage& storage_,
         const SamplingConfig& config_,
         const std::filesystem::path& db_folder_)
        : storage(storage_)
        , config(config_)
        , db_folder(db_folder_)
    {
        config.validate();
    }

    /**
     * @brief Initialize components lazily.
     *
     * When config.use_predefined_splits is true, loads the global RowMapping
     * from <db_folder>/gnn_features/<feature_name>.rmap so the SeedSelector
     * can index labels.bin/splits.bin written by graph_project. The
     * feature_name is read from <proj_dir>/gnn_meta.bin.
     */
    void init_components() {
        if (!row_mapping && config.use_predefined_splits && !db_folder.empty()) {
            // Locate gnn_meta.bin in projection directory
            auto proj_dir = std::filesystem::path(storage.get_projection_dir());
            auto meta_path = proj_dir / "gnn_meta.bin";
            if (!std::filesystem::exists(meta_path)) {
                throw std::runtime_error(
                    "usePredefinedSplits=true but gnn_meta.bin not found at: "
                    + meta_path.string()
                    + ". The projection must be created with labelProperty/splitProperty."
                );
            }
            auto meta = GnnMeta::read(meta_path);
            if (meta.feature_name.empty()) {
                throw std::runtime_error(
                    "gnn_meta.bin has empty feature_name; cannot locate RowMapping."
                );
            }
            auto rmap_path = db_folder / "gnn_features" / (meta.feature_name + ".rmap");
            if (!std::filesystem::exists(rmap_path)) {
                throw std::runtime_error(
                    "Feature RowMapping not found at: " + rmap_path.string()
                    + " (referenced by " + meta_path.string() + ")"
                );
            }
            row_mapping = std::make_unique<RowMapping>(RowMapping::open(rmap_path));
        }

        if (!seed_selector) {
            seed_selector = std::make_unique<SeedSelector>(storage, config, row_mapping.get());
        }
        if (!khop_sampler) {
            khop_sampler = std::make_unique<BasicKHopSampler>(storage, config);
        }
    }

    /**
     * @brief Report progress if callback is set.
     * @return false if cancellation requested
     */
    bool report_progress(const SamplingProgress& progress) {
        if (cancel_requested.load()) {
            return false;
        }
        if (progress_callback) {
            return progress_callback(progress);
        }
        return true;
    }

    /**
     * @brief Main sampling loop.
     */
    SamplingResult do_run(const std::filesystem::path& db_folder) {
        SamplingResult result;
        result.success = false;
        result.cancelled = false;
        result.total_samples = 0;
        result.total_seconds = 0;

        auto start_time = std::chrono::steady_clock::now();

        try {
            // Initialize components
            init_components();

            // Check if storage already exists
            if (SampleStorage::exists(db_folder, config.sample_name)) {
                result.error_message = "Sample storage already exists: " + config.sample_name;
                return result;
            }

            // Create storage
            // TODO(FourLevelStore): When RowMapping is available from the projection
            // context, pass it here to enable dense frequency tracking:
            //   SampleStorage::create(db_folder, config, row_mapping)
            // This would reduce RAM from ~9.5 GB to ~0.8 GB at 100M nodes.
            SampleStorage sample_storage = SampleStorage::create(db_folder, config);

            // Get seed split
            SeedSplit split = seed_selector->get_seed_split();

            if (split.train_seeds.empty()) {
                result.error_message = "No training seeds found";
                return result;
            }

            // Calculate totals - single pass (no epochs in sampling layer)
            uint64_t total_batches = seed_selector->total_batches_per_epoch();
            uint64_t batch_id = 0;

            // Progress tracking
            SamplingProgress progress;
            progress.total_batches = total_batches;
            progress.current_batch = 0;
            progress.current_split = SplitType::TRAIN;
            progress.samples_written = 0;

            // Generate batches ONCE (following DiskGNN architecture)
            // Using epoch=0 for initial shuffle; training will shuffle batch ORDER
            EpochBatches batches = seed_selector->generate_epoch_batches(0);

            // -----------------------------------------------------------------
            // Plan F (2026-05-11) — parallel-vs-legacy dispatch.
            //
            // `num_workers == 0`: preserve the pre-Plan-F single-threaded path
            //   bit-identically. The legacy lambda below loops sequentially
            //   through (train, val, test) batches and uses the primary
            //   sampler's persistent RNG. Outputs from earlier sessions stay
            //   reproducible.
            //
            // `num_workers >= 1`: parallel path. Each batch is re-seeded as
            //   `random_seed XOR batch_id` before sampling, so the output
            //   depends only on `(random_seed, batch_id)` and is identical
            //   across `num_workers ∈ {1, 2, 4, 20}`. Workers borrow the
            //   primary's TopologyAccessor (FourLevelTopologyStore + caches
            //   are read-only post-build) and own private LeapfrogGnnSampler,
            //   SeekBasedGnnSampler, RNG, and node_access_counts vector.
            //   `sample_storage.write_sample` is serialized via a single
            //   mutex; per-batch sampling work runs concurrently.
            // -----------------------------------------------------------------

            // Resolve effective worker count. 0 = legacy path; otherwise cap
            // at hardware_concurrency() so misconfiguration can't oversubscribe.
            uint32_t effective_workers = config.num_workers;
            if (effective_workers > 0) {
                const unsigned hw = std::thread::hardware_concurrency();
                if (hw > 0 && effective_workers > hw) {
                    effective_workers = static_cast<uint32_t>(hw);
                }
            }
            result.num_workers_used = effective_workers == 0
                ? 1
                : effective_workers;

            // Lambda to process batches for a split — LEGACY single-threaded
            // path. Used when config.num_workers == 0.
            auto process_batches_legacy = [&](
                const std::vector<std::vector<ObjectId>>& split_batches,
                SplitType split_type
            ) -> bool {
                progress.current_split = split_type;

                for (size_t i = 0; i < split_batches.size(); ++i) {
                    if (cancel_requested.load()) {
                        result.cancelled = true;
                        return false;
                    }

                    const auto& batch_seeds = split_batches[i];

                    // Sample k-hop neighborhood (no epoch parameter)
                    GraphSample sample = khop_sampler->sample(
                        batch_seeds,
                        batch_id,
                        split_type
                    );

                    // Write to storage
                    sample_storage.write_sample(sample);

                    batch_id++;
                    progress.current_batch = batch_id;
                    progress.samples_written++;

                    // Report progress periodically
                    if (progress.samples_written % progress_interval == 0) {
                        auto now = std::chrono::steady_clock::now();
                        progress.elapsed_seconds = std::chrono::duration<double>(
                            now - start_time
                        ).count();

                        double rate = progress.throughput();
                        uint64_t remaining = total_batches - progress.samples_written;
                        progress.estimated_remaining = rate > 0 ? remaining / rate : 0;

                        if (!report_progress(progress)) {
                            result.cancelled = true;
                            return false;
                        }
                    }
                }
                return true;
            };

            // -----------------------------------------------------------------
            // Plan F parallel path. Builds a flat work-queue across all three
            // splits (train, val, test) tagged with the destination split, so
            // a single atomic counter can dispatch any batch to any worker
            // and the `batch_id` assigned matches the legacy monotonic order
            // (train batches first, then val, then test).
            // -----------------------------------------------------------------
            auto process_batches_parallel = [&]() -> bool {
                struct WorkItem {
                    const std::vector<ObjectId>* seeds;
                    SplitType                    split;
                    uint64_t                     batch_id;
                };

                std::vector<WorkItem> work;
                work.reserve(
                    batches.train_batches.size()
                    + batches.validation_batches.size()
                    + batches.test_batches.size());

                auto append_split = [&](
                    const std::vector<std::vector<ObjectId>>& split_batches,
                    SplitType split_type)
                {
                    for (const auto& batch : split_batches) {
                        work.push_back({&batch, split_type, batch_id});
                        ++batch_id;
                    }
                };
                append_split(batches.train_batches,      SplitType::TRAIN);
                append_split(batches.validation_batches, SplitType::VALIDATION);
                append_split(batches.test_batches,       SplitType::TEST);

                const std::size_t total_work = work.size();

                // Build worker samplers. Worker 0 reuses the primary
                // `khop_sampler` (already through Phase 0 + four-level
                // enable). Workers 1..N-1 borrow the topology and own their
                // own Leapfrog/Seek samplers + RNG + tally vector.
                std::vector<std::unique_ptr<BasicKHopSampler>> worker_samplers;
                if (effective_workers > 1) {
                    worker_samplers.reserve(effective_workers - 1);
                    for (uint32_t w = 1; w < effective_workers; ++w) {
                        worker_samplers.push_back(
                            std::make_unique<BasicKHopSampler>(
                                storage,
                                config,
                                &khop_sampler->get_topology(),
                                w));
                    }
                }

                std::atomic<std::size_t> next_idx{0};
                std::mutex               write_mutex;

                // QueryContext is a thread_local pointer initialized only
                // for server-managed threads. Workers spawned via std::thread
                // start with _query_ctx == nullptr, so the first BPT access
                // (BufferManager::get_page_readonly reads
                // get_query_ctx().start_version) null-derefs and SIGSEGVs
                // the server. Capture the primary's ctx and install it on
                // each worker thread before any sampler call. Sampling reads
                // only the shared buffer (vp_map, protected by vp_mutex)
                // and does not touch the worker-indexed private buffer
                // (pp_map / tmp_info), so all workers can share the same
                // QueryContext without contention on the worker_index slot.
                QueryContext* primary_ctx = &get_query_ctx();

                auto worker_fn = [&](BasicKHopSampler* sampler) {
                    QueryContext::set_query_ctx(primary_ctx);
                    while (true) {
                        if (cancel_requested.load(std::memory_order_relaxed)) {
                            return;
                        }
                        const std::size_t idx =
                            next_idx.fetch_add(1, std::memory_order_relaxed);
                        if (idx >= total_work) return;

                        const WorkItem& item = work[idx];
                        sampler->reseed_for_batch(item.batch_id);
                        GraphSample sample = sampler->sample(
                            *item.seeds, item.batch_id, item.split);

                        std::lock_guard<std::mutex> lk(write_mutex);
                        sample_storage.write_sample(sample);
                        progress.samples_written++;
                        progress.current_batch  = item.batch_id + 1;
                        progress.current_split  = item.split;

                        if (progress.samples_written % progress_interval == 0) {
                            auto now = std::chrono::steady_clock::now();
                            progress.elapsed_seconds =
                                std::chrono::duration<double>(now - start_time).count();
                            double rate = progress.throughput();
                            uint64_t remaining =
                                total_batches - progress.samples_written;
                            progress.estimated_remaining =
                                rate > 0 ? remaining / rate : 0;
                            if (!report_progress(progress)) {
                                cancel_requested.store(true);
                            }
                        }
                    }
                };

                // Spawn workers. Primary runs in-thread to keep one fewer
                // OS thread alive and to match the legacy path's behavior
                // when num_workers == 1.
                std::vector<std::thread> threads;
                threads.reserve(effective_workers - 1);
                for (auto& w : worker_samplers) {
                    threads.emplace_back(worker_fn, w.get());
                }
                worker_fn(khop_sampler.get());
                for (auto& t : threads) t.join();

                // Merge per-worker tallies into the primary's vector so the
                // node_counts.bin write at the end of do_run reflects every
                // worker's contribution.
                for (const auto& w : worker_samplers) {
                    khop_sampler->merge_counts_from(w->node_access_counts());
                }

                return !cancel_requested.load();
            };

            if (effective_workers == 0) {
                // Process all splits (single pass, no epoch loop)
                if (process_batches_legacy(batches.train_batches, SplitType::TRAIN) &&
                    process_batches_legacy(batches.validation_batches, SplitType::VALIDATION)) {
                    process_batches_legacy(batches.test_batches, SplitType::TEST);
                }
            } else {
                process_batches_parallel();
                if (cancel_requested.load()) {
                    result.cancelled = true;
                }
            }

            // Finalize storage
            sample_storage.finalize();

            // Spec #13 Phase 5 (T13.2 writer half) — persist
            // `<projection_dir>/node_counts.bin` so the next
            // `gnn_offline_sample` run can warm-start the
            // Four-Level Topology Store. Guarded by
            // `useFourLevelTopologyStore`: when the user opted out of
            // the tiered cache there is no consumer for the file, so
            // skip the I/O. Failures inside `persist_node_counts_` log
            // to stderr but never throw — the sample itself already
            // succeeded; persistence is an optimization for future
            // runs.
            if (config.use_four_level_topology_store) {
                auto proj_dir =
                    std::filesystem::path(storage.get_projection_dir());
                persist_node_counts_(proj_dir,
                                     khop_sampler->node_access_counts(),
                                     config.orientation);
            }

            // Calculate final timing
            auto end_time = std::chrono::steady_clock::now();
            result.total_seconds = std::chrono::duration<double>(end_time - start_time).count();
            result.total_samples = batch_id;
            result.catalog = sample_storage.get_catalog();

            // Plan E (2026-05-11) — pull Phase 0 telemetry out of the
            // sampler so the procedure can yield it. Populated even when
            // the auto-profiler chose not to run (triggered=false) so
            // bench harnesses can tell apart "warm-start ready" from
            // "profiler ran and succeeded".
            if (khop_sampler) {
                auto rep = khop_sampler->phase0_report();
                result.phase0_triggered       = rep.triggered;
                result.phase0_succeeded       = rep.succeeded;
                result.phase0_walks_done      = static_cast<uint64_t>(rep.walks_done);
                result.phase0_lookups_done    = static_cast<uint64_t>(rep.lookups_done);
                result.phase0_elapsed_seconds = rep.elapsed_seconds;
            }

            if (!result.cancelled) {
                result.success = true;
            }

        } catch (const std::exception& e) {
            result.error_message = e.what();
            result.success = false;
        }

        return result;
    }

    /**
     * @brief Validate configuration.
     */
    std::string do_validate() {
        try {
            config.validate();
        } catch (const std::exception& e) {
            return e.what();
        }

        // Check projection has nodes
        init_components();
        if (seed_selector->total_seed_count() == 0) {
            return "Projection has no nodes";
        }

        return "";
    }

    /**
     * @brief Estimate total samples (single pass - no epochs).
     */
    uint64_t do_estimate_total_samples() {
        init_components();
        return seed_selector->total_batches_per_epoch();
    }

    /**
     * @brief Estimate storage size.
     *
     * Rough estimate based on:
     * - Each sample has ~batch_size * product(fanouts) nodes
     * - Each node is 8 bytes (ObjectId)
     * - Edges add ~50% overhead
     */
    uint64_t do_estimate_storage_bytes() {
        uint64_t nodes_per_sample = config.estimated_max_nodes_per_batch();
        uint64_t bytes_per_sample = nodes_per_sample * 8 * 2;  // nodes + edges overhead

        return do_estimate_total_samples() * bytes_per_sample;
    }
};

// =============================================================================
// OfflineSamplingEngine Public Interface
// =============================================================================

OfflineSamplingEngine::OfflineSamplingEngine(
    GQL::ProjectionStorage& storage,
    const SamplingConfig& config,
    const std::filesystem::path& db_folder
)
    : impl_(std::make_unique<Impl>(storage, config, db_folder))
{
}

OfflineSamplingEngine::~OfflineSamplingEngine() = default;

OfflineSamplingEngine::OfflineSamplingEngine(OfflineSamplingEngine&&) noexcept = default;
OfflineSamplingEngine& OfflineSamplingEngine::operator=(OfflineSamplingEngine&&) noexcept = default;

void OfflineSamplingEngine::set_progress_callback(ProgressCallback callback) {
    impl_->progress_callback = std::move(callback);
}

void OfflineSamplingEngine::set_progress_interval(uint64_t batches) {
    impl_->progress_interval = batches > 0 ? batches : 1;
}

void OfflineSamplingEngine::set_use_optimized_sampling(bool enabled) {
    impl_->use_optimized_sampling = enabled;
}

const SamplingConfig& OfflineSamplingEngine::get_config() const {
    return impl_->config;
}

SamplingResult OfflineSamplingEngine::run(const std::filesystem::path& db_folder) {
    impl_->cancel_requested.store(false);
    return impl_->do_run(db_folder);
}

void OfflineSamplingEngine::request_cancel() {
    impl_->cancel_requested.store(true);
}

bool OfflineSamplingEngine::is_cancelled() const {
    return impl_->cancel_requested.load();
}

std::string OfflineSamplingEngine::validate() {
    return impl_->do_validate();
}

uint64_t OfflineSamplingEngine::estimate_total_samples() const {
    return impl_->do_estimate_total_samples();
}

uint64_t OfflineSamplingEngine::estimate_storage_bytes() const {
    return impl_->do_estimate_storage_bytes();
}

} // namespace mdb::gnn
