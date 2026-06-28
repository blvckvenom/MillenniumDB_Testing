#include "gnn/sampling/offline_sampling_engine.h"

#include <atomic>
#include <chrono>
#include <cstdlib>
#include <exception>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <mutex>
#include <sstream>
#include <stdexcept>
#include <string>
#include <system_error>
#include <thread>

#include "gnn/common/process_memory.h"
#include "gnn/projection/gnn_meta.h"
#include "gnn/projection/symmetric_snapshot_builder.h"
#include "gnn/projection/pinned_topology_view.h"
#include "gnn/projection/topology_accessor.h"
#include "gnn/sampling/basic_khop_sampler.h"
#include "gnn/sampling/gpu_khop_sampler.h"
#include "gnn/sampling/node_counts_io.h"
#include "gnn/sampling/sample_fingerprint.h"
#include "gnn/sampling/sample_storage.h"
#include "gnn/sampling/sampling_backend_plan.h"
#include "gnn/sampling/seed_selector.h"
#include "gnn/storage/row_mapping.h"
#include "graph_models/gql/projection/topology_snapshot_reader.h"
#include "gpu/gpu_device.h"
#include "graph_models/gql/projection/projection_storage.h"
#include "query/query_context.h"

namespace mdb::gnn {

// =============================================================================
// Implementation
// =============================================================================

#ifdef GNN_CUDA_ENABLED
// AUTO eligibility for the lean tiled-symmetric GPU path. True when the user did
// not force CPU, the four-level store is in play, the orientation resolves to a
// symmetric (UNDIRECTED) sample, a baked NARROW topology_sym.csr exists, and a
// GPU is present. When true the engine builds the sampler with
// lean_symmetric_gpu set, so the four-level L1/L2 build (the papers100M OOM) is
// skipped and the symmetric slice is pinned tiled. Pure read-only probe (opens
// the sidecar header + queries the device); no ordering dependency on storage
// state established later in do_run.
bool lean_symmetric_gpu_eligible(GQL::ProjectionStorage& storage,
                                 const SamplingConfig& config) {
    if (config.sampling_backend == SamplingBackendChoice::FORCE_CPU) return false;
    if (const char* e = std::getenv("MDB_GNN_SAMPLING_BACKEND")) {
        if (std::string(e) == "cpu") return false;
    }
    if (!config.use_four_level_topology_store) return false;
    if (!config.symmetric_resolved_on(config.orientation)) return false;
    auto sym = GQL::Projection::TopologySnapshotReader::open_symmetric(
        storage.get_projection_dir());
    if (!sym.has_data() || sym.id_width() != 4) return false;
    const mdb::gpu::SystemResources res = mdb::gpu::detect_resources();
    return res.has_gpu;
}
#endif

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
            SamplingConfig cfg = config;
#ifdef GNN_CUDA_ENABLED
            if (lean_symmetric_gpu_eligible(storage, config)) {
                cfg.lean_symmetric_gpu = true;
                std::cerr << "[OfflineSamplingEngine] lean tiled-symmetric GPU "
                             "path eligible: skipping the four-level L1/L2 build "
                             "(pinning the symmetric slice tiled)\n";
            }
#endif
            khop_sampler = std::make_unique<BasicKHopSampler>(storage, cfg);
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

        // Coarse do_run phase timers (sample-loop rank-1 follow-up): the
        // direct sample-loop measurement showed the parallel batch loop is
        // only ~27s, leaving ~188s of the ~382s total unaccounted between
        // build() and the loop. These spans pin it: init (incl. the
        // four-level build), seed/batch prep, and worker-pool setup.
        auto phase_ms_ = [](std::chrono::steady_clock::time_point a,
                             std::chrono::steady_clock::time_point b) -> long long {
            return std::chrono::duration_cast<std::chrono::milliseconds>(b - a).count();
        };
        long long init_ms_ = 0, seedprep_ms_ = 0, wsetup_ms_ = 0;

        try {
            // Initialize components
            const auto t_init0_ = std::chrono::steady_clock::now();
            init_components();
            init_ms_ = phase_ms_(t_init0_, std::chrono::steady_clock::now());

            // Check if storage already exists
            if (SampleStorage::exists(db_folder, config.sample_name)) {
                result.error_message = "Sample storage already exists: " + config.sample_name;
                return result;
            }

            // Create storage. MDB_GNN_SHARD_WRITE=1 enables the sharded
            // (lock-free) parallel write: each worker writes its own shard file,
            // merged in batch_id order at the end — removing the single write
            // mutex that dominates the parallel sample loop (inlock/wall ~0.96).
            // It requires the dense (RowMapping) path + parallel workers; the
            // final batches.dat/.idx/frequency.dat are byte-identical to the
            // DENSE single-writer path and the catalog statistics match (the
            // catalog.dat file itself carries a write timestamp). Falls back to
            // the legacy sparse single writer otherwise (default OFF); that mode
            // writes a v1 frequency.dat — a different on-disk layout than the
            // dense v2, but one the frequency consumer normalizes identically.
            const bool want_shard_write = [&]() {
                const char* e = std::getenv("MDB_GNN_SHARD_WRITE");
                return e != nullptr && std::string(e) == "1"
                    && config.num_workers >= 1 && row_mapping != nullptr;
            }();
            SampleStorage sample_storage = want_shard_write
                ? SampleStorage::create(db_folder, config, *row_mapping,
                      khop_sampler->get_topology().get_node_count())
                : SampleStorage::create(db_folder, config);

            // Get seed split
            const auto t_seed0_ = std::chrono::steady_clock::now();
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
            seedprep_ms_ = phase_ms_(t_seed0_, std::chrono::steady_clock::now());

            // -----------------------------------------------------------------
            // Dynamic sampling-backend decision.
            //
            // Decide, from the detected hardware resources + the in-RAM compact
            // CSR sizes, whether the GPU neighbor-fetch path runs (the kernel
            // reads the pinned uint32 CSR via UVA, or a VRAM copy) vs the proven
            // CPU out-of-core sampler. The decision is logged and exposed via the
            // result for the procedure yields. env MDB_GNN_SAMPLING_BACKEND=
            // {auto,cpu,gpu} overrides SamplingConfig.sampling_backend (the
            // MDB_GNN_* convention, like MDB_FORCE_CPU_SORT for the sort planner).
            //
            // When the plan chooses a GPU backend AND a GPU is present AND the
            // build has CUDA, the global topology CSR is pinned and the per-batch
            // loop dispatches to the GPU kernel below; otherwise the CPU path
            // runs unchanged (byte-identical output preserved).
            bool                use_gpu_sampling = false;
            SamplingBackendPlan active_gpu_plan;
            {
                SamplingBackendChoice backend_choice = config.sampling_backend;
                if (const char* env = std::getenv("MDB_GNN_SAMPLING_BACKEND")) {
                    const std::string e(env);
                    if (e == "cpu")       backend_choice = SamplingBackendChoice::FORCE_CPU;
                    else if (e == "gpu")  backend_choice = SamplingBackendChoice::FORCE_GPU;
                    else if (e == "auto") backend_choice = SamplingBackendChoice::AUTO;
                }
                mdb::gpu::SystemResources res = mdb::gpu::detect_resources();
                const TopologyAccessor& topo = khop_sampler->get_topology();

                // Symmetric headroom: when resolved-on, the directional
                // four-level tiers (L1+L2 fwd+rev) are SUPERSEDED by the single
                // pinned undirected slice — the GPU kernel reads the slice, not
                // the tiers. So their RAM is freeable headroom for the GPU/CPU
                // decision: without this the tiers (e.g. ~15 GB on papers100M)
                // shrink MemAvailable enough that the planner rejects the GPU
                // even though the slice + tiers empirically coexist within
                // budget (measured 24.8 GB peak on a 30 GB host). Add the tier
                // RAM back so the planner sizes against the true headroom.
                if (config.symmetric_resolved_on(config.orientation)) {
                    res.ram_available += topo.four_level_ram_used();
                }

                // Size the decision on the L3 GLOBAL sidecar — the substrate
                // enable_pinned_gpu_view actually pins (whole graph, narrow
                // uint32) — NOT the warm-tier L2. present mirrors the pin
                // condition exactly (has_data + id_width==4).
                auto l3_dims = [](const GQL::Projection::TopologySnapshotReader* r)
                    -> DirCsrDims {
                    if (r != nullptr && r->has_data() && r->id_width() == 4) {
                        return DirCsrDims{r->num_nodes(), r->num_edges(), true};
                    }
                    return DirCsrDims{};
                };
                // UNDIRECTED/symmetric: enable_pinned_gpu_view pins the pre-merged
                // narrow sym slice (topology_sym.csr), NOT the two directional
                // sidecars. Those are narrow only when MDB_GNN_TOPOLOGY_UINT32 was
                // set at build time; on a default (wide uint64) projection they read
                // as absent here, which would wrongly force CPU even though the
                // narrow merged slice exists and is pinnable. So when the sym slice
                // is present, size the gate on IT (one NATURAL direction == the
                // single slice that gets pinned) instead of the directional pair.
                DirCsrDims sym_dims;
                if (config.symmetric_resolved_on(config.orientation)) {
                    auto sym = GQL::Projection::TopologySnapshotReader::open_symmetric(
                        storage.get_projection_dir());
                    if (sym.has_data() && sym.id_width() == 4) {
                        sym_dims = DirCsrDims{sym.num_nodes(), sym.num_edges(), true};
                    } else {
                        // The narrow topology_sym.csr is not baked yet — the common
                        // case on a default (wide uint64) projection and on any cold
                        // first run. The slice WILL be auto-baked narrow below
                        // (build_symmetric_snapshot) BEFORE the pin, so the GPU/CPU
                        // decision must size on what that slice WILL be rather than
                        // fall through to CPU just because the file is absent — that
                        // chicken-and-egg (the plan needs the slice present to pick
                        // GPU, but the bake runs after the plan) is exactly what
                        // pinned small graphs like cora on the CPU path. The merged
                        // undirected slice holds at most fwd+rev = 2*directed entries
                        // (reciprocal pairs only shrink it via dedup), so 2*edge_count
                        // is a safe UPPER bound for VRAM sizing: the real baked slice
                        // is <= this, hence "fits VRAM" here implies it still fits
                        // after the bake. node/edge counts are O(1) catalog reads.
                        const std::uint64_t n = topo.get_node_count();
                        const std::uint64_t e = topo.get_edge_count();
                        if (n > 0) {
                            sym_dims = DirCsrDims{n, 2ull * e, true};
                        }
                    }
                }
                SamplingBackendPlan plan = sym_dims.present
                    ? plan_sampling_backend(res, EdgeOrientation::NATURAL, sym_dims,
                                            DirCsrDims{}, backend_choice)
                    : plan_sampling_backend(res, config.orientation,
                                            l3_dims(topo.l3_fwd()),
                                            l3_dims(topo.l3_rev()), backend_choice);

                // Symmetric single-slice override: when resolved-on (AUTO =>
                // UNDIRECTED) and a pinnable substrate exists, serve ONE
                // pre-merged undirected slice. The planner sized on the
                // directional sidecars; collapse to a single FORWARD_ONLY slice
                // and flag use_symmetric so the pin path takes the merged arrays
                // (never BOTH).
                const bool sym_on = config.symmetric_resolved_on(config.orientation);
                if (sym_on && plan.backend != SamplingBackend::CPU_OUT_OF_CORE) {
                    plan.use_symmetric = true;
                    plan.directions    = GpuDirections::FORWARD_ONLY;
                }
                result.symmetric_used = sym_on;

                result.sampling_backend     = to_string(plan.backend);
                result.sampling_directions  = to_string(plan.directions);
                result.sampling_plan_reason = plan.reason;
                // FORCE_GPU with no capable GPU is a hard error: the planner
                // signals it with a reason starting "ERROR:" (it cannot honor
                // FORCE_GPU without OOMing). Surface it instead of silently
                // running CPU.
                if (plan.reason.rfind("ERROR:", 0) == 0) {
                    result.error_message = plan.reason;
                    return result;
                }
#ifdef GNN_CUDA_ENABLED
                if (plan.backend != SamplingBackend::CPU_OUT_OF_CORE) {
                    auto& topo_store = khop_sampler->get_topology();
                    bool sym_freed_before_pin = false;
                    if (plan.use_symmetric) {
                        // Ensure the pre-merged symmetric slice is baked on disk
                        // (topology_sym.csr) so enable_pinned_gpu_view OPENS +
                        // pins it (mmap, zero-copy) instead of merging the two
                        // directional sidecars in RAM on every sample. Auto-bake
                        // once when absent/stale (open_symmetric runs the
                        // two-source staleness gate); the bake is the single
                        // owner of the merge, shared with gnn_build_topology_
                        // snapshot, so the auto and explicit bakes are identical.
                        // On failure we log and fall through — materialize then
                        // does the in-RAM merge (correct, just not amortized).
                        const std::filesystem::path proj_dir =
                            storage.get_projection_dir();
                        bool sym_present =
                            GQL::Projection::TopologySnapshotReader::open_symmetric(
                                proj_dir).has_data();
                        if (!sym_present) {
                            const auto tb = std::chrono::steady_clock::now();
                            try {
                                bool refused = false;
                                mdb::gnn::build_symmetric_snapshot(
                                    storage, /*verify=*/false,
                                    /*verify_sample=*/0, &refused);
                                std::cerr << "[OfflineSamplingEngine] auto-baked "
                                          << "topology_sym.csr in "
                                          << phase_ms_(tb, std::chrono::steady_clock::now())
                                          << " ms\n";
                                // The slice is now servable from disk; re-check so
                                // the free-before-pin path below can take effect.
                                sym_present =
                                    GQL::Projection::TopologySnapshotReader::
                                        open_symmetric(proj_dir).has_data();
                            } catch (const std::exception& e) {
                                std::cerr << "[OfflineSamplingEngine] auto-bake of "
                                          << "topology_sym.csr failed (" << e.what()
                                          << "); falling back to in-RAM merge\n";
                            }
                        }
                        // Free-before-pin: when the slice is servable from the
                        // baked sidecar, materialize_symmetric_arrays() copies it
                        // from that disk mmap and never reads the directional
                        // tiers, so they (~15 GB on papers100M) are dead weight.
                        // Release them BEFORE the copy+pin so the ~13 GB heap
                        // slice never coexists with them (transient host peak
                        // ~28 -> ~13 GB). Only safe when baked: the in-RAM
                        // fallback merge (taken when !sym_present) consumes the
                        // tiers and frees them after the pin instead.
                        if (sym_present) {
                            result.rss_peak_before_free_bytes = peak_rss_bytes();
                            result.directional_ram_freed_bytes =
                                topo_store
                                    .release_directional_for_baked_symmetric();
                            reset_peak_rss();
                            sym_freed_before_pin = true;
                        }
                        // enable_pinned_gpu_view materializes the merged
                        // undirected slice (opens the baked sidecar, or merges as
                        // a fallback) AND pins it.
                        const auto t0 = std::chrono::steady_clock::now();
                        topo_store.enable_pinned_gpu_view(plan);
                        result.symmetric_ms = phase_ms_(
                            t0, std::chrono::steady_clock::now());
                        result.symmetric_ram_bytes = topo_store.symmetric_ram_bytes();
                        result.symmetric_built_ok =
                            result.symmetric_ram_bytes > 0;
                    } else {
                        topo_store.enable_pinned_gpu_view(plan);
                    }
                    const PinnedTopologyView* pv = topo_store.pinned_view();
                    if (pv != nullptr && pv->is_registered()) {
                        use_gpu_sampling = true;
                        active_gpu_plan  = plan;
                        // The merged undirected slice is now materialized AND
                        // pinned; the GPU kernel serves every node from it, so
                        // the directional fwd/rev tiers (L1/L2 heap + the two L3
                        // mmap sidecars) are dead weight. Capture the high-water
                        // RSS at this tightest point (merged slice + directional
                        // tiers both resident), release the directional tiers,
                        // then reset the kernel peak so the sampling phase that
                        // follows reports its own (much lower) high-water mark.
                        // Symmetric-only: the non-symmetric pin registers the L3
                        // sidecars directly, so those must outlive the kernel.
                        if (plan.use_symmetric && !sym_freed_before_pin) {
                            // Fallback path only: the in-RAM merge consumed the
                            // directional tiers, so they can only be freed now,
                            // after the pin. (The baked path freed them before
                            // the copy+pin above, setting sym_freed_before_pin.)
                            result.rss_peak_before_free_bytes = peak_rss_bytes();
                            result.directional_ram_freed_bytes =
                                topo_store
                                    .release_directional_after_symmetric_pin();
                            reset_peak_rss();
                        }
                    }
                }
#endif
                // CPU / no-pin path: the Part C symmetric tier (single dispatch)
                // delivers the benefit, not the flat GPU arrays. Report whether
                // that tier is active; do NOT materialize the flat merge (only
                // the GPU pin consumes it) just for telemetry — on a large graph
                // it would be a multi-GB merge with no consumer here.
                if (result.symmetric_used && !use_gpu_sampling) {
                    result.symmetric_built_ok =
                        khop_sampler->get_topology().is_symmetric_topology_built();
                }
                std::cerr << "[OfflineSamplingEngine] sampling backend = "
                          << result.sampling_backend << " (" << plan.reason << ") "
                          << (use_gpu_sampling ? "[GPU kernel active]"
                                               : "[CPU path]")
                          << "\n";
            }

            // -----------------------------------------------------------------
            // Parallel-vs-legacy dispatch for the offline k-hop sampling loop.
            //
            // `num_workers == 0`: single-threaded (legacy) path. The lambda
            //   below loops sequentially through (train, val, test) batches
            //   and uses the primary sampler's persistent RNG. Outputs from
            //   earlier sessions remain bit-identical.
            //
            // `num_workers >= 1`: parallel worker-pool path. Each batch is
            //   re-seeded as `random_seed XOR batch_id` before sampling, so
            //   the output depends only on `(random_seed, batch_id)` and is
            //   identical regardless of how many workers are used. Workers
            //   borrow the primary's TopologyAccessor (the Four-Level
            //   Topology Store and its L1/L2/L3 caches are read-only
            //   post-build) and own private LeapfrogGnnSampler,
            //   SeekBasedGnnSampler, RNG, and node_access_counts vector.
            //   `sample_storage.write_sample` is serialized via a single
            //   mutex; per-batch k-hop expansion runs concurrently.
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
            // The GPU sampling path can run the PARALLEL worker pool: the per-batch
            // GPU work is microseconds (the device-resident CSR is read-only and
            // shared across threads), so the bottleneck is the single-threaded HOST
            // post-processing (edge assembly, dedup, serialize, write). Let the pool
            // overlap that tail across batches instead of forcing one thread. Only
            // safe on the RESIDENT backend (whole CSR in VRAM): the tiled fallback
            // streams COL_IDX through a single shared pinned window buffer that is
            // NOT reentrant, so it stays on the legacy serial loop. MDB_GNN_GPU_SERIAL=1
            // forces the legacy serial GPU path regardless.
            if (use_gpu_sampling) {
                const PinnedTopologyView* pv =
                    khop_sampler->get_topology().pinned_view();
                const bool resident_ok =
                    pv != nullptr && pv->is_registered()
                    && pv->fwd() != nullptr && pv->fwd()->resident;
                const char* serial_env = std::getenv("MDB_GNN_GPU_SERIAL");
                const bool want_serial =
                    serial_env != nullptr && std::string(serial_env) == "1";
                if (!resident_ok || want_serial) {
                    effective_workers = 0;  // legacy GPU-serial loop
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

                    // Reseed per batch so the legacy (numWorkers=0) path produces
                    // output that depends ONLY on (random_seed, batch_id) — exactly
                    // like the parallel (numWorkers>=1) worker path, which calls
                    // reseed_for_batch() before every sample(). Without this, W=0
                    // used a single RNG evolving across batches (order-dependent),
                    // so the sampled subgraph — and therefore the trained model —
                    // silently DIFFERED between W=0 and any W>=1 run (verified on
                    // cora: testAcc 0.8722 vs 0.8624). numWorkers must not change
                    // the sample; gated by scripts/test_plan_f_parity.sh.
                    GraphSample sample;
#ifdef GNN_CUDA_ENABLED
                    if (use_gpu_sampling) {
                        // GPU path: the kernel samples each batch's frontier in
                        // parallel over the pinned CSR. Output is NOT bit-identical
                        // to the CPU mt19937 path (Philox RNG) but honors the same
                        // uniform-without-replacement distribution. tally_nodes
                        // keeps node_counts.bin populated for the next warm start.
                        const PinnedTopologyView* pv =
                            khop_sampler->get_topology().pinned_view();
                        std::vector<int> fanouts_i(config.fanouts.begin(),
                                                   config.fanouts.end());
                        sample = sample_khop_gpu(batch_seeds, batch_id, split_type,
                                                 fanouts_i, *pv, active_gpu_plan,
                                                 config.random_seed);
                        khop_sampler->tally_nodes(sample.all_unique_nodes);
                    } else
#endif
                    {
                        khop_sampler->reseed_for_batch(batch_id);
                        // Sample k-hop neighborhood (no epoch parameter)
                        sample = khop_sampler->sample(
                            batch_seeds,
                            batch_id,
                            split_type
                        );
                    }

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
            // Parallel worker-pool path. Builds a flat work-queue across all
            // three splits (train, val, test) tagged with the destination
            // split, so a single atomic counter can dispatch any batch to any
            // worker and the `batch_id` assigned matches the legacy monotonic
            // order (train batches first, then val, then test).
            // -----------------------------------------------------------------
            auto process_batches_parallel = [&]() -> bool {
                const auto t_wsetup0_ = std::chrono::steady_clock::now();
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

                // Single SHARED atomic access-counts array for all workers,
                // replacing each worker's private N-sized `node_access_counts`
                // vector (0.83 GB each on papers100M — the allocation that
                // capped numWorkers on a 30 GB host). Pre-sized to the
                // projection node count, zero-initialized (C++17 std::atomic
                // default-init is indeterminate, so the explicit store-0 loop
                // is required). Declared on this frame, which outlives every
                // worker (all threads are joined below before the frame returns).
                const std::size_t tally_n = static_cast<std::size_t>(
                    khop_sampler->get_topology().get_node_count());
                std::vector<std::atomic<uint64_t>> shared_counts(tally_n);
                for (std::size_t i = 0; i < tally_n; ++i) {
                    shared_counts[i].store(0, std::memory_order_relaxed);
                }
                khop_sampler->set_shared_access_counts(shared_counts.data(), tally_n);
                for (auto& w : worker_samplers) {
                    w->set_shared_access_counts(shared_counts.data(), tally_n);
                }

                std::atomic<std::size_t> next_idx{0};
                std::mutex               write_mutex;
                // Sharded path: tiny lock for the periodic progress block only
                // (the byte-write + freq tally are lock-free per worker).
                std::mutex               progress_mutex;

                // Open one shard file per worker when sharded write is enabled.
                if (want_shard_write) {
                    sample_storage.begin_sharded_write(effective_workers);
                }

                // Sample-loop sub-stage instrumentation (analogous to the
                // FourLevelTopologyStore::build() rank-1 split). Three atomic
                // microsecond accumulators across all workers answer the
                // decisive question for the 215s/56% sample loop: is it
                // SERIAL-write-bound (in_lock ≈ wall) or CONCURRENT-assemble-
                // bound (expand dominates, in_lock small)? wait_us is the
                // lock-contention indicator. Logged once to stderr after join.
                std::atomic<uint64_t> expand_us{0};   // sampler->sample() (concurrent)
                std::atomic<uint64_t> wait_us{0};      // blocked acquiring write_mutex
                std::atomic<uint64_t> inlock_us{0};    // write_sample + progress (serial)
                auto dur_us_ = [](std::chrono::steady_clock::time_point a,
                                  std::chrono::steady_clock::time_point b) -> uint64_t {
                    return static_cast<uint64_t>(
                        std::chrono::duration_cast<std::chrono::microseconds>(b - a).count());
                };

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

                // Exception safety for the worker pool: an exception escaping
                // a std::thread body — or unwinding the primary past the join
                // below — calls std::terminate(). sampler->sample() can throw
                // (e.g. the max_layer_nodes cap on dense UNDIRECTED graphs).
                // Capture the first exception, request cancellation so peers
                // wind down, and rethrow ONCE after all threads join and the
                // shared tally array is detached (no use-after-free / abort).
                std::exception_ptr worker_exception;
                std::mutex         exception_mutex;

                // MDB_GNN_PARALLEL_WRITE_PREP (default OFF): do the per-batch
                // serialize() + compute_batch_content_hash() on the worker thread,
                // OFF the write_mutex. The measured bottleneck is the serial write
                // critical section (inlock/wall ≈ 0.96 at numWorkers=16); both ops
                // are pure functions of the sample, so moving them out is
                // bit-identical (batches.dat + content_fp unchanged) and shrinks the
                // serial section to just the shared disk write + freq + bookkeeping.
                const bool parallel_write_prep = []() {
                    const char* e = std::getenv("MDB_GNN_PARALLEL_WRITE_PREP");
                    return e != nullptr && std::string(e) == "1";
                }();

                auto worker_fn = [&](BasicKHopSampler* sampler,
                                     uint32_t worker_index) {
                    QueryContext::set_query_ctx(primary_ctx);
                    try {
                    while (true) {
                        if (cancel_requested.load(std::memory_order_relaxed)) {
                            return;
                        }
                        const std::size_t idx =
                            next_idx.fetch_add(1, std::memory_order_relaxed);
                        if (idx >= total_work) return;

                        const WorkItem& item = work[idx];
                        const auto t_exp0 = std::chrono::steady_clock::now();
                        GraphSample sample;
#ifdef GNN_CUDA_ENABLED
                        if (use_gpu_sampling) {
                            // GPU sample is a pure function of (random_seed,
                            // batch_id): batch_seed = random_seed ^ batch_id inside
                            // sample_khop_gpu, so NO reseed is needed and the result
                            // is invariant to worker count/order. The kernel reads
                            // the shared device-resident CSR read-only; only per-call
                            // device scratch is mutable, so concurrent workers do not
                            // race on the graph. tally_nodes routes through the shared
                            // atomic counts installed on every worker above.
                            const PinnedTopologyView* pv =
                                sampler->get_topology().pinned_view();
                            std::vector<int> fanouts_i(config.fanouts.begin(),
                                                       config.fanouts.end());
                            sample = sample_khop_gpu(
                                *item.seeds, item.batch_id, item.split,
                                fanouts_i, *pv, active_gpu_plan,
                                config.random_seed);
                            sampler->tally_nodes(sample.all_unique_nodes);
                        } else
#endif
                        {
                            sampler->reseed_for_batch(item.batch_id);
                            sample = sampler->sample(
                                *item.seeds, item.batch_id, item.split);
                        }
                        // Off-lock prep: serialize() + content-hash on the worker
                        // (concurrent, counted as expand). Pure functions of sample.
                        // Always done in the sharded path (the write is lock-free).
                        std::string prep_data;
                        uint64_t    prep_hash = 0;
                        if (parallel_write_prep || want_shard_write) {
                            std::ostringstream pbuf(std::ios::binary);
                            sample.serialize(pbuf);
                            prep_data = pbuf.str();
                            prep_hash = compute_batch_content_hash(sample);
                        }
                        const auto t_exp1 = std::chrono::steady_clock::now();
                        expand_us.fetch_add(dur_us_(t_exp0, t_exp1),
                                            std::memory_order_relaxed);

                        if (want_shard_write) {
                            // Lock-free: worker owns its shard file; the only
                            // shared state (atomic freq array) is updated inside.
                            sample_storage.shard_write(worker_index, sample,
                                                       prep_data, prep_hash);
                            std::lock_guard<std::mutex> lk(progress_mutex);
                            progress.samples_written++;
                            progress.current_batch = item.batch_id + 1;
                            progress.current_split = item.split;
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
                            continue;
                        }

                        const auto t_wait0 = std::chrono::steady_clock::now();
                        std::lock_guard<std::mutex> lk(write_mutex);
                        const auto t_lock = std::chrono::steady_clock::now();
                        wait_us.fetch_add(dur_us_(t_wait0, t_lock),
                                          std::memory_order_relaxed);
                        if (parallel_write_prep) {
                            sample_storage.write_sample_prepared(
                                sample, prep_data, prep_hash);
                        } else {
                            sample_storage.write_sample(sample);
                        }
                        progress.samples_written++;
                        progress.current_batch  = item.batch_id + 1;
                        progress.current_split  = item.split;
                        inlock_us.fetch_add(dur_us_(t_lock,
                                            std::chrono::steady_clock::now()),
                                            std::memory_order_relaxed);

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
                    } catch (...) {
                        std::lock_guard<std::mutex> lk(exception_mutex);
                        if (!worker_exception) {
                            worker_exception = std::current_exception();
                        }
                        cancel_requested.store(true);
                    }
                };

                // Spawn workers. Primary runs in-thread to keep one fewer
                // OS thread alive and to match the legacy path's behavior
                // when num_workers == 1.
                wsetup_ms_ = phase_ms_(t_wsetup0_, std::chrono::steady_clock::now());
                const auto loop_start = std::chrono::steady_clock::now();
                std::vector<std::thread> threads;
                threads.reserve(effective_workers - 1);
                try {
                    uint32_t w_index = 1;  // worker 0 == primary (runs in-thread)
                    for (auto& w : worker_samplers) {
                        threads.emplace_back(worker_fn, w.get(), w_index++);
                    }
                } catch (...) {
                    // std::thread construction can throw std::system_error
                    // under thread/resource exhaustion. Unwinding while
                    // `threads` still holds joinable threads would call
                    // std::terminate(); wind the already-spawned workers
                    // down, join them, and detach the shared tally before
                    // rethrowing into the outer do_run handler.
                    cancel_requested.store(true);
                    for (auto& t : threads) t.join();
                    khop_sampler->set_shared_access_counts(nullptr, 0);
                    for (auto& w : worker_samplers) {
                        w->set_shared_access_counts(nullptr, 0);
                    }
                    throw;
                }
                worker_fn(khop_sampler.get(), 0);
                for (auto& t : threads) t.join();

                // Sample-loop sub-stage split. loop_wall is the parallel
                // region wall-clock; the three totals sum work across all
                // workers (so each can exceed wall). Decisive reads:
                //   inlock_total ≈ loop_wall   → serial-write-bound (attack the lock)
                //   expand_total dominates     → concurrent-assemble-bound (attack assemble)
                //   wait_total high            → lock contention
                //   effParallelism = (expand+inlock)/wall ≈ workers → good scaling
                {
                    const double loop_wall_ms =
                        std::chrono::duration<double, std::milli>(
                            std::chrono::steady_clock::now() - loop_start).count();
                    const double exp_ms = expand_us.load() / 1000.0;
                    const double wait_ms = wait_us.load() / 1000.0;
                    const double lock_ms = inlock_us.load() / 1000.0;
                    const double eff_par =
                        loop_wall_ms > 0 ? (exp_ms + lock_ms) / loop_wall_ms : 0.0;
                    std::cerr << "[OfflineSamplingEngine] sample-loop split — "
                              << "workers=" << effective_workers
                              << " loopWallMs=" << static_cast<long long>(loop_wall_ms)
                              << " expandMsTotal=" << static_cast<long long>(exp_ms)
                              << " waitMsTotal=" << static_cast<long long>(wait_ms)
                              << " inlockMsTotal=" << static_cast<long long>(lock_ms)
                              << " effParallelism=" << eff_par
                              << " (inlock/wall=" << (loop_wall_ms > 0 ? lock_ms / loop_wall_ms : 0.0)
                              << ")\n";
                    // Coarse do_run phases (locates the ~188s that is neither
                    // build() nor the sample loop): init (incl. four-level
                    // build), seed/batch prep, worker-pool setup.
                    std::cerr << "[OfflineSamplingEngine] do_run phases — "
                              << "initMs=" << init_ms_
                              << " seedPrepMs=" << seedprep_ms_
                              << " workerSetupMs=" << wsetup_ms_
                              << " (initMs includes the four-level build split logged above)\n";
                }

                // Snapshot the single shared atomic tally into the primary's
                // plain vector (no per-worker merge needed: all workers wrote
                // into the same shared array). Detach primary and workers from
                // the shared array FIRST so the about-to-be-destroyed
                // `shared_counts` stack object is never referenced after this
                // frame returns.
                {
                    std::vector<uint64_t> materialized(tally_n);
                    for (std::size_t i = 0; i < tally_n; ++i) {
                        materialized[i] =
                            shared_counts[i].load(std::memory_order_relaxed);
                    }
                    khop_sampler->set_shared_access_counts(nullptr, 0);
                    for (auto& w : worker_samplers) {
                        w->set_shared_access_counts(nullptr, 0);
                    }
                    khop_sampler->adopt_counts(std::move(materialized));
                }

                // Rethrow now — threads joined, shared tally detached. The
                // outer do_run try/catch turns this into a clean error result
                // instead of std::terminate().
                if (worker_exception) {
                    std::rethrow_exception(worker_exception);
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

            // Commit the sample only when the run completed. A cancelled run
            // must not leave a self-consistent catalog on disk: abort()
            // discards the partial sample so a re-run does not fail with
            // "already exists" and downstream consumers cannot open a
            // truncated sample. The worker-exception path (rethrow above)
            // unwinds past this point; the SampleStorage destructor then
            // aborts the partial write for the same reason.
            // Expand-stage sub-cost breakdown (env MDB_GNN_EXPAND_PROFILE=1).
            // Logged once after sampling completes, covering both the legacy and
            // parallel paths. Empty (no-op) when profiling is disabled.
            {
                std::string ep = BasicKHopSampler::dump_expand_profile();
                if (!ep.empty()) {
                    std::cerr << "[OfflineSamplingEngine] " << ep << "\n";
                }
            }

            if (result.cancelled) {
                sample_storage.abort();
            } else if (sample_storage.sharded_write_active()) {
                // Sharded path: concat shards in batch_id order into the final
                // byte-identical batches.dat/.idx/frequency.dat/catalog.
                sample_storage.merge_shards();
            } else {
                sample_storage.finalize();
            }

            // Persist `<projection_dir>/node_counts.bin` so the next
            // `gnn_offline_sample` run can warm-start the Four-Level
            // Topology Store (L1 RAM hash / L2 compact uint32 CSR /
            // L3 mmap sidecar / L4 direct B+Tree) with the per-node
            // access frequencies measured during this sample. Guarded by
            // `useFourLevelTopologyStore`: when the user opted out of the
            // tiered cache there is no consumer for the file, so skip the
            // I/O. Failures inside `node_counts_io::persist` log to stderr
            // but never throw — the sample itself already succeeded;
            // persistence is an optimization for future runs.
            if (config.use_four_level_topology_store) {
                auto proj_dir =
                    std::filesystem::path(storage.get_projection_dir());
                node_counts_io::persist(proj_dir,
                                        khop_sampler->node_access_counts(),
                                        config.orientation);
            }

            // Calculate final timing
            auto end_time = std::chrono::steady_clock::now();
            result.total_seconds = std::chrono::duration<double>(end_time - start_time).count();
            result.total_samples = batch_id;
            result.catalog = sample_storage.get_catalog();

            // Pull cold-start topology profiler telemetry out of the
            // sampler so the procedure can yield it. The cold-start
            // profiler (a degree-weighted random-walk pass using a
            // Vose-alias table) runs before the Four-Level Topology
            // Store's build() when node_counts.bin is absent. Fields are
            // populated even when the profiler chose not to run
            // (triggered=false) so bench harnesses can distinguish
            // "warm-start ready" from "profiler ran and succeeded".
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

            // RAM telemetry: the sampling-phase high-water mark (the peak was
            // reset right after the directional free on the symmetric GPU path,
            // so this reflects the post-free working set) + the current resident
            // size at end. Compared against rss_peak_before_free_bytes these show
            // the peak-RAM relief from releasing the directional tiers.
            result.rss_peak_after_free_bytes = peak_rss_bytes();
            result.rss_current_end_bytes     = current_rss_bytes();

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

void OfflineSamplingEngine::set_use_optimized_sampling(bool /*enabled*/) {
    // Inert: the SortedBatchSampler path was never wired into do_run; the
    // sampler dispatch is decided by BasicKHopSampler's strategy selection.
    // Kept as a no-op for API compatibility.
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
