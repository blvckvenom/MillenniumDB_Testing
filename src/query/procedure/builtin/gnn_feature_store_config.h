#pragma once
// Shared parser: DictOptions -> FourLevelStore::Config for the four-level feature
// store build. Used by BOTH gnn_build_feature_store (standalone verb) and
// gnn_offline_sample (the buildFeatureStore tail), so the two paths honor an
// identical option set with a single source of truth. Unknown keys are ignored
// (each caller validates its own key set); the deprecated packed-full option is
// hard-refused here so it can never be requested from either path.

#include <algorithm>
#include <cstdlib>
#include <stdexcept>
#include <string>

#include <iostream>

#include "gnn/sampling/minhash_reorderer.h"
#include "gnn/storage/four_level_store.h"
#include "gpu/gpu_device.h"
#include "query/procedure/builtin/gnn_procedure_utils.h"

namespace GQL {
namespace Procedures {

// Build a FourLevelStore::Config from optional build options. @p opts may be
// nullptr (no options map present) — defaults plus the auto-resources pass apply.
// Behavior is identical to the inline parser that previously lived in
// gnn_build_feature_store_procedure (moved here verbatim).
inline mdb::gnn::FourLevelStore::Config build_feature_store_config(const DictOptions* opts) {
    using mdb::gnn::FourLevelStore;
    using mdb::gnn::MinHashReorderer;

    FourLevelStore::Config config;
    bool gpu_budget_explicit = false;
    bool cpu_budget_explicit = false;

    if (opts != nullptr) {
        auto get_int_opt = [&](const char* key, const char* alias) {
            auto v = opts->get_int(key);
            return v ? v : opts->get_int(alias);
        };
        auto get_bool_opt = [&](const char* key, const char* alias) {
            auto v = opts->get_bool(key);
            return v ? v : opts->get_bool(alias);
        };

        if (auto v = get_int_opt("gpu_budget_mb", "gpuBudgetMb")) {
            if (*v < 0) throw std::runtime_error("gpu_budget_mb must be non-negative, got: " + std::to_string(*v));
            config.gpu.budget_bytes = static_cast<size_t>(*v) * 1024ULL * 1024ULL;
            gpu_budget_explicit = true;
        }
        if (auto v = get_int_opt("cpu_budget_mb", "cpuBudgetMb")) {
            if (*v <= 0) throw std::runtime_error("cpu_budget_mb must be positive, got: " + std::to_string(*v));
            config.cpu.budget_bytes = static_cast<size_t>(*v) * 1024ULL * 1024ULL;
            cpu_budget_explicit = true;
        }
        if (auto v = opts->get_bool("reorder")) {
            config.reorder = *v;
        }
        if (auto v = opts->get_bool("force")) {
            config.force = *v;
        }
        if (auto v = get_bool_opt("force_caches", "forceCaches"))           config.force_caches = *v;
        if (auto v = get_bool_opt("force_reorder", "forceReorder"))         config.force_reorder = *v;
        if (auto v = get_bool_opt("force_packed_slim", "forcePackedSlim"))  config.force_packed_slim = *v;
        if (auto v = get_bool_opt("force_meta", "forceMeta"))               config.force_meta = *v;
        if (auto v = opts->get_bool("buildAddrTables")) config.build_addr_tables = *v;
        if (auto v = opts->get_bool("bakeBlocks")) config.bake_blocks = *v;
        // packFullFeatures is DEPRECATED AND REMOVED (infeasible: ~18x the feature
        // matrix, no cross-batch dedup). Hard-refused from either build path.
        if (opts->get_bool("packFullFeatures")) {
            throw std::runtime_error(
                "packFullFeatures is deprecated and removed: the packed-full feature "
                "store is infeasible (~18x the feature matrix) and is no longer "
                "supported; use the default four-level feature store.");
        }
        if (auto v = opts->get_bool("writeConsolidatedSlim")) config.write_consolidated_slim = *v;
        if (auto v = opts->get_bool("cleanupIntermediate")) {
            config.cleanup_materialize_scratch = *v;
        }
        if (auto v = opts->get_string("strategy")) {
            std::string s = *v;
            std::transform(s.begin(), s.end(), s.begin(), ::toupper);
            if (s == "SEGMENTED") {
                config.minhash.strategy = MinHashReorderer::Strategy::SEGMENTED;
            } else if (s == "MULTIPASS_BOUNDED") {
                config.minhash.strategy = MinHashReorderer::Strategy::MULTIPASS_BOUNDED;
            } else {
                throw std::runtime_error(
                    "Invalid strategy: '" + *v + "'. Must be 'SEGMENTED' or 'MULTIPASS_BOUNDED'.");
            }
        }
        if (auto v = opts->get_int("numHashes")) {
            if (*v <= 0) throw std::runtime_error("numHashes must be positive, got: " + std::to_string(*v));
            config.minhash.num_hashes = static_cast<uint32_t>(*v);
        }
        if (auto v = opts->get_int("segmentSize")) {
            if (*v <= 0) throw std::runtime_error("segmentSize must be positive, got: " + std::to_string(*v));
            config.minhash.segment_size = static_cast<uint32_t>(*v);
        }
        if (auto v = opts->get_int("diskBudgetMb")) {
            if (*v < 0) throw std::runtime_error("diskBudgetMb must be non-negative, got: " + std::to_string(*v));
            config.disk_budget_bytes = static_cast<size_t>(*v) * 1024ULL * 1024ULL;
        }
    }

    // Dynamic per-hardware budgets (opt-in via env MDB_GNN_AUTO_RESOURCES=1,
    // default OFF). Budgets choose cache TIER placement only (which nodes live in
    // L1 GPU / L2 CPU vs L3/L4 disk), never the gathered feature values, so the
    // trained result is invariant to the budget. Explicit budgets always win.
    if (const char* e = std::getenv("MDB_GNN_AUTO_RESOURCES"); e && std::string(e) == "1") {
        auto res = mdb::gpu::detect_resources();
        if (!gpu_budget_explicit && res.has_gpu) {
            config.gpu.budget_bytes = res.gpu.free_vram / 4;
        }
        if (!cpu_budget_explicit && res.ram_available > 0) {
            config.cpu.budget_bytes = res.ram_available / 4;
        }
        std::cerr << "[feature_store_config] MDB_GNN_AUTO_RESOURCES: "
                  << "gpu_budget=" << (config.gpu.budget_bytes >> 20) << "MB "
                  << "cpu_budget=" << (config.cpu.budget_bytes >> 20) << "MB "
                  << "(has_gpu=" << (res.has_gpu ? 1 : 0)
                  << " free_vram=" << (res.gpu.free_vram >> 20) << "MB"
                  << " ram_avail=" << (res.ram_available >> 20) << "MB"
                  << " gpu_explicit=" << (gpu_budget_explicit ? 1 : 0)
                  << " cpu_explicit=" << (cpu_budget_explicit ? 1 : 0) << ")\n";
    }

    return config;
}

}  // namespace Procedures
}  // namespace GQL
