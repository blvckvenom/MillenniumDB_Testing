#pragma once

#include <cstddef>
#include <cstdint>
#include <string>

#include "gnn/sampling/graph_sample.h"

namespace mdb::gnn {

// =============================================================================
// Sample content fingerprint (STEP 8, approach B)
// =============================================================================
//
// A layout-independent content fingerprint used to decide whether a previously
// built feature store still matches the sample it was built from — enabling
// reuse-on-match / recompute-on-mismatch instead of the legacy hard-throw, with
// no risk of silent stale reuse.
//
// The fingerprint hashes the *actual* per-batch node set + edge connectivity,
// NOT aggregate scalar counts. Any content change — orientation, strategy,
// split ratios, seed_query, label_property, fanout, or any future knob — alters
// at least one batch's node set or edges, so the fingerprint changes. This is
// robust by construction (no enumeration of determinants required).
//
// It is *layout-independent* and numWorkers-safe: per-batch the node ids are
// sorted (independent of first-appearance order) and edge-endpoint hashes are
// XOR-folded (commutative within a batch); the cross-batch combiner (in
// SampleStorage) XOR-folds per-batch hashes keyed by batch_id (commutative
// across batches), so the value is invariant to worker completion order. It
// deliberately excludes created_at and the raw on-disk byte layout.

// FNV-64 constants — identical to compute_meta_sha_head in four_level_store.cc.
inline constexpr uint64_t kSampleFpFnvOffset = 0xCBF29CE484222325ULL;
inline constexpr uint64_t kSampleFpFnvPrime  = 0x00000100000001B3ULL;

// FNV-1a fold of the 8 little-endian bytes of `v` into running hash `h`.
inline uint64_t sample_fp_fold_u64(uint64_t h, uint64_t v) {
    for (int i = 0; i < 8; ++i) {
        h ^= static_cast<uint64_t>(v & 0xFFu);
        h *= kSampleFpFnvPrime;
        v >>= 8;
    }
    return h;
}

// FNV-1a fold of a raw byte buffer (e.g. a feature name) into running hash `h`.
inline uint64_t sample_fp_fold_bytes(uint64_t h, const void* data, size_t n) {
    const auto* p = static_cast<const unsigned char*>(data);
    for (size_t i = 0; i < n; ++i) {
        h ^= static_cast<uint64_t>(p[i]);
        h *= kSampleFpFnvPrime;
    }
    return h;
}

/**
 * @brief Layout-independent content hash of ONE mini-batch.
 *
 * Folds, in order: batch_id, split, the per-layer node-count shape, the SORTED
 * set of unique node ids, and a commutative XOR fold of per-edge global
 * (src,dst) endpoint hashes (reconstructed from layer-local indices, so the
 * result does not depend on local index ordering).
 *
 * The returned value MAY legitimately be 0. Callers that XOR-fold per batch
 * MUST keep it raw — the 0 = "UNKNOWN/absent" sentinel is applied only to the
 * final persisted sample fingerprint, not to per-batch values.
 */
uint64_t compute_batch_content_hash(const GraphSample& sample);

/**
 * @brief Derive the feature store's reuse key from the sample's content
 *        fingerprint plus the feature-store identity (feature name + dim +
 *        dtype), so a feature change forces recompute even at a constant sample.
 *
 * @return 0 if @p sample_content_fp == 0 (UNKNOWN propagates → caller must
 *         recompute). Otherwise the mixed hash with the 0→1 sentinel applied so
 *         a real key is never 0 (0 stays reserved for UNKNOWN/absent).
 */
uint64_t mix_feature_store_fingerprint(uint64_t sample_content_fp,
                                       const std::string& feature_name,
                                       uint64_t feature_dim,
                                       uint8_t dtype);

} // namespace mdb::gnn
