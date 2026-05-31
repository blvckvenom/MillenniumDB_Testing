#include "gnn/sampling/sample_fingerprint.h"

#include <algorithm>
#include <cstddef>
#include <vector>

namespace mdb::gnn {

uint64_t compute_batch_content_hash(const GraphSample& s) {
    uint64_t h = kSampleFpFnvOffset;
    h = sample_fp_fold_u64(h, s.batch_id);
    h = sample_fp_fold_u64(h, static_cast<uint64_t>(s.split));

    // Per-batch structural shape: layer count + per-layer node count.
    h = sample_fp_fold_u64(h, static_cast<uint64_t>(s.nodes_per_layer.size()));
    for (const auto& layer : s.nodes_per_layer) {
        h = sample_fp_fold_u64(h, static_cast<uint64_t>(layer.size()));
    }

    // Dominant content signal: the SORTED set of unique node ids. Sorting makes
    // the hash independent of all_unique_nodes' first-appearance ordering.
    {
        std::vector<uint64_t> ids;
        ids.reserve(s.all_unique_nodes.size());
        for (const auto& oid : s.all_unique_nodes) {
            ids.push_back(oid.id);
        }
        std::sort(ids.begin(), ids.end());
        h = sample_fp_fold_u64(h, static_cast<uint64_t>(ids.size()));
        for (uint64_t id : ids) {
            h = sample_fp_fold_u64(h, id);
        }
    }

    // Edge connectivity: commutative XOR fold of per-edge global-endpoint
    // hashes. Global (src,dst) node ids are reconstructed from layer-local
    // indices, so the result does not depend on local index ordering. Captures
    // edge direction/multiplicity — the residual orientation signal beyond the
    // node set. Bounds-guarded: a malformed index is skipped, never UB.
    uint64_t edge_acc = 0;
    for (size_t k = 0; k < s.edges_per_layer.size(); ++k) {
        if (k + 1 >= s.nodes_per_layer.size()) break;
        const auto& e = s.edges_per_layer[k];
        const auto& src_layer = s.nodes_per_layer[k + 1];
        const auto& dst_layer = s.nodes_per_layer[k];
        const size_t m = e.src_indices.size();
        if (e.dst_indices.size() != m) continue;  // inconsistent layer — skip
        for (size_t i = 0; i < m; ++i) {
            const int32_t si = e.src_indices[i];
            const int32_t di = e.dst_indices[i];
            if (si < 0 || di < 0) continue;
            if (static_cast<size_t>(si) >= src_layer.size()) continue;
            if (static_cast<size_t>(di) >= dst_layer.size()) continue;
            uint64_t eh = kSampleFpFnvOffset;
            eh = sample_fp_fold_u64(eh, static_cast<uint64_t>(k));
            eh = sample_fp_fold_u64(eh, src_layer[si].id);
            eh = sample_fp_fold_u64(eh, dst_layer[di].id);
            edge_acc ^= eh;
        }
    }
    h = sample_fp_fold_u64(h, edge_acc);

    return h;  // raw — 0-sentinel applied only to the persisted sample fp.
}

uint64_t mix_feature_store_fingerprint(uint64_t sample_content_fp,
                                       const std::string& feature_name,
                                       uint64_t feature_dim,
                                       uint8_t dtype) {
    if (sample_content_fp == 0) {
        return 0;  // UNKNOWN/legacy propagates → caller recomputes
    }
    uint64_t h = kSampleFpFnvOffset;
    h = sample_fp_fold_u64(h, sample_content_fp);
    h = sample_fp_fold_bytes(h, feature_name.data(), feature_name.size());
    h = sample_fp_fold_u64(h, feature_dim);
    h = sample_fp_fold_u64(h, static_cast<uint64_t>(dtype));
    return h == 0 ? 1 : h;  // reserve 0 for UNKNOWN/absent
}

} // namespace mdb::gnn
