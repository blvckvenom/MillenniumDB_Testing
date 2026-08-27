// gpu_khop_sampler.cu
//
// CUDA kernel for k-hop neighbor sampling. One thread per frontier node samples
// up to `fanout` neighbors WITHOUT REPLACEMENT from the pinned adjacency CSR,
// using a counter-based RNG (Philox 4x32-10) so the sample of a node depends only
// on (batch_seed, node, layer) — invariant to thread/block scheduling and worker
// count. Mirrors the CPU sampler's distribution (k = min(fanout, deg); the
// k==deg short-circuit emits every neighbor with no draw) but NOT its exact picks
// (Philox != mt19937_64).
//
// v1 uses ONE THREAD PER NODE running a scalar reservoir (Vitter Algorithm R).
// The warp-cooperative coalesced variant is a perf refinement deferred to a later
// phase; thread-per-node is correct and statistically validated here.

#include "gnn/sampling/gpu_khop_sampler.h"

#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include <cuda_runtime.h>

#include "gnn/core/cuda_context.h"  // CUDA_CHECK / CudaException
#include "gnn/projection/pinned_topology_view.h"
#include "gnn/sampling/sampling_backend_plan.h"
#include "graph_models/object_id.h"

namespace mdb::gnn {

namespace {

// ---------------------------------------------------------------------------
// Philox 4x32-10 counter-based RNG (Salmon et al., "Parallel Random Numbers:
// As Easy as 1, 2, 3", SC 2011, sec. 4.3 — the Philox-4x32 multipliers
// 0xD2511F53 / 0xCD9E8D57 come from there). Pure function of
// (key, counter): no per-thread state, so the draw for a given (node, layer,
// stream-index) is identical regardless of how threads are scheduled.
// ---------------------------------------------------------------------------
__device__ __forceinline__ uint32_t mulhilo32_(uint32_t a, uint32_t b,
                                                uint32_t& hi) {
    uint64_t product = static_cast<uint64_t>(a) * static_cast<uint64_t>(b);
    hi = static_cast<uint32_t>(product >> 32);
    return static_cast<uint32_t>(product);
}

__device__ __forceinline__ void philox_round_(uint32_t ctr[4],
                                              const uint32_t key[2]) {
    uint32_t hi0, hi1;
    uint32_t lo0 = mulhilo32_(0xD2511F53u, ctr[0], hi0);
    uint32_t lo1 = mulhilo32_(0xCD9E8D57u, ctr[2], hi1);
    uint32_t n0 = hi1 ^ ctr[1] ^ key[0];
    uint32_t n1 = lo1;
    uint32_t n2 = hi0 ^ ctr[3] ^ key[1];
    uint32_t n3 = lo0;
    ctr[0] = n0;
    ctr[1] = n1;
    ctr[2] = n2;
    ctr[3] = n3;
}

// Output word 0 of Philox for counter (node, layer, draw, 0) and key (batch_seed).
__device__ __forceinline__ uint32_t philox_u32_(uint64_t batch_seed,
                                               uint32_t node, uint32_t layer,
                                               uint32_t draw) {
    uint32_t ctr[4] = {node, layer, draw, 0u};
    uint32_t key[2] = {static_cast<uint32_t>(batch_seed),
                       static_cast<uint32_t>(batch_seed >> 32)};
#pragma unroll
    for (int i = 0; i < 10; ++i) {
        philox_round_(ctr, key);
        key[0] += 0x9E3779B9u;  // Weyl key increments: 32-bit truncations of the
        key[1] += 0xBB67AE85u;  // golden-ratio and sqrt(3)-1 constants (Salmon et
                                // al., "Parallel Random Numbers: As Easy as
                                // 1, 2, 3", SC 2011, sec. 4.2).
    }
    return ctr[0];
}

// Uniform in [0, m) via the multiply-shift reduction (Lemire, "Fast Random
// Integer Generation in an Interval", ACM TOMACS 2019). We omit the paper's
// rejection step, so a residual bias of at most m/2^32 per draw remains —
// negligible against fanout-sized m. m fits uint32.
__device__ __forceinline__ uint32_t uniform_below_(uint32_t r, uint32_t m) {
    return static_cast<uint32_t>((static_cast<uint64_t>(r) * static_cast<uint64_t>(m)) >> 32);
}

// One direction's CSR slice. row_ptr==nullptr => direction absent (single-dir).
struct CsrRange {
    const uint64_t* row_ptr;
    const uint32_t* col_idx;
    uint32_t        n_rows;
    // Global edge offset of the first edge present in `col_idx`. 0 when col_idx
    // spans the whole CSR (non-tiled). In tiled mode col_idx points at a
    // node-aligned window holding col_idx[row_ptr[u_lo] .. row_ptr[u_hi]); then
    // col_base_offset = row_ptr[u_lo] and the kernel subtracts it from row_ptr[v]
    // to land at the local index. The host only feeds this kernel frontier nodes
    // whose row lies in [u_lo, u_hi), so a node's whole neighbor list is in-window
    // (no straddling).
    uint64_t        col_base_offset = 0;
};

__device__ __forceinline__ uint64_t deg_of_(const CsrRange& c, uint32_t v) {
    if (c.row_ptr == nullptr || v >= c.n_rows) return 0;
    return c.row_ptr[v + 1] - c.row_ptr[v];
}

// One thread per frontier node. Samples up to `fanout` neighbors (across the
// concatenation of directions a + b) without replacement, writing into
// out_nbrs[i*fanout .. ] and out_counts[i].
__global__ void sample_layer_kernel(CsrRange a, CsrRange b,
                                    const uint32_t* __restrict__ frontier,
                                    uint32_t F, int fanout, uint64_t batch_seed,
                                    int layer,
                                    uint32_t* __restrict__ out_nbrs,
                                    uint32_t* __restrict__ out_counts) {
    const uint32_t stride = gridDim.x * blockDim.x;
    for (uint32_t i = blockIdx.x * blockDim.x + threadIdx.x; i < F; i += stride) {
        const uint32_t v = frontier[i];
        const uint64_t deg_a = deg_of_(a, v);
        const uint64_t deg_b = deg_of_(b, v);
        const uint64_t deg = deg_a + deg_b;
        uint32_t* out = out_nbrs + static_cast<size_t>(i) * fanout;

        if (deg == 0) {
            out_counts[i] = 0;
            continue;
        }

        // LOCAL col_idx offsets: subtract the window base (0 in the non-tiled
        // case) so the three reads below index a windowed col_idx buffer.
        const uint64_t base_a = (deg_a > 0) ? a.row_ptr[v] - a.col_base_offset : 0;
        const uint64_t base_b = (deg_b > 0) ? b.row_ptr[v] - b.col_base_offset : 0;
        const uint32_t fo = static_cast<uint32_t>(fanout);

        // k == deg short-circuit: emit every neighbor in CSR order, no draws.
        if (deg <= fo) {
            for (uint64_t j = 0; j < deg; ++j) {
                out[j] = (j < deg_a) ? a.col_idx[base_a + j]
                                     : b.col_idx[base_b + (j - deg_a)];
            }
            out_counts[i] = static_cast<uint32_t>(deg);
            continue;
        }

        // Reservoir (Vitter Algorithm R): out[0..fo) IS the reservoir.
        for (uint32_t j = 0; j < fo; ++j) {
            out[j] = (j < deg_a) ? a.col_idx[base_a + j]
                                 : b.col_idx[base_b + (j - deg_a)];
        }
        for (uint64_t j = fo; j < deg; ++j) {
            const uint32_t r =
                philox_u32_(batch_seed, v, static_cast<uint32_t>(layer),
                            static_cast<uint32_t>(j));
            const uint32_t slot = uniform_below_(r, static_cast<uint32_t>(j + 1));
            if (slot < fo) {
                out[slot] = (j < deg_a) ? a.col_idx[base_a + j]
                                        : b.col_idx[base_b + (j - deg_a)];
            }
        }
        out_counts[i] = fo;
    }
}

constexpr int kBlockSize = 256;

int grid_for_(uint32_t F) {
    long long needed = (static_cast<long long>(F) + kBlockSize - 1) / kBlockSize;
    if (needed < 1) needed = 1;
    if (needed > 65535) needed = 65535;  // grid-stride loop covers the rest
    return static_cast<int>(needed);
}

// Launch the sampling kernel over `frontier` (host dense ids) against already-on-
// device CSR ranges (a + optional b). Uploads the frontier, allocates the output,
// launches, copies back. Returns per-frontier-node the sampled neighbor ids.
std::vector<std::vector<uint32_t>> sample_layer_on_device_(
    CsrRange a, CsrRange b, const std::vector<uint32_t>& frontier, int fanout,
    uint64_t batch_seed, int layer) {
    const uint32_t F = static_cast<uint32_t>(frontier.size());
    std::vector<std::vector<uint32_t>> result(F);
    if (F == 0 || fanout <= 0) return result;

    const size_t out_len = static_cast<size_t>(F) * static_cast<size_t>(fanout);
    uint32_t* d_frontier = nullptr;
    uint32_t* d_out = nullptr;
    uint32_t* d_counts = nullptr;
    CUDA_CHECK(cudaMalloc(&d_frontier, F * sizeof(uint32_t)));
    CUDA_CHECK(cudaMalloc(&d_out, out_len * sizeof(uint32_t)));
    CUDA_CHECK(cudaMalloc(&d_counts, F * sizeof(uint32_t)));
    CUDA_CHECK(cudaMemcpy(d_frontier, frontier.data(), F * sizeof(uint32_t),
                          cudaMemcpyHostToDevice));

    sample_layer_kernel<<<grid_for_(F), kBlockSize>>>(
        a, b, d_frontier, F, fanout, batch_seed, layer, d_out, d_counts);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    std::vector<uint32_t> h_out(out_len);
    std::vector<uint32_t> h_counts(F);
    CUDA_CHECK(cudaMemcpy(h_out.data(), d_out, out_len * sizeof(uint32_t),
                          cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_counts.data(), d_counts, F * sizeof(uint32_t),
                          cudaMemcpyDeviceToHost));

    cudaFree(d_frontier);
    cudaFree(d_out);
    cudaFree(d_counts);

    for (uint32_t i = 0; i < F; ++i) {
        const uint32_t c = h_counts[i];
        result[i].assign(h_out.begin() + static_cast<size_t>(i) * fanout,
                         h_out.begin() + static_cast<size_t>(i) * fanout + c);
    }
    return result;
}

// Replicates BasicKHopSampler::Impl::build_edges: builds edges_per_layer with
// local indices. sampled_edges[k] maps dst_node -> [(src_neighbor, edge_id)] where
// dst is in layer k and src in layer k+1. Messages flow src(k+1) -> dst(k).
void build_edges_(
    GraphSample& sample,
    const std::vector<std::unordered_map<
        uint64_t, std::vector<std::pair<ObjectId, ObjectId>>>>& sampled_edges) {
    const size_t num_layers = sample.nodes_per_layer.size();
    if (num_layers < 2) return;
    sample.edges_per_layer.resize(num_layers - 1);

    std::vector<std::unordered_map<uint64_t, int32_t>> layer_mappings(num_layers);
    for (size_t layer = 0; layer < num_layers; ++layer) {
        const auto& nodes = sample.nodes_per_layer[layer];
        for (size_t i = 0; i < nodes.size(); ++i) {
            layer_mappings[layer][nodes[i].id] = static_cast<int32_t>(i);
        }
    }

    for (size_t k = 0; k < num_layers - 1; ++k) {
        auto& edges = sample.edges_per_layer[k];
        for (const auto& kv : sampled_edges[k]) {
            auto dst_it = layer_mappings[k].find(kv.first);
            if (dst_it == layer_mappings[k].end()) continue;
            const int32_t dst_idx = dst_it->second;
            for (const auto& src_edge : kv.second) {
                auto src_it = layer_mappings[k + 1].find(src_edge.first.id);
                if (src_it == layer_mappings[k + 1].end()) continue;
                edges.src_indices.push_back(src_it->second);
                edges.dst_indices.push_back(dst_idx);
                edges.edge_ids.push_back(src_edge.second);
            }
        }
    }
}

// One node-aligned COL_IDX window: nodes [u_lo, u_hi), edges col_idx[edge_lo,
// edge_hi) == col_idx[row_ptr[u_lo], row_ptr[u_hi]). Node-aligned => a node's
// whole neighbour list is wholly inside one window (never straddles a boundary).
struct Window {
    uint32_t u_lo, u_hi;
    uint64_t edge_lo, edge_hi;
};

// Partition [0, n_rows) into node-aligned windows whose edge span stays <=
// soft_cap, EXCEPT a single node whose degree alone exceeds soft_cap forms its
// own window. buffer_cap (out) = max edge span over all windows = max(soft_cap,
// max single-node degree), so one staging buffer of buffer_cap edges fits every
// window's map. row_ptr has n_rows+1 entries (row_ptr[n_rows] == total edges).
void compute_node_aligned_windows(const uint64_t* row_ptr, uint32_t n_rows,
                                  std::size_t soft_cap,
                                  std::vector<Window>& out,
                                  std::size_t& buffer_cap) {
    out.clear();
    const std::size_t cap = soft_cap > 0 ? soft_cap : 1;
    buffer_cap = cap;
    uint32_t u_lo = 0;
    while (u_lo < n_rows) {
        const uint64_t edge_lo = row_ptr[u_lo];
        uint32_t u_hi = u_lo + 1;  // always at least one node (super-hub allowed)
        while (u_hi < n_rows && (row_ptr[u_hi + 1] - edge_lo) <= cap) {
            ++u_hi;
        }
        const uint64_t edge_hi = row_ptr[u_hi];
        out.push_back(Window{u_lo, u_hi, edge_lo, edge_hi});
        const std::size_t span = static_cast<std::size_t>(edge_hi - edge_lo);
        if (span > buffer_cap) buffer_cap = span;
        u_lo = u_hi;
    }
}

}  // namespace

std::vector<std::vector<uint32_t>> gpu_sample_neighbors_for_test(
    const std::vector<uint64_t>& row_ptr, const std::vector<uint32_t>& col_idx,
    const std::vector<uint32_t>& nodes, int fanout, uint64_t batch_seed,
    int layer) {
    if (nodes.empty() || fanout <= 0 || row_ptr.size() < 2) {
        return std::vector<std::vector<uint32_t>>(nodes.size());
    }
    uint64_t* d_row_ptr = nullptr;
    uint32_t* d_col_idx = nullptr;
    CUDA_CHECK(cudaMalloc(&d_row_ptr, row_ptr.size() * sizeof(uint64_t)));
    CUDA_CHECK(cudaMalloc(&d_col_idx, col_idx.size() * sizeof(uint32_t)));
    CUDA_CHECK(cudaMemcpy(d_row_ptr, row_ptr.data(),
                          row_ptr.size() * sizeof(uint64_t),
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_col_idx, col_idx.data(),
                          col_idx.size() * sizeof(uint32_t),
                          cudaMemcpyHostToDevice));

    CsrRange a{d_row_ptr, d_col_idx, static_cast<uint32_t>(row_ptr.size() - 1)};
    CsrRange b{nullptr, nullptr, 0};
    auto result =
        sample_layer_on_device_(a, b, nodes, fanout, batch_seed, layer);

    cudaFree(d_row_ptr);
    cudaFree(d_col_idx);
    return result;
}

std::vector<std::vector<uint32_t>> gpu_sample_neighbors_tiled_for_test(
    const std::vector<uint64_t>& row_ptr, const std::vector<uint32_t>& col_idx,
    const std::vector<uint32_t>& nodes, int fanout, uint64_t batch_seed,
    int layer, std::size_t window_cap_edges) {
    const std::size_t F = nodes.size();
    std::vector<std::vector<uint32_t>> result(F);
    if (nodes.empty() || fanout <= 0 || row_ptr.size() < 2) return result;

    const uint32_t n_rows = static_cast<uint32_t>(row_ptr.size() - 1);

    // Pin ROW_PTR whole on the device (mirrors the tiled view: ROW_PTR resident,
    // COL_IDX streamed). deg_of_/base reads only need the full ROW_PTR.
    uint64_t* d_row_ptr = nullptr;
    CUDA_CHECK(cudaMalloc(&d_row_ptr, row_ptr.size() * sizeof(uint64_t)));
    CUDA_CHECK(cudaMemcpy(d_row_ptr, row_ptr.data(),
                          row_ptr.size() * sizeof(uint64_t),
                          cudaMemcpyHostToDevice));

    std::vector<Window> windows;
    std::size_t buffer_cap = 0;
    compute_node_aligned_windows(row_ptr.data(), n_rows, window_cap_edges,
                                 windows, buffer_cap);

    // One reusable pinned, device-mapped staging buffer sized to the largest
    // window span (so a super-hub window also fits).
    uint32_t* h_win = nullptr;
    uint32_t* d_win = nullptr;
    CUDA_CHECK(cudaHostAlloc(reinterpret_cast<void**>(&h_win),
                             buffer_cap * sizeof(uint32_t), cudaHostAllocMapped));
    CUDA_CHECK(cudaHostGetDevicePointer(reinterpret_cast<void**>(&d_win),
                                        h_win, 0));

    // Per window: gather the sub-frontier of nodes whose row lies in [u_lo,u_hi),
    // stage that window's COL_IDX into the pinned buffer, sample, scatter back to
    // the original frontier index. Sequential windows are safe because
    // sample_layer_on_device_ synchronises before returning.
    for (const auto& w : windows) {
        std::vector<uint32_t> sub_frontier;
        std::vector<std::size_t> orig_idx;
        for (std::size_t i = 0; i < F; ++i) {
            const uint32_t v = nodes[i];
            if (v >= w.u_lo && v < w.u_hi) {
                sub_frontier.push_back(v);
                orig_idx.push_back(i);
            }
        }
        if (sub_frontier.empty()) continue;

        const std::size_t span = static_cast<std::size_t>(w.edge_hi - w.edge_lo);
        std::memcpy(h_win, col_idx.data() + w.edge_lo, span * sizeof(uint32_t));

        CsrRange a{d_row_ptr, d_win, n_rows, w.edge_lo};
        CsrRange b{nullptr, nullptr, 0, 0};
        auto sub = sample_layer_on_device_(a, b, sub_frontier, fanout,
                                           batch_seed, layer);
        for (std::size_t j = 0; j < sub.size(); ++j) {
            result[orig_idx[j]] = std::move(sub[j]);
        }
    }

    cudaFreeHost(h_win);
    cudaFree(d_row_ptr);
    return result;
}

GraphSample sample_khop_gpu(const std::vector<ObjectId>& seeds, uint64_t batch_id,
                            SplitType split, const std::vector<int>& fanouts,
                            const PinnedTopologyView& view,
                            const SamplingBackendPlan& plan,
                            uint64_t random_seed) {
    GraphSample sample;
    sample.batch_id = batch_id;
    sample.split = split;
    if (seeds.empty() || fanouts.empty()) return sample;

    const size_t K = fanouts.size();
    sample.nodes_per_layer.resize(K + 1);
    sample.nodes_per_layer[0] = seeds;  // seeds-first, verbatim order

    const uint64_t batch_seed = random_seed ^ batch_id;  // same mix as the CPU path

    // Direction selection: REVERSE samples in-neighbors (rev CSR), NATURAL the
    // out-neighbors (fwd CSR), UNDIRECTED the union of both.
    const PinnedDirView* va = nullptr;
    const PinnedDirView* vb = nullptr;
    switch (plan.directions) {
        case GpuDirections::FORWARD_ONLY: va = view.fwd(); break;
        case GpuDirections::REVERSE_ONLY: va = view.rev(); break;
        case GpuDirections::BOTH:         va = view.fwd(); vb = view.rev(); break;
        case GpuDirections::NONE:         return sample;
    }
    if (va == nullptr) {
        throw std::runtime_error(
            "sample_khop_gpu: pinned view missing the requested direction");
    }

    CsrRange a{va->d_row_ptr, va->d_col_idx, static_cast<uint32_t>(va->n_rows)};
    CsrRange b{vb ? vb->d_row_ptr : nullptr, vb ? vb->d_col_idx : nullptr,
               vb ? static_cast<uint32_t>(vb->n_rows) : 0u};
    uint64_t dst_tag = va->dst_type_tag;  // tag<<56, re-OR'd onto dense ids
    // The symmetric slice persists tag 0, and a directional reader carries a
    // non-zero tag only when NARROW (id_width==4). On a default (wide) projection
    // the view tag is 0, so the kernel would emit tag-less neighbor ObjectIds that
    // miss the (tagged) feature row map -> silent zero-fill. Recover the real node
    // type tag from a seed ObjectId (seeds carry their real tag) so every expanded
    // neighbor resolves on ANY projection (wide or narrow). get_type() = id & TYPE_MASK.
    if (dst_tag == 0 && !seeds.empty()) {
        dst_tag = seeds.front().get_type();
    }

    // Tiled view: COL_IDX is not resident whole — it is streamed per node-aligned
    // window. The CSR is fixed across layers, so compute the windows once here.
    // buffer_cap <= va->window_cap_edges by construction (the lean store sizes the
    // window cap to >= the max node degree, so a super-hub window still fits).
    std::vector<Window> windows;
    if (va->tiled) {
        std::size_t buffer_cap = 0;
        compute_node_aligned_windows(va->h_row_ptr,
                                     static_cast<uint32_t>(va->n_rows),
                                     va->window_cap_edges, windows, buffer_cap);
    }

    // Expand one layer's frontier into per-frontier-index sampled neighbours.
    // Non-tiled: ONE launch over the whole CSR — byte-identical to the original
    // single sample_layer_on_device_ call. Tiled (symmetric FORWARD_ONLY, so vb is
    // null): partition the frontier by node-aligned window, stage that window's
    // COL_IDX (map_col_window) and launch per non-empty window with
    // CsrRange.col_base_offset, scattering results back to the frontier index.
    // Both return the same vector<vector<uint32_t>> shape, so the reassembly tail
    // below is untouched. Sequential windows reuse the one pinned buffer safely
    // (sample_layer_on_device_ synchronises before returning).
    auto expand_layer = [&](const std::vector<uint32_t>& frontier, int fanout,
                            int layer) -> std::vector<std::vector<uint32_t>> {
        if (!va->tiled) {
            return sample_layer_on_device_(a, b, frontier, fanout, batch_seed,
                                           layer);
        }
        std::vector<std::vector<uint32_t>> out(frontier.size());
        for (const auto& w : windows) {
            std::vector<uint32_t> sub;
            std::vector<std::size_t> orig;
            for (std::size_t i = 0; i < frontier.size(); ++i) {
                const uint32_t v = frontier[i];
                if (v >= w.u_lo && v < w.u_hi) {
                    sub.push_back(v);
                    orig.push_back(i);
                }
            }
            if (sub.empty()) continue;
            const uint32_t* d_col = view.map_col_window(*va, w.edge_lo, w.edge_hi);
            CsrRange wa{va->d_row_ptr, d_col,
                        static_cast<uint32_t>(va->n_rows), w.edge_lo};
            CsrRange wb{nullptr, nullptr, 0, 0};
            auto s = sample_layer_on_device_(wa, wb, sub, fanout, batch_seed,
                                             layer);
            for (std::size_t j = 0; j < s.size(); ++j) {
                out[orig[j]] = std::move(s[j]);
            }
        }
        return out;
    };

    std::vector<std::unordered_map<
        uint64_t, std::vector<std::pair<ObjectId, ObjectId>>>>
        sampled_edges(K);

    for (size_t k = 0; k < K; ++k) {
        const auto& current = sample.nodes_per_layer[k];
        std::vector<uint32_t> frontier(current.size());
        for (size_t i = 0; i < current.size(); ++i) {
            frontier[i] = static_cast<uint32_t>(current[i].get_value());
        }

        auto sampled = expand_layer(frontier, fanouts[k], static_cast<int>(k));

        std::unordered_set<uint64_t> next_set;
        for (size_t i = 0; i < current.size(); ++i) {
            const auto& nbrs = sampled[i];
            if (nbrs.empty()) continue;
            auto& vec = sampled_edges[k][current[i].id];
            vec.reserve(nbrs.size());
            for (uint32_t nd : nbrs) {
                // Reconstruct the tagged ObjectId from the tag-stripped dense id.
                ObjectId src(dst_tag | static_cast<uint64_t>(nd));
                vec.emplace_back(src, ObjectId(0));  // edge_id=0 (no eids in CSR)
                next_set.insert(src.id);
            }
        }
        sample.nodes_per_layer[k + 1].reserve(next_set.size());
        for (uint64_t id : next_set) {
            sample.nodes_per_layer[k + 1].emplace_back(id);
        }
    }

    build_edges_(sample, sampled_edges);
    sample.rebuild_unique_nodes();
    return sample;
}

}  // namespace mdb::gnn
