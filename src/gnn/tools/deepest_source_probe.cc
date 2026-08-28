// Empirical check (read-only) for a proposed deepest-layer fused gather+mean
// kernel: would it still have to READ essentially all of A_L? For each baked
// batch it computes the set of A_L nodes such a kernel must read = A_{L-1}
// (read as x_self, the prefix) UNION the distinct SOURCES of the deepest-hop
// edges (read to compute the mean). Reports read_fraction = |read_set| / |A_L|
// and unread_fraction = A_L nodes the kernel could skip. If unread ~ 0, the
// kernel reads ~100% of A_L and cannot cut the tier-read cost.
#include "gnn/sampling/sample_storage.h"
#include "gnn/sampling/graph_sample.h"

#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <stdexcept>
#include <unordered_set>

using namespace mdb::gnn;

int main(int argc, char** argv) {
    if (argc < 2) {
        std::fprintf(stderr, "usage: deepest_source_probe <sample_dir> [max_batches]\n");
        return 2;
    }
    auto st = SampleStorage::open(argv[1]);
    // NB: do NOT skip edges — the deepest edge_index is the whole point.
    const uint64_t max_b = (argc >= 3) ? std::strtoull(argv[2], nullptr, 10) : 5;

    unsigned long long sum_AL = 0, sum_ALm1 = 0, sum_src = 0, sum_read = 0, sum_unread = 0;
    unsigned long long n = 0;

    for (uint64_t b = 0; b < max_b; ++b) {
        GraphSample s;
        try { s = st.read_sample(b); } catch (const std::exception&) { break; }
        const size_t L = s.nodes_per_layer.size();           // K+1
        if (L < 2 || s.edges_per_layer.empty()) continue;
        const size_t K = L - 1;

        // A_{L-1} = union of node layers 0..K-1 (everything but the deepest hop).
        std::unordered_set<uint64_t> read_set;               // starts as A_{L-1}, then += deepest sources
        for (size_t k = 0; k + 1 < L; ++k)
            for (const auto& o : s.nodes_per_layer[k]) read_set.insert(o.get_value());
        const unsigned long long ALm1 = read_set.size();

        // deepest layer = nodes_per_layer[K]; deepest edges = edges_per_layer[K-1];
        // src_indices are LOCAL indices into nodes_per_layer[K].
        const auto& deepest_nodes = s.nodes_per_layer[K];
        const auto& de            = s.edges_per_layer[K - 1];
        std::unordered_set<int32_t> src_local;
        for (int32_t si : de.src_indices) src_local.insert(si);
        for (int32_t si : src_local)
            if (si >= 0 && static_cast<size_t>(si) < deepest_nodes.size())
                read_set.insert(deepest_nodes[si].get_value());

        // A_L and the nodes the kernel could skip (in A_L but neither x_self nor a source).
        const unsigned long long AL = s.all_unique_nodes.size();
        unsigned long long unread = 0;
        for (const auto& o : s.all_unique_nodes)
            if (!read_set.count(o.get_value())) ++unread;

        const unsigned long long readN = read_set.size();
        sum_AL += AL; sum_ALm1 += ALm1; sum_src += src_local.size();
        sum_read += readN; sum_unread += unread; ++n;

        std::fprintf(stderr,
            "  batch %llu: A_L=%llu A_{L-1}=%llu deepest_src_distinct=%zu "
            "read_set=%llu (%.3f%% of A_L) unread=%llu (%.3f%%)\n",
            (unsigned long long)b, AL, ALm1, src_local.size(), readN,
            AL ? 100.0 * readN / AL : 0.0, unread, AL ? 100.0 * unread / AL : 0.0);
    }

    std::printf("batches=%llu\n", n);
    std::printf("sum_A_L=%llu\n", sum_AL);
    std::printf("sum_read_set=%llu  (A_L nodes the kernel MUST read = A_{L-1} U deepest_sources)\n", sum_read);
    std::printf("READ_FRACTION_of_A_L=%.6f\n", sum_AL ? (double)sum_read / sum_AL : 0.0);
    std::printf("sum_unread=%llu  (A_L nodes the kernel could SKIP)\n", sum_unread);
    std::printf("UNREAD_FRACTION=%.6f\n", sum_AL ? (double)sum_unread / sum_AL : 0.0);
    return 0;
}
