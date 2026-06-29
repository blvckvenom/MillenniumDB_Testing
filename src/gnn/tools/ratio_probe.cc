// Offline probe (read-only): measures the active-set ratio that gates the
// deepest-layer aggregation memoization lever. For each baked batch it reports
//   A_L     = all_unique_nodes.size()                       (full receptive field)
//   A_{L-1} = |dedup(nodes_per_layer[0..K-1])|              (all but the deepest hop)
// and the aggregate ratio Sum(A_{L-1}) / Sum(A_L). Reuses the authoritative
// GraphSample::deserialize via SampleStorage. Edges are skipped on read.
#include "gnn/sampling/sample_storage.h"
#include "gnn/sampling/graph_sample.h"

#include <cstdio>
#include <cstdint>
#include <stdexcept>
#include <unordered_set>

using namespace mdb::gnn;

int main(int argc, char** argv) {
    if (argc < 2) {
        std::fprintf(stderr, "usage: ratio_probe <sample_dir>\n");
        return 2;
    }
    auto st = SampleStorage::open(argv[1]);
    st.set_skip_edges_on_read(true);
    st.set_skip_edge_ids_on_read(true);

    unsigned long long sum_AL = 0, sum_ALm1 = 0;
    unsigned long long n = 0;
    double ratio_min = 1e9, ratio_max = -1e9;
    std::unordered_set<uint64_t> shallow;  // reused per batch

    for (uint64_t b = 0;; ++b) {
        GraphSample s;
        try {
            s = st.read_sample(b);
        } catch (const std::exception&) {
            break;  // out of range / not found -> end of contiguous samples
        }
        const size_t L = s.nodes_per_layer.size();  // K+1 layers
        if (L == 0) continue;
        const size_t AL = s.all_unique_nodes.size();
        shallow.clear();
        // union of all layers EXCEPT the deepest (last) -> A_{L-1}
        for (size_t k = 0; k + 1 < L; ++k) {
            for (const auto& o : s.nodes_per_layer[k]) shallow.insert(o.get_value());
        }
        const size_t ALm1 = shallow.size();
        sum_AL += AL;
        sum_ALm1 += ALm1;
        ++n;
        if (AL > 0) {
            double r = static_cast<double>(ALm1) / static_cast<double>(AL);
            if (r < ratio_min) ratio_min = r;
            if (r > ratio_max) ratio_max = r;
        }
        if (b < 3) {
            std::fprintf(stderr, "  batch %llu: layers=%zu A_L=%zu A_{L-1}=%zu r=%.4f\n",
                         (unsigned long long)b, L, AL, ALm1,
                         AL ? (double)ALm1 / AL : 0.0);
        }
        if ((b % 200) == 0) {
            std::fprintf(stderr, "  ...processed %llu batches\n", (unsigned long long)b);
        }
    }

    const double ratio = sum_AL ? (double)sum_ALm1 / (double)sum_AL : 0.0;
    std::printf("batches=%llu\n", n);
    std::printf("sum_A_L(all_unique)=%llu\n", sum_AL);
    std::printf("sum_A_Lminus1(shallow union)=%llu\n", sum_ALm1);
    std::printf("RATIO_A_Lminus1_over_A_L=%.6f\n", ratio);
    std::printf("deepest_ring_only_fraction=%.6f\n", 1.0 - ratio);
    std::printf("per_batch_ratio_min=%.4f per_batch_ratio_max=%.4f\n", ratio_min, ratio_max);
    if (n) {
        std::printf("avg_A_L=%.0f avg_A_Lminus1=%.0f\n",
                    (double)sum_AL / n, (double)sum_ALm1 / n);
    }
    return 0;
}
