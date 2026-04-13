#include "gnn/output/embedding_writer.h"

#include <chrono>
#include <cstdint>
#include <utility>
#include <vector>

#include <torch/torch.h>

namespace mdb::gnn {

// =============================================================================
// Construction
// =============================================================================

EmbeddingWriter::EmbeddingWriter(
    GraphSAGEModel&         model,
    BatchAssembler&         assembler,
    SampleStorage&          sample_storage,
    const RowMapping&       row_mapping,
    const SampleCatalog&    catalog,
    Config                  config
)
    : model_(model)
    , assembler_(assembler)
    , sample_storage_(sample_storage)
    , row_mapping_(row_mapping)
    , catalog_(catalog)
    , config_(std::move(config))
{
}

// =============================================================================
// write_all — orchestrator (Phase A only; B and C are stubs)
// =============================================================================

EmbeddingWriter::Result EmbeddingWriter::write_all() {
    Result result;

    auto t0 = std::chrono::steady_clock::now();

    auto seed_embs = collect_seed_embeddings();

    auto t1 = std::chrono::steady_clock::now();
    result.inference_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

    // TODO: Phase B (infer non-seeds) -- Task 5
    // TODO: Phase C (write to projection) -- Task 6

    result.nodes_written = seed_embs.size();
    return result;
}

// =============================================================================
// Phase A: collect_seed_embeddings
// =============================================================================

std::vector<std::pair<uint64_t, torch::Tensor>> EmbeddingWriter::collect_seed_embeddings() {
    // --- Put model in inference mode ----------------------------------------
    model_.eval();
    torch::NoGradGuard no_grad;

    // --- Detect model device ------------------------------------------------
    // Same pattern as TrainingLoop::evaluate() (training_loop.cc:201).
    torch::Device device(torch::kCPU);
    if (!model_.parameters().empty()) {
        device = model_.parameters().begin()->device();
    }

    // --- Iterate all batches ------------------------------------------------
    std::vector<std::pair<uint64_t, torch::Tensor>> result;

    for (uint64_t bid = 0; bid < catalog_.total_batches; ++bid) {
        // 1. Assemble MiniBatch (features + edge_indices + labels)
        MiniBatch mini = assembler_.assemble(bid);

        // 2. Read GraphSample to recover seed ObjectIds (nodes_per_layer[0])
        GraphSample sample = sample_storage_.read_sample(bid);

        const auto& seed_oids = sample.nodes_per_layer[0];
        const auto  num_seeds = static_cast<int64_t>(seed_oids.size());

        if (num_seeds == 0) {
            continue;
        }

        // 3. Move batch tensors to model device if CUDA
        if (!device.is_cpu()) {
            mini.features = mini.features.to(device);
            for (auto& ei : mini.edge_indices) {
                ei = ei.to(device);
            }
        }

        // 4. Forward pass (hidden representation, NOT logits)
        auto emb = model_.get_embeddings(
            mini.features,
            mini.edge_indices,
            num_seeds
        );
        // emb shape: [num_seeds, hidden_dim]

        // 5. Move to CPU for storage
        emb = emb.cpu().contiguous();

        // 6. Map each seed embedding to its RowMapping index
        for (int64_t i = 0; i < num_seeds; ++i) {
            auto row_opt = row_mapping_.find(seed_oids[static_cast<size_t>(i)]);
            if (!row_opt) {
                // Defensive: skip seeds not found in RowMapping.
                // Should never happen with consistent data.
                continue;
            }
            result.emplace_back(*row_opt, emb[i].clone());
        }
    }

    return result;
}

} // namespace mdb::gnn
