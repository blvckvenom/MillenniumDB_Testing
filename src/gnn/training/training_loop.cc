#include "gnn/training/training_loop.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <stdexcept>
#include <string>

#include <torch/torch.h>

namespace mdb::gnn {

// =============================================================================
// Construction
// =============================================================================

TrainingLoop::TrainingLoop(
    GraphSAGEModel&      model,
    BatchAssembler&      assembler,
    const SampleCatalog& catalog,
    torch::optim::Adam&  optimizer,
    Config               config)
    : model_(model)
    , assembler_(assembler)
    , catalog_(catalog)
    , optimizer_(optimizer)
    , config_(std::move(config))
{
}

// =============================================================================
// Training
// =============================================================================

TrainingLoop::Result TrainingLoop::train()
{
    Result result;

    // --- Reproducibility -------------------------------------------------
    if (config_.random_seed >= 0) {
        torch::manual_seed(static_cast<uint64_t>(config_.random_seed));
    }

    // --- Device detection ---------------------------------------------------
    // Probe the first batch to discover whether the FeatureAssembler returns
    // CUDA tensors (L1 cache in GPU VRAM) or CPU tensors. If CUDA, move
    // the model so all parameters are on the same device as features.
    torch::Device device(torch::kCPU);
    {
        MiniBatch probe = assembler_.assemble(0);
        device = probe.features.device();
    }
    if (!device.is_cpu()) {
        model_.to(device);
    }

    // Optimizer is owned by the caller (accessible via optimizer_). Do NOT
    // re-create it here — doing so would discard Adam momenta carried over
    // from a resumed checkpoint.
    auto& optimizer = optimizer_;

    // Seed from resume state (defaults to fresh training)
    double   best_val_acc     = config_.start_best_val;
    uint64_t patience_counter = config_.start_patience;

    // Prepend seed_losses into result so epoch_losses history is contiguous
    for (double l : config_.seed_losses) {
        result.epoch_losses.push_back(l);
    }

    const uint64_t train_batches = catalog_.train_batches;
    const uint64_t val_batches   = catalog_.validation_batches;

    auto wall_start = std::chrono::steady_clock::now();

    for (uint64_t epoch = config_.start_epoch;
         epoch < config_.start_epoch + config_.epochs;
         ++epoch)
    {

        // === Training phase ===
        model_.train();

        double   total_loss       = 0.0;
        uint64_t num_train_batches = 0;

        for (uint64_t bid = 0; bid < train_batches; ++bid) {
            MiniBatch mini = assembler_.assemble(bid);

            // Move all batch tensors to the training device
            if (!device.is_cpu()) {
                mini.features = mini.features.to(device);
                for (auto& ei : mini.edge_indices) {
                    ei = ei.to(device);
                }
                mini.labels     = mini.labels.to(device);
                mini.label_mask = mini.label_mask.to(device);
            }

            optimizer.zero_grad();

            auto logits = model_.forward(
                mini.features,
                mini.edge_indices,
                static_cast<int64_t>(mini.num_seeds)
            );

            // Only back-prop if at least one labeled seed exists in this batch
            if (mini.label_mask.any().item<bool>()) {
                auto masked_logits = logits.index({mini.label_mask});
                auto masked_labels = mini.labels.index({mini.label_mask});

                auto loss = torch::nn::functional::cross_entropy(
                    masked_logits, masked_labels
                );

                loss.backward();
                optimizer.step();

                total_loss += loss.item<double>();
            }

            ++num_train_batches;
        }

        double avg_loss = (num_train_batches > 0)
            ? (total_loss / static_cast<double>(num_train_batches))
            : 0.0;
        result.epoch_losses.push_back(avg_loss);

        // === Validation phase ===
        double val_accuracy = evaluate(train_batches, val_batches);

        // === Checkpoint if improved ===
        if (val_accuracy > best_val_acc) {
            best_val_acc = val_accuracy;
            patience_counter = 0;
            // Checkpoint persistence is now the responsibility of the
            // on_epoch_end callback (see AutoCheckpointer in Phase 3).
        } else {
            ++patience_counter;
            if (patience_counter >= config_.patience) {
                // Stopped by patience — not a convergence stop
                result.converged = false;
                ++epoch;  // account for this epoch before break
                result.ran_epochs = epoch - config_.start_epoch;
                result.best_val_accuracy = best_val_acc;

                auto wall_end = std::chrono::steady_clock::now();
                result.train_seconds = std::chrono::duration<double>(
                    wall_end - wall_start).count();
                return result;
            }
        }

        // Fire per-epoch callback (e.g. AutoCheckpointer)
        if (config_.on_epoch_end) {
            config_.on_epoch_end(EpochEvent{
                epoch,
                avg_loss,
                val_accuracy,
                patience_counter,
                (val_accuracy > best_val_acc - 1e-12)
            });
        }

        // === Convergence check ===
        const size_t n = result.epoch_losses.size();
        if (n >= 2) {
            double delta = std::abs(
                result.epoch_losses[n - 1] - result.epoch_losses[n - 2]
            );
            if (delta < config_.tolerance) {
                result.converged  = true;
                result.ran_epochs = (epoch - config_.start_epoch) + 1;
                result.best_val_accuracy = best_val_acc;

                auto wall_end = std::chrono::steady_clock::now();
                result.train_seconds = std::chrono::duration<double>(
                    wall_end - wall_start).count();
                return result;
            }
        }
    }

    // Completed all epochs without early stopping
    result.ran_epochs        = config_.epochs;
    result.best_val_accuracy = best_val_acc;

    auto wall_end = std::chrono::steady_clock::now();
    result.train_seconds = std::chrono::duration<double>(
        wall_end - wall_start).count();
    return result;
}

// =============================================================================
// Evaluation
// =============================================================================

double TrainingLoop::evaluate(uint64_t start_batch, uint64_t count)
{
    if (count == 0) {
        return 0.0;
    }

    torch::NoGradGuard no_grad;
    model_.eval();

    int64_t correct = 0;
    int64_t total   = 0;

    for (uint64_t i = 0; i < count; ++i) {
        auto mini = assembler_.assemble(start_batch + i);

        // Move all batch tensors to the model's device
        auto dev = model_.parameters().begin()->device();
        if (!dev.is_cpu()) {
            mini.features = mini.features.to(dev);
            for (auto& ei : mini.edge_indices) {
                ei = ei.to(dev);
            }
            mini.labels     = mini.labels.to(dev);
            mini.label_mask = mini.label_mask.to(dev);
        }

        auto logits = model_.forward(
            mini.features,
            mini.edge_indices,
            static_cast<int64_t>(mini.num_seeds)
        );

        if (mini.label_mask.any().item<bool>()) {
            auto masked_logits = logits.index({mini.label_mask});
            auto masked_labels = mini.labels.index({mini.label_mask});

            auto predicted = masked_logits.argmax(1);
            correct += (predicted == masked_labels).sum().item<int64_t>();
            total   += masked_labels.size(0);
        }
    }

    // Restore training mode for subsequent use
    model_.train();

    return (total > 0) ? (static_cast<double>(correct) / static_cast<double>(total)) : 0.0;
}

} // namespace mdb::gnn
