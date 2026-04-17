#pragma once

#include <cstdint>
#include <filesystem>
#include <string>

#include <torch/torch.h>

#include "gnn/output/model_checkpoint.h"
#include "gnn/training/training_loop.h"

namespace mdb::gnn {

class GraphSAGEModel;

/**
 * @brief Writes checkpoints per policy in response to TrainingLoop events.
 *
 * Registered as TrainingLoop::Config::on_epoch_end. Also invoked explicitly
 * by the training procedure after loop.train() returns (save_final).
 */
class AutoCheckpointer {
public:
    struct Policy {
        bool        save_on_best_val = true;
        bool        save_final       = true;
        std::string best_basename    = "best_model";
        std::string final_basename   = "final_model";
    };

    AutoCheckpointer(
        GraphSAGEModel&                 model,
        torch::optim::Adam&             optimizer,
        const std::filesystem::path&    ckpt_dir,
        TrainingState                   base_state,
        Policy                          policy
    );

    AutoCheckpointer(
        GraphSAGEModel&                 model,
        torch::optim::Adam&             optimizer,
        const std::filesystem::path&    ckpt_dir,
        TrainingState                   base_state
    );

    void on_epoch_end(const TrainingLoop::EpochEvent& e);
    void save_final(const TrainingState& final_state);

    double   best_val_seen() const { return best_val_seen_; }
    uint64_t saves_written() const { return saves_written_; }

private:
    GraphSAGEModel&              model_;
    torch::optim::Adam&          optimizer_;
    std::filesystem::path        ckpt_dir_;
    TrainingState                base_state_;
    Policy                       policy_;
    double                       best_val_seen_ = 0.0;
    uint64_t                     saves_written_ = 0;
};

} // namespace mdb::gnn
