#include "gnn/output/auto_checkpointer.h"

#include "gnn/models/graphsage_model.h"

namespace mdb::gnn {

AutoCheckpointer::AutoCheckpointer(
    GraphSAGEModel&                 model,
    torch::optim::Adam&             optimizer,
    const std::filesystem::path&    ckpt_dir,
    TrainingState                   base_state,
    Policy                          policy)
    : model_(model)
    , optimizer_(optimizer)
    , ckpt_dir_(ckpt_dir)
    , base_state_(std::move(base_state))
    , policy_(std::move(policy))
{
    std::filesystem::create_directories(ckpt_dir_);
}

AutoCheckpointer::AutoCheckpointer(
    GraphSAGEModel&                 model,
    torch::optim::Adam&             optimizer,
    const std::filesystem::path&    ckpt_dir,
    TrainingState                   base_state)
    : AutoCheckpointer(model, optimizer, ckpt_dir, std::move(base_state), Policy{})
{
}

void AutoCheckpointer::on_epoch_end(const TrainingLoop::EpochEvent& e)
{
    if (!policy_.save_on_best_val) return;

    // Trust the loop's improvement flag: TrainingLoop computes is_best
    // against a best-so-far tracker seeded with Config::start_best_val
    // (the resumed checkpoint's best_val_accuracy), so it stays correct
    // across resumeFrom continuations. A local comparison starting at 0.0
    // would treat the first post-resume epoch as a new best and overwrite
    // best_model with a worse model.
    if (e.is_best) {
        best_val_seen_ = e.val_accuracy;

        TrainingState s          = base_state_;
        s.epoch                  = e.epoch + 1;             // store "next epoch to run"
        s.patience_counter       = e.patience_counter;
        s.best_val_accuracy      = static_cast<float>(e.val_accuracy);

        ModelCheckpoint::save_full(
            model_, optimizer_,
            ckpt_dir_ / policy_.best_basename,
            s);
        ++saves_written_;
    }
}

void AutoCheckpointer::save_final(const TrainingState& final_state)
{
    if (!policy_.save_final) return;
    ModelCheckpoint::save_full(
        model_, optimizer_,
        ckpt_dir_ / policy_.final_basename,
        final_state);
    ++saves_written_;
}

} // namespace mdb::gnn
