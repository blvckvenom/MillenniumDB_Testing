#pragma once

#include <array>
#include <cstdint>
#include <filesystem>
#include <optional>
#include <string>
#include <vector>

#include <torch/torch.h>

namespace mdb::gnn {

class GraphSAGEModel;  // forward decl — defined in gnn/models/graphsage_model.h

// ---------------------------------------------------------------------------
// SaveKind — whether a checkpoint carries full training state or only weights
// ---------------------------------------------------------------------------
enum class SaveKind : uint32_t {
    WeightsOnly = 0,   // model parameters only; no optimizer buffers
    Full        = 1,   // model parameters + optimizer state + TrainingState
};

// ---------------------------------------------------------------------------
// TrainingState — serialized payload of a full checkpoint
// ---------------------------------------------------------------------------
struct TrainingState {
    // Critical for resume correctness
    uint64_t epoch              = 0;
    uint64_t patience_counter   = 0;
    float    best_val_accuracy  = 0.0f;
    std::vector<double> epoch_losses;

    // Architecture validation (matches GraphSAGEConfig fields)
    int32_t  input_dim          = 0;
    int32_t  hidden_dim         = 0;
    int32_t  num_classes        = 0;
    int32_t  num_layers         = 0;
    float    dropout            = 0.0f;
    bool     normalize          = false;

    // Metadata
    std::string model_type;                         // e.g. "graphsage"
    std::string projection_name;                    // e.g. "cora_projection"
    std::array<uint8_t, 32> gnn_meta_hash{};        // SHA-256 of gnn_meta.bin
    uint64_t    creation_time_unix = 0;             // seconds since UNIX epoch
    double      total_training_time_sec = 0.0;
};

// ---------------------------------------------------------------------------
// CheckpointInfo — lightweight metadata returned by list_checkpoints
// (epoch_losses and gnn_meta_hash NOT populated — load_full/load_weights only)
// ---------------------------------------------------------------------------
struct CheckpointInfo {
    std::filesystem::path basename;   // absolute path without extension
    SaveKind    save_kind             = SaveKind::Full;
    uint64_t    epoch                 = 0;
    float       best_val_accuracy     = 0.0f;
    uint64_t    creation_time_unix    = 0;
    std::string model_type;
    std::string projection_name;
    uint64_t    pt_bytes              = 0;
    uint64_t    meta_bytes            = 0;
};

// ---------------------------------------------------------------------------
// ModelCheckpoint — stateless utility (all methods static)
// ---------------------------------------------------------------------------
class ModelCheckpoint {
public:
    // =====================================================================
    // Save
    // =====================================================================
    static void save_full(
        const GraphSAGEModel&       model,
        const torch::optim::Adam&   optimizer,
        const std::filesystem::path& basename,
        TrainingState                state
    );

    static void save_weights(
        const GraphSAGEModel&       model,
        const std::filesystem::path& basename,
        TrainingState                state
    );

    // =====================================================================
    // Load
    // =====================================================================
    static TrainingState load_full(
        GraphSAGEModel&               model,
        torch::optim::Adam&           optimizer,
        const std::filesystem::path&  basename
    );

    static TrainingState load_weights(
        GraphSAGEModel&               model,
        const std::filesystem::path&  basename
    );

    // =====================================================================
    // Discovery / lifecycle
    // =====================================================================
    static std::vector<CheckpointInfo> list_checkpoints(
        const std::filesystem::path&      dir,
        std::optional<std::string>        name_filter = std::nullopt
    );

    static bool exists(const std::filesystem::path& basename);

    static void delete_checkpoint(const std::filesystem::path& basename);

    // =====================================================================
    // Validation helpers (public so tests can exercise them directly)
    // =====================================================================
    static std::array<uint8_t, 32> compute_gnn_meta_hash(
        const std::filesystem::path& gnn_meta_bin_path
    );

    static void validate_compat(
        const TrainingState&          state,
        const std::filesystem::path&  current_gnn_meta_path,
        const std::string&            current_projection_name
    );

    // =====================================================================
    // Low-level binary I/O — exposed for unit testing but not for external use
    // =====================================================================
    static void          write_ckptmeta(const std::filesystem::path&, const TrainingState&);
    static TrainingState read_ckptmeta(const std::filesystem::path&);

private:
    static void fsync_directory(const std::filesystem::path& dir);
    static void cleanup_tmps(const std::filesystem::path& basename);
};

} // namespace mdb::gnn
