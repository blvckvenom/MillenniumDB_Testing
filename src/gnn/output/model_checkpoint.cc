#include "gnn/output/model_checkpoint.h"

#include <openssl/evp.h>

#include <fcntl.h>       // open, O_RDONLY
#include <unistd.h>      // fsync, close

#include <algorithm>
#include <cerrno>
#include <cstring>
#include <fstream>
#include <stdexcept>
#include <string>
#include <type_traits>

namespace {

constexpr uint8_t  CKPTMETA_MAGIC[8] = {'G','N','N','C','K','P','T','\0'};
constexpr uint32_t CKPTMETA_VERSION  = 1;

template<typename T>
inline void write_le(std::ofstream& f, const T& v) {
    static_assert(std::is_trivially_copyable_v<T>);
    f.write(reinterpret_cast<const char*>(&v), sizeof(T));
}

template<typename T>
inline void read_le(std::ifstream& f, T& v, const std::string& context) {
    static_assert(std::is_trivially_copyable_v<T>);
    if (!f.read(reinterpret_cast<char*>(&v), sizeof(T))) {
        throw std::runtime_error(
            "ModelCheckpoint::read_ckptmeta: truncated .ckptmeta reading " + context);
    }
}

// Force a file's data to stable storage. Must run before rename() commits a
// checkpoint name: rename durability without data durability can leave a
// committed-named but truncated/empty file after a power loss.
void fsync_file_impl(const std::filesystem::path& path) {
    int fd = ::open(path.c_str(), O_RDONLY);
    if (fd < 0) {
        throw std::runtime_error(
            "ModelCheckpoint: cannot open file for fsync: " + path.string()
            + " (errno=" + std::to_string(errno) + ")");
    }
    if (::fsync(fd) != 0) {
        int e = errno;
        ::close(fd);
        throw std::runtime_error(
            "ModelCheckpoint: fsync failed on " + path.string()
            + " (errno=" + std::to_string(e) + ")");
    }
    ::close(fd);
}

// Peek only the magic + version + save_kind (first 16 bytes) without reading
// the full payload. Used by load_full to branch on kind early.
mdb::gnn::SaveKind peek_save_kind(const std::filesystem::path& meta_path) {
    std::ifstream f(meta_path, std::ios::binary);
    if (!f) {
        throw std::runtime_error(
            "ModelCheckpoint: cannot open " + meta_path.string());
    }
    uint8_t magic[8];
    if (!f.read(reinterpret_cast<char*>(magic), 8)) {
        throw std::runtime_error(
            "ModelCheckpoint: cannot read magic from " + meta_path.string());
    }
    if (std::memcmp(magic, CKPTMETA_MAGIC, 8) != 0) {
        throw std::runtime_error(
            "ModelCheckpoint: invalid magic in " + meta_path.string());
    }
    uint32_t version = 0, kind_val = 0;
    if (!f.read(reinterpret_cast<char*>(&version), 4)) {
        throw std::runtime_error(
            "ModelCheckpoint: truncated header (version) in " + meta_path.string());
    }
    if (!f.read(reinterpret_cast<char*>(&kind_val), 4)) {
        throw std::runtime_error(
            "ModelCheckpoint: truncated header (save_kind) in " + meta_path.string());
    }
    if (version != CKPTMETA_VERSION) {
        throw std::runtime_error(
            "ModelCheckpoint: unsupported checkpoint format version "
            + std::to_string(version));
    }
    if (kind_val > 1) {
        throw std::runtime_error(
            "ModelCheckpoint: invalid save_kind " + std::to_string(kind_val)
            + " in " + meta_path.string());
    }
    return static_cast<mdb::gnn::SaveKind>(kind_val);
}

} // anon

namespace mdb::gnn {

std::array<uint8_t, 32> ModelCheckpoint::compute_gnn_meta_hash(
    const std::filesystem::path& gnn_meta_bin_path)
{
    std::ifstream f(gnn_meta_bin_path, std::ios::binary);
    if (!f) {
        throw std::runtime_error(
            "ModelCheckpoint::compute_gnn_meta_hash: cannot open "
            + gnn_meta_bin_path.string());
    }

    EVP_MD_CTX* ctx = EVP_MD_CTX_new();
    if (!ctx) {
        throw std::runtime_error("EVP_MD_CTX_new failed");
    }
    if (EVP_DigestInit_ex(ctx, EVP_sha256(), nullptr) != 1) {
        EVP_MD_CTX_free(ctx);
        throw std::runtime_error("EVP_DigestInit_ex(SHA-256) failed");
    }

    constexpr size_t BUF = 4096;
    char buf[BUF];
    while (f.read(buf, BUF) || f.gcount() > 0) {
        if (EVP_DigestUpdate(ctx, buf, static_cast<size_t>(f.gcount())) != 1) {
            EVP_MD_CTX_free(ctx);
            throw std::runtime_error("EVP_DigestUpdate failed");
        }
    }

    std::array<uint8_t, 32> digest{};
    unsigned int len = 0;
    if (EVP_DigestFinal_ex(ctx, digest.data(), &len) != 1) {
        EVP_MD_CTX_free(ctx);
        throw std::runtime_error("EVP_DigestFinal_ex failed");
    }
    EVP_MD_CTX_free(ctx);

    if (len != 32) {
        throw std::runtime_error("SHA-256 produced unexpected digest length");
    }
    return digest;
}

// .ckptmeta binary layout (all multi-byte fields little-endian, no padding
// beyond the 3 reserved bytes noted):
//   magic   8 B   "GNNCKPT\0"
//   u32           format version (currently 1)
//   u32           save_kind (0 = WeightsOnly, 1 = Full)
//   u64           epoch
//   u64           patience_counter
//   f32           best_val_accuracy
//   i64 x4        input_dim, hidden_dim, num_classes, num_layers
//   f64           dropout
//   u8            normalize (0/1), then 3 reserved zero bytes
//   f64           total_training_time_sec
//   u64           creation_time_unix
//   32 B          gnn_meta_hash (SHA-256 of gnn_meta.bin)
//   var           projection_name: u32 length + bytes
//   var           model_type:      u32 length + bytes
//   var           epoch_losses:    u32 count + count x f64
// The reader consumes fields in exactly this order; read_ckptmeta bounds every
// var-length prefix by the remaining file size before allocating.
void ModelCheckpoint::write_ckptmeta(
    const std::filesystem::path& path,
    const TrainingState&         s)
{
    // Public entry point: always writes SaveKind::Full.
    // save_full / save_weights use write_ckptmeta_impl with the intended kind.
    write_ckptmeta_impl(path, s, SaveKind::Full);
}

void ModelCheckpoint::write_ckptmeta_impl(
    const std::filesystem::path& path,
    const TrainingState&         s,
    SaveKind                     kind)
{
    std::filesystem::create_directories(path.parent_path());
    std::ofstream f(path, std::ios::binary | std::ios::trunc);
    if (!f) {
        throw std::runtime_error(
            "ModelCheckpoint::write_ckptmeta: cannot open " + path.string());
    }

    // magic + version
    f.write(reinterpret_cast<const char*>(CKPTMETA_MAGIC), 8);
    write_le(f, CKPTMETA_VERSION);

    // save_kind selected by caller (Full from public wrapper, WeightsOnly from
    // save_weights).
    uint32_t save_kind_val = static_cast<uint32_t>(kind);
    write_le(f, save_kind_val);

    // Fixed-size fields, in the exact order documented above (the reader
    // consumes them in the same order).
    write_le(f, s.epoch);
    write_le(f, s.patience_counter);
    write_le(f, s.best_val_accuracy);
    write_le(f, s.input_dim);
    write_le(f, s.hidden_dim);
    write_le(f, s.num_classes);
    write_le(f, s.num_layers);
    write_le(f, s.dropout);
    uint8_t norm = s.normalize ? 1 : 0;
    write_le(f, norm);
    uint8_t reserved[3] = {0, 0, 0};
    f.write(reinterpret_cast<const char*>(reserved), 3);
    write_le(f, s.total_training_time_sec);
    write_le(f, s.creation_time_unix);
    f.write(reinterpret_cast<const char*>(s.gnn_meta_hash.data()), 32);

    // var-length: projection_name, model_type, epoch_losses
    auto write_str = [&](const std::string& str) {
        uint32_t len = static_cast<uint32_t>(str.size());
        write_le(f, len);
        if (len > 0) f.write(str.data(), len);
    };
    write_str(s.projection_name);
    write_str(s.model_type);

    uint32_t k = static_cast<uint32_t>(s.epoch_losses.size());
    write_le(f, k);
    if (k > 0) {
        f.write(reinterpret_cast<const char*>(s.epoch_losses.data()),
                static_cast<std::streamsize>(k * sizeof(double)));
    }

    // close() performs the final flush; checking the stream before it would
    // miss a failure (e.g. ENOSPC) on that last flush and silently leave a
    // truncated file behind.
    f.close();
    if (f.fail()) {
        throw std::runtime_error(
            "ModelCheckpoint::write_ckptmeta: I/O error writing " + path.string());
    }
    fsync_file_impl(path);
}

TrainingState ModelCheckpoint::read_ckptmeta(
    const std::filesystem::path& path)
{
    std::ifstream f(path, std::ios::binary);
    if (!f) {
        throw std::runtime_error(
            "ModelCheckpoint::read_ckptmeta: cannot open " + path.string());
    }

    // Total size for plausibility-bounding the var-length section below: a
    // length prefix can never exceed the bytes that remain in the file, so a
    // corrupt or crafted header is rejected before driving any allocation.
    f.seekg(0, std::ios::end);
    const uint64_t file_bytes = static_cast<uint64_t>(f.tellg());
    f.seekg(0, std::ios::beg);

    auto remaining_bytes = [&]() -> uint64_t {
        const auto pos = f.tellg();
        if (pos < 0) return 0;
        const uint64_t upos = static_cast<uint64_t>(pos);
        return upos <= file_bytes ? file_bytes - upos : 0;
    };

    uint8_t magic[8];
    if (!f.read(reinterpret_cast<char*>(magic), 8)) {
        throw std::runtime_error(
            "ModelCheckpoint::read_ckptmeta: cannot read magic from " + path.string());
    }
    if (std::memcmp(magic, CKPTMETA_MAGIC, 8) != 0) {
        throw std::runtime_error(
            "ModelCheckpoint::read_ckptmeta: invalid magic in " + path.string()
            + " (possibly not a checkpoint file or corrupted)");
    }

    uint32_t version = 0;
    read_le(f, version, "format_version");
    if (version != CKPTMETA_VERSION) {
        throw std::runtime_error(
            "ModelCheckpoint::read_ckptmeta: unsupported checkpoint format version "
            + std::to_string(version) + " (expected "
            + std::to_string(CKPTMETA_VERSION) + ") in " + path.string());
    }

    uint32_t save_kind_val = 0;
    read_le(f, save_kind_val, "save_kind");
    if (save_kind_val > 1) {
        throw std::runtime_error(
            "ModelCheckpoint::read_ckptmeta: invalid save_kind "
            + std::to_string(save_kind_val) + " in " + path.string());
    }

    TrainingState s;
    read_le(f, s.epoch,              "epoch");
    read_le(f, s.patience_counter,   "patience_counter");
    read_le(f, s.best_val_accuracy,  "best_val_accuracy");
    read_le(f, s.input_dim,          "input_dim");
    read_le(f, s.hidden_dim,         "hidden_dim");
    read_le(f, s.num_classes,        "num_classes");
    read_le(f, s.num_layers,         "num_layers");
    read_le(f, s.dropout,            "dropout");
    uint8_t norm = 0;
    read_le(f, norm,                 "normalize");
    s.normalize = (norm != 0);
    uint8_t reserved[3];
    if (!f.read(reinterpret_cast<char*>(reserved), 3)) {
        throw std::runtime_error(
            "ModelCheckpoint::read_ckptmeta: truncated reading reserved bytes in "
            + path.string());
    }
    read_le(f, s.total_training_time_sec, "total_training_time_sec");
    read_le(f, s.creation_time_unix,      "creation_time_unix");
    if (!f.read(reinterpret_cast<char*>(s.gnn_meta_hash.data()), 32)) {
        throw std::runtime_error(
            "ModelCheckpoint::read_ckptmeta: truncated reading gnn_meta_hash in "
            + path.string());
    }

    auto read_str = [&](std::string& out, const std::string& ctx) {
        uint32_t len = 0;
        read_le(f, len, ctx + "_len");
        const uint64_t remaining = remaining_bytes();
        if (len > remaining) {
            throw std::runtime_error(
                "ModelCheckpoint::read_ckptmeta: corrupt " + ctx + " length "
                + std::to_string(len) + " exceeds remaining "
                + std::to_string(remaining) + " bytes in " + path.string());
        }
        out.resize(len);
        if (len > 0 && !f.read(out.data(), len)) {
            throw std::runtime_error(
                "ModelCheckpoint::read_ckptmeta: truncated reading " + ctx
                + " in " + path.string());
        }
    };
    read_str(s.projection_name, "projection_name");
    read_str(s.model_type,      "model_type");

    uint32_t k = 0;
    read_le(f, k, "num_epoch_losses");
    if (static_cast<uint64_t>(k) * sizeof(double) > remaining_bytes()) {
        throw std::runtime_error(
            "ModelCheckpoint::read_ckptmeta: corrupt epoch_losses count "
            + std::to_string(k) + " exceeds remaining "
            + std::to_string(remaining_bytes()) + " bytes in " + path.string());
    }
    s.epoch_losses.resize(k);
    if (k > 0 && !f.read(reinterpret_cast<char*>(s.epoch_losses.data()),
                         static_cast<std::streamsize>(k * sizeof(double)))) {
        throw std::runtime_error(
            "ModelCheckpoint::read_ckptmeta: truncated reading epoch_losses in "
            + path.string());
    }

    return s;
}

} // namespace mdb::gnn

// The following anonymous-namespace helper augments the file-level helpers
// already declared in this translation unit.
#include "gnn/models/graphsage_model.h"

#include <unistd.h>      // fsync, close
#include <fcntl.h>       // open, O_RDONLY
#include <cerrno>
#include <system_error>

namespace {

void fsync_dir_impl(const std::filesystem::path& dir) {
    int fd = ::open(dir.c_str(), O_RDONLY);
    if (fd < 0) {
        throw std::runtime_error(
            "ModelCheckpoint: cannot open directory for fsync: " + dir.string()
            + " (errno=" + std::to_string(errno) + ")");
    }
    if (::fsync(fd) != 0) {
        int e = errno;
        ::close(fd);
        throw std::runtime_error(
            "ModelCheckpoint: fsync failed on " + dir.string()
            + " (errno=" + std::to_string(e) + ")");
    }
    ::close(fd);
}

} // anon

namespace mdb::gnn {

void ModelCheckpoint::fsync_directory(const std::filesystem::path& dir) {
    fsync_dir_impl(dir);
}

void ModelCheckpoint::cleanup_tmps(const std::filesystem::path& basename) {
    std::error_code ec;
    std::filesystem::remove(basename.string() + ".pt.tmp",        ec);
    std::filesystem::remove(basename.string() + ".ckptmeta.tmp",  ec);
}

void ModelCheckpoint::save_full(
    const GraphSAGEModel&       model,
    const torch::optim::Adam&   optimizer,
    const std::filesystem::path& basename,
    TrainingState                state)
{
    std::filesystem::create_directories(basename.parent_path());
    cleanup_tmps(basename);

    auto pt_path   = basename.string() + ".pt";
    auto meta_path = basename.string() + ".ckptmeta";
    auto pt_tmp    = pt_path   + ".tmp";
    auto meta_tmp  = meta_path + ".tmp";

    try {
        // --- Write .pt (model + optimizer) via torch archive ---
        {
            torch::serialize::OutputArchive archive;
            model.save(archive);
            // LibTorch's Optimizer::save takes a non-const archive but the
            // optimizer itself is logically const for serialization. The
            // const_cast expresses that this is a read-only access at the
            // ModelCheckpoint API boundary, even though LibTorch's signature
            // predates const-correctness on this path.
            const_cast<torch::optim::Adam&>(optimizer).save(archive);
            archive.save_to(pt_tmp);
        }
        // save_to uses buffered I/O; force the payload to disk so the rename
        // below cannot commit a name whose data is still volatile.
        fsync_file_impl(pt_tmp);

        // --- Write .ckptmeta (always SaveKind::Full via public wrapper) ---
        // write_ckptmeta fsyncs the file data internally.
        write_ckptmeta(meta_tmp, state);

        // .pt first, then .ckptmeta: presence of .ckptmeta signals a
        // fully-committed checkpoint to list_checkpoints() (Task 2.4).
        std::filesystem::rename(pt_tmp, pt_path);
        try {
            std::filesystem::rename(meta_tmp, meta_path);
        } catch (...) {
            // The new .pt is already committed; the surviving old .ckptmeta
            // would silently pair stale training state with the new weights.
            // Remove it so the checkpoint reads as absent (orphan .pt)
            // instead of torn. Callers needing true rollback should save to
            // a fresh basename.
            std::error_code rec;
            std::filesystem::remove(meta_path, rec);
            throw;
        }
        fsync_directory(basename.parent_path());
    }
    catch (...) {
        std::error_code ec;
        std::filesystem::remove(pt_tmp,   ec);
        std::filesystem::remove(meta_tmp, ec);
        throw;
    }
}

TrainingState ModelCheckpoint::load_full(
    GraphSAGEModel&               model,
    torch::optim::Adam&           optimizer,
    const std::filesystem::path&  basename)
{
    auto pt_path   = basename.string() + ".pt";
    auto meta_path = basename.string() + ".ckptmeta";

    if (!std::filesystem::exists(meta_path)) {
        throw std::runtime_error(
            "ModelCheckpoint::load_full: " + meta_path + " not found");
    }
    if (!std::filesystem::exists(pt_path)) {
        throw std::runtime_error(
            "ModelCheckpoint::load_full: " + pt_path + " not found");
    }

    // Weights-only checkpoints lack optimizer state — cannot be used for
    // training resume. load_weights is the correct API for inference reloads.
    if (peek_save_kind(meta_path) != SaveKind::Full) {
        throw std::runtime_error(
            "ModelCheckpoint::load_full: " + meta_path + " is weights-only. "
            "Use load_weights or gnn_predict instead.");
    }

    auto state = read_ckptmeta(meta_path);

    // --- Architecture validation ---
    check_arch_match(state, model.config(), "load_full");

    // --- Load torch archive ---
    torch::serialize::InputArchive archive;
    try {
        archive.load_from(pt_path);
    } catch (const std::exception& e) {
        throw std::runtime_error(
            std::string("ModelCheckpoint::load_full: failed to load .pt: ") + e.what());
    }
    model.load(archive);
    optimizer.load(archive);

    return state;
}

void ModelCheckpoint::save_weights(
    const GraphSAGEModel&       model,
    const std::filesystem::path& basename,
    TrainingState                state)
{
    std::filesystem::create_directories(basename.parent_path());
    cleanup_tmps(basename);

    // Weights-only checkpoints carry no loss history
    state.epoch_losses.clear();

    auto pt_path   = basename.string() + ".pt";
    auto meta_path = basename.string() + ".ckptmeta";
    auto pt_tmp    = pt_path   + ".tmp";
    auto meta_tmp  = meta_path + ".tmp";

    try {
        {
            torch::serialize::OutputArchive archive;
            model.save(archive);           // ONLY weights — no optimizer
            archive.save_to(pt_tmp);
        }
        // save_to uses buffered I/O; force the payload to disk so the rename
        // below cannot commit a name whose data is still volatile.
        fsync_file_impl(pt_tmp);

        // write_ckptmeta_impl fsyncs the file data internally.
        write_ckptmeta_impl(meta_tmp, state, SaveKind::WeightsOnly);

        // .pt first, then .ckptmeta: presence of .ckptmeta signals a
        // fully-committed checkpoint to list_checkpoints() (Task 2.4).
        std::filesystem::rename(pt_tmp, pt_path);
        try {
            std::filesystem::rename(meta_tmp, meta_path);
        } catch (...) {
            // The new .pt is already committed; the surviving old .ckptmeta
            // would silently pair stale training state with the new weights.
            // Remove it so the checkpoint reads as absent (orphan .pt)
            // instead of torn. Callers needing true rollback should save to
            // a fresh basename.
            std::error_code rec;
            std::filesystem::remove(meta_path, rec);
            throw;
        }
        fsync_directory(basename.parent_path());
    }
    catch (...) {
        std::error_code ec;
        std::filesystem::remove(pt_tmp,   ec);
        std::filesystem::remove(meta_tmp, ec);
        throw;
    }
}

TrainingState ModelCheckpoint::load_weights(
    GraphSAGEModel&               model,
    const std::filesystem::path&  basename)
{
    auto pt_path   = basename.string() + ".pt";
    auto meta_path = basename.string() + ".ckptmeta";

    if (!std::filesystem::exists(meta_path)) {
        throw std::runtime_error(
            "ModelCheckpoint::load_weights: " + meta_path + " not found");
    }
    if (!std::filesystem::exists(pt_path)) {
        throw std::runtime_error(
            "ModelCheckpoint::load_weights: " + pt_path + " not found");
    }

    auto state = read_ckptmeta(meta_path);

    // Architecture validation (same as load_full)
    check_arch_match(state, model.config(), "load_weights");

    // Load torch archive — only model weights are read. If the .pt was produced
    // by save_full, the optimizer section is present but we ignore it.
    torch::serialize::InputArchive archive;
    try {
        archive.load_from(pt_path);
    } catch (const std::exception& e) {
        // Wrap LibTorch c10::Error with ModelCheckpoint context
        throw std::runtime_error(
            std::string("ModelCheckpoint::load_weights: failed to load .pt: ") + e.what());
    }
    model.load(archive);

    return state;
}

void ModelCheckpoint::check_arch_match(
    const TrainingState&       state,
    const GraphSAGEConfig&     cfg,
    const char*                method_name)
{
    if (state.input_dim   != cfg.input_dim   ||
        state.hidden_dim  != cfg.hidden_dim  ||
        state.num_classes != cfg.num_classes ||
        state.num_layers  != cfg.num_layers)
    {
        throw std::runtime_error(
            std::string("ModelCheckpoint::") + method_name + ": architecture mismatch. "
            "Checkpoint=(input=" + std::to_string(state.input_dim)
            + ", hidden=" + std::to_string(state.hidden_dim)
            + ", classes=" + std::to_string(state.num_classes)
            + ", layers=" + std::to_string(state.num_layers) + ") vs "
            "Model=(input=" + std::to_string(cfg.input_dim)
            + ", hidden=" + std::to_string(cfg.hidden_dim)
            + ", classes=" + std::to_string(cfg.num_classes)
            + ", layers=" + std::to_string(cfg.num_layers) + ")");
    }
}

void ModelCheckpoint::validate_compat(
    const TrainingState&          state,
    const std::filesystem::path&  current_gnn_meta_path,
    const std::string&            current_projection_name)
{
    if (state.projection_name != current_projection_name) {
        throw std::runtime_error(
            "ModelCheckpoint::validate_compat: checkpoint was trained on projection '"
            + state.projection_name + "' but current is '"
            + current_projection_name + "'. Cannot reuse across projections.");
    }

    auto current_hash = compute_gnn_meta_hash(current_gnn_meta_path);
    if (current_hash != state.gnn_meta_hash) {
        throw std::runtime_error(
            "ModelCheckpoint::validate_compat: gnn_meta.bin hash mismatch. "
            "The projection may have been regenerated with different features. "
            "Recreate the projection with the original feature set, or train a new model.");
    }
}

bool ModelCheckpoint::exists(const std::filesystem::path& basename)
{
    return std::filesystem::exists(basename.string() + ".pt")
        && std::filesystem::exists(basename.string() + ".ckptmeta");
}

void ModelCheckpoint::delete_checkpoint(const std::filesystem::path& basename)
{
    // Delete both committed files AND any orphan .tmp siblings so a user-
    // initiated delete is fully clean (Task 2.3 review fold-in).
    auto remove_or_throw = [](const std::filesystem::path& p, const char* which) {
        std::error_code ec;
        std::filesystem::remove(p, ec);
        if (ec && ec != std::errc::no_such_file_or_directory) {
            throw std::runtime_error(
                std::string("ModelCheckpoint::delete_checkpoint: failed to remove ")
                + which + " (" + p.string() + "): " + ec.message());
        }
    };

    remove_or_throw(basename.string() + ".pt",            ".pt");
    remove_or_throw(basename.string() + ".ckptmeta",      ".ckptmeta");
    remove_or_throw(basename.string() + ".pt.tmp",        ".pt.tmp");
    remove_or_throw(basename.string() + ".ckptmeta.tmp",  ".ckptmeta.tmp");
}

std::vector<CheckpointInfo> ModelCheckpoint::list_checkpoints(
    const std::filesystem::path&      dir,
    std::optional<std::string>        name_filter)
{
    std::vector<CheckpointInfo> out;
    std::error_code ec;
    if (!std::filesystem::exists(dir, ec) || !std::filesystem::is_directory(dir, ec)) {
        return out;
    }

    for (const auto& entry : std::filesystem::directory_iterator(dir)) {
        if (!entry.is_regular_file()) continue;
        const auto& p = entry.path();
        if (p.extension() != ".ckptmeta") continue;

        auto basename = p;
        basename.replace_extension();  // strip .ckptmeta
        auto pt_path = basename.string() + ".pt";
        if (!std::filesystem::exists(pt_path)) continue;  // orphan .ckptmeta

        auto fname = basename.filename().string();
        if (name_filter && *name_filter != fname) continue;

        CheckpointInfo info;
        info.basename = std::filesystem::absolute(basename);
        try {
            // peek_save_kind is in the file-scope anonymous namespace; unqualified
            // name lookup from inside namespace mdb::gnn finds it at TU scope.
            info.save_kind = peek_save_kind(p);
            auto s = read_ckptmeta(p);
            info.epoch               = s.epoch;
            info.best_val_accuracy   = s.best_val_accuracy;
            info.creation_time_unix  = s.creation_time_unix;
            info.model_type          = s.model_type;
            info.projection_name     = s.projection_name;
            info.pt_bytes   = std::filesystem::file_size(pt_path);
            info.meta_bytes = std::filesystem::file_size(p);
        } catch (const std::exception&) {
            continue;  // skip malformed files silently
        }
        out.push_back(std::move(info));
    }

    std::sort(out.begin(), out.end(),
              [](const CheckpointInfo& a, const CheckpointInfo& b) {
                  return a.creation_time_unix > b.creation_time_unix;
              });
    return out;
}

} // namespace mdb::gnn
