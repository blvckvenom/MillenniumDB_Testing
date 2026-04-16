#include "gnn/output/model_checkpoint.h"

#include <openssl/evp.h>

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

void ModelCheckpoint::write_ckptmeta(
    const std::filesystem::path& path,
    const TrainingState&         s)
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

    // save_kind default: Full (1). save_full / save_weights wrappers (Tasks 2.x)
    // will override via a private overload. For the present task the low-level
    // public entry point always writes Full.
    uint32_t save_kind_val = static_cast<uint32_t>(SaveKind::Full);
    write_le(f, save_kind_val);

    // Fixed-size fields.  Offsets match the binary layout in the spec.
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

    if (!f) {
        throw std::runtime_error(
            "ModelCheckpoint::write_ckptmeta: I/O error writing " + path.string());
    }
}

TrainingState ModelCheckpoint::read_ckptmeta(
    const std::filesystem::path& path)
{
    std::ifstream f(path, std::ios::binary);
    if (!f) {
        throw std::runtime_error(
            "ModelCheckpoint::read_ckptmeta: cannot open " + path.string());
    }

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
