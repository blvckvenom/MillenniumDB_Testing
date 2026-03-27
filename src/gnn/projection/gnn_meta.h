#pragma once

#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <string>

namespace mdb::gnn {

/**
 * @brief Projection-level GNN metadata written by graph_project and consumed by gnn_train.
 *
 * Binary file format (gnn_meta.bin), variable-length due to feature_name:
 *
 * Offset  Size  Field
 *  0       8    magic: "GNNM\0\0\0\0"
 *  8       4    version: uint32 (1)
 * 12       4    feature_dim: uint32
 * 16       8    num_nodes: uint64
 * 24       8    num_classes: uint64
 * 32       1    has_labels: uint8 (0/1)
 * 33       1    has_splits: uint8 (0/1)
 * 34       2    reserved (zeros)
 * 36       4    feature_name_len: uint32
 * 40       N    feature_name: char[N]
 *
 * Total fixed portion: 40 bytes + N bytes for feature_name.
 */
struct GnnMeta {
    // -------------------------------------------------------------------------
    // Constants
    // -------------------------------------------------------------------------
    static constexpr uint8_t  MAGIC[8]  = {'G','N','N','M','\0','\0','\0','\0'};
    static constexpr uint32_t VERSION   = 1;

    // -------------------------------------------------------------------------
    // Fields
    // -------------------------------------------------------------------------
    std::string feature_name;
    uint32_t    feature_dim  = 0;
    uint64_t    num_nodes    = 0;
    uint64_t    num_classes  = 0;
    bool        has_labels   = false;
    bool        has_splits   = false;

    // -------------------------------------------------------------------------
    // read() — deserialize from file, throws std::runtime_error on failure
    // -------------------------------------------------------------------------
    static GnnMeta read(const std::filesystem::path& path) {
        std::ifstream f(path, std::ios::binary);
        if (!f.is_open()) {
            throw std::runtime_error("GnnMeta::read: cannot open file: " + path.string());
        }

        // --- magic (8 bytes) ---
        uint8_t magic_buf[8];
        if (!f.read(reinterpret_cast<char*>(magic_buf), 8)) {
            throw std::runtime_error("GnnMeta::read: failed to read magic in: " + path.string());
        }
        if (std::memcmp(magic_buf, MAGIC, 8) != 0) {
            throw std::runtime_error("GnnMeta::read: invalid magic in: " + path.string());
        }

        // --- version (4 bytes) ---
        uint32_t version = 0;
        if (!f.read(reinterpret_cast<char*>(&version), 4)) {
            throw std::runtime_error("GnnMeta::read: failed to read version in: " + path.string());
        }
        if (version != VERSION) {
            throw std::runtime_error(
                "GnnMeta::read: unsupported version " + std::to_string(version)
                + " (expected " + std::to_string(VERSION) + ") in: " + path.string()
            );
        }

        GnnMeta m;

        // --- feature_dim (4 bytes) ---
        if (!f.read(reinterpret_cast<char*>(&m.feature_dim), 4)) {
            throw std::runtime_error("GnnMeta::read: failed to read feature_dim in: " + path.string());
        }

        // --- num_nodes (8 bytes) ---
        if (!f.read(reinterpret_cast<char*>(&m.num_nodes), 8)) {
            throw std::runtime_error("GnnMeta::read: failed to read num_nodes in: " + path.string());
        }

        // --- num_classes (8 bytes) ---
        if (!f.read(reinterpret_cast<char*>(&m.num_classes), 8)) {
            throw std::runtime_error("GnnMeta::read: failed to read num_classes in: " + path.string());
        }

        // --- has_labels (1 byte) ---
        uint8_t has_labels_raw = 0;
        if (!f.read(reinterpret_cast<char*>(&has_labels_raw), 1)) {
            throw std::runtime_error("GnnMeta::read: failed to read has_labels in: " + path.string());
        }
        m.has_labels = (has_labels_raw != 0);

        // --- has_splits (1 byte) ---
        uint8_t has_splits_raw = 0;
        if (!f.read(reinterpret_cast<char*>(&has_splits_raw), 1)) {
            throw std::runtime_error("GnnMeta::read: failed to read has_splits in: " + path.string());
        }
        m.has_splits = (has_splits_raw != 0);

        // --- reserved (2 bytes, ignored) ---
        uint8_t reserved[2];
        if (!f.read(reinterpret_cast<char*>(reserved), 2)) {
            throw std::runtime_error("GnnMeta::read: failed to read reserved bytes in: " + path.string());
        }

        // --- feature_name_len (4 bytes) ---
        uint32_t name_len = 0;
        if (!f.read(reinterpret_cast<char*>(&name_len), 4)) {
            throw std::runtime_error("GnnMeta::read: failed to read feature_name_len in: " + path.string());
        }

        // --- feature_name (N bytes) ---
        if (name_len > 0) {
            m.feature_name.resize(name_len);
            if (!f.read(m.feature_name.data(), name_len)) {
                throw std::runtime_error("GnnMeta::read: failed to read feature_name in: " + path.string());
            }
        }

        return m;
    }

    // -------------------------------------------------------------------------
    // write() — serialize to file, creates parent dirs if needed
    // -------------------------------------------------------------------------
    void write(const std::filesystem::path& path) const {
        // Create parent directories if they do not exist
        auto parent = path.parent_path();
        if (!parent.empty()) {
            std::filesystem::create_directories(parent);
        }

        std::ofstream f(path, std::ios::binary | std::ios::trunc);
        if (!f.is_open()) {
            throw std::runtime_error("GnnMeta::write: cannot open file for writing: " + path.string());
        }

        // --- magic (8 bytes) ---
        f.write(reinterpret_cast<const char*>(MAGIC), 8);

        // --- version (4 bytes) ---
        uint32_t ver = VERSION;
        f.write(reinterpret_cast<const char*>(&ver), 4);

        // --- feature_dim (4 bytes) ---
        f.write(reinterpret_cast<const char*>(&feature_dim), 4);

        // --- num_nodes (8 bytes) ---
        f.write(reinterpret_cast<const char*>(&num_nodes), 8);

        // --- num_classes (8 bytes) ---
        f.write(reinterpret_cast<const char*>(&num_classes), 8);

        // --- has_labels (1 byte) ---
        uint8_t hl = has_labels ? 1 : 0;
        f.write(reinterpret_cast<const char*>(&hl), 1);

        // --- has_splits (1 byte) ---
        uint8_t hs = has_splits ? 1 : 0;
        f.write(reinterpret_cast<const char*>(&hs), 1);

        // --- reserved (2 bytes, zeros) ---
        uint8_t reserved[2] = {0, 0};
        f.write(reinterpret_cast<const char*>(reserved), 2);

        // --- feature_name_len (4 bytes) ---
        auto name_len = static_cast<uint32_t>(feature_name.size());
        f.write(reinterpret_cast<const char*>(&name_len), 4);

        // --- feature_name (N bytes) ---
        if (name_len > 0) {
            f.write(feature_name.data(), name_len);
        }

        if (!f) {
            throw std::runtime_error("GnnMeta::write: I/O error while writing: " + path.string());
        }
    }

    // -------------------------------------------------------------------------
    // exists() — checks whether dir/gnn_meta.bin is present
    // -------------------------------------------------------------------------
    static bool exists(const std::filesystem::path& dir) {
        return std::filesystem::exists(dir / "gnn_meta.bin");
    }
};

// The constexpr static member must be defined at namespace scope in C++14,
// but in C++17 constexpr statics are implicitly inline — no out-of-line
// definition is required. The declaration above is sufficient.

} // namespace mdb::gnn
