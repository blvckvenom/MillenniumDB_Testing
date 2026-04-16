#include "gnn/output/model_checkpoint.h"

#include <openssl/evp.h>

#include <fstream>
#include <stdexcept>

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

} // namespace mdb::gnn
