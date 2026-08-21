#include "graph_models/gql/projection/spill_codec.h"

#include <algorithm>
#include <atomic>
#include <cctype>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <mutex>
#include <stdexcept>

#include "misc/ablation_registry.h"

#ifdef HAS_LZ4
  #include <lz4frame.h>
#endif

namespace GQL {

// ---------------------------------------------------------------------------
// Env + default resolution
// ---------------------------------------------------------------------------
//
// Cached after the first call. The cache is justified because the env var is
// read once at process startup and all projection builds in the same server
// must agree on the policy (mixing formats across a single build would make
// reads nondeterministic). Reading the env on each call would also force
// expensive locking in inner loops.
namespace {

std::atomic<bool>             s_resolved_cached{false};
std::atomic<SpillCompression> s_resolved_value{SpillCompression::NONE};
std::once_flag                s_warning_once;

std::string to_lower(std::string s) {
    for (char& c : s) c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
    return s;
}

SpillCompression compiled_default() {
#ifdef HAS_LZ4
    return SpillCompression::LZ4;
#else
    return SpillCompression::NONE;
#endif
}

void maybe_warn_invalid(const std::string& raw_value) {
    std::call_once(s_warning_once, [&]() {
        std::cerr << "[spill_codec] MDB_SPILL_COMPRESSION='" << raw_value
                  << "' is invalid; falling back to default. "
                     "Valid values: 'none', 'lz4'.\n";
    });
}

} // anonymous namespace

SpillCompression resolve_default_spill_compression() {
    if (s_resolved_cached.load(std::memory_order_acquire)) {
        return s_resolved_value.load(std::memory_order_relaxed);
    }

    // text() and not choice(): the comparison below is case-INSENSITIVE, so a
    // validated list would reject "LZ4", which works today. The switch is still
    // declared, and an invalid value keeps its existing stderr warning.
    const std::string env = Ablation::text("MDB_SPILL_COMPRESSION", "");
    SpillCompression resolved = compiled_default();

    if (!env.empty()) {
        std::string v = to_lower(env);
        if (v == "none" || v == "off" || v == "0") {
            resolved = SpillCompression::NONE;
        } else if (v == "lz4") {
#ifdef HAS_LZ4
            resolved = SpillCompression::LZ4;
#else
            // User asked for LZ4 but binary lacks liblz4 — warn + fall back.
            std::call_once(s_warning_once, []() {
                std::cerr << "[spill_codec] MDB_SPILL_COMPRESSION=lz4 requested "
                             "but this build has no LZ4 support. Using NONE.\n";
            });
            resolved = SpillCompression::NONE;
#endif
        } else {
            maybe_warn_invalid(env);
            // resolved remains compiled_default()
        }
    }

    s_resolved_value.store(resolved, std::memory_order_relaxed);
    s_resolved_cached.store(true, std::memory_order_release);
    return resolved;
}

// ---------------------------------------------------------------------------
// Internal helpers
// ---------------------------------------------------------------------------
namespace {

void write_header(std::ofstream& out, SpillCompression compression) {
    // Layout is little-endian on x86_64; we write byte-by-byte to be portable.
    uint8_t hdr[SpillFormat::HEADER_SIZE];
    hdr[0] = static_cast<uint8_t>(SpillFormat::MAGIC & 0xFF);
    hdr[1] = static_cast<uint8_t>((SpillFormat::MAGIC >> 8)  & 0xFF);
    hdr[2] = static_cast<uint8_t>((SpillFormat::MAGIC >> 16) & 0xFF);
    hdr[3] = static_cast<uint8_t>((SpillFormat::MAGIC >> 24) & 0xFF);
    hdr[4] = SpillFormat::VERSION;
    hdr[5] = static_cast<uint8_t>(compression);
    hdr[6] = 0; // reserved
    hdr[7] = 0;
    out.write(reinterpret_cast<const char*>(hdr), SpillFormat::HEADER_SIZE);
    if (!out) {
        throw std::runtime_error("SpillWriter: failed to write header");
    }
}

#ifdef HAS_LZ4
// Creates LZ4 frame preferences. We use a moderate block size (64 KB) and
// enable block checksums off (spill files are short-lived within a build;
// end-to-end checksums are the DB's responsibility via B+Tree on the final
// output, not on the ephemeral intermediate).
LZ4F_preferences_t make_lz4_prefs() {
    LZ4F_preferences_t prefs{};
    prefs.compressionLevel            = 0;                         // fastest
    prefs.frameInfo.blockSizeID       = LZ4F_max64KB;
    prefs.frameInfo.blockMode         = LZ4F_blockLinked;          // better ratio
    prefs.frameInfo.contentChecksumFlag = LZ4F_noContentChecksum;  // speed
    prefs.frameInfo.blockChecksumFlag  = LZ4F_noBlockChecksum;
    prefs.autoFlush                    = 0;                        // let lz4 pick
    return prefs;
}
#endif

} // anonymous namespace

// ---------------------------------------------------------------------------
// SpillWriter
// ---------------------------------------------------------------------------

SpillWriter::SpillWriter(const std::string& path, SpillCompression compression)
    : path_(path)
    , compression_(compression)
{
    // Validate + graceful fallback for LZ4 without build support. Documented
    // semantics: the writer accepts LZ4 but silently records NONE in the
    // header if LZ4 is not built in. This preserves the invariant "every
    // file written is readable by any SpillReader" without crashing.
#ifndef HAS_LZ4
    if (compression_ == SpillCompression::LZ4) {
        compression_ = SpillCompression::NONE;
    }
#endif

    out_.open(path_, std::ios::binary | std::ios::trunc);
    if (!out_) {
        throw std::runtime_error("SpillWriter: cannot open for writing: " + path_);
    }

    write_header(out_, compression_);

#ifdef HAS_LZ4
    if (compression_ == SpillCompression::LZ4) {
        LZ4F_cctx* ctx = nullptr;
        LZ4F_errorCode_t err = LZ4F_createCompressionContext(&ctx, LZ4F_VERSION);
        if (LZ4F_isError(err)) {
            throw std::runtime_error(
                std::string("SpillWriter: LZ4F_createCompressionContext failed: ") +
                LZ4F_getErrorName(err));
        }
        lz4_ctx_ = ctx;

        // Emit LZ4 frame header (independent from our 8-byte magic header).
        // The LZ4 frame header is small (~7-15 bytes) and describes block size
        // and content checksum for the decompressor.
        auto prefs = make_lz4_prefs();
        compress_buffer_.resize(LZ4F_compressBound(64 * 1024, &prefs) + 64);
        size_t written = LZ4F_compressBegin(ctx,
                                            compress_buffer_.data(),
                                            compress_buffer_.size(),
                                            &prefs);
        if (LZ4F_isError(written)) {
            throw std::runtime_error(
                std::string("SpillWriter: LZ4F_compressBegin failed: ") +
                LZ4F_getErrorName(written));
        }
        out_.write(reinterpret_cast<const char*>(compress_buffer_.data()),
                   static_cast<std::streamsize>(written));
        if (!out_) {
            throw std::runtime_error("SpillWriter: failed to write LZ4 frame begin");
        }
    }
#endif
}

SpillWriter::~SpillWriter() {
    // Swallow exceptions in destructor to preserve no-throw guarantee, but
    // log so users see failure if finalize() wasn't called explicitly.
    try {
        finalize();
    } catch (const std::exception& e) {
        std::cerr << "[SpillWriter] finalize failed in dtor: " << e.what()
                  << " (path=" << path_ << ")\n";
    }

#ifdef HAS_LZ4
    if (lz4_ctx_) {
        LZ4F_freeCompressionContext(static_cast<LZ4F_cctx*>(lz4_ctx_));
        lz4_ctx_ = nullptr;
    }
#endif
}

void SpillWriter::write(const void* data, std::size_t bytes) {
    if (finalized_) {
        throw std::runtime_error("SpillWriter: write after finalize: " + path_);
    }
    if (bytes == 0) return;

    if (compression_ == SpillCompression::NONE) {
        out_.write(static_cast<const char*>(data), static_cast<std::streamsize>(bytes));
        if (!out_) {
            throw std::runtime_error("SpillWriter: raw write failed: " + path_);
        }
        return;
    }

#ifdef HAS_LZ4
    // Chunked compression loop: LZ4F_compressUpdate takes arbitrary sizes,
    // but internally splits into blocks bounded by the frame's blockSizeID.
    // We pass the full remaining range each call; LZ4 handles partial
    // consumption and internal buffering transparently.
    auto prefs = make_lz4_prefs();
    auto* ctx  = static_cast<LZ4F_cctx*>(lz4_ctx_);

    const uint8_t* src = static_cast<const uint8_t*>(data);
    std::size_t    left = bytes;

    while (left > 0) {
        // Process in blocks of up to 64 KB so compress_buffer_ sizing is bounded.
        std::size_t take = std::min<std::size_t>(left, 64 * 1024);
        std::size_t need = LZ4F_compressBound(take, &prefs);
        if (compress_buffer_.size() < need + 64) {
            compress_buffer_.resize(need + 64);
        }

        std::size_t produced = LZ4F_compressUpdate(ctx,
                                                   compress_buffer_.data(),
                                                   compress_buffer_.size(),
                                                   src,
                                                   take,
                                                   nullptr);
        if (LZ4F_isError(produced)) {
            throw std::runtime_error(
                std::string("SpillWriter: LZ4F_compressUpdate failed: ") +
                LZ4F_getErrorName(produced));
        }
        if (produced > 0) {
            out_.write(reinterpret_cast<const char*>(compress_buffer_.data()),
                       static_cast<std::streamsize>(produced));
            if (!out_) {
                throw std::runtime_error("SpillWriter: compressed write failed: " + path_);
            }
        }
        src  += take;
        left -= take;
    }
#else
    // Should be unreachable because ctor demotes LZ4 to NONE when !HAS_LZ4.
    (void)data; (void)bytes;
    throw std::runtime_error("SpillWriter: LZ4 requested but not compiled in");
#endif
}

void SpillWriter::finalize() {
    if (finalized_) return;
    finalized_ = true;

#ifdef HAS_LZ4
    if (compression_ == SpillCompression::LZ4 && lz4_ctx_) {
        auto* ctx = static_cast<LZ4F_cctx*>(lz4_ctx_);
        std::size_t need = LZ4F_compressBound(0, nullptr) + 64;
        if (compress_buffer_.size() < need) {
            compress_buffer_.resize(need);
        }
        std::size_t produced = LZ4F_compressEnd(ctx,
                                                compress_buffer_.data(),
                                                compress_buffer_.size(),
                                                nullptr);
        if (LZ4F_isError(produced)) {
            throw std::runtime_error(
                std::string("SpillWriter: LZ4F_compressEnd failed: ") +
                LZ4F_getErrorName(produced));
        }
        if (produced > 0) {
            out_.write(reinterpret_cast<const char*>(compress_buffer_.data()),
                       static_cast<std::streamsize>(produced));
        }
    }
#endif

    out_.flush();
    out_.close();
}

// ---------------------------------------------------------------------------
// SpillReader
// ---------------------------------------------------------------------------
//
// Why the two-stage detection: spills are normally short-lived (same build
// creates and consumes them), so we could rely on always-new format. But a
// small cost in reader logic (check magic → seek back) buys robustness
// against any edge case where a partially-written legacy file is read, and
// makes the module safe to integrate incrementally without coordinating
// writer/reader upgrades.

SpillReader::SpillReader(const std::string& path)
    : path_(path)
{
    in_.open(path_, std::ios::binary);
    if (!in_) {
        throw std::runtime_error("SpillReader: cannot open for reading: " + path_);
    }

    // Try to read the 8-byte header and validate magic.
    uint8_t hdr[SpillFormat::HEADER_SIZE];
    in_.read(reinterpret_cast<char*>(hdr), SpillFormat::HEADER_SIZE);
    std::streamsize got = in_.gcount();

    bool header_ok = false;
    if (got == static_cast<std::streamsize>(SpillFormat::HEADER_SIZE)) {
        uint32_t magic = static_cast<uint32_t>(hdr[0])
                       | (static_cast<uint32_t>(hdr[1]) << 8)
                       | (static_cast<uint32_t>(hdr[2]) << 16)
                       | (static_cast<uint32_t>(hdr[3]) << 24);
        uint8_t  ver   = hdr[4];
        uint8_t  ctype = hdr[5];
        if (magic == SpillFormat::MAGIC && ver == SpillFormat::VERSION
            && (ctype == 0 || ctype == 1))
        {
            header_ok = true;
            compression_ = static_cast<SpillCompression>(ctype);
        }
    }

    if (!header_ok) {
        // Legacy: rewind and treat as raw NONE.
        in_.clear(); // clear eof/fail bits from short read
        in_.seekg(0, std::ios::beg);
        if (!in_) {
            throw std::runtime_error("SpillReader: failed to seek back on legacy detect: " + path_);
        }
        legacy_format_ = true;
        compression_   = SpillCompression::NONE;
    }

#ifdef HAS_LZ4
    if (compression_ == SpillCompression::LZ4) {
        LZ4F_dctx* ctx = nullptr;
        LZ4F_errorCode_t err = LZ4F_createDecompressionContext(&ctx, LZ4F_VERSION);
        if (LZ4F_isError(err)) {
            throw std::runtime_error(
                std::string("SpillReader: LZ4F_createDecompressionContext failed: ") +
                LZ4F_getErrorName(err));
        }
        lz4_ctx_ = ctx;
        // Reserve capacity for staging. We use size=0 with reserved capacity
        // so read() can't accidentally consume zero-filled pre-sized bytes as
        // valid data. pump_chunk() sets the actual size when it reads.
        compressed_in_.reserve(64 * 1024);
        compressed_in_.resize(64 * 1024);   // we read INTO this via in_.read()
    }
#endif
    // decompressed_ starts empty (size 0). pump_chunk() is the only writer.
    // This avoids a subtle bug where a pre-sized decompressed_ would make
    // read() serve zero-filled bytes on the first call before pump_chunk.
    (void)0;
}

SpillReader::~SpillReader() {
#ifdef HAS_LZ4
    if (lz4_ctx_) {
        LZ4F_freeDecompressionContext(static_cast<LZ4F_dctx*>(lz4_ctx_));
        lz4_ctx_ = nullptr;
    }
#endif
}

std::size_t SpillReader::read(void* dst, std::size_t max_bytes) {
    if (eof_ || max_bytes == 0) return 0;

    std::size_t   written = 0;
    auto*         out_ptr = static_cast<uint8_t*>(dst);

    while (written < max_bytes) {
        // Serve from decompressed buffer first.
        if (decompressed_pos_ < decompressed_.size()) {
            std::size_t avail = decompressed_.size() - decompressed_pos_;
            std::size_t take  = std::min(avail, max_bytes - written);
            std::memcpy(out_ptr + written, decompressed_.data() + decompressed_pos_, take);
            decompressed_pos_ += take;
            written += take;
            continue;
        }

        // Buffer empty — refill.
        if (!pump_chunk()) {
            eof_ = true;
            break;
        }
    }
    return written;
}

bool SpillReader::pump_chunk() {
    // Reset buffer for next fill.
    decompressed_.clear();
    decompressed_pos_ = 0;

    if (compression_ == SpillCompression::NONE) {
        // Raw path: read up to 256 KB from file.
        decompressed_.resize(256 * 1024);
        in_.read(reinterpret_cast<char*>(decompressed_.data()),
                 static_cast<std::streamsize>(decompressed_.size()));
        std::streamsize got = in_.gcount();
        if (got <= 0) {
            decompressed_.clear();
            return false;
        }
        decompressed_.resize(static_cast<std::size_t>(got));
        return true;
    }

#ifdef HAS_LZ4
    auto* ctx = static_cast<LZ4F_dctx*>(lz4_ctx_);

    decompressed_.resize(256 * 1024);
    std::size_t out_capacity = decompressed_.size();
    std::size_t out_used     = 0;

    // We loop until we produce something OR we hit end-of-frame with no output.
    while (out_used == 0) {
        // Refill compressed input buffer if empty.
        if (compressed_pos_ >= compressed_len_) {
            if (file_eof_reached_) {
                // No more input; frame likely complete.
                decompressed_.clear();
                return false;
            }
            in_.read(reinterpret_cast<char*>(compressed_in_.data()),
                     static_cast<std::streamsize>(compressed_in_.size()));
            std::streamsize got = in_.gcount();
            if (got <= 0) {
                file_eof_reached_ = true;
                // Try one last decompress call with zero input to flush.
                std::size_t dst_size = out_capacity;
                std::size_t src_size = 0;
                std::size_t hint = LZ4F_decompress(ctx,
                                                   decompressed_.data(), &dst_size,
                                                   compressed_in_.data(), &src_size,
                                                   nullptr);
                if (LZ4F_isError(hint)) {
                    throw std::runtime_error(
                        std::string("SpillReader: LZ4F_decompress (flush) failed: ") +
                        LZ4F_getErrorName(hint));
                }
                out_used = dst_size;
                if (out_used == 0) {
                    decompressed_.clear();
                    return false;
                }
                break;
            }
            compressed_len_ = static_cast<std::size_t>(got);
            compressed_pos_ = 0;
        }

        std::size_t dst_size = out_capacity - out_used;
        std::size_t src_size = compressed_len_ - compressed_pos_;
        std::size_t hint = LZ4F_decompress(ctx,
                                           decompressed_.data() + out_used, &dst_size,
                                           compressed_in_.data() + compressed_pos_, &src_size,
                                           nullptr);
        if (LZ4F_isError(hint)) {
            throw std::runtime_error(
                std::string("SpillReader: LZ4F_decompress failed: ") +
                LZ4F_getErrorName(hint));
        }
        out_used        += dst_size;
        compressed_pos_ += src_size;

        if (hint == 0 && out_used == 0) {
            // Frame ended without output — we're done.
            decompressed_.clear();
            return false;
        }
    }

    decompressed_.resize(out_used);
    return true;
#else
    // Unreachable: compression NONE served above.
    return false;
#endif
}

} // namespace GQL
