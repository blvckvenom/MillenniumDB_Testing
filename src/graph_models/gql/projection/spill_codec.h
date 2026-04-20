#pragma once

// Why this module exists:
// -----------------------
// External-sort spill files written by StreamingRecordBuffer are sorted
// arrays of `uint64_t` tuples. They compress very well (3-5x with LZ4) due
// to prefix redundancy in sorted integer keys and the narrow value range of
// ObjectIds. For billion-scale graphs (papers100M, MAG240M) the spill peak
// dominates disk pressure during projection build. Compressing spills
// transparently reduces peak disk by ~3-4x without touching query paths,
// B+Tree layout, or the projection ABI — spills are ephemeral by design.
//
// See docs/superpowers/thesis_analysis/2026-04-20-projection-disk-reduction-analysis.md
// for root-cause analysis, design rationale, and safety matrix.

#include <cstddef>
#include <cstdint>
#include <fstream>
#include <string>
#include <vector>

namespace GQL {

/// Compression algorithm used for spill files.
///
/// Sorted uint64 tuples (graph edge records) compress 3-5x with LZ4 in
/// practice. LZ4 was chosen over zstd because decompress speed exceeds
/// NVMe read throughput — compression is effectively free for I/O-bound
/// spill reads. See analysis doc §3.B for the full justification.
enum class SpillCompression : uint8_t {
    NONE = 0,  ///< Raw bytes. Always supported; legacy-compatible output.
    LZ4  = 1,  ///< LZ4 frame format. Requires HAS_LZ4 (liblz4) at build time.
};

/// On-disk spill file format (written when compression != NONE OR when
/// explicit header mode is requested):
///
///   offset  size  field
///   0       4     magic = 0x4C505347 ('GSPL' little-endian)
///   4       1     version = 1
///   5       1     compression_type (0=NONE, 1=LZ4)
///   6       2     reserved (must be 0)
///   8       ...   payload
///
/// When compression_type == NONE, payload is raw record bytes.
/// When compression_type == LZ4, payload is an LZ4 frame.
///
/// Legacy files without the header are still readable by SpillReader: if
/// the first 8 bytes don't match magic+version+valid-compression, the file
/// is re-read from offset 0 as raw bytes (NONE). This keeps compatibility
/// with pre-existing spills during incremental rollout, even though spills
/// are normally ephemeral within a single projection build.
struct SpillFormat {
    static constexpr uint32_t MAGIC       = 0x4C505347u; // 'GSPL' LE
    static constexpr uint8_t  VERSION     = 1u;
    static constexpr size_t   HEADER_SIZE = 8u;
};

/// Resolves the default compression algorithm from environment + build config.
///
/// Decision order:
///   1. Env var MDB_SPILL_COMPRESSION = "none" | "lz4" (case-insensitive)
///   2. Compiled-in default: LZ4 if HAS_LZ4 is defined, else NONE
///
/// Invalid env values fall back to default and print a warning once.
/// This function is thread-safe and cached after first call.
SpillCompression resolve_default_spill_compression();

/// Streaming writer for spill files with optional compression.
///
/// Construction writes the 8-byte header immediately. Subsequent `write()`
/// calls append data (buffered internally if compressed). `finalize()` (or
/// the destructor) flushes pending data and closes the file.
///
/// Not thread-safe. Not copyable.
class SpillWriter {
public:
    SpillWriter(const std::string& path, SpillCompression compression);
    ~SpillWriter();

    SpillWriter(const SpillWriter&)            = delete;
    SpillWriter& operator=(const SpillWriter&) = delete;
    SpillWriter(SpillWriter&&)                 = delete;
    SpillWriter& operator=(SpillWriter&&)      = delete;

    /// Appends `bytes` bytes from `data` to the spill.
    /// For LZ4, data is buffered and compressed in chunks internally.
    /// Throws std::runtime_error on I/O or compression failure.
    void write(const void* data, std::size_t bytes);

    /// Flushes remaining data and closes. Idempotent; safe to call multiple
    /// times. The destructor calls this automatically.
    void finalize();

    SpillCompression compression() const noexcept { return compression_; }

private:
    std::string      path_;
    SpillCompression compression_;
    std::ofstream    out_;
    bool             finalized_ = false;

#ifdef HAS_LZ4
    // Opaque LZ4F_cctx (declared void* to avoid leaking lz4frame.h into users)
    void* lz4_ctx_ = nullptr;
    std::vector<uint8_t> compress_buffer_; // scratch for LZ4 output blocks
#endif
};

/// Streaming reader for spill files. Auto-detects compression from the
/// 8-byte header; falls back to raw read if the header magic is absent
/// (legacy files).
///
/// Not thread-safe. Not copyable.
class SpillReader {
public:
    explicit SpillReader(const std::string& path);
    ~SpillReader();

    SpillReader(const SpillReader&)            = delete;
    SpillReader& operator=(const SpillReader&) = delete;
    SpillReader(SpillReader&&)                 = delete;
    SpillReader& operator=(SpillReader&&)      = delete;

    /// Reads up to `max_bytes` into `dst`. Returns the number of bytes
    /// actually produced (0 on EOF). Transparent: caller always sees
    /// uncompressed data regardless of file format.
    ///
    /// Short reads may occur if an internal buffer boundary is reached;
    /// callers should loop until they have the data they need or eof().
    std::size_t read(void* dst, std::size_t max_bytes);

    bool             eof() const noexcept          { return eof_; }
    SpillCompression compression() const noexcept  { return compression_; }
    bool             legacy_format() const noexcept { return legacy_format_; }

private:
    // Refills `decompressed_` from the file. Returns false at EOF.
    bool pump_chunk();

    std::string      path_;
    std::ifstream    in_;
    SpillCompression compression_ = SpillCompression::NONE;
    bool             legacy_format_ = false;
    bool             eof_ = false;

    // Decompressed data buffer (for LZ4) or passthrough buffer (for NONE).
    std::vector<uint8_t> decompressed_;
    std::size_t          decompressed_pos_ = 0;

#ifdef HAS_LZ4
    // Opaque LZ4F_dctx
    void* lz4_ctx_ = nullptr;
    std::vector<uint8_t> compressed_in_; // staging for raw bytes read from file
    std::size_t          compressed_pos_ = 0;
    std::size_t          compressed_len_ = 0;
    bool                 file_eof_reached_ = false;
#endif
};

} // namespace GQL
