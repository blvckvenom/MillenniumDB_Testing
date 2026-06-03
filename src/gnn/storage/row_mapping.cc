#include "gnn/storage/row_mapping.h"

#include <cerrno>
#include <cstring>
#include <stdexcept>
#include <type_traits>

#if __has_include(<execution>)
#include <execution>
#define MDB_GNN_HAS_PAR_EXECUTION 1
#endif

#include <fcntl.h>
#include <sys/mman.h>
#include <unistd.h>

namespace mdb::gnn {

static_assert(RowMapping::HEADER_SIZE % alignof(ObjectId) == 0,
              "HEADER_SIZE must be aligned to ObjectId alignment");

// Guard noexcept move operations — if sorted_index_ or build_index_flag_
// types change to non-nothrow-movable, this catches it at compile time.
static_assert(std::is_nothrow_move_constructible_v<std::vector<std::pair<uint64_t, uint64_t>>>,
              "sorted_index_ must be nothrow-movable for RowMapping noexcept move operations");
static_assert(std::is_nothrow_move_constructible_v<std::unique_ptr<std::once_flag>>,
              "build_index_flag_ must be nothrow-movable for RowMapping noexcept move operations");

namespace fs = std::filesystem;

namespace {

class FdGuard {
public:
    explicit FdGuard(int fd) : fd_(fd) {}
    ~FdGuard() { if (fd_ >= 0) ::close(fd_); }
    FdGuard(const FdGuard&) = delete;
    FdGuard& operator=(const FdGuard&) = delete;
private:
    int fd_;
};

void write_all(int fd, const void* buf, size_t count) {
    const char* p = static_cast<const char*>(buf);
    size_t remaining = count;
    while (remaining > 0) {
        ssize_t written = ::write(fd, p, remaining);
        if (written < 0) {
            if (errno == EINTR) continue;
            throw std::runtime_error(
                "RowMapping: write failed: " + std::string(std::strerror(errno)));
        }
        if (written == 0) {
            throw std::runtime_error(
                "RowMapping: write returned 0 for non-zero count — disk full or I/O error");
        }
        p += written;
        remaining -= static_cast<size_t>(written);
    }
}

} // anonymous namespace

// --- Move semantics ---

RowMapping::RowMapping(RowMapping&& other) noexcept
    : path_(std::move(other.path_)),
      mmap_ptr_(other.mmap_ptr_),
      mmap_size_(other.mmap_size_),
      count_(other.count_),
      idx_mmap_ptr_(other.idx_mmap_ptr_),
      idx_mmap_size_(other.idx_mmap_size_),
      idx_data_(other.idx_data_),
      build_index_flag_(std::move(other.build_index_flag_)),
      sorted_index_(std::move(other.sorted_index_))
{
    other.mmap_ptr_      = nullptr;
    other.mmap_size_     = 0;
    other.count_         = 0;
    other.idx_mmap_ptr_  = nullptr;
    other.idx_mmap_size_ = 0;
    other.idx_data_      = nullptr;
}

RowMapping& RowMapping::operator=(RowMapping&& other) noexcept {
    if (this != &other) {
        if (mmap_ptr_ != nullptr) {
            ::munmap(mmap_ptr_, mmap_size_);
        }
        if (idx_mmap_ptr_ != nullptr) {
            ::munmap(idx_mmap_ptr_, idx_mmap_size_);
        }
        path_             = std::move(other.path_);
        mmap_ptr_         = other.mmap_ptr_;
        mmap_size_        = other.mmap_size_;
        count_            = other.count_;
        idx_mmap_ptr_     = other.idx_mmap_ptr_;
        idx_mmap_size_    = other.idx_mmap_size_;
        idx_data_         = other.idx_data_;
        build_index_flag_ = std::move(other.build_index_flag_);
        sorted_index_     = std::move(other.sorted_index_);
        other.mmap_ptr_      = nullptr;
        other.mmap_size_     = 0;
        other.count_         = 0;
        other.idx_mmap_ptr_  = nullptr;
        other.idx_mmap_size_ = 0;
        other.idx_data_      = nullptr;
    }
    return *this;
}

RowMapping::~RowMapping() {
    if (mmap_ptr_ != nullptr) {
        ::munmap(mmap_ptr_, mmap_size_);
        mmap_ptr_ = nullptr;
    }
    if (idx_mmap_ptr_ != nullptr) {
        ::munmap(idx_mmap_ptr_, idx_mmap_size_);
        idx_mmap_ptr_ = nullptr;
    }
}

const ObjectId* RowMapping::data_ptr() const {
    return reinterpret_cast<const ObjectId*>(
        static_cast<const char*>(mmap_ptr_) + HEADER_SIZE);
}

// --- create() ---

RowMapping RowMapping::create(const fs::path& path, const std::vector<ObjectId>& ids) {
    // Overflow check
    size_t data_size = ids.size() * sizeof(ObjectId);
    if (!ids.empty() && data_size / sizeof(ObjectId) != ids.size()) {
        throw std::overflow_error("RowMapping::create: data size would overflow");
    }
    if (data_size > SIZE_MAX - HEADER_SIZE) {
        throw std::overflow_error("RowMapping::create: total file size would overflow");
    }
    size_t file_size = HEADER_SIZE + data_size;

    int fd = ::open(path.c_str(), O_RDWR | O_CREAT | O_TRUNC, 0644);
    if (fd < 0) {
        throw std::runtime_error(
            "RowMapping::create: cannot open " + path.string() + ": " + std::strerror(errno));
    }
    FdGuard guard(fd);

    // Write header: magic(4) + version(4) + count(8)
    uint32_t magic   = MAGIC;
    uint32_t version = VERSION;
    uint64_t count   = ids.size();
    write_all(fd, &magic, sizeof(magic));
    write_all(fd, &version, sizeof(version));
    write_all(fd, &count, sizeof(count));

    // Write ObjectId array (each ObjectId is 8 bytes — just its uint64_t id)
    static_assert(sizeof(ObjectId) == sizeof(uint64_t),
                  "ObjectId must be 8 bytes for direct serialization");
    if (!ids.empty()) {
        write_all(fd, ids.data(), data_size);
    }

    if (::fsync(fd) < 0) {
        throw std::runtime_error(
            "RowMapping::create: fsync failed: " + std::string(std::strerror(errno)));
    }

    // Best-effort parent directory fsync for crash consistency
    {
        int dir_fd = ::open(path.parent_path().c_str(), O_RDONLY);
        if (dir_fd >= 0) {
            ::fsync(dir_fd);
            ::close(dir_fd);
        }
    }

    // A freshly-written .rmap carries a NEW permutation, which invalidates any
    // persisted sorted-index sidecar built from the OLD permutation. Remove the
    // orphan so a later open() rebuilds the index from THIS .rmap rather than
    // silently adopting a stale <path>.idx (the v1 count-only guard could not
    // detect a same-N permutation change — root cause of the L4 feature-row
    // corruption fixed 2026-06-01). The IDX_VERSION-2 fingerprint is the
    // defense-in-depth; this removal is the direct fix at the write site.
    {
        std::error_code ec;
        fs::remove(fs::path(path.string() + ".idx"), ec);
    }

    // mmap read-only
    void* ptr = ::mmap(nullptr, file_size, PROT_READ, MAP_PRIVATE, fd, 0);
    if (ptr == MAP_FAILED) {
        throw std::runtime_error(
            "RowMapping::create: mmap failed: " + std::string(std::strerror(errno)));
    }

    RowMapping rm;
    rm.path_      = path;
    rm.mmap_ptr_  = ptr;
    rm.mmap_size_ = file_size;
    rm.count_     = count;
    // Note: build_index() is NOT called here — it runs lazily on first find().
    return rm;
}

// --- open() ---

RowMapping RowMapping::open(const fs::path& path) {
    if (!fs::exists(path)) {
        throw std::runtime_error("RowMapping::open: file not found: " + path.string());
    }

    auto file_size = fs::file_size(path);
    if (file_size < HEADER_SIZE) {
        throw std::runtime_error(
            "RowMapping::open: file too small (" + std::to_string(file_size) +
            " bytes): " + path.string());
    }

    int fd = ::open(path.c_str(), O_RDONLY);
    if (fd < 0) {
        throw std::runtime_error(
            "RowMapping::open: cannot open " + path.string() + ": " + std::strerror(errno));
    }
    FdGuard guard(fd);

    void* ptr = ::mmap(nullptr, file_size, PROT_READ, MAP_PRIVATE, fd, 0);
    if (ptr == MAP_FAILED) {
        throw std::runtime_error(
            "RowMapping::open: mmap failed: " + std::string(std::strerror(errno)));
    }

    // Validate header
    const char* base = static_cast<const char*>(ptr);
    uint32_t magic, version;
    uint64_t count;
    std::memcpy(&magic, base, sizeof(magic));
    std::memcpy(&version, base + 4, sizeof(version));
    std::memcpy(&count, base + 8, sizeof(count));

    if (magic != MAGIC || version != VERSION) {
        ::munmap(ptr, file_size);
        throw std::runtime_error(
            "RowMapping::open: invalid header in " + path.string());
    }

    // Overflow check: prevent count * sizeof(ObjectId) from wrapping
    if (count > (SIZE_MAX - HEADER_SIZE) / sizeof(ObjectId)) {
        ::munmap(ptr, file_size);
        throw std::runtime_error(
            "RowMapping::open: count in header would overflow size computation: " +
            path.string());
    }

    size_t expected = HEADER_SIZE + count * sizeof(ObjectId);
    if (file_size < expected) {
        ::munmap(ptr, file_size);
        throw std::runtime_error(
            "RowMapping::open: file truncated — expected " +
            std::to_string(expected) + " bytes, got " +
            std::to_string(file_size) + ": " + path.string());
    }

    RowMapping rm;
    rm.path_      = path;
    rm.mmap_ptr_  = ptr;
    rm.mmap_size_ = file_size;
    rm.count_     = count;
    // Fix #17: try to mmap a persisted sorted-index sidecar. If present
    // and matches (count, magic), find() will use it directly and we
    // skip the O(N log N) lazy build entirely (~30 s on papers100M).
    rm.try_load_persisted_index_();
    return rm;
}

// --- get() ---

ObjectId RowMapping::get(uint64_t row_index) const {
    if (mmap_ptr_ == nullptr) {
        throw std::runtime_error("RowMapping::get: not mapped");
    }
    if (row_index >= count_) {
        throw std::out_of_range(
            "RowMapping::get: index " + std::to_string(row_index) +
            " out of range [0, " + std::to_string(count_) + ")");
    }
    return data_ptr()[row_index];
}

// --- find() ---

std::optional<uint64_t> RowMapping::find(ObjectId target) const {
    if (mmap_ptr_ == nullptr) {
        throw std::runtime_error("RowMapping::find: not mapped");
    }

    // Fast path (Fix #17): a mmap'd sidecar `<path>.idx` was loaded at
    // open() — search it directly without paying the lazy build cost.
    if (idx_data_ != nullptr) {
        auto it = std::lower_bound(idx_data_, idx_data_ + count_,
            std::make_pair(target.id, uint64_t(0)),
            [](const auto& a, const auto& b) { return a.first < b.first; });
        if (it != idx_data_ + count_ && it->first == target.id) {
            return it->second;
        }
        return std::nullopt;
    }

    std::call_once(*build_index_flag_, [this] {
        build_index();
        // Fix #17: best-effort persist after first build, so next open
        // benefits from the fast path. Ignore failures (the sidecar is
        // strictly optional — the in-memory sorted_index_ remains valid).
        try { persist_sorted_index_(); } catch (...) { /* ignore */ }
    });

    auto it = std::lower_bound(sorted_index_.begin(), sorted_index_.end(),
        std::make_pair(target.id, uint64_t(0)),
        [](const auto& a, const auto& b) { return a.first < b.first; });

    if (it != sorted_index_.end() && it->first == target.id) {
        return it->second;
    }
    return std::nullopt;
}

// --- build_index() ---

void RowMapping::build_index() const {
    sorted_index_.resize(count_);
    const ObjectId* arr = data_ptr();
    for (uint64_t i = 0; i < count_; ++i) {
        sorted_index_[i] = {arr[i].id, i};
    }
    // Fix #18 (2026-05-13): use parallel sort when TBB is linked. With
    // libstdc++ ≥ 9 + libtbb the C++17 par_unseq policy speeds the
    // 111M-entry sort from ~30 s to ~6-8 s on a 20-core host. Falls
    // back to serial sort when <execution> is unavailable.
#ifdef MDB_GNN_HAS_PAR_EXECUTION
    std::sort(std::execution::par_unseq,
              sorted_index_.begin(), sorted_index_.end());
#else
    std::sort(sorted_index_.begin(), sorted_index_.end());
#endif
}

// --- Permutation fingerprint (2026-06-01) ---

uint64_t RowMapping::compute_perm_fingerprint_() const {
    // Order-sensitive FNV-1a-64 over the ObjectId array. A permutation change
    // reorders the array and so changes this hash, letting try_load detect a
    // .idx sidecar built from a different permutation (even at the same count).
    // O(N) — ~0.2-0.5 s on papers100M, negligible vs the ~6-8 s index build it
    // gates. Not cryptographic; staleness detection only.
    const uint64_t FNV_OFFSET = 1469598103934665603ULL;
    const uint64_t FNV_PRIME  = 1099511628211ULL;
    uint64_t h = FNV_OFFSET;
    const ObjectId* arr = data_ptr();
    for (uint64_t i = 0; i < count_; ++i) {
        h = (h ^ arr[i].id) * FNV_PRIME;
        h ^= (h >> 29);  // extra avalanche so adjacent swaps diverge fast
    }
    return h;
}

uint64_t RowMapping::perm_fingerprint() const {
    // Public accessor over the private order-sensitive fingerprint. An empty
    // mapping has no permutation to bind, so report 0 (a dependent artifact
    // treats 0 as "reorder disabled / no fingerprint to check").
    if (count_ == 0) return 0;
    return compute_perm_fingerprint_();
}

// --- Fix #17: persistent sorted-index sidecar ---

void RowMapping::persist_sorted_index_() const {
    if (count_ == 0 || sorted_index_.empty()) return;
    auto idx_path = fs::path(path_.string() + ".idx");
    auto tmp_path = fs::path(idx_path.string() + ".tmp");

    int fd = ::open(tmp_path.c_str(), O_WRONLY | O_CREAT | O_TRUNC, 0644);
    if (fd < 0) {
        // Best-effort — log via runtime_error which the caller swallows
        throw std::runtime_error(
            "RowMapping::persist_sorted_index_: cannot create " + tmp_path.string() +
            ": " + std::strerror(errno));
    }
    FdGuard guard(fd);

    // Header: idx_magic(4) + idx_version(4) + count(8) + perm_fp(8) = 24 bytes.
    // perm_fp binds this sidecar to the .rmap permutation it indexes (v2).
    uint32_t idx_magic   = IDX_MAGIC;
    uint32_t idx_version = IDX_VERSION;
    uint64_t count       = sorted_index_.size();
    uint64_t perm_fp     = compute_perm_fingerprint_();
    write_all(fd, &idx_magic,   sizeof(idx_magic));
    write_all(fd, &idx_version, sizeof(idx_version));
    write_all(fd, &count,       sizeof(count));
    write_all(fd, &perm_fp,     sizeof(perm_fp));

    // Data: count × (uint64 oid, uint64 row) pairs, sorted by oid
    size_t data_bytes = count * sizeof(std::pair<uint64_t, uint64_t>);
    write_all(fd, sorted_index_.data(), data_bytes);

    if (::fsync(fd) < 0) {
        throw std::runtime_error(
            "RowMapping::persist_sorted_index_: fsync failed: " +
            std::string(std::strerror(errno)));
    }
    std::error_code ec;
    fs::rename(tmp_path, idx_path, ec);
    if (ec) {
        fs::remove(tmp_path, ec);
        throw std::runtime_error(
            "RowMapping::persist_sorted_index_: rename failed: " + ec.message());
    }
    int dir_fd = ::open(idx_path.parent_path().c_str(), O_RDONLY);
    if (dir_fd >= 0) { ::fsync(dir_fd); ::close(dir_fd); }
}

bool RowMapping::try_load_persisted_index_() {
    if (count_ == 0) return false;
    auto idx_path = fs::path(path_.string() + ".idx");
    if (!fs::exists(idx_path)) return false;

    auto file_size = fs::file_size(idx_path);
    const size_t header_size = IDX_HEADER_SIZE;  // magic+version+count+perm_fp
    if (file_size < header_size) return false;

    int fd = ::open(idx_path.c_str(), O_RDONLY);
    if (fd < 0) return false;
    FdGuard guard(fd);

    void* ptr = ::mmap(nullptr, file_size, PROT_READ, MAP_PRIVATE, fd, 0);
    if (ptr == MAP_FAILED) return false;

    const char* base = static_cast<const char*>(ptr);
    uint32_t idx_magic, idx_version;
    uint64_t idx_count, idx_perm_fp;
    std::memcpy(&idx_magic,   base,      sizeof(idx_magic));
    std::memcpy(&idx_version, base + 4,  sizeof(idx_version));
    std::memcpy(&idx_count,   base + 8,  sizeof(idx_count));
    std::memcpy(&idx_perm_fp, base + 16, sizeof(idx_perm_fp));

    // Reject a stale, wrong-format, or wrong-permutation sidecar. The perm_fp
    // check is the fix for the same-N permutation change the old count-only
    // guard missed: a .idx built from a different .rmap permutation no longer
    // matches and we fall back to lazily rebuilding the index from THIS .rmap.
    if (idx_magic != IDX_MAGIC || idx_version != IDX_VERSION ||
        idx_count != count_ || idx_perm_fp != compute_perm_fingerprint_()) {
        ::munmap(ptr, file_size);
        return false;
    }

    size_t expected = header_size + idx_count * sizeof(std::pair<uint64_t, uint64_t>);
    if (file_size < expected) {
        ::munmap(ptr, file_size);
        return false;
    }

    idx_mmap_ptr_  = ptr;
    idx_mmap_size_ = file_size;
    idx_data_      = reinterpret_cast<const std::pair<uint64_t, uint64_t>*>(
        base + header_size);
    return true;
}

} // namespace mdb::gnn
