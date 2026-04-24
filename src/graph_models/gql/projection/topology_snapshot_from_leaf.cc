#include "graph_models/gql/projection/topology_snapshot_from_leaf.h"

#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#include <cerrno>
#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "graph_models/object_id.h"
#include "storage/page/page.h"

namespace GQL::Projection {

namespace {

// .leaf source basename for each CSR direction (mirrors
// topology_snapshot_writer.cc's source_basename_for).
const char* source_basename_for(TopologySnapshotWriter::Direction d) {
    switch (d) {
    case TopologySnapshotWriter::Direction::FORWARD: return "from_to_edge.leaf";
    case TopologySnapshotWriter::Direction::REVERSE: return "to_from_edge.leaf";
    }
    return "from_to_edge.leaf";
}

// RAII wrapper over an mmap region. Constructed mapped-and-owning; the
// destructor munmaps and closes. Move-only.
class MappedFile {
public:
    static MappedFile open_readonly(const std::filesystem::path& path) {
        int fd = ::open(path.c_str(), O_RDONLY);
        if (fd < 0) {
            int e = errno;
            throw std::runtime_error(
                "topology_snapshot_from_leaf: open failed for "
                + path.string()
                + " (errno=" + std::to_string(e) + ")");
        }

        struct stat st{};
        if (::fstat(fd, &st) != 0) {
            int e = errno;
            ::close(fd);
            throw std::runtime_error(
                "topology_snapshot_from_leaf: fstat failed for "
                + path.string()
                + " (errno=" + std::to_string(e) + ")");
        }

        const std::size_t len = static_cast<std::size_t>(st.st_size);
        if (len == 0) {
            ::close(fd);
            return MappedFile(nullptr, 0, -1);
        }

        void* addr = ::mmap(nullptr, len, PROT_READ, MAP_PRIVATE, fd, 0);
        if (addr == MAP_FAILED) {
            int e = errno;
            ::close(fd);
            throw std::runtime_error(
                "topology_snapshot_from_leaf: mmap failed for "
                + path.string()
                + " (errno=" + std::to_string(e) + ")");
        }

        // Hint the kernel that we will scan sequentially. This keeps the
        // readahead window aggressive for the two sequential passes and
        // evicts each page once we've moved past it, so the mmap of a
        // 37 GB .leaf on papers100M doesn't bloat RSS.
        ::madvise(addr, len, MADV_SEQUENTIAL);

        return MappedFile(addr, len, fd);
    }

    MappedFile(const MappedFile&)            = delete;
    MappedFile& operator=(const MappedFile&) = delete;

    MappedFile(MappedFile&& other) noexcept
        : base_(other.base_), len_(other.len_), fd_(other.fd_) {
        other.base_ = nullptr;
        other.len_  = 0;
        other.fd_   = -1;
    }

    ~MappedFile() {
        if (base_ != nullptr && len_ > 0) {
            ::munmap(base_, len_);
        }
        if (fd_ >= 0) {
            ::close(fd_);
        }
    }

    const uint8_t* data() const noexcept {
        return static_cast<const uint8_t*>(base_);
    }
    std::size_t size() const noexcept { return len_; }
    bool        empty() const noexcept { return base_ == nullptr || len_ == 0; }

private:
    MappedFile(void* base, std::size_t len, int fd)
        : base_(base), len_(len), fd_(fd) {}

    void*       base_ = nullptr;
    std::size_t len_  = 0;
    int         fd_   = -1;
};

// Walk every leaf page and invoke `visit(src_u64_raw, dst_u64, eid_u64)`
// once per record, in the exact order they were written by
// BPTLeafWriter<3>::process_block.
//
// The page layout is (see bpt_mem_import.h:31-44 and projection_storage.cc:
// build_index_streaming's write_leaf_page lambda):
//   [0]  uint32_t value_count  (number of records in this page)
//   [4]  uint32_t next_leaf    (0 on last page; else page_number+1)
//   [8]  bitset: N zero bytes (no compression)
//   [8+N] records: `value_count × (N × sizeof(uint64_t))` bytes
//   [tail] zero-padding to Page::SIZE
//
// We iterate pages by advancing Page::SIZE each step. Empty .leaf files
// (from `BPTLeafWriter::make_empty()`) are one all-zero page; value_count
// will be 0 and the loop body is skipped.
template<typename Visitor>
void visit_leaf_records(const MappedFile& mm, Visitor&& visit) {
    constexpr std::size_t kPageSize    = Page::SIZE;
    constexpr std::size_t kRecordArity = 3;
    constexpr std::size_t kRecordBytes = kRecordArity * sizeof(uint64_t);
    constexpr std::size_t kBitsetBytes = kRecordArity;   // N bytes (N==3)
    constexpr std::size_t kHeaderBytes = 2 * sizeof(uint32_t);

    if (mm.empty()) {
        return;
    }
    if (mm.size() % kPageSize != 0) {
        throw std::runtime_error(
            "topology_snapshot_from_leaf: .leaf file size ("
            + std::to_string(mm.size())
            + ") is not a multiple of Page::SIZE=" + std::to_string(kPageSize));
    }

    const uint8_t* base = mm.data();
    const std::size_t num_pages = mm.size() / kPageSize;

    for (std::size_t page = 0; page < num_pages; ++page) {
        const uint8_t* p = base + page * kPageSize;

        uint32_t value_count = 0;
        std::memcpy(&value_count, p, sizeof(uint32_t));
        // next_leaf is not consumed — we just walk pages linearly.

        if (value_count == 0) {
            // Empty leaf page: either the make_empty sentinel or a
            // trailing zero padding past the last non-empty page. In both
            // cases we have no records to emit.
            continue;
        }

        // build_index_streaming writes `no_compression` (bitset == 0). We
        // still advance past the N-byte bitset region for robustness; the
        // value itself is unused.
        const uint8_t* rec = p + kHeaderBytes + kBitsetBytes;

        // Defensive size check: page payload must fit within Page::SIZE.
        const std::size_t needed =
            kHeaderBytes + kBitsetBytes +
            static_cast<std::size_t>(value_count) * kRecordBytes;
        if (needed > kPageSize) {
            throw std::runtime_error(
                "topology_snapshot_from_leaf: corrupt leaf page "
                + std::to_string(page) + " (value_count="
                + std::to_string(value_count) + " implies "
                + std::to_string(needed) + " B > "
                + std::to_string(kPageSize) + " B Page::SIZE)");
        }

        for (uint32_t i = 0; i < value_count; ++i) {
            uint64_t k0 = 0, k1 = 0, k2 = 0;
            std::memcpy(&k0, rec + 0 * sizeof(uint64_t), sizeof(uint64_t));
            std::memcpy(&k1, rec + 1 * sizeof(uint64_t), sizeof(uint64_t));
            std::memcpy(&k2, rec + 2 * sizeof(uint64_t), sizeof(uint64_t));
            visit(k0, k1, k2);
            rec += kRecordBytes;
        }
    }
}

}  // namespace

void build_topology_snapshot_from_leaf(
    const std::filesystem::path&      projection_dir,
    TopologySnapshotWriter::Direction dir,
    uint64_t                          num_nodes,
    bool                              include_edge_ids)
{
    const std::filesystem::path leaf_path =
        projection_dir / source_basename_for(dir);

    // ---- mmap the .leaf file (both passes share the kernel page cache) ----
    // The file was just fsync'd + closed by BPTLeafWriter at the end of
    // build_index_streaming, so it is guaranteed present and complete.
    // An absent file is treated as a zero-edge projection (matches the
    // legacy post-hoc path's behavior when num_edges_ == 0).
    if (!std::filesystem::exists(leaf_path)) {
        // Still emit a valid empty CSR so downstream consumers don't
        // have to special-case missing sidecars. Zero degrees means
        // ROW_PTR = {0, 0, ..., 0} and no COL_IDX bytes.
        std::vector<uint64_t> degrees(num_nodes, 0);
        TopologySnapshotWriter writer(projection_dir, dir, num_nodes,
                                      std::move(degrees), include_edge_ids);
        writer.finalize();
        return;
    }

    MappedFile mm = MappedFile::open_readonly(leaf_path);

    // ---- Pass 1: degrees ---------------------------------------------------
    // Match the legacy BPT-iterator logic at
    // native_projection_builder.cc:2491:
    //
    //     const uint64_t src_idx = (*rec)[0] & ObjectId::VALUE_MASK;
    //     if (src_idx < num_nodes) ++degrees[src_idx];
    //
    // The .leaf stores the FULL ObjectId (with the 8-bit type tag);
    // ROW_PTR is keyed by the tag-stripped dense row id. Out-of-range
    // src is silently skipped, mirroring the post-hoc path's defensive
    // bounds check.
    std::vector<uint64_t> degrees(num_nodes, 0);
    visit_leaf_records(mm,
        [&](uint64_t k0, uint64_t /*k1*/, uint64_t /*k2*/) {
            const uint64_t src_idx = k0 & ObjectId::VALUE_MASK;
            if (src_idx < num_nodes) {
                ++degrees[src_idx];
            }
        });

    // ---- Pass 2: stream edges through the writer --------------------------
    // The writer builds ROW_PTR from degrees in its ctor; append_edge()
    // uses src.id as the row subscript, so we pass the tag-stripped
    // value. dst and edge_id keep their full ObjectId so COL_IDX /
    // EDGE_IDS match the BPT records byte-for-byte.
    TopologySnapshotWriter writer(projection_dir, dir, num_nodes,
                                  std::move(degrees), include_edge_ids);

    visit_leaf_records(mm,
        [&](uint64_t k0, uint64_t k1, uint64_t k2) {
            const uint64_t src_idx = k0 & ObjectId::VALUE_MASK;
            if (src_idx >= num_nodes) {
                // Matches pass 1's bounds check: out-of-range src records
                // are silently skipped. They were not counted in degrees,
                // so row_ptr has no slot reserved for them; calling
                // append_edge with src_idx >= num_nodes would trip the
                // writer's own assert.
                return;
            }
            writer.append_edge(ObjectId{src_idx}, ObjectId{k1}, ObjectId{k2});
        });

    writer.finalize();
}

}  // namespace GQL::Projection
