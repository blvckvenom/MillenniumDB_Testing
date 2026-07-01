#pragma once

#include <cstdint>

#include "gnn/storage/four_level_store.h"

namespace mdb::gnn {

/// POD snapshot of FourLevelStore::Stats values. The live Stats struct
/// uses std::atomic<uint64_t> so it cannot be copied; this trivially-
/// copyable POD is the safe transport for callbacks (e.g.,
/// TrainingLoop::Config::cache_stats_provider) and external prints
/// (gnn procedure layer JSON / YIELD output).
///
/// The byte-level counters allow disk-traffic accounting directly
/// comparable to published out-of-core GNN systems (total_bytes_disk()
/// matches the "Disk access volume (GB)" row of Table 1 in DiskGNN,
/// SIGMOD'25).
///
/// operator- enables per-epoch delta computation without resetting the
/// live Stats — the per-epoch print can show ΔL3 / ΔL4 disk traffic
/// between consecutive epochs while the cumulative end-of-run total
/// stays available for the JSON log.
struct CacheStatsSnapshot {
    // Per-tier node-count counters.
    uint64_t l1_hits         = 0;
    uint64_t l2_hits         = 0;
    uint64_t l3_reads        = 0;
    uint64_t l4_reads        = 0;
    uint64_t total_requests  = 0;

    // Byte-level counters.
    uint64_t l1_bytes_served = 0;
    uint64_t l2_bytes_served = 0;
    uint64_t l3_bytes_wanted = 0;
    uint64_t l3_bytes_disk   = 0;
    uint64_t l4_bytes_wanted = 0;
    uint64_t l4_bytes_disk   = 0;

    static CacheStatsSnapshot from(const FourLevelStore::Stats& s) {
        CacheStatsSnapshot snap;
        snap.l1_hits         = s.l1_hits.load();
        snap.l2_hits         = s.l2_hits.load();
        snap.l3_reads        = s.l3_reads.load();
        snap.l4_reads        = s.l4_reads.load();
        snap.total_requests  = s.total_requests.load();
        snap.l1_bytes_served = s.l1_bytes_served.load();
        snap.l2_bytes_served = s.l2_bytes_served.load();
        snap.l3_bytes_wanted = s.l3_bytes_wanted.load();
        snap.l3_bytes_disk   = s.l3_bytes_disk.load();
        snap.l4_bytes_wanted = s.l4_bytes_wanted.load();
        snap.l4_bytes_disk   = s.l4_bytes_disk.load();
        return snap;
    }

    /// Per-epoch delta. Subtraction is field-wise unsigned —
    /// the live Stats counters are monotonic non-decreasing, so the
    /// result is always non-negative when both snapshots come from
    /// the same FourLevelStore in temporal order.
    CacheStatsSnapshot operator-(const CacheStatsSnapshot& other) const {
        CacheStatsSnapshot d;
        d.l1_hits         = l1_hits         - other.l1_hits;
        d.l2_hits         = l2_hits         - other.l2_hits;
        d.l3_reads        = l3_reads        - other.l3_reads;
        d.l4_reads        = l4_reads        - other.l4_reads;
        d.total_requests  = total_requests  - other.total_requests;
        d.l1_bytes_served = l1_bytes_served - other.l1_bytes_served;
        d.l2_bytes_served = l2_bytes_served - other.l2_bytes_served;
        d.l3_bytes_wanted = l3_bytes_wanted - other.l3_bytes_wanted;
        d.l3_bytes_disk   = l3_bytes_disk   - other.l3_bytes_disk;
        d.l4_bytes_wanted = l4_bytes_wanted - other.l4_bytes_wanted;
        d.l4_bytes_disk   = l4_bytes_disk   - other.l4_bytes_disk;
        return d;
    }

    double l1_hit_ratio() const {
        return total_requests > 0
            ? static_cast<double>(l1_hits) / static_cast<double>(total_requests)
            : 0.0;
    }
    double l2_hit_ratio() const {
        return total_requests > 0
            ? static_cast<double>(l2_hits) / static_cast<double>(total_requests)
            : 0.0;
    }

    /// Total physical disk traffic across L3 + L4 — the headline number
    /// comparable to the "Disk access volume (GB)" row of DiskGNN's Table 1.
    uint64_t total_bytes_disk() const {
        return l3_bytes_disk + l4_bytes_disk;
    }

    /// Total useful feature payload extracted from disk tiers.
    uint64_t total_bytes_wanted() const {
        return l3_bytes_wanted + l4_bytes_wanted;
    }

    /// Read amplification on the L3 path: bytes_disk / bytes_wanted.
    /// 1.0 = no overhead. For scattered point lookups the worst case
    /// approaches page_size / row_size (8× for 512 B feature rows read
    /// through 4 KiB pages); page-level dedup plus the MinHash row
    /// clustering typically brings it down to ~2-4×.
    double l3_read_amplification() const {
        return l3_bytes_wanted > 0
            ? static_cast<double>(l3_bytes_disk) / static_cast<double>(l3_bytes_wanted)
            : 0.0;
    }
};

} // namespace mdb::gnn
