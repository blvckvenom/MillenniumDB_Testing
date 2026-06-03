// src/gnn/storage/consolidated_slim_reader.h
#pragma once

#include "gnn/storage/consolidated_slim.h"

#include <cstddef>
#include <cstdint>

namespace mdb::gnn {

/**
 * @brief Validate a ConsolidatedSlimHeader against the runtime's expectations.
 *
 * DiskGNN-adoption Plan 1, Phase 2.1. Called once at FourLevelStore ctor when
 * probing `packed_slim/consolidated.slim`. A false return means "refuse the
 * consolidated file, fall back to the per-batch read" — never a hard error, so
 * a stale/foreign/mismatched file degrades gracefully instead of serving wrong
 * features (the IDX_VERSION-2 discipline; a silently-adopted stale sidecar is
 * exactly what capped papers100M val_acc at 0.16).
 *
 * Checks: structural validity, feature_dim, dtype, and the two stale-rejection
 * fingerprints. `expected_perm_fp == 0` disables the permutation check (stores
 * built with reorder disabled stamp 0); `expected_meta_sha == 0` disables the
 * meta check (matches AddrTableReader's convention).
 */
bool validate_consolidated_header(const ConsolidatedSlimHeader& h,
                                  uint64_t expected_feature_dim,
                                  uint8_t  expected_dtype,
                                  uint64_t expected_perm_fp,
                                  uint64_t expected_meta_sha);

/**
 * @brief pread exactly `len` bytes from `fd` at `offset` into `dst`.
 *
 * Loops on short reads and retries EINTR. For an O_DIRECT fd the caller must
 * guarantee `offset`, `len`, and `dst` are all block-aligned — Plan 1 satisfies
 * this by construction: the consolidated per-batch offsets are 4096-aligned, the
 * worker buffer is aligned_alloc'd, and the caller reads align_up(payload, 4096)
 * (the file is ftruncated to the aligned total, so the padding read stays
 * in-file). Returns true iff all `len` bytes were read; false on any error,
 * bad fd, or premature EOF — the caller then falls back to a buffered read of
 * the same range.
 */
bool pread_exact(int fd, void* dst, size_t len, uint64_t offset);

} // namespace mdb::gnn
