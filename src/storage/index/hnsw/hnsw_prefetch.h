#pragma once

#include <cstddef>
#include <cstdint>

namespace HNSW {

/**
 * @brief Memory prefetch utilities for HNSW search optimization.
 *
 * These functions use compiler intrinsics to hint the CPU to preload data
 * into cache before it's needed, hiding memory latency behind computation.
 *
 * Cache hierarchy (typical):
 *   - L1: 32KB, ~4 cycles latency
 *   - L2: 256KB, ~12 cycles latency
 *   - L3: 8MB, ~40 cycles latency
 *   - RAM: ~100-300 cycles latency
 *
 * For 128-dim float embeddings (512 bytes), prefetching can overlap
 * memory access with distance computation for significant speedup.
 */

/**
 * @brief Prefetch embedding data into L1 cache.
 *
 * Prefetches the first several cache lines of an embedding vector.
 * Call this 2-3 neighbors ahead to hide memory latency.
 *
 * @param ptr Pointer to embedding data
 * @param dim Embedding dimension (number of floats)
 */
inline void prefetch_embedding(const float* ptr, size_t dim)
{
    // Cache line = 64 bytes = 16 floats
    // Prefetch first 4 cache lines (covers first 64 floats)
    // For 128-dim embeddings, this prefetches half the vector
    constexpr size_t FLOATS_PER_CACHE_LINE = 16;
    constexpr size_t MAX_LINES_TO_PREFETCH = 4;

    const size_t total_lines = (dim + FLOATS_PER_CACHE_LINE - 1) / FLOATS_PER_CACHE_LINE;
    const size_t lines = (total_lines < MAX_LINES_TO_PREFETCH) ? total_lines : MAX_LINES_TO_PREFETCH;

    for (size_t i = 0; i < lines; ++i) {
        // __builtin_prefetch(addr, rw, locality)
        //   rw: 0 = read, 1 = write
        //   locality: 0 = no reuse (NTA), 1 = low, 2 = moderate, 3 = high (keep in all caches)
        __builtin_prefetch(ptr + i * FLOATS_PER_CACHE_LINE, 0, 3);
    }
}

/**
 * @brief Prefetch remaining cache lines of an embedding.
 *
 * Call this after prefetch_embedding for very large embeddings
 * or when more aggressive prefetching is beneficial.
 *
 * @param ptr Pointer to embedding data
 * @param dim Embedding dimension
 * @param start_line First cache line to prefetch (0-indexed)
 */
inline void prefetch_embedding_extended(const float* ptr, size_t dim, size_t start_line)
{
    constexpr size_t FLOATS_PER_CACHE_LINE = 16;
    constexpr size_t MAX_ADDITIONAL_LINES = 4;

    const size_t total_lines = (dim + FLOATS_PER_CACHE_LINE - 1) / FLOATS_PER_CACHE_LINE;

    for (size_t i = 0; i < MAX_ADDITIONAL_LINES && (start_line + i) < total_lines; ++i) {
        __builtin_prefetch(ptr + (start_line + i) * FLOATS_PER_CACHE_LINE, 0, 2);
    }
}

/**
 * @brief Prefetch a neighbor list entry.
 *
 * Useful for prefetching the next few neighbors in the adjacency list
 * while processing the current neighbor.
 *
 * @param ptr Pointer to neighbor entry or container
 */
inline void prefetch_neighbor_entry(const void* ptr)
{
    // Low locality hint since we typically iterate once
    __builtin_prefetch(ptr, 0, 1);
}

/**
 * @brief Prefetch norm value for a node.
 *
 * @param norms_array Pointer to norms array
 * @param node_id Node ID to prefetch norm for
 */
inline void prefetch_norm(const float* norms_array, uint32_t node_id)
{
    __builtin_prefetch(norms_array + node_id, 0, 3);
}

} // namespace HNSW
