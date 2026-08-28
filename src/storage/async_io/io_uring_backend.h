#pragma once

/**
 * @file io_uring_backend.h
 * @brief io_uring implementation of AsyncIO for Linux 5.1+.
 *
 * ## io_uring Overview
 *
 * io_uring is Linux's modern async I/O API that provides:
 * - **Batched syscalls**: Submit 32-64 operations with 1 syscall
 * - **Zero-copy**: Shares memory between kernel and userspace
 * - **NVMe-aware**: Exploits multiple hardware queue pairs
 *
 * ## Performance Characteristics
 *
 * - **IOPS**: 500K+ on NVMe vs 50K with blocking I/O
 * - **Latency**: Sub-microsecond submission overhead
 * - **Throughput**: 7 GB/s+ on modern NVMe drives
 *
 * ## Memory Layout
 *
 * ```
 * ┌─────────────────────────────────────────────────────────────┐
 * │                    Submission Queue (SQ)                     │
 * │  [SQE 0] [SQE 1] [SQE 2] ... [SQE N-1]                       │
 * │  Producer: userspace, Consumer: kernel                       │
 * ├─────────────────────────────────────────────────────────────┤
 * │                    Completion Queue (CQ)                     │
 * │  [CQE 0] [CQE 1] [CQE 2] ... [CQE N-1]                       │
 * │  Producer: kernel, Consumer: userspace                       │
 * └─────────────────────────────────────────────────────────────┘
 * ```
 *
 * @note Requires liburing and Linux kernel 5.1 or later.
 * @see https://kernel.dk/io_uring.pdf for io_uring internals
 */

#ifdef HAS_IO_URING

#include "storage/async_io/async_io.h"

#include <liburing.h>

#include <atomic>
#include <unordered_map>

namespace Storage {

/**
 * @brief io_uring backend for high-performance async I/O.
 *
 * Uses liburing to interface with the Linux io_uring subsystem.
 * Batches up to queue_depth I/O operations per syscall.
 */
class IOUringBackend : public AsyncIO {
public:
    /**
     * @brief Constructs io_uring backend with specified queue depth.
     *
     * @param queue_depth Maximum concurrent I/O operations (typically 32-256)
     * @throws std::runtime_error if io_uring initialization fails
     */
    explicit IOUringBackend(size_t queue_depth);

    /**
     * @brief Destructor - waits for pending operations and cleans up ring.
     */
    ~IOUringBackend() override;

    void submit_reads(const std::vector<IORequest>& requests) override;
    void submit_writes(const std::vector<IORequest>& requests) override;
    size_t wait_completions(CompletionCallback callback) override;
    void drain() override;

    size_t pending_count() const override { return pending_count_.load(); }
    size_t queue_depth() const override { return queue_depth_; }
    const char* backend_name() const override { return "io_uring"; }

    /**
     * @brief Runtime check for io_uring kernel support.
     *
     * Attempts to initialize a probe ring to verify kernel support.
     * Result is cached after first call.
     *
     * @return true if io_uring is available
     */
    static bool is_available();

private:
    /**
     * @brief Internal helper to prepare and submit SQEs.
     *
     * @param requests Requests to submit
     * @param is_write true for write operations, false for reads
     */
    void submit_operations(const std::vector<IORequest>& requests, bool is_write);

    struct io_uring ring_;
    size_t queue_depth_;
    std::atomic<size_t> pending_count_{0};

    // Map user_data pointer to IORequest for callback context
    // io_uring uses the user_data field to identify completions
    std::unordered_map<void*, IORequest> pending_requests_;
};

} // namespace Storage

#endif // HAS_IO_URING
