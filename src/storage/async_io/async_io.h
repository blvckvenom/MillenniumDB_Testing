#pragma once

/**
 * @file async_io.h
 * @brief Abstract interface for asynchronous I/O operations.
 *
 * Provides a unified API for batched read/write operations that exploit
 * modern storage hardware (NVMe) parallelism. Supports two backends:
 *
 * 1. **io_uring** (Linux 5.1+): Batch I/O with single syscall per 32-64 ops
 * 2. **preadv** (fallback): Thread pool using vectored I/O
 *
 * ## Design Rationale
 *
 * Modern NVMe SSDs have 32-64 hardware queue pairs and can sustain
 * 500K+ IOPS when properly utilized. Traditional blocking I/O uses
 * only 1 queue pair, leaving 97%+ bandwidth unused.
 *
 * This abstraction enables:
 * - **Batch submission**: Amortize syscall overhead across many operations
 * - **Overlap compute/I/O**: Process current buffer while prefetching next
 * - **Portable fallback**: Works on all POSIX systems via thread pool
 *
 * ## Usage Pattern
 *
 * ```cpp
 * auto aio = Storage::AsyncIO::create(32);  // 32-deep queue
 *
 * // Submit batch of reads
 * std::vector<AsyncIO::IORequest> requests;
 * for (int i = 0; i < 32; ++i) {
 *     requests.push_back({fd, buffers[i], size, offsets[i], &user_data[i]});
 * }
 * aio->submit_reads(requests);
 *
 * // Wait for completions
 * aio->wait_completions([](IORequest* req, ssize_t result) {
 *     if (result > 0) {
 *         process_buffer(req->buffer, result);
 *     }
 * });
 * ```
 *
 * ## Memory Model
 *
 * - Buffers must remain valid until completion callback
 * - User data pointer passed through for callback context
 * - No internal buffering - caller manages memory
 *
 * @see io_uring_backend.h for Linux io_uring implementation
 * @see preadv_backend.h for portable fallback
 */

#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <vector>

namespace Storage {

/**
 * @brief Abstract async I/O interface with io_uring and preadv backends.
 *
 * Provides batched read/write operations that exploit NVMe parallelism.
 * Automatically selects best available backend at runtime.
 */
class AsyncIO {
public:
    /**
     * @brief Single I/O request descriptor.
     *
     * Describes one read or write operation. Multiple requests can be
     * batched together for efficient submission.
     */
    struct IORequest {
        int fd;              ///< File descriptor to read from/write to
        void* buffer;        ///< Buffer for data (must remain valid until completion)
        size_t size;         ///< Number of bytes to transfer
        off_t offset;        ///< File offset for the operation
        void* user_data;     ///< Opaque pointer passed to completion callback

        /**
         * @brief Default constructor for container compatibility.
         */
        IORequest() = default;

        /**
         * @brief Constructs an I/O request with all fields.
         */
        IORequest(int fd_, void* buffer_, size_t size_, off_t offset_, void* user_data_ = nullptr)
            : fd(fd_)
            , buffer(buffer_)
            , size(size_)
            , offset(offset_)
            , user_data(user_data_)
        {}
    };

    /**
     * @brief Callback invoked when an I/O operation completes.
     *
     * @param request Pointer to the original request
     * @param result Bytes transferred on success, or negative errno on failure
     */
    using CompletionCallback = std::function<void(IORequest* request, ssize_t result)>;

    virtual ~AsyncIO() = default;

    // Non-copyable, non-movable (implementation-specific resources)
    AsyncIO(const AsyncIO&) = delete;
    AsyncIO& operator=(const AsyncIO&) = delete;

    /**
     * @brief Submits a batch of read operations (non-blocking).
     *
     * Requests are queued for execution. Call wait_completions() to
     * receive results. Requests must remain valid until callback fires.
     *
     * @param requests Vector of read requests to submit
     * @throws std::runtime_error if queue is full or submission fails
     */
    virtual void submit_reads(const std::vector<IORequest>& requests) = 0;

    /**
     * @brief Submits a batch of write operations (non-blocking).
     *
     * Requests are queued for execution. Call wait_completions() to
     * receive results. Buffers must remain valid until callback fires.
     *
     * @param requests Vector of write requests to submit
     * @throws std::runtime_error if queue is full or submission fails
     */
    virtual void submit_writes(const std::vector<IORequest>& requests) = 0;

    /**
     * @brief Waits for at least one completion and invokes callbacks.
     *
     * Blocks until at least one submitted operation completes, then
     * invokes the callback for each completed operation. May complete
     * multiple operations per call.
     *
     * @param callback Function to call for each completion
     * @return Number of completions processed
     */
    virtual size_t wait_completions(CompletionCallback callback) = 0;

    /**
     * @brief Waits for all pending operations to complete.
     *
     * Blocks until all submitted operations have completed.
     * No callbacks are invoked - use for cleanup/shutdown.
     */
    virtual void drain() = 0;

    /**
     * @brief Returns the number of pending operations.
     */
    virtual size_t pending_count() const = 0;

    /**
     * @brief Returns the maximum queue depth.
     */
    virtual size_t queue_depth() const = 0;

    /**
     * @brief Returns the backend name for diagnostics.
     */
    virtual const char* backend_name() const = 0;

    /**
     * @brief Factory: creates best available backend.
     *
     * Tries io_uring first (Linux 5.1+), falls back to preadv thread pool.
     * Automatically detects kernel support at runtime.
     *
     * @param queue_depth Maximum concurrent operations (default: 32)
     * @param prefer_io_uring If false, skip io_uring detection (for testing)
     * @return Unique pointer to AsyncIO implementation
     */
    static std::unique_ptr<AsyncIO> create(size_t queue_depth = 32, bool prefer_io_uring = true);

protected:
    AsyncIO() = default;
};

} // namespace Storage
