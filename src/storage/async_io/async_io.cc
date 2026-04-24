/**
 * @file async_io.cc
 * @brief Factory implementation for AsyncIO backends.
 *
 * Provides runtime detection of the best available I/O backend:
 * 1. io_uring (Linux 5.1+) - fastest, batch syscalls
 * 2. preadv thread pool - universal fallback
 */

#include "storage/async_io/async_io.h"

#ifdef HAS_IO_URING
#include "storage/async_io/io_uring_backend.h"
#endif

#include "storage/async_io/preadv_backend.h"

#include <iostream>

namespace Storage {

std::unique_ptr<AsyncIO> AsyncIO::create(size_t queue_depth, bool prefer_io_uring) {
#ifdef HAS_IO_URING
    if (prefer_io_uring && IOUringBackend::is_available()) {
        try {
            auto backend = std::make_unique<IOUringBackend>(queue_depth);
            std::cout << "[AsyncIO] Using io_uring backend (queue depth: "
                      << queue_depth << ")" << std::endl;
            return backend;
        } catch (const std::exception& e) {
            std::cerr << "[AsyncIO] io_uring initialization failed: " << e.what()
                      << ", falling back to preadv" << std::endl;
        }
    }
#else
    (void)prefer_io_uring;  // Suppress unused parameter warning
#endif

    // Fallback to preadv thread pool
    // Use queue_depth / 4 threads (minimum 2, maximum 8)
    size_t num_threads = std::max(size_t(2), std::min(queue_depth / 4, size_t(8)));

    std::cout << "[AsyncIO] Using preadv backend (" << num_threads << " threads, "
              << "queue depth: " << queue_depth << ")" << std::endl;

    return std::make_unique<PreadvBackend>(num_threads, queue_depth);
}

} // namespace Storage
