/**
 * @file io_uring_backend.cc
 * @brief io_uring implementation of AsyncIO.
 */

#ifdef HAS_IO_URING

#include "storage/async_io/io_uring_backend.h"

#include <cerrno>
#include <cstring>
#include <stdexcept>
#include <mutex>

namespace Storage {

// Static availability check with caching
static std::once_flag availability_check_flag;
static bool io_uring_available = false;

bool IOUringBackend::is_available() {
    std::call_once(availability_check_flag, []() {
        // Try to create a minimal ring to test kernel support
        struct io_uring probe_ring;
        int ret = io_uring_queue_init(1, &probe_ring, 0);
        if (ret == 0) {
            io_uring_available = true;
            io_uring_queue_exit(&probe_ring);
        } else {
            io_uring_available = false;
        }
    });
    return io_uring_available;
}

IOUringBackend::IOUringBackend(size_t queue_depth)
    : queue_depth_(queue_depth)
{
    // Initialize the io_uring ring
    // IORING_SETUP_SQPOLL could be used for even lower latency but requires root
    int ret = io_uring_queue_init(static_cast<unsigned>(queue_depth_), &ring_, 0);
    if (ret < 0) {
        throw std::runtime_error(
            std::string("Failed to initialize io_uring: ") + strerror(-ret)
        );
    }
}

IOUringBackend::~IOUringBackend() {
    // Wait for all pending operations
    drain();

    // Clean up the ring
    io_uring_queue_exit(&ring_);
}

void IOUringBackend::submit_reads(const std::vector<IORequest>& requests) {
    submit_operations(requests, false);
}

void IOUringBackend::submit_writes(const std::vector<IORequest>& requests) {
    submit_operations(requests, true);
}

void IOUringBackend::submit_operations(const std::vector<IORequest>& requests, bool is_write) {
    if (requests.empty()) {
        return;
    }

    // Check if we have room in the submission queue
    if (pending_count_.load() + requests.size() > queue_depth_) {
        throw std::runtime_error("io_uring queue full - too many pending operations");
    }

    // Prepare submission queue entries (SQEs)
    for (const auto& req : requests) {
        struct io_uring_sqe* sqe = io_uring_get_sqe(&ring_);
        if (!sqe) {
            throw std::runtime_error("Failed to get SQE from io_uring");
        }

        if (is_write) {
            io_uring_prep_write(sqe, req.fd, req.buffer,
                               static_cast<unsigned>(req.size),
                               static_cast<__u64>(req.offset));
        } else {
            io_uring_prep_read(sqe, req.fd, req.buffer,
                              static_cast<unsigned>(req.size),
                              static_cast<__u64>(req.offset));
        }

        // Use a unique key for tracking - combine address with a counter
        // We store the request so we can retrieve it on completion
        void* tracking_key = const_cast<void*>(static_cast<const void*>(&req));
        io_uring_sqe_set_data(sqe, tracking_key);

        // Store a copy of the request for callback context
        pending_requests_[tracking_key] = req;
    }

    // Submit all prepared SQEs with a single syscall
    int submitted = io_uring_submit(&ring_);
    if (submitted < 0) {
        throw std::runtime_error(
            std::string("io_uring_submit failed: ") + strerror(-submitted)
        );
    }

    pending_count_.fetch_add(static_cast<size_t>(submitted));
}

size_t IOUringBackend::wait_completions(CompletionCallback callback) {
    if (pending_count_.load() == 0) {
        return 0;
    }

    // Wait for at least one completion
    struct io_uring_cqe* cqe;
    int ret = io_uring_wait_cqe(&ring_, &cqe);
    if (ret < 0) {
        throw std::runtime_error(
            std::string("io_uring_wait_cqe failed: ") + strerror(-ret)
        );
    }

    size_t completed = 0;

    // Process all available completions
    while (cqe != nullptr) {
        void* tracking_key = io_uring_cqe_get_data(cqe);

        auto it = pending_requests_.find(tracking_key);
        if (it != pending_requests_.end()) {
            // Invoke the callback with the result
            callback(&it->second, cqe->res);

            // Remove from tracking
            pending_requests_.erase(it);
        }

        // Mark this CQE as consumed
        io_uring_cqe_seen(&ring_, cqe);
        pending_count_.fetch_sub(1);
        completed++;

        // Try to get another completion (non-blocking)
        ret = io_uring_peek_cqe(&ring_, &cqe);
        if (ret < 0) {
            cqe = nullptr;  // No more completions available
        }
    }

    return completed;
}

void IOUringBackend::drain() {
    // Process all pending completions
    while (pending_count_.load() > 0) {
        wait_completions([](IORequest*, ssize_t) {
            // Discard results - we're just draining
        });
    }

    pending_requests_.clear();
}

} // namespace Storage

#endif // HAS_IO_URING
