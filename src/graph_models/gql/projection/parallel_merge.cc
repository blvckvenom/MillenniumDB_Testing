/**
 * @file parallel_merge.cc
 * @brief Implementation of parallel K-way merge with async prefetching.
 */

#include "graph_models/gql/projection/parallel_merge.h"

#include <cstring>
#include <stdexcept>
#include <iostream>

namespace GQL {

ParallelMerge::ParallelMerge(
    const std::vector<RunDescriptor>& runs,
    Storage::AsyncIO* async_io,
    size_t block_size
)
    : runs_(runs)
    , async_io_(async_io)
    , block_size_(block_size)
    , records_per_block_(block_size / RECORD_SIZE)
    , total_records_(0)
{
    // Calculate total records
    for (const auto& run : runs_) {
        total_records_ += run.total_records;
    }

    // Initialize run states
    run_states_.resize(runs_.size());
    for (size_t i = 0; i < runs_.size(); ++i) {
        auto& state = run_states_[i];
        state.fd = -1;  // Will be opened in initialize_runs()
        state.records_remaining = runs_[i].total_records;
        state.next_read_offset = 0;
        state.buffer_pos = 0;
        state.prefetch_pending = false;
        state.exhausted = (runs_[i].total_records == 0);

        // Reserve buffer capacity
        state.buffer_a.reserve(records_per_block_);
        state.buffer_b.reserve(records_per_block_);
        state.active_buffer = &state.buffer_a;
        state.prefetch_buffer = &state.buffer_b;
    }
}

ParallelMerge::~ParallelMerge() {
    // Close any remaining open file descriptors
    for (auto& state : run_states_) {
        if (state.fd >= 0) {
            close(state.fd);
            state.fd = -1;
        }
    }
}

void ParallelMerge::initialize_runs() {
    // Open all files and fill initial buffers synchronously
    for (size_t i = 0; i < runs_.size(); ++i) {
        if (run_states_[i].exhausted) {
            continue;
        }

        // Open file for reading
        int fd = open(runs_[i].file_path.c_str(), O_RDONLY);
        if (fd < 0) {
            throw std::runtime_error(
                "Failed to open run file: " + runs_[i].file_path +
                " - " + strerror(errno)
            );
        }
        run_states_[i].fd = fd;

        // Fill the active buffer synchronously first
        fill_buffer_sync(i, *run_states_[i].active_buffer);
        run_states_[i].buffer_pos = 0;
    }

    // Issue limited number of initial prefetches (respect queue depth)
    // Only prefetch for the first N runs where N = queue_depth / 2
    // This leaves room for new prefetches during merge
    if (async_io_) {
        size_t max_initial_prefetches = std::min(
            runs_.size(),
            async_io_->queue_depth() / 2
        );

        for (size_t i = 0; i < max_initial_prefetches; ++i) {
            if (!run_states_[i].exhausted && run_states_[i].records_remaining > 0) {
                issue_prefetch(i);
            }
        }

        std::cout << "[ParallelMerge] Initialized " << runs_.size() << " runs, "
                  << max_initial_prefetches << " initial prefetches" << std::endl;
    }
}

void ParallelMerge::issue_prefetch(size_t run_idx) {
    auto& state = run_states_[run_idx];

    if (state.records_remaining == 0 || state.prefetch_pending) {
        return;
    }

    // Check queue capacity - leave some room for other operations
    if (async_io_->pending_count() >= async_io_->queue_depth() - 2) {
        // Queue is nearly full, skip prefetch for now
        // The sync fallback in swap_buffers will handle this
        return;
    }

    size_t to_read = std::min(state.records_remaining, records_per_block_);
    size_t bytes_to_read = to_read * RECORD_SIZE;

    // Resize prefetch buffer to accommodate the read
    state.prefetch_buffer->resize(to_read);

    // Submit async read
    Storage::AsyncIO::IORequest req{
        state.fd,
        state.prefetch_buffer->data(),
        bytes_to_read,
        state.next_read_offset,
        reinterpret_cast<void*>(run_idx)  // Use run index as user data
    };

    async_io_->submit_reads({req});
    state.prefetch_pending = true;

    // Update offset for next read
    state.next_read_offset += static_cast<off_t>(bytes_to_read);
    state.records_remaining -= to_read;
}

void ParallelMerge::swap_buffers(size_t run_idx) {
    auto& state = run_states_[run_idx];

    if (!state.prefetch_pending) {
        // No prefetch was issued, fill synchronously
        if (state.records_remaining > 0) {
            fill_buffer_sync(run_idx, *state.prefetch_buffer);
        } else {
            state.prefetch_buffer->clear();
        }
    } else {
        // Wait for the async prefetch to complete
        bool found_our_completion = false;
        while (!found_our_completion && async_io_->pending_count() > 0) {
            async_io_->wait_completions(
                [run_idx, &found_our_completion](Storage::AsyncIO::IORequest* req, ssize_t result) {
                    size_t completed_run = reinterpret_cast<size_t>(req->user_data);
                    if (completed_run == run_idx) {
                        found_our_completion = true;
                        if (result < 0) {
                            throw std::runtime_error(
                                "Async read failed: " + std::string(strerror(-static_cast<int>(result)))
                            );
                        }
                    }
                }
            );
        }
        state.prefetch_pending = false;
    }

    // Swap buffers
    std::swap(state.active_buffer, state.prefetch_buffer);
    state.buffer_pos = 0;

    // Issue prefetch for the next block
    if (state.records_remaining > 0 && async_io_) {
        issue_prefetch(run_idx);
    }
}

EdgeAggregationRecord* ParallelMerge::get_next_record(size_t run_idx) {
    auto& state = run_states_[run_idx];

    if (state.exhausted) {
        return nullptr;
    }

    // Check if we need to switch buffers
    if (state.buffer_pos >= state.active_buffer->size()) {
        // Current buffer exhausted
        if (state.records_remaining == 0 && state.prefetch_buffer->empty() && !state.prefetch_pending) {
            // No more data
            state.exhausted = true;
            return nullptr;
        }

        // Swap to prefetch buffer
        swap_buffers(run_idx);

        // Check if new buffer is empty
        if (state.active_buffer->empty()) {
            state.exhausted = true;
            return nullptr;
        }
    }

    return &(*state.active_buffer)[state.buffer_pos++];
}

void ParallelMerge::fill_buffer_sync(size_t run_idx, std::vector<EdgeAggregationRecord>& buffer) {
    auto& state = run_states_[run_idx];

    size_t to_read = std::min(state.records_remaining, records_per_block_);
    if (to_read == 0) {
        buffer.clear();
        return;
    }

    buffer.resize(to_read);

    ssize_t bytes_read = pread(
        state.fd,
        buffer.data(),
        to_read * RECORD_SIZE,
        state.next_read_offset
    );

    if (bytes_read < 0) {
        throw std::runtime_error(
            "Failed to read from run file: " + std::string(strerror(errno))
        );
    }

    size_t actual_records = static_cast<size_t>(bytes_read) / RECORD_SIZE;
    buffer.resize(actual_records);

    state.next_read_offset += bytes_read;
    state.records_remaining -= actual_records;
}

} // namespace GQL
