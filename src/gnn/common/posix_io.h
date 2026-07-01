#pragma once

/**
 * @brief Shared POSIX I/O helpers for GNN storage components.
 *
 * Provides FdGuard (RAII file descriptor), write_all, read_all, and
 * fsync_directory. Used by FeatureMatrix, RowMapping, and PackedBatchStore.
 */

#include <cerrno>
#include <cstring>
#include <filesystem>
#include <stdexcept>
#include <string>

#include <fcntl.h>
#include <unistd.h>

namespace mdb::gnn {

/// Thread-safe strerror. Uses a local buffer to avoid static-buffer races.
inline std::string safe_strerror(int errnum) {
    char buf[256];
    // GNU strerror_r returns char* (may or may not be buf).
    // This works on glibc (Linux) which this project targets.
    return std::string(strerror_r(errnum, buf, sizeof(buf)));
}

/**
 * @brief RAII guard for POSIX file descriptors.
 *
 * Closes the fd on destruction (normal or exception). Non-copyable, non-movable.
 * Use get() to access the raw fd. Use release() to transfer ownership.
 */
class FdGuard {
public:
    explicit FdGuard(int fd) : fd_(fd) {}

    ~FdGuard() {
        if (fd_ >= 0) ::close(fd_);
    }

    FdGuard(const FdGuard&) = delete;
    FdGuard& operator=(const FdGuard&) = delete;
    FdGuard(FdGuard&&) = delete;
    FdGuard& operator=(FdGuard&&) = delete;

    int get() const { return fd_; }

    /// Release ownership — caller is responsible for closing.
    int release() {
        int fd = fd_;
        fd_ = -1;
        return fd;
    }

private:
    int fd_;
};

/**
 * @brief Write all bytes to fd, retrying on EINTR and partial writes.
 *
 * @param fd    Open file descriptor
 * @param buf   Data to write
 * @param count Number of bytes
 * @param path  File path (for error messages)
 * @throws std::runtime_error on write failure or zero-write (disk full)
 */
inline void write_all(int fd, const void* buf, size_t count, const std::string& path = "") {
    const char* p = static_cast<const char*>(buf);
    size_t remaining = count;
    while (remaining > 0) {
        ssize_t written = ::write(fd, p, remaining);
        if (written < 0) {
            if (errno == EINTR) continue;
            throw std::runtime_error(
                "write failed" + (path.empty() ? "" : " (" + path + ")") +
                ": " + std::string(safe_strerror(errno)));
        }
        if (written == 0) {
            throw std::runtime_error(
                "write returned 0 — disk full or I/O error" +
                (path.empty() ? "" : " (" + path + ")"));
        }
        p += written;
        remaining -= static_cast<size_t>(written);
    }
}

/**
 * @brief Read all bytes from fd, retrying on EINTR and partial reads.
 *
 * @param fd    Open file descriptor
 * @param buf   Destination buffer
 * @param count Number of bytes to read
 * @param path  File path (for error messages)
 * @throws std::runtime_error on read failure or unexpected EOF
 */
inline void read_all(int fd, void* buf, size_t count, const std::string& path = "") {
    char* p = static_cast<char*>(buf);
    size_t remaining = count;
    while (remaining > 0) {
        ssize_t n = ::read(fd, p, remaining);
        if (n < 0) {
            if (errno == EINTR) continue;
            throw std::runtime_error(
                "read failed" + (path.empty() ? "" : " (" + path + ")") +
                ": " + std::string(safe_strerror(errno)));
        }
        if (n == 0) {
            throw std::runtime_error(
                "unexpected EOF" + (path.empty() ? "" : " in " + path));
        }
        p += n;
        remaining -= static_cast<size_t>(n);
    }
}

/**
 * @brief Best-effort fsync of parent directory for crash consistency.
 *
 * After creating or renaming a file, the directory entry may not be durable
 * until the parent directory is also fsync'd. This is advisory — failures
 * are silently ignored.
 */
inline void fsync_directory(const std::filesystem::path& path) {
    auto parent = path.parent_path();
    if (parent.empty()) return;
    int dir_fd = ::open(parent.c_str(), O_RDONLY);
    if (dir_fd >= 0) {
        ::fsync(dir_fd);
        ::close(dir_fd);
    }
}

} // namespace mdb::gnn
