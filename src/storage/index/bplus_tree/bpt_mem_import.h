#pragma once

#include <array>
#include <bitset>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <ios>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include "misc/fatal_error.h"
#include "storage/index/bplus_tree/bplus_tree_leaf_v2.h"
#include "storage/index/bplus_tree/bpt_leaf_format.h"
#include "storage/index/record.h"
#include "storage/page/page.h"

template<std::size_t N>
class BPTLeafWriter {
public:
    static constexpr auto max_records = (Page::SIZE - 2 * sizeof(int32_t)) / (sizeof(uint64_t) * N);

    BPTLeafWriter(const std::string& filename)
    {
        file.open(filename, std::ios::out | std::ios::binary);
        buffer = new char[Page::SIZE];
    }

    ~BPTLeafWriter()
    {
        file.close();
        delete[] buffer;
    }

    void process_block(char* bytes, uint32_t size, std::bitset<N * 8> bitset, uint32_t next_block)
    {
        auto value_count = reinterpret_cast<uint32_t*>(buffer);
        auto next_leaf = reinterpret_cast<uint32_t*>(buffer + sizeof(uint32_t));

        memset(buffer, 0, Page::SIZE);
        *value_count = size;
        *next_leaf = next_block;
        std::memcpy(
            buffer + 2 * sizeof(uint32_t),
            bytes,
            N + bitset.count() + size * (sizeof(uint64_t) * N - bitset.count())
        );
        file.write(buffer, Page::SIZE);
    }

    void make_empty()
    {
        memset(buffer, 0, Page::SIZE);
        file.write(buffer, Page::SIZE);
    }

private:
    std::fstream file;

    char* buffer;
};

// ---------------------------------------------------------------------------
// BPTLeafV2Writer — bulk-load sibling of BPTLeafWriter for Spec #5 v2
// (delta + LEB128 varint) leaf pages.
//
// Unlike BPTLeafWriter — which the caller drives page-at-a-time with a
// pre-packed byte buffer + redundant-byte bitset — BPTLeafV2Writer exposes
// a streaming record-at-a-time API. The writer buffers a single 4 KB page
// internally, drives a BPTLeafV2<N> to encode each record as zigzag-delta
// varints, and transparently flushes the current page + chains a new one
// whenever the next record would overflow the v2 budget.
//
// Rationale: the v2 payload size per record is variable (1..N*10 bytes) and
// depends on the magnitude of the inter-record delta. We cannot statically
// pre-partition the record stream into "one v1-sized block per page" and
// guarantee it fits, so the writer owns the overflow decision and signals
// page boundaries back to the caller via a boolean return from
// append_record. Callers use that signal to emit a matching directory entry
// for the record that started the new page (B+Tree dir convention: the first
// leaf does NOT get a dir entry; every subsequent first-record does).
//
// Records MUST arrive in non-decreasing lexicographic order — the delta
// encoding depends on the sorted invariant. Callers also handle dedup
// upstream (BPTLeafV2Writer does not dedup).
//
// Pages are fully zero-padded to Page::SIZE = 4096 before write, matching
// BPTLeafWriter's deterministic-padding behavior.
template<std::size_t N>
class BPTLeafV2Writer {
public:
    // Open the output file and initialize the first page.
    explicit BPTLeafV2Writer(const std::string& filename)
        : file_(filename, std::ios::out | std::ios::binary)
        , page_buffer_(std::make_unique<char[]>(Page::SIZE))
        , leaf_(nullptr)
        , pages_written_(0)
        , finalized_(false)
    {
        start_new_page_();
    }

    ~BPTLeafV2Writer()
    {
        // Defensive: flush any pending page if finalize() was not called
        // (e.g., when an exception unwinds past the caller). We do not
        // rethrow from a dtor.
        try {
            if (!finalized_) {
                finalize();
            }
        } catch (...) { /* swallow */ }
        file_.close();
    }

    BPTLeafV2Writer(const BPTLeafV2Writer&)            = delete;
    BPTLeafV2Writer& operator=(const BPTLeafV2Writer&) = delete;
    BPTLeafV2Writer(BPTLeafV2Writer&&)                 = delete;
    BPTLeafV2Writer& operator=(BPTLeafV2Writer&&)      = delete;

    // Append one record.
    //
    // Returns TRUE when this call triggered a page boundary — the just-
    // appended record is the first record of a new page. The caller should
    // emit a B+Tree directory entry for this record pointing at the page
    // index returned by current_page_index(), IF the previous page existed
    // (i.e., pages_written_ > 1 after the boundary crossing — see the v1
    // convention in BPTDirWriter usage: the first leaf never gets a dir
    // entry).
    //
    // Returns FALSE when the record fit in the current page (no boundary).
    //
    // Throws std::runtime_error if a single record cannot fit on an empty
    // page (pathological N * 10-byte encoding > 4080 budget — structurally
    // impossible for N in [1..3]).
    bool append_record(const Record<N>& rec)
    {
        if (finalized_) {
            throw std::runtime_error("BPTLeafV2Writer: append_record after finalize");
        }
        if (leaf_->append_record(rec)) {
            return false;  // fit in current page
        }

        // Overflow: flush current page with a forward link to the next
        // page, then start a new page and append the record there.
        //
        // Next-page number for the ABOUT-TO-FLUSH page is "current page
        // index + 1" in the normal (non-final) case — we are opening a
        // successor right after.
        const uint32_t next_page =
            static_cast<uint32_t>(pages_written_ + 1);
        flush_current_page_(next_page);
        start_new_page_();

        if (!leaf_->append_record(rec)) {
            // Structural guarantee: one record on an empty page CANNOT
            // overflow for N in [1..3] (N * VARINT_MAX_BYTES = 30 bytes
            // max << 4080-byte page budget). This path is defensive.
            throw std::runtime_error(
                "BPTLeafV2Writer: record too large for an empty v2 page "
                "(N=" + std::to_string(N) + ")");
        }
        return true;  // caller should emit a dir entry for this record
    }

    // Flush the final page (with next_leaf = 0) and mark the writer done.
    // Must be called exactly once after the last append_record. Safe to
    // call on an empty writer (no records) — in that case it produces a
    // single empty v2 leaf page, mirroring BPTLeafWriter::make_empty's
    // "always at least one leaf" invariant.
    void finalize()
    {
        if (finalized_) return;
        // Final page: next_leaf = 0 marks the end of the leaf chain.
        flush_current_page_(/*next_block=*/0);
        finalized_ = true;
    }

    // Emit a single empty v2 leaf page and close. Equivalent to finalize()
    // on a fresh writer, provided for API symmetry with
    // BPTLeafWriter::make_empty.
    void make_empty()
    {
        if (finalized_) return;
        finalize();
    }

    // Page index assigned to the page currently being filled (0-based).
    // Useful when the caller needs to emit a dir entry pointing at the
    // just-started page after append_record returns true.
    uint32_t current_page_index() const noexcept { return pages_written_; }

    // Total number of leaf pages written so far (post-flush count).
    uint32_t num_pages() const noexcept
    {
        return finalized_ ? pages_written_ : pages_written_;
    }

private:
    void start_new_page_()
    {
        std::memset(page_buffer_.get(), 0, Page::SIZE);
        // next_leaf is patched at flush time; start with 0, overwritten by
        // flush_current_page_ before write.
        leaf_ = std::make_unique<BPTLeafV2<N>>(page_buffer_.get(),
                                               /*next_leaf=*/ 0);
    }

    void flush_current_page_(uint32_t next_block)
    {
        // The BPTLeafV2<N> writer was constructed with its next_leaf at 0.
        // We cannot reach that field except by rebuilding the header bytes
        // after flush(), so let BPTLeafV2::flush() emit its 16-byte header
        // with next_leaf=0, then overwrite bytes 8..11 with the correct
        // value. Layout (from bpt_leaf_format.h):
        //   offset 0..3  : format_version | record_width | flags | reserved
        //   offset 4..7  : value_count   (uint32 LE)
        //   offset 8..11 : next_leaf     (uint32 LE)
        //   offset 12..15: reserved2     (uint32 LE = 0)
        leaf_->flush();
        const uint32_t nb_le = next_block;  // x86_64 is LE; matches serialize_header.
        std::memcpy(page_buffer_.get() + 8, &nb_le, sizeof(uint32_t));
        file_.write(page_buffer_.get(), Page::SIZE);
        ++pages_written_;
    }

    std::fstream                     file_;
    std::unique_ptr<char[]>          page_buffer_;
    std::unique_ptr<BPTLeafV2<N>>    leaf_;
    uint32_t                         pages_written_;
    bool                             finalized_;
};

template<std::size_t N>
struct SplitData {
    const std::array<uint64_t, N>* record;
    int32_t encoded_page_number;
    bool need_split;

    SplitData(const std::array<uint64_t, N>* record, int32_t encoded_page_number, bool need_split) :
        record(record),
        encoded_page_number(encoded_page_number),
        need_split(need_split)
    { }

    SplitData() = default;
};

template<std::size_t N>
class BPTDirWriter {
private:
    std::fstream file;
    std::vector<char*> pages;

public:
    static constexpr auto max_records = (Page::SIZE - 2 * sizeof(int32_t))
                                      / (sizeof(uint64_t) * N + sizeof(int32_t));

    BPTDirWriter(const std::string& filename)
    {
        file.open(filename, std::ios::out | std::ios::binary);
        if (file.fail()) {
            WARN("Error opening file ", filename);
        }
        auto root = new char[Page::SIZE];
        memset(root, 0, Page::SIZE);
        pages.push_back(root);
    }

    ~BPTDirWriter()
    {
        for (auto page : pages) {
            file.write(page, Page::SIZE);
            delete[] page;
        }
        file.close();
    }

    uint64_t* get_keys(int32_t dir_page_number)
    {
        return reinterpret_cast<uint64_t*>(pages[dir_page_number]);
    }

    uint32_t* get_key_count(int32_t dir_page_number)
    {
        return reinterpret_cast<uint32_t*>(pages[dir_page_number] + (sizeof(uint64_t) * max_records * N));
    }

    int32_t* get_children(int32_t dir_page_number)
    {
        return reinterpret_cast<int32_t*>(
            pages[dir_page_number] + (sizeof(uint64_t) * max_records * N) + sizeof(uint32_t)
        );
    }

    SplitData<N>
        bulk_insert(const std::array<uint64_t, N>* record, int32_t dir_page_number, int32_t leaf_page_number)
    {
        uint64_t* keys = get_keys(dir_page_number);
        uint32_t* key_count = get_key_count(dir_page_number);
        int32_t* children = get_children(dir_page_number);

        SplitData<N> split_data;

        if (children[*key_count] < 0) {
            // negative number: pointer to dir
            split_data = bulk_insert(record, children[*key_count] * -1, leaf_page_number);
        } else {
            // positive number: pointer to leaf
            split_data = SplitData(record, leaf_page_number, true);
        }

        if (split_data.need_split) {
            // Case 1: no need to split this node
            if (*key_count < max_records) {
                // update key
                std::memcpy(&keys[(*key_count) * N], split_data.record->data(), N * sizeof(uint64_t));
                ++(*key_count);
                // update child
                children[*key_count] = split_data.encoded_page_number;
                return SplitData<N>(nullptr, 0, false);
            }
            // Case 2: non-root split
            else if (dir_page_number != 0)
            {
                // create new dir page
                int32_t new_page_number = pages.size();
                auto new_page = new char[Page::SIZE];
                memset(new_page, 0, Page::SIZE);
                pages.push_back(new_page);

                auto new_dir_children = get_children(new_page_number);
                auto new_dir_key_count = get_key_count(new_page_number);

                new_dir_children[0] = split_data.encoded_page_number;
                *new_dir_key_count = 0;
                return SplitData<N>(split_data.record, new_page_number * -1, true);
            }
            // Case 3: root split
            else {
                // create 2 new pages (new_lhs, new_rhs)
                int32_t lhs_page_number = pages.size();
                auto lhs_page = new char[Page::SIZE];
                memset(lhs_page, 0, Page::SIZE);
                pages.push_back(lhs_page);

                int32_t rhs_page_number = pages.size();
                auto rhs_page = new char[Page::SIZE];
                memset(rhs_page, 0, Page::SIZE);
                pages.push_back(rhs_page);

                // new_lhs has everything previous
                std::memcpy(lhs_page, pages[dir_page_number], Page::SIZE);

                // new_rhs has 0 keys and 1 record (the splitted record)
                auto rhs_key_count = get_key_count(rhs_page_number);
                auto rhs_children = get_children(rhs_page_number);
                *rhs_key_count = 0;
                rhs_children[0] = split_data.encoded_page_number;

                // new root will have the new pages as children
                std::memcpy(keys, split_data.record->data(), N * sizeof(uint64_t));
                *key_count = 1;
                children[0] = lhs_page_number * -1;
                children[1] = rhs_page_number * -1;

                return SplitData<N>(nullptr, 0, false);
            }
        } else {
            return SplitData<N>(nullptr, 0, false);
        }
    }
};
