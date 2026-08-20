// Regression gate for the separator a directory promotes when its RIGHTMOST
// child splits during BPlusTree<N>::insert().
//
// THE DEFECT
//
// Both split paths, root and non-root, stored `record` -- the record the
// insert() call happens to be carrying -- as the promoted separator, instead of
// `split->record`, the separator the child split actually returned. `record`
// sorts at or above the whole splitting child's range, so promoting it breaks
//
//     greatest_left_record < separator <= smallest_right_record
//
// and every key in [split->record, record) is routed to the LEFT sibling on
// descent while those records physically live in the RIGHT one. They are on
// disk and no descent reaches them.
//
// It surfaces as two different faults from one bad separator, and which one a
// stranded record produces depends only on where its probe stops: a probe that
// halts inside the left leaf exceeds the (node, key, MAX) bound and yields
// nothing, while one that runs off the end follows next_leaf into the right
// sibling and is answered with the first record there -- another node's data.
// An ogbn-products write-back showed both: 24 nodes with no embedding property
// despite the procedure reporting 2,449,029 written, plus one node serving a
// different node's embedding.
//
// WHY IT LIVES HERE AND NOT WITH THE CODE THAT FOUND IT
//
// The defect was found through the embedding write-back, and a gate for it
// exists in that suite. But this is a defect in core storage, not in the GNN
// module, and the GNN suite only builds under ENABLE_GNN, which is OFF by
// default -- so a default build would compile the broken code and run no check
// against it. This file is in TEST_TARGETS and therefore always builds.
//
// WHY NO OTHER TEST CAUGHT IT
//
// Two conditions must hold at once, and the suite never met the second:
//
//   1. The records must arrive through insert(). Importing data and building
//      projections both use the bulk loaders, which assemble the tree
//      bottom-up and never call it. The only callers are the string manager's
//      free-space tree and the embedding write-back.
//   2. A DIRECTORY must actually split, which takes tens of thousands of
//      records. Every other fixture sits far below that, so the whole suite
//      passes identically with and without the fix.
//
// Two properties of the data matter and are easy to get wrong. The records go
// in STRICTLY ASCENDING, which is the real shape of the key_value_node index --
// its leading columns are (key_oid, tensor_oid) and tensor ids are
// monotonically growing blob offsets -- and which makes the rightmost child the
// splitting child at EVERY split, so the defect fires on all of them instead of
// the roughly 1-in-147 chance it has under random order. And the keys are
// deliberately incompressible: the v1 leaf format elides bytes shared across a
// page, so densely packed sequential integers fit far more than the nominal
// fanout per leaf and the tree never grows deep enough to split a directory at
// all.

#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#include <unistd.h>

#include "query/query_context.h"
#include "storage/index/bplus_tree/bplus_tree.h"
#include "storage/index/record.h"
#include "storage/page/page.h"
#include "system/buffer_manager.h"
#include "system/file_manager.h"
#include "system/system.h"

namespace fs = std::filesystem;

namespace {

int failures = 0;

void check(bool ok, const std::string& what, const std::string& detail = "") {
    std::cout << (ok ? "  ok   " : "  FAIL ") << what << std::endl;
    if (!ok) {
        if (!detail.empty()) std::cout << "         " << detail << std::endl;
        ++failures;
    }
}

// One zeroed page: exactly what BPTLeafWriter<3>::make_empty() and
// BPTDirWriter<3>'s constructor emit for a brand-new index, which is the path
// ProjectionStorage::ensure_node_property_indexes() takes when it creates the
// two property indexes the embedding write-back mutates. Written directly so
// this test does not have to pull in the bulk-load writer headers.
void write_empty_bpt_page(const fs::path& path) {
    std::vector<char> zeroed(Page::SIZE, 0);
    std::ofstream out(path, std::ios::out | std::ios::binary);
    out.write(zeroed.data(), static_cast<std::streamsize>(zeroed.size()));
}

}  // namespace

int main() {
    const fs::path db = fs::temp_directory_path()
                      / ("mdb_bpt_dir_split_" + std::to_string(static_cast<long>(::getpid())));
    fs::remove_all(db);
    fs::create_directories(db);

    const std::string index_name = "node_key_value_regress";

    constexpr uint64_t NUM_RECORDS = 120000;
    constexpr uint64_t KEY_OID     = 0xF0000000000007D0ull;  // node key id 2000

    // (node_oid, key_oid, tensor_oid), the property record the write-back
    // stores. node_oid strides widely so leaf pages stay poorly compressible;
    // the ORDER is what reproduces the defect, the stride is what makes the
    // tree deep enough to reach it.
    std::vector<Record<3>> records;
    records.reserve(NUM_RECORDS);
    uint64_t seq = 0;
    for (uint64_t i = 0; i < NUM_RECORDS; ++i) {
        // splitmix64: an incompressible stand-in for a tensor blob offset.
        seq += 0x9E3779B97F4A7C15ull;
        uint64_t z = seq;
        z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull;
        z = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
        z =  z ^ (z >> 31);

        Record<3> r;
        r[0] = 0xD400000000000000ull | (i * 0x0000000100000001ull);
        r[1] = KEY_OID;
        r[2] = 0xB100000000000000ull | (z & 0x00FFFFFFFFFFFFFFull);
        records.push_back(r);
    }

    uint64_t refused      = 0;      // insert() refused a record that was fresh
    uint64_t unreachable  = 0;      // inserted, then lost behind a bad separator
    uint64_t mismatched   = 0;      // probe answered with ANOTHER record
    uint64_t first_bad    = NUM_RECORDS;
    bool     tree_is_sane = false;
    std::string check_report;

    std::cout << "B+Tree directory-split separator regression" << std::endl;
    std::cout << "  inserting " << NUM_RECORDS << " ascending records into a fresh index"
              << std::endl;

    try {
        // The System owns the FileManager / BufferManager singletons the tree
        // reaches storage through, and is torn down before the temp dir goes.
        {
            System system(db.string(),
                          8ull * 1024 * 1024,   // str_static_size
                          8ull * 1024 * 1024,   // str_dynamic_size
                          64ull * 1024 * 1024,  // shared_buffer_size (16384 pages)
                          8ull * 1024 * 1024,   // private_buffer_size
                          8ull * 1024 * 1024,   // tensor_static_size
                          8ull * 1024 * 1024,   // tensor_dynamic_size
                          1);                   // workers

            write_empty_bpt_page(db / (index_name + ".leaf"));
            write_empty_bpt_page(db / (index_name + ".dir"));

            {
                auto version_scope = buffer_manager.init_version_editable();
                get_query_ctx().prepare(*version_scope, std::chrono::seconds(600));

                BPlusTree<3> tree(index_name);

                for (const auto& r : records) {
                    if (!tree.insert(r)) ++refused;
                }

                // The invariant itself, over every entry of every directory
                // page. The tree can check this about itself; nothing was
                // asking it to.
                std::ostringstream report;
                tree_is_sane = tree.check(report);
                check_report = report.str();

                // The user-visible consequence, probed EXACTLY the way
                // ProjectionStorage::get_node_property does it: descend on
                // (node, key, 0), accept anything up to (node, key, MAX). That
                // range is what turns the corruption into the two faults.
                bool interrupted = false;
                for (uint64_t i = 0; i < NUM_RECORDS; ++i) {
                    const Record<3>& r = records[i];
                    const Record<3> lo = { r[0], r[1], 0 };
                    const Record<3> hi = { r[0], r[1], UINT64_MAX };

                    auto iter = tree.get_range(&interrupted, lo, hi);
                    const Record<3>* found = iter.next();

                    if (found == nullptr) {
                        ++unreachable;
                        if (first_bad == NUM_RECORDS) first_bad = i;
                    } else if (*found != r) {
                        ++mismatched;
                        if (first_bad == NUM_RECORDS) first_bad = i;
                    }
                }

                version_scope->commited = true;
            }
        }
        // The System is gone, so the committed pages are flushed and the
        // directory file's size on disk is its final page count.

        const uint64_t dir_pages = fs::file_size(db / (index_name + ".dir")) / Page::SIZE;

        // The System's QueryContext lived on its stack frame: do not leave the
        // thread-local pointer dangling for whatever runs next.
        QueryContext::set_query_ctx(nullptr);

        // Vacuity guard. A fresh tree has exactly one directory page; a root
        // split appends two more and each later non-root split appends one, so
        // anything under 4 means the path this test exists to cover never ran
        // and a pass would prove nothing.
        check(dir_pages >= 4,
              "a root split AND a non-root split were exercised",
              "only " + std::to_string(dir_pages) + " directory pages -- raise NUM_RECORDS");

        check(refused == 0,
              "no unique ascending record was refused");

        check(tree_is_sane,
              "every directory separator holds greatest_left < separator <= smallest_right",
              check_report);

        check(unreachable == 0,
              "every inserted record is still findable",
              std::to_string(unreachable) + " of " + std::to_string(NUM_RECORDS)
                  + " are unreachable (first at index " + std::to_string(first_bad)
                  + "): the 'wrote them all, but the property reads NULL' fault");

        check(mismatched == 0,
              "no probe is answered with a different record",
              std::to_string(mismatched)
                  + " probes returned another record: the 'one node carries another "
                    "node's data' fault");
    } catch (const std::exception& e) {
        std::cout << "  FAIL uncaught exception: " << e.what() << std::endl;
        ++failures;
    }

    fs::remove_all(db);

    if (failures == 0) {
        std::cout << "PASS: the promoted separator keeps every record reachable across "
                     "directory splits" << std::endl;
        return 0;
    }
    std::cout << "FAIL: " << failures << " check(s) failed" << std::endl;
    return 1;
}
