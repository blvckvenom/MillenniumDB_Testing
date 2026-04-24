cmake_minimum_required(VERSION 3.12)

# Set project name and version
project(MillenniumDB VERSION 1.0.0)

# Specify the C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

# Enable export of compile commands output
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Use all cores but one for compilation
include(ProcessorCount)
ProcessorCount(NUM_CORES)
if(NUM_CORES GREATER 1)
    math(EXPR PARALLEL_LEVEL "${NUM_CORES} - 1")
    set(CMAKE_BUILD_PARALLEL_LEVEL ${PARALLEL_LEVEL})
endif()

# Set output directories
set(EXECUTABLE_OUTPUT_PATH bin)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY lib)
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY lib)

# Uncomment to disable CLI
# add_definitions(-DNO_MDB_CLI)
add_definitions(-DBOOST_ASIO_NO_DEPRECATED)

if(APPLE)
    message("https://cmake.org/cmake/help/latest/variable/APPLE.html?highlight=apple")

    set(BREW_DIRS "/opt/homebrew/opt" "/usr/local/opt")

    foreach(BREW_DIR ${BREW_DIRS})
        if(EXISTS ${BREW_DIR})
            include_directories("${BREW_DIR}/openssl/include")
            include_directories("${BREW_DIR}/ncurses/include")
                include_directories("${BREW_DIR}/icu4c/include")
                link_directories("${BREW_DIR}/openssl@3/lib")
                link_directories("${BREW_DIR}/icu4c/lib")
        endif(EXISTS ${BREW_DIR})
    endforeach(BREW_DIR)

    set(NCURSES ncurses)
    set(FS "")
    set(ICU_LIBRARIES icuuc icui18n)
else()
    set(NCURSES ncursesw)
    set(FS "stdc++fs")

    # Find ICU and its required components
    find_package(ICU COMPONENTS uc i18n REQUIRED)
endif(APPLE)

# ============================================================================
# Optional: TBB for C++17 parallel algorithms (std::execution::par)
# ============================================================================
# TBB provides parallel execution policies for std::sort and other algorithms.
# This enables 3-8× speedup for sorting operations on multi-core CPUs.
#
# Install on Ubuntu/Debian: sudo apt install libtbb-dev
# ============================================================================
set(TBB_LIBS "")
find_package(TBB QUIET)
if(TBB_FOUND)
    message(STATUS "Found TBB: ${TBB_VERSION}")
    message(STATUS "  Parallel algorithms (std::execution::par) enabled")
    add_definitions(-DHAS_TBB)
    set(TBB_LIBS TBB::tbb)
else()
    message(STATUS "TBB not found - parallel algorithms will use sequential fallback")
    message(STATUS "  Install libtbb-dev for 3-8× faster sorting on multi-core CPUs")
endif()

# ============================================================================
# Optional: io_uring for high-performance async I/O (Linux 5.1+)
# ============================================================================
# io_uring provides 3-5× speedup for external sort operations on NVMe SSDs
# by enabling batch syscalls and true async I/O. Falls back to preadv thread
# pool when unavailable.
#
# Install on Ubuntu/Debian: sudo apt install liburing-dev
# ============================================================================
set(ASYNC_IO_LIBS "")
if(NOT APPLE)
    find_library(LIBURING_LIBRARY NAMES uring)
    find_path(LIBURING_INCLUDE_DIR liburing.h)

    if(LIBURING_LIBRARY AND LIBURING_INCLUDE_DIR)
        message(STATUS "Found liburing: ${LIBURING_LIBRARY}")
        message(STATUS "  io_uring support enabled for high-performance async I/O")
        add_definitions(-DHAS_IO_URING)
        include_directories(${LIBURING_INCLUDE_DIR})
        set(ASYNC_IO_LIBS ${LIBURING_LIBRARY})
    else()
        message(STATUS "liburing not found - using preadv fallback (still 2× faster than sync I/O)")
        message(STATUS "  Install liburing-dev for best performance on Linux 5.1+")
    endif()
else()
    message(STATUS "macOS detected - using preadv fallback for async I/O")
endif()

# ============================================================================
# Optional: LZ4 for transparent spill file compression
# ============================================================================
# LZ4 compresses external-sort spill files during projection build.
# Sorted uint64 tuples (graph edges) achieve 3-5× ratio in practice,
# cutting peak disk on billion-scale projections (papers100M, MAG240M).
# Decompression is faster than memcpy so reads stay I/O-bound — compression
# is effectively free. Transparent: does not affect B+Tree layout, query
# paths, or persisted projection data.
#
# Install on Ubuntu/Debian: sudo apt install liblz4-dev
#
# See docs/superpowers/thesis_analysis/2026-04-20-projection-disk-reduction-analysis.md §3.B
# ============================================================================
set(LZ4_LIBS "")
find_library(LZ4_LIBRARY NAMES lz4)
find_path(LZ4_INCLUDE_DIR lz4frame.h)

if(LZ4_LIBRARY AND LZ4_INCLUDE_DIR)
    message(STATUS "Found LZ4: ${LZ4_LIBRARY}")
    message(STATUS "  Transparent spill compression enabled (3-5× disk savings)")
    add_definitions(-DHAS_LZ4)
    include_directories(${LZ4_INCLUDE_DIR})
    set(LZ4_LIBS ${LZ4_LIBRARY})
else()
    message(STATUS "LZ4 not found - spill files will be written uncompressed")
    message(STATUS "  Install liblz4-dev for 3-5× smaller spill files")
endif()

# Define the compiler flags
set(CMAKE_CXX_FLAGS "-Wall -Wextra -pedantic -Wno-deprecated -pthread -march=native -funroll-loops")
set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -O0 -g3 -fsanitize=undefined,address -fno-omit-frame-pointer")
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -O3 -g0")

if(PROFILE)
    add_definitions(-DBOOST_STACKTRACE_USE_ADDR2LINE)
endif(PROFILE)


# When compiling with Clang generate debug info specifically for lldb
if(CMAKE_CXX_COMPILER_ID STREQUAL Clang)
    set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -glldb -fstandalone-debug")
endif()

# Enable interprocedural optimization if supported and BUILD_TYPE is RELEASE
string(TOUPPER "${CMAKE_BUILD_TYPE}" BUILD_TYPE)
if(BUILD_TYPE STREQUAL "RELEASE")
    include(CheckIPOSupported)
    check_ipo_supported(RESULT ipo_result OUTPUT ipo_output)
    if(ipo_result)
        set(CMAKE_INTERPROCEDURAL_OPTIMIZATION TRUE)
    else()
        message(WARNING "IPO is not supported:\n${ipo_output}")
    endif(ipo_result)
endif(BUILD_TYPE STREQUAL "RELEASE")

# Add include directories
include_directories(${CMAKE_SOURCE_DIR}/src)

# Define sources
file(GLOB_RECURSE SRCS src/*.cc)
file(GLOB_RECURSE BIN_SRCS src/bin/*.cc)
file(GLOB_RECURSE TEST_SRCS src/tests/*.cc)
file(GLOB_RECURSE GNN_SRCS src/gnn/*.cc)
file(GLOB GNN_PROCEDURE_SRCS src/query/procedure/builtin/gnn_*_procedure.cc)
# .cu files are not matched by the src/*.cc glob, so only .cc exclusion is needed
file(GLOB_RECURSE GPU_SRCS src/gpu/*.cc)
# Remove unwanted files from common sources
list(REMOVE_ITEM SRCS ${BIN_SRCS})
list(REMOVE_ITEM SRCS ${TEST_SRCS})
list(REMOVE_ITEM SRCS ${GNN_SRCS})
list(REMOVE_ITEM SRCS ${GNN_PROCEDURE_SRCS})
list(REMOVE_ITEM SRCS ${GPU_SRCS})

message(STATUS "Build MillenniumDB's binaries")
# Third party dependencies
add_subdirectory(${CMAKE_SOURCE_DIR}/third_party/antlr4-runtime-4.13.1)
include_directories(${CMAKE_SOURCE_DIR}/third_party/antlr4-runtime-4.13.1/src)
include_directories(${CMAKE_SOURCE_DIR}/third_party/boost_1_82/include)
include_directories(${CMAKE_SOURCE_DIR}/third_party/libnpy)
include_directories(SYSTEM ${CMAKE_SOURCE_DIR}/third_party/ska_sort)

# Optional: GNN Module with LibTorch
option(ENABLE_GNN "Enable GNN training module with LibTorch" OFF)

# enable_testing() must be called before add_subdirectory(src/gnn) so that
# add_test() calls in the GNN subdirectory are registered with ctest.
enable_testing()

# Optional: CUDA for GPU-accelerated sort
find_package(CUDAToolkit QUIET)
if(CUDAToolkit_FOUND)
    enable_language(CUDA)
    message(STATUS "Found CUDA Toolkit: ${CUDAToolkit_VERSION}")
endif()

add_subdirectory(${CMAKE_SOURCE_DIR}/src/gpu)

if(ENABLE_GNN)
    message(STATUS "GNN module enabled - looking for LibTorch...")

    # Define preprocessor macro for conditional compilation
    add_definitions(-DENABLE_GNN)

    # Find LibTorch for the main binary
    set(CMAKE_PREFIX_PATH "${CMAKE_SOURCE_DIR}/third_party/libtorch" ${CMAKE_PREFIX_PATH})
    find_package(Torch REQUIRED)

    # Add LibTorch include directories globally for GNN procedures
    include_directories(${TORCH_INCLUDE_DIRS})

    add_subdirectory(${CMAKE_SOURCE_DIR}/src/gnn)
endif()

# Define targets
add_library(millenniumdb STATIC ${SRCS})
target_link_libraries(millenniumdb PUBLIC mdb_gpu)
target_compile_definitions(millenniumdb PUBLIC SERD_STATIC)
set(BUILD_TARGETS
    mdb
    mdb_leaf_dump
)
set(PROFILE_TARGETS
    profile-query
    profile-update
)
set(TEST_TARGETS
    available_ram-test
    compare_datetime
    iri_prefixes-test
    # native_projection_builder_test  # DISABLED: needs API updates
    # native_scanner_test  # DISABLED: needs initialized GQLModel with imported data
    # project_procedure_test  # DISABLED: needs ProcedureContext updates
    projection_storage_test
    projection_features_test
    spill_codec_test
    # property_projection_test  # DISABLED: needs pre-imported test database
    regular_path_expr_to_rpq_dfa
    scsu-test
    tensor_operations
    text_index_normalize_tokenize
    variable_set
)
# Build targets
foreach(target ${BUILD_TARGETS})
    add_executable(${target} ${CMAKE_SOURCE_DIR}/src/bin/${target}.cc)
    target_link_libraries(${target}
        millenniumdb
        antlr4_cpp_runtime
        ${FS}
        ssl
        crypto
        ${ICU_LIBRARIES}
        ${NCURSES}
        ${ASYNC_IO_LIBS}
        ${TBB_LIBS}
        ${LZ4_LIBS}
    )
    # Link GNN module and LibTorch when enabled
    if(ENABLE_GNN)
        # millenniumdb is listed again after mdb_gnn_core so that any symbol
        # defined in libmillenniumdb.a that is first referenced from
        # libmdb_gnn_core.a (e.g. Spec #4-B T4.7's TopologySnapshotReader,
        # whose only consumer under LTO is mdb_gnn/projection/topology_accessor)
        # is still resolved by the single-pass ld.bfd static-archive linker.
        target_link_libraries(${target} mdb_gnn_core ${TORCH_LIBRARIES} millenniumdb)
    endif()
endforeach(target)

# Test targets
foreach(target ${TEST_TARGETS})
    add_executable(${target} ${CMAKE_SOURCE_DIR}/src/tests/${target}.cc)
    set_target_properties(${target} PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/tests")
    target_link_libraries(${target}
        millenniumdb
        antlr4_cpp_runtime
        ${FS}
        ssl
        crypto
        ${ICU_LIBRARIES}
        ${ASYNC_IO_LIBS}
        ${TBB_LIBS}
        ${LZ4_LIBS}
    )
    # Link GNN module and LibTorch when enabled
    if(ENABLE_GNN)
        # See comment above BUILD_TARGETS loop for why millenniumdb is
        # re-linked after mdb_gnn_core under ENABLE_GNN (T4.7 link-order).
        target_link_libraries(${target} mdb_gnn_core ${TORCH_LIBRARIES} millenniumdb)
    endif()
    add_test(
        NAME ${target}
        COMMAND $<TARGET_FILE:${target}>
        WORKING_DIRECTORY tests
    )
endforeach(target)

# ============================================================================
# Radix-Partition Sort unit tests (TDD phase-gated)
# ============================================================================
# Registered here (not in a per-directory src/tests/CMakeLists.txt) to match
# the existing project layout where test targets live in the top-level file.
#
# The test source references partition_file.h which is added in Task 5.
# Until that file exists, the test target is NOT registered so the RED state
# does not pollute `ctest` or CI green contracts. Once Task 5 lands and the
# header exists, the test is registered automatically on the next CMake
# configure; tests may still fail to compile/link or fail assertions until
# Tasks 6-11 implement the RadixPartitionSort<N> methods — this is the
# expected TDD GREEN drive-in path. The main `mdb` target is unaffected
# regardless, because radix_partition_sort.h is only included from this
# test file.
#
# Precedent for EXISTS-gated test registration: src/gnn/CMakeLists.txt.
# ============================================================================
find_package(GTest QUIET)
if(GTest_FOUND AND EXISTS "${CMAKE_SOURCE_DIR}/src/graph_models/gql/projection/partition_file.h")
    add_executable(radix_partition_sort_test
        ${CMAKE_SOURCE_DIR}/src/tests/radix_partition_sort_test.cc
    )
    set_target_properties(radix_partition_sort_test PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/tests"
    )
    # GTest::gtest_main must come BEFORE millenniumdb so its main() wins
    # over the main() in src/third_party/serd/serdi.cc (which is pulled into
    # libmillenniumdb.a by the recursive src/*.cc glob). Static-archive link
    # order determines which main() actually resolves.
    target_link_libraries(radix_partition_sort_test
        GTest::gtest_main
        GTest::gtest
        millenniumdb
        antlr4_cpp_runtime
        ${FS}
        ssl
        crypto
        ${ICU_LIBRARIES}
        ${ASYNC_IO_LIBS}
        ${TBB_LIBS}
        ${LZ4_LIBS}
    )
    # Match the ENABLE_GNN-conditional link done by the TEST_TARGETS foreach
    # above — libmillenniumdb.a references vtables defined in mdb_gnn_core
    # when ENABLE_GNN=ON, so this test target would fail to link otherwise.
    if(ENABLE_GNN)
        target_link_libraries(radix_partition_sort_test mdb_gnn_core ${TORCH_LIBRARIES})
    endif()
    add_test(
        NAME RadixPartitionSortTests
        COMMAND $<TARGET_FILE:radix_partition_sort_test>
        WORKING_DIRECTORY tests
    )
elseif(NOT GTest_FOUND)
    message(STATUS "GTest not found - radix_partition_sort_test disabled")
else()
    message(STATUS "partition_file.h not present yet (Task 5 pending) - radix_partition_sort_test registration deferred")
endif()

# ============================================================================
# Serial scan pipeline unit tests (Spec #2, TDD-gated)
# ============================================================================
if(GTest_FOUND AND
   EXISTS "${CMAKE_SOURCE_DIR}/src/graph_models/gql/projection/edge_keep_bitmap.h" AND
   EXISTS "${CMAKE_SOURCE_DIR}/src/graph_models/gql/projection/edge_filter.h")
    add_executable(serial_scan_test
        ${CMAKE_SOURCE_DIR}/src/tests/serial_scan_test.cc
    )
    set_target_properties(serial_scan_test PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/tests"
    )
    # GTest::gtest_main must come BEFORE millenniumdb so its main() wins
    # over the main() in src/third_party/serd/serdi.cc (which is pulled into
    # libmillenniumdb.a by the recursive src/*.cc glob). Static-archive link
    # order determines which main() actually resolves. Same precedent as
    # radix_partition_sort_test above.
    target_link_libraries(serial_scan_test
        GTest::gtest_main
        GTest::gtest
        millenniumdb
        antlr4_cpp_runtime
        ${FS}
        ssl
        crypto
        ${ICU_LIBRARIES}
        ${ASYNC_IO_LIBS}
        ${TBB_LIBS}
        ${LZ4_LIBS}
    )
    # Match the ENABLE_GNN-conditional link done by the TEST_TARGETS foreach
    # above — libmillenniumdb.a references vtables defined in mdb_gnn_core
    # when ENABLE_GNN=ON, so this test target would fail to link otherwise.
    if(ENABLE_GNN)
        target_link_libraries(serial_scan_test mdb_gnn_core ${TORCH_LIBRARIES})
    endif()
    add_test(
        NAME SerialScanTests
        COMMAND $<TARGET_FILE:serial_scan_test>
        WORKING_DIRECTORY tests
    )
endif()

# ============================================================================
# IndexSet unit tests (Spec #3 T3.3)
# ============================================================================
# Registers the index_set_test target when GTest is available and the
# corresponding header exists. Mirrors the EXISTS-gated registration
# precedent of radix_partition_sort_test / serial_scan_test so that the
# test only wires in after its sources land.
if(GTest_FOUND AND
   EXISTS "${CMAKE_SOURCE_DIR}/src/graph_models/gql/projection/index_set.h")
    add_executable(index_set_test
        ${CMAKE_SOURCE_DIR}/src/tests/index_set_test.cc
    )
    set_target_properties(index_set_test PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/tests"
    )
    # GTest::gtest_main must come BEFORE millenniumdb so its main() wins
    # over the main() in src/third_party/serd/serdi.cc (which is pulled into
    # libmillenniumdb.a by the recursive src/*.cc glob). Static-archive link
    # order determines which main() actually resolves. Same precedent as
    # radix_partition_sort_test / serial_scan_test above.
    target_link_libraries(index_set_test
        GTest::gtest_main
        GTest::gtest
        millenniumdb
        antlr4_cpp_runtime
        ${FS}
        ssl
        crypto
        ${ICU_LIBRARIES}
        ${ASYNC_IO_LIBS}
        ${TBB_LIBS}
        ${LZ4_LIBS}
    )
    # Match the ENABLE_GNN-conditional link done by the TEST_TARGETS foreach
    # above — libmillenniumdb.a references vtables defined in mdb_gnn_core
    # when ENABLE_GNN=ON, so this test target would fail to link otherwise.
    if(ENABLE_GNN)
        target_link_libraries(index_set_test mdb_gnn_core ${TORCH_LIBRARIES})
    endif()
    add_test(
        NAME IndexSetTests
        COMMAND $<TARGET_FILE:index_set_test>
        WORKING_DIRECTORY tests
    )
endif()

# ============================================================================
# Missing-index query-layer diagnostic unit tests (Spec #3 T3.9)
# ============================================================================
# Registers projection_missing_index_test when GTest is available and the
# corresponding source exists. Validates the `projection_index_name` /
# `minimum_preset_for` helpers added to index_set.{h,cc} in T3.9, plus the
# format of the QueryException message produced by GQLModel getters when a
# query hits an index elided by the active IndexSet preset.
if(GTest_FOUND AND
   EXISTS "${CMAKE_SOURCE_DIR}/src/tests/projection_missing_index_test.cc")
    add_executable(projection_missing_index_test
        ${CMAKE_SOURCE_DIR}/src/tests/projection_missing_index_test.cc
    )
    set_target_properties(projection_missing_index_test PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/tests"
    )
    target_link_libraries(projection_missing_index_test
        GTest::gtest_main
        GTest::gtest
        millenniumdb
        antlr4_cpp_runtime
        ${FS}
        ssl
        crypto
        ${ICU_LIBRARIES}
        ${ASYNC_IO_LIBS}
        ${TBB_LIBS}
        ${LZ4_LIBS}
    )
    if(ENABLE_GNN)
        target_link_libraries(projection_missing_index_test mdb_gnn_core ${TORCH_LIBRARIES})
    endif()
    add_test(
        NAME ProjectionMissingIndexTests
        COMMAND $<TARGET_FILE:projection_missing_index_test>
        WORKING_DIRECTORY tests
    )
endif()

# ============================================================================
# TopologySnapshot CSR file format unit tests (Spec #4-B T4.3)
# ============================================================================
# Registers topology_snapshot_format_test when GTest is available and the
# corresponding header exists. Mirrors the EXISTS-gated registration
# precedent of radix_partition_sort_test / serial_scan_test / index_set_test
# so that the test only wires in after its source lands.
if(GTest_FOUND AND
   EXISTS "${CMAKE_SOURCE_DIR}/src/graph_models/gql/projection/topology_snapshot.h")
    add_executable(topology_snapshot_format_test
        ${CMAKE_SOURCE_DIR}/src/tests/topology_snapshot_format_test.cc
    )
    set_target_properties(topology_snapshot_format_test PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/tests"
    )
    # GTest::gtest_main must come BEFORE millenniumdb so its main() wins
    # over the main() in src/third_party/serd/serdi.cc (which is pulled into
    # libmillenniumdb.a by the recursive src/*.cc glob). Static-archive link
    # order determines which main() actually resolves. Same precedent as
    # radix_partition_sort_test / serial_scan_test / index_set_test above.
    target_link_libraries(topology_snapshot_format_test
        GTest::gtest_main
        GTest::gtest
        millenniumdb
        antlr4_cpp_runtime
        ${FS}
        ssl
        crypto
        ${ICU_LIBRARIES}
        ${ASYNC_IO_LIBS}
        ${TBB_LIBS}
        ${LZ4_LIBS}
    )
    if(ENABLE_GNN)
        target_link_libraries(topology_snapshot_format_test mdb_gnn_core ${TORCH_LIBRARIES})
    endif()
    add_test(
        NAME TopologySnapshotFormatTests
        COMMAND $<TARGET_FILE:topology_snapshot_format_test>
        WORKING_DIRECTORY tests
    )
endif()

# ============================================================================
# B+Tree v2 leaf header format unit tests (Spec #5 T5.3)
# ============================================================================
# Registers bpt_leaf_v2_format_test when GTest is available and the source
# file exists. Follows the precedent of topology_snapshot_format_test.
if(GTest_FOUND AND
   EXISTS "${CMAKE_SOURCE_DIR}/src/tests/bpt_leaf_v2_format_test.cc"
   AND TARGET GTest::gtest_main)
    add_executable(bpt_leaf_v2_format_test
        ${CMAKE_SOURCE_DIR}/src/tests/bpt_leaf_v2_format_test.cc
    )
    set_target_properties(bpt_leaf_v2_format_test PROPERTIES
        CXX_STANDARD 17
        CXX_STANDARD_REQUIRED ON
        RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/tests"
    )
    # GTest::gtest_main must come BEFORE millenniumdb so its main() wins
    # over the main() in src/third_party/serd/serdi.cc (which is pulled into
    # libmillenniumdb.a by the recursive src/*.cc glob). Same precedent as
    # topology_snapshot_format_test above.
    target_link_libraries(bpt_leaf_v2_format_test
        GTest::gtest_main
        GTest::gtest
        millenniumdb
        antlr4_cpp_runtime
        ${FS}
        ssl
        crypto
        ${ICU_LIBRARIES}
        ${ASYNC_IO_LIBS}
        ${TBB_LIBS}
        ${LZ4_LIBS}
    )
    if(ENABLE_GNN)
        target_link_libraries(bpt_leaf_v2_format_test mdb_gnn_core ${TORCH_LIBRARIES})
    endif()
    add_test(
        NAME bpt_leaf_v2_format_test
        COMMAND $<TARGET_FILE:bpt_leaf_v2_format_test>
        WORKING_DIRECTORY tests
    )
endif()

# ============================================================================
# B+Tree v3 (CSR_HYBRID) leaf header format unit tests (Spec #8 T8.3)
# ============================================================================
# Registers bpt_leaf_csr_format_test when GTest is available and the source
# file exists. Follows the precedent of bpt_leaf_v2_format_test (Spec #5 T5.3).
if(GTest_FOUND AND
   EXISTS "${CMAKE_SOURCE_DIR}/src/tests/bpt_leaf_csr_format_test.cc"
   AND TARGET GTest::gtest_main)
    add_executable(bpt_leaf_csr_format_test
        ${CMAKE_SOURCE_DIR}/src/tests/bpt_leaf_csr_format_test.cc
    )
    set_target_properties(bpt_leaf_csr_format_test PROPERTIES
        CXX_STANDARD 17
        CXX_STANDARD_REQUIRED ON
        RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/tests"
    )
    # GTest::gtest_main must come BEFORE millenniumdb so its main() wins
    # over the main() in src/third_party/serd/serdi.cc (which is pulled into
    # libmillenniumdb.a by the recursive src/*.cc glob). Same precedent as
    # bpt_leaf_v2_format_test above.
    target_link_libraries(bpt_leaf_csr_format_test
        GTest::gtest_main
        GTest::gtest
        millenniumdb
        antlr4_cpp_runtime
        ${FS}
        ssl
        crypto
        ${ICU_LIBRARIES}
        ${ASYNC_IO_LIBS}
        ${TBB_LIBS}
        ${LZ4_LIBS}
    )
    if(ENABLE_GNN)
        target_link_libraries(bpt_leaf_csr_format_test mdb_gnn_core ${TORCH_LIBRARIES})
    endif()
    add_test(
        NAME bpt_leaf_csr_format_test
        COMMAND $<TARGET_FILE:bpt_leaf_csr_format_test>
        WORKING_DIRECTORY tests
    )
endif()

# ============================================================================
# B+Tree v3 CSR_HYBRID leaf reader unit tests (Spec #8 T8.4)
# ============================================================================
# Registers bpt_leaf_csr_reader_test when GTest is available and the source file
# exists. Follows the precedent of bpt_leaf_csr_format_test.
if(GTest_FOUND AND
   EXISTS "${CMAKE_SOURCE_DIR}/src/tests/bpt_leaf_csr_reader_test.cc"
   AND TARGET GTest::gtest_main)
    add_executable(bpt_leaf_csr_reader_test
        ${CMAKE_SOURCE_DIR}/src/tests/bpt_leaf_csr_reader_test.cc
    )
    set_target_properties(bpt_leaf_csr_reader_test PROPERTIES
        CXX_STANDARD 17
        CXX_STANDARD_REQUIRED ON
        RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/tests"
    )
    # GTest::gtest_main must come BEFORE millenniumdb so its main() wins
    # over the main() in src/third_party/serd/serdi.cc (which is pulled into
    # libmillenniumdb.a by the recursive src/*.cc glob). Same precedent as
    # bpt_leaf_csr_format_test above.
    target_link_libraries(bpt_leaf_csr_reader_test
        GTest::gtest_main
        GTest::gtest
        millenniumdb
        antlr4_cpp_runtime
        ${FS}
        ssl
        crypto
        ${ICU_LIBRARIES}
        ${ASYNC_IO_LIBS}
        ${TBB_LIBS}
        ${LZ4_LIBS}
    )
    if(ENABLE_GNN)
        target_link_libraries(bpt_leaf_csr_reader_test mdb_gnn_core ${TORCH_LIBRARIES})
    endif()
    add_test(
        NAME bpt_leaf_csr_reader_test
        COMMAND $<TARGET_FILE:bpt_leaf_csr_reader_test>
        WORKING_DIRECTORY tests
    )
endif()

# ============================================================================
# B+Tree LEB128 varint codec unit tests (Spec #5 T5.4)
# ============================================================================
# Registers varint_test when GTest is available and the source file exists.
# Follows the precedent of bpt_leaf_v2_format_test.
if(GTest_FOUND AND
   EXISTS "${CMAKE_SOURCE_DIR}/src/tests/varint_test.cc"
   AND TARGET GTest::gtest_main)
    add_executable(varint_test
        ${CMAKE_SOURCE_DIR}/src/tests/varint_test.cc
    )
    set_target_properties(varint_test PROPERTIES
        CXX_STANDARD 17
        CXX_STANDARD_REQUIRED ON
        RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/tests"
    )
    # GTest::gtest_main must come BEFORE millenniumdb so its main() wins
    # over the main() in src/third_party/serd/serdi.cc (which is pulled into
    # libmillenniumdb.a by the recursive src/*.cc glob). Same precedent as
    # bpt_leaf_v2_format_test above.
    target_link_libraries(varint_test
        GTest::gtest_main
        GTest::gtest
        millenniumdb
        antlr4_cpp_runtime
        ${FS}
        ssl
        crypto
        ${ICU_LIBRARIES}
        ${ASYNC_IO_LIBS}
        ${TBB_LIBS}
        ${LZ4_LIBS}
    )
    if(ENABLE_GNN)
        target_link_libraries(varint_test mdb_gnn_core ${TORCH_LIBRARIES})
    endif()
    add_test(
        NAME varint_test
        COMMAND $<TARGET_FILE:varint_test>
        WORKING_DIRECTORY tests
    )
endif()

# ============================================================================
# B+Tree v2 leaf writer unit tests (Spec #5 T5.7)
# ============================================================================
# Registers bpt_leaf_v2_writer_test when GTest is available and the source
# file exists. Follows the precedent of varint_test.
if(GTest_FOUND AND
   EXISTS "${CMAKE_SOURCE_DIR}/src/tests/bpt_leaf_v2_writer_test.cc"
   AND TARGET GTest::gtest_main)
    add_executable(bpt_leaf_v2_writer_test
        ${CMAKE_SOURCE_DIR}/src/tests/bpt_leaf_v2_writer_test.cc
    )
    set_target_properties(bpt_leaf_v2_writer_test PROPERTIES
        CXX_STANDARD 17
        CXX_STANDARD_REQUIRED ON
        RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/tests"
    )
    # GTest::gtest_main must come BEFORE millenniumdb so its main() wins
    # over the main() in src/third_party/serd/serdi.cc (which is pulled into
    # libmillenniumdb.a by the recursive src/*.cc glob). Same precedent as
    # varint_test above.
    target_link_libraries(bpt_leaf_v2_writer_test
        GTest::gtest_main
        GTest::gtest
        millenniumdb
        antlr4_cpp_runtime
        ${FS}
        ssl
        crypto
        ${ICU_LIBRARIES}
        ${ASYNC_IO_LIBS}
        ${TBB_LIBS}
        ${LZ4_LIBS}
    )
    if(ENABLE_GNN)
        target_link_libraries(bpt_leaf_v2_writer_test mdb_gnn_core ${TORCH_LIBRARIES})
    endif()
    add_test(
        NAME bpt_leaf_v2_writer_test
        COMMAND $<TARGET_FILE:bpt_leaf_v2_writer_test>
        WORKING_DIRECTORY tests
    )
endif()

# ============================================================================
# B+Tree v2 leaf reader unit tests (Spec #5 T5.8)
# ============================================================================
# Registers bpt_leaf_v2_reader_test when GTest is available and the source file
# exists. Follows the precedent of bpt_leaf_v2_writer_test.
if(GTest_FOUND AND
   EXISTS "${CMAKE_SOURCE_DIR}/src/tests/bpt_leaf_v2_reader_test.cc"
   AND TARGET GTest::gtest_main)
    add_executable(bpt_leaf_v2_reader_test
        ${CMAKE_SOURCE_DIR}/src/tests/bpt_leaf_v2_reader_test.cc
    )
    set_target_properties(bpt_leaf_v2_reader_test PROPERTIES
        CXX_STANDARD 17
        CXX_STANDARD_REQUIRED ON
        RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/tests"
    )
    # GTest::gtest_main must come BEFORE millenniumdb so its main() wins
    # over the main() in src/third_party/serd/serdi.cc (which is pulled into
    # libmillenniumdb.a by the recursive src/*.cc glob). Same precedent as
    # bpt_leaf_v2_writer_test above.
    target_link_libraries(bpt_leaf_v2_reader_test
        GTest::gtest_main
        GTest::gtest
        millenniumdb
        antlr4_cpp_runtime
        ${FS}
        ssl
        crypto
        ${ICU_LIBRARIES}
        ${ASYNC_IO_LIBS}
        ${TBB_LIBS}
        ${LZ4_LIBS}
    )
    if(ENABLE_GNN)
        target_link_libraries(bpt_leaf_v2_reader_test mdb_gnn_core ${TORCH_LIBRARIES})
    endif()
    add_test(
        NAME bpt_leaf_v2_reader_test
        COMMAND $<TARGET_FILE:bpt_leaf_v2_reader_test>
        WORKING_DIRECTORY tests
    )
endif()

# ============================================================================
# B+Tree polymorphic leaf dispatch unit tests (Spec #5 T5.9)
# ============================================================================
# Registers bpt_iter_dispatch_test when GTest is available and the source file
# exists. Follows the precedent of bpt_leaf_v2_reader_test.
if(GTest_FOUND AND
   EXISTS "${CMAKE_SOURCE_DIR}/src/tests/bpt_iter_dispatch_test.cc"
   AND TARGET GTest::gtest_main)
    add_executable(bpt_iter_dispatch_test
        ${CMAKE_SOURCE_DIR}/src/tests/bpt_iter_dispatch_test.cc
    )
    set_target_properties(bpt_iter_dispatch_test PROPERTIES
        CXX_STANDARD 17
        CXX_STANDARD_REQUIRED ON
        RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/tests"
    )
    # GTest::gtest_main must come BEFORE millenniumdb so its main() wins
    # over the main() in src/third_party/serd/serdi.cc (which is pulled into
    # libmillenniumdb.a by the recursive src/*.cc glob). Same precedent as
    # bpt_leaf_v2_reader_test above.
    target_link_libraries(bpt_iter_dispatch_test
        GTest::gtest_main
        GTest::gtest
        millenniumdb
        antlr4_cpp_runtime
        ${FS}
        ssl
        crypto
        ${ICU_LIBRARIES}
        ${ASYNC_IO_LIBS}
        ${TBB_LIBS}
        ${LZ4_LIBS}
    )
    if(ENABLE_GNN)
        target_link_libraries(bpt_iter_dispatch_test mdb_gnn_core ${TORCH_LIBRARIES})
    endif()
    add_test(
        NAME bpt_iter_dispatch_test
        COMMAND $<TARGET_FILE:bpt_iter_dispatch_test>
        WORKING_DIRECTORY tests
    )
endif()

# Registers projection_catalog_v5_test when GTest is available and the source
# file exists. Follows the precedent of bpt_iter_dispatch_test.
if(GTest_FOUND AND
   EXISTS "${CMAKE_SOURCE_DIR}/src/tests/projection_catalog_v5_test.cc"
   AND TARGET GTest::gtest_main)
    add_executable(projection_catalog_v5_test
        ${CMAKE_SOURCE_DIR}/src/tests/projection_catalog_v5_test.cc
    )
    set_target_properties(projection_catalog_v5_test PROPERTIES
        CXX_STANDARD 17
        CXX_STANDARD_REQUIRED ON
        RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/tests"
    )
    # GTest::gtest_main must come BEFORE millenniumdb so its main() wins
    # over the main() in src/third_party/serd/serdi.cc (which is pulled into
    # libmillenniumdb.a by the recursive src/*.cc glob). Same precedent as
    # bpt_iter_dispatch_test above.
    target_link_libraries(projection_catalog_v5_test
        GTest::gtest_main
        GTest::gtest
        millenniumdb
        antlr4_cpp_runtime
        ${FS}
        ssl
        crypto
        ${ICU_LIBRARIES}
        ${ASYNC_IO_LIBS}
        ${TBB_LIBS}
        ${LZ4_LIBS}
    )
    if(ENABLE_GNN)
        target_link_libraries(projection_catalog_v5_test mdb_gnn_core ${TORCH_LIBRARIES})
    endif()
    add_test(
        NAME projection_catalog_v5_test
        COMMAND $<TARGET_FILE:projection_catalog_v5_test>
        WORKING_DIRECTORY tests
    )
endif()

# Registers projection_catalog_v6_test when GTest is available and the source
# file exists. Follows the precedent of projection_catalog_v5_test.
if(GTest_FOUND AND
   EXISTS "${CMAKE_SOURCE_DIR}/src/tests/projection_catalog_v6_test.cc"
   AND TARGET GTest::gtest_main)
    add_executable(projection_catalog_v6_test
        ${CMAKE_SOURCE_DIR}/src/tests/projection_catalog_v6_test.cc
    )
    set_target_properties(projection_catalog_v6_test PROPERTIES
        CXX_STANDARD 17
        CXX_STANDARD_REQUIRED ON
        RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/tests"
    )
    # GTest::gtest_main must come BEFORE millenniumdb so its main() wins
    # over the main() in src/third_party/serd/serdi.cc (which is pulled into
    # libmillenniumdb.a by the recursive src/*.cc glob). Same precedent as
    # projection_catalog_v5_test above.
    target_link_libraries(projection_catalog_v6_test
        GTest::gtest_main
        GTest::gtest
        millenniumdb
        antlr4_cpp_runtime
        ${FS}
        ssl
        crypto
        ${ICU_LIBRARIES}
        ${ASYNC_IO_LIBS}
        ${TBB_LIBS}
        ${LZ4_LIBS}
    )
    if(ENABLE_GNN)
        target_link_libraries(projection_catalog_v6_test mdb_gnn_core ${TORCH_LIBRARIES})
    endif()
    add_test(
        NAME projection_catalog_v6_test
        COMMAND $<TARGET_FILE:projection_catalog_v6_test>
        WORKING_DIRECTORY tests
    )
endif()

# Registers projection_leaffmt_config_test when GTest is available and the
# source file exists. Follows the precedent of projection_catalog_v5_test.
if(GTest_FOUND AND
   EXISTS "${CMAKE_SOURCE_DIR}/src/tests/projection_leaffmt_config_test.cc"
   AND TARGET GTest::gtest_main)
    add_executable(projection_leaffmt_config_test
        ${CMAKE_SOURCE_DIR}/src/tests/projection_leaffmt_config_test.cc
    )
    set_target_properties(projection_leaffmt_config_test PROPERTIES
        CXX_STANDARD 17
        CXX_STANDARD_REQUIRED ON
        RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/tests"
    )
    # GTest::gtest_main must come BEFORE millenniumdb so its main() wins
    # over the main() in src/third_party/serd/serdi.cc (which is pulled into
    # libmillenniumdb.a by the recursive src/*.cc glob). Same precedent as
    # projection_catalog_v5_test above.
    target_link_libraries(projection_leaffmt_config_test
        GTest::gtest_main
        GTest::gtest
        millenniumdb
        antlr4_cpp_runtime
        ${FS}
        ssl
        crypto
        ${ICU_LIBRARIES}
        ${ASYNC_IO_LIBS}
        ${TBB_LIBS}
        ${LZ4_LIBS}
    )
    if(ENABLE_GNN)
        target_link_libraries(projection_leaffmt_config_test mdb_gnn_core ${TORCH_LIBRARIES})
    endif()
    add_test(
        NAME projection_leaffmt_config_test
        COMMAND $<TARGET_FILE:projection_leaffmt_config_test>
        WORKING_DIRECTORY tests
    )
endif()

# Registers bpt_leaf_v2_fuzz_test when GTest is available and the source file
# exists. Follows the precedent of projection_leaffmt_config_test.
if(GTest_FOUND AND
   EXISTS "${CMAKE_SOURCE_DIR}/src/tests/bpt_leaf_v2_fuzz_test.cc"
   AND TARGET GTest::gtest_main)
    add_executable(bpt_leaf_v2_fuzz_test
        ${CMAKE_SOURCE_DIR}/src/tests/bpt_leaf_v2_fuzz_test.cc
    )
    set_target_properties(bpt_leaf_v2_fuzz_test PROPERTIES
        CXX_STANDARD 17
        CXX_STANDARD_REQUIRED ON
        RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/tests"
    )
    # GTest::gtest_main must come BEFORE millenniumdb so its main() wins
    # over the main() in src/third_party/serd/serdi.cc (which is pulled into
    # libmillenniumdb.a by the recursive src/*.cc glob). Same precedent as
    # projection_leaffmt_config_test above.
    target_link_libraries(bpt_leaf_v2_fuzz_test
        GTest::gtest_main
        GTest::gtest
        millenniumdb
        antlr4_cpp_runtime
        ${FS}
        ssl
        crypto
        ${ICU_LIBRARIES}
        ${ASYNC_IO_LIBS}
        ${TBB_LIBS}
        ${LZ4_LIBS}
    )
    if(ENABLE_GNN)
        target_link_libraries(bpt_leaf_v2_fuzz_test mdb_gnn_core ${TORCH_LIBRARIES})
    endif()
    add_test(
        NAME bpt_leaf_v2_fuzz_test
        COMMAND $<TARGET_FILE:bpt_leaf_v2_fuzz_test>
        WORKING_DIRECTORY tests
    )
    # Release wall-clock is ~51 s on benito_pc; Debug/ASan is typically 3-5x
    # slower (see plan doc §T5.14 "Verification" step 4). 360 s accommodates
    # both.
    set_tests_properties(bpt_leaf_v2_fuzz_test PROPERTIES TIMEOUT 360)
endif()

# ============================================================================
# TopologySnapshot CSR writer unit tests (Spec #4-B T4.4)
# ============================================================================
# Same EXISTS-gated registration pattern as topology_snapshot_format_test.
# The writer depends on the T4.3 header plus the new .cc — gate on both.
if(GTest_FOUND AND
   EXISTS "${CMAKE_SOURCE_DIR}/src/graph_models/gql/projection/topology_snapshot_writer.cc")
    add_executable(topology_snapshot_writer_test
        ${CMAKE_SOURCE_DIR}/src/tests/topology_snapshot_writer_test.cc
    )
    set_target_properties(topology_snapshot_writer_test PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/tests"
    )
    # GTest::gtest_main must come BEFORE millenniumdb so its main() wins
    # over the main() in src/third_party/serd/serdi.cc (which is pulled into
    # libmillenniumdb.a by the recursive src/*.cc glob). Static-archive link
    # order determines which main() actually resolves. Same precedent as the
    # format_test registration above.
    target_link_libraries(topology_snapshot_writer_test
        GTest::gtest_main
        GTest::gtest
        millenniumdb
        antlr4_cpp_runtime
        ${FS}
        ssl
        crypto
        ${ICU_LIBRARIES}
        ${ASYNC_IO_LIBS}
        ${TBB_LIBS}
        ${LZ4_LIBS}
    )
    if(ENABLE_GNN)
        target_link_libraries(topology_snapshot_writer_test mdb_gnn_core ${TORCH_LIBRARIES})
    endif()
    add_test(
        NAME TopologySnapshotWriterTests
        COMMAND $<TARGET_FILE:topology_snapshot_writer_test>
        WORKING_DIRECTORY tests
    )
endif()

# ============================================================================
# TopologySnapshot CSR reader unit tests (Spec #4-B T4.5)
# ============================================================================
# Same EXISTS-gated registration pattern as topology_snapshot_writer_test.
# The reader depends on the T4.3 header plus the new .cc — gate on both.
if(GTest_FOUND AND
   EXISTS "${CMAKE_SOURCE_DIR}/src/graph_models/gql/projection/topology_snapshot_reader.cc")
    add_executable(topology_snapshot_reader_test
        ${CMAKE_SOURCE_DIR}/src/tests/topology_snapshot_reader_test.cc
    )
    set_target_properties(topology_snapshot_reader_test PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/tests"
    )
    # GTest::gtest_main must come BEFORE millenniumdb so its main() wins
    # over the main() in src/third_party/serd/serdi.cc (which is pulled into
    # libmillenniumdb.a by the recursive src/*.cc glob). Static-archive link
    # order determines which main() actually resolves. Same precedent as the
    # writer_test registration above.
    target_link_libraries(topology_snapshot_reader_test
        GTest::gtest_main
        GTest::gtest
        millenniumdb
        antlr4_cpp_runtime
        ${FS}
        ssl
        crypto
        ${ICU_LIBRARIES}
        ${ASYNC_IO_LIBS}
        ${TBB_LIBS}
        ${LZ4_LIBS}
    )
    if(ENABLE_GNN)
        target_link_libraries(topology_snapshot_reader_test mdb_gnn_core ${TORCH_LIBRARIES})
    endif()
    add_test(
        NAME TopologySnapshotReaderTests
        COMMAND $<TARGET_FILE:topology_snapshot_reader_test>
        WORKING_DIRECTORY tests
    )
endif()

# ============================================================================
# NativeProjectionBuilder × TopologySnapshot integration test (Spec #4-B T4.6)
# ============================================================================
# EXISTS-gated so the target is only configured once the integration source
# lands. Shares the GTest linking recipe with the other projection tests.
if(GTest_FOUND AND
   EXISTS "${CMAKE_SOURCE_DIR}/src/tests/native_projection_builder_topology_snapshot_test.cc")
    add_executable(native_projection_builder_topology_snapshot_test
        ${CMAKE_SOURCE_DIR}/src/tests/native_projection_builder_topology_snapshot_test.cc
    )
    set_target_properties(native_projection_builder_topology_snapshot_test PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/tests"
    )
    target_link_libraries(native_projection_builder_topology_snapshot_test
        GTest::gtest_main
        GTest::gtest
        millenniumdb
        antlr4_cpp_runtime
        ${FS}
        ssl
        crypto
        ${ICU_LIBRARIES}
        ${ASYNC_IO_LIBS}
        ${TBB_LIBS}
        ${LZ4_LIBS}
    )
    if(ENABLE_GNN)
        target_link_libraries(native_projection_builder_topology_snapshot_test mdb_gnn_core ${TORCH_LIBRARIES})
    endif()
    add_test(
        NAME NativeProjectionBuilderTopologySnapshotTests
        COMMAND $<TARGET_FILE:native_projection_builder_topology_snapshot_test>
        WORKING_DIRECTORY tests
    )
endif()

# ============================================================================
# TopologyAccessor × CSR fast-path integration test (Spec #4-B T4.7)
# ============================================================================
# Verifies the mmap fast-path integration in
# src/gnn/projection/topology_accessor.cc and the per-direction fallback
# behavior when sidecars are absent / stale / corrupt. Requires ENABLE_GNN
# because the accessor itself lives in mdb_gnn_core.
if(GTest_FOUND AND ENABLE_GNN AND
   EXISTS "${CMAKE_SOURCE_DIR}/src/tests/topology_accessor_csr_path_test.cc")
    add_executable(topology_accessor_csr_path_test
        ${CMAKE_SOURCE_DIR}/src/tests/topology_accessor_csr_path_test.cc
    )
    set_target_properties(topology_accessor_csr_path_test PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/tests"
        BUILD_RPATH "${CMAKE_SOURCE_DIR}/third_party/libtorch/lib"
        INSTALL_RPATH "${CMAKE_SOURCE_DIR}/third_party/libtorch/lib"
    )
    # GTest::gtest_main must come BEFORE millenniumdb so its main() wins
    # over serdi's main(), and mdb_gnn_core must come before the trailing
    # millenniumdb so the T4.7 link-order fix applies here too.
    target_link_libraries(topology_accessor_csr_path_test
        GTest::gtest_main
        GTest::gtest
        mdb_gnn_core
        millenniumdb
        antlr4_cpp_runtime
        ${FS}
        ssl
        crypto
        ${ICU_LIBRARIES}
        ${ASYNC_IO_LIBS}
        ${TBB_LIBS}
        ${LZ4_LIBS}
        ${TORCH_LIBRARIES}
    )
    add_test(
        NAME TopologyAccessorCsrPathTests
        COMMAND $<TARGET_FILE:topology_accessor_csr_path_test>
        WORKING_DIRECTORY tests
    )
endif()

# ============================================================================
# Multi-layer k-hop sampling determinism tests (Spec #4-B T4.11)
# ============================================================================
# Complements T4.7 (single-layer sample_neighbors determinism). Exercises
# TopologyAccessor::sample_khop_neighbors under the 3×N matrix of
# orientations × fanout shapes, asserting bit-identical output between the
# CSR fast-path and the B+Tree fallback for fixed RNG seeds. Also covers
# the mixed-path scenario where topology_rev.csr is deleted after build.
if(GTest_FOUND AND ENABLE_GNN AND
   EXISTS "${CMAKE_SOURCE_DIR}/src/tests/gnn_sampling_snapshot_test.cc")
    add_executable(gnn_sampling_snapshot_test
        ${CMAKE_SOURCE_DIR}/src/tests/gnn_sampling_snapshot_test.cc
    )
    set_target_properties(gnn_sampling_snapshot_test PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/tests"
        BUILD_RPATH "${CMAKE_SOURCE_DIR}/third_party/libtorch/lib"
        INSTALL_RPATH "${CMAKE_SOURCE_DIR}/third_party/libtorch/lib"
    )
    # Link order mirrors topology_accessor_csr_path_test: gtest_main before
    # millenniumdb (wins over serdi's main), and mdb_gnn_core before the
    # trailing millenniumdb so symbols from the GNN module resolve first.
    target_link_libraries(gnn_sampling_snapshot_test
        GTest::gtest_main
        GTest::gtest
        mdb_gnn_core
        millenniumdb
        antlr4_cpp_runtime
        ${FS}
        ssl
        crypto
        ${ICU_LIBRARIES}
        ${ASYNC_IO_LIBS}
        ${TBB_LIBS}
        ${LZ4_LIBS}
        ${TORCH_LIBRARIES}
    )
    add_test(
        NAME GnnSamplingSnapshotTests
        COMMAND $<TARGET_FILE:gnn_sampling_snapshot_test>
        WORKING_DIRECTORY tests
    )
endif()

# ============================================================================
# gnn_build_topology_snapshot procedure unit tests (Spec #4-B T4.9)
# ============================================================================
# EXISTS-gated so the target is only configured once the procedure source
# lands. Shares the GTest linking recipe with the other projection tests.
# Requires ENABLE_GNN because the procedure itself lives in mdb_gnn_core.
if(GTest_FOUND AND ENABLE_GNN AND
   EXISTS "${CMAKE_SOURCE_DIR}/src/query/procedure/builtin/gnn_build_topology_snapshot_procedure.cc")
    add_executable(gnn_build_topology_snapshot_procedure_test
        ${CMAKE_SOURCE_DIR}/src/tests/gnn_build_topology_snapshot_procedure_test.cc
    )
    set_target_properties(gnn_build_topology_snapshot_procedure_test PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/tests"
        BUILD_RPATH "${CMAKE_SOURCE_DIR}/third_party/libtorch/lib"
        INSTALL_RPATH "${CMAKE_SOURCE_DIR}/third_party/libtorch/lib"
    )
    # GTest::gtest_main must come BEFORE millenniumdb so its main() wins
    # over serdi's main(). mdb_gnn_core carries the procedure impl, and
    # millenniumdb provides ProjectionStorage + BPlusTree — the T4.7
    # link-order rule (mdb_gnn_core before trailing millenniumdb) applies.
    target_link_libraries(gnn_build_topology_snapshot_procedure_test
        GTest::gtest_main
        GTest::gtest
        mdb_gnn_core
        millenniumdb
        antlr4_cpp_runtime
        ${FS}
        ssl
        crypto
        ${ICU_LIBRARIES}
        ${ASYNC_IO_LIBS}
        ${TBB_LIBS}
        ${LZ4_LIBS}
        ${TORCH_LIBRARIES}
    )
    add_test(
        NAME GnnBuildTopologySnapshotProcedureTests
        COMMAND $<TARGET_FILE:gnn_build_topology_snapshot_procedure_test>
        WORKING_DIRECTORY tests
    )
endif()

# ============================================================================
# Isolated TopologyAccessor micro-benchmark (Spec #4-B T4.17)
# ============================================================================
# Standalone binary that measures `TopologyAccessor::sample_khop_neighbors`
# wall-clock on a preexisting projection under two modes (CSR sidecars
# present vs parked). Built only when ENABLE_GNN because it links the GNN
# module. No gtest dependency — it's a plain CLI binary but mirrors the
# link recipe used by the snapshot/accessor unit tests.
if(ENABLE_GNN AND
   EXISTS "${CMAKE_SOURCE_DIR}/src/bench/bench_topology_micro.cc")
    add_executable(bench_topology_micro
        ${CMAKE_SOURCE_DIR}/src/bench/bench_topology_micro.cc
    )
    set_target_properties(bench_topology_micro PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
        BUILD_RPATH "${CMAKE_SOURCE_DIR}/third_party/libtorch/lib"
        INSTALL_RPATH "${CMAKE_SOURCE_DIR}/third_party/libtorch/lib"
    )
    # Link order mirrors gnn_sampling_snapshot_test: mdb_gnn_core before
    # the trailing millenniumdb so GNN symbols resolve first.
    target_link_libraries(bench_topology_micro
        mdb_gnn_core
        millenniumdb
        antlr4_cpp_runtime
        ${FS}
        ssl
        crypto
        ${ICU_LIBRARIES}
        ${ASYNC_IO_LIBS}
        ${TBB_LIBS}
        ${LZ4_LIBS}
        ${TORCH_LIBRARIES}
    )
endif()

if(PROFILE)
    # Profile targets
    foreach(target ${PROFILE_TARGETS})
        add_executable(${target} ${CMAKE_SOURCE_DIR}/src/bin/${target}.cc)
        target_link_libraries(${target}
            millenniumdb
            antlr4_cpp_runtime
            ${FS}
            ssl
            crypto
            ${ICU_LIBRARIES}
            dl
            -Wl,-no-as-needed
            -no-pie
            -fno-pie
            tcmalloc
            profiler
        )
    endforeach(target)
endif(PROFILE)

install(TARGETS ${BUILD_TARGETS} RUNTIME DESTINATION bin)
