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

# USE_TCMALLOC — link tcmalloc_minimal as the process-wide allocator (opt-in,
# default OFF). The offline-sampling loop builds + tears down ~11K small result
# vectors + 5 unordered_{map,set} per batch across up to 16 workers, hammering
# glibc ptmalloc2; tcmalloc's thread-local free lists relieve that with ZERO
# source change and bit-identical output. Default OFF keeps the legacy allocator
# so a build can A/B it. Requires `apt install google-perftools
# libgoogle-perftools-dev` (provides libtcmalloc_minimal).
option(USE_TCMALLOC "Link tcmalloc_minimal as the global allocator" OFF)
if(USE_TCMALLOC)
    find_library(TCMALLOC_LIB NAMES tcmalloc_minimal tcmalloc)
    if(NOT TCMALLOC_LIB)
        message(FATAL_ERROR
            "USE_TCMALLOC=ON but libtcmalloc_minimal not found — "
            "install: apt install google-perftools libgoogle-perftools-dev")
    endif()
    message(STATUS "USE_TCMALLOC: linking ${TCMALLOC_LIB} as the global allocator")
endif()


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

# ============================================================================
# Optional: GNN Module with LibTorch
# ============================================================================
# ENABLE_GNN builds the in-database GNN training module (mdb_gnn_core) and
# compiles the gnn_* GQL procedures (graph_project feature ingestion,
# gnn_offline_sample, gnn_build_feature_store, gnn_train, gnn_predict, ...).
#
# It is OFF by default because it requires LibTorch (expected under
# third_party/libtorch) and lengthens the build considerably. With
# ENABLE_GNN=OFF the gnn_* procedures do not exist in the server, so calling
# one fails at procedure lookup — this is the most common cause of "missing
# feature" reports on fresh builds. If a gnn_* call fails, check this flag
# first:
#
#   cmake -B build/Release -D CMAKE_BUILD_TYPE=Release -D ENABLE_GNN=ON
#
# CUDA is independent of this flag: it is detected separately below, and the
# GNN module falls back to CPU execution when no GPU toolkit is present.
# ============================================================================
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
# millenniumdb and mdb_gnn_core are mutually dependent static archives:
# projection code in millenniumdb calls mdb::gnn::RowMapping/FeatureMatrix
# (feature ingestion), while mdb_gnn_core consumes the GQL projection layer
# (TopologySnapshotReader, ProjectionStorage). Declaring the cycle makes
# CMake repeat both archives on consumer link lines, which single-pass
# linking requires; Release -O3 masked the cycle by dead-stripping the
# back-references that -O0 Debug objects keep.
# (The inverse edge, mdb_gnn_core -> millenniumdb, is declared in
# src/gnn/CMakeLists.txt — CMP0079 requires each half in its own directory.)
if(ENABLE_GNN)
    target_link_libraries(millenniumdb PUBLIC mdb_gnn_core)
endif()
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
    bpt_dir_split_separator_test
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
        # libmdb_gnn_core.a (e.g. TopologySnapshotReader,
        # whose only consumer under LTO is mdb_gnn/projection/topology_accessor)
        # is still resolved by the single-pass ld.bfd static-archive linker.
        target_link_libraries(${target} mdb_gnn_core ${TORCH_LIBRARIES} millenniumdb)
    endif()
    # tcmalloc must come LAST + --no-as-needed so its malloc/free strong symbols
    # win over libc and the lib is not dropped by the linker. Overrides the
    # allocator process-wide for the executable.
    if(USE_TCMALLOC)
        target_link_libraries(${target} -Wl,--no-as-needed ${TCMALLOC_LIB})
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
        # re-linked after mdb_gnn_core under ENABLE_GNN (single-pass
        # static-archive link order).
        target_link_libraries(${target} mdb_gnn_core ${TORCH_LIBRARIES} millenniumdb)
    endif()
    add_test(
        NAME ${target}
        COMMAND $<TARGET_FILE:${target}>
        WORKING_DIRECTORY tests
    )
endforeach(target)

# ============================================================================
# Radix-Partition Sort unit tests
# ============================================================================
# Registered here (not in a per-directory src/tests/CMakeLists.txt) to match
# the existing project layout where test targets live in the top-level file.
#
# The test source references partition_file.h. While that header is absent
# from the tree the test target is NOT registered, so a missing-source state
# does not pollute `ctest` or CI; once the header exists the test registers
# automatically on the next CMake configure. The main `mdb` target is
# unaffected either way, because radix_partition_sort.h is only included
# from this test file.
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
    message(STATUS "partition_file.h not present - radix_partition_sort_test registration deferred")
endif()

# ============================================================================
# CLASSIC external-sort spill/merge roundtrip unit tests
# ----------------------------------------------------------------------------
# Pins the correctness of GQL::ExternalRecordSort<N>'s spill -> sort_run_files
# -> merge_runs path (the CLASSIC projection sorter backend). Guards the
# task_arena fan-out cap and the K-way merge against future refactors.
# Mirrors the radix_partition_sort_test target above.
# ============================================================================
if(GTest_FOUND)
    add_executable(external_record_sort_test
        ${CMAKE_SOURCE_DIR}/src/tests/external_record_sort_test.cc
    )
    set_target_properties(external_record_sort_test PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/tests"
    )
    # GTest::gtest_main must come BEFORE millenniumdb so its main() wins over
    # the main() in src/third_party/serd/serdi.cc (same constraint as the
    # radix_partition_sort_test target above).
    target_link_libraries(external_record_sort_test
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
        target_link_libraries(external_record_sort_test mdb_gnn_core ${TORCH_LIBRARIES})
    endif()
    add_test(
        NAME ExternalRecordSortSpillTests
        COMMAND $<TARGET_FILE:external_record_sort_test>
        WORKING_DIRECTORY tests
    )
endif()

# ============================================================================
# Parallel edge-scan abort/exception-propagation unit tests
# ============================================================================
if(GTest_FOUND)
    add_executable(native_scanner_parallel_abort_test
        ${CMAKE_SOURCE_DIR}/src/tests/native_scanner_parallel_abort_test.cc
    )
    set_target_properties(native_scanner_parallel_abort_test PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/tests"
    )
    # Same link-order constraint as radix_partition_sort_test: gtest_main
    # must precede millenniumdb so its main() wins over serdi.cc's.
    target_link_libraries(native_scanner_parallel_abort_test
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
        target_link_libraries(native_scanner_parallel_abort_test mdb_gnn_core ${TORCH_LIBRARIES})
    endif()
    add_test(
        NAME NativeScannerParallelAbortTests
        COMMAND $<TARGET_FILE:native_scanner_parallel_abort_test>
        WORKING_DIRECTORY tests
    )
endif()

# ============================================================================
# Partitioned edge-scan seam unit tests (builder-level PARALLEL scan mode)
# ============================================================================
if(GTest_FOUND)
    add_executable(native_scanner_partitioned_test
        ${CMAKE_SOURCE_DIR}/src/tests/native_scanner_partitioned_test.cc
    )
    set_target_properties(native_scanner_partitioned_test PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/tests"
    )
    # gtest_main must precede millenniumdb so its main() wins over serdi.cc's.
    target_link_libraries(native_scanner_partitioned_test
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
        target_link_libraries(native_scanner_partitioned_test mdb_gnn_core ${TORCH_LIBRARIES})
    endif()
    add_test(
        NAME NativeScannerPartitionedTests
        COMMAND $<TARGET_FILE:native_scanner_partitioned_test>
        WORKING_DIRECTORY tests
    )
endif()

# ============================================================================
# Serial scan pipeline unit tests
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
# IndexSet unit tests
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
# Missing-index query-layer diagnostic unit tests
# ============================================================================
# Registers projection_missing_index_test when GTest is available and the
# corresponding source exists. Validates the `projection_index_name` /
# `minimum_preset_for` helpers in index_set.{h,cc}, plus the
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
# TopologySnapshot CSR file format unit tests
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
# Symmetric topology per-node merge unit tests
# ============================================================================
# Registers topology_symmetric_merge_test when GTest is available and the
# header-only merge helper exists. Same EXISTS-gated precedent as
# topology_snapshot_format_test above.
if(GTest_FOUND AND
   EXISTS "${CMAKE_SOURCE_DIR}/src/graph_models/gql/projection/topology_symmetric_merge.h")
    add_executable(topology_symmetric_merge_test
        ${CMAKE_SOURCE_DIR}/src/tests/topology_symmetric_merge_test.cc
    )
    set_target_properties(topology_symmetric_merge_test PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/tests"
    )
    # GTest::gtest_main before millenniumdb so its main() wins over the serdi.cc
    # main() pulled into libmillenniumdb.a by the recursive src/*.cc glob.
    target_link_libraries(topology_symmetric_merge_test
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
        target_link_libraries(topology_symmetric_merge_test mdb_gnn_core ${TORCH_LIBRARIES})
    endif()
    add_test(
        NAME TopologySymmetricMergeTests
        COMMAND $<TARGET_FILE:topology_symmetric_merge_test>
        WORKING_DIRECTORY tests
    )
    set(ASAN_SUPP_FILE "${CMAKE_SOURCE_DIR}/tests/asan_suppressions.txt")
    if(EXISTS "${ASAN_SUPP_FILE}")
        set_property(TEST TopologySymmetricMergeTests PROPERTY
            ENVIRONMENT "LSAN_OPTIONS=suppressions=${ASAN_SUPP_FILE}")
    endif()
endif()

# ============================================================================
# B+Tree v2 leaf header format unit tests
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
# B+Tree v3 (CSR_HYBRID) leaf header format unit tests
# ============================================================================
# Registers bpt_leaf_csr_format_test when GTest is available and the source
# file exists. Follows the precedent of bpt_leaf_v2_format_test.
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
# B+Tree v3 CSR_HYBRID leaf reader unit tests
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
# B+Tree v3 CSR_HYBRID leaf writer unit tests
# ============================================================================
# Registers bpt_leaf_csr_writer_test when GTest is available and the source
# file exists. Follows the precedent of bpt_leaf_csr_reader_test.
if(GTest_FOUND AND
   EXISTS "${CMAKE_SOURCE_DIR}/src/tests/bpt_leaf_csr_writer_test.cc"
   AND TARGET GTest::gtest_main)
    add_executable(bpt_leaf_csr_writer_test
        ${CMAKE_SOURCE_DIR}/src/tests/bpt_leaf_csr_writer_test.cc
    )
    set_target_properties(bpt_leaf_csr_writer_test PROPERTIES
        CXX_STANDARD 17
        CXX_STANDARD_REQUIRED ON
        RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/tests"
    )
    # GTest::gtest_main must come BEFORE millenniumdb so its main() wins
    # over the main() in src/third_party/serd/serdi.cc (which is pulled into
    # libmillenniumdb.a by the recursive src/*.cc glob). Same precedent as
    # bpt_leaf_csr_reader_test above.
    target_link_libraries(bpt_leaf_csr_writer_test
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
        target_link_libraries(bpt_leaf_csr_writer_test mdb_gnn_core ${TORCH_LIBRARIES})
    endif()
    add_test(
        NAME bpt_leaf_csr_writer_test
        COMMAND $<TARGET_FILE:bpt_leaf_csr_writer_test>
        WORKING_DIRECTORY tests
    )
endif()

# ============================================================================
# B+Tree v3 CSR_HYBRID multipage integration test
# ============================================================================
# Registers bpt_leaf_csr_multipage_test — end-to-end writer + reader +
# ContinuationTag dispatch across hub chains. Guards the hub-chain fixes
# (reader tuple counting across continuation pages, ContinuationTag
# construction, and writer leaf-chain linkage).
if(GTest_FOUND AND
   EXISTS "${CMAKE_SOURCE_DIR}/src/tests/bpt_leaf_csr_multipage_test.cc"
   AND TARGET GTest::gtest_main)
    add_executable(bpt_leaf_csr_multipage_test
        ${CMAKE_SOURCE_DIR}/src/tests/bpt_leaf_csr_multipage_test.cc
    )
    set_target_properties(bpt_leaf_csr_multipage_test PROPERTIES
        CXX_STANDARD 17
        CXX_STANDARD_REQUIRED ON
        RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/tests"
    )
    target_link_libraries(bpt_leaf_csr_multipage_test
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
        target_link_libraries(bpt_leaf_csr_multipage_test mdb_gnn_core ${TORCH_LIBRARIES})
    endif()
    add_test(
        NAME bpt_leaf_csr_multipage_test
        COMMAND $<TARGET_FILE:bpt_leaf_csr_multipage_test>
        WORKING_DIRECTORY tests
    )
endif()

# ============================================================================
# B+Tree v3 CSR_HYBRID leaf fuzz tests
# ============================================================================
# Registers bpt_leaf_csr_fuzz_test when GTest is available and the source file
# exists. Follows the precedent of bpt_leaf_v2_fuzz_test.
if(GTest_FOUND AND
   EXISTS "${CMAKE_SOURCE_DIR}/src/tests/bpt_leaf_csr_fuzz_test.cc"
   AND TARGET GTest::gtest_main)
    add_executable(bpt_leaf_csr_fuzz_test
        ${CMAKE_SOURCE_DIR}/src/tests/bpt_leaf_csr_fuzz_test.cc
    )
    set_target_properties(bpt_leaf_csr_fuzz_test PROPERTIES
        CXX_STANDARD 17
        CXX_STANDARD_REQUIRED ON
        RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/tests"
    )
    # GTest::gtest_main must come BEFORE millenniumdb so its main() wins
    # over the main() in src/third_party/serd/serdi.cc (pulled into
    # libmillenniumdb.a by the recursive src/*.cc glob). Same precedent as
    # bpt_leaf_csr_writer_test above.
    target_link_libraries(bpt_leaf_csr_fuzz_test
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
        target_link_libraries(bpt_leaf_csr_fuzz_test mdb_gnn_core ${TORCH_LIBRARIES})
    endif()
    add_test(
        NAME bpt_leaf_csr_fuzz_test
        COMMAND $<TARGET_FILE:bpt_leaf_csr_fuzz_test>
        WORKING_DIRECTORY tests
    )
    # The 500k Release iterations take on the order of a minute on a
    # desktop-class CPU; Debug/ASan is typically 3-5x slower. 180 s
    # accommodates Release with margin; Debug/ASan runs may need more.
    set_tests_properties(bpt_leaf_csr_fuzz_test PROPERTIES TIMEOUT 180)
endif()

# ============================================================================
# B+Tree LEB128 varint codec unit tests
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
# B+Tree v2 leaf writer unit tests
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
# B+Tree v2 leaf reader unit tests
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
# B+Tree polymorphic leaf dispatch unit tests
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

# Registers projection_graph_storage_config_test when GTest is available and
# the source file exists. Follows the precedent of
# projection_leaffmt_config_test.
if(GTest_FOUND AND
   EXISTS "${CMAKE_SOURCE_DIR}/src/tests/projection_graph_storage_config_test.cc"
   AND TARGET GTest::gtest_main)
    add_executable(projection_graph_storage_config_test
        ${CMAKE_SOURCE_DIR}/src/tests/projection_graph_storage_config_test.cc
    )
    set_target_properties(projection_graph_storage_config_test PROPERTIES
        CXX_STANDARD 17
        CXX_STANDARD_REQUIRED ON
        RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/tests"
    )
    # GTest::gtest_main must come BEFORE millenniumdb so its main() wins
    # over the main() in src/third_party/serd/serdi.cc (which is pulled into
    # libmillenniumdb.a by the recursive src/*.cc glob). Same precedent as
    # projection_catalog_v6_test above.
    target_link_libraries(projection_graph_storage_config_test
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
        target_link_libraries(projection_graph_storage_config_test
            mdb_gnn_core ${TORCH_LIBRARIES})
    endif()
    add_test(
        NAME projection_graph_storage_config_test
        COMMAND $<TARGET_FILE:projection_graph_storage_config_test>
        WORKING_DIRECTORY tests
    )
endif()

# Registers graph_storage_integration_test when GTest is available and the
# source file exists. Integration-level regressions for the CSR_HYBRID write
# dispatch (edge indexes), leaf-format invariants, and sidecar supersession:
# a CSR_HYBRID projection must not also emit topology_*.csr sidecars, since
# its v3 leaves already provide the O(1) neighbour access the sidecar exists
# to give.
if(GTest_FOUND AND
   EXISTS "${CMAKE_SOURCE_DIR}/src/tests/graph_storage_integration_test.cc"
   AND TARGET GTest::gtest_main)
    add_executable(graph_storage_integration_test
        ${CMAKE_SOURCE_DIR}/src/tests/graph_storage_integration_test.cc
    )
    set_target_properties(graph_storage_integration_test PROPERTIES
        CXX_STANDARD 17
        CXX_STANDARD_REQUIRED ON
        RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/tests"
    )
    target_link_libraries(graph_storage_integration_test
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
        target_link_libraries(graph_storage_integration_test
            mdb_gnn_core ${TORCH_LIBRARIES})
    endif()
    add_test(
        NAME graph_storage_integration_test
        COMMAND $<TARGET_FILE:graph_storage_integration_test>
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
    # Release wall-clock is on the order of a minute on a desktop-class
    # CPU; Debug/ASan is typically 3-5x slower. 360 s accommodates both.
    set_tests_properties(bpt_leaf_v2_fuzz_test PROPERTIES TIMEOUT 360)
endif()

# ============================================================================
# TopologySnapshot CSR writer unit tests
# ============================================================================
# Same EXISTS-gated registration pattern as topology_snapshot_format_test.
# The writer depends on the topology_snapshot.h format header plus its own
# .cc, so the gate keys on the .cc landing.
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
# TopologySnapshot CSR reader unit tests
# ============================================================================
# Same EXISTS-gated registration pattern as topology_snapshot_writer_test.
# The reader depends on the topology_snapshot.h format header plus its own
# .cc, so the gate keys on the .cc landing.
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
# TopologyWalkProfiler unit tests
# ============================================================================
# EXISTS-gated on both the test file AND the profiler source so the
# target only configures once the profiler implementation is present.
if(GTest_FOUND AND ENABLE_GNN AND
   EXISTS "${CMAKE_SOURCE_DIR}/src/tests/topology_walk_profiler_test.cc" AND
   EXISTS "${CMAKE_SOURCE_DIR}/src/gnn/projection/topology_walk_profiler.cc")
    add_executable(topology_walk_profiler_test
        ${CMAKE_SOURCE_DIR}/src/tests/topology_walk_profiler_test.cc
    )
    set_target_properties(topology_walk_profiler_test PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/tests"
    )
    target_link_libraries(topology_walk_profiler_test
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
        mdb_gnn_core
        ${TORCH_LIBRARIES}
    )
    add_test(
        NAME TopologyWalkProfilerTests
        COMMAND $<TARGET_FILE:topology_walk_profiler_test>
        WORKING_DIRECTORY tests
    )
endif()

# ============================================================================
# NativeProjectionBuilder × TopologySnapshot integration test
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
# TopologyAccessor × CSR fast-path integration test
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
    # millenniumdb so the static-archive link-order fix applies here too.
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
# TopologyAccessor adjacency cache unit tests
# ============================================================================
# Validates TopologyAccessor's in-memory adjacency cache mode. Cache vs
# B+Tree paths must produce bit-identical neighbour sets and bit-identical
# k-hop sampling output under a fixed RNG seed. Mirrors the linker recipe
# of topology_accessor_csr_path_test (gtest_main first, mdb_gnn_core before
# millenniumdb).
if(GTest_FOUND AND ENABLE_GNN AND
   EXISTS "${CMAKE_SOURCE_DIR}/src/tests/topology_accessor_adjacency_cache_test.cc")
    add_executable(topology_accessor_adjacency_cache_test
        ${CMAKE_SOURCE_DIR}/src/tests/topology_accessor_adjacency_cache_test.cc
    )
    set_target_properties(topology_accessor_adjacency_cache_test PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/tests"
        BUILD_RPATH "${CMAKE_SOURCE_DIR}/third_party/libtorch/lib"
        INSTALL_RPATH "${CMAKE_SOURCE_DIR}/third_party/libtorch/lib"
    )
    target_link_libraries(topology_accessor_adjacency_cache_test
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
        NAME TopologyAccessorAdjacencyCacheTests
        COMMAND $<TARGET_FILE:topology_accessor_adjacency_cache_test>
        WORKING_DIRECTORY tests
    )
endif()

# ============================================================================
# Topology frequency profiler tests
# ============================================================================
# Validates the frequency profiler that the Four-Level Topology Store uses
# to seed L1 / L2 / L3 tier assignment. Covers the cold-start (degree
# proxy) path, warm-start rejection of malformed node_counts.bin files,
# mixed orientations, and the greedy tier sizing helper. Linker recipe mirrors topology_accessor_adjacency_cache_test.
if(GTest_FOUND AND ENABLE_GNN AND
   EXISTS "${CMAKE_SOURCE_DIR}/src/tests/topology_frequency_profiler_test.cc")
    add_executable(topology_frequency_profiler_test
        ${CMAKE_SOURCE_DIR}/src/tests/topology_frequency_profiler_test.cc
    )
    set_target_properties(topology_frequency_profiler_test PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/tests"
        BUILD_RPATH "${CMAKE_SOURCE_DIR}/third_party/libtorch/lib"
        INSTALL_RPATH "${CMAKE_SOURCE_DIR}/third_party/libtorch/lib"
    )
    target_link_libraries(topology_frequency_profiler_test
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
        NAME TopologyFrequencyProfilerTests
        COMMAND $<TARGET_FILE:topology_frequency_profiler_test>
        WORKING_DIRECTORY tests
    )
    # Suppress known CUDA driver leak (cuInit ~408 bytes) so ctest exits 0.
    set(ASAN_SUPP_FILE "${CMAKE_SOURCE_DIR}/tests/asan_suppressions.txt")
    if(EXISTS "${ASAN_SUPP_FILE}")
        set_property(TEST TopologyFrequencyProfilerTests PROPERTY
            ENVIRONMENT "LSAN_OPTIONS=suppressions=${ASAN_SUPP_FILE}")
    endif()
endif()

# ============================================================================
# L1HashCache tests
# ============================================================================
# Pure data-structure tests for the hot-tier RAM hash cache (lifted from
# the monolithic adjacency cache): insert
# respects tier_assignment, round-trip parity, large-degree safety, and the
# total_bytes() contract anchored against kL1NodeFixedOverhead /
# kL1PerEdgeBytes from topology_frequency_profiler.h.
if(GTest_FOUND AND ENABLE_GNN AND
   EXISTS "${CMAKE_SOURCE_DIR}/src/tests/l1_hash_cache_test.cc")
    add_executable(l1_hash_cache_test
        ${CMAKE_SOURCE_DIR}/src/tests/l1_hash_cache_test.cc
    )
    set_target_properties(l1_hash_cache_test PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/tests"
        BUILD_RPATH "${CMAKE_SOURCE_DIR}/third_party/libtorch/lib"
        INSTALL_RPATH "${CMAKE_SOURCE_DIR}/third_party/libtorch/lib"
    )
    target_link_libraries(l1_hash_cache_test
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
        NAME L1HashCacheTests
        COMMAND $<TARGET_FILE:l1_hash_cache_test>
        WORKING_DIRECTORY tests
    )
    set(ASAN_SUPP_FILE "${CMAKE_SOURCE_DIR}/tests/asan_suppressions.txt")
    if(EXISTS "${ASAN_SUPP_FILE}")
        set_property(TEST L1HashCacheTests PROPERTY
            ENVIRONMENT "LSAN_OPTIONS=suppressions=${ASAN_SUPP_FILE}")
    endif()
endif()

# ============================================================================
# L2CompactCsr tests
# ============================================================================
# Pure data-structure tests for the warm-tier compact CSR cache: build/lookup
# round-trip, out-of-order insert identity, freeze immutability, isolated
# nodes, total_bytes contract.
if(GTest_FOUND AND ENABLE_GNN AND
   EXISTS "${CMAKE_SOURCE_DIR}/src/tests/l2_compact_csr_test.cc")
    add_executable(l2_compact_csr_test
        ${CMAKE_SOURCE_DIR}/src/tests/l2_compact_csr_test.cc
    )
    set_target_properties(l2_compact_csr_test PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/tests"
        BUILD_RPATH "${CMAKE_SOURCE_DIR}/third_party/libtorch/lib"
        INSTALL_RPATH "${CMAKE_SOURCE_DIR}/third_party/libtorch/lib"
    )
    target_link_libraries(l2_compact_csr_test
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
        NAME L2CompactCsrTests
        COMMAND $<TARGET_FILE:l2_compact_csr_test>
        WORKING_DIRECTORY tests
    )
    set(ASAN_SUPP_FILE "${CMAKE_SOURCE_DIR}/tests/asan_suppressions.txt")
    if(EXISTS "${ASAN_SUPP_FILE}")
        set_property(TEST L2CompactCsrTests PROPERTY
            ENVIRONMENT "LSAN_OPTIONS=suppressions=${ASAN_SUPP_FILE}")
    endif()
endif()

# sampling_backend_plan tests — pure GPU/CPU sampling-backend decision table.
if(GTest_FOUND AND ENABLE_GNN AND
   EXISTS "${CMAKE_SOURCE_DIR}/src/tests/sampling_backend_plan_test.cc")
    add_executable(sampling_backend_plan_test
        ${CMAKE_SOURCE_DIR}/src/tests/sampling_backend_plan_test.cc
    )
    set_target_properties(sampling_backend_plan_test PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/tests"
        BUILD_RPATH "${CMAKE_SOURCE_DIR}/third_party/libtorch/lib"
        INSTALL_RPATH "${CMAKE_SOURCE_DIR}/third_party/libtorch/lib"
    )
    target_link_libraries(sampling_backend_plan_test
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
        NAME SamplingBackendPlanTests
        COMMAND $<TARGET_FILE:sampling_backend_plan_test>
        WORKING_DIRECTORY tests
    )
    set(ASAN_SUPP_FILE "${CMAKE_SOURCE_DIR}/tests/asan_suppressions.txt")
    if(EXISTS "${ASAN_SUPP_FILE}")
        set_property(TEST SamplingBackendPlanTests PROPERTY
            ENVIRONMENT "LSAN_OPTIONS=suppressions=${ASAN_SUPP_FILE}")
    endif()
endif()

# pinned_topology_view tests — cudaHostRegister round-trip for the GPU substrate.
if(GTest_FOUND AND ENABLE_GNN AND
   EXISTS "${CMAKE_SOURCE_DIR}/src/tests/pinned_topology_view_test.cc")
    add_executable(pinned_topology_view_test
        ${CMAKE_SOURCE_DIR}/src/tests/pinned_topology_view_test.cc
    )
    set_target_properties(pinned_topology_view_test PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/tests"
        BUILD_RPATH "${CMAKE_SOURCE_DIR}/third_party/libtorch/lib"
        INSTALL_RPATH "${CMAKE_SOURCE_DIR}/third_party/libtorch/lib"
    )
    target_link_libraries(pinned_topology_view_test
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
        NAME PinnedTopologyViewTests
        COMMAND $<TARGET_FILE:pinned_topology_view_test>
        WORKING_DIRECTORY tests
    )
    set(ASAN_SUPP_FILE "${CMAKE_SOURCE_DIR}/tests/asan_suppressions.txt")
    if(EXISTS "${ASAN_SUPP_FILE}")
        set_property(TEST PinnedTopologyViewTests PROPERTY
            ENVIRONMENT "LSAN_OPTIONS=suppressions=${ASAN_SUPP_FILE}")
    endif()
endif()

# symmetric_topology_view tests — useSymmetricTopology resolution + in-RAM merge
# + single-slice FORWARD_ONLY pin (GPU-UVA). CI-friendly (no-GPU = documented no-op).
if(GTest_FOUND AND ENABLE_GNN AND
   EXISTS "${CMAKE_SOURCE_DIR}/src/tests/symmetric_topology_view_test.cc")
    add_executable(symmetric_topology_view_test
        ${CMAKE_SOURCE_DIR}/src/tests/symmetric_topology_view_test.cc
    )
    set_target_properties(symmetric_topology_view_test PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/tests"
        BUILD_RPATH "${CMAKE_SOURCE_DIR}/third_party/libtorch/lib"
        INSTALL_RPATH "${CMAKE_SOURCE_DIR}/third_party/libtorch/lib"
    )
    target_link_libraries(symmetric_topology_view_test
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
        NAME SymmetricTopologyViewTests
        COMMAND $<TARGET_FILE:symmetric_topology_view_test>
        WORKING_DIRECTORY tests
    )
    set(ASAN_SUPP_FILE "${CMAKE_SOURCE_DIR}/tests/asan_suppressions.txt")
    if(EXISTS "${ASAN_SUPP_FILE}")
        set_property(TEST SymmetricTopologyViewTests PROPERTY
            ENVIRONMENT "LSAN_OPTIONS=suppressions=${ASAN_SUPP_FILE}")
    endif()
endif()

# release-directional-tiers + in-process RSS probe tests — the symmetric GPU pin
# frees the directional tiers once the merged slice is pinned; process_memory.h
# surfaces the peak/current RSS so the relief is observable.
if(GTest_FOUND AND ENABLE_GNN AND
   EXISTS "${CMAKE_SOURCE_DIR}/src/tests/four_level_topology_release_directional_test.cc")
    add_executable(four_level_topology_release_directional_test
        ${CMAKE_SOURCE_DIR}/src/tests/four_level_topology_release_directional_test.cc
    )
    set_target_properties(four_level_topology_release_directional_test PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/tests"
        BUILD_RPATH "${CMAKE_SOURCE_DIR}/third_party/libtorch/lib"
        INSTALL_RPATH "${CMAKE_SOURCE_DIR}/third_party/libtorch/lib"
    )
    target_link_libraries(four_level_topology_release_directional_test
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
        NAME FourLevelTopologyReleaseDirectionalTests
        COMMAND $<TARGET_FILE:four_level_topology_release_directional_test>
        WORKING_DIRECTORY tests
    )
    set(ASAN_SUPP_FILE "${CMAKE_SOURCE_DIR}/tests/asan_suppressions.txt")
    if(EXISTS "${ASAN_SUPP_FILE}")
        set_property(TEST FourLevelTopologyReleaseDirectionalTests PROPERTY
            ENVIRONMENT "LSAN_OPTIONS=suppressions=${ASAN_SUPP_FILE}")
    endif()
endif()

# gpu_khop_sampler tests — statistical validation of the GPU reservoir+Philox
# sampling kernel (short-circuit, no-replacement, uniformity, determinism).
if(GTest_FOUND AND ENABLE_GNN AND
   EXISTS "${CMAKE_SOURCE_DIR}/src/tests/gpu_khop_sampler_test.cc")
    add_executable(gpu_khop_sampler_test
        ${CMAKE_SOURCE_DIR}/src/tests/gpu_khop_sampler_test.cc
    )
    set_target_properties(gpu_khop_sampler_test PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/tests"
        BUILD_RPATH "${CMAKE_SOURCE_DIR}/third_party/libtorch/lib"
        INSTALL_RPATH "${CMAKE_SOURCE_DIR}/third_party/libtorch/lib"
    )
    target_link_libraries(gpu_khop_sampler_test
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
        NAME GpuKhopSamplerTests
        COMMAND $<TARGET_FILE:gpu_khop_sampler_test>
        WORKING_DIRECTORY tests
    )
    set(ASAN_SUPP_FILE "${CMAKE_SOURCE_DIR}/tests/asan_suppressions.txt")
    if(EXISTS "${ASAN_SUPP_FILE}")
        set_property(TEST GpuKhopSamplerTests PROPERTY
            ENVIRONMENT "LSAN_OPTIONS=suppressions=${ASAN_SUPP_FILE}")
    endif()
endif()

# ============================================================================
# FourLevelTopologyStore dispatcher tests
# ============================================================================
# Validates the tier-dispatch skeleton: L1/L2 hits, L3 sidecar absence
# falls through to L4, L4 BPT fallback, defensive throws on missing
# fallback configuration, isolated nodes across tiers, multi-tier graph
# consistency, out-of-range row_lookup behavior. The build orchestration
# is exercised separately; this suite pins only the dispatch contract.
if(GTest_FOUND AND ENABLE_GNN AND
   EXISTS "${CMAKE_SOURCE_DIR}/src/tests/four_level_topology_store_test.cc")
    add_executable(four_level_topology_store_test
        ${CMAKE_SOURCE_DIR}/src/tests/four_level_topology_store_test.cc
    )
    set_target_properties(four_level_topology_store_test PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/tests"
        BUILD_RPATH "${CMAKE_SOURCE_DIR}/third_party/libtorch/lib"
        INSTALL_RPATH "${CMAKE_SOURCE_DIR}/third_party/libtorch/lib"
    )
    target_link_libraries(four_level_topology_store_test
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
        NAME FourLevelTopologyStoreTests
        COMMAND $<TARGET_FILE:four_level_topology_store_test>
        WORKING_DIRECTORY tests
    )
    set(ASAN_SUPP_FILE "${CMAKE_SOURCE_DIR}/tests/asan_suppressions.txt")
    if(EXISTS "${ASAN_SUPP_FILE}")
        set_property(TEST FourLevelTopologyStoreTests PROPERTY
            ENVIRONMENT "LSAN_OPTIONS=suppressions=${ASAN_SUPP_FILE}")
    endif()
endif()

# ============================================================================
# TopologyAccessor four-level integration tests
# ============================================================================
# End-to-end check that enable_four_level_store() over a live ProjectionStorage
# returns byte-identical neighbour sets vs the BPT-direct path, that the
# default-disabled path preserves the sidecar / adjacency-cache behaviour,
# and that SamplingConfig::validate() rejects the incompatible combination
# (four-level store enabled while the adjacency cache is disabled).
if(GTest_FOUND AND ENABLE_GNN AND
   EXISTS "${CMAKE_SOURCE_DIR}/src/tests/topology_accessor_four_level_integration_test.cc")
    add_executable(topology_accessor_four_level_integration_test
        ${CMAKE_SOURCE_DIR}/src/tests/topology_accessor_four_level_integration_test.cc
    )
    set_target_properties(topology_accessor_four_level_integration_test PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/tests"
        BUILD_RPATH "${CMAKE_SOURCE_DIR}/third_party/libtorch/lib"
        INSTALL_RPATH "${CMAKE_SOURCE_DIR}/third_party/libtorch/lib"
    )
    target_link_libraries(topology_accessor_four_level_integration_test
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
        NAME TopologyAccessorFourLevelIntegrationTests
        COMMAND $<TARGET_FILE:topology_accessor_four_level_integration_test>
        WORKING_DIRECTORY tests
    )
    set(ASAN_SUPP_FILE "${CMAKE_SOURCE_DIR}/tests/asan_suppressions.txt")
    if(EXISTS "${ASAN_SUPP_FILE}")
        set_property(TEST TopologyAccessorFourLevelIntegrationTests PROPERTY
            ENVIRONMENT "LSAN_OPTIONS=suppressions=${ASAN_SUPP_FILE}")
    endif()
endif()

# ============================================================================
# Four-Level Topology Store tagged-ObjectId dispatch regression
# ============================================================================
# Guards the row_lookup_ tag-mask fix: production projections store nodes
# with ObjectId.id = (MASK_NODE | row_idx), and pre-fix the dispatcher
# treated the tag bits as part of the row index — landing every lookup in
# the L4 BPT-fallback path and emitting the "exceeds tier_lookup_ size"
# warning observed by scripts/bench_four_level_topology.sh. The 3 tests
# here exercise the same row_lookup_ closure that the bench's
# `gnn_offline_sample` invocation drives, but in-process so a gtest can
# pin the regression without spinning up the HTTP server.
if(GTest_FOUND AND ENABLE_GNN AND
   EXISTS "${CMAKE_SOURCE_DIR}/src/tests/four_level_topology_tag_dispatch_test.cc")
    add_executable(four_level_topology_tag_dispatch_test
        ${CMAKE_SOURCE_DIR}/src/tests/four_level_topology_tag_dispatch_test.cc
    )
    set_target_properties(four_level_topology_tag_dispatch_test PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/tests"
        BUILD_RPATH "${CMAKE_SOURCE_DIR}/third_party/libtorch/lib"
        INSTALL_RPATH "${CMAKE_SOURCE_DIR}/third_party/libtorch/lib"
    )
    target_link_libraries(four_level_topology_tag_dispatch_test
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
        NAME FourLevelTopologyTagDispatchTests
        COMMAND $<TARGET_FILE:four_level_topology_tag_dispatch_test>
        WORKING_DIRECTORY tests
    )
    set(ASAN_SUPP_FILE "${CMAKE_SOURCE_DIR}/tests/asan_suppressions.txt")
    if(EXISTS "${ASAN_SUPP_FILE}")
        set_property(TEST FourLevelTopologyTagDispatchTests PROPERTY
            ENVIRONMENT "LSAN_OPTIONS=suppressions=${ASAN_SUPP_FILE}")
    endif()
endif()

# ============================================================================
# FourLevelTopologyStore symmetric (pre-merged undirected) tier unit tests
# ============================================================================
if(GTest_FOUND AND ENABLE_GNN AND
   EXISTS "${CMAKE_SOURCE_DIR}/src/tests/four_level_topology_sym_test.cc")
    add_executable(four_level_topology_sym_test
        ${CMAKE_SOURCE_DIR}/src/tests/four_level_topology_sym_test.cc
    )
    set_target_properties(four_level_topology_sym_test PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/tests"
        BUILD_RPATH "${CMAKE_SOURCE_DIR}/third_party/libtorch/lib"
        INSTALL_RPATH "${CMAKE_SOURCE_DIR}/third_party/libtorch/lib"
    )
    target_link_libraries(four_level_topology_sym_test
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
        NAME FourLevelTopologySymTests
        COMMAND $<TARGET_FILE:four_level_topology_sym_test>
        WORKING_DIRECTORY tests
    )
    set(ASAN_SUPP_FILE "${CMAKE_SOURCE_DIR}/tests/asan_suppressions.txt")
    if(EXISTS "${ASAN_SUPP_FILE}")
        set_property(TEST FourLevelTopologySymTests PROPERTY
            ENVIRONMENT "LSAN_OPTIONS=suppressions=${ASAN_SUPP_FILE}")
    endif()
endif()

# ============================================================================
# FourLevelTopologyStore warm-start activation tests
# ============================================================================
# Exercises the round-trip between gnn_offline_sample's `node_counts.bin`
# writer and the TopologyFrequencyProfiler reader, plus the activation
# of FourLevelTopologyStore::compute_l3_minhash_reorder_'s warm-start
# branch. Cold-start path remains the default and is preserved.
if(GTest_FOUND AND ENABLE_GNN AND
   EXISTS "${CMAKE_SOURCE_DIR}/src/tests/four_level_topology_warm_start_test.cc")
    add_executable(four_level_topology_warm_start_test
        ${CMAKE_SOURCE_DIR}/src/tests/four_level_topology_warm_start_test.cc
    )
    set_target_properties(four_level_topology_warm_start_test PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/tests"
        BUILD_RPATH "${CMAKE_SOURCE_DIR}/third_party/libtorch/lib"
        INSTALL_RPATH "${CMAKE_SOURCE_DIR}/third_party/libtorch/lib"
    )
    target_link_libraries(four_level_topology_warm_start_test
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
        NAME FourLevelTopologyWarmStartTests
        COMMAND $<TARGET_FILE:four_level_topology_warm_start_test>
        WORKING_DIRECTORY tests
    )
    set(ASAN_SUPP_FILE "${CMAKE_SOURCE_DIR}/tests/asan_suppressions.txt")
    if(EXISTS "${ASAN_SUPP_FILE}")
        set_property(TEST FourLevelTopologyWarmStartTests PROPERTY
            ENVIRONMENT "LSAN_OPTIONS=suppressions=${ASAN_SUPP_FILE}")
    endif()
endif()

# ============================================================================
# Multi-layer k-hop sampling determinism tests
# ============================================================================
# Complements the single-layer sample_neighbors determinism tests. Exercises
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
# gnn_build_topology_snapshot procedure unit tests
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
    # millenniumdb provides ProjectionStorage + BPlusTree — the usual
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
# Symmetric topology parallel-edge guard unit tests
# ============================================================================
# Constructs ProjectionStorage + scans a BPlusTree<3>, so it needs the full
# millenniumdb link (same recipe as gnn_build_topology_snapshot_procedure_test).
if(GTest_FOUND AND ENABLE_GNN AND
   EXISTS "${CMAKE_SOURCE_DIR}/src/tests/topology_symmetric_parallel_edge_test.cc")
    add_executable(topology_symmetric_parallel_edge_test
        ${CMAKE_SOURCE_DIR}/src/tests/topology_symmetric_parallel_edge_test.cc
    )
    set_target_properties(topology_symmetric_parallel_edge_test PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/tests"
        BUILD_RPATH "${CMAKE_SOURCE_DIR}/third_party/libtorch/lib"
        INSTALL_RPATH "${CMAKE_SOURCE_DIR}/third_party/libtorch/lib"
    )
    target_link_libraries(topology_symmetric_parallel_edge_test
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
        NAME TopologySymmetricParallelEdgeTests
        COMMAND $<TARGET_FILE:topology_symmetric_parallel_edge_test>
        WORKING_DIRECTORY tests
    )
    set(ASAN_SUPP_FILE "${CMAKE_SOURCE_DIR}/tests/asan_suppressions.txt")
    if(EXISTS "${ASAN_SUPP_FILE}")
        set_property(TEST TopologySymmetricParallelEdgeTests PROPERTY
            ENVIRONMENT "LSAN_OPTIONS=suppressions=${ASAN_SUPP_FILE}")
    endif()
endif()

# ============================================================================
# Post-hoc symmetric topology bake unit tests
# ============================================================================
# Builds a TopologyAccessor (needs torch + mdb_gnn_core) and a ProjectionStorage,
# so it uses the same full link recipe as gnn_build_topology_snapshot_procedure_test.
if(GTest_FOUND AND ENABLE_GNN AND
   EXISTS "${CMAKE_SOURCE_DIR}/src/tests/gnn_build_topology_snapshot_symmetric_test.cc")
    add_executable(gnn_build_topology_snapshot_symmetric_test
        ${CMAKE_SOURCE_DIR}/src/tests/gnn_build_topology_snapshot_symmetric_test.cc
    )
    set_target_properties(gnn_build_topology_snapshot_symmetric_test PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/tests"
        BUILD_RPATH "${CMAKE_SOURCE_DIR}/third_party/libtorch/lib"
        INSTALL_RPATH "${CMAKE_SOURCE_DIR}/third_party/libtorch/lib"
    )
    target_link_libraries(gnn_build_topology_snapshot_symmetric_test
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
        NAME GnnBuildTopologySnapshotSymmetricTests
        COMMAND $<TARGET_FILE:gnn_build_topology_snapshot_symmetric_test>
        WORKING_DIRECTORY tests
    )
    set(ASAN_SUPP_FILE "${CMAKE_SOURCE_DIR}/tests/asan_suppressions.txt")
    if(EXISTS "${ASAN_SUPP_FILE}")
        set_property(TEST GnnBuildTopologySnapshotSymmetricTests PROPERTY
            ENVIRONMENT "LSAN_OPTIONS=suppressions=${ASAN_SUPP_FILE}")
    endif()
endif()

# ============================================================================
# GNN procedure metadata consistency tests (yield_fields vs runtime yields)
# ============================================================================
if(GTest_FOUND AND ENABLE_GNN AND
   EXISTS "${CMAKE_SOURCE_DIR}/src/tests/gnn_train_procedure_metadata_test.cc")
    add_executable(gnn_train_procedure_metadata_test
        ${CMAKE_SOURCE_DIR}/src/tests/gnn_train_procedure_metadata_test.cc
    )
    set_target_properties(gnn_train_procedure_metadata_test PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/tests"
        BUILD_RPATH "${CMAKE_SOURCE_DIR}/third_party/libtorch/lib"
        INSTALL_RPATH "${CMAKE_SOURCE_DIR}/third_party/libtorch/lib"
    )
    target_link_libraries(gnn_train_procedure_metadata_test
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
        NAME GnnTrainProcedureMetadataTests
        COMMAND $<TARGET_FILE:gnn_train_procedure_metadata_test>
        WORKING_DIRECTORY tests
    )
endif()

# ============================================================================
# SampleStorage commit/abort protocol tests
# ============================================================================
if(GTest_FOUND AND ENABLE_GNN AND
   EXISTS "${CMAKE_SOURCE_DIR}/src/tests/sample_storage_test.cc")
    add_executable(sample_storage_test
        ${CMAKE_SOURCE_DIR}/src/tests/sample_storage_test.cc
    )
    set_target_properties(sample_storage_test PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/tests"
        BUILD_RPATH "${CMAKE_SOURCE_DIR}/third_party/libtorch/lib"
        INSTALL_RPATH "${CMAKE_SOURCE_DIR}/third_party/libtorch/lib"
    )
    target_link_libraries(sample_storage_test
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
        NAME SampleStorageTests
        COMMAND $<TARGET_FILE:sample_storage_test>
        WORKING_DIRECTORY tests
    )
endif()

# sample_storage_shard tests — golden compare: sharded write == legacy write.
if(GTest_FOUND AND ENABLE_GNN AND
   EXISTS "${CMAKE_SOURCE_DIR}/src/tests/sample_storage_shard_test.cc")
    add_executable(sample_storage_shard_test
        ${CMAKE_SOURCE_DIR}/src/tests/sample_storage_shard_test.cc
    )
    set_target_properties(sample_storage_shard_test PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/tests"
        BUILD_RPATH "${CMAKE_SOURCE_DIR}/third_party/libtorch/lib"
        INSTALL_RPATH "${CMAKE_SOURCE_DIR}/third_party/libtorch/lib"
    )
    target_link_libraries(sample_storage_shard_test
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
        NAME SampleStorageShardTests
        COMMAND $<TARGET_FILE:sample_storage_shard_test>
        WORKING_DIRECTORY tests
    )
endif()

# ============================================================================
# Isolated TopologyAccessor micro-benchmark
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

# ============================================================================
# BatchedEdgeAggregator unit tests
# ============================================================================
# Registers batched_edge_aggregator_test when GTest is available and the
# corresponding header exists. Mirrors the EXISTS-gated registration
# precedent of index_set_test / topology_snapshot_format_test so the test
# only wires in after its sources land.
if(GTest_FOUND AND
   EXISTS "${CMAKE_SOURCE_DIR}/src/graph_models/gql/projection/batched_edge_aggregator.h"
   AND TARGET GTest::gtest_main)
    add_executable(batched_edge_aggregator_test
        ${CMAKE_SOURCE_DIR}/src/tests/batched_edge_aggregator_test.cc
    )
    set_target_properties(batched_edge_aggregator_test PROPERTIES
        CXX_STANDARD 17
        CXX_STANDARD_REQUIRED ON
        RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/tests"
    )
    # GTest::gtest_main must come BEFORE millenniumdb so its main() wins
    # over the main() in src/third_party/serd/serdi.cc (which is pulled into
    # libmillenniumdb.a by the recursive src/*.cc glob). Static-archive link
    # order determines which main() actually resolves.
    target_link_libraries(batched_edge_aggregator_test
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
        target_link_libraries(batched_edge_aggregator_test
            mdb_gnn_core ${TORCH_LIBRARIES})
    endif()
    add_test(
        NAME batched_edge_aggregator_test
        COMMAND $<TARGET_FILE:batched_edge_aggregator_test>
        WORKING_DIRECTORY tests
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
