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



# Define targets
add_library(millenniumdb STATIC ${SRCS})
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
