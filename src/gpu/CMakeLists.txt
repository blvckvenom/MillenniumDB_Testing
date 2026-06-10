# ============================================================================
# GPU Module CMake Configuration
# ============================================================================
#
# GPU-accelerated operations for MillenniumDB (sort, filter, transform).
# CUDA is optional: when CUDAToolkit is not found, only CPU fallback code
# is compiled and MDB_GPU_ENABLED is NOT defined.
#
# This module has zero MillenniumDB source dependencies.  It only links
# CUDA::cudart (when available) and exposes MDB_GPU_ENABLED to consumers.
# ============================================================================

add_library(mdb_gpu STATIC
    gpu_device.cc
    resource_planner.cc
    sort/gpu_sort.cc
    sort/gpu_sort_fallback.cc
)

target_include_directories(mdb_gpu PUBLIC ${CMAKE_SOURCE_DIR}/src)

# Link TBB when available (HAS_TBB is defined globally by root CMakeLists.txt)
if(TBB_FOUND)
    target_link_libraries(mdb_gpu PUBLIC TBB::tbb)
endif()

# Link LZ4 when available so gpu_sort can decompress SpillCodec-encoded
# spill files (produced by StreamingRecordBuffer). HAS_LZ4 and the LZ4
# paths are set globally by root CMakeLists.txt. mdb_gpu duplicates the
# minimal SpillCodec reader logic locally (see src/gpu/sort/gpu_sort.cc
# read_spill_file) so it can decompress without depending on
# graph_models/gql/projection/spill_codec.{h,cc} — preserving the "zero
# MillenniumDB dependencies" rule declared at the top of this file.
if(LZ4_LIBRARY AND LZ4_INCLUDE_DIR)
    target_include_directories(mdb_gpu PUBLIC ${LZ4_INCLUDE_DIR})
    target_link_libraries(mdb_gpu PUBLIC ${LZ4_LIBRARY})
endif()

# ----------------------------------------------------------------------------
# Optional CUDA support
# ----------------------------------------------------------------------------
# CUDAToolkit_FOUND and enable_language(CUDA) are handled at project scope
# in the root CMakeLists.txt before add_subdirectory(src/gpu).
if(CUDAToolkit_FOUND)
    target_sources(mdb_gpu PRIVATE
        sort/gpu_radix_sort.cu
        ops/gpu_filter.cu
        ops/gpu_membership.cu
        ops/gpu_transform.cu
        tests/benchmark_gpu_kernels.cu
    )
    target_link_libraries(mdb_gpu PUBLIC CUDA::cudart)
    target_compile_definitions(mdb_gpu PUBLIC MDB_GPU_ENABLED)

    # Multi-architecture: Turing through Hopper + Blackwell + PTX forward-compat
    set_target_properties(mdb_gpu PROPERTIES
        CUDA_STANDARD 17
        CUDA_STANDARD_REQUIRED ON
        CUDA_ARCHITECTURES "75;80;86;89;90"
    )

    message(STATUS "GPU module: CUDA enabled (${CUDAToolkit_VERSION})")
else()
    message(STATUS "GPU module: CUDA not found, CPU-only fallback")
endif()

# ----------------------------------------------------------------------------
# Tests
# ----------------------------------------------------------------------------
if(BUILD_TESTING)
    find_package(GTest QUIET)

    if(GTest_FOUND)
        add_executable(test_gpu
            tests/test_gpu_device.cc
            tests/test_gpu_sort.cc
            tests/test_gpu_ops.cc
        )
        set_target_properties(test_gpu PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/tests")
        target_link_libraries(test_gpu mdb_gpu GTest::gtest_main)
        add_test(NAME GpuTests COMMAND test_gpu)

        message(STATUS "GPU tests enabled")
    else()
        message(STATUS "GTest not found. GPU tests disabled.")
    endif()
endif()

# ----------------------------------------------------------------------------
# Benchmark executable (manual run, not registered with ctest)
# ----------------------------------------------------------------------------
add_executable(benchmark_gpu_sort tests/benchmark_gpu_sort.cc)
target_link_libraries(benchmark_gpu_sort mdb_gpu)
set_target_properties(benchmark_gpu_sort PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")
