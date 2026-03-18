# ============================================================================
# GNN Module CMake Configuration
# ============================================================================
#
# This module provides LibTorch-based GNN training capabilities for MillenniumDB.
# Enable with: cmake -D ENABLE_GNN=ON
#
# Requirements:
#   - LibTorch installed in third_party/libtorch/
#   - CUDA Toolkit (optional, for GPU acceleration)
#
# ============================================================================

cmake_minimum_required(VERSION 3.18)

# ----------------------------------------------------------------------------
# Find LibTorch
# ----------------------------------------------------------------------------
set(CMAKE_PREFIX_PATH "${CMAKE_SOURCE_DIR}/third_party/libtorch" ${CMAKE_PREFIX_PATH})
find_package(Torch REQUIRED)

message(STATUS "Found Torch: ${TORCH_INCLUDE_DIRS}")
message(STATUS "Torch libraries: ${TORCH_LIBRARIES}")

# Check for CUDA support in LibTorch
if(NOT TORCH_CUDA_LIBRARIES)
    message(WARNING "LibTorch was built without CUDA support. GPU acceleration disabled.")
    set(GNN_CUDA_ENABLED OFF)
else()
    set(GNN_CUDA_ENABLED ON)
    message(STATUS "LibTorch CUDA support enabled")
endif()

# ----------------------------------------------------------------------------
# Compiler Flags
# ----------------------------------------------------------------------------
# Parse TORCH_CXX_FLAGS into a CMake list for target_compile_options.
# This avoids polluting global CMAKE_CXX_FLAGS which would affect ALL targets.
separate_arguments(TORCH_CXX_FLAGS_LIST UNIX_COMMAND "${TORCH_CXX_FLAGS}")

# Ensure C++17 compatibility (MDB requirement)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# ----------------------------------------------------------------------------
# GNN Core Library
# ----------------------------------------------------------------------------
add_library(mdb_gnn_core STATIC
    # Phase 0: Core LibTorch Foundation
    core/cuda_context.cc
    core/tensor_utils.cc
    core/memory_pool.cc
    core/sparse_ops.cc

    # GNN Tensor Storage (independent from MDB tensors)
    storage/inmemory_gnn_tensor_store.cc
    storage/file_gnn_tensor_store.cc
    storage/gnn_tensor_converter.cc

    # Phase 0: Storage - FeatureMatrix (redesign)
    storage/feature_matrix.cc
    storage/row_mapping.cc
    storage/packed_batch_store.cc
    storage/batch_materializer.cc

    # Phase 1: Projection Integration
    projection/feature_accessor.cc
    projection/topology_accessor.cc
    projection/gnn_projection_adapter.cc

    # Phase 2: Offline Sampling Engine
    sampling/graph_sample.cc
    sampling/seed_selector.cc
    sampling/basic_khop_sampler.cc
    sampling/sorted_batch_sampler.cc
    sampling/reservoir_sampler.cc
    sampling/sample_storage.cc
    sampling/offline_sampling_engine.cc
    sampling/minhash_reorderer.cc

    # Phase 7: Leapfrog Triejoin Optimization
    sampling/leapfrog_gnn_sampler.cc

    # Phase 8: True Leapfrog Triejoin (Seek-Based Sampling)
    sampling/seekable_edge_iter.cc
    sampling/seek_based_gnn_sampler.cc
    sampling/parallel_cursor_merger.cc

    # Phase 5: GQL Procedures for GNN Sampling
    ${CMAKE_SOURCE_DIR}/src/query/procedure/builtin/gnn_offline_sample_procedure.cc
    ${CMAKE_SOURCE_DIR}/src/query/procedure/builtin/gnn_sample_list_procedure.cc
    ${CMAKE_SOURCE_DIR}/src/query/procedure/builtin/gnn_sample_info_procedure.cc
    ${CMAKE_SOURCE_DIR}/src/query/procedure/builtin/gnn_sample_drop_procedure.cc

    # Phase 12: HNSW Index over GNN Embeddings
    ${CMAKE_SOURCE_DIR}/src/query/procedure/builtin/gnn_hnsw_create_procedure.cc
    ${CMAKE_SOURCE_DIR}/src/query/procedure/builtin/gnn_hnsw_find_similar_procedure.cc
    ${CMAKE_SOURCE_DIR}/src/query/procedure/builtin/gnn_hnsw_list_procedure.cc
    ${CMAKE_SOURCE_DIR}/src/query/procedure/builtin/gnn_hnsw_drop_procedure.cc
    ${CMAKE_SOURCE_DIR}/src/query/procedure/builtin/gnn_hnsw_info_procedure.cc

    # Phase 3: Batch Materialization (L3 reorder + L4 packing)
    ${CMAKE_SOURCE_DIR}/src/query/procedure/builtin/gnn_materialize_batches_procedure.cc
)

target_include_directories(mdb_gnn_core PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_SOURCE_DIR}/src  # For accessing graph_models, storage, etc.
    ${TORCH_INCLUDE_DIRS}
)

target_link_libraries(mdb_gnn_core PUBLIC
    ${TORCH_LIBRARIES}
)

# LibTorch C++ ABI flag (required for binary compatibility with LibTorch).
# Note: -foperator-names is no longer needed here — the global -fno-operator-names
# flag was removed since no code in the project uses alternative tokens as identifiers.
target_compile_options(mdb_gnn_core PRIVATE ${TORCH_CXX_FLAGS_LIST})
target_compile_definitions(mdb_gnn_core PRIVATE _GLIBCXX_USE_CXX11_ABI=1)

# Link with MDB libraries if they exist
if(TARGET mdb_storage)
    target_link_libraries(mdb_gnn_core PUBLIC mdb_storage)
endif()

if(TARGET mdb_graph_models)
    target_link_libraries(mdb_gnn_core PUBLIC mdb_graph_models)
endif()

# ----------------------------------------------------------------------------
# Conditional CUDA Configuration
# ----------------------------------------------------------------------------
if(GNN_CUDA_ENABLED)
    find_package(CUDAToolkit)

    if(CUDAToolkit_FOUND)
        target_compile_definitions(mdb_gnn_core PUBLIC GNN_CUDA_ENABLED)
        target_link_libraries(mdb_gnn_core PUBLIC CUDA::cudart)

        # CUDA architecture (adjust for your GPU)
        # 70=V100, 75=T4, 80=A100, 86=RTX30xx, 89=RTX40xx, 90=H100
        if(DEFINED CMAKE_CUDA_ARCHITECTURES)
            set_property(TARGET mdb_gnn_core PROPERTY CUDA_ARCHITECTURES ${CMAKE_CUDA_ARCHITECTURES})
        else()
            set_property(TARGET mdb_gnn_core PROPERTY CUDA_ARCHITECTURES 70 75 80 86 89 90)
        endif()

        message(STATUS "CUDA Toolkit found: ${CUDAToolkit_VERSION}")
    else()
        message(WARNING "CUDA Toolkit not found. Disabling CUDA support.")
        set(GNN_CUDA_ENABLED OFF)
    endif()
endif()

# ----------------------------------------------------------------------------
# RPATH for LibTorch libraries
# ----------------------------------------------------------------------------
set_target_properties(mdb_gnn_core PROPERTIES
    INSTALL_RPATH "${CMAKE_SOURCE_DIR}/third_party/libtorch/lib"
    BUILD_WITH_INSTALL_RPATH TRUE
)

# ----------------------------------------------------------------------------
# Tests
# ----------------------------------------------------------------------------
if(BUILD_TESTING)
    find_package(GTest QUIET)

    if(GTest_FOUND)
        # test_gnn_core: only build if source files exist (created in later phases)
        if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/tests/test_cuda_context.cc")
            add_executable(test_gnn_core
                # Phase 0: Core tests
                tests/test_cuda_context.cc
                tests/test_tensor_utils.cc
                tests/test_memory_pool.cc
                tests/test_sparse_ops.cc

                # GNN Tensor Storage tests
                tests/test_gnn_tensor_store.cc

                # Phase 1: Projection integration tests
                tests/test_projection_accessors.cc

                # DVM-1: TopologyAccessor correctness verification
                tests/test_topology_accessor_correctness.cc

                # DVM-2: UNDIRECTED orientation bidirectional traversal
                tests/test_undirected_orientation.cc
            )

            target_link_libraries(test_gnn_core
                mdb_gnn_core
                GTest::gtest_main
            )

            # Set RPATH for test executable to find LibTorch at runtime
            set_target_properties(test_gnn_core PROPERTIES
                BUILD_RPATH "${CMAKE_SOURCE_DIR}/third_party/libtorch/lib"
                INSTALL_RPATH "${CMAKE_SOURCE_DIR}/third_party/libtorch/lib"
            )

            add_test(NAME GNNCoreTests COMMAND test_gnn_core)
        endif()

        # GNN Storage tests (Step 0: FeatureMatrix + RowMapping)
        # Multiple .cc files → one executable, filtered via --gtest_filter
        add_executable(test_gnn_storage
            tests/test_feature_matrix.cc
            tests/test_row_mapping.cc
        )
        # Note: ObjectId is header-only for construction/comparison;
        # operator<< lives in query/id.cc (linked via millenniumdb) but tests don't need it
        target_link_libraries(test_gnn_storage mdb_gnn_core GTest::gtest_main)
        set_target_properties(test_gnn_storage PROPERTIES
            BUILD_RPATH "${CMAKE_SOURCE_DIR}/third_party/libtorch/lib"
            INSTALL_RPATH "${CMAKE_SOURCE_DIR}/third_party/libtorch/lib"
        )
        add_test(NAME GnnStorageTests COMMAND test_gnn_storage)

        # PackedBatchStore tests (Step 1: L4 per-batch packed features)
        add_executable(test_packed_batch_store
            tests/test_packed_batch_store.cc
        )
        target_link_libraries(test_packed_batch_store mdb_gnn_core GTest::gtest_main)
        set_target_properties(test_packed_batch_store PROPERTIES
            BUILD_RPATH "${CMAKE_SOURCE_DIR}/third_party/libtorch/lib"
            INSTALL_RPATH "${CMAKE_SOURCE_DIR}/third_party/libtorch/lib"
        )
        add_test(NAME PackedBatchStoreTests COMMAND test_packed_batch_store)

        # MinHash Reorderer tests (Step 2: L3 disk cache reordering)
        add_executable(test_minhash_reorderer
            tests/test_minhash_reorderer.cc
        )
        target_link_libraries(test_minhash_reorderer mdb_gnn_core GTest::gtest_main)
        set_target_properties(test_minhash_reorderer PROPERTIES
            BUILD_RPATH "${CMAKE_SOURCE_DIR}/third_party/libtorch/lib"
            INSTALL_RPATH "${CMAKE_SOURCE_DIR}/third_party/libtorch/lib"
        )
        add_test(NAME MinHashReordererTests COMMAND test_minhash_reorderer)

        # GraphSample + SamplingConfig tests (Phase 2 components, isolable)
        add_executable(test_graph_sample
            tests/test_graph_sample.cc
        )
        target_link_libraries(test_graph_sample mdb_gnn_core GTest::gtest_main)
        target_compile_options(test_graph_sample PRIVATE ${TORCH_CXX_FLAGS_LIST})
        target_compile_definitions(test_graph_sample PRIVATE _GLIBCXX_USE_CXX11_ABI=1)
        set_target_properties(test_graph_sample PROPERTIES
            BUILD_RPATH "${CMAKE_SOURCE_DIR}/third_party/libtorch/lib"
            INSTALL_RPATH "${CMAKE_SOURCE_DIR}/third_party/libtorch/lib"
        )
        add_test(NAME GraphSampleTests COMMAND test_graph_sample)

        # BatchMaterializer tests (L3+L4 pipeline)
        add_executable(test_batch_materializer
            tests/test_batch_materializer.cc
        )
        # Link gtest_main FIRST so its main() takes precedence over serdi's main()
        # in the millenniumdb static library. SampleStorage needs Logger from millenniumdb.
        target_link_libraries(test_batch_materializer
            GTest::gtest_main
            mdb_gnn_core
            millenniumdb
            antlr4_cpp_runtime
            ssl crypto
            ${TORCH_LIBRARIES}
        )
        set_target_properties(test_batch_materializer PROPERTIES
            BUILD_RPATH "${CMAKE_SOURCE_DIR}/third_party/libtorch/lib"
            INSTALL_RPATH "${CMAKE_SOURCE_DIR}/third_party/libtorch/lib"
        )
        add_test(NAME BatchMaterializerTests COMMAND test_batch_materializer)

        message(STATUS "GNN tests enabled")
    else()
        message(STATUS "GTest not found. GNN tests disabled.")
    endif()
endif()

# ----------------------------------------------------------------------------
# Installation
# ----------------------------------------------------------------------------
install(TARGETS mdb_gnn_core
    ARCHIVE DESTINATION lib
    LIBRARY DESTINATION lib
)

install(DIRECTORY core/
    DESTINATION include/gnn/core
    FILES_MATCHING PATTERN "*.h"
)

install(DIRECTORY projection/
    DESTINATION include/gnn/projection
    FILES_MATCHING PATTERN "*.h"
)

install(DIRECTORY storage/
    DESTINATION include/gnn/storage
    FILES_MATCHING PATTERN "*.h"
)

install(DIRECTORY sampling/
    DESTINATION include/gnn/sampling
    FILES_MATCHING PATTERN "*.h"
)
