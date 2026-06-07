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
# Optional io_uring support for L3 direct I/O
# ----------------------------------------------------------------------------
find_package(PkgConfig QUIET)
if(PkgConfig_FOUND)
    pkg_check_modules(URING QUIET liburing)
endif()
if(URING_FOUND)
    message(STATUS "io_uring support enabled (liburing ${URING_VERSION})")
else()
    message(STATUS "liburing not found — using pread fallback for L3 direct I/O")
endif()

# ----------------------------------------------------------------------------
# GNN Core Library
# ----------------------------------------------------------------------------
add_library(mdb_gnn_core STATIC
    # Phase 0: Core LibTorch Foundation
    core/cuda_context.cc
    core/tensor_utils.cc
    core/memory_pool.cc
    core/sparse_ops.cc
    core/feature_assembler.cc

    # GNN Tensor Storage (independent from MDB tensors)
    storage/inmemory_gnn_tensor_store.cc
    storage/file_gnn_tensor_store.cc
    storage/gnn_tensor_converter.cc

    # Phase 0: Storage - FeatureMatrix (redesign)
    storage/feature_matrix.cc
    storage/row_mapping.cc
    storage/packed_batch_store.cc
    storage/batch_materializer.cc
    storage/cpu_cache.cc
    storage/gpu_cache.cc
    storage/four_level_store.cc
    storage/addr_table_writer.cc
    storage/addr_table_reader.cc
    storage/direct_io_reader.cc
    storage/consolidated_slim_reader.cc
    storage/block_store.cc

    # Phase 1: Projection Integration
    projection/feature_accessor.cc
    projection/topology_accessor.cc
    projection/gnn_projection_adapter.cc

    # Spec #13 Phase 1: Frequency profiler for Four-Level Topology Store
    projection/topology_frequency_profiler.cc

    # Plan E Phase 0 (2026-05-11): random-walk profiler to bootstrap
    # node_counts.bin on first run (resolves cold-start L3 thrashing).
    projection/topology_walk_profiler.cc

    # Spec #13 Phase 2: L1/L2 caches + dispatcher (T13.4 / T13.5 / T13.6)
    projection/l1_hash_cache.cc
    projection/l2_compact_csr.cc
    projection/four_level_topology_store.cc

    # Phase 2: Offline Sampling Engine
    sampling/graph_sample.cc
    sampling/seed_selector.cc
    sampling/basic_khop_sampler.cc
    sampling/sorted_batch_sampler.cc
    sampling/reservoir_sampler.cc
    sampling/sample_storage.cc
    sampling/sample_fingerprint.cc
    sampling/offline_sampling_engine.cc
    sampling/node_counts_io.cc
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

    # Phase 3: Four-Level Feature Store build procedure (L1-L4)
    ${CMAKE_SOURCE_DIR}/src/query/procedure/builtin/gnn_build_feature_store_procedure.cc

    # Spec #4-B T4.9: Post-hoc topology snapshot build procedure
    ${CMAKE_SOURCE_DIR}/src/query/procedure/builtin/gnn_build_topology_snapshot_procedure.cc

    # Phase 4: GNN Training procedure (GraphSAGE training orchestration)
    ${CMAKE_SOURCE_DIR}/src/query/procedure/builtin/gnn_train_procedure.cc

    # Phase 6: GNN Prediction procedure (load checkpoint + inference)
    ${CMAKE_SOURCE_DIR}/src/query/procedure/builtin/gnn_predict_procedure.cc

    # Phase 6: GNN checkpoint listing procedure
    ${CMAKE_SOURCE_DIR}/src/query/procedure/builtin/gnn_list_checkpoints_procedure.cc

    # Phase 6: GNN checkpoint existence-check procedure
    ${CMAKE_SOURCE_DIR}/src/query/procedure/builtin/gnn_checkpoint_exists_procedure.cc

    # Phase 6: GNN checkpoint delete procedure
    ${CMAKE_SOURCE_DIR}/src/query/procedure/builtin/gnn_checkpoint_delete_procedure.cc

    # Phase 4: Training Pipeline
    training/label_store.cc
    training/split_store.cc
    training/batch_assembler.cc
    training/graph_block_builder.cc
    training/batch_timing_log.cc
    training/npy_writer.cc

    # Gate (A): feature<->label integrity check (2026-06-02)
    training/feature_label_integrity.cc

    # Phase 4: GNN Models
    models/graphsage_model.cc

    # Task 9: Training Loop
    training/training_loop.cc

    # Spec C3 stage 1: Async batch prefetcher
    training/async_batch_prefetcher.cc

    # Phase 6: Output / Embedding Writer
    output/embedding_writer.cc
    output/model_checkpoint.cc
    output/auto_checkpointer.cc
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

# Optional io_uring linkage for DirectIoReader
if(URING_FOUND)
    target_compile_definitions(mdb_gnn_core PUBLIC ENABLE_IO_URING)
    target_link_libraries(mdb_gnn_core PUBLIC ${URING_LIBRARIES})
    target_include_directories(mdb_gnn_core PUBLIC ${URING_INCLUDE_DIRS})
endif()

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
# CUDA Feature Assembler (requires nvcc + CUDAToolkit)
# ----------------------------------------------------------------------------
if(GNN_CUDA_ENABLED AND CUDAToolkit_FOUND)
    enable_language(CUDA)
    target_sources(mdb_gnn_core PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/core/feature_assembler.cu
    )
    set_source_files_properties(
        ${CMAKE_CURRENT_SOURCE_DIR}/core/feature_assembler.cu
        PROPERTIES LANGUAGE CUDA
    )
    target_compile_options(mdb_gnn_core PRIVATE
        $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr -std=c++17>
    )
    target_compile_definitions(mdb_gnn_core PUBLIC ENABLE_CUDA_ASSEMBLER)
    message(STATUS "CUDA Feature Assembler enabled")
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
        target_link_libraries(test_gnn_storage mdb_gnn_core GTest::gtest_main ${TBB_LIBS})
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

        # Block store tests (offline computation-graph bake)
        add_executable(test_block_store
            tests/test_block_store.cc
        )
        target_link_libraries(test_block_store mdb_gnn_core GTest::gtest_main ${TORCH_LIBRARIES})
        target_compile_options(test_block_store PRIVATE ${TORCH_CXX_FLAGS_LIST})
        target_compile_definitions(test_block_store PRIVATE _GLIBCXX_USE_CXX11_ABI=1)
        set_target_properties(test_block_store PROPERTIES
            BUILD_RPATH "${CMAKE_SOURCE_DIR}/third_party/libtorch/lib"
            INSTALL_RPATH "${CMAKE_SOURCE_DIR}/third_party/libtorch/lib"
        )
        add_test(NAME BlockStoreTests COMMAND test_block_store)

        # Sample content fingerprint tests (STEP 8 — staleness, isolable)
        add_executable(test_sample_fingerprint
            tests/test_sample_fingerprint.cc
        )
        target_link_libraries(test_sample_fingerprint mdb_gnn_core GTest::gtest_main)
        target_compile_options(test_sample_fingerprint PRIVATE ${TORCH_CXX_FLAGS_LIST})
        target_compile_definitions(test_sample_fingerprint PRIVATE _GLIBCXX_USE_CXX11_ABI=1)
        set_target_properties(test_sample_fingerprint PROPERTIES
            BUILD_RPATH "${CMAKE_SOURCE_DIR}/third_party/libtorch/lib"
            INSTALL_RPATH "${CMAKE_SOURCE_DIR}/third_party/libtorch/lib"
        )
        add_test(NAME SampleFingerprintTests COMMAND test_sample_fingerprint)

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

        # SampleStorage deserialized-sample cache tests (train hot path)
        add_executable(test_sample_storage_cache
            tests/test_sample_storage_cache.cc
        )
        target_link_libraries(test_sample_storage_cache
            GTest::gtest_main
            mdb_gnn_core
            millenniumdb
            antlr4_cpp_runtime
            ssl crypto
            ${TORCH_LIBRARIES}
        )
        set_target_properties(test_sample_storage_cache PROPERTIES
            BUILD_RPATH "${CMAKE_SOURCE_DIR}/third_party/libtorch/lib"
            INSTALL_RPATH "${CMAKE_SOURCE_DIR}/third_party/libtorch/lib"
        )
        add_test(NAME SampleStorageCacheTests COMMAND test_sample_storage_cache)

        # FourLevelStore tests (CpuCache L2, later GpuCache L1 + coordinator)
        add_executable(test_four_level_store
            tests/test_four_level_store.cc
        )
        # Link gtest_main FIRST so its main() takes precedence over serdi's main()
        target_link_libraries(test_four_level_store
            GTest::gtest_main
            mdb_gnn_core
            millenniumdb
            antlr4_cpp_runtime
            ssl crypto
            ${TORCH_LIBRARIES}
        )
        set_target_properties(test_four_level_store PROPERTIES
            BUILD_RPATH "${CMAKE_SOURCE_DIR}/third_party/libtorch/lib"
            INSTALL_RPATH "${CMAKE_SOURCE_DIR}/third_party/libtorch/lib"
        )
        add_test(NAME FourLevelStoreTests COMMAND test_four_level_store)

        # Suppress known CUDA driver leak (cuInit 408 bytes) in ASan builds
        set(ASAN_SUPP_FILE "${CMAKE_SOURCE_DIR}/tests/asan_suppressions.txt")
        if(EXISTS "${ASAN_SUPP_FILE}")
            set_property(TEST FourLevelStoreTests PROPERTY
                ENVIRONMENT "LSAN_OPTIONS=suppressions=${ASAN_SUPP_FILE}")
            set_property(TEST BatchMaterializerTests PROPERTY
                ENVIRONMENT "LSAN_OPTIONS=suppressions=${ASAN_SUPP_FILE}")
        endif()

        # DirectIoReader tests (O_DIRECT + io_uring, with pread fallback)
        add_executable(test_direct_io
            tests/test_direct_io.cc
        )
        target_link_libraries(test_direct_io
            GTest::gtest_main
            mdb_gnn_core
        )
        set_target_properties(test_direct_io PROPERTIES
            BUILD_RPATH "${CMAKE_SOURCE_DIR}/third_party/libtorch/lib"
            INSTALL_RPATH "${CMAKE_SOURCE_DIR}/third_party/libtorch/lib"
        )
        add_test(NAME DirectIoTests COMMAND test_direct_io)

        # Feature Assembler tests (CUDA kernel + LibTorch fallback)
        add_executable(test_feature_assembler
            tests/test_feature_assembler.cc
        )
        target_link_libraries(test_feature_assembler
            GTest::gtest_main
            mdb_gnn_core
            ${TORCH_LIBRARIES}
        )
        set_target_properties(test_feature_assembler PROPERTIES
            BUILD_RPATH "${CMAKE_SOURCE_DIR}/third_party/libtorch/lib"
            INSTALL_RPATH "${CMAKE_SOURCE_DIR}/third_party/libtorch/lib"
        )
        add_test(NAME FeatureAssemblerTests COMMAND test_feature_assembler)
        if(EXISTS "${ASAN_SUPP_FILE}")
            set_property(TEST FeatureAssemblerTests PROPERTY
                ENVIRONMENT "LSAN_OPTIONS=suppressions=${ASAN_SUPP_FILE}")
        endif()

        # LabelStore tests (Phase 4: training labels)
        if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/tests/test_label_store.cc")
            add_executable(test_label_store tests/test_label_store.cc)
            target_link_libraries(test_label_store mdb_gnn_core GTest::gtest_main)
            set_target_properties(test_label_store PROPERTIES
                BUILD_RPATH "${CMAKE_SOURCE_DIR}/third_party/libtorch/lib"
                INSTALL_RPATH "${CMAKE_SOURCE_DIR}/third_party/libtorch/lib"
            )
            add_test(NAME LabelStoreTests COMMAND test_label_store)
        endif()

        # BatchTimingLog tests (DiskGNN parity: per-batch CSV instrumentation)
        add_executable(test_batch_timing_log tests/test_batch_timing_log.cc)
        target_link_libraries(test_batch_timing_log PRIVATE mdb_gnn_core gtest gtest_main)
        add_test(NAME BatchTimingLogTests COMMAND test_batch_timing_log)

        # AddrTable format header tests (Path 4 Task 1: on-disk format + header invariants)
        add_executable(test_addr_table_format tests/test_addr_table_format.cc)
        target_link_libraries(test_addr_table_format PRIVATE mdb_gnn_core gtest gtest_main)
        add_test(NAME AddrTableFormatTests COMMAND test_addr_table_format)

        # AddrTable writer tests (Path 4 Task 2: classify + serialize per-batch sidecar)
        add_executable(test_addr_table_writer tests/test_addr_table_writer.cc)
        target_link_libraries(test_addr_table_writer PRIVATE mdb_gnn_core gtest gtest_main)
        add_test(NAME AddrTableWriterTests COMMAND test_addr_table_writer)

        # ConsolidatedSlim format tests (DiskGNN-adoption Plan 1 Phase 0.1:
        # on-disk header for the consolidated cold-feature file)
        add_executable(test_consolidated_slim_format tests/test_consolidated_slim_format.cc)
        target_link_libraries(test_consolidated_slim_format PRIVATE mdb_gnn_core gtest gtest_main)
        add_test(NAME ConsolidatedSlimFormatTests COMMAND test_consolidated_slim_format)

        # ConsolidatedSlim reader tests (DiskGNN-adoption Plan 1 Phase 2:
        # header validation + O_DIRECT-capable pread_exact)
        add_executable(test_consolidated_slim_reader tests/test_consolidated_slim_reader.cc)
        target_link_libraries(test_consolidated_slim_reader PRIVATE mdb_gnn_core gtest gtest_main)
        add_test(NAME ConsolidatedSlimReaderTests COMMAND test_consolidated_slim_reader)

        # AddrTable reader tests (Path 4 Task 3: open + validate + parse 9 arrays)
        add_executable(test_addr_table_reader tests/test_addr_table_reader.cc)
        target_link_libraries(test_addr_table_reader PRIVATE mdb_gnn_core gtest gtest_main)
        add_test(NAME AddrTableReaderTests COMMAND test_addr_table_reader)

        # SplitStore tests (Task 3: split assignments for training)
        if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/tests/test_split_store.cc")
            add_executable(test_split_store tests/test_split_store.cc)
            target_link_libraries(test_split_store mdb_gnn_core GTest::gtest_main)
            set_target_properties(test_split_store PROPERTIES
                BUILD_RPATH "${CMAKE_SOURCE_DIR}/third_party/libtorch/lib"
                INSTALL_RPATH "${CMAKE_SOURCE_DIR}/third_party/libtorch/lib"
            )
            add_test(NAME SplitStoreTests COMMAND test_split_store)
        endif()

        # BatchAssembler tests (Task 6: assemble MiniBatch from features+topology+labels)
        if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/tests/test_batch_assembler.cc")
            add_executable(test_batch_assembler tests/test_batch_assembler.cc)
            # Link gtest_main FIRST so its main() takes precedence over serdi's main()
            # in the millenniumdb static library. SampleStorage needs Logger from millenniumdb.
            target_link_libraries(test_batch_assembler
                GTest::gtest_main
                mdb_gnn_core
                millenniumdb
                antlr4_cpp_runtime
                ssl crypto
                ${TORCH_LIBRARIES}
            )
            set_target_properties(test_batch_assembler PROPERTIES
                BUILD_RPATH "${CMAKE_SOURCE_DIR}/third_party/libtorch/lib"
                INSTALL_RPATH "${CMAKE_SOURCE_DIR}/third_party/libtorch/lib"
            )
            add_test(NAME BatchAssemblerTests COMMAND test_batch_assembler)
            if(EXISTS "${ASAN_SUPP_FILE}")
                set_property(TEST BatchAssemblerTests PROPERTY
                    ENVIRONMENT "LSAN_OPTIONS=suppressions=${ASAN_SUPP_FILE}")
            endif()
        endif()

        # Gate (A): feature<->label integrity tests (2026-06-02)
        if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/tests/test_feature_label_integrity.cc")
            add_executable(test_feature_label_integrity tests/test_feature_label_integrity.cc)
            target_link_libraries(test_feature_label_integrity mdb_gnn_core GTest::gtest_main)
            set_target_properties(test_feature_label_integrity PROPERTIES
                BUILD_RPATH "${CMAKE_SOURCE_DIR}/third_party/libtorch/lib"
                INSTALL_RPATH "${CMAKE_SOURCE_DIR}/third_party/libtorch/lib"
            )
            add_test(NAME FeatureLabelIntegrityTests COMMAND test_feature_label_integrity)
            if(EXISTS "${ASAN_SUPP_FILE}")
                set_property(TEST FeatureLabelIntegrityTests PROPERTY
                    ENVIRONMENT "LSAN_OPTIONS=suppressions=${ASAN_SUPP_FILE}")
            endif()
        endif()

        # NpyWriter tests (Task 8: export tensors as .npy files)
        if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/tests/test_npy_writer.cc")
            add_executable(test_npy_writer tests/test_npy_writer.cc)
            target_link_libraries(test_npy_writer mdb_gnn_core GTest::gtest_main)
            set_target_properties(test_npy_writer PROPERTIES
                BUILD_RPATH "${CMAKE_SOURCE_DIR}/third_party/libtorch/lib"
                INSTALL_RPATH "${CMAKE_SOURCE_DIR}/third_party/libtorch/lib"
            )
            add_test(NAME NpyWriterTests COMMAND test_npy_writer)
        endif()

        # TrainingLoop tests (Task 9: epoch iteration, early stopping, validation)
        if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/tests/test_training_loop.cc")
            add_executable(test_training_loop tests/test_training_loop.cc)
            target_link_libraries(test_training_loop
                GTest::gtest_main
                mdb_gnn_core
                millenniumdb
                antlr4_cpp_runtime
                ssl crypto
                ${TORCH_LIBRARIES}
            )
            set_target_properties(test_training_loop PROPERTIES
                BUILD_RPATH "${CMAKE_SOURCE_DIR}/third_party/libtorch/lib"
                INSTALL_RPATH "${CMAKE_SOURCE_DIR}/third_party/libtorch/lib"
            )
            add_test(NAME TrainingLoopTests COMMAND test_training_loop)
            if(EXISTS "${ASAN_SUPP_FILE}")
                set_property(TEST TrainingLoopTests PROPERTY
                    ENVIRONMENT "LSAN_OPTIONS=suppressions=${ASAN_SUPP_FILE}")
            endif()
        endif()

        # Spec C3 stage 3: CUDA streams (Module 1) tests
        if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/tests/test_stage3_streams.cc")
            add_executable(test_stage3_streams tests/test_stage3_streams.cc)
            target_link_libraries(test_stage3_streams
                GTest::gtest_main
                mdb_gnn_core
                millenniumdb
                antlr4_cpp_runtime
                ssl crypto
                ${TORCH_LIBRARIES}
            )
            set_target_properties(test_stage3_streams PROPERTIES
                BUILD_RPATH "${CMAKE_SOURCE_DIR}/third_party/libtorch/lib"
                INSTALL_RPATH "${CMAKE_SOURCE_DIR}/third_party/libtorch/lib"
            )
            add_test(NAME Stage3StreamsTests COMMAND test_stage3_streams)
            if(EXISTS "${ASAN_SUPP_FILE}")
                set_property(TEST Stage3StreamsTests PROPERTY
                    ENVIRONMENT "LSAN_OPTIONS=suppressions=${ASAN_SUPP_FILE}")
            endif()
        endif()

        # Spec C3 stage 1: AsyncBatchPrefetcher tests
        if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/tests/test_async_batch_prefetcher.cc")
            add_executable(test_async_batch_prefetcher tests/test_async_batch_prefetcher.cc)
            target_link_libraries(test_async_batch_prefetcher
                GTest::gtest_main
                mdb_gnn_core
                millenniumdb
                antlr4_cpp_runtime
                ssl crypto
                ${TORCH_LIBRARIES}
            )
            set_target_properties(test_async_batch_prefetcher PROPERTIES
                BUILD_RPATH "${CMAKE_SOURCE_DIR}/third_party/libtorch/lib"
                INSTALL_RPATH "${CMAKE_SOURCE_DIR}/third_party/libtorch/lib"
            )
            add_test(NAME AsyncBatchPrefetcherTests COMMAND test_async_batch_prefetcher)
            if(EXISTS "${ASAN_SUPP_FILE}")
                set_property(TEST AsyncBatchPrefetcherTests PROPERTY
                    ENVIRONMENT "LSAN_OPTIONS=suppressions=${ASAN_SUPP_FILE}")
            endif()
        endif()

        add_executable(pipeline_overlap_test
            ${CMAKE_SOURCE_DIR}/src/tests/pipeline_overlap_test.cc
        )
        target_link_libraries(pipeline_overlap_test mdb_gnn_core GTest::gtest_main ${TBB_LIBS})
        set_target_properties(pipeline_overlap_test PROPERTIES
            BUILD_RPATH "${CMAKE_SOURCE_DIR}/third_party/libtorch/lib"
            INSTALL_RPATH "${CMAKE_SOURCE_DIR}/third_party/libtorch/lib"
        )
        add_test(NAME PipelineOverlapTests COMMAND pipeline_overlap_test)

        add_executable(page_cache_hint_test
            ${CMAKE_SOURCE_DIR}/src/tests/page_cache_hint_test.cc
        )
        target_link_libraries(page_cache_hint_test mdb_gnn_core GTest::gtest_main)
        set_target_properties(page_cache_hint_test PROPERTIES
            BUILD_RPATH "${CMAKE_SOURCE_DIR}/third_party/libtorch/lib"
            INSTALL_RPATH "${CMAKE_SOURCE_DIR}/third_party/libtorch/lib"
        )
        add_test(NAME PageCacheHintTests COMMAND page_cache_hint_test)

        # TrainingLoop resume / callback tests (Phase 6 Sub-2 Task 4.3)
        if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/tests/test_training_loop_resume.cc")
            add_executable(test_training_loop_resume tests/test_training_loop_resume.cc)
            target_link_libraries(test_training_loop_resume
                GTest::gtest_main
                mdb_gnn_core
                millenniumdb
                antlr4_cpp_runtime
                ssl crypto
                ${TORCH_LIBRARIES}
            )
            set_target_properties(test_training_loop_resume PROPERTIES
                BUILD_RPATH "${CMAKE_SOURCE_DIR}/third_party/libtorch/lib"
                INSTALL_RPATH "${CMAKE_SOURCE_DIR}/third_party/libtorch/lib"
            )
            add_test(NAME TrainingLoopResumeTests COMMAND test_training_loop_resume)
            if(EXISTS "${ASAN_SUPP_FILE}")
                set_property(TEST TrainingLoopResumeTests PROPERTY
                    ENVIRONMENT "LSAN_OPTIONS=suppressions=${ASAN_SUPP_FILE}")
            endif()
        endif()

        # GnnMeta tests (Task 1: metadata reader/writer)
        if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/tests/test_gnn_meta.cc")
            add_executable(test_gnn_meta tests/test_gnn_meta.cc)
            target_link_libraries(test_gnn_meta mdb_gnn_core GTest::gtest_main)
            set_target_properties(test_gnn_meta PROPERTIES
                BUILD_RPATH "${CMAKE_SOURCE_DIR}/third_party/libtorch/lib"
                INSTALL_RPATH "${CMAKE_SOURCE_DIR}/third_party/libtorch/lib"
            )
            add_test(NAME GnnMetaTests COMMAND test_gnn_meta)
        endif()

        # GraphSAGE MEAN model tests (Task 7)
        add_executable(test_graphsage_model tests/test_graphsage_model.cc)
        target_link_libraries(test_graphsage_model mdb_gnn_core GTest::gtest_main)
        set_target_properties(test_graphsage_model PROPERTIES
            BUILD_RPATH "${CMAKE_SOURCE_DIR}/third_party/libtorch/lib"
            INSTALL_RPATH "${CMAKE_SOURCE_DIR}/third_party/libtorch/lib"
        )
        add_test(NAME GraphSAGEModelTests COMMAND test_graphsage_model)

        # Active-set-shrinking refactor: determinism + shape regression gate
        add_executable(test_active_set_equivalence tests/test_active_set_equivalence.cc)
        target_link_libraries(test_active_set_equivalence mdb_gnn_core GTest::gtest_main)
        set_target_properties(test_active_set_equivalence PROPERTIES
            BUILD_RPATH "${CMAKE_SOURCE_DIR}/third_party/libtorch/lib"
            INSTALL_RPATH "${CMAKE_SOURCE_DIR}/third_party/libtorch/lib"
        )
        add_test(NAME ActiveSetEquivalenceTests COMMAND test_active_set_equivalence)

        # EmbeddingWriter tests (Task 8: Phase A/B logic in isolation)
        if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/tests/test_embedding_writer.cc")
            add_executable(test_embedding_writer tests/test_embedding_writer.cc)
            # Link gtest_main FIRST so its main() takes precedence over serdi's main()
            # in the millenniumdb static library. SampleStorage needs Logger from millenniumdb.
            target_link_libraries(test_embedding_writer
                GTest::gtest_main
                mdb_gnn_core
                millenniumdb
                antlr4_cpp_runtime
                ssl crypto
                ${TORCH_LIBRARIES}
            )
            set_target_properties(test_embedding_writer PROPERTIES
                BUILD_RPATH "${CMAKE_SOURCE_DIR}/third_party/libtorch/lib"
                INSTALL_RPATH "${CMAKE_SOURCE_DIR}/third_party/libtorch/lib"
            )
            add_test(NAME EmbeddingWriterTests COMMAND test_embedding_writer)
            if(EXISTS "${ASAN_SUPP_FILE}")
                set_property(TEST EmbeddingWriterTests PROPERTY
                    ENVIRONMENT "LSAN_OPTIONS=suppressions=${ASAN_SUPP_FILE}")
            endif()
        endif()

        # Model Checkpoint tests (Phase 6 Sub-2: save/load/list/exists/delete)
        if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/tests/test_model_checkpoint.cc")
            add_executable(test_model_checkpoint
                tests/test_model_checkpoint.cc
            )
            target_link_libraries(test_model_checkpoint
                GTest::gtest_main
                mdb_gnn_core
                millenniumdb
                antlr4_cpp_runtime
                ssl crypto
                ${TORCH_LIBRARIES}
            )
            set_target_properties(test_model_checkpoint PROPERTIES
                BUILD_RPATH "${CMAKE_SOURCE_DIR}/third_party/libtorch/lib"
                INSTALL_RPATH "${CMAKE_SOURCE_DIR}/third_party/libtorch/lib"
            )
            add_test(NAME ModelCheckpointTests COMMAND test_model_checkpoint)
            if(EXISTS "${ASAN_SUPP_FILE}")
                set_property(TEST ModelCheckpointTests PROPERTY
                    ENVIRONMENT "LSAN_OPTIONS=suppressions=${ASAN_SUPP_FILE}")
            endif()
        endif()

        # AutoCheckpointer tests (Phase 6 Sub-2)
        add_executable(test_auto_checkpointer tests/test_auto_checkpointer.cc)
        target_link_libraries(test_auto_checkpointer
            GTest::gtest_main
            mdb_gnn_core
            millenniumdb
            antlr4_cpp_runtime
            ssl crypto
            ${TORCH_LIBRARIES}
        )
        set_target_properties(test_auto_checkpointer PROPERTIES
            BUILD_RPATH "${CMAKE_SOURCE_DIR}/third_party/libtorch/lib"
            INSTALL_RPATH "${CMAKE_SOURCE_DIR}/third_party/libtorch/lib"
        )
        add_test(NAME AutoCheckpointerTests COMMAND test_auto_checkpointer)
        if(EXISTS "${ASAN_SUPP_FILE}")
            set_property(TEST AutoCheckpointerTests PROPERTY
                ENVIRONMENT "LSAN_OPTIONS=suppressions=${ASAN_SUPP_FILE}")
        endif()

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
