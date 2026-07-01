#pragma once

#include <gtest/gtest.h>

#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

#include <unistd.h> // getpid

#include "gnn/storage/feature_matrix.h"
#include "gnn/storage/feature_matrix_header.h"
#include "gnn/storage/gnn_dtype.h"

namespace fs = std::filesystem;
using namespace mdb::gnn;

/**
 * @brief Shared fixture for GNN storage tests.
 *
 * Creates a temporary directory scoped to the current PID,
 * so parallel test runs (e.g. CI) don't collide.
 */
class GnnStorageTest : public ::testing::Test {
protected:
    fs::path test_dir_;

    void SetUp() override {
        test_dir_ = fs::temp_directory_path()
            / ("mdb_test_gnn_" + std::to_string(getpid()));
        fs::create_directories(test_dir_);
    }

    void TearDown() override {
        fs::remove_all(test_dir_);
    }

    fs::path test_path(const std::string& name) {
        return test_dir_ / name;
    }
};

/**
 * @brief Parameter struct for parameterized dtype tests.
 */
struct DtypeTestParam {
    GnnDtype    dtype;
    std::string name;
};
