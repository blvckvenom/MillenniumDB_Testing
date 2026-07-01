/**
 * @file test_model_checkpoint_corrupt_meta.cc
 * @brief Robustness tests for ModelCheckpoint::read_ckptmeta against corrupt
 *        or crafted var-length fields.
 *
 * The .ckptmeta tail encodes projection_name / model_type / epoch_losses with
 * raw uint32 length prefixes. read_ckptmeta must reject a length that exceeds
 * the bytes remaining in the file BEFORE driving any allocation — otherwise a
 * corrupt or attacker-placed file in the checkpoints directory (scanned by
 * list_checkpoints) can trigger multi-GiB zero-filled allocations.
 */

#include "test_helpers.h"

#include <cstdint>
#include <fstream>
#include <string>

#include "gnn/output/model_checkpoint.h"

using ModelCheckpointCorruptMetaTest = GnnStorageTest;

namespace {

// Minimal valid state with EMPTY var-length fields, so the serialized tail is
// exactly [projection_name_len][model_type_len][num_epoch_losses]
// (3 x uint32 = the last 12 bytes of the file).
mdb::gnn::TrainingState empty_tail_state() {
    mdb::gnn::TrainingState s;
    s.epoch       = 3;
    s.input_dim   = 8;
    s.hidden_dim  = 4;
    s.num_classes = 2;
    s.num_layers  = 1;
    return s;
}

// Overwrite 4 bytes at `offset_from_end` (counted back from EOF) with `value`.
void patch_u32_at_tail(
    const std::filesystem::path& path,
    std::streamoff               offset_from_end,
    uint32_t                     value)
{
    std::fstream f(path, std::ios::binary | std::ios::in | std::ios::out);
    ASSERT_TRUE(f.is_open()) << "cannot open " << path;
    f.seekp(-offset_from_end, std::ios::end);
    f.write(reinterpret_cast<const char*>(&value), sizeof(value));
    ASSERT_TRUE(f.good()) << "patch failed on " << path;
}

} // anon

// ---------------------------------------------------------------------------
// RejectsOversizedStringLength — a projection_name length claiming ~4 GiB the
// file does not contain must throw, not resize(0xFFFFFFFF).
// ---------------------------------------------------------------------------
TEST_F(ModelCheckpointCorruptMetaTest, RejectsOversizedStringLength) {
    auto path = test_dir_ / "corrupt_str.ckptmeta";
    mdb::gnn::ModelCheckpoint::write_ckptmeta(path, empty_tail_state());

    // projection_name_len is the 3rd uint32 from EOF (tail layout:
    // [proj_len][model_len][num_losses]).
    patch_u32_at_tail(path, 12, 0xFFFFFFFFu);

    try {
        mdb::gnn::ModelCheckpoint::read_ckptmeta(path);
        FAIL() << "expected read_ckptmeta to reject the corrupt length";
    } catch (const std::runtime_error& e) {
        EXPECT_NE(std::string(e.what()).find("exceeds remaining"),
                  std::string::npos)
            << "unexpected error: " << e.what();
    }
}

// ---------------------------------------------------------------------------
// RejectsOversizedEpochLossCount — 2^32-1 doubles would be a ~32 GiB
// value-initialized allocation if taken at face value.
// ---------------------------------------------------------------------------
TEST_F(ModelCheckpointCorruptMetaTest, RejectsOversizedEpochLossCount) {
    auto path = test_dir_ / "corrupt_losses.ckptmeta";
    mdb::gnn::ModelCheckpoint::write_ckptmeta(path, empty_tail_state());

    // num_epoch_losses is the last uint32 in the file.
    patch_u32_at_tail(path, 4, 0xFFFFFFFFu);

    try {
        mdb::gnn::ModelCheckpoint::read_ckptmeta(path);
        FAIL() << "expected read_ckptmeta to reject the corrupt count";
    } catch (const std::runtime_error& e) {
        EXPECT_NE(std::string(e.what()).find("exceeds remaining"),
                  std::string::npos)
            << "unexpected error: " << e.what();
    }
}

// ---------------------------------------------------------------------------
// ValidTailStillRoundtrips — the remaining-bytes bound must not reject a
// well-formed file whose var-length fields exactly consume the tail.
// ---------------------------------------------------------------------------
TEST_F(ModelCheckpointCorruptMetaTest, ValidTailStillRoundtrips) {
    auto path = test_dir_ / "valid.ckptmeta";
    auto s = empty_tail_state();
    s.projection_name = "proj";
    s.model_type      = "graphsage";
    s.epoch_losses    = {0.5, 0.25};
    mdb::gnn::ModelCheckpoint::write_ckptmeta(path, s);

    auto r = mdb::gnn::ModelCheckpoint::read_ckptmeta(path);
    EXPECT_EQ(r.projection_name, "proj");
    EXPECT_EQ(r.model_type, "graphsage");
    ASSERT_EQ(r.epoch_losses.size(), 2u);
    EXPECT_DOUBLE_EQ(r.epoch_losses[0], 0.5);
    EXPECT_DOUBLE_EQ(r.epoch_losses[1], 0.25);
}
