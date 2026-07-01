#include "test_helpers.h"

#include "gnn/storage/vram_calib_io.h"

// Fixture alias — file roundtrip needs a temp dir; the pure parser tests don't.
using VramCalibIoTest = GnnStorageTest;

// ===========================================================================
// Pure format/parse tests (no filesystem)
// ===========================================================================

TEST(VramCalibTextTest, FormatMatchesCanonicalLayout) {
    VramCalib c;
    c.recommended_l1_cache_mb = 5426;
    c.measured_peak_mb        = 14999;
    c.measured_reserve_mb     = 9879;
    c.total_vram_mb           = 15817;
    // Byte-exact layout gnn_train has always written: 4 key=value lines, each
    // newline-terminated, in this fixed order.
    EXPECT_EQ(format_vram_calib_text(c),
              "recommended_l1_cache_mb=5426\n"
              "measured_peak_mb=14999\n"
              "measured_reserve_mb=9879\n"
              "total_vram_mb=15817\n");
}

TEST(VramCalibTextTest, ParseRoundTrip) {
    VramCalib in;
    in.recommended_l1_cache_mb = 5426;
    in.measured_peak_mb        = 14999;
    in.measured_reserve_mb     = 9879;
    in.total_vram_mb           = 15817;
    auto out = parse_vram_calib_text(format_vram_calib_text(in));
    ASSERT_TRUE(out.has_value());
    EXPECT_EQ(out->recommended_l1_cache_mb, 5426u);
    EXPECT_EQ(out->measured_peak_mb,        14999u);
    EXPECT_EQ(out->measured_reserve_mb,     9879u);
    EXPECT_EQ(out->total_vram_mb,           15817u);
}

TEST(VramCalibTextTest, ParsesRealCalibFileContents) {
    // The exact text persisted on the papers100M/16GB box.
    const std::string text =
        "recommended_l1_cache_mb=5426\n"
        "measured_peak_mb=14999\n"
        "measured_reserve_mb=9879\n"
        "total_vram_mb=15817\n";
    auto out = parse_vram_calib_text(text);
    ASSERT_TRUE(out.has_value());
    EXPECT_EQ(out->recommended_l1_cache_mb, 5426u);
}

TEST(VramCalibTextTest, MissingKeysDefaultToZero) {
    auto out = parse_vram_calib_text("recommended_l1_cache_mb=2048\n");
    ASSERT_TRUE(out.has_value());
    EXPECT_EQ(out->recommended_l1_cache_mb, 2048u);
    EXPECT_EQ(out->measured_peak_mb,    0u);
    EXPECT_EQ(out->measured_reserve_mb, 0u);
    EXPECT_EQ(out->total_vram_mb,       0u);
}

TEST(VramCalibTextTest, ToleratesCrlfWhitespaceAndUnknownLines) {
    const std::string text =
        "# a comment line with no equals\r\n"
        "recommended_l1_cache_mb = 4096 \r\n"
        "unknown_key=999\r\n"
        "measured_peak_mb=12000\r\n";
    auto out = parse_vram_calib_text(text);
    ASSERT_TRUE(out.has_value());
    EXPECT_EQ(out->recommended_l1_cache_mb, 4096u);
    EXPECT_EQ(out->measured_peak_mb,        12000u);
}

TEST(VramCalibTextTest, EmptyOrUnrecognizedReturnsNullopt) {
    EXPECT_FALSE(parse_vram_calib_text("").has_value());
    EXPECT_FALSE(parse_vram_calib_text("no keys here\njust prose\n").has_value());
}

TEST(VramCalibTextTest, NonNumericValueSkippedNotAdopted) {
    // A garbled value must not be silently read as 0 for that key while still
    // accepting the good keys around it.
    const std::string text =
        "recommended_l1_cache_mb=not_a_number\n"
        "measured_peak_mb=12000\n";
    auto out = parse_vram_calib_text(text);
    ASSERT_TRUE(out.has_value());                 // measured_peak_mb is valid
    EXPECT_EQ(out->recommended_l1_cache_mb, 0u);  // garbled line skipped
    EXPECT_EQ(out->measured_peak_mb,        12000u);
}

// ===========================================================================
// File I/O roundtrip + absence handling
// ===========================================================================

TEST_F(VramCalibIoTest, WriteThenReadRoundTrip) {
    auto path = test_dir_ / "node_features_vram_calib.txt";
    VramCalib in;
    in.recommended_l1_cache_mb = 5900;
    in.measured_peak_mb        = 14800;
    in.measured_reserve_mb     = 9300;
    in.total_vram_mb           = 15817;
    ASSERT_TRUE(write_vram_calib(path, in));

    auto out = read_vram_calib(path);
    ASSERT_TRUE(out.has_value());
    EXPECT_EQ(out->recommended_l1_cache_mb, 5900u);
    EXPECT_EQ(out->measured_peak_mb,        14800u);
    EXPECT_EQ(out->measured_reserve_mb,     9300u);
    EXPECT_EQ(out->total_vram_mb,           15817u);
}

TEST_F(VramCalibIoTest, ReadMissingFileReturnsNullopt) {
    auto path = test_dir_ / "does_not_exist_vram_calib.txt";
    EXPECT_FALSE(read_vram_calib(path).has_value());
}
