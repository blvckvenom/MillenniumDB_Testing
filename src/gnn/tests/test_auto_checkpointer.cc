#include "test_helpers.h"

#include "gnn/models/graphsage_model.h"
#include "gnn/output/auto_checkpointer.h"

using AutoCheckpointerTest = GnnStorageTest;

namespace {

std::unique_ptr<mdb::gnn::GraphSAGEModel> make_model() {
    torch::manual_seed(42);
    mdb::gnn::GraphSAGEConfig c;
    c.input_dim = 8; c.hidden_dim = 4; c.num_classes = 2; c.num_layers = 2;
    c.dropout = 0.5; c.normalize = false;
    return std::make_unique<mdb::gnn::GraphSAGEModel>(c);
}

mdb::gnn::TrainingState base_state() {
    mdb::gnn::TrainingState s;
    s.input_dim = 8; s.hidden_dim = 4; s.num_classes = 2; s.num_layers = 2;
    s.dropout = 0.5f; s.normalize = false;
    s.model_type = "graphsage"; s.projection_name = "test";
    for (size_t i = 0; i < s.gnn_meta_hash.size(); ++i) s.gnn_meta_hash[i] = uint8_t(i);
    return s;
}

} // anon

TEST_F(AutoCheckpointerTest, CtorCreatesCheckpointDir) {
    auto m = make_model();
    torch::optim::Adam opt(m->parameters(), torch::optim::AdamOptions(0.01));
    auto dir = test_dir_ / "never_created" / "checkpoints";

    mdb::gnn::AutoCheckpointer ac(*m, opt, dir, base_state());
    EXPECT_TRUE(std::filesystem::is_directory(dir));
}

TEST_F(AutoCheckpointerTest, OnEpochEndSavesOnBestVal) {
    auto m = make_model();
    torch::optim::Adam opt(m->parameters(), torch::optim::AdamOptions(0.01));
    auto dir = test_dir_ / "ckpts";
    mdb::gnn::AutoCheckpointer ac(*m, opt, dir, base_state());

    mdb::gnn::TrainingLoop::EpochEvent e1{0, 0.8, 0.60, 0, true};
    mdb::gnn::TrainingLoop::EpochEvent e2{1, 0.7, 0.55, 1, false};
    mdb::gnn::TrainingLoop::EpochEvent e3{2, 0.6, 0.70, 0, true};

    ac.on_epoch_end(e1);
    ac.on_epoch_end(e2);
    ac.on_epoch_end(e3);

    EXPECT_EQ(ac.saves_written(), 2u);  // e1 and e3, not e2
    EXPECT_DOUBLE_EQ(ac.best_val_seen(), 0.70);
    EXPECT_TRUE(std::filesystem::exists(dir / "best_model.pt"));
    EXPECT_TRUE(std::filesystem::exists(dir / "best_model.ckptmeta"));
}

TEST_F(AutoCheckpointerTest, SaveFinalWritesFinalModel) {
    auto m = make_model();
    torch::optim::Adam opt(m->parameters(), torch::optim::AdamOptions(0.01));
    auto dir = test_dir_ / "ckpts";
    mdb::gnn::AutoCheckpointer ac(*m, opt, dir, base_state());

    auto s = base_state();
    s.epoch = 50;
    s.best_val_accuracy = 0.81f;
    ac.save_final(s);

    EXPECT_TRUE(std::filesystem::exists(dir / "final_model.pt"));
    EXPECT_TRUE(std::filesystem::exists(dir / "final_model.ckptmeta"));
    EXPECT_EQ(ac.saves_written(), 1u);
}

TEST_F(AutoCheckpointerTest, PolicyAllFalseSavesNothing) {
    auto m = make_model();
    torch::optim::Adam opt(m->parameters(), torch::optim::AdamOptions(0.01));
    auto dir = test_dir_ / "ckpts";

    mdb::gnn::AutoCheckpointer::Policy p;
    p.save_on_best_val = false;
    p.save_final       = false;
    mdb::gnn::AutoCheckpointer ac(*m, opt, dir, base_state(), p);

    mdb::gnn::TrainingLoop::EpochEvent e{0, 0.8, 0.9, 0, true};
    ac.on_epoch_end(e);
    ac.save_final(base_state());

    EXPECT_EQ(ac.saves_written(), 0u);
    EXPECT_FALSE(std::filesystem::exists(dir / "best_model.pt"));
    EXPECT_FALSE(std::filesystem::exists(dir / "final_model.pt"));
}

TEST_F(AutoCheckpointerTest, MultipleImprovementsKeepOverwriting) {
    auto m = make_model();
    torch::optim::Adam opt(m->parameters(), torch::optim::AdamOptions(0.01));
    auto dir = test_dir_ / "ckpts";
    mdb::gnn::AutoCheckpointer ac(*m, opt, dir, base_state());

    for (uint64_t i = 0; i < 5; ++i) {
        // Each epoch improves val_accuracy
        mdb::gnn::TrainingLoop::EpochEvent e{
            i, 1.0 - 0.1 * i, 0.5 + 0.05 * (i + 1), 0, true};
        ac.on_epoch_end(e);
    }
    EXPECT_EQ(ac.saves_written(), 5u);
    EXPECT_DOUBLE_EQ(ac.best_val_seen(), 0.75);

    // Only one best_model file exists (overwritten)
    EXPECT_TRUE(std::filesystem::exists(dir / "best_model.pt"));
}

TEST_F(AutoCheckpointerTest, TiedValAccNotConsideredImprovement) {
    auto m = make_model();
    torch::optim::Adam opt(m->parameters(), torch::optim::AdamOptions(0.01));
    auto dir = test_dir_ / "ckpts";
    mdb::gnn::AutoCheckpointer ac(*m, opt, dir, base_state());

    mdb::gnn::TrainingLoop::EpochEvent e1{0, 0.8, 0.70, 0, true};
    mdb::gnn::TrainingLoop::EpochEvent e2{1, 0.7, 0.70, 1, false};  // tie
    ac.on_epoch_end(e1);
    ac.on_epoch_end(e2);

    EXPECT_EQ(ac.saves_written(), 1u);  // only e1
}

TEST_F(AutoCheckpointerTest, CustomBasenamesHonored) {
    auto m = make_model();
    torch::optim::Adam opt(m->parameters(), torch::optim::AdamOptions(0.01));
    auto dir = test_dir_ / "ckpts";

    mdb::gnn::AutoCheckpointer::Policy p;
    p.best_basename  = "custom_best";
    p.final_basename = "custom_final";
    mdb::gnn::AutoCheckpointer ac(*m, opt, dir, base_state(), p);

    ac.on_epoch_end({0, 0.5, 0.8, 0, true});
    ac.save_final(base_state());

    EXPECT_TRUE(std::filesystem::exists(dir / "custom_best.pt"));
    EXPECT_TRUE(std::filesystem::exists(dir / "custom_final.pt"));
    EXPECT_FALSE(std::filesystem::exists(dir / "best_model.pt"));
}

TEST_F(AutoCheckpointerTest, BaseStateFieldsPersistedInSaves) {
    auto m = make_model();
    torch::optim::Adam opt(m->parameters(), torch::optim::AdamOptions(0.01));
    auto dir = test_dir_ / "ckpts";

    auto s = base_state();
    s.projection_name = "distinctive_name";
    mdb::gnn::AutoCheckpointer ac(*m, opt, dir, s);
    ac.on_epoch_end({0, 0.5, 0.7, 0, true});

    auto loaded = mdb::gnn::ModelCheckpoint::read_ckptmeta(
        dir / "best_model.ckptmeta");
    EXPECT_EQ(loaded.projection_name, "distinctive_name");
    EXPECT_FLOAT_EQ(loaded.best_val_accuracy, 0.7f);
    EXPECT_EQ(loaded.epoch, 1u);   // stored as "next epoch to run"
}
