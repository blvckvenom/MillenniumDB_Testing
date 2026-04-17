#pragma once

#include "query/procedure/procedure.h"

namespace GQL::Procedures {

class GnnPredictProcedure : public Procedure {
public:
    std::string name() const override { return "gnn_predict"; }
    std::string qualified_name() const override { return "gnn_predict"; }
    std::string description() const override {
        return "Load a trained GNN model from a checkpoint and run inference over "
               "all nodes in a sample. Optionally writes embeddings back to the "
               "projection as a tensor property.";
    }

    std::vector<Parameter> parameters() const override {
        return {
            Parameter("sampleName",      ParamType::STRING, true,
                "Existing sample with batches to run inference on"),
            Parameter("featureName",     ParamType::STRING, true,
                "Registered feature name (must match checkpoint)"),
            Parameter("checkpointPath",  ParamType::STRING, true,
                "Checkpoint basename (relative or absolute, no extension)"),
            Parameter("options",         ParamType::ANY,    false,
                "Options: {writeProperty, exportEmbeddings, outputDir}"),
        };
    }

    std::vector<YieldField> yield_fields() const override {
        return {
            YieldField{"checkpointPath",         YieldType::STRING, "Absolute path of the loaded checkpoint"},
            YieldField{"checkpointEpoch",        YieldType::INT,    "Epoch recorded in the checkpoint"},
            YieldField{"checkpointValAccuracy",  YieldType::FLOAT,  "Best val accuracy recorded in the checkpoint"},
            YieldField{"numBatches",             YieldType::INT,    "Number of batches processed"},
            YieldField{"numSeedNodes",           YieldType::INT,    "Total seed nodes across all batches"},
            YieldField{"embeddingDim",           YieldType::INT,    "Hidden dimension = embedding dim"},
            YieldField{"nodesWritten",           YieldType::INT,    "Seed nodes whose embeddings were written (0 if writeProperty unset)"},
            YieldField{"nodesInferred",          YieldType::INT,    "Non-seed nodes inferred during write-back"},
            YieldField{"inferenceMillis",        YieldType::FLOAT,  "Wall-clock time for non-seed inference"},
            YieldField{"writeMillis",            YieldType::FLOAT,  "Wall-clock time for projection writes"},
            YieldField{"l1HitRatio",             YieldType::FLOAT,  "FourLevelStore L1 hit ratio"},
            YieldField{"l2HitRatio",             YieldType::FLOAT,  "FourLevelStore L2 hit ratio"},
            YieldField{"l3Reads",                YieldType::INT,    "FourLevelStore L3 read count"},
            YieldField{"l4Reads",                YieldType::INT,    "FourLevelStore L4 read count"},
        };
    }

    void execute(ProcedureContext& ctx) override;
};

} // namespace GQL::Procedures
