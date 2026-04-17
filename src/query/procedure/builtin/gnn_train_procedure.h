#pragma once

#include "query/procedure/procedure.h"

namespace GQL::Procedures {

class GnnTrainProcedure : public Procedure {
public:
    std::string name() const override { return "gnn_train"; }
    std::string qualified_name() const override { return "gnn_train"; }
    std::string description() const override {
        return "Train a GraphSAGE model on a pre-sampled graph projection with offline "
               "batches, labels, and train/val/test splits. Exports model checkpoint, "
               "training log, and optionally node embeddings.";
    }

    std::vector<Parameter> parameters() const override {
        return {
            Parameter("sampleName",  ParamType::STRING, true,  "Existing sample set name"),
            Parameter("featureName", ParamType::STRING, true,  "Registered feature name (e.g. 'node_features')"),
            Parameter("options",     ParamType::ANY,    false,
                "Options: {model, hiddenDim, dropout, epochs, lr, patience, tolerance, "
                "normalize, randomSeed, outputDir, exportEmbeddings, writeProperty, "
                "resumeFrom, saveOnBestVal, saveFinal}"),
        };
    }

    std::vector<YieldField> yield_fields() const override {
        return {
            YieldField{"modelName",       YieldType::STRING, "Model identifier (e.g. 'graphsage')"},
            YieldField{"ranEpochs",       YieldType::INT,    "Number of epochs executed"},
            YieldField{"didConverge",     YieldType::BOOL,   "Stopped by convergence or early stopping"},
            YieldField{"bestValAccuracy", YieldType::FLOAT,  "Best validation accuracy achieved"},
            YieldField{"testAccuracy",    YieldType::FLOAT,  "Test accuracy (-1.0 if no test split)"},
            YieldField{"trainSeconds",    YieldType::FLOAT,  "Total wall-clock training time in seconds"},
            YieldField{"l1HitRatio",      YieldType::FLOAT,  "FourLevelStore L1 (GPU) cache hit ratio"},
            YieldField{"l2HitRatio",      YieldType::FLOAT,  "FourLevelStore L2 (CPU pinned) cache hit ratio"},
            YieldField{"l3Reads",         YieldType::INT,    "FourLevelStore L3 (disk reordered) read count"},
            YieldField{"l4Reads",         YieldType::INT,    "FourLevelStore L4 (packed batch) read count"},
            YieldField{"nodesWritten",    YieldType::INT,    "Nodes whose embeddings were written to projection (0 if writeProperty not set)"},
            YieldField{"nodesInferred",   YieldType::INT,    "Non-seed nodes inferred during write-back (0 if writeProperty not set)"},
            YieldField{"inferenceMillis", YieldType::FLOAT,  "Wall-clock time for non-seed inference in ms (0.0 if writeProperty not set)"},
            YieldField{"writeMillis",     YieldType::FLOAT,  "Wall-clock time for projection writes in ms (0.0 if writeProperty not set)"},
            YieldField{"bestCheckpointPath",  YieldType::STRING, "Absolute path (no extension) to best_model checkpoint; empty if disabled or no improvement"},
            YieldField{"finalCheckpointPath", YieldType::STRING, "Absolute path (no extension) to final_model checkpoint; empty if disabled"},
            YieldField{"resumedFromEpoch",    YieldType::INT,    "Epoch index from which training resumed (0 if fresh training)"},
        };
    }

    void execute(ProcedureContext& ctx) override;
};

} // namespace GQL::Procedures
