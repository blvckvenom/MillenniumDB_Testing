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
                "normalize, randomSeed, outputDir, exportEmbeddings}"),
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
        };
    }

    void execute(ProcedureContext& ctx) override;
};

} // namespace GQL::Procedures
