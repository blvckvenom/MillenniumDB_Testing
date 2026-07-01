#pragma once
#include "query/procedure/procedure.h"

namespace GQL::Procedures {

class GnnListCheckpointsProcedure : public Procedure {
public:
    std::string name() const override { return "gnn_list_checkpoints"; }
    std::string qualified_name() const override { return "gnn_list_checkpoints"; }
    std::string description() const override {
        return "List GNN model checkpoints stored under a projection's "
               "gnn_output directory. Optional outputDir and name filters.";
    }
    std::vector<Parameter> parameters() const override {
        return {
            Parameter("projectionName", ParamType::STRING, true,
                "Projection whose gnn_output/ tree to scan"),
            Parameter("outputDir",      ParamType::STRING, false,
                "Restrict to one output_dir (optional; scans all if omitted)"),
            Parameter("name",           ParamType::STRING, false,
                "Filter by checkpoint basename (e.g. 'best_model')"),
        };
    }
    std::vector<YieldField> yield_fields() const override {
        return {
            YieldField{"basename",        YieldType::STRING, "Absolute basename without extension"},
            YieldField{"outputDir",       YieldType::STRING, "Containing output_dir name"},
            YieldField{"saveKind",        YieldType::STRING, "'full' or 'weights_only'"},
            YieldField{"epoch",           YieldType::INT,    "Epoch recorded in the checkpoint"},
            YieldField{"bestValAccuracy", YieldType::FLOAT,  "Best val accuracy recorded"},
            YieldField{"modelType",       YieldType::STRING, "e.g. 'graphsage'"},
            YieldField{"projectionName",  YieldType::STRING, "Projection recorded in checkpoint"},
            YieldField{"creationTime",    YieldType::INT,    "Unix seconds"},
            YieldField{"ptBytes",         YieldType::INT,    ".pt file size in bytes"},
            YieldField{"metaBytes",       YieldType::INT,    ".ckptmeta file size in bytes"},
        };
    }
    void execute(ProcedureContext& ctx) override;
};

} // namespace GQL::Procedures
