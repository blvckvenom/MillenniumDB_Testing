#pragma once
#include "query/procedure/procedure.h"

namespace GQL::Procedures {

class GnnCheckpointExistsProcedure : public Procedure {
public:
    std::string name() const override { return "gnn_checkpoint_exists"; }
    std::string qualified_name() const override { return "gnn_checkpoint_exists"; }
    std::string description() const override {
        return "Check whether a named GNN checkpoint exists (both .pt and .ckptmeta present).";
    }
    std::vector<Parameter> parameters() const override {
        return {
            Parameter("projectionName", ParamType::STRING, true, "Projection name"),
            Parameter("outputDir",      ParamType::STRING, true, "Output dir under gnn_output/"),
            Parameter("name",           ParamType::STRING, true, "Checkpoint basename"),
        };
    }
    std::vector<YieldField> yield_fields() const override {
        return {
            YieldField{"exists",   YieldType::BOOL,   "true iff both .pt and .ckptmeta are present"},
            YieldField{"basename", YieldType::STRING, "Absolute path without extension (whether it exists or not)"},
        };
    }
    void execute(ProcedureContext& ctx) override;
};

} // namespace GQL::Procedures
