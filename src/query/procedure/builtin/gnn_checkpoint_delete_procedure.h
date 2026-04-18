#pragma once
#include "query/procedure/procedure.h"

namespace GQL::Procedures {

class GnnCheckpointDeleteProcedure : public Procedure {
public:
    std::string name() const override { return "gnn_checkpoint_delete"; }
    std::string qualified_name() const override { return "gnn_checkpoint_delete"; }
    std::string description() const override {
        return "Delete a named GNN checkpoint (.pt and .ckptmeta). "
               "Idempotent: missing files are not an error.";
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
            YieldField{"deleted",     YieldType::BOOL,   "Always true unless the filesystem error is fatal (in which case the procedure throws)"},
            YieldField{"basename",    YieldType::STRING, "Absolute basename operated on"},
            YieldField{"ptDeleted",   YieldType::BOOL,   "true iff the .pt file existed and was deleted"},
            YieldField{"metaDeleted", YieldType::BOOL,   "true iff the .ckptmeta file existed and was deleted"},
        };
    }
    void execute(ProcedureContext& ctx) override;
};

} // namespace GQL::Procedures
