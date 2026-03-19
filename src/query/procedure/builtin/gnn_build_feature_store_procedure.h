#pragma once

#include "query/procedure/procedure.h"

namespace GQL::Procedures {

class GnnBuildFeatureStoreProcedure : public Procedure {
public:
    std::string name() const override { return "gnn_build_feature_store"; }
    std::string qualified_name() const override { return "gnn_build_feature_store"; }
    std::string description() const override {
        return "Build a four-level hierarchical feature store (L1 GPU cache, "
               "L2 CPU cache, L3 disk cache, L4 packed slim batches)";
    }

    std::vector<Parameter> parameters() const override {
        return {
            Parameter("sampleName",  ParamType::STRING, true,  "Existing sample set name"),
            Parameter("featureName", ParamType::STRING, true,  "Registered feature name"),
            Parameter("options",     ParamType::ANY,    false, "Options: {gpu_budget_mb, cpu_budget_mb, reorder, force, strategy, numHashes, segmentSize}"),
        };
    }

    std::vector<YieldField> yield_fields() const override {
        return {
            YieldField{"sampleName",    YieldType::STRING, "Sample set name"},
            YieldField{"featureName",   YieldType::STRING, "Feature name used"},
            YieldField{"l1Nodes",       YieldType::INT,    "Nodes in GPU cache (L1)"},
            YieldField{"l2Nodes",       YieldType::INT,    "Nodes in CPU cache (L2)"},
            YieldField{"l3Nodes",       YieldType::INT,    "Shared nodes (L3, freq > 1)"},
            YieldField{"l4Nodes",       YieldType::INT,    "Unique nodes (L4, freq == 1)"},
            YieldField{"gpuAvailable",  YieldType::BOOL,   "Whether CUDA was available"},
            YieldField{"buildTimeMs",   YieldType::INT,    "Total build time in milliseconds"},
        };
    }

    void execute(ProcedureContext& ctx) override;
};

} // namespace GQL::Procedures
