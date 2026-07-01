#pragma once

#include "query/procedure/procedure.h"

namespace GQL::Procedures {

class GnnMaterializeBatchesProcedure : public Procedure {
public:
    std::string name() const override { return "gnn_materialize_batches"; }
    std::string qualified_name() const override { return "gnn_materialize_batches"; }
    std::string description() const override {
        return "Materialize packed feature batches from offline sampling output "
               "(L3 MinHash reordering + L4 packed batch files)";
    }

    std::vector<Parameter> parameters() const override {
        return {
            Parameter("sampleName",  ParamType::STRING, true,  "Existing sample set name"),
            Parameter("featureName", ParamType::STRING, true,  "Registered feature name (e.g. 'node_features')"),
            Parameter("options",     ParamType::ANY,    false, "Options: {reorder, strategy, numHashes, segmentSize, force}"),
        };
    }

    std::vector<YieldField> yield_fields() const override {
        return {
            YieldField{"sampleName",    YieldType::STRING, "Sample set name"},
            YieldField{"featureName",   YieldType::STRING, "Feature name used"},
            YieldField{"totalBatches",  YieldType::INT,    "Number of packed batch files"},
            YieldField{"reordered",     YieldType::BOOL,   "Whether L3 reordering was performed"},
            YieldField{"reorderTimeMs", YieldType::INT,    "L3 reordering time in milliseconds"},
            YieldField{"packTimeMs",    YieldType::INT,    "L4 packing time in milliseconds"},
            YieldField{"totalTimeMs",   YieldType::INT,    "Total wall time in milliseconds"},
            YieldField{"packedDir",     YieldType::STRING, "Path to packed batch directory"},
        };
    }

    void execute(ProcedureContext& ctx) override;
};

} // namespace GQL::Procedures
