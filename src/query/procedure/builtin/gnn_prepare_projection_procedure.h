#pragma once

#include "query/procedure/procedure.h"

namespace GQL::Procedures {

class GnnPrepareProjectionProcedure : public Procedure {
public:
    std::string name() const override { return "gnn_prepare_projection"; }
    std::string qualified_name() const override { return "gnn_prepare_projection"; }
    std::string description() const override {
        return "Prepare GNN metadata, labels, and splits for an existing graph projection";
    }

    std::vector<Parameter> parameters() const override {
        return {
            Parameter("projectionName", ParamType::STRING, true, "Projection to prepare"),
            Parameter("options", ParamType::ANY, true,
                "Options: {includeFeatures: STRING, labelProperty: STRING, splitProperty: STRING}"),
        };
    }

    std::vector<YieldField> yield_fields() const override {
        return {
            YieldField{"projectionName", YieldType::STRING, "Projection prepared"},
            YieldField{"featureName", YieldType::STRING, "FeatureMatrix name attached to the projection"},
            YieldField{"nodeCount", YieldType::INT, "Number of rows in the feature RowMapping"},
            YieldField{"featureDim", YieldType::INT, "Feature dimension"},
            YieldField{"numClasses", YieldType::INT, "Number of distinct integer labels found"},
            YieldField{"hasLabels", YieldType::BOOL, "Whether labels.bin was written"},
            YieldField{"hasSplits", YieldType::BOOL, "Whether splits.bin was written"},
            YieldField{"labelsPath", YieldType::STRING, "Path to labels.bin, or empty when not written"},
            YieldField{"splitsPath", YieldType::STRING, "Path to splits.bin, or empty when not written"},
            YieldField{"metaPath", YieldType::STRING, "Path to gnn_meta.bin"},
        };
    }

    void execute(ProcedureContext& ctx) override;
};

} // namespace GQL::Procedures
