#pragma once

#include "query/procedure/procedure.h"

namespace GQL::Procedures {

class GnnCreateFeaturesFromQueryProcedure : public Procedure {
public:
    std::string name() const override { return "gnn_create_features_from_query"; }
    std::string qualified_name() const override { return "gnn_create_features_from_query"; }
    std::string description() const override {
        return "Create native GNN feature files (.fmat/.rmap) from a GQL query result";
    }

    std::vector<Parameter> parameters() const override {
        return {
            Parameter("featureName", ParamType::STRING, true,  "Feature name to create"),
            Parameter("query",       ParamType::STRING, true,  "GQL query returning node_id and float features"),
            Parameter("options",     ParamType::ANY,    false, "Options: {normalize: 'none'|'zscore', appendToFeature: STRING}"),
        };
    }

    std::vector<YieldField> yield_fields() const override {
        return {
            YieldField{"featureName",       YieldType::STRING, "Feature name created"},
            YieldField{"nodeCount",         YieldType::INT,    "Number of feature rows"},
            YieldField{"featureDim",        YieldType::INT,    "Feature dimension"},
            YieldField{"baseFeatureDim",    YieldType::INT,    "Base feature dimension when appending, otherwise 0"},
            YieldField{"queryFeatureDim",   YieldType::INT,    "Query-generated feature dimension"},
            YieldField{"fmatPath",          YieldType::STRING, "Path to FeatureMatrix file"},
            YieldField{"rmapPath",          YieldType::STRING, "Path to RowMapping file"},
            YieldField{"normalized",        YieldType::BOOL,   "Whether z-score normalization was applied to query feature columns"},
            YieldField{"appendedToFeature", YieldType::STRING, "Base feature name when appending, otherwise empty"},
        };
    }

    void execute(ProcedureContext& ctx) override;
};

} // namespace GQL::Procedures
