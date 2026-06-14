#pragma once

#include "query/procedure/procedure.h"

namespace GQL::Procedures {

class GnnCreateStructuralFeaturesProcedure : public Procedure {
public:
    std::string name() const override { return "gnn_create_structural_features"; }
    std::string qualified_name() const override { return "gnn_create_structural_features"; }
    std::string description() const override {
        return "Create native GNN structural node features (.fmat/.rmap) from a node label and edge type";
    }

    std::vector<Parameter> parameters() const override {
        return {
            Parameter("featureName", ParamType::STRING, true,  "Feature name to create"),
            Parameter("nodeLabel",   ParamType::STRING, true,  "Node label to featurize"),
            Parameter("edgeType",    ParamType::STRING, true,  "Edge type used for structural counts"),
            Parameter("options",     ParamType::ANY,    false, "Options: {normalize: 'none'|'zscore', appendToFeature: STRING}"),
        };
    }

    std::vector<YieldField> yield_fields() const override {
        return {
            YieldField{"featureName",          YieldType::STRING, "Feature name created"},
            YieldField{"nodeCount",            YieldType::INT,    "Number of feature rows"},
            YieldField{"featureDim",           YieldType::INT,    "Feature dimension"},
            YieldField{"baseFeatureDim",       YieldType::INT,    "Base feature dimension when appending, otherwise 0"},
            YieldField{"structuralFeatureDim", YieldType::INT,    "Structural feature dimension"},
            YieldField{"fmatPath",             YieldType::STRING, "Path to FeatureMatrix file"},
            YieldField{"rmapPath",             YieldType::STRING, "Path to RowMapping file"},
            YieldField{"normalized",           YieldType::BOOL,   "Whether z-score normalization was applied"},
            YieldField{"appendedToFeature",    YieldType::STRING, "Base feature name when appending, otherwise empty"},
        };
    }

    void execute(ProcedureContext& ctx) override;
};

} // namespace GQL::Procedures
