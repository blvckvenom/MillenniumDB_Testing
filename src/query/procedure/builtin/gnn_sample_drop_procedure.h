#pragma once

#include <string>
#include <vector>

#include "query/procedure/procedure.h"
#include "query/procedure/procedure_context.h"

namespace GQL {
namespace Procedures {

/**
 * @brief Deletes a GNN sample set and all its data.
 *
 * Removes the sample directory and all files within it.
 * This operation is irreversible.
 *
 * ## Syntax
 *
 * @code{.gql}
 *   CALL gnn_sample_drop(sampleName)
 *   YIELD success, message
 * @endcode
 *
 * ## Parameters
 *
 * | Name | Type | Required | Description |
 * |------|------|----------|-------------|
 * | sampleName | STRING | Yes | Name of the sample set to delete |
 *
 * ## Examples
 *
 * @code{.gql}
 *   -- Delete a sample set
 *   CALL gnn_sample_drop('old_samples')
 *   YIELD success, message
 *   RETURN success, message;
 *
 *   -- Check before deleting
 *   CALL gnn_sample_list() YIELD sampleName
 *   WHERE sampleName = 'old_samples'
 *   RETURN sampleName;
 *   -- If exists:
 *   CALL gnn_sample_drop('old_samples') YIELD success;
 * @endcode
 *
 * @see gnn_sample_list() to list all sample sets
 * @see gnn_sample_info() to get information before deleting
 * @see gnn_offline_sample() to create new sample sets
 */
class GnnSampleDropProcedure : public Procedure {
public:
    std::string name() const override {
        return "gnn_sample_drop";
    }

    std::string qualified_name() const override {
        return "gnn_sample_drop";
    }

    std::string description() const override {
        return "Deletes a GNN sample set and all its data. "
               "This operation is irreversible.";
    }

    std::vector<Parameter> parameters() const override {
        return {
            Parameter("sampleName", ParamType::STRING, true,
                "Name of the sample set to delete")
        };
    }

    std::vector<YieldField> yield_fields() const override {
        return {
            YieldField{"success", YieldType::BOOL,
                "TRUE if the sample set was deleted successfully"},
            YieldField{"message", YieldType::STRING,
                "Status message describing the result"}
        };
    }

    void execute(ProcedureContext& ctx) override;
};

} // namespace Procedures
} // namespace GQL
