#include "query/procedure/builtin/gnn_predict_procedure.h"

#include <stdexcept>

#include "query/procedure/procedure_context.h"

namespace GQL::Procedures {

void GnnPredictProcedure::execute(ProcedureContext& /*ctx*/) {
    throw std::runtime_error("gnn_predict: not yet implemented");
}

} // namespace GQL::Procedures
