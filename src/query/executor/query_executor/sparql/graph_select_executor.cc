#include "graph_select_executor.h"

#include "graph_models/rdf_model/conversions.h"

using namespace SPARQL;

uint64_t GraphSelectExecutor::execute(std::ostream& os) {
    uint64_t result_count = 0;
    binding = std::make_unique<Binding>(get_query_ctx().get_var_size());
    root->begin(*binding);

    while (root->next()) {
        result_count++;
        auto sep = "";
        for (auto var : projection_vars) {
            auto value = (*binding)[var];
            if (!value.is_null()) {
                os << sep;
                write_and_escape_ttl(os, value);
                sep = " ";
            }
        }
        os << " .\n";
    }
    return result_count;
}

void GraphSelectExecutor::analyze(std::ostream& os, bool /*print_stats*/, int indent) const {
    os << std::string(indent, ' ') << "GraphSelectExecutor(";
    for (size_t i = 0; i < projection_vars.size(); i++) {
        if (i != 0) os << ", ";
        os << '?' << get_query_ctx().get_var_name(projection_vars[i]);
    }
    os << ")\n";

    BindingIterPrinter printer(os, false, indent + 2);
    root->accept_visitor(printer);
}
