#include "query/executor/binding_iter/procedure/gds_graph_drop.h"
#include "graph_models/common/conversions.h"
#include "graph_models/gql/conversions.h"
#include "graph_models/gql/gql_graph_catalog.h"
#include "query/parser/expr/gql/expr_term.h"
#include "query/parser/op/gql/op_procedure.h"
#include "query/query_context.h"
#include <filesystem>
#include <iostream>
#include <memory>
#include <vector>

using namespace GQL;

bool test_drop_alias()
{
    QueryContext ctx;
    QueryContext::set_query_ctx(&ctx);
    std::string dir = (std::filesystem::temp_directory_path() / "gql_catalog_drop").string();
    std::filesystem::remove_all(dir);
    GqlGraphCatalog catalog(dir);
    Value node("*");
    Value rel("*");
    Map cfg;
    catalog.project("g1", node, rel, cfg);

    VarId gVar = ctx.get_or_create_var("g");
    VarId nVar = ctx.get_or_create_var("n");
    auto available = OpProcedure::get_procedure_available_yield_variable_names(
        OpProcedure::ProcedureType::GDS_GRAPH_DROP);

    std::vector<VarId> yields;
    yields.reserve(available.size());
    for (const auto& name : available) {
        if (name == "graphName") {
            yields.push_back(gVar);
        } else if (name == "nodeCount") {
            yields.push_back(nVar);
        } else {
            yields.push_back(ctx.get_internal_var());
        }
    }

    std::vector<std::unique_ptr<Expr>> args;
    args.emplace_back(std::make_unique<ExprTerm>(Conversions::pack_string_simple("g1")));

    GdsGraphDrop iter(catalog, std::move(args), std::move(yields));
    Binding binding(ctx.get_var_size());
    iter.begin(binding);

    if (!iter.next()) {
        std::cerr << "drop failed\n";
        return true;
    }

    if (Conversions::unpack_string(binding[gVar]) != "g1") {
        std::cerr << "graphName mismatch\n";
        return true;
    }
    if (Common::Conversions::unpack_int(binding[nVar]) <= 0) {
        std::cerr << "nodeCount invalid\n";
        return true;
    }
    return false;
}

int main()
{
    bool error = false;
    if (test_drop_alias())
        error = true;
    return error ? 1 : 0;
}
