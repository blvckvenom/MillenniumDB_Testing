#include "query/executor/binding_iter/procedure/gds_graph_list.h"
#include "graph_models/common/conversions.h"
#include "graph_models/gql/conversions.h"
#include "graph_models/gql/gql_graph_catalog.h"
#include "query/query_context.h"
#include <filesystem>
#include <iostream>
#include <memory>
#include <vector>

using namespace GQL;

bool test_no_filter()
{
    QueryContext ctx;
    QueryContext::set_query_ctx(&ctx);
    std::string dir = (std::filesystem::temp_directory_path() / "gql_catalog_test").string();
    std::filesystem::remove_all(dir);
    GqlGraphCatalog catalog(dir);
    Value node("*");
    Value rel("*");
    Map cfg;
    catalog.project("g1", node, rel, cfg);
    catalog.project("g2", node, rel, cfg);

    VarId graphNameVar = ctx.get_or_create_var("graphName");
    VarId nodeCountVar = ctx.get_or_create_var("nodeCount");
    std::vector<std::unique_ptr<Expr>> args;
    std::vector<VarId> yields { graphNameVar, nodeCountVar };
    GdsGraphList iter(catalog, std::move(args), std::move(yields));
    Binding binding(ctx.get_var_size());
    iter.begin(binding);
    std::vector<std::string> names;
    std::vector<int64_t> counts;
    while (iter.next()) {
        names.push_back(Conversions::unpack_string(binding[graphNameVar]));
        counts.push_back(Common::Conversions::unpack_int(binding[nodeCountVar]));
    }
    if (names.size() != 2) {
        std::cerr << "expected 2 entries\n";
        return true;
    }
    bool has_g1 = (names[0] == "g1" || names[1] == "g1");
    bool has_g2 = (names[0] == "g2" || names[1] == "g2");
    if (!has_g1 || !has_g2) {
        std::cerr << "missing names\n";
        return true;
    }
    if (counts[0] == 0 || counts[1] == 0) {
        std::cerr << "counts zero\n";
        return true;
    }
    return false;
}

bool test_with_filter()
{
    QueryContext ctx;
    QueryContext::set_query_ctx(&ctx);
    std::string dir = (std::filesystem::temp_directory_path() / "gql_catalog_test2").string();
    std::filesystem::remove_all(dir);
    GqlGraphCatalog catalog(dir);
    Value node("*");
    Value rel("*");
    Map cfg;
    catalog.project("g1", node, rel, cfg);
    catalog.project("g2", node, rel, cfg);

    VarId graphNameVar = ctx.get_or_create_var("graphName");
    VarId nodeCountVar = ctx.get_or_create_var("nodeCount");
    std::vector<std::unique_ptr<Expr>> args;
    args.emplace_back(std::make_unique<ExprTerm>(Conversions::pack_string_simple("g1")));
    std::vector<VarId> yields { graphNameVar, nodeCountVar };
    GdsGraphList iter(catalog, std::move(args), std::move(yields));
    Binding binding(ctx.get_var_size());
    iter.begin(binding);
    std::vector<std::string> names;
    while (iter.next()) {
        names.push_back(Conversions::unpack_string(binding[graphNameVar]));
    }
    if (names.size() != 1 || names[0] != "g1") {
        std::cerr << "filter failed\n";
        return true;
    }
    return false;
}

int main()
{
    bool error = false;
    if (test_no_filter())
        error = true;
    if (test_with_filter())
        error = true;
    return error ? 1 : 0;
}
