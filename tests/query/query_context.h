#pragma once
#include "query/var_id.h"
class QueryContext {
public:
    const char* get_var_name(VarId) const { return ""; }
};
inline QueryContext& get_query_ctx() { static QueryContext ctx; return ctx; }
