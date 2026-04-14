#include "query_context.h"

#include <iostream>
#include "graph_models/gql/projection/projection_query_context.h"
#include "system/string_manager.h"
#include "system/tmp_manager.h"

QueryContext::QueryContext() {
    buffer1 = new char[StringManager::MAX_STRING_SIZE];
    buffer2 = new char[StringManager::MAX_STRING_SIZE];
}

QueryContext::QueryContext(QueryContext&& other) :
    buffer1(std::exchange(other.buffer1, nullptr)),
    buffer2(std::exchange(other.buffer2, nullptr))
{ }

QueryContext::~QueryContext() {
    delete[] buffer1;
    delete[] buffer2;
    // projection_ctx will be automatically destroyed here with complete type
}

// Note: prepare() is now implemented inline in query_context.h (from origin/main)
// The inline implementation is identical to the previous .cc implementation

void QueryContext::clear_active_projection() {
    active_projection.clear();
    projection_ctx.reset();
}

void QueryContext::load_projection(const std::string& proj_name) {
    std::cerr << "[QueryContext] load_projection() called for '" << proj_name << "'" << std::endl;
    active_projection = proj_name;
    projection_ctx = std::make_unique<GQL::ProjectionQueryContext>(proj_name);
    std::cerr << "[QueryContext] Projection context created, valid=" << projection_ctx->is_valid() << std::endl;
}

void QueryContext::unload_projection() {
    active_projection.clear();
    projection_ctx.reset();
}
