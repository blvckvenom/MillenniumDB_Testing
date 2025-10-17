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

void QueryContext::prepare(BufferManager::VersionScope& version_scope, std::chrono::seconds timeout) {
    blank_node_ids.clear();
    blank_node_count = 0;

    var_ctx.internal_var_counter = 0;
    var_ctx.var_names = {};
    var_ctx.var_map.clear();

    // NOTE: Do NOT reset projection context here!
    // The projection is set during query parsing (visitUseGraphClause)
    // and must persist through query execution.
    // active_projection.clear();  // REMOVED - projection set by parser must persist
    // projection_ctx.reset();      // REMOVED - projection set by parser must persist

    const auto start = std::chrono::system_clock::now();
    thread_info.interruption_requested = false;
    thread_info.time_start = start;
    thread_info.timeout = start + timeout;

    start_version  = version_scope.start_version;
    result_version = version_scope.start_version + (version_scope.is_editable ? 1 : 0);

    cancellation_token = get_uuid();

    tmp_manager.reset(thread_info.worker_index);
}

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
