#include "http_gql_session.h"

#include <boost/beast/ssl.hpp>

#include "misc/logger.h"
#include "misc/trim.h"
#include "network/server/server.h"
#include "network/server/session/http/gql_request_parser.h"
#include "network/server/session/http/request_parser.h"
#include "network/server/session/http/response/http_response_buffer.h"
#include "network/server/session/http/response/return_codes_helper.h"
#include "query/optimizer/property_graph_model/executor_constructor.cc"
#include "query/parser/gql_query_parser.h"

using namespace boost;
using namespace MDBServer;
namespace http = beast::http;
namespace chrono = std::chrono;

template<typename stream_t>
HttpGQLSession<stream_t>::HttpGQLSession(
    Server& server,
    stream_t&& stream,
    http::request<http::string_body>&& request,
    chrono::seconds query_timeout
) :
    server(server),
    stream(std::move(stream)),
    request(std::move(request)),
    query_timeout(query_timeout)
{
    worker = get_query_ctx().thread_info.worker_index;
}

template<typename stream_t>
HttpGQLSession<stream_t>::~HttpGQLSession()
{
    if constexpr (std::is_same_v<stream_t, asio::ip::tcp::socket>) {
        if (stream.is_open()) {
            stream.close();
        }
    } else if constexpr (std::is_same_v<stream_t, beast::ssl_stream<asio::ip::tcp::socket>>) {
        if (stream.next_layer().is_open()) {
            stream.next_layer().close();
        }
    }
}

template<typename stream_t>
void HttpGQLSession<stream_t>::run(std::unique_ptr<HttpGQLSession> obj)
{
    HttpResponseBuffer response_buffer(obj->stream);

    std::ostream response_ostream(&response_buffer);

    // Without this line ConnectionException won't be caught properly
    response_ostream.exceptions(std::ifstream::failbit | std::ifstream::badbit);

    auto request_type = GQL::RequestParser::get_request_type(obj->request.target());

    if (request_type == Protocol::RequestType::HEALTH_CHECK) {
        return send_ok(response_ostream);
    }

    if (request_type == Protocol::RequestType::AUTH) {
        auto&& [user, pass] = Common::RequestParser::parse_auth(obj->request);
        auto&& [auth_token, valid_until] = obj->server.create_auth_token(user, pass);
        if (auth_token.empty()) {
            return send_unauthorized(response_ostream);
        } else {
            return send_token_authorized(response_ostream, auth_token, valid_until);
        }
    }

    auto auth_token = Common::RequestParser::get_auth(obj->request);

    if (request_type == Protocol::RequestType::CANCEL) {
        auto&& [worker_id, cancel_token] = Common::RequestParser::parse_cancel(obj->request);
        auto cancel_res = obj->server.try_cancel(worker_id, cancel_token);
        if (cancel_res) {
            return send_ok(response_ostream);
        } else {
            return send_bad_request(response_ostream, "Not Canceled");
        }
    }

    auto&& [query, response_type] = GQL::RequestParser::parse_query(obj->request);

    if (!obj->server.authorize(request_type, auth_token)) {
        return send_unauthorized(response_ostream);
    }

    if (request_type == Protocol::RequestType::UPDATE) {
        obj->execute_update_query(query, response_ostream);
    } else /* (request_type == Protocol::RequestType::QUERY) */ {
        obj->execute_readonly_query(query, response_ostream, response_type);
    }
}

template<typename stream_t>
void HttpGQLSession<stream_t>::execute_readonly_query(
    const std::string& query,
    std::ostream& os,
    GQL::ReturnType response_type
)
{
    // Check if query contains PROJECT function - if so, we need an editable scope
    // to persist the projection data to disk
    bool needs_editable = (query.find("PROJECT") != std::string::npos
                        || query.find("project") != std::string::npos
                        || query.find("CALL") != std::string::npos
                        || query.find("call") != std::string::npos);

    // Declared here because the destruction need to be after calling execute_query_plan
    std::unique_ptr<BufferManager::VersionScope> version_scope;

    // Hold update_execution_mutex for the entire execution when needs_editable is true.
    // This serializes CALL/PROJECT mutations so they don't run concurrently.
    // Non-editable (read-only) queries are NOT affected and still run concurrently.
    std::unique_lock<std::mutex> update_lock(server.update_execution_mutex, std::defer_lock);

    if (needs_editable) {
        // Use editable scope for queries that modify data (e.g., PROJECT)
        update_lock.lock();
        version_scope = buffer_manager.init_version_editable();
    } else {
        // Use readonly scope for pure read queries
        version_scope = buffer_manager.init_version_readonly();
    }

    {
        std::lock_guard<std::mutex> lock(server.thread_info_vec_mutex);
        get_query_ctx().prepare(*version_scope, query_timeout);
    }

    std::unique_ptr<QueryExecutor> physical_plan;
    try {
        auto logical_plan = create_readonly_logical_plan(query);

        // Load projection context if USE GRAPH was specified
        auto& ctx = get_query_ctx();
        if (ctx.is_using_projection()) {
            ctx.load_projection(ctx.active_projection);
        }

        physical_plan = create_readonly_physical_plan(*logical_plan, response_type);
    } catch (const QueryParsingException& e) {
        std::string msg = "Parsing Exception. Line " + std::to_string(e.line)
                        + ", col: " + std::to_string(e.column) + ": " + e.what();

        logger.error() << msg << "\nQuery:\n" << query;
        return send_bad_request(os, msg);
    } catch (const QueryException& e) {
        std::string msg = e.what();
        logger.error() << "Worker " << worker << ": " << msg;
        return send_bad_request(os, msg);
    } catch (const LogicException& e) {
        std::string msg = e.what();
        logger.error() << "Worker " << worker << ": " << msg;
        return send_internal_error(os, msg);
    }

    if (physical_plan == nullptr) {
        return;
    }

    try {
        execute_readonly_query_plan(*physical_plan, os, response_type);

        // If we used an editable scope, mark it as committed so changes persist
        if (needs_editable) {
            version_scope->commited = true;
        }
    } catch (const ConnectionException& e) {
        logger.error() << "Connection(" << worker << ") exception: " << e.what();
    } catch (const InterruptedException& e) {
        // Handled in execute_readonly_query_plan
    } catch (const QueryExecutionException& e) {
        // Handled in execute_readonly_query_plan
    }
}

template<typename stream_t>
std::unique_ptr<Op> HttpGQLSession<stream_t>::create_readonly_logical_plan(const std::string& query)
{
    const auto start_parser = chrono::system_clock::now();

    auto logical_plan = GQL::QueryParser::get_query_plan(query);
    parser_duration = chrono::system_clock::now() - start_parser;
    return logical_plan;
}

template<typename stream_t>
std::unique_ptr<QueryExecutor>
    HttpGQLSession<stream_t>::create_readonly_physical_plan(Op& logical_plan, GQL::ReturnType response_type)
{
    const auto start_optimizer = chrono::system_clock::now();

    GQL::ExecutorConstructor executor_constructor(response_type);
    logical_plan.accept_visitor(executor_constructor);

    optimizer_duration = chrono::system_clock::now() - start_optimizer;
    return std::move(executor_constructor.executor);
}

template<typename stream_t>
void HttpGQLSession<stream_t>::execute_readonly_query_plan(
    QueryExecutor& physical_plan,
    std::ostream& os,
    GQL::ReturnType return_type
)
{
    const auto execution_start = chrono::system_clock::now();

    try {
        send_query_header(os, get_content_type(return_type));

        logger.debug([&physical_plan](std::ostream& os) {
            physical_plan.analyze(os, false);
        });

        const auto result_count = physical_plan.execute(os);
        execution_duration = chrono::system_clock::now() - execution_start;

        logger.debug([&physical_plan](std::ostream& os) {
            physical_plan.analyze(os, true);
        });

        logger.info() << "Worker " << worker << " results: " << result_count << "\n"
                      << "Parser    : " << parser_duration.count() << " ms\n"
                      << "Optimizer : " << optimizer_duration.count() << " ms\n"
                      << "Execution : " << execution_duration.count() << " ms";
    } catch (const InterruptedException& e) {
        execution_duration = chrono::system_clock::now() - execution_start;
        logger.error() << "Worker " << worker << " timed out after " << execution_duration.count() << " ms";
    } catch (const QueryExecutionException& e) {
        execution_duration = chrono::system_clock::now() - execution_start;
        logger.error() << "Worker " << worker << ": " << e.what();
    } catch (const std::exception& e) {
        logger.error() << "Worker " << worker << " Unexpected Exception: " << e.what();
    } catch (...) {
        logger.error() << "Worker " << worker << " Unknown exception";
    }
}

template<typename stream_t>
void HttpGQLSession<stream_t>::execute_update_query(const std::string& /* query */, std::ostream& /* os */)
{ }

template class MDBServer::HttpGQLSession<asio::ip::tcp::socket>;
template class MDBServer::HttpGQLSession<boost::beast::ssl_stream<asio::ip::tcp::socket>>;
