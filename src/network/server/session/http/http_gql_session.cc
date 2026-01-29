#include "http_gql_session.h"

#include <iomanip>

#include "misc/logger.h"
#include "misc/trim.h"
#include "network/server/server.h"
#include "network/server/session/http/gql_request_parser.h"
#include "network/server/session/http/request_parser.h"
#include "network/server/session/http/response/http_response_buffer.h"
#include "query/optimizer/property_graph_model/executor_constructor.cc"
#include "query/parser/gql_query_parser.h"

using namespace boost;
using namespace MDBServer;
namespace beast = boost::beast;
namespace http = beast::http;

HttpGQLSession::HttpGQLSession(
    Server& server,
    stream_type&& stream,
    http::request<http::string_body>&& request,
    std::chrono::seconds query_timeout
) :
    server(server),
    stream(std::move(stream)),
    request(std::move(request)),
    query_timeout(query_timeout)
{ }

HttpGQLSession::~HttpGQLSession()
{
    if (stream.socket().is_open()) {
        stream.close();
    }
}

void HttpGQLSession::run(std::unique_ptr<HttpGQLSession> obj)
{
    HttpResponseBuffer response_buffer(obj->stream);

    std::ostream response_ostream(&response_buffer);

    // Without this line ConnectionException won't be caught properly
    response_ostream.exceptions(std::ifstream::failbit | std::ifstream::badbit);

    auto request_type = GQL::RequestParser::get_request_type(obj->request.target());

    if (request_type == Protocol::RequestType::AUTH) {
        auto&& [user, pass] = Common::RequestParser::parse_auth(obj->request);
        auto&& [auth_token, valid_until] = obj->server.create_auth_token(user, pass);
        if (auth_token.empty()) {
            response_ostream << "HTTP/1.1 401 Unauthorized\r\nWWW-Authenticate: Bearer\r\n\r\n";
        } else {
            auto valid_until_t = std::chrono::system_clock::to_time_t(valid_until);
            response_ostream << "HTTP/1.1 200 OK\r\n"
                                "Content-Type: application/json; charset=utf-8\r\n"
                                "{\"token\":\""
                             << auth_token << "\",\"valid_until\":\""
                             << std::put_time(std::localtime(&valid_until_t), "%Y-%m-%d %T") << "\"}";
        }
        return;
    }

    auto auth_token = Common::RequestParser::get_auth(obj->request);

    if (request_type == Protocol::RequestType::CANCEL) {
        auto&& [worker_id, cancel_token] = Common::RequestParser::parse_cancel(obj->request);
        auto cancel_res = obj->server.try_cancel(worker_id, cancel_token);
        if (cancel_res) {
            response_ostream << "HTTP/1.1 200 OK\r\n\r\n";
        } else {
            response_ostream << "HTTP/1.1 400 Bad Request\r\n\r\n";
        }
        return;
    }

    auto&& [query, response_type] = GQL::RequestParser::parse_query(obj->request);

    if (!obj->server.authorize(request_type, auth_token)) {
        response_ostream << "HTTP/1.1 401 Unauthorized\r\nWWW-Authenticate: Bearer\r\n\r\n";
        return;
    }

    // After parsing the query we don't want to have a connection timeout
    obj->stream.expires_never();

    logger(Category::Info) << "\nQuery received:\n" << trim_string(query) << "\n";

    if (request_type == Protocol::RequestType::UPDATE) {
        obj->execute_update_query(query, response_ostream);
    } else /* (request_type == Protocol::RequestType::QUERY) */ {
        obj->execute_readonly_query(query, response_ostream, response_type);
    }
}

void HttpGQLSession::execute_readonly_query(
    const std::string& query,
    std::ostream& os,
    GQL::ReturnType response_type
)
{
    // Check if query contains PROJECT function - if so, we need an editable scope
    // to persist the projection data to disk
    bool needs_editable = (query.find("PROJECT") != std::string::npos
                        || query.find("project") != std::string::npos);

    // Declared here because the destruction need to be after calling execute_query_plan
    std::unique_ptr<BufferManager::VersionScope> version_scope;

    if (needs_editable) {
        // Use editable scope for queries that modify data (e.g., PROJECT)
        std::lock_guard<std::mutex> lock(server.update_execution_mutex);
        version_scope = buffer_manager.init_version_editable();
    } else {
        // Use readonly scope for pure read queries
        version_scope = buffer_manager.init_version_readonly();
    }

    {
        std::lock_guard<std::mutex> lock(server.thread_info_vec_mutex);
        get_query_ctx().prepare(*version_scope, query_timeout);
    }
    logger(Category::Info) << "Cancellation: " << get_query_ctx().thread_info.worker_index << ' '
                           << get_query_ctx().cancellation_token;

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
        std::string msg = "Query Parsing Exception. Line " + std::to_string(e.line)
                        + ", col: " + std::to_string(e.column) + ": " + e.what();
        logger(Category::Error) << msg;

        os << "HTTP/1.1 400 Bad Request\r\n"
              "Content-Type: text/plain\r\n"
              "\r\n"
           << std::string(msg);
        return;
    } catch (const QueryException& e) {
        logger(Category::Error) << "Query Exception: " << e.what();

        os << "HTTP/1.1 400 Bad Request\r\n"
              "Content-Type: text/plain\r\n"
              "\r\n"
           << std::string(e.what());
    } catch (const LogicException& e) {
        logger(Category::Error) << "Logic Exception: " << e.what();

        os << "HTTP/1.1 500 Internal Server Error\r\n"
              "Content-Type: text/plain\r\n"
              "\r\n"
           << std::string(e.what());
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
        logger(Category::Error) << "Connection Exception: " << e.what();
    } catch (const InterruptedException& e) {
        // Handled in execute_readonly_query_plan
    } catch (const QueryExecutionException& e) {
        // Handled in execute_readonly_query_plan
    }
}

std::unique_ptr<Op> HttpGQLSession::create_readonly_logical_plan(const std::string& query)
{
    const auto start_parser = std::chrono::system_clock::now();

    auto logical_plan = GQL::QueryParser::get_query_plan(query);
    parser_duration = std::chrono::system_clock::now() - start_parser;
    return logical_plan;
}

std::unique_ptr<QueryExecutor>
    HttpGQLSession::create_readonly_physical_plan(Op& logical_plan, GQL::ReturnType response_type)
{
    const auto start_optimizer = std::chrono::system_clock::now();

    GQL::ExecutorConstructor executor_constructor(response_type);
    logical_plan.accept_visitor(executor_constructor);

    optimizer_duration = std::chrono::system_clock::now() - start_optimizer;
    return std::move(executor_constructor.executor);
}

void HttpGQLSession::execute_readonly_query_plan(
    QueryExecutor& physical_plan,
    std::ostream& os,
    GQL::ReturnType return_type
)
{
    const auto execution_start = std::chrono::system_clock::now();

    try {
        os << "HTTP/1.1 200 OK\r\n"
              "Server: MillenniumDB\r\n";

        switch (return_type) {
        case GQL::ReturnType::CSV:
            os << "Content-Type: text/csv; charset=utf-8\r\n";
            break;
        case GQL::ReturnType::TSV:
            os << "Content-Type: text/tab-separated-values; charset=utf-8\r\n";
            break;
        default:
            throw LogicException(
                "Response type not implemented: " + std::to_string(static_cast<int>(return_type))
            );
        }
        os << "Access-Control-Allow-Origin: *\r\n"
              "Access-Control-Allow-Headers: Origin, X-Requested-With, Content-Type, Accept, "
              "Authorization\r\n"
              "Access-Control-Allow-Methods: GET, POST\r\n"
              "\r\n";

        logger.log(Category::PhysicalPlan, [&physical_plan](std::ostream& os) {
            physical_plan.analyze(os, false);
            os << '\n';
        });

        const auto result_count = physical_plan.execute(os);
        execution_duration = std::chrono::system_clock::now() - execution_start;

        logger.log(Category::ExecutionStats, [&physical_plan](std::ostream& os) {
            physical_plan.analyze(os, true);
            os << '\n';
        });

        logger(Category::Info) << "Results: " << result_count << '\n'
                               << "Parser duration:    " << parser_duration.count() << " ms\n"
                               << "Optimizer duration: " << optimizer_duration.count() << " ms\n"
                               << "Execution duration: " << execution_duration.count() << " ms";
    } catch (const InterruptedException& e) {
        execution_duration = std::chrono::system_clock::now() - execution_start;
        logger(Category::Info
        ) << "Timeout thrown after "
          << std::chrono::duration_cast<std::chrono::milliseconds>(execution_duration).count() << " ms";
        // Write timeout error to response body
        os << "\n# Error: Query timeout after "
           << std::chrono::duration_cast<std::chrono::milliseconds>(execution_duration).count() << " ms\n";
    } catch (const QueryExecutionException& e) {
        execution_duration = std::chrono::system_clock::now() - execution_start;
        logger(Category::Error) << e.what();
        // Write error to response body
        os << "\n# Error: " << e.what() << "\n";
    } catch (const std::exception& e) {
        logger(Category::Error) << "Unexpected Exception: " << e.what();
        // Write error to response body so client sees the issue
        os << "\n# Error: " << e.what() << "\n";
    } catch (...) {
        logger(Category::Error) << "Unknown exception";
        os << "\n# Error: Unknown exception during query execution\n";
    }
}

void HttpGQLSession::execute_update_query(const std::string& /* query */, std::ostream& /* os */) { }
