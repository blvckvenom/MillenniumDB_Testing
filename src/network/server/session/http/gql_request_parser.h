#pragma once

#include <boost/beast.hpp>

#include "network/server/protocol.h"
#include "network/sparql/response_type.h"
#include "query/executor/query_executor/gql/return_executor.h"

namespace GQL { namespace RequestParser {

inline MDBServer::Protocol::RequestType get_request_type(const std::string_view& /* str */)
{
    // if (str.find("/update") != std::string::npos) {
    //     return MDBServer::Protocol::RequestType::UPDATE;
    // }
    // if (str.find("/cancel") != std::string::npos) {
    //     return MDBServer::Protocol::RequestType::CANCEL;
    // }
    // if (str.find("/auth") != std::string::npos) {
    //     return MDBServer::Protocol::RequestType::AUTH;
    // }
    return MDBServer::Protocol::RequestType::QUERY;
}

inline std::pair<std::string, ReturnType>
    parse_query(boost::beast::http::request<boost::beast::http::string_body>& req)
{
    ReturnType response_type = ReturnType::CSV;

    for (auto& header : req) {
        std::string accept_value = header.value();
        if (header.name_string() == "Accept") {
            if (accept_value.find("text/tab-separated-values") != std::string::npos) {
                response_type = ReturnType::TSV;
            }
        }
        // else CSV by default
    }

    // If the query was sent using GET, parse it from the URL
    if (req.method() == boost::beast::http::verb::get) {
        std::string target = std::string(req.target());
        auto qpos = target.find('?');
        if (qpos != std::string::npos && qpos + 1 < target.size()) {
            std::string query_part = target.substr(qpos + 1);
            if (query_part.rfind("query=", 0) == 0) {
                query_part = query_part.substr(6);
            } else if (query_part.rfind("q=", 0) == 0) {
                query_part = query_part.substr(2);
            }
            // Decode percent-encoding
            std::string decoded;
            for (size_t i = 0; i < query_part.size(); ++i) {
                if (query_part[i] == '%' && i + 2 < query_part.size()) {
                    std::string hex = query_part.substr(i + 1, 2);
                    decoded.push_back(static_cast<char>(std::stoi(hex, nullptr, 16)));
                    i += 2;
                } else if (query_part[i] == '+') {
                    decoded.push_back(' ');
                } else {
                    decoded.push_back(query_part[i]);
                }
            }
            return std::make_pair(decoded, response_type);
        }
        return std::make_pair("", response_type);
    }

    // Otherwise we assume that the POST body contains the query
    if (req.method() != boost::beast::http::verb::post) {
        return std::make_pair("", response_type);
    }

    return std::make_pair(req.body(), response_type);
}
}} // namespace GQL::RequestParser
