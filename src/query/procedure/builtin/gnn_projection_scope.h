#pragma once

#include <string>

#include "query/query_context.h"

namespace GQL {
namespace Procedures {

/// Routes this thread's QueryContext at one graph for the duration of a scope
/// and puts it back afterwards.
///
/// Property scans resolve their B+Tree through GQLModel::get_key_value_node()
/// and friends, which ask the context whether a projection is loaded. The
/// resulting iterator holds that tree by reference, so the routing has to
/// outlive the whole scan. Restoring on the way out matters because a procedure
/// runs inside a query that may itself be operating on a projection.
class ProjectionScope {
public:
    explicit ProjectionScope(const std::string& projection) :
        previous_ { get_query_ctx().active_projection }
    {
        if (projection == previous_) {
            return;
        }
        if (projection.empty()) {
            get_query_ctx().clear_active_projection();
        } else {
            get_query_ctx().load_projection(projection);
        }
        switched_ = true;
    }

    ~ProjectionScope()
    {
        if (!switched_) {
            return;
        }
        try {
            if (previous_.empty()) {
                get_query_ctx().clear_active_projection();
            } else {
                get_query_ctx().load_projection(previous_);
            }
        } catch (...) {
            // A destructor must not throw. Leaving the context pointing at the
            // wrong graph is bad, but tearing down the server is worse.
        }
    }

    ProjectionScope(const ProjectionScope&) = delete;
    ProjectionScope& operator=(const ProjectionScope&) = delete;

private:
    std::string previous_;
    bool switched_ { false };
};

} // namespace Procedures
} // namespace GQL
