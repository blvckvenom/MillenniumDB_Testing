#pragma once

#include <cassert>
#include <chrono>
#include <ostream>
#include <random>
#include <set>
#include <string>
#include <unordered_map>
#include <vector>

#include <boost/uuid/random_generator.hpp>
#include <boost/uuid/uuid_io.hpp>

#include "graph_models/object_id.h"
#include "query/id.h"
#include "query/var_id.h"
#include "system/buffer_manager.h"
#include "system/string_manager.h"
#include "system/tmp_manager.h"

// Forward declaration to avoid circular dependencies
namespace GQL {
    class ProjectionQueryContext;
}

struct ThreadInfo {
    bool interruption_requested = false;

    uint_fast32_t worker_index = 0;

    std::chrono::system_clock::time_point timeout;
    std::chrono::system_clock::time_point time_start;
};


class QueryContext {
struct VarContext {
    uint64_t internal_var_counter = 0;

    std::vector<std::string> var_names;

    // maps var_name -> pos
    std::unordered_map<std::string, uint64_t> var_map;
};

public:
    ThreadInfo thread_info;

    uint64_t start_version = 0;
    uint64_t result_version = 0;

    std::string cancellation_token;

    // Used only by BindingExprBNode of the RDF model.
    std::unordered_map<std::string, uint64_t> blank_node_ids;

    // Active projection name for GQL USE GRAPH clause
    std::string active_projection;

    // Projection query context (loaded when active_projection is set)
    std::unique_ptr<GQL::ProjectionQueryContext> projection_ctx;

    // Debug prints ObjectIds
    static inline std::ostream& (*_debug_print)(std::ostream&, ObjectId);

    static constexpr uint64_t buffer_size = 64 * 1024;

private:
    VarContext var_ctx;

    // Used only by BindingExprBNode of the RDF model.
    uint64_t blank_node_count = 0;

    std::default_random_engine rand_generator = std::default_random_engine(
        std::chrono::system_clock::now().time_since_epoch().count()
    );

    // Random number generator
    std::uniform_real_distribution<double> distribution = std::uniform_real_distribution<double>(0.0, 1.0);

    static inline boost::uuids::random_generator uuid_generator;

    // buffers for general use
    char* buffer1;
    char* buffer2;

    std::map<VarId, ObjectId> edge_directions;

public:
    // Constructors and destructor defined in .cc file to handle unique_ptr with incomplete type
    QueryContext();
    QueryContext(QueryContext&& other);
    QueryContext(const QueryContext& other) = delete;
    ~QueryContext();

    // Cleans up everything. Must be called before parsing the query
    // Defined in .cc file to handle unique_ptr operations
    void prepare(BufferManager::VersionScope& version_scope, std::chrono::seconds timeout);

    // Projection management methods (for GQL USE GRAPH support)
    bool is_using_projection() const {
        return !active_projection.empty();
    }

    void set_active_projection(const std::string& projection_name) {
        active_projection = projection_name;
    }

    // Methods that manipulate projection_ctx defined in .cc file
    void clear_active_projection();
    void load_projection(const std::string& proj_name);
    void unload_projection();

    std::string get_uuid() {
        boost::uuids::uuid uuid = uuid_generator();
        return boost::uuids::to_string(uuid);
    }

    // returns a random double between 0 (inclusive) and 1 (exclusive) with uniform distribution
    double get_rand() {
        return distribution(rand_generator);
    }

    void define_alias(VarId var_id, const std::string& alias) {
        var_ctx.var_map.insert({ alias, var_id.id });
    }

    VarId get_or_create_var(const std::string& var_name) {
        auto found = var_ctx.var_map.find(var_name);
        if (found != var_ctx.var_map.end()) {
            return VarId(found->second);
        } else {
            uint64_t new_id = var_ctx.var_names.size();
            var_ctx.var_names.push_back(var_name);
            var_ctx.var_map.insert({ var_name, new_id });
            return VarId(new_id);
        }
    }

    VarId get_var(const std::string& var_name, bool* found) {
        auto it = var_ctx.var_map.find(var_name);
        if (it != var_ctx.var_map.end()) {
            *found = true;
            return VarId(it->second);
        } else {
           *found = false;
           return VarId(0);
        }
    }

    const std::string& get_var_name(VarId var_id) {
        assert(var_ctx.var_names.size() > var_id.id);
        return var_ctx.var_names[var_id.id];
    }

    uint64_t get_var_size() const {
        return var_ctx.var_names.size();
    }

    // generates a new internal var.
    VarId get_internal_var() {
        uint64_t new_id = var_ctx.var_names.size();
        var_ctx.var_names.push_back("." + std::to_string(var_ctx.internal_var_counter++));
        return VarId(new_id);
    }

    std::set<VarId> get_all_vars() {
        std::set<VarId> res;
        for (unsigned i = 0; i < var_ctx.var_names.size(); i++) {
            res.emplace(i);
        }
        return res;
    }

    std::set<VarId> get_non_internal_vars() {
        std::set<VarId> res;
        for (unsigned i = 0; i < var_ctx.var_names.size(); i++) {
            if (var_ctx.var_names[i][0] != '.') {
                res.emplace(i);
            }
        }
        return res;
    }

    // only for RDF model
    uint64_t get_new_blank_node() {
        return blank_node_count++;
    }

    void debug_print(std::ostream& os, ObjectId oid) {
        _debug_print(os, oid);
    }

    void debug_print(std::ostream& os, Id id) {
        if (id.is_OID()) {
            _debug_print(os, id.get_OID());
        } else {
            os << get_var_name(id.get_var());
        }
    }

    bool is_internal(VarId var) {
        return var_ctx.var_names[var.id][0] == '.';
    }

    bool is_blank(VarId var) {
        return var_ctx.var_names[var.id].find("_:") == 0;
    }

    static inline thread_local QueryContext* _query_ctx;

    static inline void set_query_ctx(QueryContext* ctx) {
        _query_ctx = ctx;
    }

    char* get_buffer1() {
        return buffer1;
    }

    char* get_buffer2() {
        return buffer2;
    }
};

inline QueryContext& get_query_ctx() {
    return *QueryContext::_query_ctx;
}
