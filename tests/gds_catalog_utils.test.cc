#include "query/executor/binding_iter/procedure/gds_catalog_utils.h"
#include "graph_models/common/conversions.h"
#include "graph_models/gql/conversions.h"
#include <chrono>
#include <iostream>

using namespace std::chrono;

bool test_assign_catalog_field() {
    GQL::GqlGraphCatalog::GraphListEntry entry;
    entry.graphName = "g";
    entry.database = "db";
    entry.databaseLocation = "/tmp/db";
    entry.configuration = "{}";
    entry.nodeCount = 1;
    entry.relationshipCount = 2;
    entry.schema = "s";
    entry.schemaWithOrientation = "so";
    entry.degreeDistribution = "dd";
    entry.density = 0.5;
    entry.creationTime = system_clock::time_point{milliseconds{1000}};
    entry.modificationTime = system_clock::time_point{milliseconds{2000}};
    entry.sizeInBytes = 42;
    entry.memoryUsage = "1k";

    bool error = false;
    auto check = [&](const std::string& field, ObjectId expected) {
        auto got = assign_catalog_field(entry, field);
        if (got != expected) {
            std::cerr << "mismatch for " << field << "\n";
            error = true;
        }
    };

    check("graphName", GQL::Conversions::pack_string_simple(entry.graphName));
    check("database", GQL::Conversions::pack_string_simple(entry.database));
    check("databaseLocation", GQL::Conversions::pack_string_simple(entry.databaseLocation));
    check("configuration", GQL::Conversions::pack_string_simple(entry.configuration));
    check("nodeCount", Common::Conversions::pack_int(static_cast<int64_t>(entry.nodeCount)));
    check("relationshipCount", Common::Conversions::pack_int(static_cast<int64_t>(entry.relationshipCount)));
    check("schema", GQL::Conversions::pack_string_simple(entry.schema));
    check("schemaWithOrientation", GQL::Conversions::pack_string_simple(entry.schemaWithOrientation));
    check("degreeDistribution", GQL::Conversions::pack_string_simple(entry.degreeDistribution));
    check("density", Common::Conversions::pack_double(entry.density));
    check("creationTime", Common::Conversions::pack_int(static_cast<int64_t>(
        duration_cast<milliseconds>(entry.creationTime.time_since_epoch()).count())));
    check("modificationTime", Common::Conversions::pack_int(static_cast<int64_t>(
        duration_cast<milliseconds>(entry.modificationTime.time_since_epoch()).count())));
    check("sizeInBytes", Common::Conversions::pack_int(static_cast<int64_t>(entry.sizeInBytes)));
    check("memoryUsage", GQL::Conversions::pack_string_simple(entry.memoryUsage));
    check("unknown", ObjectId::get_null());

    return error;
}

int main() {
    return test_assign_catalog_field() ? 1 : 0;
}

