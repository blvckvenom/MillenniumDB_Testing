#pragma once

#include <string>
#include <chrono>

#include "graph_models/gql/gql_graph_catalog.h"
#include "graph_models/common/conversions.h"
#include "graph_models/gql/conversions.h"

inline ObjectId assign_catalog_field(
    const GQL::GqlGraphCatalog::GraphListEntry& entry,
    const std::string& field_name) {
    using namespace std::chrono;
    if (field_name == "graphName") {
        return GQL::Conversions::pack_string_simple(entry.graphName);
    } else if (field_name == "database") {
        return GQL::Conversions::pack_string_simple(entry.database);
    } else if (field_name == "databaseLocation") {
        return GQL::Conversions::pack_string_simple(entry.databaseLocation);
    } else if (field_name == "configuration") {
        return GQL::Conversions::pack_string_simple(entry.configuration);
    } else if (field_name == "schema") {
        return GQL::Conversions::pack_string_simple(entry.schema);
    } else if (field_name == "schemaWithOrientation") {
        return GQL::Conversions::pack_string_simple(entry.schemaWithOrientation);
    } else if (field_name == "degreeDistribution") {
        return GQL::Conversions::pack_string_simple(entry.degreeDistribution);
    } else if (field_name == "memoryUsage") {
        return GQL::Conversions::pack_string_simple(entry.memoryUsage);
    } else if (field_name == "nodeCount") {
        return Common::Conversions::pack_int(static_cast<int64_t>(entry.nodeCount));
    } else if (field_name == "relationshipCount") {
        return Common::Conversions::pack_int(static_cast<int64_t>(entry.relationshipCount));
    } else if (field_name == "density") {
        return Common::Conversions::pack_float(static_cast<float>(entry.density));
    } else if (field_name == "creationTime") {
        return Common::Conversions::pack_int(static_cast<int64_t>(
            duration_cast<milliseconds>(entry.creationTime.time_since_epoch()).count()));
    } else if (field_name == "modificationTime") {
        return Common::Conversions::pack_int(static_cast<int64_t>(
            duration_cast<milliseconds>(entry.modificationTime.time_since_epoch()).count()));
    } else if (field_name == "sizeInBytes") {
        return Common::Conversions::pack_int(static_cast<int64_t>(entry.sizeInBytes));
    }
    return ObjectId::get_null();
}

