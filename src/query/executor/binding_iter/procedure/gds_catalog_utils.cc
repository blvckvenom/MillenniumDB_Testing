//===----------------------------------------------------------------------===//
//
// This file is part of MillenniumDB
//
// MillenniumDB is free software: you can redistribute it and/or modify
// it under the terms of the GNU Affero General Public License as published
// by the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.
//
// MillenniumDB is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU Affero General Public License for more details.
//
// You should have received a copy of the GNU Affero General Public License
// along with MillenniumDB.  If not, see <https://www.gnu.org/licenses/>.
//
//===----------------------------------------------------------------------===//

#include "gds_catalog_utils.h"

#include <chrono>

ObjectId assign_catalog_field(
    const GQL::GqlGraphCatalog::GraphListEntry& entry,
    const std::string& name
) {
    if (name == "graphName") {
        return GQL::Conversions::pack_string_simple(entry.graphName);
    } else if (name == "database") {
        return GQL::Conversions::pack_string_simple(entry.database);
    } else if (name == "databaseLocation") {
        return GQL::Conversions::pack_string_simple(entry.databaseLocation);
    } else if (name == "configuration") {
        return GQL::Conversions::pack_string_simple(entry.configuration);
    } else if (name == "schema") {
        return GQL::Conversions::pack_string_simple(entry.schema);
    } else if (name == "schemaWithOrientation") {
        return GQL::Conversions::pack_string_simple(entry.schemaWithOrientation);
    } else if (name == "degreeDistribution") {
        return GQL::Conversions::pack_string_simple(entry.degreeDistribution);
    } else if (name == "memoryUsage") {
        return GQL::Conversions::pack_string_simple(entry.memoryUsage);
    } else if (name == "nodeCount") {
        return Common::Conversions::pack_int(static_cast<int64_t>(entry.nodeCount));
    } else if (name == "relationshipCount") {
        return Common::Conversions::pack_int(static_cast<int64_t>(entry.relationshipCount));
    } else if (name == "density") {
        return Common::Conversions::pack_double(entry.density);
    } else if (name == "creationTime") {
        auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                      entry.creationTime.time_since_epoch()).count();
        return Common::Conversions::pack_int(static_cast<int64_t>(ms));
    } else if (name == "modificationTime") {
        auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                      entry.modificationTime.time_since_epoch()).count();
        return Common::Conversions::pack_int(static_cast<int64_t>(ms));
    } else if (name == "sizeInBytes") {
        return Common::Conversions::pack_int(static_cast<int64_t>(entry.sizeInBytes));
    }
    return ObjectId::get_null();
}

