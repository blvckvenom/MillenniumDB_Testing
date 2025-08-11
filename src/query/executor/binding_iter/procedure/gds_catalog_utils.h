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

#pragma once

#include <string>

#include "graph_models/gql/gql_graph_catalog.h"
#include "graph_models/common/conversions.h"
#include "graph_models/gql/conversions.h"

/**
 * Helper function to map a catalog field name to an ObjectId value.
 * Unknown field names return ObjectId::get_null().
 */
ObjectId assign_catalog_field(
    const GQL::GqlGraphCatalog::GraphListEntry& entry,
    const std::string& name
);

