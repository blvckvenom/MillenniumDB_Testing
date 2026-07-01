#include "gnn_sample_list_procedure.h"

#include <chrono>
#include <filesystem>
#include <stdexcept>

#include "gnn/sampling/sample_catalog.h"
#include "gnn_procedure_utils.h"

using namespace GQL;
using namespace GQL::Procedures;
using namespace mdb::gnn;

void GnnSampleListProcedure::execute(ProcedureContext& ctx) {
    // Get database folder
    std::string db_folder = get_db_folder();

    // Samples are stored in <db_folder>/samples/
    std::filesystem::path samples_root = std::filesystem::path(db_folder) / "samples";

    // If samples directory doesn't exist, return empty result
    if (!std::filesystem::exists(samples_root)) {
        return;  // No samples - zero rows is valid
    }

    if (!std::filesystem::is_directory(samples_root)) {
        return;  // samples is not a directory - unusual but handle gracefully
    }

    // Iterate through sample directories
    for (const auto& entry : std::filesystem::directory_iterator(samples_root)) {
        if (!entry.is_directory()) {
            continue;  // Skip non-directories
        }

        std::filesystem::path sample_dir = entry.path();

        // Check if this is a valid sample directory (has catalog.dat)
        if (!SampleCatalog::exists(sample_dir)) {
            continue;  // Not a valid sample directory
        }

        // Load catalog
        try {
            SampleCatalog catalog = SampleCatalog::load(sample_dir);

            // Convert timestamp to Unix epoch
            auto epoch = std::chrono::duration_cast<std::chrono::seconds>(
                catalog.created_at.time_since_epoch()
            ).count();

            // Yield row
            ctx.yield("sampleName", ctx.create_string(catalog.sample_name));
            ctx.yield("projectionName", ctx.create_string(catalog.projection_name));
            ctx.yield("totalBatches", ctx.create_int(static_cast<int64_t>(catalog.total_batches)));
            ctx.yield("trainBatches", ctx.create_int(static_cast<int64_t>(catalog.train_batches)));
            ctx.yield("validationBatches", ctx.create_int(static_cast<int64_t>(catalog.validation_batches)));
            ctx.yield("testBatches", ctx.create_int(static_cast<int64_t>(catalog.test_batches)));
            ctx.yield("uniqueNodes", ctx.create_int(static_cast<int64_t>(catalog.unique_nodes)));
            ctx.yield("createdAt", ctx.create_int(static_cast<int64_t>(epoch)));
            ctx.yield_row();

        } catch (const std::exception& e) {
            // Skip corrupted/invalid catalogs - log but don't fail
            // Could add logging here if needed
            continue;
        }
    }
}
