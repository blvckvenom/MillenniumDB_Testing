#include "gnn_sample_info_procedure.h"

#include <chrono>
#include <filesystem>
#include <sstream>
#include <stdexcept>

#include "gnn/sampling/sample_catalog.h"
#include "gnn/sampling/sample_storage.h"
#include "gnn_procedure_utils.h"

using namespace GQL;
using namespace GQL::Procedures;
using namespace mdb::gnn;

void GnnSampleInfoProcedure::execute(ProcedureContext& ctx) {
    // Validate argument count
    if (ctx.arguments.size() != 1) {
        throw std::runtime_error(
            "gnn.sample_info() requires exactly 1 argument: sampleName\n"
            "Usage: CALL gnn.sample_info('mySamples') YIELD ..."
        );
    }

    // Get sample name
    std::string sample_name;
    try {
        sample_name = ctx.get_string_argument(0);
    } catch (const std::exception& e) {
        throw std::runtime_error(
            "Invalid sampleName parameter: " + std::string(e.what()) + "\n\n"
            "The parameter must be a STRING containing the sample set name.\n"
            "Example: CALL gnn.sample_info('training_v1')"
        );
    }

    if (sample_name.empty()) {
        throw std::runtime_error(
            "Invalid sample name: name cannot be empty.\n"
            "Provide a non-empty string as the argument.\n"
            "Example: CALL gnn.sample_info('training_v1')"
        );
    }

    validate_safe_name(sample_name, "sampleName");

    // Get database folder
    std::string db_folder = get_db_folder();

    // Check if sample exists
    if (!SampleStorage::exists(db_folder, sample_name)) {
        // Provide helpful error with available samples
        std::filesystem::path samples_root = std::filesystem::path(db_folder) / "samples";
        std::string available;

        if (std::filesystem::exists(samples_root) && std::filesystem::is_directory(samples_root)) {
            std::vector<std::string> sample_names;
            for (const auto& entry : std::filesystem::directory_iterator(samples_root)) {
                if (entry.is_directory() && SampleCatalog::exists(entry.path())) {
                    sample_names.push_back(entry.path().filename().string());
                }
            }

            if (sample_names.empty()) {
                available = "No sample sets exist. Create one first with:\n"
                            "  CALL gnn.offline_sample('projection', 'name', [15, 10])";
            } else {
                available = "Available sample sets: [";
                for (size_t i = 0; i < sample_names.size(); i++) {
                    if (i > 0) available += ", ";
                    available += "'" + sample_names[i] + "'";
                }
                available += "]";
            }
        } else {
            available = "No sample sets exist. Create one first with:\n"
                        "  CALL gnn.offline_sample('projection', 'name', [15, 10])";
        }

        throw std::runtime_error(
            "Sample set '" + sample_name + "' not found.\n\n" + available
        );
    }

    // Load catalog
    std::filesystem::path storage_path = SampleStorage::get_storage_path(db_folder, sample_name);
    SampleCatalog catalog = SampleCatalog::load(storage_path);

    // Convert timestamp to Unix epoch
    auto epoch = std::chrono::duration_cast<std::chrono::seconds>(
        catalog.created_at.time_since_epoch()
    ).count();

    // Format fanouts as comma-separated string
    std::ostringstream fanouts_ss;
    for (size_t i = 0; i < catalog.fanouts.size(); i++) {
        if (i > 0) fanouts_ss << ",";
        fanouts_ss << catalog.fanouts[i];
    }

    // Yield results
    ctx.yield("sampleName", ctx.create_string(catalog.sample_name));
    ctx.yield("projectionName", ctx.create_string(catalog.projection_name));
    ctx.yield("totalBatches", ctx.create_int(static_cast<int64_t>(catalog.total_batches)));
    ctx.yield("trainBatches", ctx.create_int(static_cast<int64_t>(catalog.train_batches)));
    ctx.yield("validationBatches", ctx.create_int(static_cast<int64_t>(catalog.validation_batches)));
    ctx.yield("testBatches", ctx.create_int(static_cast<int64_t>(catalog.test_batches)));
    ctx.yield("uniqueNodes", ctx.create_int(static_cast<int64_t>(catalog.unique_nodes)));
    ctx.yield("totalEdges", ctx.create_int(static_cast<int64_t>(catalog.total_edges)));
    ctx.yield("batchSize", ctx.create_int(static_cast<int64_t>(catalog.batch_size)));
    ctx.yield("numLayers", ctx.create_int(static_cast<int64_t>(catalog.num_layers())));
    ctx.yield("fanouts", ctx.create_string(fanouts_ss.str()));
    ctx.yield("randomSeed", ctx.create_int(static_cast<int64_t>(catalog.random_seed)));
    ctx.yield("createdAt", ctx.create_int(static_cast<int64_t>(epoch)));
    ctx.yield("storagePath", ctx.create_string(storage_path.string()));
    ctx.yield_row();
}
