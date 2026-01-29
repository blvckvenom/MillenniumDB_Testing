#include "gnn_sample_drop_procedure.h"

#include <filesystem>
#include <stdexcept>

#include "gnn/sampling/sample_storage.h"
#include "system/file_manager.h"

using namespace GQL;
using namespace GQL::Procedures;
using namespace mdb::gnn;

void GnnSampleDropProcedure::execute(ProcedureContext& ctx) {
    // Validate argument count
    if (ctx.arguments.size() < 1) {
        throw std::runtime_error(
            "gnn.sample_drop() requires 1 argument: sampleName\n"
            "Usage: CALL gnn.sample_drop('mySamples') YIELD success, message"
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
            "Example: CALL gnn.sample_drop('old_samples')"
        );
    }

    if (sample_name.empty()) {
        throw std::runtime_error(
            "Invalid sample name: name cannot be empty.\n"
            "Provide a non-empty string as the argument.\n"
            "Example: CALL gnn.sample_drop('old_samples')"
        );
    }

    // Get database folder
    std::string db_folder = file_manager.get_file_path("");
    if (!db_folder.empty() && db_folder.back() == '/') {
        db_folder.pop_back();
    }

    // Check if sample exists
    if (!SampleStorage::exists(db_folder, sample_name)) {
        ctx.yield("success", ctx.create_bool(false));
        ctx.yield("message", ctx.create_string(
            "Sample set '" + sample_name + "' does not exist"
        ));
        ctx.yield_row();
        return;
    }

    // Get storage path
    std::filesystem::path storage_path = SampleStorage::get_storage_path(db_folder, sample_name);

    // Attempt to delete the directory
    try {
        std::error_code ec;
        std::uintmax_t removed_count = std::filesystem::remove_all(storage_path, ec);

        if (ec) {
            ctx.yield("success", ctx.create_bool(false));
            ctx.yield("message", ctx.create_string(
                "Failed to delete sample set '" + sample_name + "': " + ec.message()
            ));
            ctx.yield_row();
            return;
        }

        if (removed_count == 0) {
            ctx.yield("success", ctx.create_bool(false));
            ctx.yield("message", ctx.create_string(
                "Sample set '" + sample_name + "' was not found or already deleted"
            ));
            ctx.yield_row();
            return;
        }

        ctx.yield("success", ctx.create_bool(true));
        ctx.yield("message", ctx.create_string(
            "Sample set '" + sample_name + "' has been deleted (" +
            std::to_string(removed_count) + " files removed)"
        ));
        ctx.yield_row();

    } catch (const std::exception& e) {
        ctx.yield("success", ctx.create_bool(false));
        ctx.yield("message", ctx.create_string(
            "Failed to delete sample set '" + sample_name + "': " + e.what()
        ));
        ctx.yield_row();
    }
}
