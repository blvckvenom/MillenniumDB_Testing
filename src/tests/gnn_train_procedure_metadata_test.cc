#include <algorithm>
#include <iostream>
#include <string>
#include <vector>

#include "query/procedure/builtin/gnn_train_procedure.h"

// Unit tests for GnnTrainProcedure metadata.
//
// The GQL parser expands YIELD * from yield_fields() and binds NULL for any
// explicit YIELD name absent from the result row, so yield_fields() must
// declare every field execute() actually yields — a name missing from the
// declaration is silently dropped from YIELD * results.

int main() {
    std::cout << "Testing GnnTrainProcedure metadata..." << std::endl;
    int test_count = 0;
    int passed = 0;

    GQL::Procedures::GnnTrainProcedure proc;
    const auto fields = proc.yield_fields();

    auto declared = [&fields](const std::string& name) {
        return std::any_of(
            fields.begin(), fields.end(),
            [&name](const GQL::Procedure::YieldField& f) { return f.name == name; });
    };

    // Every name passed to ctx.yield() in GnnTrainProcedure::execute()
    // (Step 13), in emission order.
    const std::vector<std::string> yielded = {
        "modelName",
        "ranEpochs",
        "didConverge",
        "bestValAccuracy",
        "testAccuracy",
        "trainSeconds",
        "assembleSeconds",
        "forwardSeconds",
        "backwardSeconds",
        "l1HitRatio",
        "l2HitRatio",
        "l3Reads",
        "l4Reads",
        "l3BytesDisk",
        "l4BytesDisk",
        "totalBytesDisk",
        "l3ReadAmplification",
        "nodesWritten",
        "nodesInferred",
        "inferenceMillis",
        "writeMillis",
        "bestCheckpointPath",
        "finalCheckpointPath",
        "resumedFromEpoch",
        "effectivePrefetchWorkers",
        "useAddrTablesEffective",
        "addrTableLoadUs",
        "testAccuracyAtBestVal",
        "bestValEpoch",
    };

    // Test 1: every yielded field is declared in yield_fields().
    test_count++;
    {
        std::cout << "Test 1: all yielded fields declared in yield_fields()...";
        std::vector<std::string> missing;
        for (const auto& name : yielded) {
            if (!declared(name)) {
                missing.push_back(name);
            }
        }
        if (missing.empty()) {
            std::cout << " PASS" << std::endl;
            passed++;
        } else {
            std::cout << " FAIL (missing:";
            for (const auto& name : missing) {
                std::cout << " " << name;
            }
            std::cout << ")" << std::endl;
        }
    }

    // Test 2: no declared field is dead metadata (declared but never yielded).
    test_count++;
    {
        std::cout << "Test 2: no declared-but-never-yielded fields...";
        std::vector<std::string> dead;
        for (const auto& f : fields) {
            if (std::find(yielded.begin(), yielded.end(), f.name) == yielded.end()) {
                dead.push_back(f.name);
            }
        }
        if (dead.empty()) {
            std::cout << " PASS" << std::endl;
            passed++;
        } else {
            std::cout << " FAIL (dead:";
            for (const auto& name : dead) {
                std::cout << " " << name;
            }
            std::cout << ")" << std::endl;
        }
    }

    // Test 3: declared field names are unique.
    test_count++;
    {
        std::cout << "Test 3: yield_fields() names are unique...";
        std::vector<std::string> names;
        names.reserve(fields.size());
        for (const auto& f : fields) {
            names.push_back(f.name);
        }
        std::sort(names.begin(), names.end());
        if (std::adjacent_find(names.begin(), names.end()) == names.end()) {
            std::cout << " PASS" << std::endl;
            passed++;
        } else {
            std::cout << " FAIL (duplicate name found)" << std::endl;
        }
    }

    std::cout << passed << "/" << test_count << " tests passed" << std::endl;
    return (passed == test_count) ? 0 : 1;
}
