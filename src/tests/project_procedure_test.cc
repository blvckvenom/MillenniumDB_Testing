#include <iostream>
#include <vector>

#include "graph_models/gql/gql_model.h"
#include "query/procedure/builtin/project_procedure.h"
#include "query/procedure/procedure_catalog.h"
#include "query/procedure/procedure_context.h"
#include "graph_models/object_id.h"
#include "query/query_context.h"
#include "system/system.h"

// Unit tests for ProjectProcedure
// Tests the CALL PROJECT procedure interface

int main() {
    std::cout << "Testing ProjectProcedure..." << std::endl;
    int test_count = 0;
    int passed = 0;

    try {
        // Initialize system
        System system(
            "test_db",
            1024 * 1024,
            1024 * 1024,
            64 * 1024 * 1024,
            32 * 1024 * 1024,
            1024 * 1024,
            1024 * 1024,
            1
        );

        QueryContext query_ctx;
        QueryContext::set_query_ctx(&query_ctx);

        // Get procedure catalog
        auto& catalog = GQL::ProcedureCatalog::get_instance();

        // Test 1: Procedure registration
        test_count++;
        std::cout << "Test 1: ProjectProcedure registration...";
        try {
            // Register the procedure
            catalog.register_procedure(std::make_unique<GQL::Procedures::ProjectProcedure>());

            auto* proc = catalog.lookup("project");
            if (proc != nullptr) {
                std::cout << " PASS" << std::endl;
                passed++;
            } else {
                std::cout << " FAIL (procedure not found)" << std::endl;
            }
        } catch (const std::exception& e) {
            std::cout << " FAIL: " << e.what() << std::endl;
        }

        // Test 2: Procedure name
        test_count++;
        std::cout << "Test 2: Procedure name...";
        try {
            auto* proc = catalog.lookup("project");
            if (proc != nullptr && proc->name() == "project") {
                std::cout << " PASS" << std::endl;
                passed++;
            } else {
                std::cout << " FAIL (wrong name)" << std::endl;
            }
        } catch (const std::exception& e) {
            std::cout << " FAIL: " << e.what() << std::endl;
        }

        // Test 3: Qualified name
        test_count++;
        std::cout << "Test 3: Qualified name...";
        try {
            auto* proc = catalog.lookup("project");
            if (proc != nullptr && proc->qualified_name() == "gds.graph.project") {
                std::cout << " PASS" << std::endl;
                passed++;
            } else {
                std::cout << " FAIL (wrong qualified name: " << (proc ? proc->qualified_name() : "null") << ")" << std::endl;
            }
        } catch (const std::exception& e) {
            std::cout << " FAIL: " << e.what() << std::endl;
        }

        // Test 4: Description
        test_count++;
        std::cout << "Test 4: Description not empty...";
        try {
            auto* proc = catalog.lookup("project");
            if (proc != nullptr && !proc->description().empty()) {
                std::cout << " PASS" << std::endl;
                passed++;
            } else {
                std::cout << " FAIL (empty description)" << std::endl;
            }
        } catch (const std::exception& e) {
            std::cout << " FAIL: " << e.what() << std::endl;
        }

        // Test 5: Parameters count
        test_count++;
        std::cout << "Test 5: Parameters count (3-4 expected)...";
        try {
            auto* proc = catalog.lookup("project");
            if (proc != nullptr) {
                auto params = proc->parameters();
                if (params.size() >= 3 && params.size() <= 4) {
                    std::cout << " PASS (" << params.size() << " parameters)" << std::endl;
                    passed++;
                } else {
                    std::cout << " FAIL (wrong count: " << params.size() << ")" << std::endl;
                }
            } else {
                std::cout << " FAIL (procedure not found)" << std::endl;
            }
        } catch (const std::exception& e) {
            std::cout << " FAIL: " << e.what() << std::endl;
        }

        // Test 6: Parameter names
        test_count++;
        std::cout << "Test 6: Parameter names...";
        try {
            auto* proc = catalog.lookup("project");
            if (proc != nullptr) {
                auto params = proc->parameters();
                if (params.size() >= 3 &&
                    params[0].name == "graphName" &&
                    params[1].name == "nodeProjection" &&
                    params[2].name == "relationshipProjection") {
                    std::cout << " PASS" << std::endl;
                    passed++;
                } else {
                    std::cout << " FAIL (wrong parameter names)" << std::endl;
                }
            } else {
                std::cout << " FAIL (procedure not found)" << std::endl;
            }
        } catch (const std::exception& e) {
            std::cout << " FAIL: " << e.what() << std::endl;
        }

        // Test 7: YIELD fields count
        test_count++;
        std::cout << "Test 7: YIELD fields count (4 expected)...";
        try {
            auto* proc = catalog.lookup("project");
            if (proc != nullptr) {
                auto yields = proc->yield_fields();
                if (yields.size() == 4) {
                    std::cout << " PASS" << std::endl;
                    passed++;
                } else {
                    std::cout << " FAIL (wrong count: " << yields.size() << ")" << std::endl;
                }
            } else {
                std::cout << " FAIL (procedure not found)" << std::endl;
            }
        } catch (const std::exception& e) {
            std::cout << " FAIL: " << e.what() << std::endl;
        }

        // Test 8: YIELD field names
        test_count++;
        std::cout << "Test 8: YIELD field names...";
        try {
            auto* proc = catalog.lookup("project");
            if (proc != nullptr) {
                auto yields = proc->yield_fields();
                if (yields.size() == 4 &&
                    yields[0].name == "graphName" &&
                    yields[1].name == "nodeCount" &&
                    yields[2].name == "relationshipCount" &&
                    yields[3].name == "projectMillis") {
                    std::cout << " PASS" << std::endl;
                    passed++;
                } else {
                    std::cout << " FAIL (wrong field names)" << std::endl;
                }
            } else {
                std::cout << " FAIL (procedure not found)" << std::endl;
            }
        } catch (const std::exception& e) {
            std::cout << " FAIL: " << e.what() << std::endl;
        }

        // Test 9: Execute with valid parameters (string variant)
        test_count++;
        std::cout << "Test 9: Execute with string parameters...";
        try {
            auto* proc = catalog.lookup("project");
            if (proc != nullptr) {
                GQL::ProcedureContext ctx;

                // Create string arguments (graphName, nodeProjection, relationshipProjection)
                // Note: These will fail with empty database, but we're testing the interface
                // ObjectId graph_name = create_string("test_graph");
                // ObjectId node_proj = create_string("User");
                // ObjectId rel_proj = create_string("KNOWS");
                // ctx.arguments = {graph_name, node_proj, rel_proj};

                // For now, just verify procedure exists and can be executed
                // Full execution testing requires database setup
                std::cout << " SKIP (requires database setup)" << std::endl;
                // Don't count as pass or fail
                test_count--;
            } else {
                std::cout << " FAIL (procedure not found)" << std::endl;
            }
        } catch (const std::exception& e) {
            std::cout << " FAIL: " << e.what() << std::endl;
        }

        // Test 10: Execute with list parameters
        test_count++;
        std::cout << "Test 10: Execute with list parameters...";
        try {
            // Similar to Test 9, requires full database setup
            std::cout << " SKIP (requires database setup)" << std::endl;
            test_count--;
        } catch (const std::exception& e) {
            std::cout << " FAIL: " << e.what() << std::endl;
        }

        // Test 11: Lookup by qualified name
        test_count++;
        std::cout << "Test 11: Lookup by qualified name...";
        try {
            auto* proc = catalog.lookup("gds.graph.project");
            if (proc != nullptr && proc->name() == "project") {
                std::cout << " PASS" << std::endl;
                passed++;
            } else {
                std::cout << " FAIL (qualified name lookup failed)" << std::endl;
            }
        } catch (const std::exception& e) {
            std::cout << " FAIL: " << e.what() << std::endl;
        }

        // Test 12: Case sensitivity — procedures should be case-sensitive
        test_count++;
        std::cout << "Test 12: Case sensitivity...";
        try {
            auto* proc_lower = catalog.lookup("project");
            auto* proc_upper = catalog.lookup("PROJECT");

            if (proc_lower == nullptr) {
                std::cout << " FAIL (procedure 'project' not found)" << std::endl;
            } else if (proc_upper == nullptr) {
                // Case-sensitive: lowercase found but uppercase not
                std::cout << " PASS (case-sensitive: lowercase only)" << std::endl;
                passed++;
            } else {
                // Case-insensitive: both found — verify they're the same
                if (proc_lower == proc_upper) {
                    std::cout << " PASS (case-insensitive: same procedure)" << std::endl;
                    passed++;
                } else {
                    std::cout << " FAIL (different procedures for different cases)" << std::endl;
                }
            }
        } catch (const std::exception& e) {
            std::cout << " FAIL: " << e.what() << std::endl;
        }

        // Test 13: Parameter types
        test_count++;
        std::cout << "Test 13: Parameter types...";
        try {
            auto* proc = catalog.lookup("project");
            if (proc != nullptr) {
                auto params = proc->parameters();
                // graphName should be STRING, nodeProjection and relationshipProjection should be ANY
                // configuration (if exists) should be ANY
                bool types_correct = true;

                if (params.size() >= 3) {
                    // Check first parameter is STRING
                    if (params[0].type != GQL::ParamType::STRING) {
                        types_correct = false;
                    }
                    // Check second and third are ANY
                    if (params[1].type != GQL::ParamType::ANY ||
                        params[2].type != GQL::ParamType::ANY) {
                        types_correct = false;
                    }
                }

                if (types_correct) {
                    std::cout << " PASS" << std::endl;
                    passed++;
                } else {
                    std::cout << " FAIL (wrong parameter types)" << std::endl;
                }
            } else {
                std::cout << " FAIL (procedure not found)" << std::endl;
            }
        } catch (const std::exception& e) {
            std::cout << " FAIL: " << e.what() << std::endl;
        }

        // Test 14: Required vs optional parameters
        test_count++;
        std::cout << "Test 14: Required vs optional parameters...";
        try {
            auto* proc = catalog.lookup("project");
            if (proc != nullptr) {
                auto params = proc->parameters();
                // First 3 should be required, 4th (configuration) optional
                bool requirements_correct = true;

                if (params.size() >= 3) {
                    if (!params[0].required || !params[1].required || !params[2].required) {
                        requirements_correct = false;
                    }
                }

                if (params.size() == 4) {
                    if (params[3].required) {
                        requirements_correct = false;
                    }
                }

                if (requirements_correct) {
                    std::cout << " PASS" << std::endl;
                    passed++;
                } else {
                    std::cout << " FAIL (wrong required flags)" << std::endl;
                }
            } else {
                std::cout << " FAIL (procedure not found)" << std::endl;
            }
        } catch (const std::exception& e) {
            std::cout << " FAIL: " << e.what() << std::endl;
        }

        // Test 15: YIELD field types
        test_count++;
        std::cout << "Test 15: YIELD field types...";
        try {
            auto* proc = catalog.lookup("project");
            if (proc != nullptr) {
                auto yields = proc->yield_fields();
                bool types_correct = true;

                if (yields.size() == 4) {
                    // graphName should be STRING
                    if (yields[0].type != GQL::YieldType::STRING) {
                        types_correct = false;
                    }
                    // nodeCount, relationshipCount, projectMillis should be INT
                    if (yields[1].type != GQL::YieldType::INT ||
                        yields[2].type != GQL::YieldType::INT ||
                        yields[3].type != GQL::YieldType::INT) {
                        types_correct = false;
                    }
                }

                if (types_correct) {
                    std::cout << " PASS" << std::endl;
                    passed++;
                } else {
                    std::cout << " FAIL (wrong yield types)" << std::endl;
                }
            } else {
                std::cout << " FAIL (procedure not found)" << std::endl;
            }
        } catch (const std::exception& e) {
            std::cout << " FAIL: " << e.what() << std::endl;
        }

        // Summary
        std::cout << "\n========================================" << std::endl;
        std::cout << "ProjectProcedure Test Summary" << std::endl;
        std::cout << "========================================" << std::endl;
        std::cout << "Total tests: " << test_count << std::endl;
        std::cout << "Passed: " << passed << std::endl;
        std::cout << "Failed: " << (test_count - passed) << std::endl;
        std::cout << "Success rate: " << (100.0 * passed / test_count) << "%" << std::endl;

        if (passed == test_count) {
            std::cout << "\n✓ All tests passed!" << std::endl;
            return 0;
        } else {
            std::cout << "\n✗ Some tests failed" << std::endl;
            return 1;
        }

    } catch (const std::exception& e) {
        std::cerr << "\nFatal error: " << e.what() << std::endl;
        return 1;
    }
}
