#pragma once

#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

#include "query/procedure/procedure.h"

namespace GQL {

/**
 * Global registry for stored procedures.
 * Manages registration and lookup of procedures callable via CALL statements.
 * Thread-safe singleton.
 */
class ProcedureCatalog {
public:
    /**
     * Gets the singleton instance of the procedure catalog.
     *
     * @return Reference to the catalog instance.
     */
    static ProcedureCatalog& get_instance();

    /**
     * Registers a procedure in the catalog.
     * The procedure's qualified name is used as the lookup key.
     *
     * @param proc Unique pointer to the procedure to register.
     * @throws std::runtime_error if a procedure with the same name already exists.
     */
    void register_procedure(std::unique_ptr<Procedure> proc);

    /**
     * Looks up a procedure by name.
     * Supports both simple names ("project") and qualified names ("gds.graph.project").
     *
     * @param name The procedure name to look up.
     * @return Pointer to the procedure, or nullptr if not found.
     */
    Procedure* lookup(const std::string& name);

    /**
     * Looks up a procedure by qualified name parts.
     *
     * @param parts Vector of name parts (e.g., ["gds", "graph", "project"]).
     * @return Pointer to the procedure, or nullptr if not found.
     */
    Procedure* lookup_qualified(const std::vector<std::string>& parts);

    /**
     * Lists all registered procedure names.
     *
     * @return Vector of procedure qualified names.
     */
    std::vector<std::string> list_procedures() const;

    /**
     * Gets all registered procedures.
     *
     * @return Vector of procedure pointers.
     */
    std::vector<Procedure*> get_all_procedures() const;

    /**
     * Checks if a procedure with the given name exists.
     *
     * @param name The procedure name to check.
     * @return true if the procedure exists, false otherwise.
     */
    bool exists(const std::string& name);

private:
    ProcedureCatalog() = default;

    // Prevent copying
    ProcedureCatalog(const ProcedureCatalog&) = delete;
    ProcedureCatalog& operator=(const ProcedureCatalog&) = delete;

    std::unordered_map<std::string, std::unique_ptr<Procedure>> procedures;
    mutable std::mutex mutex;  // Thread-safe access
};

} // namespace GQL
