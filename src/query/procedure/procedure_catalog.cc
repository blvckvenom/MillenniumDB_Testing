#include "procedure_catalog.h"

#include <sstream>
#include <stdexcept>

namespace GQL {

ProcedureCatalog& ProcedureCatalog::get_instance()
{
    static ProcedureCatalog instance;
    return instance;
}

void ProcedureCatalog::register_procedure(std::unique_ptr<Procedure> proc)
{
    std::lock_guard<std::mutex> lock(mutex);

    std::string qualified_name = proc->qualified_name();

    if (procedures.count(qualified_name)) {
        throw std::runtime_error(
            "Procedure '" + qualified_name + "' is already registered"
        );
    }

    procedures[qualified_name] = std::move(proc);
}

Procedure* ProcedureCatalog::lookup(const std::string& name)
{
    std::lock_guard<std::mutex> lock(mutex);

    auto it = procedures.find(name);
    if (it == procedures.end()) {
        return nullptr;
    }

    return it->second.get();
}

Procedure* ProcedureCatalog::lookup_qualified(const std::vector<std::string>& parts)
{
    // Join parts with dots
    std::ostringstream oss;
    for (size_t i = 0; i < parts.size(); i++) {
        if (i > 0) {
            oss << ".";
        }
        oss << parts[i];
    }

    return lookup(oss.str());
}

std::vector<std::string> ProcedureCatalog::list_procedures() const
{
    std::lock_guard<std::mutex> lock(mutex);

    std::vector<std::string> names;
    names.reserve(procedures.size());

    for (const auto& [name, proc] : procedures) {
        names.push_back(name);
    }

    return names;
}

std::vector<Procedure*> ProcedureCatalog::get_all_procedures() const
{
    std::lock_guard<std::mutex> lock(mutex);

    std::vector<Procedure*> procs;
    procs.reserve(procedures.size());

    for (const auto& [name, proc] : procedures) {
        procs.push_back(proc.get());
    }

    return procs;
}

bool ProcedureCatalog::exists(const std::string& name)
{
    std::lock_guard<std::mutex> lock(mutex);
    return procedures.count(name) > 0;
}

} // namespace GQL
