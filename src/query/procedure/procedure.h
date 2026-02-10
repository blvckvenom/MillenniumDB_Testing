#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace GQL {

// Forward declaration
class ProcedureContext;

/**
 * Base class for all stored procedures callable via CALL statement.
 * Procedures can accept parameters and return results via YIELD.
 */
class Procedure {
public:
    /**
     * Parameter type specification for procedure arguments.
     */
    enum class ParamType : uint8_t {
        STRING,
        INT,
        FLOAT,
        DOUBLE,
        BOOL,
        LIST,
        ANY  // Accepts any type
    };

    /**
     * Yield field type specification for procedure outputs.
     */
    enum class YieldType : uint8_t {
        STRING,
        INT,
        FLOAT,
        DOUBLE,
        BOOL,
        NODE,
        EDGE,
        PATH,
        LIST
    };

    /**
     * Parameter metadata describing a procedure argument.
     */
    struct Parameter {
        std::string name;
        ParamType type;
        bool required;
        std::string description;

        Parameter(
            std::string name_,
            ParamType type_,
            bool required_,
            std::string description_ = ""
        ) :
            name(std::move(name_)),
            type(type_),
            required(required_),
            description(std::move(description_))
        { }
    };

    /**
     * Yield field metadata describing a procedure output field.
     */
    struct YieldField {
        std::string name;
        YieldType type;
        std::string description;

        YieldField(
            std::string name_,
            YieldType type_,
            std::string description_ = ""
        ) :
            name(std::move(name_)),
            type(type_),
            description(std::move(description_))
        { }
    };

    virtual ~Procedure() = default;

    /**
     * Returns the simple name of the procedure (e.g., "project").
     */
    virtual std::string name() const = 0;

    /**
     * Returns the fully qualified name of the procedure (e.g., "gds.graph.project").
     * By default returns the simple name.
     */
    virtual std::string qualified_name() const
    {
        return name();
    }

    /**
     * Returns a description of what the procedure does.
     */
    virtual std::string description() const
    {
        return "";
    }

    /**
     * Returns the parameter specification for this procedure.
     */
    virtual std::vector<Parameter> parameters() const = 0;

    /**
     * Returns the yield field specification for this procedure.
     */
    virtual std::vector<YieldField> yield_fields() const = 0;

    /**
     * Executes the procedure with the given context.
     * The context provides access to arguments and allows yielding results.
     *
     * @param ctx The procedure execution context.
     * @throws std::runtime_error if execution fails.
     */
    virtual void execute(ProcedureContext& ctx) = 0;
};

} // namespace GQL
