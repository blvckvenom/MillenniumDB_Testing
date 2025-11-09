#pragma once

#include <iostream>

#include "query/procedure/procedure.h"
#include "query/procedure/procedure_context.h"

namespace GQL {
namespace Procedures {

/**
 * Simple test procedure that yields a greeting message.
 * Used to validate basic CALL execution with YIELD.
 * No parameters, yields a "message" field.
 */
class TestHello : public Procedure {
public:
    std::string name() const override
    {
        return "test_hello";
    }

    std::string qualified_name() const override
    {
        return "test_hello";
    }

    std::string description() const override
    {
        return "Test procedure that yields a greeting message";
    }

    std::vector<Parameter> parameters() const override
    {
        return {};  // No parameters
    }

    std::vector<YieldField> yield_fields() const override
    {
        return {
            YieldField{"message", YieldType::STRING, "The greeting message"}
        };
    }

    void execute(ProcedureContext& ctx) override
    {
        // Yield one row with a greeting message
        ctx.yield("message", ctx.create_string("Hello, World!"));
        ctx.yield_row();
    }
};

} // namespace Procedures
} // namespace GQL
