#pragma once

#include "graph_models/gql/conversions.h"
#include "query/executor/binding_iter/binding_expr/binding_expr.h"
#include <memory>
#include <optional>
#include <string>

namespace GQL {

class BindingExprCast : public BindingExpr {
public:
    std::unique_ptr<BindingExpr> operand;
    GQL_OID::GenericType targetType;
    std::optional<GQL_OID::GenericSubType> targetNumericSubType;

    BindingExprCast(
        std::unique_ptr<BindingExpr> operand,
        GQL_OID::GenericType targetType,
        std::optional<GQL_OID::GenericSubType> targetNumericSubType = std::nullopt
    ) :
        operand(std::move(operand)),
        targetType(targetType),
        targetNumericSubType(targetNumericSubType)
    { }

    ObjectId eval(const Binding& binding) override
    {
        auto operand_oid = operand->eval(binding);
        if (operand_oid.is_null()) {
            return ObjectId::get_null();
        }

        auto sourceType = GQL_OID::get_generic_type(operand_oid);

        if (sourceType == targetType
            && !(targetType == GQL_OID::GenericType::NUMERIC && targetNumericSubType.has_value()))
        {
            return operand_oid;
        }

        switch (targetType) {
        case GQL_OID::GenericType::BOOL:
            switch (sourceType) {
            case GQL_OID::GenericType::NUMERIC: {
                auto number = GQL::Conversions::to_double(operand_oid);
                auto boolean = number != 0;
                return GQL::Conversions::pack_bool(boolean);
            }
            case GQL_OID::GenericType::STRING:
                return GQL::Conversions::pack_bool(GQL::Conversions::to_lexical_str(operand_oid) == "true");
            default:
                return ObjectId::get_null();
            }

        case GQL_OID::GenericType::NUMERIC:
            switch (sourceType) {
            case GQL_OID::GenericType::NUMERIC: {
                if (!targetNumericSubType.has_value()) {
                    return operand_oid;
                }

                switch (*targetNumericSubType) {
                case GQL_OID::GenericSubType::INTEGER:
                    return GQL::Conversions::pack_int(
                        static_cast<int64_t>(GQL::Conversions::to_double(operand_oid))
                    );
                case GQL_OID::GenericSubType::FLOAT:
                    return GQL::Conversions::pack_float(GQL::Conversions::to_float(operand_oid));
                case GQL_OID::GenericSubType::DOUBLE:
                    return GQL::Conversions::pack_double(GQL::Conversions::to_double(operand_oid));
                default:
                    return operand_oid;
                }
            }
            case GQL_OID::GenericType::BOOL: {
                auto boolean = GQL::Conversions::to_boolean(operand_oid);
                auto boolean_value = boolean == GQL::Conversions::pack_bool(true);
                if (targetNumericSubType == GQL_OID::GenericSubType::FLOAT) {
                    return GQL::Conversions::pack_float(boolean_value ? 1.0f : 0.0f);
                } else if (targetNumericSubType == GQL_OID::GenericSubType::DOUBLE) {
                    return GQL::Conversions::pack_double(boolean_value ? 1.0 : 0.0);
                }
                return GQL::Conversions::pack_int(boolean_value ? 1 : 0);
            }
            case GQL_OID::GenericType::STRING: {
                std::string str = GQL::Conversions::to_lexical_str(operand_oid);
                try {
                    if (targetNumericSubType == GQL_OID::GenericSubType::FLOAT) {
                        return GQL::Conversions::pack_float(std::stof(str));
                    } else if (targetNumericSubType == GQL_OID::GenericSubType::DOUBLE) {
                        return GQL::Conversions::pack_double(std::stod(str));
                    }
                    return GQL::Conversions::pack_int(static_cast<int64_t>(std::stod(str)));
                } catch (...) {
                    return ObjectId::get_null();
                }
            }
            default:
                return ObjectId::get_null();
            }

        case GQL_OID::GenericType::DATE:
            if (sourceType == GQL_OID::GenericType::STRING) {
                try {
                    return ObjectId(DateTime::from_dateTime(Conversions::to_lexical_str(operand_oid)));
                } catch (...) {
                    return ObjectId::get_null();
                }
            }
            return ObjectId::get_null();

        case GQL_OID::GenericType::STRING:
            switch (sourceType) {
            case GQL_OID::GenericType::NUMERIC:
            case GQL_OID::GenericType::BOOL:
            case GQL_OID::GenericType::DATE:
                return GQL::Conversions::pack_string_simple(GQL::Conversions::to_lexical_str(operand_oid));
            default:
                return ObjectId::get_null();
            }

        default:
            return ObjectId::get_null();
        }
    }

    void accept_visitor(BindingExprVisitor& visitor) override
    {
        visitor.visit(*this);
    }

    void print(std::ostream& os, std::vector<BindingIter*>& ops) const override
    {
        os << "CAST(";
        operand->print(os, ops);
        os << " AS ";
        if (targetType == GQL_OID::GenericType::BOOL) {
            os << "BOOL";
        } else if (targetType == GQL_OID::GenericType::NUMERIC) {
            if (targetNumericSubType == GQL_OID::GenericSubType::INTEGER) {
                os << "INTEGER";
            } else if (targetNumericSubType == GQL_OID::GenericSubType::FLOAT) {
                os << "FLOAT";
            } else if (targetNumericSubType == GQL_OID::GenericSubType::DOUBLE) {
                os << "DOUBLE";
            } else {
                os << "NUMERIC";
            }
        } else if (targetType == GQL_OID::GenericType::DATE) {
            os << "DATE";
        } else if (targetType == GQL_OID::GenericType::STRING) {
            os << "STRING";
        }
        os << ")";
    }
};

} // namespace GQL
