#pragma once

#include <memory>
#include <string>
#include <vector>
#include <unordered_map>

#include "query/executor/binding_iter.h"
#include "query/parser/op/gql/op_call.h"
#include "query/var_id.h"

namespace GQL {

class CallBindingIter : public BindingIter {
public:
    virtual ~CallBindingIter() = default;

    virtual void print(std::ostream& os, int indent, bool stats) const override = 0;
    virtual void assign_nulls() override = 0;

protected:
    virtual void _begin(Binding& parent_binding) override = 0;
    virtual bool _next() override = 0;
    virtual void _reset() override = 0;

    std::vector<VarId> yield_vars;
    uint64_t result_count = 0;
    bool exhausted = false;
    Binding* parent_binding = nullptr;
};

class CallNamedBindingIter : public CallBindingIter {
public:
    CallNamedBindingIter(
        const std::string& procedure_name,
        const std::vector<std::unique_ptr<Expr>>& arguments,
        const std::vector<VarId>& yield_vars,
        const std::vector<std::string>& yield_fields
    );

    void print(std::ostream& os, int indent, bool stats) const override;
    void assign_nulls() override;

protected:
    void _begin(Binding& parent_binding) override;
    bool _next() override;
    void _reset() override;

private:
    std::string procedure_name;
    std::vector<std::unique_ptr<Expr>> arguments;
    std::vector<std::string> yield_fields;
    std::vector<std::unordered_map<std::string, std::string>> procedure_results;
    size_t current_result_index = 0;

    void execute_db_labels();
    void execute_db_property_keys();
    void execute_db_relationship_types();
};

class CallInlineBindingIter : public CallBindingIter {
public:
    CallInlineBindingIter(
        std::unique_ptr<BindingIter> subquery_iter,
        const std::vector<VarId>& yield_vars
    );

    void print(std::ostream& os, int indent, bool stats) const override;
    void assign_nulls() override;

protected:
    void _begin(Binding& parent_binding) override;
    bool _next() override;
    void _reset() override;

private:
    std::unique_ptr<BindingIter> subquery_iter;
};

} // namespace GQL
