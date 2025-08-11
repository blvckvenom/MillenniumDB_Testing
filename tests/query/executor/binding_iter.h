#pragma once
#include "query/executor/binding.h"
class BindingIterVisitor;
class BindingIter {
public:
    virtual ~BindingIter() = default;
    void begin(Binding& b) { _begin(b); }
    bool next() { return _next(); }
    void reset() { _reset(); }
    virtual void assign_nulls() = 0;
    virtual void accept_visitor(BindingIterVisitor&) = 0;
protected:
    virtual void _begin(Binding&) = 0;
    virtual bool _next() = 0;
    virtual void _reset() = 0;
};
class BindingIterVisitor {};
