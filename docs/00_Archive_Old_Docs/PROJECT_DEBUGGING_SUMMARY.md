# PROJECT Function Debugging Summary

## Date
2025-10-14

## Issue Description
The query `MATCH (p:Paper) RETURN PROJECT('all_papers')` was failing with:
- **Error**: "PROJECT() projection name cannot be empty"
- **Evidence from terminal**: Query plan showed `PROJECT("")` instead of `PROJECT("all_papers")`

## Root Causes Identified via GDB

### 1. Empty ObjectId Value
**Location**: During expression parsing
**Finding**: The `ExprTerm` contained ObjectId `0x4200000000000000`
- Type mask: `0x42` = STRING_SIMPLE_TMP ✓ (correct)
- Value: `0x00...` = Empty (all zeros) ✗ (incorrect)

**Analysis**: The string "all_papers" was never packed into the ObjectId's value portion.

### 2. Unregistered Variables
**Location**: QueryContext at runtime
**Finding**: `var_names = std::vector of length 0`
- No variables were registered at all (not even `?p` from MATCH clause)
- Caused assertion failure when trying to print VarId(2)
- This was a secondary issue preventing error messages from displaying properly

## Fixes Implemented

### Fix 1: Add Null Checks in visitGqlProjectFunction
**File**: `src/query/parser/grammar/gql/query_visitor.cc:1533-1565`

**Changes**:
- Added null check for `ctx->projectionName`
- Added validation that `visit()` produces a valid expression
- Added debug logging (when DEBUG_GQL_QUERY_VISITOR is defined)

**Rationale**: Defensive programming to catch parse errors early and provide clear error messages.

### Fix 2: Add ObjectId Validation in String Literal Parsing
**File**: `src/query/parser/grammar/gql/query_visitor.cc:1721-1731, 1762-1772`

**Changes**:
- After calling `pack_string_simple()`, validate the resulting ObjectId
- Check if value portion is zero when string is non-empty
- Throw detailed error if packing fails

**Affected Functions**:
- `visitSingleQuotedCharacterSequence()`
- `visitDoubleQuotedCharacterSequence()`

**Rationale**: Catches the exact bug we saw in GDB - detects when string packing silently fails.

### Fix 3: Improve Error Messages in AggProject
**File**: `src/query/executor/binding_iter/aggregation/gql/agg_project.h:48-75`

**Changes**:
- Added debug logging of ObjectId values (when DEBUG_GQL_QUERY_VISITOR is defined)
- Enhanced error messages to include:
  - ObjectId value in hexadecimal
  - Type information
  - Context about what failed

**Added headers**: `<iostream>` and `<sstream>`

**Rationale**: Better diagnostics for troubleshooting similar issues in the future.

## Testing Instructions

### Basic Test (Release Mode)
```bash
# Start server
build/Release/bin/mdb server data/dbs/gql/posts

# In browser (http://localhost:4321), execute:
MATCH (p:Paper)
RETURN PROJECT('all_papers')
```

**Expected Result**:
- Either the query succeeds and creates the projection
- OR we get a clear, detailed error message explaining what went wrong

### Debug Mode Test (For Detailed Logging)
```bash
# 1. Enable debug logging
# Edit src/query/parser/grammar/gql/query_visitor.cc line 13:
# Uncomment: #define DEBUG_GQL_QUERY_VISITOR

# 2. Build Debug version
cmake -B build/Debug -D CMAKE_BUILD_TYPE=Debug
cmake --build build/Debug

# 3. Run server
build/Debug/bin/mdb server data/dbs/gql/posts 2>&1 | tee debug_output.log

# 4. Execute query in browser
# Check debug_output.log for detailed trace
```

## GDB Investigation Steps Used

### Breakpoints Set
1. `agg_project.h:60` - Error throw location
2. `agg_project.h:43` - Expression evaluation
3. `expr_to_binding_expr.cc:641` - ExprAggProject conversion
4. `expr_to_binding_expr.cc:22` - ExprTerm creation

### Key GDB Commands Used
```gdb
# Inspect variables
print expr
print expr.projection_name_expr
print *expr.projection_name_expr
print ((GQL::ExprTerm*)expr.projection_name_expr.get())->term
print /x ((GQL::ExprTerm*)expr.projection_name_expr.get())->term.id

# Navigate stack
backtrace
frame N
info locals

# Examine QueryContext
print *this
print this->var_ctx.var_names
```

### Critical Discovery
At `expr_to_binding_expr.cc:641`, the ObjectId value was:
```
$7 = 0x4200000000000000
```

Decoding:
- Bits 56-63 (0x42): Type = STRING_SIMPLE_TMP
- Bits 0-55 (0x00...): Value = EMPTY

This proved the string literal "all_papers" was never stored.

## Files Modified

1. `src/query/parser/grammar/gql/query_visitor.cc`
   - Added validation in `visitGqlProjectFunction()`
   - Added ObjectId checks in string literal visitors

2. `src/query/executor/binding_iter/aggregation/gql/agg_project.h`
   - Enhanced error messages
   - Added debug logging
   - Added required headers

## Next Steps

1. **Monitor for the actual root cause**: These fixes catch symptoms but may not address the underlying issue
   - Why is `pack_string_simple()` returning an empty ObjectId?
   - Is there a bug in `Conversions::pack_string_simple()`?
   - Is the grammar/parser mishandling the string literal?

2. **If the error still occurs with detailed messages**, investigate:
   - `src/graph_models/gql/conversions.cc:213-229` (`pack_string_simple` implementation)
   - ANTLR grammar rule for `characterStringLiteral`
   - String extraction from `ctx->projectionName->getText()`

3. **Consider adding unit tests** for:
   - String literal parsing with PROJECT function
   - ObjectId creation from strings of various lengths
   - Edge cases (empty strings, very long strings, special characters)

## References

- GDB Tutorial: `GDB_DEBUGGING_GUIDE.md`
- PROJECT Implementation: `PROJECT_FUNCTION_COMPLETE_GUIDE.md`
- GQL Grammar: `src/query/parser/grammar/gql/GQLParser.g4:1425`
