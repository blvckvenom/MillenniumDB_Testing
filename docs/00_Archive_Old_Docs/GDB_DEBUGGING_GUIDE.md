# GDB Debugging Guide for PROJECT Function Issue

## Problem Summary
The query `MATCH (p:Paper) RETURN PROJECT('all_papers')` is failing with:
- **Error**: "PROJECT() projection name cannot be empty"
- **Root Cause**: The projection name 'all_papers' is not being properly passed to the aggregation function
- **Evidence**: Query plan shows `PROJECT("")` instead of `PROJECT("all_papers")`

## Key Source Locations

### Error Location
- **File**: `src/query/executor/binding_iter/aggregation/gql/agg_project.h:60`
- **Function**: `AggProject::initialize_if_needed()`
- **Code**: Throws exception when `projection_name.empty()` is true

### Expression Parsing and Transformation
1. **Expression Creation**: `src/query/parser/grammar/gql/query_visitor.cc`
   - Creates `ExprAggProject` with `projection_name_expr`
2. **Expression to Binding**: `src/query/optimizer/property_graph_model/expr_to_binding_expr.cc:639-642`
   - Converts `ExprAggProject` → `AggProject` binding iterator
3. **Template Function**: `expr_to_binding_expr.cc:644-676`
   - Line 661: `expr->accept_visitor(*this)` - Converts projection name expression
   - Line 662: Creates `AggProject` with converted binding expression

---

## GDB Setup and Basic Commands

### 1. Build in Debug Mode (Already Done)
```bash
# You've already done this, but for reference:
cmake -B build/Debug -D CMAKE_BUILD_TYPE=Debug
cmake --build build/Debug
```

### 2. Starting the Server with GDB

**Option A: Start server directly in GDB**
```bash
gdb --args build/Debug/bin/mdb server data/dbs/gql/posts
```

**Option B: Attach GDB to running server**
```bash
# Terminal 1: Start server
build/Debug/bin/mdb server data/dbs/gql/posts

# Terminal 2: Find process ID and attach
ps aux | grep mdb
gdb -p <PID>
```

### 3. Essential GDB Commands

```gdb
# Run the program
(gdb) run

# Set breakpoints
(gdb) break filename.cc:line_number
(gdb) break function_name
(gdb) break ClassName::method_name

# Continue execution
(gdb) continue   # or just 'c'

# Step through code
(gdb) next       # or 'n' - step over (don't enter functions)
(gdb) step       # or 's' - step into functions
(gdb) finish     # execute until current function returns

# Print variables
(gdb) print variable_name           # or 'p'
(gdb) print *pointer                # dereference pointer
(gdb) print object.member
(gdb) print object->member          # for pointers

# Print strings
(gdb) print projection_name         # C++ std::string
(gdb) print projection_name.c_str() # C-style view

# Examine memory
(gdb) x/s address                   # examine as string
(gdb) x/4x address                  # examine 4 hex values

# Inspect objects
(gdb) print *this                   # current object
(gdb) print expr                    # expression object
(gdb) print *expr                   # dereference unique_ptr

# View call stack
(gdb) backtrace  # or 'bt' - show call stack
(gdb) frame n    # switch to frame n
(gdb) up         # move up call stack
(gdb) down       # move down call stack

# List source code
(gdb) list       # show source around current line
(gdb) list function_name

# Watch variables (break when value changes)
(gdb) watch variable_name

# Delete breakpoints
(gdb) info breakpoints              # list all breakpoints
(gdb) delete n                      # delete breakpoint n
(gdb) delete                        # delete all breakpoints

# Quit
(gdb) quit       # or 'q'
```

---

## Debugging Strategy for PROJECT Issue

### Phase 1: Confirm the Error Location

Start GDB and set a breakpoint at the error:

```bash
gdb --args build/Debug/bin/mdb server data/dbs/gql/posts
```

```gdb
(gdb) break agg_project.h:60
(gdb) run
```

Now in your browser, execute the query:
```
MATCH (p:Paper)
RETURN PROJECT('all_papers')
```

When it hits the breakpoint:
```gdb
(gdb) print projection_name
(gdb) print projection_name.size()
(gdb) print projection_name_oid
(gdb) backtrace
```

**Expected findings**:
- `projection_name` is empty
- `projection_name_oid` might be NULL or corrupted

### Phase 2: Track Expression Construction

Set breakpoints earlier in the pipeline:

```gdb
(gdb) delete     # clear previous breakpoints
(gdb) break GQL::ExprToBindingExpr::visit(GQL::ExprAggProject&)
(gdb) run
```

Execute the query again. When it breaks:

```gdb
# Inspect the ExprAggProject object
(gdb) print expr
(gdb) print expr.projection_name_expr
(gdb) print *expr.projection_name_expr

# Step through the function
(gdb) step

# When inside check_and_make_aggregate at line 661
(gdb) print expr
(gdb) print *expr
(gdb) step       # Step into expr->accept_visitor(*this)

# After accept_visitor returns
(gdb) print tmp
(gdb) print *tmp
```

**What to look for**:
- Is `expr.projection_name_expr` a valid pointer?
- What type is it? (likely `ExprTerm` with a string literal)
- After `accept_visitor`, what does `tmp` contain?

### Phase 3: Examine the Expression Evaluation

Set a breakpoint where the expression is evaluated:

```gdb
(gdb) delete
(gdb) break agg_project.h:43
(gdb) run
```

Execute the query. When it breaks at line 43:

```gdb
# Inspect the expression before evaluation
(gdb) print expr
(gdb) print *expr
(gdb) print binding

# Step to line 43 and execute the eval
(gdb) next

# After eval, inspect the result
(gdb) print name_oid
(gdb) print name_oid.id
(gdb) print /x name_oid.id   # Print in hexadecimal

# Continue to line 53 where unpack_string is called
(gdb) until 53
(gdb) step       # Step into unpack_string
```

**What to look for**:
- Is `name_oid` valid (non-NULL)?
- What is the ObjectId's type?
- Does `unpack_string` correctly extract the string?

### Phase 4: Deep Dive into String Unpacking

If the issue is in string unpacking:

```gdb
(gdb) delete
(gdb) break GQL::Conversions::unpack_string
(gdb) run
```

Execute the query:

```gdb
(gdb) print oid
(gdb) print /x oid.id
(gdb) step       # Step through unpack_string

# Watch how the string is constructed
(gdb) print result   # (if there's a result variable)
```

---

## Specific Debugging Scenarios

### Scenario A: Expression is NULL

```gdb
(gdb) break expr_to_binding_expr.cc:661
(gdb) run
# Execute query
(gdb) print expr
(gdb) if expr == nullptr
> print "Expression is NULL!"
> end
```

### Scenario B: Expression is Wrong Type

```gdb
(gdb) break expr_to_binding_expr.cc:661
(gdb) run
# Execute query
(gdb) print *expr
(gdb) print typeid(*expr).name()   # Get actual type
```

### Scenario C: Binding Expression is Corrupted

```gdb
(gdb) break expr_to_binding_expr.cc:662
(gdb) run
# Execute query
(gdb) print tmp
(gdb) print *tmp
(gdb) print typeid(*tmp).name()
```

### Scenario D: ObjectId is Malformed

```gdb
(gdb) break agg_project.h:43
(gdb) run
# Execute query
(gdb) next       # Execute the eval
(gdb) print name_oid
(gdb) print /x name_oid.id
(gdb) print (int)((name_oid.id >> 56) & 0xFF)   # Extract type bits
```

---

## Automated Debugging Script

Save this as `debug_project.gdb`:

```gdb
# Set breakpoints at key locations
break agg_project.h:43
break agg_project.h:53
break agg_project.h:60
break expr_to_binding_expr.cc:661
break expr_to_binding_expr.cc:662

# Configure display
set print pretty on
set print object on
set print static-members on

# Run the program
run

# Commands to execute at each breakpoint
commands
  backtrace 3
  info locals
  continue
end
```

Run with:
```bash
gdb --args build/Debug/bin/mdb server data/dbs/gql/posts -x debug_project.gdb
```

---

## Advanced Techniques

### Conditional Breakpoints

Break only when specific conditions are met:

```gdb
# Break only when projection_name is empty
(gdb) break agg_project.h:59 if projection_name.empty()

# Break only when ObjectId is NULL
(gdb) break agg_project.h:43 if name_oid.is_null()
```

### Pretty Printing C++ Objects

```gdb
# Enable Python pretty printers (if available)
(gdb) set print pretty on

# For std::string
(gdb) print projection_name

# For std::unique_ptr
(gdb) print *expr
```

### Logging to File

```gdb
(gdb) set logging on
(gdb) set logging file debug_output.txt
# All output now goes to file
```

---

## Quick Reference Card

| Task | Command |
|------|---------|
| Start debugging | `gdb --args build/Debug/bin/mdb server data/dbs/gql/posts` |
| Set breakpoint | `break file.cc:123` or `break function_name` |
| Run program | `run` |
| Continue | `continue` or `c` |
| Step over | `next` or `n` |
| Step into | `step` or `s` |
| Print variable | `print var` or `p var` |
| Print string | `print str.c_str()` |
| Show backtrace | `backtrace` or `bt` |
| List source | `list` |
| Watch variable | `watch var` |
| Quit | `quit` or `q` |

---

## Next Steps After Identifying the Issue

Once you find where the string is lost or corrupted:

1. **Document the findings**:
   - What is the value at each step?
   - Where does it change from correct to incorrect?

2. **Share the backtrace**:
   ```gdb
   (gdb) backtrace full
   ```

3. **Share variable values**:
   ```gdb
   (gdb) info locals
   (gdb) print relevant_variable
   ```

4. **Test hypothesis**:
   - If you suspect a specific line, try modifying the code
   - Rebuild in Debug mode
   - Re-test with GDB

---

## Common Issues and Solutions

### Issue: "No debugging symbols found"
**Solution**: Ensure you're using the Debug build:
```bash
file build/Debug/bin/mdb
# Should say "not stripped"
```

### Issue: "Cannot find function"
**Solution**: Use TAB completion:
```gdb
(gdb) break GQL::Agg<TAB>
```

### Issue: Optimizations hiding variables
**Solution**: Already fixed - Debug build uses `-O0`

### Issue: Multi-threaded debugging
If the server uses threads:
```gdb
(gdb) info threads        # List all threads
(gdb) thread <n>          # Switch to thread n
(gdb) set scheduler-locking on   # Only current thread runs
```

---

## Example Session

Here's what a typical debugging session might look like:

```bash
$ gdb --args build/Debug/bin/mdb server data/dbs/gql/posts
GNU gdb (Ubuntu 12.1-0ubuntu1~22.04.2) 12.1
...
(gdb) break agg_project.h:43
Breakpoint 1 at 0x555555678abc: file src/.../agg_project.h, line 43.

(gdb) run
Starting program: build/Debug/bin/mdb server data/dbs/gql/posts
...
MillenniumDB HTTP/WebSocket server listening on http://localhost:1234
...

# Execute query in browser, then...

Breakpoint 1, GQL::AggProject::initialize_if_needed() at agg_project.h:43
43          ObjectId name_oid = expr->eval(*binding);

(gdb) print expr
$1 = std::unique_ptr<BindingExpr> = {get() = 0x7ffff4567890}

(gdb) print *expr
$2 = {<BindingExpr> = {...}, ...}

(gdb) next
44          projection_name_oid = name_oid;

(gdb) print name_oid
$3 = {id = 0x...}

(gdb) print /x name_oid.id
$4 = 0x...

(gdb) continue
```

---

## Contact Points for Additional Help

After gathering debugging information, provide:
1. Full backtrace (`bt full`)
2. Variable values at each key point
3. Any patterns or observations
4. Modified files (if any experimental fixes)
