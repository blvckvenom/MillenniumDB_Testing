# GDB script for debugging PROJECT function issue
# Usage: gdb --args build/Debug/bin/mdb server data/dbs/gql/posts -x debug_project.gdb

# Configure output for better readability
set print pretty on
set print object on
set print static-members on
set print vtbl on
set print demangle on
set demangle-style gnu-v3
set pagination off

# Set breakpoints at critical locations

# 1. Where the error occurs
break src/query/executor/binding_iter/aggregation/gql/agg_project.h:60
commands 1
  silent
  printf "\n=== BREAKPOINT 1: Error thrown - projection_name is empty ===\n"
  printf "projection_name = '%s'\n", projection_name.c_str()
  printf "projection_name.size() = %zu\n", projection_name.size()
  printf "projection_name_oid = %lx\n", projection_name_oid.id
  backtrace 5
  continue
end

# 2. Where projection name is evaluated
break src/query/executor/binding_iter/aggregation/gql/agg_project.h:43
commands 2
  silent
  printf "\n=== BREAKPOINT 2: Evaluating projection name expression ===\n"
  printf "expr pointer = %p\n", expr.get()
  if expr.get() != 0
    printf "Expression type: %s\n", typeid(*expr).name()
  else
    printf "expr is NULL!\n"
  end
  continue
end

# 3. After evaluation, before unpacking
break src/query/executor/binding_iter/aggregation/gql/agg_project.h:49
commands 3
  silent
  printf "\n=== BREAKPOINT 3: After eval, checking ObjectId type ===\n"
  printf "name_oid.id = 0x%lx\n", name_oid.id
  printf "Type extracted: %d\n", (int)((name_oid.id >> 56) & 0xFF)
  continue
end

# 4. After unpack_string
break src/query/executor/binding_iter/aggregation/gql/agg_project.h:54
commands 4
  silent
  printf "\n=== BREAKPOINT 4: After unpack_string ===\n"
  printf "projection_name = '%s'\n", projection_name.c_str()
  printf "projection_name.size() = %zu\n", projection_name.size()
  if projection_name.size() == 0
    printf "WARNING: String is empty after unpacking!\n"
    backtrace 8
  end
  continue
end

# 5. Where ExprAggProject is converted to AggProject
break src/query/optimizer/property_graph_model/expr_to_binding_expr.cc:641
commands 5
  silent
  printf "\n=== BREAKPOINT 5: Converting ExprAggProject ===\n"
  printf "expr.var = %d\n", expr.var.id
  printf "expr.projection_name_expr = %p\n", expr.projection_name_expr.get()
  if expr.projection_name_expr.get() != 0
    printf "projection_name_expr type: %s\n", typeid(*expr.projection_name_expr).name()
  else
    printf "projection_name_expr is NULL!\n"
  end
  continue
end

# 6. Inside check_and_make_aggregate, after accept_visitor
break src/query/optimizer/property_graph_model/expr_to_binding_expr.cc:662
commands 6
  silent
  printf "\n=== BREAKPOINT 6: Creating AggProject with binding expression ===\n"
  printf "tmp pointer = %p\n", tmp.get()
  if tmp.get() != 0
    printf "tmp type: %s\n", typeid(*tmp).name()
  else
    printf "tmp is NULL!\n"
  end
  continue
end

# Print startup message
printf "\n"
printf "========================================\n"
printf "  GDB Debug Session for PROJECT Issue  \n"
printf "========================================\n"
printf "\n"
printf "Breakpoints set at:\n"
printf "  1. agg_project.h:60  - Error location\n"
printf "  2. agg_project.h:43  - Expression eval\n"
printf "  3. agg_project.h:49  - After eval\n"
printf "  4. agg_project.h:54  - After unpack_string\n"
printf "  5. expr_to_binding_expr.cc:641 - ExprAggProject conversion\n"
printf "  6. expr_to_binding_expr.cc:662 - AggProject creation\n"
printf "\n"
printf "All breakpoints are set to auto-continue with debug output.\n"
printf "The program will run and print diagnostic information.\n"
printf "\n"
printf "To stop at a specific breakpoint, use: disable <breakpoint_number>\n"
printf "Example: disable 1  (then breakpoint 1 won't auto-continue)\n"
printf "\n"
printf "Starting server...\n"
printf "========================================\n"
printf "\n"

# Start the program
run
