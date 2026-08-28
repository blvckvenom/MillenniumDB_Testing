# MillenniumDB Copilot Instructions

## Project Overview

**MillenniumDB** is a research-oriented graph database management system (DBMS) supporting multiple graph models and query languages. The system is in active development and not production-ready.

### Supported Models
- **RDF Model**: SPARQL 1.1 queries
- **Quad Model (QM)**: Custom Cypher-like language (MQL), single edge labels, directed edges only  
- **GQL Model**: GQL standard (early implementation), undirected edges, multiple edge labels

**Critical**: Each model is completely isolated. Once data is imported in a model, only that model's query language can be used.

## Build System

### Dependencies
- **C++17** required (`-std=c++17`)
- **Boost 1.82** must be manually installed in `third_party/boost_1_82/include/`
- **ANTLR4 runtime** in `third_party/antlr4-runtime-4.13.1/`
- System packages: CMake ≥3.12, GCC ≥8.1, OpenSSL, ICU, ncursesw

### Build Commands
```bash
# Release build (use for normal development)
cmake -B build/Release -D CMAKE_BUILD_TYPE=Release
cmake --build build/Release -j$(nproc)

# Debug build (includes AddressSanitizer & UndefinedBehaviorSanitizer)
cmake -B build/Debug -D CMAKE_BUILD_TYPE=Debug
cmake --build build/Debug -j$(nproc)
```

**Important**: Debug builds automatically enable `-fsanitize=undefined,address`. The main executable is `build/<BuildType>/bin/mdb`.

### VS Code Tasks
Use pre-configured tasks in `.vscode/tasks.json`:
- `Debug Build` / `Release Build` - Build with dependencies
- `Debug Configure` / `Release Configure` - Run CMake configuration

## Architecture

### ObjectId System (`src/graph_models/object_id.h`)
**Everything is an ObjectId**: 64-bit value with 8-bit type prefix (4-bit generic type + 2-bit subtype + 2-bit modifier) and 56-bit payload.

- **Storage modes**: inline (small values), external (dictionary reference), tmp (temporary)
- **Types**: NULL, nodes, IRIs, strings, numerics, datetimes, edges, paths, tensors, lists, etc.
- Node/edge IDs, property values, query results - all are ObjectIds

### Directory Structure
- `src/bin/` - Entry point (`mdb.cc`) and CLI tools
- `src/graph_models/` - Model-specific implementations (RDF, Quad, GQL)
  - `common/` - Shared code across models
  - `object_id.h` - Central ObjectId type
- `src/storage/` - Persistent layer (B+trees, pages, catalog)
- `src/query/` - Query processing pipeline
  - `parser/grammar/` - ANTLR4 grammars (separate per model)
  - `executor/binding_iter/` - Iterator-based execution
  - `optimizer/` - Query optimization
- `src/import/` - Data importers per model
- `src/network/` - HTTP/WebSocket server
- `tests/` - Integration tests (sparql/, mql/, gql/)

### Query Execution Flow
1. **Parse**: ANTLR4 grammar → AST (`src/query/parser/grammar/{sparql,mql,gql}/`)
2. **Visit**: Visitor pattern converts AST → Expr tree (`src/query/parser/expr/`)
3. **Rewrite**: Query rewriting passes (`src/query/rewriter/`)
4. **Optimize**: Convert Expr → BindingExpr (`src/query/optimizer/`)
5. **Execute**: Iterator-based volcano model (`src/query/executor/binding_iter/`)

### Storage Layer
- Custom B+tree implementations for all indexes
- Model-specific indexing (e.g., RDF uses SPO/POS/OSP permutations)
- String dictionary for external storage (strings > inline threshold)
- Model catalog stored in database directory

## Development Workflows

### Running the Server
```bash
# Import data (automatically detects model from file extension)
build/Release/bin/mdb import data/example/gql/posts/posts.gql data/dbs/gql/posts

# Start server (web UI at localhost:4321, API at localhost:1234)
build/Release/bin/mdb server data/dbs/gql/posts

# Query via script
echo "MATCH (a)-[b]->(c) RETURN * LIMIT 10" > query.gql
bash scripts/query query.gql
```

### Testing
```bash
# Run all tests (auto-builds Debug, creates Python venv)
./scripts/run-tests

# Run specific test suites
./scripts/run-tests sparql   # SPARQL integration tests
./scripts/run-tests gql      # GQL integration tests
./scripts/run-tests unit     # C++ unit tests via ctest
```

Unit tests are defined in `CMakeLists.txt:108-149` and located in `src/tests/`.

### Debugging

**GDB with VS Code**: Use `.vscode/launch.json` configurations:
- **Server** - Debug server process
- **Import** - Debug import process
- **CLI Launch** - Debug interactive CLI

**Manual GDB**:
```bash
gdb --args build/Debug/bin/mdb server data/dbs/gql/posts
```

See `GDB_DEBUGGING_GUIDE.md` for detailed debugging instructions.

## Code Conventions

### Naming
- **Classes**: PascalCase (`AggProject`, `ProjectionStorage`)
- **Functions/Variables**: snake_case (`process_binding`, `projection_name`)
- **Member variables**: snake_case with trailing underscore for privates (`projection_storage_`)
- **Constants**: UPPER_SNAKE_CASE (`MAX_BUFFER_SIZE`)

### File Organization
- **Headers**: `.h` extension (NOT `.hpp`)
- **Source**: `.cc` extension (NOT `.cpp`, except ANTLR-generated files)
- **Templates**: Typically header-only
- One class per file, filename matches class name (snake_case)

### Common Patterns
- **Visitor Pattern**: Used extensively for Expr tree traversal
- **Iterator Pattern**: Query execution uses binding iterators
- **RAII**: Scope guards, unique_ptr for ownership
- **Exceptions**: Runtime errors for query semantic issues

## Key Implementation Details

### Adding GQL Aggregate Functions
Recent addition: `PROJECT()` aggregate function. Pattern to follow:

1. **Grammar** (`src/query/parser/grammar/gql/GQLParser.g4`):
   ```antlr
   aggregateFunction
       : ...
       | PROJECT LEFT_PAREN projectionName=characterStringLiteral RIGHT_PAREN  #gqlProjectFunction
   ```

2. **AST Expression** (`src/query/parser/expr/gql/expr_agg_project.h`):
   ```cpp
   class ExprAggProject : public Expr {
       std::unique_ptr<Expr> projection_name_expr;
       VarId var;
   };
   ```

3. **Visitor** (`src/query/parser/grammar/gql/query_visitor.cc`):
   ```cpp
   std::any visitGqlProjectFunction(GQLParser::GqlProjectFunctionContext* ctx) {
       visit(ctx->projectionName);
       auto expr = std::move(current_expr);
       current_expr = std::make_unique<ExprAggProject>(std::move(expr), agg_var);
   }
   ```

4. **Executor** (`src/query/executor/binding_iter/aggregation/gql/agg_project.h`):
   Implement aggregation logic with `begin()`, `next()`, `reset()`, `process()`.

5. **Optimizer** (`src/query/optimizer/gql/expr_to_binding_expr.cc`):
   Convert `ExprAggProject` → `AggProject` binding iterator.

### String Handling
- Use `get_query_ctx().get_or_create_object_id()` to convert strings to ObjectIds
- Extract strings from ObjectIds using `StringManager::get_str()`
- Single-quoted strings in queries are converted during parsing

### Projection Feature (Custom Extension)
- **CLI commands**: `list-projections`, `inspect-projection`, `drop-projection`
- **Storage**: Projections stored in `<db_folder>/projections/<name>/`
- **Indexes**: B+trees for nodes, edges, properties (separate from main DB)
- See `PROJECT_FUNCTION_COMPLETE_GUIDE.md` for full documentation

## Common Pitfalls

1. **ObjectId validation**: Always check ObjectId type before extracting payload
   ```cpp
   if (oid.get_generic_type() != ObjectId::GenericType::STRING_SIMPLE) {
       throw std::runtime_error("Expected string, got: " + std::to_string(oid.id));
   }
   ```

2. **Empty string literals**: Parser may create empty string ObjectIds - validate:
   ```cpp
   std::string str = StringManager::get_str(name_oid);
   if (str.empty()) throw std::runtime_error("String cannot be empty");
   ```

3. **Model isolation**: Never mix RDF/Quad/GQL code paths - check `Catalog::get_model_id()`

4. **Build regeneration**: After modifying ANTLR grammars, manually regenerate parsers:
   ```bash
   cd src/query/parser/grammar/gql
   ./generate.sh  # Requires antlr-4.13.1-complete.jar in /usr/local/lib/
   ```

5. **Sanitizer errors**: Debug builds fail fast on memory issues - this is intentional

## Documentation

- **Wiki**: `MillenniumDB.wiki/` (Git submodule)
- **Setup**: `MillenniumDB.wiki/Setup.md`
- **SPARQL Status**: `MillenniumDB.wiki/SPARQL-Implementation-Status.md`
- **Example data**: `data/example/{rdf,qm,gql}/`
- **Project docs**: `PROJECT_IMPLEMENTATION_COMPLETE.md`, `PROJECT_FUNCTION_COMPLETE_GUIDE.md`

## AI Agent Guidelines

- **Read context first**: Check existing implementation patterns before suggesting changes
- **Respect model boundaries**: Don't suggest RDF solutions for GQL problems
- **Test your changes**: Use `./scripts/run-tests` to verify modifications
- **Follow ObjectId patterns**: All values flow through the ObjectId system
- **Check sanitizers**: Debug build errors often reveal real issues
- **Preserve iterator semantics**: Query execution relies on correct iterator behavior
