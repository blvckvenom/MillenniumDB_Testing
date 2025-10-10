# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

MillenniumDB is a graph-oriented database management system (DBMS) developed by the Millennium Institute for Foundational Research on Data (IMFD). It supports multiple graph models and query languages:

- **RDF Model**: SPARQL 1.1 support (see wiki: SPARQL-Implementation-Status.md)
- **Quad Model (QM)**: Property graphs with single edge labels and directed edges, using a custom Cypher-like query language (MQL)
- **GQL Model**: Property graphs supporting the GQL standard with undirected edges and multiple edge labels (early implementation, still missing functionality)

The project is in active development and not production-ready. Each graph model has its own query language - once you import data in one model, you must use that model's query language.

## Setup and Dependencies

### System Requirements
- x86-64 Linux distribution (Windows via WSL, MacOS on Apple Silicon supported)
- GCC >= 8.1
- CMake >= 3.12
- Git
- libssl
- ncursesw and less (for CLI)
- Python >= 3.8 with venv (for tests)
- ICU library (libicu-dev)
- Boost 1.82 (must be manually installed)

### Install Dependencies

Ubuntu/Debian:
```bash
sudo apt update && sudo apt install git g++ cmake libssl-dev libncurses-dev less python3 python3-venv libicu-dev
```

MacOS:
```bash
brew install cmake ncurses openssl@3 icu4c
```

### Install Boost 1.82

Boost must be manually downloaded and placed in the third_party directory:
```bash
wget -q --show-progress https://archives.boost.io/release/1.82.0/source/boost_1_82_0.tar.gz
tar -xf boost_1_82_0.tar.gz
mkdir -p $MDB_HOME/third_party/boost_1_82/include
mv boost_1_82_0/boost $MDB_HOME/third_party/boost_1_82/include
rm -r boost_1_82_0.tar.gz boost_1_82_0
```

## Build Commands

### Standard Builds

```bash
# Release build (recommended for normal use)
cmake -B build/Release -D CMAKE_BUILD_TYPE=Release
cmake --build build/Release -j <n>  # Use -j <n> for parallel compilation

# Debug build (with sanitizers)
cmake -B build/Debug -D CMAKE_BUILD_TYPE=Debug
cmake --build build/Debug

# Profile build (requires gperftools/tcmalloc)
cmake -B build/Profile -D CMAKE_BUILD_TYPE=Release -D PROFILE=ON
cmake --build build/Profile
```

The main executable is `mdb` and will be located in `build/<BuildType>/bin/mdb`.

### Verify Build
```bash
build/Release/bin/mdb help
```

## Testing

### Run All Tests
```bash
./scripts/run-tests
```

### Run Specific Test Suites
```bash
./scripts/run-tests sparql    # SPARQL integration tests
./scripts/run-tests mql        # MQL integration tests (Quad Model)
./scripts/run-tests gql        # GQL integration tests
./scripts/run-tests unit       # Unit tests via ctest
```

The test script automatically:
1. Sets up a Python virtual environment with dependencies
2. Builds the Debug configuration
3. Runs Python-based integration tests and ctest unit tests

Unit test executables are defined in CMakeLists.txt:108-149 and located in `src/tests/`.

## Common Usage Workflows

### Creating and Running a Database

**RDF Model Example:**
```bash
# 1. Import data
build/Release/bin/mdb import data/example/rdf/cora/cora.ttl data/dbs/rdf/cora

# 2. Start server
build/Release/bin/mdb server data/dbs/rdf/cora

# 3. Query via web interface: http://localhost:4321
# Or via HTTP: http://localhost:1234/sparql
# Or use the query script:
echo "SELECT * WHERE { ?s ?p ?o . } LIMIT 10" > my_query.rq
bash scripts/query my_query.rq
```

**Quad Model Example:**
```bash
build/Release/bin/mdb import data/example/qm/cora/cora.qm data/dbs/qm/cora
build/Release/bin/mdb server data/dbs/qm/cora
echo "MATCH (?from)-[?edge :?type]->(?to) RETURN * LIMIT 10" > my_query.rq
bash scripts/query my_query.rq
```

**GQL Example:**
```bash
build/Release/bin/mdb import data/example/gql/posts/posts.gql data/dbs/gql/posts
build/Release/bin/mdb server data/dbs/gql/posts
echo "MATCH (from)-[edge]->(to) RETURN * LIMIT 10" > my_query.rq
bash scripts/query my_query.rq
```

### MDB Subcommands

**Server:**
```bash
build/Release/bin/mdb server <db_folder> [OPTIONS]
# Options: --port (default: 1234), --timeout (default: 60s), --threads,
#          --browser (default: true), --browser-port (default: 4321),
#          buffer size options, admin credentials
```

**Import:**
```bash
# From files
build/Release/bin/mdb import <files...> <db_folder> [OPTIONS]
# Formats: .ttl, .nt, .n3, .rdf (RDF), .gql (GQL), .qm (Quad Model)

# From stdin (useful for compressed files)
bzcat file.ttl.bz2 | build/Release/bin/mdb import <db_folder> --format ttl

# Options: --buffer-strings, --buffer-tensors, --prefixes (RDF),
#          --btree-permutations (RDF: 3, 4, or 6, default 4)
```

**CSV Import:**
```bash
build/Release/bin/mdb csv-import <gql|quad> <db_folder> --nodes <files> --edges <files>
# Options: --buffer-strings, --buffer-tensors, --list-separator
```

**CLI (Interactive):**
```bash
build/Release/bin/mdb cli <db_folder> [OPTIONS]
```

**Dump (Export):**
```bash
build/Release/bin/mdb dump <db_folder> <output_prefix> <format>
# RDF formats: nt, ttl
# Quad Model formats: qm, json
# GQL: Not yet supported
```

**Query Script:**
```bash
./scripts/query <query_file> [CSV|TSV|JSON|XML]
# Sends POST request to http://localhost:1234/sparql
```

## Code Architecture

### High-Level Directory Structure

- **src/bin/**: Entry point (`mdb.cc`) with command-line argument parsing
- **src/cli/**: Interactive CLI implementation
- **src/graph_models/**: Core graph model implementations
  - `common/`: Shared code across models
  - `rdf_model/`: RDF/SPARQL implementation
  - `quad_model/`: Quad Model/MQL implementation
  - `gql/`: GQL implementation
  - `object_id.h`: Central ObjectId type (64-bit with 8-bit type prefix)
- **src/storage/**: Persistent storage layer
  - `index/`: B+tree implementations and index structures
  - `page/`: Page-based storage management
  - `catalog/`: Database metadata and model identification
  - `dictionary/`: External string storage
  - `tuple_collection/`: Intermediate result storage
- **src/query/**: Query processing pipeline
  - `parser/`: ANTLR4-based parsers (separate for SPARQL, MQL, GQL)
  - `executor/`: Iterator-based query execution (`binding_iter/`)
  - `optimizer/`: Query optimization
  - `rewriter/`: Query rewriting passes
  - `update/`: Update operations
- **src/import/**: Data importers for each model and format
- **src/network/**: HTTP server and client protocol handling
- **src/system/**: System initialization and resource management
- **src/third_party/**: Embedded dependencies (serd, murmur3, xxHash, dragonbox)
- **third_party/**: External dependencies (ANTLR4 runtime, Boost headers)
- **tests/**: Integration test suites (sparql/, mql/, gql/)
- **MillenniumDB.wiki/**: Comprehensive documentation (cloned as submodule)

### Key Architectural Concepts

**ObjectId System (src/graph_models/object_id.h):**
All database values are represented as 64-bit ObjectIds:
- 8 bits: Type information (4-bit generic type, 2-bit subtype, 2-bit modifier)
- 56 bits: Value payload

Type encoding supports:
- Storage modes: inline (small values), external (dictionary reference), tmp (temporary results)
- Type hierarchy: NULL, anonymous nodes, named nodes, IRIs, strings (simple/XSD/language/datatype), numerics (int/decimal/float/double), datetime types, booleans, edges, paths, tensors, lists, dictionaries, GQL-specific types (nodes, directed/undirected edges, labels, keys)

**Model Isolation:**
Each graph model (RDF, Quad, GQL) is completely separate:
- Different import formats and logic
- Different query parsers (ANTLR grammars in `src/query/parser/grammar/`)
- Different query execution strategies
- Shared storage layer with different indexing strategies (e.g., RDF uses SPO/POS/OSP permutations)

**Storage Layer:**
- Custom B+tree implementations for all indexes
- Page-based storage with buffer management
- String dictionary for external string storage (strings > inline threshold)
- Model-specific catalog stored in database directory

**Query Execution:**
- Iterator-based volcano model
- Binding iterators in `src/query/executor/binding_iter/` produce result rows
- Worst-case optimal join algorithms (WCO joins for SPARQL)
- Regular path query evaluation using specialized data structures

**Tensor Support:**
MillenniumDB supports tensor operations (floats and doubles) as first-class types. See MillenniumDB.wiki/Working-with-tensors.md for details.

## Development Notes

- **Language:** C++17 with `-std=c++17` required
- **Compiler flags:** `-march=native` for CPU-specific optimizations, `-fno-operator-names` to avoid conflicts
- **Debug builds:** Include AddressSanitizer and UndefinedBehaviorSanitizer (`-fsanitize=undefined,address`)
- **Release builds:** `-O3` optimization with optional IPO (interprocedural optimization)
- **Profile builds:** Link with tcmalloc and gperftools for profiling, require `PROFILE=ON` flag
- **CLI:** Can be disabled at compile time with `-DNO_MDB_CLI` definition
- **Parallel compilation:** CMake automatically uses (NUM_CORES - 1) for parallel builds

## Documentation References

- Main wiki: `MillenniumDB.wiki/` (GitHub wiki cloned as Git submodule)
- Setup guide: `MillenniumDB.wiki/Setup.md`
- Usage examples: `MillenniumDB.wiki/Creating-and-running-a-database.md`
- Model documentation: `MillenniumDB.wiki/Database-models.md`, `Quad-Model.md`, `MQL.md`, `GQL*.md`
- SPARQL status: `MillenniumDB.wiki/SPARQL-Implementation-Status.md`
- Example data: `data/example/{rdf,qm,gql}/`
