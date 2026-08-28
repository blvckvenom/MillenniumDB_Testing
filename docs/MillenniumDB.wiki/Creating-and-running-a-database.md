
# Creating and running a database

In this page, we show an example of the basic usage of MillenniumDB. First, we need to build MillenniumDB as it is explained in [[Setup]].

If you are using Docker, see the page [[Docker]].

Some steps have different commands for each model, so make sure to be consistent.

---

## GQL

### 1. Create database

To create a database, run this command:

```bash
build/Release/bin/mdb import data/example/gql/posts/posts.gql data/dbs/gql/posts
```

### 2. Start the server

After creating the database, the server can be started with the following command:

```bash
build/Release/bin/mdb server data/dbs/gql/posts
```

### 3. Querying

For querying MillenniumDB you have the following alternatives:

- Directly from the web interface at <http://localhost:4321>

- Via HTTP request at <http://localhost:1234>. You can do this using `curl`:

```bash
curl -H "Accept:text/csv" \
    --data "MATCH (from)-[edge]->(to) RETURN * LIMIT 10" \
    -X POST http://localhost:1234
```

- Via [MillenniumDB Driver Javascript](https://github.com/MillenniumDB/MillenniumDB-driver-javascript)

You can also run this script to pass a file containing a query.

```bash
echo "MATCH (from)-[edge]->(to) RETURN * LIMIT 10" > my_query.rq
bash scripts/query my_query.rq
```

---

## RDF Model

### 1. Create database

```bash
build/Release/bin/mdb import data/example/rdf/cora/cora.ttl data/dbs/rdf/cora
```

### 2. Start the server

```bash
build/Release/bin/mdb server data/dbs/rdf/cora
```

### 3. Querying

- Web interface: <http://localhost:4321>
- HTTP endpoint: <http://localhost:1234/sparql>

```bash
echo "SELECT * WHERE { ?s ?p ?o . } LIMIT 10" > my_query.rq
bash scripts/query my_query.rq
```

---

## Quad Model (MQL)

### 1. Create database

```bash
build/Release/bin/mdb import data/example/qm/cora/cora.qm data/dbs/qm/cora
```

### 2. Start the server

```bash
build/Release/bin/mdb server data/dbs/qm/cora
```

### 3. Querying

```bash
echo "MATCH (?from)-[?edge :?type]->(?to) RETURN * LIMIT 10" > my_query.rq
bash scripts/query my_query.rq
```

---

## MDB Subcommands

### Server

Start the database server:

```bash
build/Release/bin/mdb server <db_folder> [OPTIONS]
```

**Options:**
| Option | Default | Description |
|--------|---------|-------------|
| `--port` | 1234 | HTTP server port |
| `--timeout` | 60s | Query timeout |
| `--threads` | auto | Worker threads |
| `--browser` | true | Enable web interface |
| `--browser-port` | 4321 | Web interface port |

### Import

Import data from files:

```bash
# From files
build/Release/bin/mdb import <files...> <db_folder> [OPTIONS]
```

**Supported formats:**
- RDF: `.ttl`, `.nt`, `.n3`, `.rdf`
- GQL: `.gql`
- Quad Model: `.qm`

**From stdin (useful for compressed files):**

```bash
bzcat file.ttl.bz2 | build/Release/bin/mdb import <db_folder> --format ttl
```

**Options:**
| Option | Description |
|--------|-------------|
| `--buffer-strings` | String buffer size |
| `--buffer-tensors` | Tensor buffer size |
| `--prefixes` | RDF prefix file |
| `--btree-permutations` | RDF only: 3, 4, or 6 (default: 4) |

### CSV Import

Import from CSV files:

```bash
build/Release/bin/mdb csv-import <gql|quad> <db_folder> --nodes <files> --edges <files>
```

**Options:**
| Option | Description |
|--------|-------------|
| `--buffer-strings` | String buffer size |
| `--buffer-tensors` | Tensor buffer size |
| `--list-separator` | List value separator |

### CLI (Interactive)

Start interactive command-line interface:

```bash
build/Release/bin/mdb cli <db_folder> [OPTIONS]
```

### Dump (Export)

Export database contents:

```bash
build/Release/bin/mdb dump <db_folder> <output_prefix> <format>
```

**Supported formats:**
| Model | Formats |
|-------|---------|
| RDF | `nt`, `ttl` |
| Quad Model | `qm`, `json` |
| GQL | Not yet supported |

### Query Script

Send queries to running server:

```bash
./scripts/query <query_file> [CSV|TSV|JSON|XML]
```

Sends POST request to `http://localhost:1234/sparql`.
