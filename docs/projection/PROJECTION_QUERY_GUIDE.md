# Projection Query Guide

## Current Status (2025-10-15)

### ✅ What Works
- **Creating projections** via `PROJECT()` function
- **Persistent storage** to disk
- **Inspecting projections** with enhanced CLI tool

### ❌ What's Not Yet Implemented
- **Querying projections** via `USE GRAPH` clause (see `USE_GRAPH_IMPLEMENTATION_PLAN.md`)

---

## How to View Projection Content

Since GQL `USE GRAPH` queries are not yet implemented, use the **enhanced projection inspector tool** to view projection data.

### Tool Location
```bash
build/Release/tests/projection_inspect_enhanced
```

### Basic Usage

```bash
# Show projection summary
projection_inspect_enhanced <db_folder> <projection_name> --summary

# Show all data (table format)
projection_inspect_enhanced <db_folder> <projection_name>

# Show only nodes
projection_inspect_enhanced <db_folder> <projection_name> --nodes

# Show only edges
projection_inspect_enhanced <db_folder> <projection_name> --edges
```

### Output Formats

#### 1. **Table Format** (default, human-readable)
```bash
projection_inspect_enhanced data/dbs/gql/posts my_projection
```

**Output:**
```
Nodes (74 total):
------------------------------------------------------------
   Index |              Node ID | Type
------------------------------------------------------------
       1 |   0xd400000000000000 | NODE
       2 |   0xd400000000000002 | NODE
       3 |   0xd400000000000003 | NODE
...
------------------------------------------------------------

Edges (75 total):
------------------------------------------------------------------------------------------
   Index |            From Node |              To Node |              Edge ID | Direction
------------------------------------------------------------------------------------------
       1 |   0xd400000000000000 |   0xd400000000000058 |   0xe00000000000001b | DIRECTED
       2 |   0xd400000000000000 |   0xd40000000000005f |   0xe00000000000002d | DIRECTED
...
------------------------------------------------------------------------------------------

==========================================
Projection Summary: my_projection
==========================================
  Total Nodes:        74
  Total Edges:        75
    Directed:         75
    Undirected:       0
==========================================
```

#### 2. **CSV Format** (for data analysis)
```bash
# Export nodes to CSV
projection_inspect_enhanced data/dbs/gql/posts my_projection --nodes --format csv > nodes.csv

# Export edges to CSV
projection_inspect_enhanced data/dbs/gql/posts my_projection --edges --format csv > edges.csv
```

**Output:**
```csv
index,node_id,type
1,0xd400000000000000,NODE
2,0xd400000000000002,NODE
3,0xd400000000000003,NODE
...
```

```csv
index,from_node,to_node,edge_id,direction
1,0xd400000000000000,0xd400000000000058,0xe00000000000001b,directed
2,0xd400000000000000,0xd40000000000005f,0xe00000000000002d,directed
...
```

#### 3. **JSON Format** (for programmatic access)
```bash
# Export to JSON file
projection_inspect_enhanced data/dbs/gql/posts my_projection --format json --output result.json
```

**Output:**
```json
{
  "projection": "my_projection",
  "node_count": 74,
  "edge_count": 75,
  "nodes": {
    "type": "nodes",
    "total": 74,
    "limit": 100,
    "nodes": [
      {
        "index": 1,
        "node_id": "0xd400000000000000",
        "type": "NODE"
      },
      ...
    ]
  },
  "edges": {
    "type": "edges",
    "total": 75,
    "limit": 100,
    "edges": [
      {
        "index": 1,
        "from_node": "0xd400000000000000",
        "to_node": "0xd400000000000058",
        "edge_id": "0xe00000000000001b",
        "direction": "directed"
      },
      ...
    ]
  }
}
```

### Advanced Options

#### Limit Output
```bash
# Show first 10 items
projection_inspect_enhanced data/dbs/gql/posts my_projection --limit 10

# Show all items (no limit)
projection_inspect_enhanced data/dbs/gql/posts my_projection --all
```

#### Decimal IDs Instead of Hex
```bash
projection_inspect_enhanced data/dbs/gql/posts my_projection --no-hex
```

**Output:**
```
index,node_id,type
1,15205959375887073280,NODE
2,15205959375887073282,NODE
...
```

#### Write to File
```bash
# Write CSV to file
projection_inspect_enhanced data/dbs/gql/posts my_projection --format csv --output data.csv

# Write JSON to file
projection_inspect_enhanced data/dbs/gql/posts my_projection --format json --output data.json
```

---

## Complete Examples

### Example 1: Create and Inspect a Projection

```bash
# Start server
build/Release/bin/mdb server data/dbs/gql/posts --port 1234 &

# Create projection of recent papers
curl -X POST http://localhost:1234/gql \
  --data "MATCH (a:Paper)-[e:CITES]->(b:Paper) WHERE a.year > 2020 RETURN PROJECT('recent_citations')"

# Inspect summary
build/Release/tests/projection_inspect_enhanced data/dbs/gql/posts recent_citations --summary

# Expected output:
# ==========================================
# Projection Summary: recent_citations
# ==========================================
#   Total Nodes:        42
#   Total Edges:        38
#     Directed:         38
#     Undirected:       0
# ==========================================
```

### Example 2: Export for Analysis

```bash
# Create projection
curl -X POST http://localhost:1234/gql \
  --data "MATCH (author:Author)-[:WROTE]->(paper:Paper) RETURN PROJECT('authorship')"

# Export edges to CSV for analysis
build/Release/tests/projection_inspect_enhanced data/dbs/gql/posts authorship \
  --edges --format csv --output /tmp/authorship_edges.csv

# Import into Python/Pandas
# import pandas as pd
# df = pd.read_csv('/tmp/authorship_edges.csv')
# print(df.head())
```

### Example 3: JSON API Integration

```bash
# Create projection
curl -X POST http://localhost:1234/gql \
  --data "MATCH (n)-[e]->(m) LIMIT 100 RETURN PROJECT('sample_graph')"

# Export as JSON
build/Release/tests/projection_inspect_enhanced data/dbs/gql/posts sample_graph \
  --format json --output /tmp/graph.json

# Use in web application
# fetch('/tmp/graph.json')
#   .then(response => response.json())
#   .then(data => {
#     console.log(`Nodes: ${data.node_count}, Edges: ${data.edge_count}`);
#     // Visualize with D3.js, vis.js, etc.
#   });
```

### Example 4: Compare Projections

```bash
# Create two projections
curl -X POST http://localhost:1234/gql \
  --data "MATCH (n:Paper) WHERE n.year = 2020 RETURN PROJECT('papers_2020')"

curl -X POST http://localhost:1234/gql \
  --data "MATCH (n:Paper) WHERE n.year = 2021 RETURN PROJECT('papers_2021')"

# Compare sizes
echo "2020:"
build/Release/tests/projection_inspect_enhanced data/dbs/gql/posts papers_2020 --summary

echo "2021:"
build/Release/tests/projection_inspect_enhanced data/dbs/gql/posts papers_2021 --summary
```

---

## All Available Options

```
Usage: projection_inspect_enhanced <db_folder> <projection_name> [options]

Options:
  --format <table|csv|json>   Output format (default: table)
  --nodes                     Show only nodes
  --edges                     Show only edges
  --summary                   Show only summary statistics
  --limit N                   Limit output to N items (default: 100)
  --all                       Show all items (no limit)
  --no-hex                    Show decimal IDs instead of hexadecimal
  --output FILE               Write output to file instead of stdout
  --help                      Show this help message
```

---

## File Formats

### CSV Schema

**Nodes CSV:**
| Column   | Type   | Description                    |
|----------|--------|--------------------------------|
| index    | int    | Sequential number (1-based)    |
| node_id  | string | Node ObjectId (hex or decimal) |
| type     | string | Always "NODE"                  |

**Edges CSV:**
| Column     | Type   | Description                        |
|------------|--------|------------------------------------|
| index      | int    | Sequential number (1-based)        |
| from_node  | string | Source node ObjectId               |
| to_node    | string | Target node ObjectId               |
| edge_id    | string | Edge ObjectId                      |
| direction  | string | "directed" or "undirected"         |

### JSON Schema

```json
{
  "projection": "string",     // Projection name
  "node_count": 0,            // Total node count
  "edge_count": 0,            // Total edge count
  "nodes": {
    "type": "nodes",
    "total": 0,               // Total nodes
    "limit": 0,               // Items returned (may be less than total)
    "nodes": [
      {
        "index": 0,           // 1-based sequential index
        "node_id": "string",  // Hex or decimal
        "type": "NODE"
      }
    ]
  },
  "edges": {
    "type": "edges",
    "total": 0,
    "limit": 0,
    "edges": [
      {
        "index": 0,
        "from_node": "string",
        "to_node": "string",
        "edge_id": "string",
        "direction": "directed|undirected"
      }
    ]
  }
}
```

---

## Troubleshooting

### Problem: "Projection not found"
```
Error: Cannot open projection: data/dbs/gql/posts/projections/my_projection/nodes.leaf
```

**Solution:** Verify the projection exists and path is correct
```bash
# List all projections
build/Release/bin/mdb list-projections data/dbs/gql/posts

# Check projection directory
ls -la data/dbs/gql/posts/projections/
```

### Problem: Empty projection (0 nodes, 0 edges)
```
Projection Summary: my_projection
Total Nodes:        0
Total Edges:        0
```

**Possible causes:**
1. Query returned no results
2. Server was not committed properly (old issue - should be fixed)
3. Projection was created but immediately deleted

**Solution:** Recreate projection and verify server is running with proper persistence

---

## Future: Querying Projections with GQL

Once `USE GRAPH` is implemented (see `USE_GRAPH_IMPLEMENTATION_PLAN.md`), you'll be able to query projections directly:

```sql
-- Future syntax (not yet implemented)
USE GRAPH my_projection
MATCH (n)-[e]->(m)
RETURN n, e, m
LIMIT 10
```

---

## Related Files

- **Implementation Plan:** `USE_GRAPH_IMPLEMENTATION_PLAN.md` - Full plan for USE GRAPH support
- **Inspector Tool:** `src/tests/projection_inspect_enhanced.cc` - Source code
- **Original Tool:** `src/tests/projection_inspect.cc` - Basic inspector (still available)
- **Projection Storage:** `src/graph_models/gql/projection/projection_storage.{h,cc}`

---

**Last Updated:** 2025-10-15
**Tool Version:** projection_inspect_enhanced 1.0
**Status:** Production Ready
