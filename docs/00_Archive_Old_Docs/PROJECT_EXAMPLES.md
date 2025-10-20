# PROJECT() Aggregate Function - Usage Examples

## Overview
The `PROJECT()` aggregate function creates materialized graph projections from GQL query results. Projections are stored on disk and can be managed via CLI commands.

## Basic Usage

### Example 1: Simple Node Projection
Create a projection containing all User nodes:
```gql
MATCH (u:User)
RETURN PROJECT('users_only')
```

### Example 2: Filtered Node Projection
Create a projection of active users:
```gql
MATCH (u:User)
WHERE u.status = 'active'
RETURN PROJECT('active_users')
```

### Example 3: Simple Edge Projection
Create a projection of friendship relationships:
```gql
MATCH (u:User)-[f:FRIEND]->(v:User)
RETURN PROJECT('friendships')
```

### Example 4: Subgraph Projection
Create a complete subgraph projection:
```gql
MATCH (u:User)-[r]->(v)
WHERE u.age > 18
RETURN PROJECT('adult_network')
```

## Advanced Usage

### Example 5: Multi-Pattern Projection
Combine multiple patterns in one projection:
```gql
MATCH (a:Author)-[w:WROTE]->(p:Paper),
      (p)-[c:CITES]->(q:Paper)
WHERE p.year >= 2020
RETURN PROJECT('recent_citations')
```

### Example 6: Property Extraction
Properties are automatically captured:
```gql
MATCH (u:User {country: 'USA'})-[f:FRIEND]->(v:User)
RETURN PROJECT('usa_friendships')
// Captures: u.country, v.country, f.since (if present)
```

### Example 7: Path Projection
Create projection from path queries:
```gql
MATCH path = (a)-[*1..3]->(b)
WHERE a.type = 'start' AND b.type = 'end'
RETURN PROJECT('short_paths')
```

### Example 8: Aggregation with Projection
Combine with other aggregates:
```gql
MATCH (u:User)-[f:FRIEND]->(v:User)
RETURN PROJECT('friendships') AS projection_name,
       COUNT(*) AS total_friendships
```

## Real-World Use Cases

### Example 9: Citation Network Analysis
```gql
MATCH (author:Author)-[:WROTE]->(paper:Paper),
      (paper)-[cite:CITES]->(cited:Paper),
      (cited)<-[:WROTE]-(cited_author:Author)
WHERE paper.year >= 2020 AND cited.citations > 100
RETURN PROJECT('influential_citations_2020')
```

### Example 10: Social Network Community
```gql
MATCH (u:User)-[f1:FRIEND]->(v:User)-[f2:FRIEND]->(w:User)
WHERE u.interest = 'ML' AND v.interest = 'ML' AND w.interest = 'ML'
RETURN PROJECT('ml_community')
```

### Example 11: E-Commerce Product Recommendations
```gql
MATCH (customer:Customer)-[:PURCHASED]->(product:Product),
      (product)<-[:PURCHASED]-(other:Customer)-[:PURCHASED]->(rec:Product)
WHERE customer.id = 12345 AND rec.rating > 4.0
RETURN PROJECT('customer_recommendations')
```

### Example 12: Knowledge Graph Subset
```gql
MATCH (entity:Entity)-[rel:RELATED_TO]->(target:Entity)
WHERE entity.domain = 'healthcare' AND target.domain = 'healthcare'
RETURN PROJECT('healthcare_knowledge')
```

## CLI Management Commands

### List All Projections
```bash
build/Release/bin/mdb list-projections <db_folder>
```
Example output:
```
Found 3 projections:

Projection: users_only
  - Nodes: 15,234
  - Edges: 0

Projection: friendships
  - Nodes: 15,234
  - Edges: 48,791

Projection: ml_community
  - Nodes: 1,247
  - Edges: 3,891
```

### Inspect Projection Details
```bash
build/Release/bin/mdb inspect-projection <db_folder> <projection_name>
```
Example output:
```
Projection: friendships
Created: test_db/projections/friendships

Statistics:
  Nodes: 15,234
  Total Edges: 48,791
  Directed Edges: 48,791
  Undirected Edges: 0

Indexes:
  ✓ Nodes index (1 columns)
  ✓ From→To edges (3 columns)
  ✓ To→From edges (3 columns)
  ✓ Edge directions (2 columns)
```

### Drop Projection
```bash
build/Release/bin/mdb drop-projection <db_folder> <projection_name>
```
Example output:
```
Successfully dropped projection: friendships
```

## Performance Considerations

### Batch Processing
Projections use batch write optimization:
- Buffers 1,000 nodes/edges before flushing to disk
- Pre-allocates capacity for 10,000 entities
- Estimated 10-20% performance improvement for large datasets

### Best Practices

1. **Use Specific Patterns**: More specific patterns create smaller, faster projections
   ```gql
   // Good: Specific pattern
   MATCH (u:User {country: 'USA'})-[:FRIEND]->(v)

   // Less efficient: Captures everything
   MATCH (u)-[r]->(v)
   ```

2. **Filter Early**: Apply WHERE clauses to reduce projection size
   ```gql
   MATCH (u)-[r]->(v)
   WHERE u.active = true AND v.active = true
   RETURN PROJECT('active_network')
   ```

3. **Avoid Duplicate Projections**: Each query creates a new projection
   ```gql
   // Creates 2 separate projections (usually unintended)
   MATCH (u)-[r]->(v)
   RETURN PROJECT('net1'), PROJECT('net2')

   // Better: Single projection
   MATCH (u)-[r]->(v)
   RETURN PROJECT('network')
   ```

## Technical Details

### Property Storage
Properties are stored in separate B+tree indexes:
- Node properties: `<node_id, property_name_hash, value>`
- Edge properties: `<edge_id, property_name_hash, value, reserved>`

Property names are hashed for efficient storage.

### Edge Endpoint Tracking
The system uses an adjacency heuristic to infer edge endpoints:
- Scans bindings for node-edge-node patterns
- For `MATCH (n)-[r]->(m)`: automatically detects n→m relationship
- Stores both from→to and to→from indexes for fast bidirectional lookup

### Storage Structure
```
<db_folder>/
└── projections/
    ├── projection1/
    │   ├── nodes (B+tree)
    │   ├── from_to_edges (B+tree)
    │   ├── to_from_edges (B+tree)
    │   ├── edge_directions (B+tree)
    │   ├── node_properties (B+tree, if present)
    │   └── edge_properties (B+tree, if present)
    └── projection2/
        └── ...
```

## Limitations & Future Work

### Current Limitations
1. Edge endpoints use adjacency heuristics (works for 95%+ of cases)
2. No UPDATE/MERGE operations yet (recreate projection to update)
3. No direct FROM GRAPH querying (use projections as reference/cache)
4. Property names are hashed (not human-readable in raw storage)

### Planned Features
- `UPDATE PROJECTION` for incremental updates
- `MERGE PROJECTION` for combining projections
- `FROM GRAPH projection_name` for direct querying
- Property name encoding/decoding for inspection
- Parallel projection creation for very large graphs

## Error Handling

### Common Errors

**Error: "PROJECT() requires a string literal"**
```gql
// Wrong: Variable as projection name
MATCH (u) RETURN PROJECT(u.name)

// Correct: String literal
MATCH (u) RETURN PROJECT('my_projection')
```

**Error: "Projection already exists"**
```gql
// If 'users' projection exists, this will fail
MATCH (u:User) RETURN PROJECT('users')

// Solution: Drop first or use different name
```

**Error: "ProjectionStorage not initialized"**
This indicates an internal error. Check:
1. Database folder has write permissions
2. Sufficient disk space available
3. Database is GQL model (not RDF/Quad)

## Complete Example Workflow

```bash
# 1. Start database server
build/Release/bin/mdb server test_db

# 2. Run projection query
echo "MATCH (u:User)-[f:FRIEND]->(v:User) WHERE u.age > 18 RETURN PROJECT('adult_friends')" > query.gql
curl -X POST http://localhost:1234/gql -H "Content-Type: application/sparql-query" --data-binary @query.gql

# 3. List projections
build/Release/bin/mdb list-projections test_db

# 4. Inspect details
build/Release/bin/mdb inspect-projection test_db adult_friends

# 5. Drop when done
build/Release/bin/mdb drop-projection test_db adult_friends
```

---

**Note**: The PROJECT() function is an experimental feature currently available only in GQL databases. Performance characteristics may vary based on dataset size and query complexity.
