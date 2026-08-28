# Path pattern prefix

When defining path patterns, we can specify a search prefix and a path mode. The **path mode** will filter results if elements are repeated in a path, and the **search prefix** will select the paths by their length.

When using a search prefix, the results are partitioned, so that each path from a partition has the same first and last node. Then, a result is selected from each partition.

Search prefix  | Description
---------------|--------------
`ALL`          | All paths. The default search prefix.
`ANY`          | Any path for each partition.
`ALL SHORTEST` | All shortest paths.
`ANY SHORTEST` | Any shortest path for each partition.
`SHORTEST k`   | The k shortest paths for each partition.

Path mode      | Description
---------------|--------------
`WALK`         | Does not filter any path. The default path mode.
`TRAIL`        | Filters paths with repeated edges.
`ACYCLIC`      | Filters paths with repeated nodes.
`SIMPLE`       | Filters paths with repeated nodes, unless the repeated nodes are the first and last nodes.

## Examples

In this query, we return paths of followers, without any repeated nodes. If this query does not contain the `ACYCLIC` keyword, then it is an invalid query.

```txt
MATCH p = ACYCLIC (a)-[:Follows]->*(b)
RETURN p
```

In this query, we return again a path of followers, but selecting all shortest paths for each pair of followers.

```txt
MATCH p = ALL SHORTEST ACYCLIC (a)-[:Follows]->*(b)
RETURN p
```

In this query, we return all trails of length 2.

```txt
MATCH p = TRAIL ()->{2}()
RETURN p
```
