# PROJECT <alias> AS { ... } status

MillenniumDB's GQL dialect now parses and executes `PROJECT <alias> AS { ... }` blocks. The binder
still enforces that the enclosed subquery is read-only, terminates with `RETURN`, and exposes at
least one node or edge binding (aliases derived from node/edge expressions are accepted). Violations
raise a `QuerySemanticException` with a descriptive message.

At planning time the property-graph pipeline lowers `OpProject` into a `ProjectGraph` binding
iterator that runs the subquery once, materializes its node/edge output into an in-memory
`VirtualGraph`, and registers it under `<alias>` for the remainder of the statement list. `FROM
<alias>` prefixes on `MATCH` clauses activate the virtual graph: the planner wraps the graph pattern
with a `VirtualGraphFilter`, restricting node and edge bindings to the previously materialized
subgraph. Subsequent clauses can reuse the yielded bindings just like they would against the default
database graph.

**Limitations**

* Correlated PROJECT blocks (referencing outer variables) remain unsupported; each block executes
  once per query.
* Projected properties beyond node/edge identifiers are currently ignored when constructing the
  virtual graph.
* Streaming executors continue to reject PROJECT statements.
