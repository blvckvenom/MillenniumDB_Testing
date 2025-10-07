# CALL { ... } [YIELD ...] support roadmap

MillenniumDB now parses `CALL { ... } [YIELD ...]` blocks into a dedicated AST node and performs
basic semantic validation (the binder confirms that the subquery is read-only, terminates with
`RETURN`, and that each `YIELD` symbol is produced by the inner `RETURN`). The output bindings of the
subquery are registered in the outer scope so later statements can reference them.

Runtime support currently materializes each `CALL` subquery into a temporary in-memory table via the
`CallTable` binding iterator. The subquery executes once, its `YIELD` columns are cached, and the
outer pipeline consumes the cached rows, making the yielded variables behave like bindings produced
by a derived table. This enables subsequent statements (e.g., `MATCH`, `FILTER`, `RETURN`) to read
those bindings without re-executing the subquery. The implementation still targets read-only,
uncorrelated subqueries.

Remaining improvements under consideration include:

1. **Streaming / spill-aware materialization** – allow very large `CALL` results to stream or spill
   instead of buffering them entirely in memory.
2. **EXPLAIN output** – enrich the textual/JSON explainers with explicit `CALL` operator nodes so the
   cached subquery appears in execution plans.
3. **Correlated CALL blocks** – if future work allows references to outer variables inside the block,
   define and implement row-by-row semantics.
