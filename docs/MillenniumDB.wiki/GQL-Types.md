# Types

Each variable in a GQL query has a type. The type of variables for node and edge patterns are defined like this:

* The type of a node is node.
* The type of an edge is edge.
* The type of a path is path.

<!-- * The ISO also mentions binding table as the type of a binding table. -->

When a variable is mentioned in a where clause or a search condition, it is said to be a reference. The **degree of reference** is defined as follows.

* The degree of reference of a variable in a union or under the `?` quantifier is conditional singleton.
* The degree of reference of a variable in a repetition is group.
* Otherwise, the degree of reference is unconditional singleton.

The degree of reference of a variable depends on the scope from which they are being referenced.

If a union declares a variable in its both operands and in at least one of them, the variable has a group degree of reference, then the union exposes the variable with a group degree of reference.

## Well-defined queries

Well-defined queries are queries that do not present any inconsistencies regarding to the types of variables. A query that is not well-defined cannot be evaluated. For a query to be well-defined, it must follow these rules.

* In a concatenation of patterns, the variables appearing in both sides are exposed as unconditional singleton.
* In a graph pattern, if more than one path pattern contains the same variable, then the variables are exposed as unconditional singleton.
* Path variables and subpath variables cannot be quantified.

## Examples

```txt
MATCH (a:Reader)(-[e:Follows]-> WHERE e.since = "2021-06-16")+(b)
RETURN e
```

In this query, the variables `a` and `b`  are of node type. The variable `e` is of edge type, it has unconditional singleton degree of reference when it is referenced in the where clause and it has group degree of reference when it is referenced in the return statement.

```txt
MATCH p = (a)~[e]~(a)
RETURN a, p
```

In this query, the variable `a` is of type node, the variable `e` is of type edge, the variable `p` is of type path. The variable `a` appears twice, but it has singleton degree of reference, since it binds to a single value.

```txt
MATCH (a)-[a]->
RETURN a
```

This query is not well-defined, since it the variable `a` is declared twice with a different type.

```txt
MATCH ((:Reader)-[b:Follows]->{,2}(:Reader) | (:Author)~[b:Knows]~(:Author))
RETURN b
```

In this query, the reference of the variable `b` in the return statement has a group degree of reference. The variable `b` is declared in both operands of the path union, but in the first operand it has a group degree of reference.
