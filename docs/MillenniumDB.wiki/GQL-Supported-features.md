# Supported features

## Import

We support the import of data using a CSV file or a GQL file. Both formats allows us to declare:

* Nodes
* Edges
  * Directed
  * Undirected
* Multiple labels
* Properties

## Patterns

These types of patterns can be present in a match statement.

* Node patterns
* Edge patterns
  * Full and abbreviated forms.
  * Directed and undirected edges
* Parenthesized path patterns
* Path pattern union
  * This operand maintains duplicates.
* Node variables, edge variables and path variables.
  * Subpath variables are supported but cannot be referenced.
* Element pattern predicate
  * Where clause
  * Property specification
* Graph pattern where clause

## Quantifiers

These quantifiers are supported over edges and parenthesized path patterns.

* Quantified path primary
  * Asterisk, plus, fixed quantifier and general quantifier
* Questioned path primary

Currently, path mode and path search prefixes are not supported, so a quantified pattern always returns any shortest path.

## Statement

These statements are supported.

* Match statement
  * The simple match statement is supported. The optional match statement is not supported yet.
* Let statement
* Filter statement
* Order by and page statement
  * `ASC`, `DESC` and ordering by nulls is supported.
  * `OFFSET`/`SKIP` and `LIMIT` are supported.
* Return statement
  * The return statement can contain a list of expressions or the `*` symbol.
  * Supports `DISTINCT` and `ALL`.
  * Supports the group by clause
* Call statement
  * `CALL` with procedure name and arguments is supported.
  * `YIELD` clause for selecting procedure output fields is supported.

## Expression

These are the supported expressions.

* Label expression: `&` (and), `|` (or) operators and the wildcard symbol `*` are supported.
* Logical operators: `AND`, `OR`, `NOT`, `XOR`.
* Comparison operators: `=`, `<`, `>`, `<=`, `>=`, `<>`.
* Arithmetic operators: `+`, `-`, `*`, `/`.
* Parenthesized expressions
* Functions
* String concatenation
* Case specification
* Boolean test: `IS` or `IS NOT`, with values `TRUE`, `FALSE` or `NULL`/`UNKNOWN`.

Also, the return statement can contain aggregate functions.
