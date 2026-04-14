# Paths

GQL queries allows us to reference and return paths. For this, we use path variables.

## Path variables

Path variables allows us to reference paths. For example, we can return all edges and their source and destination nodes using the variable `p`.

```txt
MATCH p = ()-[]->()
RETURN p
```

There are functions that we can use in path variables. For example, `PATH_LENGTH` or `ELEMENTS`.

```txt
MATCH p = ({name:"John Doe"})->(:Book)
RETURN PATH_LENGTH(p)
```

Using the `PATH_LENGTH` function does is not really useful in this query, because the path has fixed length. Sometimes we want to find more complex paths, so then we can use unions, multiset alternation and quantifiers.

## Parenthesized paths

Paths can have parenthesized subpaths.

For example, in this query we have the author node and the edge in a parenthesis.

```txt
MATCH ((a:Author)-[:Published]->)(:Book)
RETURN a
```

Parenthesized path patterns can also contain a where clause.

```txt
MATCH ((e:Editor)~(a:Author) WHERE e.age = a.age)->(:Book)
RETURN a, e
```

This will be useful when using the union operator and quantifiers.

## Union

We can use the path pattern union operator to combine the results from its operands.

For example, we can obtain in a single query all books with their authors and authors that know each other.

```txt
MATCH p = (:Author)->(:Book) | (:Author)~[:Knows]~(:Author)
RETURN p
```

In this example, we use the union operator inside a parenthesis.

```txt
MATCH p = (:Author)~[:Knows]~((:Author) | (:Editor))
RETURN p
```

## Multiset Alternation

The multiset alternation operator combine the results similarly to the union operator, but duplicate results are removed from the working table.

For example, we can use the union operator `|` to obtain all authors and all editors.

```txt
MATCH (a:Author) | (a:Editor)
RETURN a
```

When using the union operator `|`, duplicate results are discarded. If we want to get all the results including duplicates, then we can use the multiset alternation operator `|+|`.

```txt
MATCH (a:Author) |+| (a:Editor)
RETURN a
```
