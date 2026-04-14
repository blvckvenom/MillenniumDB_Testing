

# Statements

GQL queries are composed by a sequence of statements. These statements are evaluated one by one and each of them updates the working table, which is a temporary table that stores partial results.

The statements are the following.

* `RETURN`
* `MATCH`
* `LET`
* `FOR`
* `GROUP BY`
* `OFFSET`
* `LIMIT`

## RETURN

The return statement evaluates expressions and returns them as results.

### Examples

In this query, we pass a numerical constant as an expression.

```txt
RETURN 100
```

In this query, we pass a mathematical expression.

```txt
RETURN (1 + 2 + 3 + 4) / 4
```

In this query, we pass a string.

```txt
RETURN "Hello GQL!"
```

We can also return multiple expressions and use the keyword `AS` to define aliases.

```txt
RETURN 100 AS number, "Hello GQL!" AS message
```

These examples only return constants, which does not seem to be that useful. To search for nodes and edges in the graph database, we need the statement `MATCH`.

## GROUP BY

The return statement can also contain a group by clause. This clause groups the records by the specified items. The items in the group by clause are called grouping elements.

For a query that contains a group by clause to be valid, for each return item we have two cases:

* The alias of the return item is a grouping element.
* The alias of the return item is not a grouping element and the return item contains only aggregate functions or variables that are grouping elements.

Also, a query that has simply an asterisk (*) in the return statement cannot contain a group by clause.

### Examples

For example, in this query we obtain the average age of authors for each location. The GQL Standard does not allow grouping by a property like `a.location`, so we need to use an alias like `location`.

```txt
MATCH (a:Author)
RETURN a.location AS location, AVG(a.age) GROUP BY location
```

We can also group by an element pattern. In this query, we group by the node variable `a` to get the number of books published by each author.

```txt
MATCH (a:Author)-[:Published]->
RETURN a.name, COUNT(*) AS n_published GROUP BY a
```

The following query is not valid, because `a.age` it does not have an alias, it is not an aggregate function and `a` is not a grouping element.

```txt
MATCH (a:Author)
RETURN a.age GROUP a.name
```

And this query is also invalid since the return statement has an asterisk and the group by clause.

```txt
MATCH (a:Author)
RETURN * GROUP a
```

## MATCH

The match statement defines a graph pattern to match. It can also have a where clause to filter results.

### Examples

In this query, we declare variables in graph pattern and return them in the return statement.

```txt
MATCH (a:Author)-[:Published]->(b:Book)
RETURN a, b
```

We can also obtain the properties of an element.

```txt
MATCH (a:Author)
RETURN a.name, a.age
```

We can add a where clause in the match statement to filter results.

```txt
MATCH (a: Author) WHERE a.age <= 20
RETURN a.name, a.age
```

## LET

The let statement allows us to define variables to use in other statements. It adds a column to the working table.

### Examples

In this query, we define the variables `x` and `y`.

```txt
LET x = 1, y = 2
RETURN x, y
```

We can define variables and use them in the match statement.

```txt
LET author_name = "Jane Doe"
MATCH (a:Author {name: author_name})
RETURN a
```

We can define expressions as well.

```txt
MATCH (a:Author)~(b:Author)
LET age_avg = (a.age + b.age) / 2
RETURN a.name, b.name, age_avg
```

## FOR

The for statement is used to unnest a list or a group variable and join it with the working table.

### Examples

In this example, we define a list and return each element as a different result.

```txt
FOR elem IN [1, 2, 3]
RETURN elem
```

In this example, we join the elements of the list with the authors. As a result, we obtain 3 results for each author.

```txt
MATCH (a:Author)
FOR elem IN [1, 2, 3]
RETURN a, elem
```

In this example, the variable `e` has a group degree of reference, so `e` can bind to one or more edges (see section [[Types|GQL Types]]). Using a for statement, we get each edge as a different result.

```txt
MATCH ()-[e]->{1,4}()
FOR elem IN e
RETURN elem
```

## FILTER

The filter statement allows us to filter records in the working table that do not pass the expression conditions.

### Examples

In this example, we filter the results where `a` is equal to `b`.

```txt
MATCH (a:Author)~(b:Author)
FILTER a <> b
RETURN a, b
```

We can use the let statement to define variables and use them in the filter statement.

```txt
MATCH (b:Book)
LET b_rating = b.rating
FILTER b_rating >= 7
RETURN b
```

## ORDER BY

The order by statement sorts the working table by the specified expressions.

### Examples

In this query, we obtain all the authors sorted by name.

```txt
MATCH (a:Author)
ORDER BY a.name
RETURN a.name
```

By default, the sort is ascendent. We can explicitly specify the order with `ASC` and `DESC`.

```txt
MATCH (a:Author)
ORDER BY a.name DESC
RETURN a
```

If there are null results, we can specify whether they appear first or last using `NULLS FIRST` and `NULLS LAST`. In this example, books don't have the property name, so the result will contain null values.

```txt
MATCH (a:Author) | (:Book)
ORDER BY a.name DESC NULLS LAST
RETURN a.name
```

## OFFSET

The offset statement skips the specified number of records in the working table. It is equivalent to the keyword `SKIP`.

### Examples

In this query, we skip the first 3 results obtained.

```txt
MATCH (a:Author)
OFFSET 3
RETURN a
```

In this query, we skip the 3 youngest authors using the order by statement.

```txt
MATCH (a:Author)
ORDER BY a.age ASC
SKIP 3
RETURN a.name, a.age
```

## LIMIT

The limit statement limits the number of records in the working table.

### Examples

In this query, we obtain at most 10 authors.

```txt
MATCH (a:Author)
LIMIT 10
RETURN a
```
