# Patterns

To match subgraphs in the graph database, we use patterns.

The smallest pattern category is the element pattern. There are two types of element patterns: the node pattern and the edge pattern.

A **node pattern** is an element that matches a node in a graph. In a query, a node pattern is represented like this:

```txt
()
```

An **edge pattern** is an element that matches an edge in a graph. In a query, an edge pattern is represented like this

```txt
->
```

This pattern matches a directed edge between two nodes. There are different types of edge patterns which are specified in the following table.

| Edge pattern                      | Full form | Abbreviated form   |
|-----------------------------------|-----------|--------------------|
| Edge pointing right               | `-[]->`   | `->`               |
| Edge pointing left                | `<-[]-`   | `->`               |
| Edge undirected                   | `~[]~`    | `~`                |
| Edge pointing right or undirected | `~[]~>`   | `~>`               |
| Edge pointing left or undirected  | `<~[]~`   | `<~`               |
| Edge pointing left or right       | `<-[]->`  | `<->`              |
| Any type of edge                  | `-[]-`    | `-`                |

In a GQL query, we want to search for **graph patterns** . A graph pattern is a list of path patterns. A **path pattern** is a sequence of element patterns which can be node patterns or edge patterns.

```txt
(a)-[e]->(b), (c)~[f]~(d)
```

This example has two path patterns separated by commas. The element patterns are the following.

* Node patterns: `(a)` , `(b)` , `(c)`, `(d)`
* Edge patterns: `-[e]->` , `~[f]~`

In a path pattern, the sequence of element patterns does not need to alternate between node patterns and edge patterns, and it does not need to start and end with a node. There are three cases:

1. If the sequence starts or ends with an edge, a node is inserted in that position. For example, the query `(a)->` is equivalent to `(a)->()`.
2. If there are two adjacent edges, a node is inserted between them. For example, the query `-[e1]->~[e2]~` is equivalent to `()-[e1]->()~[e2]~()`.
3. If there are two adjacent nodes, they are considered the same node. For example, in the query `(n1)(n2)`, the variables `n1` and `n2` are bound to the same node.

## Element pattern filler

An element pattern can have a filler which consists in a variable declaration, a label expression and a predicate. Each of these elements is optional.

### Variables

A **variable declaration** allows us to reference an element in other statements. For example, we can declare an edge variable `e` and the source and destination node variables `n1` and `n2`.

```txt
(n1)-[e]->(n2)
```

Since the variables are optional, a pattern like `()-[]->()` is also valid.

### Label expression

A **label expression** allows us to match elements with a certain label set. For example, if we want to define a pattern for an author, we can declare the node variable `a` and set the label `Author`.

```txt
(a:Author)
```

For example, we can define a pattern for a relationship between author and book.

```txt
(a:Author)-[:Published]->(b:Book)
```

In this example, we use the labels `Author` , `Published` and `Book` . As we are interested in the authors and the books they have published, we do not need to specify a variable for the directed edge.

Label expressions can be used with the logical operators `&` for conjunction, `|` for disjunction and `!` for negation. This allows to define a pattern like this:

```txt
(a:(Author & !Editor) | Illustrator)
```

Here we define a node that is an author and not an editor or an illustrator. We can also use the wildcard label `%`. For example, we can define an edge with any label like this:

```txt
-[:%]->
```

### Predicate

The predicate of the pattern filler can be a where clause or a property specification.

A **property specification** allows us to define the properties of an element pattern. For example, we can define a node named John Doe and declare the variable `a`.

```txt
(a {name: "Jane Doe", age: 42})
```

We can define properties for edges as well.

```txt
-[e {since: "1999-11-16"}]->
```

The property specification is useful when we need to define exact values, but it is not enough for when we need a range of values. For more complex conditions, we use the where clause.

The **where clause** allows us to filter element patterns. We use the keyword `WHERE` within the element pattern and specify the expressions. We can also use the logical operators `AND` , `OR` and `NOT` .

For example, we can define a pattern for authors born between the years 1950 and 1960.

```txt
(a:Author WHERE a.age >= 65 AND a.age <= 100)
```

Notice that we can include the variable declaration, the label expression and the where clause in an element pattern.
