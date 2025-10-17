# Graph Models

We support three different graph models, each graph model has its corresponding query language (i.e. once you import an RDF graph, you must use SPARQL to query it, our property graph query language will not work in that graph).

## RDF Model

We can import RDF graphs in `TTL` and `NT` formats, also `RDF/XML` is partially supported (we advice you to prefer `TTL`), We support most of [SPARQL](https://www.w3.org/TR/sparql11-query/), with exceptions explained in [[SPARQL Implementation Status]].

## Property Graph Model

The definition of the graph model and how to create a graph file is [[explained here|Quad Model]]. The query language is inspired on Cypher and [[defined here|MQL]].

## GQL

GQL (Graph Query Language) is a query language for [[property graphs|GQL Property graphs]].

GQL allows us to access the elements in the graph, filter and aggregate results. We can also insert new data and modify the existent data. As we are working with a property graph model, GQL can match subgraphs and paths in the graph database.

For example, a query looks like this.

```txt
MATCH (n1:Author)-[:Published]->(p:Book)
WHERE n1.name = "John Doe"
RETURN n1.name, book.title
LIMIT 50
```

This model is still in development. If you are interested, check out the [[GQL docs|GQL]].
