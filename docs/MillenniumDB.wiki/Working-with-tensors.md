# Working with tensors

As you may notice this dataset contains tensors by the predicate `ex:embedding` in RDF and property `embedding` in MQL. In MillenniumDB we support 1-dimensional tensors natively, and they can be declared with the literal with datatype syntax:

```sparql
# SPARQL
"[1.0, 1.5, 3.2]"^^mdbtype:tensorFloat
```

```sparql
// MQL
tensorFloat("[1.0, 1.5, 3.2]")
```

You can read further about our tensor implementation in this document.

Using the same database from the previous step, we can operate this tensors with all the arithmetic operations as you would expect from a vectorial engine.

For example, imagine that we would like to get the top-100 most similar Academic Articles given a fixed article with id `101811` by cosine distance:

```sparql
# SPARQL
PREFIX bibo: <http://purl.org/ontology/bibo/>
PREFIX ex: <http://example.com/>

SELECT ?similarArticle ?distance
WHERE {
  BIND(ex:101811 AS ?fixedArticle)
  ?fixedArticle   ex:embedding ?fixedEmbedding .
  ?similarArticle a bibo:AcademicArticle .
  ?similarArticle ex:embedding ?similarEmbedding .
  BIND(mdbfn:cosineDistance(?fixedEmbedding, ?similarEmbedding) AS ?distance)
}
ORDER BY ?distance
LIMIT 100
```

```plaintext
// MQL
LET ?fixedArticle = art101811
MATCH (?similarArticle :AcademicArticle)
LET ?distance = COSINE_DISTANCE(?fixedArticle.embedding, ?similarArticle.embedding)
ORDER BY ?distance
RETURN ?similarArticle, ?distance
LIMIT 100
```

This queries may run instantly because the instance has a low amount of tensors, but if you have millions of tensors, this may not be the ideal query for your case. Heres where the indexes come into play. An update query for creating an HNSW tensor index would be something like the following:

```sparql
# SPARQL
PREFIX ex: <http://example.com/>
CREATE HNSW INDEX "my_index" WITH {
    "predicate"     = ex:embedding,
    "dimension"     = 1433,
    "maxCandidates" = 128,
    "maxEdges"      = 16,
    "metric"        = "cosineDistance"
}
```

```plaintext
// MQL
CREATE HNSW INDEX "my_index" WITH {
    "property"      = "embedding",
    "dimension"     = 1433,
    "maxCandidates" = 128,
    "maxEdges"      = 16,
    "metric"        = "cosineDistance"
}
```

As updates are restricted in the frontend, you can execute and update via curl:

```bash
# SPARQL update
curl -H "Content-Type:application/sparql-update" \
    --data "PREFIX ex: <http://example.com/>
            CREATE HNSW INDEX \"my_index\" WITH { \
                \"predicate\"     = ex:embedding, \
                \"dimension\"     = 1433, \
                \"maxCandidates\" = 128, \
                \"maxEdges\"      = 16, \
                \"metric\"        = \"cosineDistance\" \
            }" \
    -X POST http://localhost:1234/update
```

```bash
# MQL update
curl \
    --data "CREATE HNSW INDEX \"my_index\" WITH { \
                \"property\"      = \"embedding\", \
                \"dimension\"     = 1433, \
                \"maxCandidates\" = 128, \
                \"maxEdges\"      = 16, \
                \"metric\"        = \"cosineDistance\" \
            }" \
    -X POST http://localhost:1234
```

After the index creation, you can do the same query in a ANN-manner:

```sparql
# SPARQL
PREFIX bibo: <http://purl.org/ontology/bibo/>
PREFIX ex: <http://example.com/>

SELECT ?similarArticle ?distance
WHERE {
  BIND(ex:101811 AS ?fixedArticle)
  ?fixedArticle ex:embedding ?fixedEmbedding .
  HNSW_TOP_K("my_index", ?fixedEmbedding, 100, 1000) AS (?similarArticle, ?distance)
}
```

```plaintext
// MQL
LET ?fixedArticle = art101811
CALL HNSW_TOP_K("my_index", ?fixedArticle.embedding, 100, 1000)
     YIELD ?object AS ?similarArticle, ?distance
RETURN ?similarArticle, ?distance
```

For removing a HNSW index, make sure that the database is not running and remove its directory:

```bash
# RDF
rm -r data/dbs/rdf/YOUR_DB_NAME/hnsw_index/my_index
```

```bash
# Quad Model
rm -r data/dbs/qm/YOUR_DB_NAME/hnsw_index/my_index
```

It is recommended to tweak the HNSW construction and querying index arguments given the specific database instance.
