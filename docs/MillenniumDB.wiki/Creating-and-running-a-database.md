
# Creating and running a database

In this page, we show an example of the basic usage of MillenniumDB. First, we need to build MillenniumDB as it is explained in [[Setup]].

If you are using Docker, see the page [[Docker]].

Some steps have different commands for each model, so make sure to be consistent.


## GQL

### 1. Create database

To create a database, run this command:

```bash
build/Release/bin/mdb import data/example/gql/posts/posts.gql data/dbs/gql/posts
```

### 2. Start the server

After creating the database, the server can be started with the following command:

```bash
build/Release/bin/mdb server data/dbs/gql/posts
```

### 3. Querying

For querying MillenniumDB you have the following alternatives:

- Directly from the web interface at <http://localhost:4321>

- Via HTTP request at <http://localhost:1234>. You can do this using `curl`:

```bash
curl -H "Accept:text/csv" \
    --data "MATCH (from)-[edge]->(to) RETURN * LIMIT 10" \
    -X POST http://localhost:1234
```

- Via [MillenniumDB Driver Javascript](https://github.com/MillenniumDB/MillenniumDB-driver-javascript)

You can also run this script to pass a file containing a query.

```bash
echo "MATCH (from)-[edge]->(to) RETURN * LIMIT 10" > my_query.rq
bash scripts/query my_query.rq
```
