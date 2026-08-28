# Docker

We supply a Dockerfile to build and run MillenniumDB using Docker.

## Build the Docker Image

To build a Docker image of MillenniumDB run the following:

```bash
docker build -t mdb .
```

## Creating a Database

Put any `.ttl` files into the `data` directory and from the repository root directory run:

```bash
docker run --rm --volume "$PWD"/data:/data mdb \
    mdb import \
    /data/example/rdf/cora/cora.ttl  \
    /data/dbs/rdf/cora
```

You can change `/data/example/rdf/cora.ttl` to the path of of your `.ttl` and
`/data/dbs/rdf/cora` to the directory where you want the database to be created. The `.ttl` files and database directories have to be inside `data`. The `.ttl` file must not be a symbolic link to a `.ttl` file but a real one. Also the `.ttl` file must exist or else the DB will be created empty.

## Running a Server

To run the server with the previously created database use:

```bash
docker run --rm --volume "$PWD"/data:/data -p 1234:1234 -p 4321:4321 mdb \
    mdb server /data/dbs/rdf/cora
```

## Executing a Query

Go to <http://localhost:4321/> to see the web interface (available while running the server).

Also we provide a script to make queries using the console:

```bash
bash scripts/query <query-file>
```

## Remove the Database

To remove the database that was created just delete the directory:

```bash
rm -r data/example-rdf-database
```

Depending on your Docker configuration you may have to use sudo.
