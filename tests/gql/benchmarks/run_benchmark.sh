#!/bin/bash
# Automated benchmark runner for MillenniumDB projection performance
# Usage: ./tests/gql/benchmarks/run_benchmark.sh

set -e

# Configuration
DB_PATH="data/dbs/gql/posts"
SERVER_PORT=1234
BENCHMARK_SCRIPT="tests/gql/benchmarks/projection_benchmark.py"

echo "================================================================"
echo "MillenniumDB Projection Benchmark - Automated Runner"
echo "================================================================"
echo ""

# Check if database exists
if [ ! -d "$DB_PATH" ]; then
    echo "Creating test database at $DB_PATH..."
    build/Release/bin/mdb import data/example/gql/posts/posts.gql "$DB_PATH"
    echo "✓ Database created"
    echo ""
fi

# Check if Python dependencies are installed
if ! python3 -c "import requests" 2>/dev/null; then
    echo "Installing Python dependencies..."
    pip3 install requests
    echo ""
fi

# Check if server is already running
if curl -s "http://localhost:$SERVER_PORT" > /dev/null 2>&1; then
    echo "Server is already running at port $SERVER_PORT"
    echo "Using existing server instance"
    echo ""

    # Run benchmark with existing server
    python3 "$BENCHMARK_SCRIPT"
    exit $?
fi

# Start server in background
echo "Starting MillenniumDB server..."
echo "  Database: $DB_PATH"
echo "  Port: $SERVER_PORT"
echo ""

# Start server with output redirected
build/Release/bin/mdb server "$DB_PATH" --port "$SERVER_PORT" > /tmp/mdb_benchmark_server.log 2>&1 &
SERVER_PID=$!

echo "Server started (PID: $SERVER_PID)"
echo ""

# Wait for server to be ready
echo "Waiting for server to initialize..."
MAX_WAIT=30
ELAPSED=0

while ! curl -s "http://localhost:$SERVER_PORT" > /dev/null 2>&1; do
    sleep 1
    ELAPSED=$((ELAPSED + 1))

    if [ $ELAPSED -ge $MAX_WAIT ]; then
        echo "✗ Server failed to start within $MAX_WAIT seconds"
        echo "Server log:"
        tail -20 /tmp/mdb_benchmark_server.log
        kill $SERVER_PID 2>/dev/null || true
        exit 1
    fi

    # Check if server process is still running
    if ! kill -0 $SERVER_PID 2>/dev/null; then
        echo "✗ Server process died"
        echo "Server log:"
        cat /tmp/mdb_benchmark_server.log
        exit 1
    fi
done

echo "✓ Server ready (took ${ELAPSED}s)"
echo ""

# Run benchmark
echo "Running benchmark suite..."
echo ""

python3 "$BENCHMARK_SCRIPT"
BENCHMARK_EXIT_CODE=$?

echo ""
echo "Shutting down server..."
kill $SERVER_PID 2>/dev/null || true
wait $SERVER_PID 2>/dev/null || true
echo "✓ Server stopped"

# Clean up log file
rm -f /tmp/mdb_benchmark_server.log

echo ""
if [ $BENCHMARK_EXIT_CODE -eq 0 ]; then
    echo "================================================================"
    echo "✅ Benchmark completed successfully!"
    echo "================================================================"
else
    echo "================================================================"
    echo "✗ Benchmark failed with exit code $BENCHMARK_EXIT_CODE"
    echo "================================================================"
fi

exit $BENCHMARK_EXIT_CODE
