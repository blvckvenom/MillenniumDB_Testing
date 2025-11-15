#!/bin/bash

# Native Projection Performance Benchmark
# Compares Native vs Cypher projection performance

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
MDB_BIN="$PROJECT_ROOT/build/Release/bin/mdb"
DATA_DIR="$PROJECT_ROOT/data/benchmark"
DB_DIR="$PROJECT_ROOT/test_benchmark_db"

echo "========================================="
echo "Native Projection Performance Benchmark"
echo "========================================="
echo ""

# Check if mdb binary exists
if [ ! -f "$MDB_BIN" ]; then
    echo "Error: mdb binary not found at $MDB_BIN"
    echo "Please build Release version first: cmake -B build/Release -DCMAKE_BUILD_TYPE=Release && cmake --build build/Release"
    exit 1
fi

# Function to clean up database
cleanup_db() {
    echo "Cleaning up database..."
    rm -rf "$DB_DIR"
}

# Function to create benchmark data
create_benchmark_data() {
    local size=$1
    local filename="$DATA_DIR/benchmark_${size}.gql"

    echo "Creating benchmark data for $size graph..."
    mkdir -p "$DATA_DIR"

    cat > "$filename" << EOF
// Benchmark data: $size
// Generated automatically for performance testing

EOF

    case $size in
        "small")
            # 1K nodes, 5K edges
            node_count=1000
            edge_count=5000
            ;;
        "medium")
            # 10K nodes, 50K edges
            node_count=10000
            edge_count=50000
            ;;
        "large")
            # 100K nodes, 500K edges
            node_count=100000
            edge_count=500000
            ;;
        *)
            echo "Error: Unknown size $size"
            exit 1
            ;;
    esac

    # Generate User nodes
    echo "Generating $node_count User nodes..."
    for ((i=1; i<=node_count; i++)); do
        echo "INSERT (:User {id: $i, name: 'User_$i'})" >> "$filename"
        if (( i % 1000 == 0 )); then
            echo -ne "  Progress: $i/$node_count nodes\r"
        fi
    done
    echo ""

    # Generate KNOWS edges
    echo "Generating $edge_count KNOWS edges..."
    for ((i=1; i<=edge_count; i++)); do
        from=$((RANDOM % node_count + 1))
        to=$((RANDOM % node_count + 1))
        if [ $from -ne $to ]; then
            echo "MATCH (u1:User {id: $from}), (u2:User {id: $to}) INSERT (u1)-[:KNOWS]->(u2)" >> "$filename"
        fi
        if (( i % 1000 == 0 )); then
            echo -ne "  Progress: $i/$edge_count edges\r"
        fi
    done
    echo ""

    echo "Benchmark data created: $filename"
}

# Function to import database
import_database() {
    local size=$1
    local datafile="$DATA_DIR/benchmark_${size}.gql"

    if [ ! -f "$datafile" ]; then
        create_benchmark_data "$size"
    fi

    echo "Importing $size database..."
    cleanup_db

    time "$MDB_BIN" import "$datafile" "$DB_DIR"

    echo "Database imported successfully"
}

# Function to benchmark native projection
benchmark_native() {
    local size=$1

    echo ""
    echo "--- Native Projection Benchmark ($size) ---"

    # Start server in background
    "$MDB_BIN" server "$DB_DIR" --port 8080 --browser false &
    SERVER_PID=$!

    # Wait for server to start
    sleep 2

    # Run projection
    echo "Running CALL PROJECT('native_$size', 'User', 'KNOWS')..."

    start_time=$(date +%s%3N)

    curl -s -H "Content-Type:application/sparql-query" \
         --data "CALL PROJECT('native_$size', 'User', 'KNOWS') YIELD projectMillis RETURN projectMillis" \
         -X POST http://localhost:8080/sparql > /tmp/native_result.txt

    end_time=$(date +%s%3N)
    total_time=$((end_time - start_time))

    # Extract projectMillis from result
    project_millis=$(grep -oP '\d+' /tmp/native_result.txt | head -n 1)

    echo "Native projection time: ${project_millis}ms (internal)"
    echo "Total time (with HTTP): ${total_time}ms"

    # Stop server
    kill $SERVER_PID
    wait $SERVER_PID 2>/dev/null || true

    echo "$project_millis"
}

# Function to benchmark Cypher projection (theoretical)
benchmark_cypher() {
    local size=$1

    echo ""
    echo "--- Cypher Projection Benchmark ($size) ---"

    # Note: This is a theoretical benchmark
    # Actual Cypher projection syntax depends on implementation

    # Start server in background
    "$MDB_BIN" server "$DB_DIR" --port 8080 --browser false &
    SERVER_PID=$!

    # Wait for server to start
    sleep 2

    # Run pattern-based projection (slower)
    echo "Running pattern-based projection..."

    start_time=$(date +%s%3N)

    # Cypher-style projection query
    curl -s -H "Content-Type:application/sparql-query" \
         --data "MATCH (u1:User)-[r:KNOWS]->(u2:User) RETURN count(*) AS edge_count" \
         -X POST http://localhost:8080/sparql > /tmp/cypher_result.txt

    end_time=$(date +%s%3N)
    total_time=$((end_time - start_time))

    echo "Cypher pattern matching time: ${total_time}ms"

    # Stop server
    kill $SERVER_PID
    wait $SERVER_PID 2>/dev/null || true

    echo "$total_time"
}

# Function to run complete benchmark
run_benchmark() {
    local size=$1

    echo ""
    echo "========================================="
    echo "Benchmarking: $size graph"
    echo "========================================="

    # Import database
    import_database "$size"

    # Benchmark native projection
    native_time=$(benchmark_native "$size")

    # Benchmark Cypher projection
    cypher_time=$(benchmark_cypher "$size")

    # Calculate speedup
    if [ "$cypher_time" -gt 0 ]; then
        speedup=$(echo "scale=2; $cypher_time / $native_time" | bc)
        echo ""
        echo "========================================="
        echo "Results for $size graph:"
        echo "  Native:  ${native_time}ms"
        echo "  Cypher:  ${cypher_time}ms"
        echo "  Speedup: ${speedup}x"
        echo "========================================="
    else
        echo ""
        echo "Warning: Cypher benchmark failed"
    fi
}

# Main execution
echo "This benchmark compares Native vs Cypher projection performance"
echo "Expected speedup: 2-3x for Native projection"
echo ""

# Check if benchmark data exists
if [ ! -d "$DATA_DIR" ]; then
    echo "Creating benchmark data directory..."
    mkdir -p "$DATA_DIR"
fi

# Run benchmarks
if [ $# -eq 0 ]; then
    # Run all benchmarks
    run_benchmark "small"
    run_benchmark "medium"
    # run_benchmark "large"  # Uncomment for large dataset (takes longer)
else
    # Run specific benchmark
    run_benchmark "$1"
fi

# Cleanup
cleanup_db

echo ""
echo "========================================="
echo "Benchmark complete!"
echo "========================================="
echo ""
echo "Summary:"
echo "- Native projection uses direct B+Tree scanning"
echo "- Cypher projection uses pattern matching"
echo "- Expected speedup: 2-3x for Native"
echo ""
echo "For more detailed profiling, use:"
echo "  build/Profile/bin/mdb server test_db"
echo "  And analyze with gperftools"
