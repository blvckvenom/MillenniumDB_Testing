# GQL Projection Benchmarks

Comprehensive benchmark suite for measuring projection creation performance in MillenniumDB GQL implementation.

## Overview

This benchmark suite measures **server-side** projection creation time to evaluate the B+Tree write batching optimization. Unlike CLI-based tests, server-mode eliminates the 20-22 second startup overhead, providing accurate performance measurements.

## Prerequisites

1. **Python 3.8+** with `requests` library:
   ```bash
   pip3 install requests
   ```

2. **Test database** (posts example):
   ```bash
   # If database doesn't exist, create it:
   build/Release/bin/mdb import data/example/gql/posts/posts.gql data/dbs/gql/posts
   ```

## Running Benchmarks

### Method 1: Automated (Recommended)

```bash
# Start server and run benchmark in one command
./tests/gql/benchmarks/run_benchmark.sh
```

### Method 2: Manual

```bash
# Terminal 1: Start server
build/Release/bin/mdb server data/dbs/gql/posts

# Terminal 2: Run benchmark
python3 tests/gql/benchmarks/projection_benchmark.py
```

## Benchmark Configuration

The suite runs multiple test configurations:

- **Small projection**: User → Friend (typically 50 nodes, 50 edges)
- **Medium projection**: Post → HasTag (varies by dataset)

Each configuration runs 5 iterations to calculate statistical metrics (mean, median, stddev).

## Interpreting Results

### Key Metrics

- **Mean Time**: Average projection creation time across iterations
- **Throughput**: Edges per second (higher is better)
- **Std Deviation**: Consistency across runs (lower is better)

### Expected Performance (with batching optimization)

**Baseline (before optimization)**:
- Throughput: ~50-70 edges/sec
- Mean time for 50 edges: ~0.7-1.0s

**Optimized (with sorted batching)**:
- Throughput: ~150-250 edges/sec (3-5× improvement)
- Mean time for 50 edges: ~0.2-0.3s

### Sample Output

```
======================================================================
MillenniumDB Projection Creation Benchmark
======================================================================
Server: http://localhost:1234
Timestamp: 2025-11-14T17:45:00
Iterations per test: 5

Testing: Small projection (User-Friend)
----------------------------------------------------------------------
  Iteration 1/5... ✓ 0.245s (50 nodes, 50 edges)
  Iteration 2/5... ✓ 0.238s (50 nodes, 50 edges)
  Iteration 3/5... ✓ 0.242s (50 nodes, 50 edges)
  Iteration 4/5... ✓ 0.240s (50 nodes, 50 edges)
  Iteration 5/5... ✓ 0.244s (50 nodes, 50 edges)

======================================================================
BENCHMARK RESULTS SUMMARY
======================================================================

📊 Small projection (User-Friend)
   Configuration: User → Friend
   Graph size: 50 nodes, 50 edges

   ⏱️  Performance:
      Mean time:      0.242s
      Median time:    0.242s
      Min time:       0.238s
      Max time:       0.245s
      Std deviation:  0.003s

   📈 Throughput:     206.6 edges/sec

======================================================================
✅ Benchmark completed successfully!
======================================================================
```

## Output Files

Results are saved to JSON for further analysis:

- `benchmark_results_YYYYMMDD_HHMMSS.json`: Detailed metrics in JSON format

Example:
```json
{
  "Small projection (User-Friend)": {
    "node_label": "User",
    "edge_label": "Friend",
    "iterations": 5,
    "avg_nodes": 50,
    "avg_edges": 50,
    "mean_time": 0.242,
    "median_time": 0.242,
    "min_time": 0.238,
    "max_time": 0.245,
    "stdev_time": 0.003,
    "throughput_edges_per_sec": 206.6
  }
}
```

## Troubleshooting

### Server Not Running

**Error**: `Cannot connect to server at http://localhost:1234`

**Solution**: Start the server first:
```bash
build/Release/bin/mdb server data/dbs/gql/posts
```

### Python Dependencies Missing

**Error**: `ModuleNotFoundError: No module named 'requests'`

**Solution**: Install dependencies:
```bash
pip3 install requests
```

### Database Not Found

**Error**: Server fails to start (database missing)

**Solution**: Import test data:
```bash
build/Release/bin/mdb import data/example/gql/posts/posts.gql data/dbs/gql/posts
```

## Advanced Usage

### Custom Server URL

```python
benchmark = ProjectionBenchmark(server_url="http://localhost:5000")
```

### Custom Iterations

```python
results = benchmark.run_benchmark_suite(iterations=10)
```

### Single Projection Test

```python
benchmark = ProjectionBenchmark()
metrics = benchmark.create_projection("my_proj", "User", "Friend")
print(f"Created in {metrics['elapsed']:.3f}s")
```

## Integration with CI/CD

The benchmark suite returns exit code 0 on success, 1 on failure:

```bash
python3 tests/gql/benchmarks/projection_benchmark.py
if [ $? -eq 0 ]; then
    echo "Benchmarks passed"
else
    echo "Benchmarks failed"
    exit 1
fi
```

## Performance Regression Detection

To detect performance regressions, compare results against baseline:

```bash
# Save baseline
python3 projection_benchmark.py > baseline.json

# After changes, compare
python3 projection_benchmark.py > current.json
# Use jq or Python to compare throughput metrics
```

## Related Documentation

- **M1.4 Implementation Plan**: `investigacion_project_2025_11_01/M1.4_IMPLEMENTATION_PLAN.md`
- **Week 3-8 Roadmap**: Covers B+Tree batching optimization
- **Projection Architecture**: `docs/projection_architecture.md` (if exists)

---

**Last Updated**: November 14, 2025
**Version**: 1.0
**Maintainer**: MillenniumDB Team
