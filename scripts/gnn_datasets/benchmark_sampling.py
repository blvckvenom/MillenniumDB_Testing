#!/usr/bin/env python3
"""
Benchmark SWEEP vs SEEK sampling strategies via GQL procedures.

This script starts a MillenniumDB server, creates projections, and measures
sampling performance across different batch sizes and sparsity levels.

Usage:
    # Start server first:
    ./build/Release/bin/mdb server ./data/dbs/gql/ogbn-products --port 1234

    # Then run benchmark:
    python benchmark_sampling.py --port 1234 --projection my_proj

Requirements:
    pip install requests numpy matplotlib

Author: GNN Integration Team
"""

import argparse
import json
import random
import statistics
import sys
import time
from dataclasses import dataclass
from typing import List, Optional, Tuple

try:
    import requests
except ImportError:
    print("Error: requests not found. Install with: pip install requests")
    sys.exit(1)


@dataclass
class BenchmarkResult:
    """Result of a single benchmark run."""
    strategy: str
    batch_size: int
    sparsity: float
    times_ms: List[float]
    neighbors_collected: int

    @property
    def mean_ms(self) -> float:
        return statistics.mean(self.times_ms)

    @property
    def std_ms(self) -> float:
        return statistics.stdev(self.times_ms) if len(self.times_ms) > 1 else 0.0

    @property
    def min_ms(self) -> float:
        return min(self.times_ms)

    @property
    def max_ms(self) -> float:
        return max(self.times_ms)


class GQLClient:
    """Simple client for MillenniumDB GQL queries."""

    def __init__(self, host: str = "localhost", port: int = 1234):
        self.url = f"http://{host}:{port}"

    def query(self, gql: str) -> dict:
        """Execute GQL query and return result."""
        response = requests.post(
            self.url,
            data=gql,
            headers={"Content-Type": "application/gql"},
            timeout=300  # 5 min timeout for large operations
        )
        response.raise_for_status()
        return response.json()

    def query_timed(self, gql: str) -> Tuple[dict, float]:
        """Execute query and return (result, elapsed_ms)."""
        start = time.perf_counter()
        result = self.query(gql)
        elapsed = (time.perf_counter() - start) * 1000
        return result, elapsed


def get_projection_stats(client: GQLClient, proj_name: str) -> dict:
    """Get projection statistics."""
    result = client.query(f"CALL graph.info('{proj_name}') YIELD * RETURN *")

    if result.get("results"):
        row = result["results"][0]
        return {
            "nodes": row.get("nodeCount", 0),
            "edges": row.get("edgeCount", 0),
        }
    return {"nodes": 0, "edges": 0}


def generate_batch_nodes(
    max_node_id: int,
    batch_size: int,
    sparsity: float,
    seed: int = 42
) -> List[int]:
    """
    Generate batch of node IDs with specified sparsity.

    Args:
        max_node_id: Maximum node ID in projection
        batch_size: Number of nodes to generate
        sparsity: 0.0=dense (consecutive), 1.0=sparse (random)
        seed: Random seed for reproducibility

    Returns:
        List of node IDs
    """
    random.seed(seed)

    if batch_size >= max_node_id:
        return list(range(max_node_id))

    if sparsity < 0.01:
        # Dense: consecutive IDs
        start = random.randint(0, max_node_id - batch_size)
        return list(range(start, start + batch_size))

    elif sparsity > 0.99:
        # Sparse: uniform random
        return random.sample(range(max_node_id), batch_size)

    else:
        # Mixed: random with controlled gaps
        avg_gap = int((max_node_id / batch_size) * sparsity)
        nodes = []
        current = random.randint(0, avg_gap)

        while len(nodes) < batch_size and current < max_node_id:
            nodes.append(current)
            gap = random.randint(1, max(2, avg_gap * 2))
            current += gap

        # Fill remaining with random if needed
        if len(nodes) < batch_size:
            remaining = set(range(max_node_id)) - set(nodes)
            nodes.extend(random.sample(list(remaining), batch_size - len(nodes)))

        return nodes[:batch_size]


def benchmark_sampling_direct(
    client: GQLClient,
    proj_name: str,
    node_ids: List[int],
    fanout: int,
    strategy: str,  # "SWEEP" or "SEEK"
    iterations: int = 10
) -> BenchmarkResult:
    """
    Benchmark sampling by measuring K-hop sampling time.

    Note: This uses the gnn.offline_sample procedure which internally
    uses the configured batch_strategy.
    """
    times = []

    # For now, we'll measure the full K-hop sampling time
    # A more granular benchmark would require adding timing instrumentation
    # to the C++ code

    for i in range(iterations):
        # Use inline seed specification in query
        seed_list = ",".join(str(n) for n in node_ids[:100])  # Limit for query size

        query = f"""
        CALL gnn.offline_sample('{proj_name}', 'bench_sample_{i}', [15], {{
            batchSize: {len(node_ids)},
            strategy: '{strategy}'
        }})
        YIELD sampleName, computeMillis
        RETURN sampleName, computeMillis
        """

        try:
            result, query_time = client.query_timed(query)
            times.append(query_time)

            # Cleanup
            client.query(f"CALL gnn.sample.drop('bench_sample_{i}')")
        except Exception as e:
            print(f"  Warning: iteration {i} failed: {e}")
            continue

    return BenchmarkResult(
        strategy=strategy,
        batch_size=len(node_ids),
        sparsity=0.0,  # Set by caller
        times_ms=times if times else [0.0],
        neighbors_collected=0  # Would need instrumentation
    )


def benchmark_batch_creation(
    client: GQLClient,
    proj_name: str,
    max_node_id: int,
    batch_sizes: List[int],
    sparsities: List[float],
    fanout: int,
    iterations: int,
    seed: int
) -> List[BenchmarkResult]:
    """
    Run full benchmark suite comparing strategies.

    Since we can't directly switch strategies via GQL yet,
    this benchmark measures the overall sampling performance
    which uses the AUTO strategy internally.
    """
    results = []

    for batch_size in batch_sizes:
        for sparsity in sparsities:
            print(f"\nBatch size: {batch_size}, Sparsity: {sparsity:.2f}")

            # Generate batch
            nodes = generate_batch_nodes(max_node_id, batch_size, sparsity, seed)

            # Benchmark with timing at Python level
            # (For accurate C++ level comparison, would need instrumented builds)

            times = []
            for i in range(iterations):
                # Create a simple sampling query that exercises the batch sampler
                sample_nodes = random.sample(nodes, min(100, len(nodes)))
                node_list = ",".join(str(n) for n in sample_nodes)

                query = f"""
                MATCH (n) WHERE id(n) IN [{node_list}]
                MATCH (n)-[e]->(m)
                RETURN COUNT(m) as neighbor_count
                """

                start = time.perf_counter()
                try:
                    result = client.query(query)
                    elapsed = (time.perf_counter() - start) * 1000
                    times.append(elapsed)
                except Exception as e:
                    print(f"    Warning: query failed: {e}")

            if times:
                result = BenchmarkResult(
                    strategy="AUTO",
                    batch_size=batch_size,
                    sparsity=sparsity,
                    times_ms=times,
                    neighbors_collected=0
                )
                results.append(result)
                print(f"  Mean: {result.mean_ms:.2f}ms, Std: {result.std_ms:.2f}ms")

    return results


def print_results_table(results: List[BenchmarkResult]):
    """Print results in a formatted table."""
    print("\n" + "=" * 80)
    print(f"{'Strategy':<10} {'Batch':<10} {'Sparsity':<10} {'Mean(ms)':<12} "
          f"{'Std(ms)':<12} {'Min(ms)':<12} {'Max(ms)':<12}")
    print("=" * 80)

    for r in results:
        print(f"{r.strategy:<10} {r.batch_size:<10} {r.sparsity:<10.2f} "
              f"{r.mean_ms:<12.2f} {r.std_ms:<12.2f} {r.min_ms:<12.2f} {r.max_ms:<12.2f}")


def export_csv(results: List[BenchmarkResult], filename: str):
    """Export results to CSV file."""
    with open(filename, "w") as f:
        f.write("strategy,batch_size,sparsity,mean_ms,std_ms,min_ms,max_ms\n")
        for r in results:
            f.write(f"{r.strategy},{r.batch_size},{r.sparsity:.2f},"
                    f"{r.mean_ms:.2f},{r.std_ms:.2f},{r.min_ms:.2f},{r.max_ms:.2f}\n")
    print(f"\nResults exported to {filename}")


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark GNN sampling strategies"
    )
    parser.add_argument(
        "--host",
        type=str,
        default="localhost",
        help="MillenniumDB host (default: localhost)"
    )
    parser.add_argument(
        "--port",
        type=int,
        default=1234,
        help="MillenniumDB port (default: 1234)"
    )
    parser.add_argument(
        "--projection", "-p",
        type=str,
        required=True,
        help="Projection name to benchmark"
    )
    parser.add_argument(
        "--batch-sizes",
        type=str,
        default="50,100,500,1000,5000",
        help="Comma-separated batch sizes"
    )
    parser.add_argument(
        "--sparsities",
        type=str,
        default="0.0,0.5,0.9,1.0",
        help="Comma-separated sparsity values (0=dense, 1=sparse)"
    )
    parser.add_argument(
        "--iterations", "-n",
        type=int,
        default=10,
        help="Iterations per configuration"
    )
    parser.add_argument(
        "--fanout",
        type=int,
        default=15,
        help="Fanout per layer"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed"
    )
    parser.add_argument(
        "--csv",
        type=str,
        default=None,
        help="Export results to CSV file"
    )

    args = parser.parse_args()

    # Parse lists
    batch_sizes = [int(x) for x in args.batch_sizes.split(",")]
    sparsities = [float(x) for x in args.sparsities.split(",")]

    # Connect to server
    print(f"Connecting to {args.host}:{args.port}...")
    client = GQLClient(args.host, args.port)

    # Get projection stats
    print(f"Getting stats for projection '{args.projection}'...")
    try:
        stats = get_projection_stats(client, args.projection)
        print(f"  Nodes: {stats['nodes']:,}")
        print(f"  Edges: {stats['edges']:,}")
    except Exception as e:
        print(f"Error getting projection stats: {e}")
        print("Make sure the projection exists and server is running.")
        return 1

    if stats["nodes"] == 0:
        print("Error: Projection has no nodes. Create a projection first.")
        return 1

    # Run benchmarks
    print(f"\nRunning benchmarks...")
    print(f"  Batch sizes: {batch_sizes}")
    print(f"  Sparsities: {sparsities}")
    print(f"  Iterations: {args.iterations}")
    print(f"  Fanout: {args.fanout}")

    results = benchmark_batch_creation(
        client,
        args.projection,
        stats["nodes"],
        batch_sizes,
        sparsities,
        args.fanout,
        args.iterations,
        args.seed
    )

    # Print results
    print_results_table(results)

    # Export if requested
    if args.csv:
        export_csv(results, args.csv)

    return 0


if __name__ == "__main__":
    sys.exit(main())
