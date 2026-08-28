#!/usr/bin/env python3
"""
MillenniumDB Projection Creation Benchmark Suite

Measures server-side projection creation performance to evaluate B+Tree write batching optimization.

Usage:
    1. Start server: build/Release/bin/mdb server data/dbs/gql/posts
    2. Run benchmark: python3 tests/gql/benchmarks/projection_benchmark.py

Requirements:
    - Python 3.8+
    - requests library (pip install requests)
"""

import json
import time
import requests
import statistics
import sys
from typing import Dict, List, Tuple
from datetime import datetime


class ProjectionBenchmark:
    def __init__(self, server_url: str = "http://localhost:1234"):
        self.server_url = server_url
        self.gql_endpoint = f"{server_url}/gql"

    def execute_query(self, query: str) -> Tuple[Dict, float]:
        """
        Execute GQL query via HTTP API and measure server-side execution time.

        Returns:
            (response_data, elapsed_seconds)
        """
        headers = {"Content-Type": "application/sparql-query"}

        start_time = time.perf_counter()
        response = requests.post(self.gql_endpoint, data=query, headers=headers)
        elapsed = time.perf_counter() - start_time

        response.raise_for_status()
        return response.json(), elapsed

    def create_projection(self, name: str, node_label: str, edge_label: str) -> Dict:
        """
        Create a graph projection and measure performance.

        Returns:
            metrics dict with timing and metadata
        """
        query = f'''
            CALL graph_project(
                "{name}",
                "{node_label}",
                "{edge_label}"
            )
            YIELD graphName, nodeCount, relationshipCount
            RETURN graphName, nodeCount, relationshipCount
        '''

        try:
            data, elapsed = self.execute_query(query)

            # Extract result from response
            if 'results' in data and 'bindings' in data['results']:
                bindings = data['results']['bindings']
                if bindings:
                    result = bindings[0]
                    return {
                        'success': True,
                        'elapsed': elapsed,
                        'graph_name': result.get('graphName', {}).get('value'),
                        'node_count': int(result.get('nodeCount', {}).get('value', 0)),
                        'relationship_count': int(result.get('relationshipCount', {}).get('value', 0))
                    }

            return {'success': False, 'elapsed': elapsed, 'error': 'No results returned'}

        except Exception as e:
            return {'success': False, 'elapsed': 0, 'error': str(e)}

    def drop_projection(self, name: str) -> bool:
        """Drop a projection if it exists."""
        query = f'CALL graph_drop("{name}")'
        try:
            self.execute_query(query)
            return True
        except:
            return False

    def run_benchmark_suite(self, iterations: int = 5) -> Dict:
        """
        Run comprehensive benchmark suite with multiple iterations.

        Args:
            iterations: Number of times to repeat each test

        Returns:
            benchmark results with statistics
        """
        print("=" * 70)
        print("MillenniumDB Projection Creation Benchmark")
        print("=" * 70)
        print(f"Server: {self.server_url}")
        print(f"Timestamp: {datetime.now().isoformat()}")
        print(f"Iterations per test: {iterations}")
        print()

        # Test configurations: (name_prefix, node_label, edge_label, description)
        test_configs = [
            ("bench_small", "User", "Friend", "Small projection (User-Friend)"),
            ("bench_medium", "Post", "HasTag", "Medium projection (Post-HasTag)"),
        ]

        results = {}

        for name_prefix, node_label, edge_label, description in test_configs:
            print(f"Testing: {description}")
            print("-" * 70)

            timings = []
            node_counts = []
            edge_counts = []

            for i in range(iterations):
                projection_name = f"{name_prefix}_{i}"

                # Clean up previous projection if exists
                self.drop_projection(projection_name)

                # Create projection and measure
                print(f"  Iteration {i+1}/{iterations}...", end=" ", flush=True)

                metrics = self.create_projection(projection_name, node_label, edge_label)

                if metrics['success']:
                    timings.append(metrics['elapsed'])
                    node_counts.append(metrics['node_count'])
                    edge_counts.append(metrics['relationship_count'])
                    print(f"✓ {metrics['elapsed']:.3f}s ({metrics['node_count']} nodes, {metrics['relationship_count']} edges)")
                else:
                    print(f"✗ Error: {metrics.get('error', 'Unknown error')}")

                # Clean up
                self.drop_projection(projection_name)
                time.sleep(0.5)  # Brief pause between iterations

            # Calculate statistics
            if timings:
                results[description] = {
                    'node_label': node_label,
                    'edge_label': edge_label,
                    'iterations': len(timings),
                    'avg_nodes': statistics.mean(node_counts),
                    'avg_edges': statistics.mean(edge_counts),
                    'mean_time': statistics.mean(timings),
                    'median_time': statistics.median(timings),
                    'min_time': min(timings),
                    'max_time': max(timings),
                    'stdev_time': statistics.stdev(timings) if len(timings) > 1 else 0,
                    'throughput_edges_per_sec': statistics.mean(edge_counts) / statistics.mean(timings) if statistics.mean(timings) > 0 else 0
                }

            print()

        return results

    def print_summary(self, results: Dict):
        """Print benchmark summary with performance metrics."""
        print("=" * 70)
        print("BENCHMARK RESULTS SUMMARY")
        print("=" * 70)
        print()

        for test_name, metrics in results.items():
            print(f"📊 {test_name}")
            print(f"   Configuration: {metrics['node_label']} → {metrics['edge_label']}")
            print(f"   Graph size: {metrics['avg_nodes']:.0f} nodes, {metrics['avg_edges']:.0f} edges")
            print(f"   ")
            print(f"   ⏱️  Performance:")
            print(f"      Mean time:      {metrics['mean_time']:.3f}s")
            print(f"      Median time:    {metrics['median_time']:.3f}s")
            print(f"      Min time:       {metrics['min_time']:.3f}s")
            print(f"      Max time:       {metrics['max_time']:.3f}s")
            print(f"      Std deviation:  {metrics['stdev_time']:.3f}s")
            print(f"   ")
            print(f"   📈 Throughput:     {metrics['throughput_edges_per_sec']:.1f} edges/sec")
            print()

        print("=" * 70)
        print("✅ Benchmark completed successfully!")
        print("=" * 70)


def main():
    """Main entry point for benchmark suite."""

    # Check server availability
    server_url = "http://localhost:1234"

    print(f"Checking server availability at {server_url}...")
    try:
        response = requests.get(server_url, timeout=2)
        print("✓ Server is running\n")
    except requests.exceptions.ConnectionError:
        print(f"✗ Error: Cannot connect to server at {server_url}")
        print("\nPlease start the server first:")
        print("  build/Release/bin/mdb server data/dbs/gql/posts")
        print()
        sys.exit(1)

    # Run benchmark
    benchmark = ProjectionBenchmark(server_url)
    results = benchmark.run_benchmark_suite(iterations=5)
    benchmark.print_summary(results)

    # Save results to JSON
    output_file = f"benchmark_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\n📄 Results saved to: {output_file}")


if __name__ == "__main__":
    main()
