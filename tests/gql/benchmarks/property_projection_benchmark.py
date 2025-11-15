#!/usr/bin/env python3
"""
MillenniumDB Property Projection Benchmark Suite

Measures property projection performance overhead and query speedup.

Benchmark Scenarios:
1. Property Extraction Overhead: Compare baseline vs node/edge/full properties
2. Query Performance: Main graph vs projection query times
3. Memory & Disk: Resource usage with properties

Usage:
    1. Start server: build/Release/bin/mdb server data/dbs/gql/posts
    2. Run benchmark: python3 tests/gql/benchmarks/property_projection_benchmark.py

Requirements:
    - Python 3.8+
    - requests library (pip install requests)
"""

import json
import time
import requests
import statistics
import sys
import os
from typing import Dict, List, Tuple, Optional
from datetime import datetime


class PropertyProjectionBenchmark:
    def __init__(self, server_url: str = "http://localhost:1234", db_path: str = None):
        self.server_url = server_url
        self.gql_endpoint = f"{server_url}/gql"
        self.db_path = db_path

    def execute_query(self, query: str) -> Tuple[Dict, float]:
        """
        Execute GQL query via HTTP API and measure execution time.

        Returns:
            (response_data, elapsed_seconds)
        """
        headers = {"Content-Type": "application/sparql-query"}

        start_time = time.perf_counter()
        response = requests.post(self.gql_endpoint, data=query, headers=headers)
        elapsed = time.perf_counter() - start_time

        response.raise_for_status()
        return response.json(), elapsed

    def create_projection(self, name: str, node_label: str, edge_label: str,
                         config: Optional[Dict] = None) -> Dict:
        """
        Create a graph projection with optional property configuration.

        Args:
            name: Projection name
            node_label: Node label to project
            edge_label: Edge label to project
            config: Optional dict with nodeProperties and/or relationshipProperties

        Returns:
            metrics dict with timing and metadata
        """
        # Build config string for GQL query
        config_str = ""
        if config:
            config_parts = []
            if 'nodeProperties' in config:
                props = config['nodeProperties']
                if isinstance(props, list):
                    props_str = "[" + ", ".join(f"'{p}'" for p in props) + "]"
                else:
                    props_str = f"'{props}'"
                config_parts.append(f"nodeProperties: {props_str}")

            if 'relationshipProperties' in config:
                props = config['relationshipProperties']
                if isinstance(props, list):
                    props_str = "[" + ", ".join(f"'{p}'" for p in props) + "]"
                else:
                    props_str = f"'{props}'"
                config_parts.append(f"relationshipProperties: {props_str}")

            if config_parts:
                config_str = ", {" + ", ".join(config_parts) + "}"

        query = f'''
            CALL graph_project(
                "{name}",
                "{node_label}",
                "{edge_label}"{config_str}
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
                        'relationship_count': int(result.get('relationshipCount', {}).get('value', 0)),
                        'config': config or {}
                    }

            return {'success': False, 'elapsed': elapsed, 'error': 'No results returned'}

        except Exception as e:
            return {'success': False, 'elapsed': 0, 'error': str(e)}

    def drop_projection(self, name: str) -> bool:
        """Drop a projection if it exists."""
        query = f'CALL graph_drop("{name}")'
        try:
            self.execute_query(query)
            time.sleep(0.2)  # Brief pause for cleanup
            return True
        except:
            return False

    def get_projection_disk_size(self, projection_name: str) -> Optional[int]:
        """
        Get disk size of projection in bytes.

        Returns None if db_path not provided or projection doesn't exist.
        """
        if not self.db_path:
            return None

        proj_path = os.path.join(self.db_path, "projections", projection_name)
        if not os.path.exists(proj_path):
            return None

        total_size = 0
        for dirpath, dirnames, filenames in os.walk(proj_path):
            for filename in filenames:
                filepath = os.path.join(dirpath, filename)
                if os.path.isfile(filepath):
                    total_size += os.path.getsize(filepath)

        return total_size

    def scenario_1_property_overhead(self, iterations: int = 3) -> Dict:
        """
        Benchmark 1: Property Extraction Overhead

        Compares projection creation time with different property configurations.
        """
        print("=" * 70)
        print("SCENARIO 1: Property Extraction Overhead")
        print("=" * 70)
        print()

        # Test configurations: (name, description, config)
        configs = [
            ("baseline", "Baseline (no properties)", {}),
            ("node_props", "Node properties (age, name)",
             {'nodeProperties': ['age', 'name']}),
            ("edge_props", "Edge properties (since)",
             {'relationshipProperties': ['since']}),
            ("full_props", "Full properties (node + edge)",
             {'nodeProperties': ['age', 'name'], 'relationshipProperties': ['since']}),
        ]

        results = {}

        for config_name, description, config in configs:
            print(f"Testing: {description}")
            print("-" * 70)

            timings = []
            node_counts = []
            edge_counts = []
            disk_sizes = []

            for i in range(iterations):
                projection_name = f"bench_{config_name}_{i}"

                # Clean up previous projection
                self.drop_projection(projection_name)

                # Create projection and measure
                print(f"  Iteration {i+1}/{iterations}...", end=" ", flush=True)

                metrics = self.create_projection(projection_name, "User", "Friend", config)

                if metrics['success']:
                    timings.append(metrics['elapsed'])
                    node_counts.append(metrics['node_count'])
                    edge_counts.append(metrics['relationship_count'])

                    # Measure disk size if possible
                    disk_size = self.get_projection_disk_size(projection_name)
                    if disk_size is not None:
                        disk_sizes.append(disk_size)

                    print(f"✓ {metrics['elapsed']:.3f}s ({metrics['node_count']} nodes, {metrics['relationship_count']} edges)")
                else:
                    print(f"✗ Error: {metrics.get('error', 'Unknown error')}")

                # Clean up
                self.drop_projection(projection_name)
                time.sleep(0.3)

            # Calculate statistics
            if timings:
                avg_time = statistics.mean(timings)
                avg_edges = statistics.mean(edge_counts)

                results[config_name] = {
                    'description': description,
                    'config': config,
                    'iterations': len(timings),
                    'avg_nodes': statistics.mean(node_counts),
                    'avg_edges': avg_edges,
                    'mean_time': avg_time,
                    'median_time': statistics.median(timings),
                    'min_time': min(timings),
                    'max_time': max(timings),
                    'stdev_time': statistics.stdev(timings) if len(timings) > 1 else 0,
                    'throughput_edges_per_sec': avg_edges / avg_time if avg_time > 0 else 0,
                }

                if disk_sizes:
                    results[config_name]['avg_disk_size_mb'] = statistics.mean(disk_sizes) / (1024 * 1024)

            print()

        # Calculate overhead vs baseline
        if 'baseline' in results:
            baseline_time = results['baseline']['mean_time']
            for key in results:
                if key != 'baseline':
                    overhead = ((results[key]['mean_time'] - baseline_time) / baseline_time) * 100
                    results[key]['overhead_vs_baseline_pct'] = overhead

        return results

    def scenario_2_query_performance(self, iterations: int = 3) -> Dict:
        """
        Benchmark 2: Query Performance Comparison

        Compares query execution time on main graph vs projection.
        """
        print("=" * 70)
        print("SCENARIO 2: Query Performance (Main Graph vs Projection)")
        print("=" * 70)
        print()

        # Create test projection with properties
        projection_name = "bench_query_perf"
        self.drop_projection(projection_name)

        print("Creating projection for query tests...")
        config = {'nodeProperties': ['age', 'name']}
        metrics = self.create_projection(projection_name, "User", "Friend", config)

        if not metrics['success']:
            print(f"✗ Failed to create projection: {metrics.get('error')}")
            return {}

        print(f"✓ Projection created: {metrics['node_count']} nodes, {metrics['relationship_count']} edges")
        print()

        # Test queries
        test_queries = [
            ("Topology query", "MATCH (n)-[r]->(m) RETURN COUNT(*)", "Count all edges"),
            ("Node count", "MATCH (n:User) RETURN COUNT(n)", "Count User nodes"),
        ]

        results = {}

        for query_name, query_pattern, description in test_queries:
            print(f"Testing: {description}")
            print("-" * 70)

            main_graph_times = []
            projection_times = []

            for i in range(iterations):
                print(f"  Iteration {i+1}/{iterations}...", end=" ", flush=True)

                # Query on main graph
                try:
                    _, main_elapsed = self.execute_query(query_pattern)
                    main_graph_times.append(main_elapsed)
                except Exception as e:
                    print(f"✗ Main graph error: {e}")
                    continue

                # Query on projection
                projection_query = f"USE GRAPH {projection_name}; {query_pattern}"
                try:
                    _, proj_elapsed = self.execute_query(projection_query)
                    projection_times.append(proj_elapsed)
                except Exception as e:
                    print(f"✗ Projection error: {e}")
                    continue

                print(f"✓ Main: {main_elapsed:.3f}s, Projection: {proj_elapsed:.3f}s")

            # Calculate statistics
            if main_graph_times and projection_times:
                avg_main = statistics.mean(main_graph_times)
                avg_proj = statistics.mean(projection_times)
                speedup = avg_main / avg_proj if avg_proj > 0 else 0

                results[query_name] = {
                    'description': description,
                    'query': query_pattern,
                    'main_graph_mean_time': avg_main,
                    'projection_mean_time': avg_proj,
                    'speedup': speedup,
                    'iterations': len(main_graph_times)
                }

            print()

        # Clean up
        self.drop_projection(projection_name)

        return results

    def print_scenario_1_summary(self, results: Dict):
        """Print Scenario 1 summary (Property Overhead)."""
        print("=" * 70)
        print("SCENARIO 1 RESULTS: Property Extraction Overhead")
        print("=" * 70)
        print()

        if not results:
            print("No results to display.")
            return

        # Print baseline first if available
        if 'baseline' in results:
            metrics = results['baseline']
            print(f"📊 Baseline (No Properties)")
            print(f"   Mean time:      {metrics['mean_time']:.3f}s")
            print(f"   Throughput:     {metrics['throughput_edges_per_sec']:.1f} edges/sec")
            if 'avg_disk_size_mb' in metrics:
                print(f"   Disk size:      {metrics['avg_disk_size_mb']:.2f} MB")
            print()

        # Print property configurations
        for key, metrics in results.items():
            if key == 'baseline':
                continue

            print(f"📊 {metrics['description']}")
            print(f"   Mean time:      {metrics['mean_time']:.3f}s")
            print(f"   Throughput:     {metrics['throughput_edges_per_sec']:.1f} edges/sec")

            if 'overhead_vs_baseline_pct' in metrics:
                overhead = metrics['overhead_vs_baseline_pct']
                symbol = "📈" if overhead > 0 else "📉"
                print(f"   {symbol} Overhead:      {overhead:+.1f}% vs baseline")

            if 'avg_disk_size_mb' in metrics:
                print(f"   Disk size:      {metrics['avg_disk_size_mb']:.2f} MB")
                if 'baseline' in results and 'avg_disk_size_mb' in results['baseline']:
                    baseline_disk = results['baseline']['avg_disk_size_mb']
                    disk_overhead = ((metrics['avg_disk_size_mb'] - baseline_disk) / baseline_disk) * 100
                    print(f"   Disk overhead:  {disk_overhead:+.1f}% vs baseline")

            print()

    def print_scenario_2_summary(self, results: Dict):
        """Print Scenario 2 summary (Query Performance)."""
        print("=" * 70)
        print("SCENARIO 2 RESULTS: Query Performance Comparison")
        print("=" * 70)
        print()

        if not results:
            print("No results to display.")
            return

        for query_name, metrics in results.items():
            print(f"📊 {metrics['description']}")
            print(f"   Query: {metrics['query']}")
            print(f"   Main graph:     {metrics['main_graph_mean_time']:.3f}s")
            print(f"   Projection:     {metrics['projection_mean_time']:.3f}s")

            speedup = metrics['speedup']
            if speedup > 1:
                print(f"   ⚡ Speedup:      {speedup:.2f}x faster on projection")
            elif speedup < 1:
                print(f"   ⏱️  Speedup:      {1/speedup:.2f}x slower on projection")
            else:
                print(f"   ⚖️  Speedup:      ~same performance")

            print()

    def save_results_json(self, results: Dict, filename: str = "property_projection_benchmark_results.json"):
        """Save benchmark results to JSON file."""
        output = {
            'timestamp': datetime.now().isoformat(),
            'server_url': self.server_url,
            'scenarios': results
        }

        with open(filename, 'w') as f:
            json.dump(output, f, indent=2)

        print(f"✅ Results saved to: {filename}")
        print()

    def run_full_benchmark(self, iterations: int = 3):
        """Run all benchmark scenarios."""
        print("\n")
        print("╔" + "=" * 68 + "╗")
        print("║" + " MillenniumDB Property Projection Benchmark Suite ".center(68) + "║")
        print("╚" + "=" * 68 + "╝")
        print()
        print(f"Server: {self.server_url}")
        print(f"Timestamp: {datetime.now().isoformat()}")
        print(f"Iterations per test: {iterations}")
        print()

        all_results = {}

        # Scenario 1: Property Overhead
        try:
            scenario1_results = self.scenario_1_property_overhead(iterations)
            all_results['scenario_1_property_overhead'] = scenario1_results
            self.print_scenario_1_summary(scenario1_results)
        except Exception as e:
            print(f"✗ Scenario 1 failed: {e}")
            print()

        # Scenario 2: Query Performance
        try:
            scenario2_results = self.scenario_2_query_performance(iterations)
            all_results['scenario_2_query_performance'] = scenario2_results
            self.print_scenario_2_summary(scenario2_results)
        except Exception as e:
            print(f"✗ Scenario 2 failed: {e}")
            print()

        # Save results
        self.save_results_json(all_results)

        return all_results


def main():
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(description='MillenniumDB Property Projection Benchmark')
    parser.add_argument('--server', default='http://localhost:1234', help='Server URL (default: http://localhost:1234)')
    parser.add_argument('--iterations', type=int, default=3, help='Iterations per test (default: 3)')
    parser.add_argument('--db-path', help='Database path for disk size measurements (optional)')

    args = parser.parse_args()

    # Check server availability
    try:
        response = requests.get(args.server, timeout=2)
        print(f"✓ Server reachable at {args.server}")
    except:
        print(f"✗ Cannot connect to server at {args.server}")
        print("  Please start the server first:")
        print("  build/Release/bin/mdb server data/dbs/gql/posts")
        sys.exit(1)

    # Run benchmark
    benchmark = PropertyProjectionBenchmark(args.server, args.db_path)
    benchmark.run_full_benchmark(args.iterations)


if __name__ == "__main__":
    main()
