from __future__ import annotations

from pathlib import Path

# Host and port the server will listen on
HOST = "127.0.0.1"
PORT = 8080

# Time between checks for server initialization
SLEEP_DELAY = 0.01

# Maximum time in seconds that the server will wait for a query
TIMEOUT = 60

# Assume that the script is run from the root directory
CWD = Path().cwd()
ROOT_TEST_DIR = CWD / "tests/gql"
TEST_SUITE_DIR = ROOT_TEST_DIR / "test_suites"
TESTING_DBS_DIR = ROOT_TEST_DIR / "tmp/dbs"
SERVER_LOGS_DIR = ROOT_TEST_DIR / "tmp/server-logs"

# Executables
EXECUTABLE = CWD / "build/Debug/bin/mdb"

# Width of each column of test outputs
OUTPUT_COLUMN_WIDTH = 50

LOGGING_LEVELS = {
    "SUMMARY": True,
    "ERROR": True,
    "WARNING": True,
    "OUTPUT": True,
    "CORRECT": False,
    "BEGIN": False,
    "END": False,
    "SKIPPED": False,
    "DEBUG": False,
    "SERVER_LOG": False,
}

# Test suites to run
TEST_SUITES: list[str] = [
    "simple",
    "edges",
    "list",
    "order_by",
    "alias",
    "sequence",
    "expressions",
    "repeated_variables",
    "quantifier_aggregation",
    "let_statement",
    "filter_statement",
    "order_by_statement",
    "datetime",
    "path_binding",
    "group_by",
    "call_procedure",
    "projection_native",
    "projection_properties",
    "projection_comprehensive",
    "projection_advanced",
    "projection_adaptive_buffer",
    "projection_no_labels",
    "list_exprs",
]

# GNN test suites require ENABLE_GNN=ON build with LibTorch. Add suites here when GNN integration tests are created.
GNN_TEST_SUITES: list[str] = [
]


def get_test_suites(executable: Path | None = None) -> list[str]:
    """Return test suites, filtering GNN suites if binary doesn't support them."""
    import subprocess

    if executable is None:
        executable = EXECUTABLE

    has_gnn = False
    try:
        result = subprocess.run(
            [str(executable), "help"],
            capture_output=True, text=True, timeout=5
        )
        has_gnn = "gnn" in result.stdout.lower()
    except Exception:
        pass

    suites = list(TEST_SUITES)
    if has_gnn:
        suites.extend(GNN_TEST_SUITES)
    return suites


# Tests with the following query files fill be ignored
IGNORED_TESTS: set[str] = set()
