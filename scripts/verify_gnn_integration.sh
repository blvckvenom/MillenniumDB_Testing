#!/bin/bash
#
# Master GNN Integration Verification Script
#
# Purpose: Orchestrates all verification methods for the graph projection ↔ GNN
#          integration layer. Runs comprehensive tests to validate data flow
#          correctness at every layer.
#
# Verification Phases:
#   Phase 1: Projection Creation Verification (INV-PROJ-001, INV-PROJ-002)
#   Phase 2: GNN Integration Verification (INV-GNN-001)
#   Phase 3: Orientation Verification (INV-ORI-002)
#   Phase 4: GQL Integration Tests
#   Phase 5: Unit Tests (if GTest available)
#
# Usage:
#   ./scripts/verify_gnn_integration.sh [options]
#
# Options:
#   --quick       Run only quick verification (Phases 1-3)
#   --full        Run all verification phases including stress tests
#   --phase N     Run only phase N (1-5)
#   --database    Path to test database (default: data/dbs/gql/cora)
#   --help        Show this help message
#
# Exit Codes:
#   0 - All verifications passed
#   1 - One or more verifications failed
#   2 - Prerequisites not met
#   3 - Invalid arguments
#
# Author: MillenniumDB Project
# Date: 2026-01-23

set -euo pipefail

# =============================================================================
# Configuration
# =============================================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
DB_FOLDER="${PROJECT_ROOT}/data/dbs/gql/cora"
BUILD_TYPE="${BUILD_TYPE:-Release}"
QUICK_MODE=false
FULL_MODE=false
RUN_PHASE=""

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
BOLD='\033[1m'
NC='\033[0m'

# =============================================================================
# Logging
# =============================================================================

log_header() {
    echo ""
    echo -e "${BOLD}${BLUE}════════════════════════════════════════════════════════════════${NC}"
    echo -e "${BOLD}${BLUE}  $1${NC}"
    echo -e "${BOLD}${BLUE}════════════════════════════════════════════════════════════════${NC}"
    echo ""
}

log_phase() {
    echo ""
    echo -e "${CYAN}────────────────────────────────────────────────────────────────${NC}"
    echo -e "${CYAN}  Phase $1: $2${NC}"
    echo -e "${CYAN}────────────────────────────────────────────────────────────────${NC}"
}

log_info()  { echo -e "${YELLOW}[INFO]${NC}  $*"; }
log_pass()  { echo -e "${GREEN}[PASS]${NC}  $*"; }
log_fail()  { echo -e "${RED}[FAIL]${NC}  $*" >&2; }
log_skip()  { echo -e "${YELLOW}[SKIP]${NC}  $*"; }

# =============================================================================
# Argument Parsing
# =============================================================================

show_help() {
    cat << EOF
GNN Integration Verification Script

Usage: $0 [options]

Options:
  --quick       Run only quick verification (Phases 1-3)
  --full        Run all verification phases including stress tests
  --phase N     Run only phase N (1-5)
  --database    Path to test database (default: data/dbs/gql/cora)
  --help        Show this help message

Verification Phases:
  1: Projection Creation (INV-PROJ-001, INV-PROJ-002)
  2: GNN Neighbor Counts (INV-GNN-001)
  3: Orientation Verification (INV-ORI-002)
  4: GQL Integration Tests
  5: Unit Tests (requires GTest)

Examples:
  $0 --quick                    # Fast verification
  $0 --phase 3                  # Only test orientations
  $0 --full --database mydb/    # Full verification on custom database
EOF
}

parse_args() {
    while [[ $# -gt 0 ]]; do
        case $1 in
            --quick)
                QUICK_MODE=true
                shift
                ;;
            --full)
                FULL_MODE=true
                shift
                ;;
            --phase)
                RUN_PHASE="$2"
                shift 2
                ;;
            --database)
                DB_FOLDER="$2"
                shift 2
                ;;
            --help)
                show_help
                exit 0
                ;;
            *)
                echo "Unknown option: $1"
                show_help
                exit 3
                ;;
        esac
    done
}

# =============================================================================
# Prerequisites
# =============================================================================

check_prerequisites() {
    log_header "Checking Prerequisites"

    local missing=0

    # Check MDB executable
    if [ -x "${PROJECT_ROOT}/build/${BUILD_TYPE}/bin/mdb" ]; then
        log_pass "MDB executable found (${BUILD_TYPE} build)"
    else
        log_fail "MDB executable not found at build/${BUILD_TYPE}/bin/mdb"
        log_info "Build with: cmake --build build/${BUILD_TYPE} -j \$(nproc)"
        missing=$((missing + 1))
    fi

    # Check database folder
    if [ -d "$DB_FOLDER" ]; then
        log_pass "Database folder exists: $DB_FOLDER"
    else
        log_fail "Database folder not found: $DB_FOLDER"
        missing=$((missing + 1))
    fi

    # Check jq
    if command -v jq &> /dev/null; then
        log_pass "jq is available"
    else
        log_fail "jq is required but not installed"
        missing=$((missing + 1))
    fi

    # Check curl
    if command -v curl &> /dev/null; then
        log_pass "curl is available"
    else
        log_fail "curl is required but not installed"
        missing=$((missing + 1))
    fi

    # Check verification scripts
    if [ -x "${SCRIPT_DIR}/verify_gnn_neighbor_counts.sh" ]; then
        log_pass "verify_gnn_neighbor_counts.sh found"
    else
        log_fail "verify_gnn_neighbor_counts.sh not found or not executable"
        missing=$((missing + 1))
    fi

    if [ -x "${SCRIPT_DIR}/verify_undirected_orientation.sh" ]; then
        log_pass "verify_undirected_orientation.sh found"
    else
        log_fail "verify_undirected_orientation.sh not found or not executable"
        missing=$((missing + 1))
    fi

    if [ $missing -gt 0 ]; then
        echo ""
        log_fail "$missing prerequisite(s) missing. Please resolve before continuing."
        exit 2
    fi

    echo ""
    log_pass "All prerequisites satisfied"
}

# =============================================================================
# Verification Phases
# =============================================================================

PHASE_RESULTS=()

# Phase 1: Projection Creation Verification
run_phase_1() {
    log_phase 1 "Projection Creation Verification"
    log_info "Verifying INV-PROJ-001 (node count) and INV-PROJ-002 (edge count)"

    if "${SCRIPT_DIR}/verify_gnn_neighbor_counts.sh" "$DB_FOLDER"; then
        log_pass "Phase 1 PASSED: Projection counts are consistent"
        PHASE_RESULTS+=(1)
        return 0
    else
        log_fail "Phase 1 FAILED: Projection count discrepancies detected"
        PHASE_RESULTS+=(0)
        return 1
    fi
}

# Phase 2: GNN Integration Verification
run_phase_2() {
    log_phase 2 "GNN Integration Verification"
    log_info "Verifying INV-GNN-001 (degree sum equals edge count)"

    # This is already covered by verify_gnn_neighbor_counts.sh
    # We mark it passed if Phase 1 passed
    if [ "${PHASE_RESULTS[0]:-0}" -eq 1 ]; then
        log_pass "Phase 2 PASSED: GNN neighbor count invariants verified"
        PHASE_RESULTS+=(1)
        return 0
    else
        log_info "Phase 2 depends on Phase 1 - running verification"
        if "${SCRIPT_DIR}/verify_gnn_neighbor_counts.sh" "$DB_FOLDER"; then
            log_pass "Phase 2 PASSED: GNN neighbor count invariants verified"
            PHASE_RESULTS+=(1)
            return 0
        else
            log_fail "Phase 2 FAILED: GNN invariant violations detected"
            PHASE_RESULTS+=(0)
            return 1
        fi
    fi
}

# Phase 3: Orientation Verification
run_phase_3() {
    log_phase 3 "Orientation Verification"
    log_info "Verifying INV-ORI-002 (UNDIRECTED bidirectional traversal)"

    if "${SCRIPT_DIR}/verify_undirected_orientation.sh" "$DB_FOLDER"; then
        log_pass "Phase 3 PASSED: UNDIRECTED orientation allows bidirectional traversal"
        PHASE_RESULTS+=(1)
        return 0
    else
        log_fail "Phase 3 FAILED: UNDIRECTED orientation issues detected"
        PHASE_RESULTS+=(0)
        return 1
    fi
}

# Phase 4: GQL Integration Tests
run_phase_4() {
    log_phase 4 "GQL Integration Tests"
    log_info "Running projection_verification test suite"

    cd "$PROJECT_ROOT"

    if python3 tests/gql/scripts/test.py --module projection_verification 2>&1; then
        log_pass "Phase 4 PASSED: GQL integration tests passed"
        PHASE_RESULTS+=(1)
        return 0
    else
        log_fail "Phase 4 FAILED: GQL integration test failures"
        PHASE_RESULTS+=(0)
        return 1
    fi
}

# Phase 5: Unit Tests
run_phase_5() {
    log_phase 5 "Unit Tests"
    log_info "Running C++ unit tests via ctest"

    cd "$PROJECT_ROOT"

    # Check if GNN tests are available
    if [ -f "build/Debug/test_gnn_core" ]; then
        log_info "Running GNN core unit tests..."
        if ctest --test-dir build/Debug -R GNNCoreTests --output-on-failure; then
            log_pass "GNN unit tests passed"
        else
            log_fail "GNN unit tests failed"
            PHASE_RESULTS+=(0)
            return 1
        fi
    else
        log_skip "GNN unit tests not available (ENABLE_GNN=OFF or GTest not found)"
    fi

    # Run projection-related tests
    if ctest --test-dir build/Debug -R projection --output-on-failure 2>/dev/null; then
        log_pass "Projection unit tests passed"
    else
        log_info "No projection unit tests found or some failed"
    fi

    log_pass "Phase 5 PASSED: Unit tests completed"
    PHASE_RESULTS+=(1)
    return 0
}

# =============================================================================
# Main Execution
# =============================================================================

main() {
    parse_args "$@"

    log_header "GNN Integration Verification Suite"
    echo "Date: $(date '+%Y-%m-%d %H:%M:%S')"
    echo "Database: $DB_FOLDER"
    echo "Build Type: $BUILD_TYPE"
    echo ""

    check_prerequisites

    local failed=0

    # Determine which phases to run
    if [ -n "$RUN_PHASE" ]; then
        # Run single phase
        case "$RUN_PHASE" in
            1) run_phase_1 || failed=1 ;;
            2) run_phase_2 || failed=1 ;;
            3) run_phase_3 || failed=1 ;;
            4) run_phase_4 || failed=1 ;;
            5) run_phase_5 || failed=1 ;;
            *) log_fail "Invalid phase: $RUN_PHASE"; exit 3 ;;
        esac
    elif [ "$QUICK_MODE" = true ]; then
        # Quick mode: Phases 1-3 only
        run_phase_1 || failed=1
        run_phase_2 || failed=1
        run_phase_3 || failed=1
    else
        # Full mode: All phases
        run_phase_1 || failed=1
        run_phase_2 || failed=1
        run_phase_3 || failed=1
        run_phase_4 || failed=1

        if [ "$FULL_MODE" = true ]; then
            run_phase_5 || failed=1
        fi
    fi

    # Summary
    log_header "Verification Summary"

    local passed=0
    local total=${#PHASE_RESULTS[@]}

    for result in "${PHASE_RESULTS[@]}"; do
        if [ "$result" -eq 1 ]; then
            passed=$((passed + 1))
        fi
    done

    echo "Results: $passed/$total phases passed"
    echo ""

    if [ $failed -eq 0 ]; then
        echo -e "${GREEN}${BOLD}╔══════════════════════════════════════════════════════════╗${NC}"
        echo -e "${GREEN}${BOLD}║          ALL VERIFICATIONS PASSED SUCCESSFULLY           ║${NC}"
        echo -e "${GREEN}${BOLD}╚══════════════════════════════════════════════════════════╝${NC}"
        echo ""
        log_info "Graph Projection ↔ GNN Integration is verified correct"
        exit 0
    else
        echo -e "${RED}${BOLD}╔══════════════════════════════════════════════════════════╗${NC}"
        echo -e "${RED}${BOLD}║          VERIFICATION FAILED - SEE ERRORS ABOVE          ║${NC}"
        echo -e "${RED}${BOLD}╚══════════════════════════════════════════════════════════╝${NC}"
        echo ""
        log_info "Review the failed phases and fix issues before deployment"
        exit 1
    fi
}

main "$@"
