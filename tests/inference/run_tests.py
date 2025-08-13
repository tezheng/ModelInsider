#!/usr/bin/env python3
"""
Test Runner for ONNXAutoProcessor Test Suite

This script provides convenient ways to run different categories of tests
with appropriate configurations and reporting.

Usage Examples:
    python run_tests.py --smoke          # Quick smoke tests (5 min)
    python run_tests.py --sanity         # Core functionality tests (15 min)
    python run_tests.py --unit           # Unit tests (30 min)
    python run_tests.py --integration    # Integration tests (45 min)
    python run_tests.py --performance    # Performance benchmarks (20 min)
    python run_tests.py --all            # All tests
    python run_tests.py --ci             # CI-friendly test subset

Author: Generated for TEZ-144 ONNX AutoProcessor Test Implementation
"""

import argparse
import subprocess
import sys
import time
from pathlib import Path


class TestRunner:
    """Test runner with different execution modes."""

    def __init__(self, test_dir: Path | None = None):
        self.test_dir = test_dir or Path(__file__).parent
        self.total_start_time = None

    def run_command(self, cmd: list[str], description: str) -> bool:
        """Run a command and return success status."""
        print(f"\n{'=' * 60}")
        print(f"🧪 {description}")
        print(f"{'=' * 60}")
        print(f"Command: {' '.join(cmd)}")
        print()

        start_time = time.time()

        try:
            result = subprocess.run(
                cmd, cwd=self.test_dir, check=False, capture_output=False
            )

            elapsed_time = time.time() - start_time

            if result.returncode == 0:
                print(f"\n✅ {description} PASSED in {elapsed_time:.1f}s")
                return True
            else:
                print(
                    f"\n❌ {description} FAILED in {elapsed_time:.1f}s (exit code: {result.returncode})"
                )
                return False

        except KeyboardInterrupt:
            print(f"\n⚠️  {description} INTERRUPTED by user")
            return False
        except Exception as e:
            print(f"\n💥 {description} ERROR: {e}")
            return False

    def smoke_tests(self) -> bool:
        """Run smoke tests - quick validation (5 minutes)."""
        cmd = [
            "python",
            "-m",
            "pytest",
            "-m",
            "smoke",
            "-v",
            "--tb=short",
            "--durations=5",
            "test_onnx_auto_processor.py",
        ]
        return self.run_command(cmd, "Smoke Tests (5 min)")

    def sanity_tests(self) -> bool:
        """Run sanity tests - core features (15 minutes)."""
        cmd = [
            "python",
            "-m",
            "pytest",
            "-m",
            "sanity",
            "-v",
            "--tb=short",
            "--durations=5",
            "test_onnx_auto_processor.py",
        ]
        return self.run_command(cmd, "Sanity Tests (15 min)")

    def unit_tests(self) -> bool:
        """Run unit tests - component isolation (30 minutes)."""
        cmd = [
            "python",
            "-m",
            "pytest",
            "-m",
            "unit",
            "-v",
            "--tb=short",
            "--durations=10",
            "test_onnx_auto_processor.py",
        ]
        return self.run_command(cmd, "Unit Tests (30 min)")

    def integration_tests(self) -> bool:
        """Run integration tests - end-to-end workflows (45 minutes)."""
        cmd = [
            "python",
            "-m",
            "pytest",
            "-m",
            "integration",
            "-v",
            "--tb=long",
            "--durations=10",
            "test_onnx_auto_processor.py",
        ]
        return self.run_command(cmd, "Integration Tests (45 min)")

    def performance_tests(self) -> bool:
        """Run performance tests - benchmarks (20 minutes)."""
        cmd = [
            "python",
            "-m",
            "pytest",
            "-m",
            "performance",
            "-v",
            "--tb=short",
            "--durations=0",  # Show all durations for performance analysis
            "test_onnx_auto_processor.py",
        ]
        return self.run_command(cmd, "Performance Tests (20 min)")

    def multimodal_tests(self) -> bool:
        """Run multimodal-specific tests."""
        cmd = [
            "python",
            "-m",
            "pytest",
            "-m",
            "multimodal",
            "-v",
            "--tb=short",
            "--durations=5",
            "test_onnx_auto_processor.py",
        ]
        return self.run_command(cmd, "Multimodal Tests")

    def fast_tests(self) -> bool:
        """Run all tests except slow ones - good for development."""
        cmd = [
            "python",
            "-m",
            "pytest",
            "-m",
            "not slow",
            "-v",
            "--tb=short",
            "--durations=10",
            "test_onnx_auto_processor.py",
        ]
        return self.run_command(cmd, "Fast Tests (excludes slow)")

    def ci_tests(self) -> bool:
        """Run CI-friendly test subset."""
        cmd = [
            "python",
            "-m",
            "pytest",
            "-m",
            "not (slow or requires_gpu or requires_models)",
            "-v",
            "--tb=short",
            "--maxfail=3",  # Fail fast for CI
            "--durations=5",
            "test_onnx_auto_processor.py",
        ]
        return self.run_command(cmd, "CI Tests")

    def all_tests(self) -> bool:
        """Run all tests."""
        cmd = [
            "python",
            "-m",
            "pytest",
            "-v",
            "--tb=short",
            "--durations=20",
            "test_onnx_auto_processor.py",
        ]
        return self.run_command(cmd, "All Tests")

    def test_discovery(self) -> bool:
        """Show test discovery without running."""
        cmd = [
            "python",
            "-m",
            "pytest",
            "--collect-only",
            "-q",
            "test_onnx_auto_processor.py",
        ]
        return self.run_command(cmd, "Test Discovery")

    def run_progressive_validation(self) -> bool:
        """Run tests in order of increasing complexity/time."""
        print(f"\n{'=' * 60}")
        print("🚀 PROGRESSIVE VALIDATION WORKFLOW")
        print(f"{'=' * 60}")

        self.total_start_time = time.time()
        results = []

        # Run tests in progressive order
        test_stages = [
            ("Smoke Tests", self.smoke_tests),
            ("Sanity Tests", self.sanity_tests),
            ("Unit Tests", self.unit_tests),
            ("Integration Tests", self.integration_tests),
            ("Performance Tests", self.performance_tests),
        ]

        for stage_name, test_func in test_stages:
            print(f"\n📍 Starting {stage_name}...")
            success = test_func()
            results.append((stage_name, success))

            if not success:
                print(f"\n❌ {stage_name} failed - stopping progressive validation")
                self._print_summary(results)
                return False

        self._print_summary(results)
        return all(success for _, success in results)

    def _print_summary(self, results: list[tuple]) -> None:
        """Print test execution summary."""
        total_time = time.time() - self.total_start_time if self.total_start_time else 0

        print(f"\n{'=' * 60}")
        print("📊 TEST EXECUTION SUMMARY")
        print(f"{'=' * 60}")

        passed = sum(1 for _, success in results if success)
        total = len(results)

        for stage_name, success in results:
            status = "✅ PASS" if success else "❌ FAIL"
            print(f"{status} {stage_name}")

        print(f"\nResults: {passed}/{total} stages passed")
        print(f"Total time: {total_time:.1f}s")

        if passed == total:
            print("🎉 All test stages completed successfully!")
        else:
            print(f"⚠️  {total - passed} stage(s) failed")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Run ONNXAutoProcessor tests with different configurations",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_tests.py --smoke          # Quick validation (5 min)
  python run_tests.py --sanity         # Core features (15 min)
  python run_tests.py --unit           # Component tests (30 min)
  python run_tests.py --integration    # End-to-end tests (45 min)
  python run_tests.py --performance    # Benchmarks (20 min)
  python run_tests.py --fast           # All except slow tests
  python run_tests.py --ci             # CI-friendly subset
  python run_tests.py --all            # Complete test suite
  python run_tests.py --progressive    # Run in order of complexity

Test Categories:
  🚬 Smoke Tests: Basic functionality verification (5 min)
  ✅ Sanity Tests: Core feature validation (15 min)  
  🔧 Unit Tests: Component isolation testing (30 min)
  🔄 Integration Tests: End-to-end workflows (45 min)
  ⚡ Performance Tests: Speed/memory benchmarks (20 min)
        """,
    )

    # Test category options
    test_group = parser.add_mutually_exclusive_group(required=True)
    test_group.add_argument(
        "--smoke", action="store_true", help="Run smoke tests (5 min)"
    )
    test_group.add_argument(
        "--sanity", action="store_true", help="Run sanity tests (15 min)"
    )
    test_group.add_argument(
        "--unit", action="store_true", help="Run unit tests (30 min)"
    )
    test_group.add_argument(
        "--integration", action="store_true", help="Run integration tests (45 min)"
    )
    test_group.add_argument(
        "--performance", action="store_true", help="Run performance tests (20 min)"
    )
    test_group.add_argument(
        "--multimodal", action="store_true", help="Run multimodal tests only"
    )
    test_group.add_argument(
        "--fast", action="store_true", help="Run all except slow tests"
    )
    test_group.add_argument(
        "--ci", action="store_true", help="Run CI-friendly test subset"
    )
    test_group.add_argument("--all", action="store_true", help="Run all tests")
    test_group.add_argument(
        "--discover", action="store_true", help="Show test discovery without running"
    )
    test_group.add_argument(
        "--progressive", action="store_true", help="Run progressive validation workflow"
    )

    # Additional options
    parser.add_argument(
        "--test-dir",
        type=Path,
        help="Directory containing tests (default: current directory)",
    )

    args = parser.parse_args()

    # Create test runner
    runner = TestRunner(args.test_dir)

    # Run selected test category
    success = False

    if args.smoke:
        success = runner.smoke_tests()
    elif args.sanity:
        success = runner.sanity_tests()
    elif args.unit:
        success = runner.unit_tests()
    elif args.integration:
        success = runner.integration_tests()
    elif args.performance:
        success = runner.performance_tests()
    elif args.multimodal:
        success = runner.multimodal_tests()
    elif args.fast:
        success = runner.fast_tests()
    elif args.ci:
        success = runner.ci_tests()
    elif args.all:
        success = runner.all_tests()
    elif args.discover:
        success = runner.test_discovery()
    elif args.progressive:
        success = runner.run_progressive_validation()

    # Exit with appropriate code
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
