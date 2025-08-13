"""
Test runner for ONNX inference test suite.

Provides convenient ways to run different categories of tests
and generate reports.
"""

import pytest
import sys
from pathlib import Path

def run_unit_tests():
    """Run only unit tests."""
    return pytest.main([
        str(Path(__file__).parent),
        "-v",
        "-m", "unit",
        "--tb=short"
    ])

def run_integration_tests():
    """Run only integration tests."""
    return pytest.main([
        str(Path(__file__).parent),
        "-v", 
        "-m", "integration",
        "--tb=short"
    ])

def run_smoke_tests():
    """Run only smoke tests."""
    return pytest.main([
        str(Path(__file__).parent),
        "-v",
        "-m", "smoke", 
        "--tb=short"
    ])

def run_sanity_tests():
    """Run only sanity tests."""
    return pytest.main([
        str(Path(__file__).parent),
        "-v",
        "-m", "sanity",
        "--tb=short"
    ])

def run_all_tests():
    """Run all tests."""
    return pytest.main([
        str(Path(__file__).parent),
        "-v",
        "--tb=short"
    ])

def run_fast_tests():
    """Run smoke and unit tests (fast subset)."""
    return pytest.main([
        str(Path(__file__).parent),
        "-v",
        "-m", "smoke or unit",
        "--tb=short"
    ])

def run_with_coverage():
    """Run all tests with coverage report."""
    return pytest.main([
        str(Path(__file__).parent),
        "-v",
        "--cov=onnx_tokenizer",
        "--cov=enhanced_pipeline", 
        "--cov=auto_model_loader",
        "--cov=inference_utils",
        "--cov-report=html",
        "--cov-report=term-missing",
        "--tb=short"
    ])

def run_failed_only():
    """Run only previously failed tests."""
    return pytest.main([
        str(Path(__file__).parent),
        "-v",
        "--lf",  # last failed
        "--tb=short"
    ])

def run_specific_test(test_path):
    """Run a specific test file or test function."""
    return pytest.main([
        test_path,
        "-v",
        "--tb=short"
    ])

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="ONNX Inference Test Runner")
    parser.add_argument(
        "test_type",
        choices=["unit", "integration", "smoke", "sanity", "all", "fast", "coverage", "failed"],
        help="Type of tests to run"
    )
    parser.add_argument(
        "--file",
        help="Specific test file to run"
    )
    
    args = parser.parse_args()
    
    if args.file:
        exit_code = run_specific_test(args.file)
    else:
        test_runners = {
            "unit": run_unit_tests,
            "integration": run_integration_tests,
            "smoke": run_smoke_tests,
            "sanity": run_sanity_tests,
            "all": run_all_tests,
            "fast": run_fast_tests,
            "coverage": run_with_coverage,
            "failed": run_failed_only
        }
        
        exit_code = test_runners[args.test_type]()
    
    sys.exit(exit_code)