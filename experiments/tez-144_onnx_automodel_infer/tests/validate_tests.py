#!/usr/bin/env python3
"""
Test Validation Script

Validates that the test files are properly structured and can be imported.
This is a basic validation to ensure the test suite is ready to run.
"""

import ast
import sys
from pathlib import Path


def validate_test_file(file_path: Path) -> dict:
    """Validate a Python test file structure."""
    results = {
        "file": str(file_path),
        "valid": True,
        "errors": [],
        "test_count": 0,
        "fixture_count": 0,
        "marker_count": 0
    }
    
    try:
        with open(file_path, encoding='utf-8') as f:
            content = f.read()
        
        # Parse the AST
        tree = ast.parse(content, filename=str(file_path))
        
        # Count functions and methods
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                if node.name.startswith('test_'):
                    results["test_count"] += 1
                elif any(d.id == 'fixture' for d in node.decorator_list 
                        if isinstance(d, ast.Name)):
                    results["fixture_count"] += 1
            
            # Count pytest markers
            if isinstance(node, ast.Call):
                if (isinstance(node.func, ast.Attribute) and 
                    isinstance(node.func.value, ast.Attribute) and
                    isinstance(node.func.value.value, ast.Name) and
                    node.func.value.value.id == 'pytest' and
                    node.func.value.attr == 'mark'):
                    results["marker_count"] += 1
    
    except SyntaxError as e:
        results["valid"] = False
        results["errors"].append(f"Syntax error: {e}")
    except Exception as e:
        results["valid"] = False
        results["errors"].append(f"Error parsing file: {e}")
    
    return results

def main():
    """Validate all test files."""
    test_dir = Path(__file__).parent
    
    print("🧪 Validating ONNXAutoProcessor Test Suite")
    print("=" * 60)
    
    test_files = [
        "test_utils.py",
        "test_onnx_auto_processor.py",
        "conftest.py"
    ]
    
    total_tests = 0
    total_fixtures = 0
    all_valid = True
    
    for test_file in test_files:
        file_path = test_dir / test_file
        
        if not file_path.exists():
            print(f"❌ {test_file}: File not found")
            all_valid = False
            continue
        
        results = validate_test_file(file_path)
        
        if results["valid"]:
            print(f"✅ {test_file}: Valid")
            print(f"   📊 Tests: {results['test_count']}")
            print(f"   🔧 Fixtures: {results['fixture_count']}")
            print(f"   🏷️  Markers: {results['marker_count']}")
            
            total_tests += results["test_count"]
            total_fixtures += results["fixture_count"]
        else:
            print(f"❌ {test_file}: Invalid")
            for error in results["errors"]:
                print(f"   💥 {error}")
            all_valid = False
        
        print()
    
    # Validate configuration files
    config_files = [
        ("pytest.ini", "Pytest configuration"),
        ("run_tests.py", "Test runner script"),
        ("README.md", "Test documentation")
    ]
    
    for config_file, description in config_files:
        file_path = test_dir / config_file
        if file_path.exists():
            print(f"✅ {config_file}: {description} present")
        else:
            print(f"❌ {config_file}: {description} missing")
            all_valid = False
    
    print("=" * 60)
    print("📊 VALIDATION SUMMARY")
    print("=" * 60)
    print(f"Total test functions found: {total_tests}")
    print(f"Total fixtures found: {total_fixtures}")
    print(f"All files valid: {'✅ Yes' if all_valid else '❌ No'}")
    
    if all_valid:
        print("\n🎉 Test suite validation successful!")
        print("\nNext steps:")
        print("1. Run smoke tests: pytest -m smoke -v")
        print("2. Run full validation: python run_tests.py --progressive")
        print("3. Run specific category: pytest -m sanity -v")
    else:
        print("\n⚠️  Test suite validation failed!")
        print("Please fix the issues above before running tests.")
    
    return 0 if all_valid else 1

if __name__ == "__main__":
    sys.exit(main())