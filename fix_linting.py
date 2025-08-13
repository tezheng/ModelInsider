#!/usr/bin/env python3
"""Fix linting issues in test files."""

import re
from pathlib import Path

def fix_unused_variables(file_path: Path) -> None:
    """Remove or use unused variables."""
    content = file_path.read_text()
    
    # Fix specific patterns
    replacements = {
        # Remove unused variable assignments
        r"(\s+)model_meta = metadata\[model_name\]\n": "",
        r"(\s+)strategies = list\(results\.keys\(\)\)\n": "",
        r"(\s+)common_ops = torch_functions & functional_functions\n": "",
        r"(\s+)graphml_str = converter\.convert\(simple_onnx_model\)\n": r"\1converter.convert(simple_onnx_model)\n",
        r"(\s+)key_elem = ET\.SubElement\(.*?\)\n": "",
        r"(\s+)graph = ET\.SubElement\(.*?\)\n": r"\1ET.SubElement(root, 'graph', id='G', edgedefault='directed')\n",
        r"(\s+)actual_edges = len\(.*?\)\n": "",
        r"(\s+)stats = exporter\.export\(": r"\1exporter.export(",
        
        # Fix f-strings without placeholders
        r'f"Operation type count mismatches:\\n"': r'"Operation type count mismatches:\n"',
        r'f"Node structure mismatches:\\n"': r'"Node structure mismatches:\n"',
        
        # Fix except Exception as e patterns
        r"except Exception as e:\n(\s+)# If export fails.*\n(\s+)pass": "except Exception:\n\\1# If export fails, check if it was expected\n\\2pass",
        
        # Fix converter = in error handling
        r"(\s+)converter = ONNXToGraphMLConverter\(htp_metadata_path=str\(metadata_file\)\)\n": "",
    }
    
    for pattern, replacement in replacements.items():
        content = re.sub(pattern, replacement, content, flags=re.MULTILINE)
    
    file_path.write_text(content)


def fix_broad_exceptions(file_path: Path) -> None:
    """Replace broad Exception catches with specific ones."""
    content = file_path.read_text()
    
    replacements = {
        # Fix broad exception catches
        r"with pytest\.raises\(Exception\):(\s+)# Could be IsADirectoryError.*": 
            "with pytest.raises((IsADirectoryError, PermissionError, OSError)):  # Directory-related errors",
        r"with pytest\.raises\(Exception\):\n(\s+)converter\.convert\(malformed_onnx_file\)":
            "with pytest.raises((onnx.onnx_cpp2py_export.checker.ValidationError, ValueError)):\n\\1converter.convert(malformed_onnx_file)",
        r"with pytest\.raises\(Exception\):(\s+)# Should handle parse errors":
            "with pytest.raises((ET.ParseError, ValueError)):  # XML parse errors",
    }
    
    for pattern, replacement in replacements.items():
        content = re.sub(pattern, replacement, content, flags=re.MULTILINE)
    
    file_path.write_text(content)


def fix_contextlib_suppress(file_path: Path) -> None:
    """Use contextlib.suppress instead of try-except-pass."""
    content = file_path.read_text()
    
    # Add import if needed
    if "contextlib.suppress" not in content and "try:" in content and "except Exception:" in content:
        if "import contextlib" not in content:
            content = re.sub(r"(from __future__ import annotations\n)", r"\1import contextlib\n", content)
    
    # Replace try-except-pass with contextlib.suppress
    pattern = r"(\s+)try:\n(\s+)# This is the core method.*\n(\s+)exporter\._convert_model_to_onnx.*\n(\s+)except Exception:\n(\s+)# We expect.*\n(\s+)pass"
    replacement = r"\1with contextlib.suppress(Exception):\n\2# This is the core method that should preserve dict structure\n\3exporter._convert_model_to_onnx(mock_model, self.output_path, {})"
    
    content = re.sub(pattern, replacement, content, flags=re.MULTILINE)
    
    file_path.write_text(content)


def fix_multiple_with_statements(file_path: Path) -> None:
    """Combine nested with statements."""
    content = file_path.read_text()
    
    # Pattern to match nested with statements
    pattern = r"(\s+)with (.*?) as (.*?):\n(\s+)with (.*?) as (.*?):"
    
    # Check if this pattern exists and combine them
    def replace_nested_with(match):
        indent = match.group(1)
        ctx1 = match.group(2)
        var1 = match.group(3) if match.group(3) else ""
        ctx2 = match.group(5)
        var2 = match.group(6) if match.group(6) else ""
        
        if var1 and var2:
            return f"{indent}with {ctx1} as {var1}, {ctx2} as {var2}:"
        elif var1:
            return f"{indent}with {ctx1} as {var1}, {ctx2}:"
        elif var2:
            return f"{indent}with {ctx1}, {ctx2} as {var2}:"
        else:
            return f"{indent}with {ctx1}, {ctx2}:"
    
    content = re.sub(pattern, replace_nested_with, content)
    
    file_path.write_text(content)


def main():
    """Fix all linting issues."""
    test_files = [
        "tests/fixtures/base_test.py",
        "tests/graphml/test_converter.py",
        "tests/graphml/test_depth_validation.py",
        "tests/graphml/test_error_handling.py",
        "tests/graphml/test_graphml_structure.py",
        "tests/test_sam_export_regression.py",
        "tests/unit/test_core/test_operation_config.py",
        "tests/unit/test_core/test_topology_preservation.py",
        "tests/test_hierarchy_exporter.py",
    ]
    
    for file_name in test_files:
        file_path = Path(file_name)
        if file_path.exists():
            print(f"Fixing {file_name}...")
            fix_unused_variables(file_path)
            fix_broad_exceptions(file_path)
            fix_contextlib_suppress(file_path)
            fix_multiple_with_statements(file_path)
    
    print("Done fixing linting issues!")


if __name__ == "__main__":
    main()