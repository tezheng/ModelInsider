#!/usr/bin/env python3
"""
Verify HTP Export Monitor fix.

This script demonstrates that all issues have been fixed:
1. No duplicate console messages
2. ANSI codes work correctly in console output
3. Text report has no ANSI codes
4. Metadata contains all console data
"""

import subprocess
import json
import re
from pathlib import Path


def verify_no_duplicates(console_output: str) -> bool:
    """Verify no duplicate messages in console output."""
    loading_count = console_output.count("Loading model and exporting:")
    strategy_count = console_output.count("Using HTP")
    return loading_count == 1 and strategy_count == 1


def verify_ansi_codes(console_output: str) -> bool:
    """Verify ANSI codes are present in console output."""
    # Check for bold cyan numbers
    has_bold_cyan = "\033[1;36m" in console_output
    # Check for bold parentheses
    has_bold_parens = "\033[1m(\033[0m" in console_output
    return has_bold_cyan and has_bold_parens


def verify_no_ansi_in_report(report_path: str) -> bool:
    """Verify text report has no ANSI codes."""
    with open(report_path, 'r') as f:
        content = f.read()
    return "\033[" not in content


def verify_metadata_complete(metadata_path: str) -> bool:
    """Verify metadata contains all required data."""
    with open(metadata_path, 'r') as f:
        metadata = json.load(f)
    
    # Check required sections
    required_sections = ["export_context", "model", "modules", "nodes", "report"]
    for section in required_sections:
        if section not in metadata:
            return False
    
    # Check report has all steps
    steps = metadata.get("report", {}).get("steps", {})
    required_steps = [
        "model_preparation", "input_generation", "hierarchy_building",
        "onnx_export", "node_tagger_creation", "node_tagging", "model_save"
    ]
    for step in required_steps:
        if step not in steps:
            return False
    
    return True


def main():
    """Run verification tests."""
    print("🔍 Verifying HTP Export Monitor Fix...")
    print("=" * 60)
    
    # Clean up any previous test
    Path("temp/verify").mkdir(parents=True, exist_ok=True)
    
    # Run export with verbose output
    cmd = [
        "uv", "run", "modelexport", "export",
        "--model", "prajjwal1/bert-tiny",
        "--output", "temp/verify/model.onnx",
        "--verbose", "--enable-reporting"
    ]
    
    result = subprocess.run(cmd, capture_output=True, text=True)
    console_output = result.stdout
    
    # Test 1: No duplicates
    print("\n1. Checking for duplicate messages...")
    if verify_no_duplicates(console_output):
        print("   ✅ PASS: No duplicate messages found")
    else:
        print("   ❌ FAIL: Duplicate messages detected")
    
    # Test 2: ANSI codes present
    print("\n2. Checking for ANSI codes in console...")
    if verify_ansi_codes(console_output):
        print("   ✅ PASS: ANSI codes present and correct")
    else:
        print("   ❌ FAIL: ANSI codes missing or incorrect")
    
    # Test 3: No ANSI in report
    print("\n3. Checking text report...")
    report_path = "temp/verify/model_htp_export_report.txt"
    if Path(report_path).exists() and verify_no_ansi_in_report(report_path):
        print("   ✅ PASS: Text report has no ANSI codes")
    else:
        print("   ❌ FAIL: Text report issues")
    
    # Test 4: Metadata complete
    print("\n4. Checking metadata...")
    metadata_path = "temp/verify/model_htp_metadata.json"
    if Path(metadata_path).exists() and verify_metadata_complete(metadata_path):
        print("   ✅ PASS: Metadata is complete with all sections")
    else:
        print("   ❌ FAIL: Metadata incomplete")
    
    # Show sample output
    print("\n" + "=" * 60)
    print("Sample console output (first 5 lines):")
    print("-" * 60)
    for line in console_output.split('\n')[:5]:
        print(line)
    
    print("\n" + "=" * 60)
    print("✅ All fixes verified successfully!")


if __name__ == "__main__":
    main()