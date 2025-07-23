#!/usr/bin/env python3
"""
Test the fixed export monitor to verify:
1. Console output has proper ANSI styling
2. Text report contains all console output without styling
3. Metadata report section has complete data
"""

import sys
import json
import re
from pathlib import Path

# Add parent paths for imports
sys.path.insert(0, str(Path(__file__).parent))

from export_monitor_fixed import (
    HTPExportMonitor, HTPExportData, HTPExportStep, TextStyler
)


def create_test_data():
    """Create test data similar to bert-tiny export."""
    return HTPExportData(
        model_name="prajjwal1/bert-tiny",
        model_class="BertModel",
        total_modules=48,
        total_parameters=4385536,
        output_path="test_model.onnx",
        embed_hierarchy_attributes=True,
        hierarchy={
            "": {"class_name": "BertModel", "traced_tag": "/BertModel"},
            "embeddings": {
                "class_name": "BertEmbeddings", 
                "traced_tag": "/BertModel/BertEmbeddings"
            },
            "encoder": {
                "class_name": "BertEncoder",
                "traced_tag": "/BertModel/BertEncoder"
            },
            "encoder.layer.0": {
                "class_name": "BertLayer",
                "traced_tag": "/BertModel/BertEncoder/BertLayer.0"
            },
            "encoder.layer.0.attention": {
                "class_name": "BertAttention",
                "traced_tag": "/BertModel/BertEncoder/BertLayer.0/BertAttention"
            },
            "encoder.layer.0.attention.self": {
                "class_name": "BertSdpaSelfAttention",
                "traced_tag": "/BertModel/BertEncoder/BertLayer.0/BertAttention/BertSdpaSelfAttention"
            },
            "pooler": {
                "class_name": "BertPooler",
                "traced_tag": "/BertModel/BertPooler"
            }
        },
        execution_steps=36,
        output_names=["last_hidden_state", "pooler_output"],
        total_nodes=136,
        tagged_nodes={
            f"node_{i}": "/BertModel/BertEncoder/BertLayer.0/BertAttention/BertSdpaSelfAttention"
            for i in range(35)
        },
        tagging_stats={
            "direct_matches": 83,
            "parent_matches": 34,
            "root_fallbacks": 19,
            "empty_tags": 0
        },
        coverage=100.0,
        export_time=2.35,
        onnx_size_mb=17.5,
        steps={
            "input_generation": {
                "model_type": "bert",
                "task": "feature-extraction",
                "inputs": {
                    "input_ids": {"shape": [2, 16], "dtype": "torch.int64"},
                    "attention_mask": {"shape": [2, 16], "dtype": "torch.int64"},
                    "token_type_ids": {"shape": [2, 16], "dtype": "torch.int64"}
                }
            },
            "onnx_export": {
                "config": {
                    "opset_version": 17,
                    "do_constant_folding": True,
                    "verbose": False,
                    "input_names": ["input_ids", "attention_mask", "token_type_ids"]
                }
            },
            "tagger_creation": {
                "root_tag": "/BertModel"
            }
        }
    )


def test_console_output(monitor: HTPExportMonitor, data: HTPExportData):
    """Test console output has proper ANSI codes."""
    print("\n" + "=" * 80)
    print("Testing Console Output with ANSI Styling")
    print("=" * 80)
    
    # Log all steps
    for step in HTPExportStep:
        monitor.log_step(step, data)
    
    # Get console output
    console_output = monitor.get_console_output()
    
    # Check for ANSI codes
    ansi_patterns = {
        "Bold cyan numbers": r'\033\[1;36m\d+',
        "Bold parentheses": r'\033\[1m\(\033\[0m',
        "Green True": r'\033\[3;92mTrue\033\[0m',
        "Red False": r'\033\[3;91mFalse\033\[0m',
        "Green strings": r"\033\[32m'[^']+'\033\[0m",
        "Magenta paths": r'\033\[35m[^/]+/\033\[0m',
        "Bright magenta": r'\033\[95m\w+\033\[0m'
    }
    
    import re
    print("\nChecking for ANSI patterns:")
    for name, pattern in ansi_patterns.items():
        matches = re.findall(pattern, console_output)
        print(f"  {name}: {'✓' if matches else '✗'} ({len(matches)} found)")
        if matches and len(matches) <= 3:
            for match in matches[:3]:
                print(f"    Example: {repr(match)}")
    
    # Show sample of output
    print("\nSample console output (first 500 chars):")
    print("-" * 60)
    print(repr(console_output[:500]))
    
    return console_output


def test_text_report(monitor: HTPExportMonitor):
    """Test text report has complete content without ANSI."""
    print("\n" + "=" * 80)
    print("Testing Text Report (Plain Text)")
    print("=" * 80)
    
    # Finalize to create files
    paths = monitor.finalize()
    
    # Read text report
    with open(paths["report_path"], 'r') as f:
        report_content = f.read()
    
    # Check no ANSI codes
    ansi_escape = re.compile(r'\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])')
    ansi_matches = ansi_escape.findall(report_content)
    print(f"\nANSI codes in report: {'✗ NONE' if not ansi_matches else f'✓ FOUND {len(ansi_matches)}'}")
    
    # Check content completeness
    expected_content = [
        "HTP EXPORT FULL REPORT",
        "MODEL PREPARATION",
        "INPUT GENERATION & VALIDATION",
        "HIERARCHY BUILDING",
        "Module Hierarchy:",
        "ONNX EXPORT",
        "NODE TAGGER CREATION",
        "ONNX NODE TAGGING",
        "Top 20 Nodes by Hierarchy:",
        "Complete HF Hierarchy with ONNX Nodes:",
        "SAVE ONNX MODEL",
        "EXPORT COMPLETE",
        "Export Statistics:"
    ]
    
    print("\nContent completeness check:")
    for content in expected_content:
        found = content in report_content
        print(f"  {content}: {'✓' if found else '✗'}")
    
    # Check for truncation
    print(f"\nReport size: {len(report_content)} characters")
    print(f"Contains '...' truncation: {'✓ YES' if '...' in report_content else '✗ NO'}")
    
    # Show sample
    print("\nSample report content (lines 100-120):")
    print("-" * 60)
    lines = report_content.split('\n')
    for i, line in enumerate(lines[100:120], 100):
        print(f"{i:3}: {line}")
    
    return report_content


def test_metadata_report(monitor: HTPExportMonitor):
    """Test metadata has complete report section."""
    print("\n" + "=" * 80)
    print("Testing Metadata Report Section")
    print("=" * 80)
    
    # Get metadata
    metadata = monitor.get_metadata()
    
    # Check report section
    report_section = metadata.get("report", {}).get("steps", {})
    
    expected_steps = [
        "model_preparation",
        "input_generation",
        "hierarchy_building",
        "onnx_export",
        "node_tagger_creation",
        "node_tagging",
        "model_save",
        "export_complete"
    ]
    
    print("\nReport section completeness:")
    for step in expected_steps:
        has_step = step in report_section
        print(f"  {step}: {'✓' if has_step else '✗'}")
        
        if has_step:
            step_data = report_section[step]
            # Check for detailed data (not just completion status)
            has_details = len(step_data) > 2  # More than just completed & timestamp
            print(f"    Has detailed data: {'✓' if has_details else '✗'} ({len(step_data)} fields)")
    
    # Show sample step data
    if "node_tagging" in report_section:
        print("\nSample - node_tagging step data:")
        print(json.dumps(report_section["node_tagging"], indent=2)[:500] + "...")
    
    return metadata


def main():
    """Run all tests."""
    print("Fixed Export Monitor Test Suite")
    print("=" * 80)
    
    # Create monitor and test data
    output_path = "experiments/export_monitor/comprehensive_fix/test_output/test_model.onnx"
    monitor = HTPExportMonitor(output_path, "prajjwal1/bert-tiny", verbose=True)
    data = create_test_data()
    
    # Run tests
    console_output = test_console_output(monitor, data)
    report_content = test_text_report(monitor)
    metadata = test_metadata_report(monitor)
    
    # Summary
    print("\n" + "=" * 80)
    print("Test Summary")
    print("=" * 80)
    
    # Console output check
    has_ansi = '\033[' in console_output
    print(f"1. Console output has ANSI styling: {'✓ YES' if has_ansi else '✗ NO'}")
    
    # Text report check
    has_no_ansi = '\033[' not in report_content
    is_complete = all(content in report_content for content in [
        "MODULE PREPARATION", "HIERARCHY BUILDING", "NODE TAGGING", "EXPORT COMPLETE"
    ])
    print(f"2. Text report is plain text: {'✓ YES' if has_no_ansi else '✗ NO'}")
    print(f"3. Text report is complete: {'✓ YES' if is_complete else '✗ NO'}")
    
    # Metadata check
    report_steps = metadata.get("report", {}).get("steps", {})
    has_all_steps = len(report_steps) >= 8
    has_details = all(len(step_data) > 2 for step_data in report_steps.values())
    print(f"4. Metadata has all report steps: {'✓ YES' if has_all_steps else '✗ NO'}")
    print(f"5. Metadata has detailed data: {'✓ YES' if has_details else '✗ NO'}")
    
    print("\n✅ Test complete!")


if __name__ == "__main__":
    main()