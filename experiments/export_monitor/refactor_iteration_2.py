#!/usr/bin/env python3
"""
Refactoring iteration 2: Extract hardcoded values and magic numbers
"""

import re
from pathlib import Path

def analyze_hardcoded_values():
    """Analyze export_monitor.py for hardcoded values."""
    
    export_monitor_path = Path("/home/zhengte/modelexport_allmodels/modelexport/strategies/htp/export_monitor.py")
    
    with open(export_monitor_path) as f:
        content = f.read()
    
    # Find all hardcoded values
    hardcoded_values = {
        "Display Constants": [],
        "Magic Numbers": [],
        "Hardcoded Strings": [],
        "File Suffixes": [],
        "Formatting": []
    }
    
    # Find magic numbers
    numbers = re.findall(r'\b\d+\b', content)
    for num in set(numbers):
        if int(num) > 1 and num not in ['17', '100']:  # Skip common ones
            lines = [i+1 for i, line in enumerate(content.splitlines()) if num in line]
            if lines:
                hardcoded_values["Magic Numbers"].append(f"{num} on lines: {lines[:5]}")
    
    # Find hardcoded strings
    strings = re.findall(r'"([^"]+)"', content) + re.findall(r"'([^']+)'", content)
    for s in set(strings):
        if len(s) > 3 and not s.startswith('%') and not s.startswith('{'):
            count = content.count(f'"{s}"') + content.count(f"'{s}'")
            if count > 1:
                hardcoded_values["Hardcoded Strings"].append(f'"{s}" used {count} times')
    
    # Find file suffixes
    suffixes = [
        "_htp_metadata.json",
        "_htp_export_report.txt",
        "_full_report.txt"
    ]
    for suffix in suffixes:
        if suffix in content:
            hardcoded_values["File Suffixes"].append(suffix)
    
    # Find formatting constants
    formatting = [
        ("SEPARATOR_LENGTH = 80", 80),
        ("MODULE_TREE_MAX_LINES = 100", 100),
        ("NODE_TREE_MAX_LINES = 30", 30),
        ("TOP_NODES_COUNT = 20", 20),
        ("width=80", 80),
        ("width=120", 120),
        ('"-" * 60', 60),
        ('"-" * 30', 30),
        ('"=" * 80', 80)
    ]
    for pattern, value in formatting:
        if str(value) in content:
            hardcoded_values["Formatting"].append(f"{pattern}")
    
    # Print findings
    print("🔍 HARDCODED VALUES ANALYSIS")
    print("=" * 60)
    
    for category, items in hardcoded_values.items():
        if items:
            print(f"\n{category}:")
            for item in items[:10]:  # Limit output
                print(f"  - {item}")
    
    return hardcoded_values

def create_config_class():
    """Create a configuration class to replace hardcoded values."""
    
    config_code = '''
class HTPExportConfig:
    """Configuration for HTP Export Monitor - no hardcoded values."""
    
    # Display formatting
    SEPARATOR_WIDTH = 80
    CONSOLE_WIDTH = 80
    WIDE_CONSOLE_WIDTH = 120
    
    # Tree display limits
    MODULE_TREE_MAX_LINES = 100  # Full module hierarchy
    NODE_TREE_MAX_LINES = 50     # ONNX nodes with operations (increased from 30)
    TOP_NODES_DISPLAY_COUNT = 20  # Top N nodes by hierarchy
    MAX_OPERATION_TYPES = 5       # Operations to show per module
    
    # Section separators
    MAJOR_SEPARATOR = "=" * SEPARATOR_WIDTH
    MINOR_SEPARATOR = "-" * 60
    SHORT_SEPARATOR = "-" * 30
    
    # Depth limits
    MAX_TREE_DEPTH = 4            # Maximum nesting depth for trees
    NODE_DETAIL_MAX_DEPTH = 3     # Show operation details up to this depth
    
    # File naming
    METADATA_SUFFIX = "_htp_metadata.json"
    REPORT_SUFFIX = "_htp_export_report.txt"
    FULL_REPORT_SUFFIX = "_full_report.txt"
    
    # Export settings
    DEFAULT_OPSET_VERSION = 17
    DEFAULT_CONSTANT_FOLDING = True
    DEFAULT_ONNX_VERBOSE = False
    
    # Step display
    TOTAL_EXPORT_STEPS = 8
    
    # Formatting templates
    STEP_HEADER_TEMPLATE = "📋 STEP {current}/{total}: {title}"
    NODE_COUNT_TEMPLATE = "{class_name}: {name} ({count} nodes)"
    OPERATION_TEMPLATE = "{op_type} ({count} ops)"
    OPERATION_SINGLE_TEMPLATE = "{op_type}: {node_name}"
    
    # Console emoji/icons
    ICONS = {
        "success": "✅",
        "error": "❌", 
        "warning": "⚠️",
        "info": "ℹ️",
        "progress": "🔄",
        "complete": "🎉",
        "tree": "🌳",
        "stats": "📊",
        "file": "📄",
        "folder": "📁",
        "tag": "🏷️",
        "build": "🏗️",
        "gear": "🔧",
        "package": "📦",
        "chart": "📈",
        "target": "🎯",
        "robot": "🤖",
        "brain": "🧠",
        "clipboard": "📋"
    }
'''
    
    return config_code

def create_refactored_methods():
    """Create refactored methods using config."""
    
    refactored_code = '''
# Example refactored methods using config

def _print_header(self, text: str) -> None:
    """Print section header using config."""
    print("")
    print(HTPExportConfig.MAJOR_SEPARATOR)
    print(text)
    print(HTPExportConfig.MAJOR_SEPARATOR)

def _print_minor_header(self, text: str) -> None:
    """Print minor section header."""
    print(f"\\n{HTPExportConfig.ICONS['tree']} {text}:")
    print(HTPExportConfig.MINOR_SEPARATOR)

def _format_step_header(self, step_num: int, title: str) -> str:
    """Format step header using template."""
    return HTPExportConfig.STEP_HEADER_TEMPLATE.format(
        current=step_num,
        total=HTPExportConfig.TOTAL_EXPORT_STEPS,
        title=title
    )

# Replace print() with rich console
def _console_print(self, message: str, style: str = None) -> None:
    """Print using rich console instead of print()."""
    if hasattr(self, 'console') and self.console:
        self.console.print(message, style=style or "")
    else:
        print(message)
'''
    
    return refactored_code

def main():
    """Run refactoring analysis."""
    print("🔧 EXPORT MONITOR REFACTORING - ITERATION 2")
    print("=" * 60)
    
    # Analyze hardcoded values
    analyze_hardcoded_values()
    
    # Create config class
    print("\n📝 PROPOSED CONFIG CLASS:")
    print(create_config_class())
    
    # Show refactored examples
    print("\n📝 EXAMPLE REFACTORED METHODS:")
    print(create_refactored_methods())
    
    print("\n✅ Refactoring plan ready!")
    print("\nNext steps:")
    print("1. Add HTPExportConfig class to export_monitor.py")
    print("2. Replace all hardcoded values with config references")
    print("3. Replace print() with rich console methods")
    print("4. Test that everything still works")

if __name__ == "__main__":
    main()