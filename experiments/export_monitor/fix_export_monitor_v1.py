#!/usr/bin/env python3
"""
Fix export monitor issues - Iteration 1
Goal: Fix the Module Hierarchy tree to show full nested structure like baseline
"""

import json
from pathlib import Path

def analyze_hierarchy_data():
    """Analyze the hierarchy data structure to understand the issue."""
    
    # Load metadata from iteration 1
    metadata_path = Path("experiments/export_monitor/iterations/iteration_001/model_htp_metadata.json")
    with open(metadata_path) as f:
        metadata = json.load(f)
    
    modules = metadata.get("modules", {})
    
    print("🔍 Analyzing hierarchy data structure...")
    print(f"Total modules: {len(modules)}")
    print("\nModule paths:")
    
    # Sort paths to see structure
    paths = sorted(modules.keys())
    for path in paths:
        print(f"  '{path}' -> {modules[path].get('class_name')}")
    
    # Check for encoder layer paths
    print("\n🔍 Checking encoder layer paths:")
    encoder_paths = [p for p in paths if 'encoder' in p]
    for path in encoder_paths:
        print(f"  '{path}'")
    
    # Debug the path format
    print("\n🔍 Path format analysis:")
    if "encoder.layer.0" in modules:
        print("✓ Has 'encoder.layer.0' format")
    if "encoder/layer/0" in modules:
        print("✓ Has 'encoder/layer/0' format")
    
    # Check empty path
    if "" in modules:
        print(f"\n🔍 Root module (empty path): {modules['']}")

def create_fix():
    """Create a fix for the hierarchy tree display."""
    
    print("\n📝 Creating fix for HTPConsoleWriter._print_hierarchy_tree...")
    
    fix_code = '''
# Fix for _print_hierarchy_tree method in HTPConsoleWriter
# The issue is that the hierarchy paths don't match the expected format

def _print_hierarchy_tree(self, hierarchy: dict, max_lines: int | None = None) -> None:
    """Print module hierarchy as a tree."""
    if max_lines is None:
        max_lines = self.MODULE_TREE_MAX_LINES
        
    print("\\n🌳 Module Hierarchy:")
    print("-" * 60)
    
    # Build tree using Rich
    root_info = hierarchy.get("", {})
    root_name = root_info.get("class_name", "Model")
    tree = Tree(root_name)
    
    def add_children(parent_tree: Tree, parent_path: str, level: int = 0):
        # Get all paths and sort them
        all_paths = sorted(hierarchy.keys())
        
        # Find direct children of parent_path
        children = []
        for path in all_paths:
            if not path or path == parent_path:
                continue
                
            # Check if this is a direct child
            if parent_path:
                # Must start with parent path + dot
                if not path.startswith(parent_path + "."):
                    continue
                # Get the part after parent path
                remainder = path[len(parent_path) + 1:]
                # Should not have dots (direct child only)
                if "." in remainder:
                    continue
            else:
                # For root, check if path has no dots (direct child)
                if "." not in path:
                    children.append(path)
                    continue
                else:
                    continue
            
            children.append(path)
        
        # Add children to tree
        for child_path in children:
            info = hierarchy.get(child_path, {})
            class_name = info.get("class_name", "Unknown")
            
            # Get the child's name (last part of path)
            if "." in child_path:
                child_name = child_path.split(".")[-1]
            else:
                child_name = child_path
            
            # Create tree node
            child_tree = parent_tree.add(f"{class_name}: {child_name}")
            
            # Recursively add children
            add_children(child_tree, child_path, level + 1)
    
    # Start building from root
    add_children(tree, "")
    
    # Render the tree
    import io
    from rich.console import Console
    buffer = io.StringIO()
    temp_console = Console(file=buffer, width=80, force_terminal=True)
    temp_console.print(tree)
    
    # Get the rendered output
    lines = buffer.getvalue().splitlines()
    
    # Print all lines (don't truncate the hierarchy tree)
    for line in lines:
        print(line)
'''
    
    print(fix_code)
    
    print("\n✅ Fix ready to apply!")
    print("\nTo apply:")
    print("1. Replace the _print_hierarchy_tree method in export_monitor.py")
    print("2. Run the iteration script again to test")

if __name__ == "__main__":
    analyze_hierarchy_data()
    create_fix()