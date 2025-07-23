#!/usr/bin/env python3
"""
Debug the tree rendering to understand why it's only showing 4 lines
"""

from rich.tree import Tree
from rich.console import Console
import io
import json

def debug_tree_rendering():
    """Debug why the tree is only showing 4 lines."""
    
    # Load actual hierarchy data
    with open("experiments/export_monitor/iterations/iteration_001/model_htp_metadata.json") as f:
        metadata = json.load(f)
    
    hierarchy = metadata["modules"]
    
    print(f"Total modules in hierarchy: {len(hierarchy)}")
    print("\nBuilding tree...")
    
    # Build tree using the exact same logic
    root_info = hierarchy.get("", {})
    root_name = root_info.get("class_name", "Model")
    tree = Tree(root_name)
    
    added_count = 0
    
    def add_children(parent_tree: Tree, parent_path: str, level: int = 0):
        nonlocal added_count
        
        # Debug: print what we're looking for
        print(f"\n{'  ' * level}Looking for children of '{parent_path}' (level {level}):")
        
        children = []
        for path, info in hierarchy.items():
            if not path or path == parent_path:
                continue
            
            # Check if direct child
            if parent_path:
                if not path.startswith(parent_path + "."):
                    continue
                suffix = path[len(parent_path) + 1:]
                if "." in suffix:
                    if level <= 1:  # Debug for first two levels
                        print(f"{'  ' * (level + 1)}Skipping {path} - suffix '{suffix}' contains dot")
                    continue
            else:
                if "." in path:
                    continue
            
            children.append((path, info))
            print(f"{'  ' * (level + 1)}Found child: {path}")
        
        children.sort(key=lambda x: x[0])
        
        for path, info in children:
            class_name = info.get("class_name", "Unknown")
            child_name = path.split(".")[-1]
            
            child_tree = parent_tree.add(f"{class_name}: {child_name}")
            added_count += 1
            add_children(child_tree, path, level + 1)
    
    add_children(tree, "")
    
    print(f"\nTotal nodes added to tree: {added_count}")
    
    # Render the tree
    buffer = io.StringIO()
    temp_console = Console(file=buffer, width=80, force_terminal=True)
    temp_console.print(tree)
    
    lines = buffer.getvalue().splitlines()
    print(f"\nRendered tree has {len(lines)} lines")
    
    print("\nFull tree output:")
    print("-" * 60)
    for i, line in enumerate(lines):
        print(f"{i+1:3}: {line}")

if __name__ == "__main__":
    debug_tree_rendering()