#!/usr/bin/env python3
"""
Fix for iteration 1: Properly handle hierarchy tree with components containing dots
"""

def create_fixed_print_hierarchy_tree():
    """Create the fixed _print_hierarchy_tree method."""
    
    fixed_code = '''
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
    
    # Build a parent-child mapping first to handle components with dots
    parent_to_children = {}
    for path in hierarchy:
        if not path:  # Skip root
            continue
        
        # Find the parent by looking for the longest existing prefix
        parent_path = ""
        path_parts = path.split(".")
        
        # Try to find the longest matching parent
        for i in range(len(path_parts) - 1, 0, -1):
            potential_parent = ".".join(path_parts[:i])
            if potential_parent in hierarchy:
                parent_path = potential_parent
                break
        
        # Add to parent's children list
        if parent_path not in parent_to_children:
            parent_to_children[parent_path] = []
        parent_to_children[parent_path].append(path)
    
    def add_children(parent_tree: Tree, parent_path: str, level: int = 0):
        # Get children from our mapping
        children = parent_to_children.get(parent_path, [])
        
        # Sort children
        children.sort()
        
        # Add each child to tree
        for child_path in children:
            info = hierarchy.get(child_path, {})
            class_name = info.get("class_name", "Unknown")
            
            # Get display name - everything after the parent path
            if parent_path:
                display_name = child_path[len(parent_path) + 1:]
            else:
                display_name = child_path
            
            # Create tree node
            child_tree = parent_tree.add(f"{class_name}: {display_name}")
            
            # Recursively add this child's children
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
    
    # Print all lines up to max_lines
    for i, line in enumerate(lines):
        if i >= max_lines:
            remaining = len(lines) - i
            if remaining > 0:
                print(f"... and {remaining} more lines (truncated for console)")
            break
        print(line)
    
    # Show line count summary
    if len(lines) <= max_lines:
        print(f"(showing {len(lines)}/{len(lines)} lines)")
    else:
        print(f"(showing {max_lines}/{len(lines)} lines)")
'''
    
    return fixed_code

def test_fix():
    """Test the fix with sample data."""
    print("Testing hierarchy tree fix...")
    
    # Create test hierarchy
    hierarchy = {
        "": {"class_name": "BertModel"},
        "embeddings": {"class_name": "BertEmbeddings"},
        "encoder": {"class_name": "BertEncoder"},
        "encoder.layer.0": {"class_name": "BertLayer"},
        "encoder.layer.0.attention": {"class_name": "BertAttention"},
        "encoder.layer.0.attention.self": {"class_name": "BertSdpaSelfAttention"},
        "encoder.layer.0.attention.output": {"class_name": "BertSelfOutput"},
        "encoder.layer.1": {"class_name": "BertLayer"},
        "pooler": {"class_name": "BertPooler"},
    }
    
    # Build parent-child mapping
    parent_to_children = {}
    for path in hierarchy:
        if not path:  # Skip root
            continue
        
        # Find the parent by looking for the longest existing prefix
        parent_path = ""
        path_parts = path.split(".")
        
        # Try to find the longest matching parent
        for i in range(len(path_parts) - 1, 0, -1):
            potential_parent = ".".join(path_parts[:i])
            if potential_parent in hierarchy:
                parent_path = potential_parent
                break
        
        # Add to parent's children list
        if parent_path not in parent_to_children:
            parent_to_children[parent_path] = []
        parent_to_children[parent_path].append(path)
    
    print("\nParent-child mapping:")
    for parent, children in sorted(parent_to_children.items()):
        print(f"  '{parent}' -> {children}")
    
    print("\n✅ Fix ready to apply!")
    print("\nNext steps:")
    print("1. Replace _print_hierarchy_tree in export_monitor.py")
    print("2. Run iteration test again")

if __name__ == "__main__":
    print(create_fixed_print_hierarchy_tree())
    print("\n" + "="*80 + "\n")
    test_fix()