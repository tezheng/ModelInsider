#!/usr/bin/env python3
"""
Fix for iteration 2: Fix the Complete HF Hierarchy with ONNX Nodes display
"""

def create_fixed_print_hierarchy_with_nodes():
    """Create the fixed _print_hierarchy_with_nodes method."""
    
    fixed_code = '''
def _print_hierarchy_with_nodes(self, hierarchy: dict, tagged_nodes: dict[str, str], total_nodes: int) -> None:
    """Print hierarchy tree with ONNX node operations."""
    from collections import defaultdict
    
    print("\\n🌳 Complete HF Hierarchy with ONNX Nodes:")
    print("-" * 60)
    
    # Group nodes by tag and operation type
    nodes_by_tag = defaultdict(lambda: defaultdict(list))
    for node_name, tag in tagged_nodes.items():
        # Extract operation type from node name
        op_type = node_name.split('_')[0] if '_' in node_name else node_name
        nodes_by_tag[tag][op_type].append(node_name)
    
    # Build tree with node counts
    root_info = hierarchy.get("", {})
    root_name = root_info.get("class_name", "Model")
    
    print(f"{root_name} ({total_nodes} ONNX nodes)")
    
    # Build parent-child mapping to handle components with dots
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
    
    def print_module_with_nodes(path: str, level: int = 1, line_count: list | None = None):
        if line_count is None:
            line_count = [0]
            
        # Increase limit to show more content
        if line_count[0] >= 50:  # Increased from 30
            return
            
        if level > 4:  # Increased from 3 to show more depth
            return
            
        # Get children from our mapping
        children = parent_to_children.get(path, [])
        
        # Sort children
        children.sort()
        
        # Print each child with its nodes
        for child_path in children:
            if line_count[0] >= 50:  # Increased limit
                break
                
            child_info = hierarchy.get(child_path, {})
            class_name = child_info.get("class_name", "Unknown")
            
            # Get display name - everything after the parent path
            if path:
                display_name = child_path[len(path) + 1:]
            else:
                display_name = child_path
            
            tag = child_info.get("traced_tag", "")
            
            # Count nodes for this tag
            node_count = len([n for n, t in tagged_nodes.items() if t == tag])
            
            indent = "│   " * (level - 1) + "├── "
            print(f"{indent}{class_name}: {display_name} ({node_count} nodes)")
            line_count[0] += 1
            
            # Show operation breakdown for this module
            if tag in nodes_by_tag and level <= 3 and node_count > 0:
                ops = nodes_by_tag[tag]
                sorted_ops = sorted(ops.items(), key=lambda x: len(x[1]), reverse=True)
                
                for op_type, op_nodes in sorted_ops[:5]:  # Show top 5 operation types
                    if line_count[0] >= 50:  # Increased limit
                        break
                        
                    op_indent = "│   " * level + "├── "
                    count = len(op_nodes)
                    if count > 1:
                        print(f"{op_indent}{op_type} ({count} ops)")
                    else:
                        # For single ops, show the full name
                        print(f"{op_indent}{op_type}: {op_nodes[0]}")
                    line_count[0] += 1
            
            # Recurse for children
            print_module_with_nodes(child_path, level + 1, line_count)
    
    line_counter = [1]  # Start at 1 for the root line
    print_module_with_nodes("", line_count=line_counter)
    
    # Add truncation notice
    lines_shown = line_counter[0]
    if lines_shown < total_nodes:
        print(f"... and {total_nodes - lines_shown} more lines (truncated for console)")
    print(f"(showing {lines_shown}/{total_nodes} lines)")
'''
    
    return fixed_code

if __name__ == "__main__":
    print(create_fixed_print_hierarchy_with_nodes())
    print("\n✅ Fix ready to apply!")
    print("\nNext steps:")
    print("1. Replace _print_hierarchy_with_nodes in export_monitor.py")
    print("2. Run iteration test again")