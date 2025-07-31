#!/usr/bin/env python3
"""
Improved hierarchy tree implementation merging best features from both versions.

Combines:
- Universal design and clarity from htp
- Sophisticated compound component handling from export_monitor
- Efficient single-pass algorithm
- Custom sorting with numeric and attention-aware ordering
"""

def find_immediate_children_improved(parent_path: str, hierarchy: dict) -> list:
    """
    Find immediate children of a module path with improved compound handling.
    
    This implementation combines the best of both approaches:
    - Clear universal design that works with any module hierarchy
    - Sophisticated handling of compound components (e.g., layer.0)
    - Efficient single-pass algorithm
    - Smart sorting with numeric awareness
    
    Args:
        parent_path: Parent module path (empty string for root)
        hierarchy: Full hierarchy dictionary
        
    Returns:
        Sorted list of immediate child paths
    """
    if parent_path == "":
        # Root case: find top-level modules (no dots in path)
        return sorted([p for p in hierarchy if p and '.' not in p])
    
    # Non-root case: find children under parent path
    prefix = parent_path + "."
    immediate_children = set()
    
    for path in hierarchy:
        if not path.startswith(prefix) or path == parent_path:
            continue
            
        # Extract the portion after parent path
        suffix = path[len(prefix):]
        
        if not suffix:
            continue
        
        # Check for immediate vs nested children
        if '.' not in suffix:
            # Simple immediate child (e.g., parent.child)
            immediate_children.add(path)
        else:
            # Compound pattern handling (e.g., parent.layer.0)
            parts = suffix.split('.')
            
            # Check if this is a numbered pattern (name.number)
            if len(parts) == 2 and parts[1].isdigit():
                # This is a compound immediate child (e.g., encoder.layer.0)
                immediate_children.add(path)
            else:
                # Multi-level descendant - find the immediate child in the path
                # For "encoder.layer.0.attention.self", if parent is "encoder",
                # the immediate child is "encoder.layer"
                immediate_child = prefix + parts[0]
                if immediate_child in hierarchy:
                    immediate_children.add(immediate_child)
    
    # Convert to list and apply smart sorting
    children = list(immediate_children)
    
    def smart_sort_key(path: str):
        """
        Smart sorting key that handles:
        - Numeric indices (0, 1, 2... not 0, 10, 11...)
        - Attention component ordering (self before output)
        - Alphabetical ordering for others
        """
        # Extract just the child part for sorting
        child_part = path[len(prefix):] if path.startswith(prefix) else path
        parts = child_part.split('.')
        
        result = []
        for part in parts:
            if part.isdigit():
                # Numeric part - sort as integer
                result.append((0, int(part)))
            elif part == "self":
                # Special case: self comes first in attention
                result.append((1, "a_self"))
            elif part == "output":
                # Special case: output comes last in attention
                result.append((1, "z_output"))
            else:
                # Regular string sorting
                result.append((1, part))
        
        return result
    
    children.sort(key=smart_sort_key)
    return children


def print_hierarchy_tree_improved(hierarchy: dict, tagged_nodes: dict = None,
                                  max_lines: int = None, use_rich: bool = False) -> str:
    """
    Print module hierarchy tree with improved formatting.
    
    Features:
    - Shows ONNX node counts when tagged_nodes provided
    - Supports both plain text and Rich library rendering
    - Optional line truncation for console output
    - Shows operation details for modules
    
    Args:
        hierarchy: Module hierarchy dictionary
        tagged_nodes: Optional dict of node_name -> tag mappings
        max_lines: Optional maximum lines (None for full output)
        use_rich: Whether to use Rich library for rendering
        
    Returns:
        Formatted hierarchy tree as string
    """
    lines = []
    
    def count_module_nodes(tag: str) -> int:
        """Count nodes belonging to a module (including descendants)."""
        if not tagged_nodes or not tag:
            return 0
        return len([n for n, t in tagged_nodes.items() 
                   if t == tag or t.startswith(tag + "/")])
    
    def print_module(path: str, prefix: str = "", is_last: bool = True):
        """Recursively print module and its children."""
        info = hierarchy.get(path, {})
        class_name = info.get("class_name", "Unknown")
        tag = info.get("traced_tag", "")
        
        # Count nodes for this module
        node_count = count_module_nodes(tag) if path else len(tagged_nodes or {})
        
        # Format module line
        if path:
            connector = "└── " if is_last else "├── "
            line = f"{prefix}{connector}{class_name}: {path}"
            if tagged_nodes:
                line += f" ({node_count} nodes)"
        else:
            line = f"{class_name}"
            if tagged_nodes:
                line += f" ({node_count} ONNX nodes)"
        
        lines.append(line)
        
        # Update prefix for children
        new_prefix = prefix + ("    " if is_last else "│   ") if path else ""
        
        # Get and print children
        children = find_immediate_children_improved(path, hierarchy)
        
        # Optionally show operations for this module
        if tagged_nodes and tag and node_count > 0 and path:
            # Check if module has direct operations
            child_tags = [hierarchy.get(c, {}).get("traced_tag", "") for c in children]
            child_nodes = sum(count_module_nodes(ct) for ct in child_tags)
            
            if child_nodes < node_count:
                # This module has direct operations
                # Group operations by type
                from collections import defaultdict
                ops_by_type = defaultdict(list)
                
                for node_name, node_tag in tagged_nodes.items():
                    if node_tag == tag:
                        # Extract operation type
                        op_type = node_name.split("/")[-1].split("_")[0]
                        ops_by_type[op_type].append(node_name)
                
                # Print operations
                op_items = sorted(ops_by_type.items())
                for i, (op_type, nodes) in enumerate(op_items):
                    is_last_op = (i == len(op_items) - 1) and not children
                    op_connector = "└── " if is_last_op else "├── "
                    
                    if len(nodes) > 1:
                        lines.append(f"{new_prefix}{op_connector}{op_type} ({len(nodes)} ops)")
                    else:
                        lines.append(f"{new_prefix}{op_connector}{op_type}")
        
        # Print children
        for i, child in enumerate(children):
            is_last_child = (i == len(children) - 1)
            print_module(child, new_prefix, is_last_child)
    
    # Start from root
    print_module("")
    
    # Handle truncation if requested
    if max_lines and len(lines) > max_lines:
        truncated_lines = lines[:max_lines]
        remaining = len(lines) - max_lines
        truncated_lines.append(f"... and {remaining} more lines (truncated for console)")
        truncated_lines.append(f"(showing {max_lines}/{len(lines)} lines)")
        return "\n".join(truncated_lines)
    
    return "\n".join(lines)


# Example usage and test
if __name__ == "__main__":
    # Test hierarchy with compound components
    test_hierarchy = {
        "": {"class_name": "BertModel", "traced_tag": "/BertModel"},
        "embeddings": {"class_name": "BertEmbeddings", "traced_tag": "/BertModel/Embeddings"},
        "encoder": {"class_name": "BertEncoder", "traced_tag": "/BertModel/Encoder"},
        "encoder.layer": {"class_name": "ModuleList", "traced_tag": "/BertModel/Encoder/Layer"},
        "encoder.layer.0": {"class_name": "BertLayer", "traced_tag": "/BertModel/Encoder/Layer/0"},
        "encoder.layer.0.attention": {"class_name": "BertAttention", "traced_tag": "/BertModel/Encoder/Layer/0/Attention"},
        "encoder.layer.0.attention.self": {"class_name": "BertSelfAttention", "traced_tag": "/BertModel/Encoder/Layer/0/Attention/Self"},
        "encoder.layer.0.attention.output": {"class_name": "BertSelfOutput", "traced_tag": "/BertModel/Encoder/Layer/0/Attention/Output"},
        "encoder.layer.1": {"class_name": "BertLayer", "traced_tag": "/BertModel/Encoder/Layer/1"},
        "pooler": {"class_name": "BertPooler", "traced_tag": "/BertModel/Pooler"},
    }
    
    # Test finding children
    print("Root children:", find_immediate_children_improved("", test_hierarchy))
    print("Encoder children:", find_immediate_children_improved("encoder", test_hierarchy))
    print("Layer.0 children:", find_immediate_children_improved("encoder.layer.0", test_hierarchy))
    
    print("\nHierarchy Tree:")
    print(print_hierarchy_tree_improved(test_hierarchy))