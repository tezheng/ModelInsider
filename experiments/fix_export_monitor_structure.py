#!/usr/bin/env python3
"""
Fix for export_monitor.py structure issues.

The problem is that _find_immediate_children was partially moved out of the class,
causing syntax errors. This file shows the correct structure.
"""

# The _find_immediate_children function should be completely OUTSIDE any class
def _find_immediate_children(parent_path: str, hierarchy: dict) -> list:
    """
    Find immediate children of a module path with improved compound handling.
    
    This implementation combines:
    - Clear universal design that works with any module hierarchy
    - Sophisticated handling of compound components (e.g., layer.0)
    - Efficient single-pass algorithm
    - Smart sorting with numeric awareness
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
                immediate_child = prefix + parts[0]
                if immediate_child in hierarchy:
                    immediate_children.add(immediate_child)
    
    # Convert to list and apply smart sorting
    children = list(immediate_children)
    
    def smart_sort_key(path: str):
        """Smart sorting key for numeric indices and attention components."""
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


# Then the HTPReportWriter class with its methods properly indented
class HTPReportWriter:
    # ... other methods ...
    
    def _generate_full_hierarchy(self, data):
        """Generate complete hierarchy tree without truncation."""
        if not data.hierarchy or not data.tagged_nodes:
            return ""
        
        lines = []
        hierarchy = data.hierarchy
        tagged_nodes = data.tagged_nodes
        
        # Same logic as console but WITHOUT truncation
        def print_module_and_ops(path: str, prefix: str = "", is_last: bool = True):
            info = hierarchy.get(path, {})
            class_name = info.get("class_name", "Unknown")
            tag = info.get("traced_tag", "")
            
            # Count nodes for this module
            if path:
                module_tag_prefix = tag
                node_count = len([n for n, t in tagged_nodes.items() if t == tag or t.startswith(tag + "/")])
            else:
                node_count = len(tagged_nodes)
            
            # Print module line
            if path:
                connector = "└── " if is_last else "├── "
                lines.append(f"{prefix}{connector}{class_name}: {path} ({node_count} nodes)")
                new_prefix = prefix + ("    " if is_last else "│   ")
            else:
                lines.append(f"{class_name} ({node_count} ONNX nodes)")
                new_prefix = ""
            
            # Find and print children
            children = _find_immediate_children(path, hierarchy)
            
            # Special sort to handle numeric indices and maintain order
            def sort_key(p):
                parts = p.split(".")
                result = []
                for part in parts:
                    try:
                        result.append((0, int(part)))
                    except ValueError:
                        if part == "self":
                            result.append((1, "a_self"))
                        elif part == "output":
                            result.append((1, "z_output"))
                        else:
                            result.append((1, part))
                return result
            
            children.sort(key=sort_key)
            
            # Print operations for this module
            if tag and node_count > 0 and path != "":
                # Check if this module has direct operations
                has_direct_ops = True
                if children:
                    child_tags = [hierarchy.get(c, {}).get("traced_tag", "") for c in children]
                    child_node_count = sum(len([n for n, t in tagged_nodes.items() if t == ct]) for ct in child_tags)
                    if child_node_count >= node_count:
                        has_direct_ops = False
                
                if has_direct_ops:
                    # Group operations by type
                    from collections import defaultdict
                    ops_by_type = defaultdict(list)
                    for node_name in [n for n, t in tagged_nodes.items() if t == tag]:
                        if "/" in node_name:
                            parts = node_name.split("/")
                            op_type = parts[-1].split("_")[0] if "_" in parts[-1] else parts[-1]
                        else:
                            op_type = node_name.split("_")[0] if "_" in node_name else node_name
                        ops_by_type[op_type].append(node_name)
                    
                    # Print each operation type
                    op_items = sorted(ops_by_type.items())
                    for i, (op_type, op_nodes) in enumerate(op_items):
                        is_last_op = (i == len(op_items) - 1) and len(children) == 0
                        op_connector = "└── " if is_last_op else "├── "
                        
                        if len(op_nodes) > 1:
                            lines.append(f"{new_prefix}{op_connector}{op_type} ({len(op_nodes)} ops)")
                        else:
                            # Single op - check if it needs full path
                            node_name = op_nodes[0]
                            if any(x in node_name for x in ["LayerNorm", "Gather", "Gemm", "Tanh", "Div", "Shape", "Slice", "Softmax", "MatMul", "Add", "Relu"]):
                                if "/" in node_name:
                                    lines.append(f"{new_prefix}{op_connector}{op_type}: {node_name}")
                                else:
                                    lines.append(f"{new_prefix}{op_connector}{op_type}")
                            else:
                                lines.append(f"{new_prefix}{op_connector}{op_type}")
            
            # Print children
            for i, child in enumerate(children):
                is_last_child = (i == len(children) - 1)
                print_module_and_ops(child, new_prefix, is_last_child)
        
        print_module_and_ops("")
        return "\n".join(lines)
    
    def _get_nodes_by_tag(self, tagged_nodes: dict) -> dict:
        """Get nodes grouped by tag."""
        from collections import defaultdict
        nodes_by_tag = defaultdict(lambda: defaultdict(list))
        for node_name, tag in tagged_nodes.items():
            op_type = node_name.split('_')[0] if '_' in node_name else node_name
            nodes_by_tag[tag][op_type].append(node_name)
        return dict(nodes_by_tag)
    
    def flush(self):
        """Write the complete report with full hierarchy (no truncation)."""
        # ... implementation ...
        pass