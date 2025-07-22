"""
Shared hierarchy building utilities for HTP export.

This module provides shared logic for building module hierarchy trees
that both console output and report generation can use consistently.
The logic is extracted from the working console implementation.
"""

from __future__ import annotations

from typing import Any


def find_immediate_children(parent_path: str, hierarchy: dict[str, Any]) -> list[str]:
    """
    Find immediate children of a path using the WORKING logic from console writer.
    
    This is the core hierarchy traversal logic that handles compound patterns
    like 'layer.0', 'blocks.1', etc. correctly.
    
    Args:
        parent_path: Parent module path (empty string for root)
        hierarchy: Module hierarchy dictionary from TracingHierarchyBuilder
        
    Returns:
        List of immediate child paths, properly sorted
        
    Examples:
        >>> # Root children
        >>> find_immediate_children("", {"embeddings": {...}, "encoder": {...}})
        ["embeddings", "encoder"]
        
        >>> # With compound patterns  
        >>> find_immediate_children("encoder", {
        ...     "encoder.layer.0": {...},
        ...     "encoder.layer.1": {...},
        ...     "encoder.layer.0.attention": {...}
        ... })
        ["encoder.layer.0", "encoder.layer.1"]  # layer.0 is immediate despite having dots
    """
    if parent_path == "":
        # Root case
        return sorted([p for p in hierarchy if p and "." not in p])
    
    # Non-root case
    prefix = parent_path + "."
    immediate = []
    
    for path in hierarchy:
        if not path.startswith(prefix) or path == parent_path:
            continue
        
        suffix = path[len(prefix):]
        
        # Check if immediate child - this is the KEY logic that was missing from reports!
        if "." not in suffix:
            # Simple immediate child (e.g., "attention" under "encoder")
            immediate.append(path)
        elif suffix.count(".") == 1 and suffix.split(".")[1].isdigit():
            # Compound pattern like layer.0 - treat as immediate child
            # This handles ResNet patterns: encoder.layer.0, encoder.layer.1, etc.
            immediate.append(path)
    
    # Custom sort that handles numeric parts properly
    def sort_key(path):
        parts = path.split(".")
        result = []
        for part in parts:
            if part.isdigit():
                result.append((0, int(part)))  # Numbers sort first, numerically
            else:
                result.append((1, part))       # Text sorts second, alphabetically
        return result
    
    return sorted(immediate, key=sort_key)


def build_ascii_tree(
    hierarchy: dict[str, Any], 
    max_depth: int | None = None, 
    max_lines: int | None = None,
    show_counts: bool = False,
    node_counts: dict[str, int] | None = None
) -> list[str]:
    """
    Generate ASCII tree representation of module hierarchy.
    
    Args:
        hierarchy: Module hierarchy dictionary
        max_depth: Maximum tree depth (None for unlimited)
        max_lines: Maximum output lines (None for unlimited)
        show_counts: Whether to show node counts
        node_counts: Dictionary mapping hierarchy tags to node counts
        
    Returns:
        List of tree lines (can be joined with newlines)
    """
    lines = []
    
    # Find root
    root_info = hierarchy.get("")
    # Handle both dict and ModuleInfo objects
    if hasattr(root_info, "class_name"):
        root_name = root_info.class_name if root_info else "Model"
        root_tag = root_info.traced_tag if hasattr(root_info, "traced_tag") else "/Model"
    else:
        root_name = root_info.get("class_name", "Model") if root_info else "Model"
        root_tag = root_info.get("traced_tag", "/Model") if root_info else "/Model"
    
    # Add root with optional count
    if show_counts and node_counts:
        root_count = node_counts.get(root_tag, 0)
        lines.append(f"{root_name} ({root_count} nodes)")
    else:
        lines.append(root_name)
    
    def add_children(parent_path: str, prefix: str = "", is_last: bool = True, depth: int = 0):
        # Check depth limit
        if max_depth is not None and depth >= max_depth:
            children = find_immediate_children(parent_path, hierarchy)
            if children:
                lines.append(f"{prefix}└── ... ({len(children)} more)")
            return
        
        # Check line limit  
        if max_lines is not None and len(lines) >= max_lines:
            return
            
        # Find immediate children using shared logic
        children = find_immediate_children(parent_path, hierarchy)
        
        for i, child_path in enumerate(children):
            # Check line limit
            if max_lines is not None and len(lines) >= max_lines:
                break
                
            child_info = hierarchy.get(child_path)
            if not child_info:
                continue
            
            is_last_child = i == len(children) - 1
            
            # Build the tree line
            if parent_path == "":  # Direct children of root
                line_prefix = "└── " if is_last_child else "├── "
                continuation = "    " if is_last_child else "│   "
            else:
                line_prefix = prefix + ("└── " if is_last_child else "├── ")
                continuation = prefix + ("    " if is_last_child else "│   ")
            
            # Show class name and full path as scope
            display_name = child_path
            # Handle both dict and ModuleInfo objects
            if hasattr(child_info, "class_name"):
                class_name = child_info.class_name
                child_tag = child_info.traced_tag if hasattr(child_info, "traced_tag") else ""
            else:
                class_name = child_info.get("class_name", "Unknown")
                child_tag = child_info.get("traced_tag", "")
            
            # Build line with optional count
            if show_counts and node_counts and child_tag:
                child_count = node_counts.get(child_tag, 0)
                line = f"{line_prefix}{class_name}: {display_name} ({child_count} nodes)"
            else:
                line = f"{line_prefix}{class_name}: {display_name}"
            lines.append(line)
            
            # Recursively add children
            add_children(child_path, continuation, is_last_child, depth + 1)
    
    # Start from root
    add_children("", "", True, 0)
    
    return lines


def count_nodes_per_tag(tagged_nodes: dict[str, str]) -> dict[str, int]:
    """
    Count nodes per hierarchy tag.
    
    Args:
        tagged_nodes: Dictionary mapping node names to hierarchy tags
        
    Returns:
        Dictionary mapping hierarchy tags to node counts
    """
    from collections import defaultdict
    
    node_counts = defaultdict(int)
    for _node_name, tag in tagged_nodes.items():
        # Count nodes for each level of the hierarchy  
        parts = tag.split("/")
        for i in range(1, len(parts) + 1):
            prefix = "/".join(parts[:i])
            if prefix:
                node_counts[prefix] += 1
    return dict(node_counts)


def count_direct_and_total_nodes(tagged_nodes: dict[str, str]) -> tuple[dict[str, int], dict[str, int]]:
    """
    Count direct nodes (not in children) and total nodes (including children) per hierarchy tag.
    
    Args:
        tagged_nodes: Dictionary mapping node names to hierarchy tags
        
    Returns:
        Tuple of (direct_counts, total_counts) dictionaries
    """
    from collections import defaultdict
    
    direct_counts = defaultdict(int)
    total_counts = defaultdict(int)
    
    # First pass: count direct nodes
    for _node_name, tag in tagged_nodes.items():
        if tag:
            direct_counts[tag] += 1
    
    # Second pass: accumulate total counts (direct + children)
    for _node_name, tag in tagged_nodes.items():
        # Count for all parent paths
        parts = tag.split("/")
        for i in range(1, len(parts) + 1):
            prefix = "/".join(parts[:i])
            if prefix:
                total_counts[prefix] += 1
    
    return dict(direct_counts), dict(total_counts)


def build_rich_tree(hierarchy: dict[str, Any], show_counts: bool = False, tagged_nodes: dict[str, str] | None = None):
    """
    Build Rich Tree object for console display.
    
    Args:
        hierarchy: Module hierarchy dictionary
        show_counts: Whether to show node counts from tagged_nodes
        tagged_nodes: Optional ONNX node mapping for counts
        
    Returns:
        Rich Tree object ready for console display
    """
    from rich.text import Text
    from rich.tree import Tree
    
    # Count nodes per hierarchy if provided
    node_counts = count_nodes_per_tag(tagged_nodes) if tagged_nodes else None
    
    # Find root
    root_info = hierarchy.get("")
    # Handle both dict and ModuleInfo objects
    if hasattr(root_info, "class_name"):
        root_name = root_info.class_name if root_info else "Model"
        root_tag = root_info.traced_tag if hasattr(root_info, "traced_tag") else "/Model"
    else:
        root_name = root_info.get("class_name", "Model") if root_info else "Model"
        root_tag = root_info.get("traced_tag", "/Model") if root_info else "/Model"
    
    # Create root with optional count
    if show_counts and node_counts:
        root_count = node_counts.get(root_tag, 0)
        tree_label = f"[bold]{root_name}[/bold] ([bold cyan]{root_count}[/bold cyan] nodes)"
    else:
        tree_label = f"[bold]{root_name}[/bold]"
    
    tree = Tree(tree_label)
    
    def add_rich_children(parent_node, parent_path: str):
        children = find_immediate_children(parent_path, hierarchy)
        
        for child_path in children:
            child_info = hierarchy.get(child_path)
            if not child_info:
                continue
            
            # Create styled text
            # Handle both dict and ModuleInfo objects
            if hasattr(child_info, "class_name"):
                child_class = child_info.class_name
                child_tag = child_info.traced_tag if hasattr(child_info, "traced_tag") else ""
            else:
                child_class = child_info.get("class_name", "Unknown")
                child_tag = child_info.get("traced_tag", "")
            display_name = child_path
            
            node_text = Text()
            node_text.append(child_class, style="bold")
            node_text.append(": ", style="")
            node_text.append(display_name, style="dim")
            
            # Add count if available
            if show_counts and node_counts and child_tag:
                child_count = node_counts.get(child_tag, 0)
                node_text.append(" (", style="")
                node_text.append(str(child_count), style="bold cyan")
                node_text.append(" nodes)", style="")
            
            # Add to tree and recurse
            child_node = parent_node.add(node_text)
            add_rich_children(child_node, child_path)
    
    # Build tree
    add_rich_children(tree, "")
    
    return tree