#!/usr/bin/env python3
"""
Test the fixed hierarchy tree logic
"""

from rich.console import Console
from rich.tree import Tree


def test_original_logic():
    """Test the original logic that's not working."""
    print("🔍 Testing ORIGINAL logic:")
    
    hierarchy = {
        "": {"class_name": "BertModel"},
        "embeddings": {"class_name": "BertEmbeddings"},
        "encoder": {"class_name": "BertEncoder"},
        "encoder.layer.0": {"class_name": "BertLayer"},
        "encoder.layer.0.attention": {"class_name": "BertAttention"},
        "encoder.layer.1": {"class_name": "BertLayer"},
        "pooler": {"class_name": "BertPooler"},
    }
    
    tree = Tree("BertModel")
    
    def add_children_original(parent_tree: Tree, parent_path: str):
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
                    continue
            else:
                if "." in path:
                    continue
            
            children.append((path, info))
        
        children.sort(key=lambda x: x[0])
        
        for path, info in children:
            class_name = info.get("class_name", "Unknown")
            child_name = path.split(".")[-1]
            child_tree = parent_tree.add(f"{class_name}: {child_name}")
            add_children_original(child_tree, path)
    
    add_children_original(tree, "")
    
    console = Console()
    console.print(tree)
    print()

def test_fixed_logic():
    """Test the fixed logic."""
    print("✅ Testing FIXED logic:")
    
    hierarchy = {
        "": {"class_name": "BertModel"},
        "embeddings": {"class_name": "BertEmbeddings"},
        "encoder": {"class_name": "BertEncoder"},
        "encoder.layer.0": {"class_name": "BertLayer"},
        "encoder.layer.0.attention": {"class_name": "BertAttention"},
        "encoder.layer.0.attention.self": {"class_name": "BertSdpaSelfAttention"},
        "encoder.layer.0.attention.output": {"class_name": "BertSelfOutput"},
        "encoder.layer.0.intermediate": {"class_name": "BertIntermediate"},
        "encoder.layer.0.intermediate.intermediate_act_fn": {"class_name": "GELUActivation"},
        "encoder.layer.0.output": {"class_name": "BertOutput"},
        "encoder.layer.1": {"class_name": "BertLayer"},
        "encoder.layer.1.attention": {"class_name": "BertAttention"},
        "encoder.layer.1.attention.self": {"class_name": "BertSdpaSelfAttention"},
        "encoder.layer.1.attention.output": {"class_name": "BertSelfOutput"},
        "encoder.layer.1.intermediate": {"class_name": "BertIntermediate"},
        "encoder.layer.1.intermediate.intermediate_act_fn": {"class_name": "GELUActivation"},
        "encoder.layer.1.output": {"class_name": "BertOutput"},
        "pooler": {"class_name": "BertPooler"},
    }
    
    tree = Tree("BertModel")
    
    def add_children_fixed(parent_tree: Tree, parent_path: str, level: int = 0):
        # Find direct children
        children = []
        
        for path, info in hierarchy.items():
            if not path or path == parent_path:
                continue
            
            # Check if this is a direct child of parent_path
            if parent_path:
                # Child must start with parent_path + "."
                if not path.startswith(parent_path + "."):
                    continue
                # Get the part after parent path
                remainder = path[len(parent_path) + 1:]
                # If remainder has dots, it's not a direct child
                if "." in remainder:
                    continue
            else:
                # For root (empty parent_path), direct children have no dots
                if "." in path:
                    continue
            
            children.append((path, info))
        
        # Sort children by path
        children.sort(key=lambda x: x[0])
        
        # Add each child to the tree
        for path, info in children:
            class_name = info.get("class_name", "Unknown")
            # Get just the last component of the path
            child_name = path.split(".")[-1]
            
            # Add to tree
            child_tree = parent_tree.add(f"{class_name}: {child_name}")
            
            # Recursively add this child's children
            add_children_fixed(child_tree, path, level + 1)
    
    add_children_fixed(tree, "")
    
    console = Console()
    console.print(tree)

if __name__ == "__main__":
    test_original_logic()
    print("\n" + "="*60 + "\n")
    test_fixed_logic()