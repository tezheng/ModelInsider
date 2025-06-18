#!/usr/bin/env python3
"""
Analyze ViT model structure to understand hierarchy
"""

import torch
import torch.nn as nn
from transformers import AutoModel
import json
from typing import Dict, List, Any

def analyze_vit_structure():
    """Analyze the hierarchical structure of google/vit-base-patch16-224"""
    print("=== Analyzing ViT Model Structure ===")
    
    # Load the model
    model_name = "google/vit-base-patch16-224"
    model = AutoModel.from_pretrained(model_name)
    
    # Extract complete module hierarchy
    hierarchy = {}
    module_to_operations = {}
    
    print(f"Model type: {type(model).__name__}")
    print(f"Model architecture:\n{model}\n")
    
    # Extract module hierarchy with detailed information
    for name, module in model.named_modules():
        if name:  # Skip root module
            parts = name.split('.')
            hierarchy[name] = {
                'type': type(module).__name__,
                'parameters': sum(p.numel() for p in module.parameters()),
                'trainable_params': sum(p.numel() for p in module.parameters() if p.requires_grad),
                'direct_children': len(list(module.children())),
                'depth': len(parts),
                'parent': '.'.join(parts[:-1]) if len(parts) > 1 else 'root',
                'leaf_name': parts[-1],
                'is_leaf': len(list(module.children())) == 0
            }
    
    # Analyze hierarchy statistics
    total_modules = len(hierarchy)
    max_depth = max(h['depth'] for h in hierarchy.values())
    leaf_modules = sum(1 for h in hierarchy.values() if h['is_leaf'])
    
    print(f"Hierarchy Analysis:")
    print(f"  Total modules: {total_modules}")
    print(f"  Max depth: {max_depth}")
    print(f"  Leaf modules: {leaf_modules}")
    print(f"  Total parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Group by depth
    by_depth = {}
    for name, info in hierarchy.items():
        depth = info['depth']
        if depth not in by_depth:
            by_depth[depth] = []
        by_depth[depth].append(name)
    
    print(f"\nModules by depth:")
    for depth in sorted(by_depth.keys()):
        print(f"  Depth {depth}: {len(by_depth[depth])} modules")
        if depth <= 3:  # Show examples for shallow depths
            for name in by_depth[depth][:3]:
                print(f"    - {name}: {hierarchy[name]['type']}")
    
    # Analyze module types
    module_types = {}
    for info in hierarchy.values():
        module_type = info['type']
        if module_type not in module_types:
            module_types[module_type] = 0
        module_types[module_type] += 1
    
    print(f"\nModule types:")
    for module_type, count in sorted(module_types.items(), key=lambda x: x[1], reverse=True):
        print(f"  {module_type}: {count}")
    
    # Extract specific patterns for ViT
    attention_modules = [name for name in hierarchy.keys() if 'attention' in name]
    layer_modules = [name for name in hierarchy.keys() if 'layer.' in name]
    
    print(f"\nViT-specific patterns:")
    print(f"  Attention modules: {len(attention_modules)}")
    print(f"  Transformer layers: {len([name for name in layer_modules if name.count('.') == 3])}")  # layer.N modules
    
    # Show sample attention hierarchy
    print(f"\nSample attention hierarchy (first layer):")
    layer_0_modules = [name for name in hierarchy.keys() if name.startswith('encoder.layer.0')]
    for name in sorted(layer_0_modules):
        print(f"  {name}: {hierarchy[name]['type']} (leaf: {hierarchy[name]['is_leaf']})")
    
    return model, hierarchy

def create_hierarchy_mapping(hierarchy: Dict[str, Any]) -> Dict[str, List[str]]:
    """Create mapping from parent modules to their children"""
    parent_to_children = {}
    
    for name, info in hierarchy.items():
        parent = info['parent']
        if parent not in parent_to_children:
            parent_to_children[parent] = []
        parent_to_children[parent].append(name)
    
    return parent_to_children

def save_hierarchy_analysis(hierarchy: Dict[str, Any], filename: str = "vit_hierarchy.json"):
    """Save hierarchy analysis to JSON file"""
    # Convert to serializable format
    serializable_hierarchy = {}
    for name, info in hierarchy.items():
        serializable_hierarchy[name] = {
            'type': info['type'],
            'parameters': info['parameters'],
            'depth': info['depth'],
            'parent': info['parent'],
            'leaf_name': info['leaf_name'],
            'is_leaf': info['is_leaf']
        }
    
    with open(filename, 'w') as f:
        json.dump(serializable_hierarchy, f, indent=2)
    
    print(f"\nHierarchy saved to {filename}")

if __name__ == "__main__":
    model, hierarchy = analyze_vit_structure()
    parent_to_children = create_hierarchy_mapping(hierarchy)
    save_hierarchy_analysis(hierarchy)
    
    print(f"\n=== Sample Parent-Child Relationships ===")
    # Show some examples
    for parent in list(parent_to_children.keys())[:5]:
        children = parent_to_children[parent]
        print(f"{parent}: {len(children)} children")
        for child in children[:3]:  # Show first 3 children
            print(f"  - {child}")