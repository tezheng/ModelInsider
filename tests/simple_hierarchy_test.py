#!/usr/bin/env python3
"""
Simple test for ONNX hierarchy preservation
"""

import torch
import torch.nn as nn
import onnx
import json
from typing import Dict, List, Any

class SimpleHierarchicalModel(nn.Module):
    """A simple hierarchical model that works with ONNX export"""
    
    def __init__(self):
        super().__init__()
        self.embeddings = nn.Sequential(
            nn.Linear(10, 20),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        
        self.encoder = nn.Sequential(
            nn.Linear(20, 15),
            nn.ReLU(),
            nn.Linear(15, 10),
            nn.ReLU()
        )
        
        self.classifier = nn.Linear(10, 3)
    
    def forward(self, x):
        x = self.embeddings(x)
        x = self.encoder(x)
        return self.classifier(x)

def extract_hierarchy(model: nn.Module) -> Dict[str, Dict[str, Any]]:
    """Extract module hierarchy"""
    hierarchy = {}
    
    for name, module in model.named_modules():
        if name:  # Skip root
            parts = name.split('.')
            hierarchy[name] = {
                'type': type(module).__name__,
                'depth': len(parts),
                'parent': '.'.join(parts[:-1]) if len(parts) > 1 else 'root',
                'leaf_name': parts[-1],
                'is_leaf': len(list(module.children())) == 0
            }
    
    return hierarchy

def export_with_metadata(model: nn.Module, dummy_input: torch.Tensor, output_path: str):
    """Export model with hierarchy metadata"""
    
    # Extract hierarchy first
    hierarchy = extract_hierarchy(model)
    
    print(f"Exporting model with {len(hierarchy)} modules")
    
    # Standard export
    torch.onnx.export(
        model,
        dummy_input,
        output_path,
        export_params=True,
        opset_version=11,
        input_names=['input'],
        output_names=['output'],
        verbose=False
    )
    
    # Load and enhance with metadata
    onnx_model = onnx.load(output_path)
    
    # Add hierarchy as metadata
    hierarchy_meta = onnx.StringStringEntryProto()
    hierarchy_meta.key = "module_hierarchy"
    hierarchy_meta.value = json.dumps(hierarchy, indent=2)
    onnx_model.metadata_props.append(hierarchy_meta)
    
    # Enhanced node naming strategy
    # Use parameter names to map operations to modules
    param_to_module = {}
    for name, module in model.named_modules():
        if name:
            for param_name, param in module.named_parameters(recurse=False):
                full_param_name = f"{name}.{param_name}"
                param_to_module[full_param_name.replace('.', '_')] = name
    
    print(f"Parameter to module mapping: {len(param_to_module)} entries")
    
    # Enhance node names and add attributes
    enhanced_nodes = 0
    for i, node in enumerate(onnx_model.graph.node):
        # Try to find source module from inputs
        source_module = None
        
        for input_name in node.input:
            # Check if input matches a parameter
            for param_key, module_name in param_to_module.items():
                if param_key in input_name:
                    source_module = module_name
                    break
            if source_module:
                break
        
        if source_module:
            # Add module path attribute
            attr = onnx.AttributeProto()
            attr.name = "source_module"
            attr.type = onnx.AttributeProto.STRING
            attr.s = source_module.encode('utf-8')
            node.attribute.append(attr)
            
            # Enhance node name
            if not node.name or not source_module in node.name:
                node.name = f"{source_module}.{node.op_type}_{i}"
            
            enhanced_nodes += 1
    
    # Save enhanced model
    enhanced_path = output_path.replace('.onnx', '_enhanced.onnx')
    onnx.save(onnx_model, enhanced_path)
    
    print(f"Enhanced {enhanced_nodes} nodes with hierarchy info")
    print(f"Enhanced model saved to: {enhanced_path}")
    
    return onnx_model, hierarchy

def analyze_enhanced_model(onnx_path: str):
    """Analyze the enhanced ONNX model"""
    print(f"\n=== Analyzing Enhanced Model: {onnx_path} ===")
    
    onnx_model = onnx.load(onnx_path)
    
    # Check metadata
    hierarchy_found = False
    for prop in onnx_model.metadata_props:
        if prop.key == "module_hierarchy":
            hierarchy = json.loads(prop.value)
            hierarchy_found = True
            print(f"Found hierarchy metadata with {len(hierarchy)} modules")
            break
    
    if not hierarchy_found:
        print("No hierarchy metadata found")
        return
    
    # Analyze nodes
    nodes_with_module_info = 0
    module_to_ops = {}
    
    for node in onnx_model.graph.node:
        source_module = None
        
        # Check for source_module attribute
        for attr in node.attribute:
            if attr.name == "source_module":
                source_module = attr.s.decode('utf-8')
                break
        
        if source_module:
            nodes_with_module_info += 1
            if source_module not in module_to_ops:
                module_to_ops[source_module] = []
            module_to_ops[source_module].append({
                'name': node.name,
                'op_type': node.op_type
            })
    
    print(f"Nodes with module info: {nodes_with_module_info}/{len(onnx_model.graph.node)}")
    
    # Show module to operations mapping
    print(f"\nModule to operations mapping:")
    for module, ops in module_to_ops.items():
        print(f"  {module}: {len(ops)} operations")
        for op in ops[:2]:  # Show first 2
            print(f"    - {op['op_type']} ({op['name']})")

def test_hierarchy_grouping(onnx_path: str):
    """Test grouping operations by hierarchy"""
    print(f"\n=== Testing Hierarchy Grouping ===")
    
    onnx_model = onnx.load(onnx_path)
    
    # Extract hierarchy
    hierarchy = None
    for prop in onnx_model.metadata_props:
        if prop.key == "module_hierarchy":
            hierarchy = json.loads(prop.value)
            break
    
    if not hierarchy:
        print("No hierarchy found")
        return
    
    # Group by hierarchy levels
    by_depth = {}
    for module_name, info in hierarchy.items():
        depth = info['depth']
        if depth not in by_depth:
            by_depth[depth] = []
        by_depth[depth].append(module_name)
    
    print("Modules by depth:")
    for depth in sorted(by_depth.keys()):
        print(f"  Depth {depth}: {len(by_depth[depth])} modules")
        for module in by_depth[depth][:3]:  # Show first 3
            print(f"    - {module}: {hierarchy[module]['type']}")
    
    # Group operations by parent modules
    print(f"\nGrouping operations by parent modules:")
    
    # Find all operations for each top-level module
    top_level_modules = [name for name, info in hierarchy.items() if info['depth'] == 1]
    
    for top_module in top_level_modules:
        # Find all child modules
        child_modules = [name for name in hierarchy.keys() if name.startswith(top_module + '.')]
        all_modules = [top_module] + child_modules
        
        # Count operations for this module group
        total_ops = 0
        for node in onnx_model.graph.node:
            for attr in node.attribute:
                if attr.name == "source_module":
                    source_module = attr.s.decode('utf-8')
                    if source_module in all_modules:
                        total_ops += 1
                    break
        
        print(f"  {top_module}: {total_ops} operations (across {len(all_modules)} modules)")

def main():
    """Main test function"""
    print("=== Simple Hierarchy Preservation Test ===\n")
    
    # Create test model
    model = SimpleHierarchicalModel()
    print(f"Model architecture:\n{model}\n")
    
    # Extract and show hierarchy
    hierarchy = extract_hierarchy(model)
    print("Module hierarchy:")
    for name, info in hierarchy.items():
        indent = "  " * info['depth']
        print(f"{indent}{name}: {info['type']}")
    
    # Test export
    dummy_input = torch.randn(1, 10)
    output_path = "simple_hierarchical.onnx"
    
    enhanced_model, hierarchy = export_with_metadata(model, dummy_input, output_path)
    
    # Analyze results
    analyze_enhanced_model(output_path.replace('.onnx', '_enhanced.onnx'))
    
    # Test hierarchy grouping
    test_hierarchy_grouping(output_path.replace('.onnx', '_enhanced.onnx'))

if __name__ == "__main__":
    main()