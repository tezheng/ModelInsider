#!/usr/bin/env python3
"""
Research ONNX metadata and node naming capabilities for hierarchy preservation
"""

import torch
import torch.nn as nn
import onnx
from onnx import helper, numpy_helper
import json
from typing import Dict, List, Any, Optional
from pathlib import Path

class HierarchicalModel(nn.Module):
    """A simple hierarchical model to test metadata capabilities"""
    
    def __init__(self):
        super().__init__()
        self.embeddings = nn.Sequential(
            nn.Linear(784, 256, bias=True),  # embeddings.linear
            nn.ReLU(),                       # embeddings.relu
            nn.Dropout(0.1)                  # embeddings.dropout
        )
        
        self.encoder = nn.ModuleList([
            self._make_layer(256, 128, layer_idx=0),
            self._make_layer(128, 64, layer_idx=1),
        ])
        
        self.classifier = nn.Linear(64, 10)
    
    def _make_layer(self, in_dim: int, out_dim: int, layer_idx: int):
        """Create a transformer-like layer"""
        layer = nn.Sequential()
        layer.add_module(f'attention', nn.MultiheadAttention(in_dim, num_heads=4, batch_first=True))
        layer.add_module(f'norm1', nn.LayerNorm(in_dim))
        layer.add_module(f'feed_forward', nn.Sequential(
            nn.Linear(in_dim, out_dim * 2),
            nn.ReLU(),
            nn.Linear(out_dim * 2, out_dim)
        ))
        layer.add_module(f'norm2', nn.LayerNorm(out_dim))
        return layer
    
    def forward(self, x):
        # x shape: (batch, seq_len, features)
        x = self.embeddings(x)
        
        for i, layer in enumerate(self.encoder):
            # Simplified attention + FFN
            if i == 0:
                attn_out, _ = layer.attention(x, x, x)
                x = layer.norm1(x + attn_out)
                ff_out = layer.feed_forward(x)
                x = layer.norm2(x + ff_out)
            else:
                # For second layer, adjust dimensions
                attn_out, _ = layer.attention(x[:, :, :128], x[:, :, :128], x[:, :, :128])
                x_adjusted = x[:, :, :128]
                x = layer.norm1(x_adjusted + attn_out)
                ff_out = layer.feed_forward(x)
                x = layer.norm2(x + ff_out)
        
        # Global average pooling and classification
        x = x.mean(dim=1)  # (batch, features)
        return self.classifier(x)

def extract_module_hierarchy(model: nn.Module) -> Dict[str, Dict[str, Any]]:
    """Extract module hierarchy information"""
    hierarchy = {}
    
    for name, module in model.named_modules():
        if name:  # Skip root module
            parts = name.split('.')
            hierarchy[name] = {
                'type': type(module).__name__,
                'depth': len(parts),
                'parent': '.'.join(parts[:-1]) if len(parts) > 1 else 'root',
                'leaf_name': parts[-1],
                'is_leaf': len(list(module.children())) == 0,
                'parameters': [param_name for param_name, _ in module.named_parameters(recurse=False)]
            }
    
    return hierarchy

def export_with_hierarchy_info(model: nn.Module, dummy_input: torch.Tensor, 
                              output_path: str, hierarchy: Dict[str, Dict[str, Any]]):
    """Export model to ONNX with hierarchy information preserved"""
    
    # Standard ONNX export
    torch.onnx.export(
        model,
        dummy_input,
        output_path,
        export_params=True,
        opset_version=11,
        do_constant_folding=True,
        input_names=['input'],
        output_names=['output'],
        verbose=False
    )
    
    # Load the exported model
    onnx_model = onnx.load(output_path)
    
    print(f"Original ONNX model nodes: {len(onnx_model.graph.node)}")
    
    # Add hierarchy metadata to the graph
    hierarchy_metadata = onnx.StringStringEntryProto()
    hierarchy_metadata.key = "module_hierarchy"
    hierarchy_metadata.value = json.dumps(hierarchy)
    onnx_model.metadata_props.append(hierarchy_metadata)
    
    # Try to map operations to modules through parameter names
    # This is a heuristic approach
    param_to_module = {}
    for module_name, module_info in hierarchy.items():
        for param_name in module_info['parameters']:
            full_param_name = f"{module_name}.{param_name}" if module_name else param_name
            param_to_module[full_param_name] = module_name
    
    # Add module path information to nodes where possible
    node_to_module_mapping = {}
    
    for i, node in enumerate(onnx_model.graph.node):
        # Try to infer module path from node inputs (parameters)
        potential_module = None
        
        for input_name in node.input:
            # Check if this input corresponds to a model parameter
            for param_name, module_name in param_to_module.items():
                if param_name.replace('.', '_') in input_name:
                    potential_module = module_name
                    break
            if potential_module:
                break
        
        if potential_module:
            # Add custom attribute to the node
            attr = onnx.AttributeProto()
            attr.name = "source_module"
            attr.type = onnx.AttributeProto.STRING
            attr.s = potential_module.encode('utf-8')
            node.attribute.append(attr)
            
            node_to_module_mapping[i] = potential_module
            
            # Also update node name to include module path
            if not node.name:
                node.name = f"{potential_module}_{node.op_type}_{i}"
            else:
                node.name = f"{potential_module}_{node.name}"
    
    # Save the modified model
    modified_path = output_path.replace('.onnx', '_with_hierarchy.onnx')
    onnx.save(onnx_model, modified_path)
    
    print(f"Enhanced ONNX model saved to: {modified_path}")
    print(f"Nodes with module mapping: {len(node_to_module_mapping)}")
    
    return onnx_model, node_to_module_mapping

def analyze_onnx_metadata(onnx_path: str):
    """Analyze the metadata and node structure of an ONNX model"""
    print(f"\n=== Analyzing ONNX Model: {onnx_path} ===")
    
    onnx_model = onnx.load(onnx_path)
    
    # Analyze graph metadata
    print("Graph metadata:")
    for prop in onnx_model.metadata_props:
        print(f"  {prop.key}: {prop.value[:100]}..." if len(prop.value) > 100 else f"  {prop.key}: {prop.value}")
    
    # Analyze nodes
    print(f"\nNodes: {len(onnx_model.graph.node)}")
    
    # Count nodes by operation type
    op_counts = {}
    nodes_with_custom_attrs = 0
    nodes_with_names = 0
    
    for node in onnx_model.graph.node:
        op_type = node.op_type
        op_counts[op_type] = op_counts.get(op_type, 0) + 1
        
        if node.name:
            nodes_with_names += 1
        
        # Check for custom attributes
        custom_attrs = [attr for attr in node.attribute if attr.name == "source_module"]
        if custom_attrs:
            nodes_with_custom_attrs += 1
    
    print(f"Nodes with names: {nodes_with_names}")
    print(f"Nodes with custom module attributes: {nodes_with_custom_attrs}")
    
    print(f"\nOperation types:")
    for op_type, count in sorted(op_counts.items(), key=lambda x: x[1], reverse=True):
        print(f"  {op_type}: {count}")
    
    # Show sample nodes with hierarchy info
    print(f"\nSample nodes with hierarchy info:")
    count = 0
    for node in onnx_model.graph.node:
        if count >= 5:
            break
        
        custom_attrs = [attr for attr in node.attribute if attr.name == "source_module"]
        if custom_attrs:
            module_path = custom_attrs[0].s.decode('utf-8')
            print(f"  {node.name}: {node.op_type} -> module: {module_path}")
            count += 1

def test_hierarchy_retrieval(onnx_path: str) -> Dict[str, List[str]]:
    """Test retrieving hierarchy information from ONNX model"""
    print(f"\n=== Testing Hierarchy Retrieval ===")
    
    onnx_model = onnx.load(onnx_path)
    
    # Extract hierarchy metadata
    hierarchy_metadata = None
    for prop in onnx_model.metadata_props:
        if prop.key == "module_hierarchy":
            hierarchy_metadata = json.loads(prop.value)
            break
    
    if not hierarchy_metadata:
        print("No hierarchy metadata found")
        return {}
    
    print(f"Retrieved hierarchy for {len(hierarchy_metadata)} modules")
    
    # Group nodes by module
    module_to_nodes = {}
    
    for i, node in enumerate(onnx_model.graph.node):
        module_path = None
        
        # Check custom attributes
        for attr in node.attribute:
            if attr.name == "source_module":
                module_path = attr.s.decode('utf-8')
                break
        
        if module_path:
            if module_path not in module_to_nodes:
                module_to_nodes[module_path] = []
            module_to_nodes[module_path].append({
                'index': i,
                'name': node.name,
                'op_type': node.op_type
            })
    
    print(f"\nModule to nodes mapping:")
    for module_path, nodes in module_to_nodes.items():
        print(f"  {module_path}: {len(nodes)} operations")
        for node in nodes[:2]:  # Show first 2 operations
            print(f"    - {node['op_type']} ({node['name']})")
    
    return module_to_nodes

def main():
    """Main research function"""
    print("=== ONNX Metadata and Hierarchy Research ===\n")
    
    # Create test model
    model = HierarchicalModel()
    
    # Extract hierarchy
    hierarchy = extract_module_hierarchy(model)
    
    print(f"Test model hierarchy:")
    print(f"  Total modules: {len(hierarchy)}")
    print(f"  Max depth: {max(info['depth'] for info in hierarchy.values())}")
    
    # Show hierarchy structure
    print(f"\nModule hierarchy:")
    for name, info in hierarchy.items():
        indent = "  " * info['depth']
        print(f"{indent}{name}: {info['type']} (leaf: {info['is_leaf']})")
    
    # Create dummy input
    dummy_input = torch.randn(2, 10, 784)  # batch_size=2, seq_len=10, features=784
    
    # Export with hierarchy information
    output_path = "hierarchical_model.onnx"
    enhanced_model, node_mapping = export_with_hierarchy_info(
        model, dummy_input, output_path, hierarchy
    )
    
    # Analyze the exported models
    analyze_onnx_metadata(output_path)
    analyze_onnx_metadata(output_path.replace('.onnx', '_with_hierarchy.onnx'))
    
    # Test hierarchy retrieval
    module_to_nodes = test_hierarchy_retrieval(output_path.replace('.onnx', '_with_hierarchy.onnx'))
    
    return hierarchy, module_to_nodes

if __name__ == "__main__":
    hierarchy, module_to_nodes = main()