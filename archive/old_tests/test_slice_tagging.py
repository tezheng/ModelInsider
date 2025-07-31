"""
Test slice operation tagging to verify if we actually fixed the issue or just bypassed it.
"""

import json
import tempfile

import torch
import torch.nn as nn

from modelexport.hierarchy_exporter import HierarchyExporter


class SimpleSliceModel(nn.Module):
    """Simple model that slices input tensor in forward pass."""
    
    def __init__(self):
        super().__init__()
        # Add a meaningful layer so we're not just pure PyTorch
        self.processor = nn.Linear(10, 10)
        
    def forward(self, x):
        # Slice the input tensor using different methods
        sliced = x[1:4]  # This should create ONNX Slice node
        
        # Process through our layer to create meaningful hierarchy
        processed = self.processor(sliced)
        
        return processed


def test_slice_tagging():
    """Test that slice operations can be properly tagged."""
    
    print("=== Testing Slice Operation Tagging ===")
    
    # Create model and inputs
    model = SimpleSliceModel()
    model.eval()
    inputs = torch.randn(5, 10)  # Input that will be sliced to (3, 10)
    
    print(f"Model: {model}")
    print(f"Input shape: {inputs.shape}")
    
    # Test with both strategies
    for strategy in ["usage_based", "htp"]:
        print(f"\n--- Testing {strategy} strategy ---")
        
        exporter = HierarchyExporter(strategy=strategy)
        
        with tempfile.NamedTemporaryFile(suffix='.onnx') as tmp:
            # Export with hierarchy preservation
            result = exporter.export(
                model=model,
                example_inputs=inputs,
                output_path=tmp.name
            )
            
            print(f"Export result: {result}")
            
            # Load and analyze ONNX model
            import onnx
            onnx_model = onnx.load(tmp.name)
            
            # Find slice operations in ONNX
            slice_nodes = []
            all_nodes = []
            
            for node in onnx_model.graph.node:
                all_nodes.append((node.name or f"unnamed_{node.op_type}", node.op_type))
                if node.op_type == 'Slice':
                    slice_nodes.append(node)
            
            print(f"Total ONNX nodes: {len(all_nodes)}")
            print(f"Slice nodes found: {len(slice_nodes)}")
            
            # Print all nodes for debugging
            print("All ONNX nodes:")
            for name, op_type in all_nodes:
                print(f"  {name}: {op_type}")
            
            # Get tag mapping
            tag_mapping = exporter.get_tag_mapping()
            print(f"Tag mapping size: {len(tag_mapping)}")
            
            # Check if slice nodes have tags
            slice_nodes_with_tags = 0
            slice_nodes_without_tags = 0
            
            for node_name, node_info in tag_mapping.items():
                if node_info.get('op_type') == 'Slice':
                    tags = node_info.get('tags', [])
                    if tags:
                        slice_nodes_with_tags += 1
                        print(f"  Slice node '{node_name}' HAS tags: {tags}")
                    else:
                        slice_nodes_without_tags += 1
                        print(f"  Slice node '{node_name}' has NO tags")
            
            print(f"Slice nodes with tags: {slice_nodes_with_tags}")
            print(f"Slice nodes without tags: {slice_nodes_without_tags}")
            
            # Load hierarchy JSON if it exists
            hierarchy_file = tmp.name.replace('.onnx', '_hierarchy.json')
            try:
                with open(hierarchy_file) as f:
                    hierarchy_data = json.load(f)
                    
                print(f"Hierarchy file loaded: {hierarchy_file}")
                print(f"Tagged operations in hierarchy: {hierarchy_data.get('summary', {}).get('tagged_operations', 0)}")
                
                # Look for slice operations in the hierarchy
                node_tags = hierarchy_data.get('node_tags', {})
                slice_in_hierarchy = [name for name, info in node_tags.items() 
                                    if info.get('op_type') == 'Slice']
                print(f"Slice operations in hierarchy: {len(slice_in_hierarchy)}")
                for slice_name in slice_in_hierarchy:
                    slice_info = node_tags[slice_name]
                    print(f"  {slice_name}: tags={slice_info.get('tags', [])}")
                    
            except FileNotFoundError:
                print("No hierarchy JSON file found")
            
            # Verify we can tag slice operations
            if slice_nodes:
                print(f"\n✅ ONNX contains {len(slice_nodes)} Slice node(s)")
                if slice_nodes_with_tags > 0:
                    print(f"✅ {slice_nodes_with_tags} Slice node(s) have tags - TAGGING WORKS!")
                elif slice_nodes_without_tags > 0:
                    print(f"❌ {slice_nodes_without_tags} Slice node(s) have no tags - TAGGING FAILED")
                    print("   This suggests we bypassed rather than fixed the issue")
                else:
                    print("❓ No slice nodes found in tag mapping - investigation needed")
            else:
                print("❌ No Slice nodes found in ONNX - model didn't generate expected operations")


class HuggingFaceStyleSliceModel(nn.Module):
    """Model that mimics HuggingFace patterns with slicing."""
    
    def __init__(self):
        super().__init__()
        # Create a more HF-like structure
        self.embeddings = nn.Embedding(100, 10)
        self.layer_norm = nn.LayerNorm(10)  # This should be tagged (whitelist)
        
    def forward(self, input_ids):
        # Get embeddings
        embeddings = self.embeddings(input_ids)
        
        # Slice embeddings (common in HF models for sequence processing)
        sliced_embeddings = embeddings[1:-1]  # Remove first and last tokens
        
        # Normalize
        normalized = self.layer_norm(sliced_embeddings)
        
        return normalized


def test_hf_style_slice_tagging():
    """Test slice tagging in a more HuggingFace-like model."""
    
    print("\n\n=== Testing HF-Style Slice Operation Tagging ===")
    
    model = HuggingFaceStyleSliceModel()
    model.eval()
    inputs = torch.randint(0, 100, (5,))  # 5 token sequence
    
    print(f"HF-style model: {model}")
    print(f"Input shape: {inputs.shape}")
    
    # Test with HTP strategy (should be better for this)
    exporter = HierarchyExporter(strategy="htp")
    
    with tempfile.NamedTemporaryFile(suffix='.onnx') as tmp:
        result = exporter.export(
            model=model,
            example_inputs=inputs,
            output_path=tmp.name
        )
        
        print(f"Export result: {result}")
        
        # Analyze results
        import onnx
        onnx_model = onnx.load(tmp.name)
        
        slice_nodes = [node for node in onnx_model.graph.node if node.op_type == 'Slice']
        print(f"Slice nodes in HF-style model: {len(slice_nodes)}")
        
        tag_mapping = exporter.get_tag_mapping()
        
        # Check for LayerNorm tagging (should work due to whitelist)
        layernorm_nodes = [name for name, info in tag_mapping.items() 
                          if 'LayerNorm' in info.get('op_type', '')]
        print(f"LayerNorm nodes found: {len(layernorm_nodes)}")
        
        # Check slice tagging
        tagged_slices = sum(1 for name, info in tag_mapping.items() 
                           if info.get('op_type') == 'Slice' and info.get('tags', []))
        print(f"Tagged slice operations: {tagged_slices}")
        
        if tagged_slices > 0:
            print("✅ HF-style slice tagging works!")
        else:
            print("❌ HF-style slice tagging failed")


if __name__ == "__main__":
    test_slice_tagging()
    test_hf_style_slice_tagging()