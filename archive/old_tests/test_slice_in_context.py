"""
Test slice operation tagging when slice happens within a module context.
"""

import tempfile

import torch
import torch.nn as nn

from modelexport.hierarchy_exporter import HierarchyExporter


class SlicingModule(nn.Module):
    """Custom module that performs slicing within its forward method."""
    
    def forward(self, x):
        # This slice should be captured with the module's context
        sliced = x[1:4]
        return sliced


class ContextualSliceModel(nn.Module):
    """Model where slice operation happens within a hooked module."""
    
    def __init__(self):
        super().__init__()
        self.slicer = SlicingModule()
        self.processor = nn.Linear(10, 10)
        
    def forward(self, x):
        # Slice within hooked module
        sliced = self.slicer(x)
        
        # Process result
        processed = self.processor(sliced)
        return processed


def test_contextual_slice_tagging():
    """Test slice tagging when slice happens within module context."""
    
    print("=== Testing Contextual Slice Operation Tagging ===")
    
    model = ContextualSliceModel()
    model.eval()
    inputs = torch.randn(5, 10)
    
    print(f"Model: {model}")
    print(f"Input shape: {inputs.shape}")
    
    # Test with HTP strategy
    exporter = HierarchyExporter(strategy="htp")
    
    with tempfile.NamedTemporaryFile(suffix='.onnx') as tmp:
        print("\n--- Exporting with HTP strategy ---")
        result = exporter.export(
            model=model,
            example_inputs=inputs,
            output_path=tmp.name
        )
        
        print(f"Export result: {result}")
        
        # Check slice operation tracking
        print(f"Slice operations tracked: {len(exporter._slice_operations)}")
        for i, slice_op in enumerate(exporter._slice_operations):
            print(f"  {i}: {slice_op}")
        
        # Load and analyze ONNX model
        import onnx
        onnx_model = onnx.load(tmp.name)
        
        slice_nodes = [node for node in onnx_model.graph.node if node.op_type == 'Slice']
        print(f"\nONNX Slice nodes found: {len(slice_nodes)}")
        
        # Check tag mapping
        tag_mapping = exporter.get_tag_mapping()
        slice_nodes_tagged = 0
        
        for node_name, node_info in tag_mapping.items():
            if node_info.get('op_type') == 'Slice':
                tags = node_info.get('tags', [])
                if tags:
                    slice_nodes_tagged += 1
                    print(f"✅ Slice node '{node_name}' tagged with: {tags}")
                else:
                    print(f"❌ Slice node '{node_name}' has no tags")
        
        print(f"\nSummary:")
        print(f"  Slice operations tracked during forward pass: {len(exporter._slice_operations)}")
        print(f"  Slice nodes in ONNX: {len(slice_nodes)}")
        print(f"  Slice nodes tagged: {slice_nodes_tagged}")
        
        if slice_nodes_tagged > 0:
            print("✅ SUCCESS: Slice operations are being tagged!")
        else:
            print("❌ FAILED: Slice operations are not being tagged")
        
        # Load hierarchy JSON for additional verification
        hierarchy_file = tmp.name.replace('.onnx', '_hierarchy.json')
        try:
            import json
            with open(hierarchy_file) as f:
                hierarchy_data = json.load(f)
            
            print(f"\nHierarchy metadata:")
            htp_metadata = hierarchy_data.get('htp_metadata', {})
            print(f"  Slice operations in metadata: {len(htp_metadata.get('slice_operations', []))}")
            for slice_op in htp_metadata.get('slice_operations', []):
                print(f"    {slice_op}")
        
        except FileNotFoundError:
            print("No hierarchy JSON file found")


def test_root_level_slice():
    """Test what happens with root-level slice operations."""
    
    print("\n\n=== Testing Root-Level Slice Operation ===")
    
    class RootSliceModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.processor = nn.Linear(10, 10)
            
        def forward(self, x):
            # This slice happens at root level - should we tag it differently?
            sliced = x[1:4]
            return self.processor(sliced)
    
    model = RootSliceModel()
    model.eval()
    inputs = torch.randn(5, 10)
    
    exporter = HierarchyExporter(strategy="htp")
    
    with tempfile.NamedTemporaryFile(suffix='.onnx') as tmp:
        result = exporter.export(
            model=model,
            example_inputs=inputs,
            output_path=tmp.name
        )
        
        print(f"Root-level slice operations tracked: {len(exporter._slice_operations)}")
        
        tag_mapping = exporter.get_tag_mapping()
        slice_nodes = {name: info for name, info in tag_mapping.items() 
                      if info.get('op_type') == 'Slice'}
        
        for name, info in slice_nodes.items():
            tags = info.get('tags', [])
            print(f"Root-level slice node '{name}': tags = {tags}")


if __name__ == "__main__":
    test_contextual_slice_tagging()
    test_root_level_slice()