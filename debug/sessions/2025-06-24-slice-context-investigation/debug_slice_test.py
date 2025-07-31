"""
Debug test to understand why slice operations aren't being tracked.
"""

import tempfile

import torch
import torch.nn as nn

from modelexport.hierarchy_exporter import HierarchyExporter


class DebugSliceModel(nn.Module):
    """Model for debugging slice operation tracking."""
    
    def __init__(self):
        super().__init__()
        self.processor = nn.Linear(10, 10)
        
    def forward(self, x):
        print(f"In forward: input shape = {x.shape}")
        # Add explicit slice operation
        sliced = x[1:4]
        print(f"After slice: shape = {sliced.shape}")
        
        # Process through layer
        processed = self.processor(sliced)
        print(f"After processing: shape = {processed.shape}")
        
        return processed


def test_slice_debugging():
    """Debug slice operation tracking step by step."""
    
    print("=== Debugging Slice Operation Tracking ===")
    
    model = DebugSliceModel()
    model.eval()
    inputs = torch.randn(5, 10)
    
    print(f"Model: {model}")
    print(f"Input shape: {inputs.shape}")
    
    # Test with HTP strategy (our implementation)
    exporter = HierarchyExporter(strategy="htp")
    
    print("\n--- Step 1: Check if hooks are registered ---")
    exporter._reset_state()
    exporter._model = model
    exporter._register_hooks(model)
    
    print(f"Pre-hooks registered: {len(exporter._pre_hooks)}")
    print(f"Post-hooks registered: {len(exporter._post_hooks)}")
    
    # Test forward pass with hooks
    print("\n--- Step 2: Test forward pass with hooks ---")
    print("Current tag stack:", exporter._tag_stack)
    print("Current context:", exporter._operation_context)
    
    with torch.no_grad():
        output = model(inputs)
    
    print(f"After forward pass:")
    print(f"Tag stack: {exporter._tag_stack}")
    print(f"Operation context: {dict(exporter._operation_context)}")
    
    print("\n--- Step 3: Test slice operation patching ---")
    # Manually test the getitem patch
    exporter._patch_tensor_getitem()
    
    print("Testing manual slice operation...")
    test_tensor = torch.randn(5, 10)
    
    # Set a fake current tag for testing
    exporter._tag_stack.append("/TestModel/TestLayer")
    
    sliced_result = test_tensor[1:4]
    print(f"Slice operations captured: {len(exporter._slice_operations)}")
    print(f"Slice operations: {exporter._slice_operations}")
    
    exporter._unpatch_tensor_getitem()
    
    print("\n--- Step 4: Full export test ---")
    
    with tempfile.NamedTemporaryFile(suffix='.onnx') as tmp:
        result = exporter.export(
            model=model,
            example_inputs=inputs,
            output_path=tmp.name
        )
        
        print(f"Export result: {result}")
        print(f"Final slice operations tracked: {len(exporter._slice_operations)}")
        print(f"Slice operations: {exporter._slice_operations}")
        
        # Check tag mapping for slice nodes
        tag_mapping = exporter.get_tag_mapping()
        slice_nodes = {name: info for name, info in tag_mapping.items() 
                      if info.get('op_type') == 'Slice'}
        
        print(f"Slice nodes in tag mapping: {slice_nodes}")


if __name__ == "__main__":
    test_slice_debugging()