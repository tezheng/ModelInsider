#!/usr/bin/env python3
"""Debug HTP implementation to understand what's happening."""

import tempfile

import torch

from modelexport.hierarchy_exporter import HierarchyExporter


class SimpleModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = torch.nn.Linear(10, 5)
        self.layer2 = torch.nn.Linear(5, 2)
        
    def forward(self, x):
        x = self.layer1(x)      # Linear -> Gemm/MatMul
        x = torch.tanh(x)       # Tanh
        x = self.layer2(x)      # Linear -> Gemm/MatMul  
        x = torch.relu(x)       # Relu
        return x

def debug_htp():
    print("=== Debugging HTP Implementation ===")
    
    model = SimpleModel()
    model.eval()
    example_input = torch.randn(1, 10)
    
    print(f"Model: {model}")
    print(f"Input shape: {example_input.shape}")
    
    # Test HTP export
    exporter = HierarchyExporter(strategy="htp")
    
    with tempfile.NamedTemporaryFile(suffix='.onnx', delete=False) as tmp:
        print(f"\nExporting with HTP strategy to: {tmp.name}")
        
        result = exporter.export(
            model=model,
            example_inputs=example_input,
            output_path=tmp.name
        )
        
        print(f"\nExport result: {result}")
        
        # Check operation trace
        print(f"\nOperation trace length: {len(exporter._operation_trace)}")
        for i, trace in enumerate(exporter._operation_trace[:5]):
            print(f"  Trace {i}: {trace}")
        
        # Check native regions
        print(f"\nNative operation regions: {len(exporter._native_op_regions)}")
        for i, region in enumerate(exporter._native_op_regions):
            print(f"  Region {i}: {region}")
        
        # Check tag mapping
        print(f"\nTag mapping entries: {len(exporter._tag_mapping)}")
        tagged_ops = 0
        for name, info in list(exporter._tag_mapping.items())[:5]:
            tags = info.get('tags', [])
            if tags:
                tagged_ops += 1
            print(f"  {name}: {info['op_type']} -> {tags}")
        
        print(f"\nTagged operations: {tagged_ops}/{len(exporter._tag_mapping)}")
        
        # Check if hooks were registered
        print(f"\nHook registration info:")
        print(f"  Pre-hooks: {len(exporter._pre_hooks)}")
        print(f"  Post-hooks: {len(exporter._post_hooks)}")
        
        # Check patched operations
        print(f"\nPatched operations: {len(exporter._patched_operations)}")
        for (module, op_name), original in list(exporter._patched_operations.items())[:5]:
            print(f"  {module.__name__}.{op_name}: {original}")

if __name__ == "__main__":
    debug_htp()