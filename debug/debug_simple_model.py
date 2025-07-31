#!/usr/bin/env python3
"""
Debug script to see what tags are generated for the simple model.
"""

import tempfile

import torch
import torch.nn as nn

from modelexport.hierarchy_exporter import HierarchyExporter


class SimpleModel(nn.Module):
    """Simple test model for parameter mapping tests."""
    
    def __init__(self):
        super().__init__()
        self.embedding = nn.Embedding(100, 64)
        self.linear1 = nn.Linear(64, 32)
        self.linear2 = nn.Linear(32, 10)
        
    def forward(self, x):
        x = self.embedding(x)
        x = torch.relu(self.linear1(x))
        x = self.linear2(x)
        return x

def debug_simple_model():
    print("🔍 Debug: Simple Model Tags")
    print("=" * 50)
    
    model = SimpleModel()
    inputs = torch.tensor([[1, 2, 3]])
    
    exporter = HierarchyExporter()
    
    with tempfile.NamedTemporaryFile(suffix='.onnx', delete=False) as f:
        temp_path = f.name
    
    try:
        # Export model
        result = exporter.export(model, inputs, temp_path)
        print(f"✅ Export completed: {result}")
        
        # Check parameter mapping
        param_mappings = exporter._param_to_module
        print(f"✅ Parameter mappings: {len(param_mappings)}")
        
        for param_name, module_context in param_mappings.items():
            print(f"  {param_name}: {module_context['tag']}")
        
        # Check tag mapping
        tag_stats = exporter._compute_tag_statistics()
        print(f"\nTag distribution:")
        for tag, count in sorted(tag_stats.items(), key=lambda x: x[1], reverse=True):
            print(f"  {tag}: {count} operations")
            
    finally:
        import os
        os.unlink(temp_path)
        if os.path.exists(temp_path.replace('.onnx', '_hierarchy.json')):
            os.unlink(temp_path.replace('.onnx', '_hierarchy.json'))

if __name__ == "__main__":
    debug_simple_model()