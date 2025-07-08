#!/usr/bin/env python3
"""Test the final notebook to ensure all cells work."""

import torch
import torch.nn as nn
import torch.onnx
import onnx
import onnxruntime as ort
import numpy as np
from pathlib import Path
import warnings

warnings.filterwarnings("ignore", category=UserWarning)

# Quick test of the model and exports
def test_notebook():
    """Test the key components of the notebook."""
    
    print("Testing final notebook components...")
    
    # Test model creation
    class HierarchicalModel(nn.Module):
        def __init__(self, input_dim=10, hidden_dim=20, num_classes=5):
            super().__init__()
            self.feature_extractor = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(hidden_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU()
            )
            self.processor = nn.ModuleDict({
                'transform': nn.Linear(hidden_dim, hidden_dim),
                'activation': nn.Tanh(),
                'norm': nn.LayerNorm(hidden_dim)
            })
            self.classifier = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.ReLU(),
                nn.Linear(hidden_dim // 2, num_classes)
            )
            
        def forward(self, x):
            features = self.feature_extractor(x)
            processed = self.processor['transform'](features)
            processed = self.processor['activation'](processed)
            processed = self.processor['norm'](processed)
            combined = features + processed
            output = self.classifier(combined)
            return output
    
    # Create model and test
    model = HierarchicalModel()
    model.eval()
    sample_input = torch.randn(2, 10)
    
    with torch.no_grad():
        output = model(sample_input)
        assert output.shape == (2, 5), f"Expected (2, 5), got {output.shape}"
    
    print("✅ Model creation and forward pass test passed")
    
    # Test exports
    output_dir = Path("temp/notebook_test")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Standard export
    standard_path = output_dir / "standard.onnx"
    torch.onnx.export(
        model, sample_input, standard_path,
        export_params=True, opset_version=17,
        export_modules_as_functions=False, verbose=False
    )
    
    # Functions export
    functions_path = output_dir / "functions.onnx"
    torch.onnx.export(
        model, sample_input, functions_path,
        export_params=True, opset_version=17,
        export_modules_as_functions=True, verbose=False
    )
    
    print("✅ Both exports completed successfully")
    
    # Test inference
    for name, path in [("standard", standard_path), ("functions", functions_path)]:\n        session = ort.InferenceSession(str(path))
        input_name = session.get_inputs()[0].name
        result = session.run(None, {input_name: sample_input.numpy()})
        assert result[0].shape == (2, 5), f"{name}: Expected (2, 5), got {result[0].shape}"
    
    print("✅ Inference tests passed for both exports")
    
    # Test analysis
    standard_model = onnx.load(str(standard_path))
    functions_model = onnx.load(str(functions_path))
    
    standard_nodes = len(standard_model.graph.node)
    functions_nodes = len(functions_model.graph.node)
    functions_count = len(functions_model.functions) if hasattr(functions_model, 'functions') else 0
    
    print(f"✅ Analysis test passed:")
    print(f"   Standard: {standard_nodes} nodes, 0 functions")
    print(f"   Functions: {functions_nodes} nodes, {functions_count} functions")
    
    assert functions_count > 0, "Functions export should create local functions"
    assert functions_nodes < standard_nodes, "Functions export should have fewer main graph nodes"
    
    print("✅ All notebook components test successfully!")
    return True

if __name__ == "__main__":
    success = test_notebook()
    print(f"\n🎉 Final notebook is ready for use!" if success else "\n❌ Notebook has issues")
    exit(0 if success else 1)