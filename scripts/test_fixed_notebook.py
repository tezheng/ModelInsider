#!/usr/bin/env python3
"""Test the fixed notebook cells to ensure they work."""

import warnings
from pathlib import Path

import onnx
import onnxruntime as ort
import torch
import torch.nn as nn
import torch.onnx

warnings.filterwarnings("ignore", category=UserWarning)

class HierarchicalModel(nn.Module):
    """A hierarchical model that demonstrates clear module structure for ONNX export."""
    
    def __init__(self, input_dim=10, hidden_dim=20, num_classes=5):
        super().__init__()
        
        # Feature extraction layers
        self.feature_extractor = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU()
        )
        
        # Processing layers with residual connection
        self.processor = nn.ModuleDict({
            'transform': nn.Linear(hidden_dim, hidden_dim),
            'activation': nn.Tanh(),
            'norm': nn.LayerNorm(hidden_dim)
        })
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, num_classes)
        )
        
    def forward(self, x):
        # Feature extraction
        features = self.feature_extractor(x)
        
        # Processing with residual connection
        processed = self.processor['transform'](features)
        processed = self.processor['activation'](processed)
        processed = self.processor['norm'](processed)
        
        # Add residual connection
        combined = features + processed
        
        # Classification
        output = self.classifier(combined)
        
        return output

def test_model_and_export():
    """Test model creation and export."""
    
    # Create output directory
    output_dir = Path("temp/onnx_structure_analysis")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"PyTorch version: {torch.__version__}")
    print(f"ONNX version: {onnx.__version__}")
    
    # Create model
    model = HierarchicalModel()
    model.eval()
    
    # Create sample input
    sample_input = torch.randn(2, 10)
    
    print("Model architecture:")
    print(model)
    print(f"\nTotal parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Test forward pass
    with torch.no_grad():
        output = model(sample_input)
        print(f"\nOutput shape: {output.shape}")
        print("✓ Model forward pass successful")
    
    # Test exports
    paths = {}
    
    # Standard export
    print(f"\nExporting with export_modules_as_functions=False...")
    output_path = output_dir / "model_standard.onnx"
    
    with torch.no_grad():
        torch.onnx.export(
            model,
            sample_input,
            output_path,
            export_params=True,
            opset_version=17,
            do_constant_folding=True,
            input_names=['input'],
            output_names=['output'],
            dynamic_axes={
                'input': {0: 'batch_size'},
                'output': {0: 'batch_size'}
            },
            export_modules_as_functions=False,
            verbose=False
        )
    
    print(f"✓ Exported to: {output_path.name}")
    paths['standard'] = output_path
    
    # Functions export
    print(f"\nExporting with export_modules_as_functions=True...")
    output_path = output_dir / "model_all_functions.onnx"
    
    with torch.no_grad():
        torch.onnx.export(
            model,
            sample_input,
            output_path,
            export_params=True,
            opset_version=17,
            do_constant_folding=True,
            input_names=['input'],
            output_names=['output'],
            dynamic_axes={
                'input': {0: 'batch_size'},
                'output': {0: 'batch_size'}
            },
            export_modules_as_functions=True,
            verbose=False
        )
    
    print(f"✓ Exported to: {output_path.name}")
    paths['all_functions'] = output_path
    
    # Test both models
    for name, path in paths.items():
        print(f"\nTesting {name} model...")
        try:
            # Load and validate
            onnx_model = onnx.load(str(path))
            onnx.checker.check_model(onnx_model)
            print("✓ Model validation passed")
            
            # Test inference
            session = ort.InferenceSession(str(path))
            input_name = session.get_inputs()[0].name
            result = session.run(None, {input_name: sample_input.numpy()})
            print(f"✓ Inference test passed - output shape: {result[0].shape}")
            
            # Basic analysis
            graph = onnx_model.graph
            num_functions = len(onnx_model.functions) if hasattr(onnx_model, 'functions') else 0
            print(f"  Nodes: {len(graph.node)}, Functions: {num_functions}")
            
        except Exception as e:
            print(f"✗ Test failed: {e}")
            return False
    
    print(f"\n✅ All tests passed! Notebook should work correctly.")
    return True

if __name__ == "__main__":
    success = test_model_and_export()
    exit(0 if success else 1)