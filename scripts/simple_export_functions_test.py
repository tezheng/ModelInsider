#!/usr/bin/env python3
"""
Simple test script to analyze export_modules_as_functions behavior.
This version avoids the selective export that caused the PyTorch internal error.
"""

from pathlib import Path

import onnx
import torch
import torch.nn as nn


class SimpleModel(nn.Module):
    """Simple model for testing export_modules_as_functions."""
    
    def __init__(self):
        super().__init__()
        self.layer1 = nn.Linear(10, 20)
        self.activation = nn.ReLU()
        self.layer2 = nn.Linear(20, 5)
        
    def forward(self, x):
        x = self.layer1(x)
        x = self.activation(x)
        x = self.layer2(x)
        return x


def export_and_compare():
    """Export with both modes and compare."""
    
    output_dir = Path("temp/export_functions_simple")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create model and input
    model = SimpleModel()
    model.eval()
    sample_input = torch.randn(1, 10)
    
    print("=== EXPORT_MODULES_AS_FUNCTIONS EXPERIMENT ===")
    print(f"PyTorch: {torch.__version__}, ONNX: {onnx.__version__}")
    
    # Export with functions=False
    print("\n1. Export with functions=False (default):")
    torch.onnx.export(
        model, sample_input,
        output_dir / "model_functions_false.onnx",
        opset_version=17,
        export_modules_as_functions=False,
        verbose=False
    )
    
    model_false = onnx.load(str(output_dir / "model_functions_false.onnx"))
    print(f"   Nodes: {len(model_false.graph.node)}")
    print(f"   Functions: {len(model_false.functions) if hasattr(model_false, 'functions') else 0}")
    
    # Export with functions=True
    print("\n2. Export with functions=True:")
    torch.onnx.export(
        model, sample_input,
        output_dir / "model_functions_true.onnx",
        opset_version=17,
        export_modules_as_functions=True,
        verbose=False
    )
    
    model_true = onnx.load(str(output_dir / "model_functions_true.onnx"))
    print(f"   Nodes: {len(model_true.graph.node)}")
    print(f"   Functions: {len(model_true.functions) if hasattr(model_true, 'functions') else 0}")
    
    if hasattr(model_true, 'functions') and len(model_true.functions) > 0:
        print("\n   Local functions created:")
        for func in model_true.functions:
            print(f"   - {func.name} ({len(func.node)} internal nodes)")
    
    print("\n=== ANALYSIS ===")
    print(f"\n🔍 KEY FINDINGS:")
    print(f"   • Functions=False: {len(model_false.graph.node)} nodes, {len(model_false.functions) if hasattr(model_false, 'functions') else 0} functions")
    print(f"   • Functions=True: {len(model_true.graph.node)} nodes, {len(model_true.functions) if hasattr(model_true, 'functions') else 0} functions")
    
    if hasattr(model_true, 'functions') and len(model_true.functions) > 0:
        print(f"   • Functions=True created {len(model_true.functions)} local functions")
        print(f"   • Main graph nodes reduced by {len(model_false.graph.node) - len(model_true.graph.node)}")
        print(f"   • This demonstrates module-level hierarchy preservation")
    else:
        print(f"   • Functions=True did not create local functions (may not be supported)")
    
    print(f"\n⚖️  RELEVANCE TO MODELEXPORT:")
    print(f"   • export_modules_as_functions: MODULE-level hierarchy (entire modules → functions)")
    print(f"   • ModelExport HTP: OPERATION-level hierarchy (individual ops → module tags)")
    print(f"   • Different granularity: complementary but not equivalent")
    print(f"   • ModelExport provides finer-grained traceability")
    
    print(f"\n🎯 CONCLUSION:")
    print(f"   • export_modules_as_functions does NOT replace ModelExport requirements")
    print(f"   • It works at module level, not operation level")
    print(f"   • Could be complementary for dual-level hierarchy")
    print(f"   • Continue with HTP strategy as primary approach")


if __name__ == "__main__":
    export_and_compare()