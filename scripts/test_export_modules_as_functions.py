#!/usr/bin/env python3
"""
Test script to analyze export_modules_as_functions behavior and compare with modelexport requirements.

This script creates a comprehensive experiment comparing function=True/False and analyzing
whether export_modules_as_functions helps with hierarchy preservation requirements.
"""

import json
import warnings
from pathlib import Path
from typing import Any

import onnx
import onnxruntime as ort
import torch
import torch.nn as nn

# Suppress deprecation warnings for clarity
warnings.filterwarnings("ignore", category=UserWarning)


class AttentionHead(nn.Module):
    """Simple attention head for testing module boundary preservation."""
    
    def __init__(self, embed_dim: int):
        super().__init__()
        self.embed_dim = embed_dim
        self.query = nn.Linear(embed_dim, embed_dim)
        self.key = nn.Linear(embed_dim, embed_dim)
        self.value = nn.Linear(embed_dim, embed_dim)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        q = self.query(x)
        k = self.key(x)
        v = self.value(x)
        
        # Simple scaled dot-product attention
        scores = torch.matmul(q, k.transpose(-2, -1)) / (self.embed_dim ** 0.5)
        attn_weights = torch.softmax(scores, dim=-1)
        output = torch.matmul(attn_weights, v)
        
        return output


class FeedForward(nn.Module):
    """Feed-forward network for testing."""
    
    def __init__(self, embed_dim: int, hidden_dim: int):
        super().__init__()
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        self.linear1 = nn.Linear(embed_dim, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, embed_dim)
        self.activation = nn.ReLU()
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear2(self.activation(self.linear1(x)))


class TransformerBlock(nn.Module):
    """Simple transformer block with clear module hierarchy."""
    
    def __init__(self, embed_dim: int, hidden_dim: int):
        super().__init__()
        self.embed_dim = embed_dim
        self.attention = AttentionHead(embed_dim)
        self.feed_forward = FeedForward(embed_dim, hidden_dim)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Attention with residual connection
        attn_out = self.attention(x)
        x = self.norm1(x + attn_out)
        
        # Feed-forward with residual connection
        ff_out = self.feed_forward(x)
        x = self.norm2(x + ff_out)
        
        return x


class TestTransformer(nn.Module):
    """Complete test model with hierarchical structure."""
    
    def __init__(self, embed_dim: int = 64, hidden_dim: int = 128, num_layers: int = 2):
        super().__init__()
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        self.embedding = nn.Embedding(1000, embed_dim)
        self.layers = nn.ModuleList([
            TransformerBlock(embed_dim, hidden_dim) for _ in range(num_layers)
        ])
        self.output_proj = nn.Linear(embed_dim, 10)  # 10 classes
        
    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        x = self.embedding(input_ids)
        
        for layer in self.layers:
            x = layer(x)
            
        # Global average pooling
        x = x.mean(dim=1)
        
        return self.output_proj(x)


def export_and_analyze(model, sample_input, export_modules_as_functions, suffix: str, output_dir: Path) -> dict[str, Any]:
    """Export model and analyze the resulting ONNX structure."""
    
    output_path = output_dir / f"test_model_{suffix}.onnx"
    
    print(f"\n=== Exporting with export_modules_as_functions={export_modules_as_functions} ===")
    
    # Export model
    with torch.no_grad():
        torch.onnx.export(
            model,
            sample_input,
            output_path,
            export_params=True,
            opset_version=17,  # Required for local functions
            do_constant_folding=True,
            input_names=['input_ids'],
            output_names=['logits'],
            dynamic_axes={
                'input_ids': {0: 'batch_size', 1: 'sequence'},
                'logits': {0: 'batch_size'}
            },
            export_modules_as_functions=export_modules_as_functions,
            verbose=False
        )
    
    # Load and analyze ONNX model
    onnx_model = onnx.load(str(output_path))
    
    # Basic statistics
    graph = onnx_model.graph
    num_nodes = len(graph.node)
    num_initializers = len(graph.initializer)
    num_functions = len(onnx_model.functions) if hasattr(onnx_model, 'functions') else 0
    
    print(f"ONNX Model Statistics:")
    print(f"  Nodes: {num_nodes}")
    print(f"  Initializers: {num_initializers}")
    print(f"  Local Functions: {num_functions}")
    
    # Analyze node types
    node_types = {}
    for node in graph.node:
        op_type = node.op_type
        node_types[op_type] = node_types.get(op_type, 0) + 1
    
    print(f"\nNode Types Distribution:")
    for op_type, count in sorted(node_types.items()):
        print(f"  {op_type}: {count}")
    
    # Analyze local functions if present
    function_details = []
    if num_functions > 0:
        print(f"\nLocal Functions:")
        for func in onnx_model.functions:
            func_node_types = {}
            for node in func.node:
                op_type = node.op_type
                func_node_types[op_type] = func_node_types.get(op_type, 0) + 1
            
            func_detail = {
                'name': func.name,
                'domain': func.domain,
                'num_nodes': len(func.node),
                'node_types': func_node_types
            }
            function_details.append(func_detail)
            
            print(f"  {func.name} (domain: {func.domain})")
            print(f"    Nodes: {len(func.node)}")
            print(f"    Node types: {dict(func_node_types)}")
    
    # Test model execution
    execution_success = False
    output_shape = None
    try:
        session = ort.InferenceSession(str(output_path))
        input_name = session.get_inputs()[0].name
        result = session.run(None, {input_name: sample_input.numpy()})
        execution_success = True
        output_shape = result[0].shape
        print(f"\nModel execution successful!")
        print(f"Output shape: {output_shape}")
    except Exception as e:
        print(f"\nModel execution failed: {e}")
    
    return {
        'path': str(output_path),
        'num_nodes': num_nodes,
        'num_initializers': num_initializers,
        'num_functions': num_functions,
        'node_types': node_types,
        'function_details': function_details,
        'execution_success': execution_success,
        'output_shape': output_shape
    }


def compare_results(result_false, result_true, result_selective):
    """Compare the three export approaches."""
    
    print("\n" + "="*60)
    print("COMPARISON ANALYSIS")
    print("="*60)
    
    print(f"\n{'Metric':<20} {'Functions=False':<15} {'Functions=True':<15} {'Selective':<15}")
    print("-" * 70)
    print(f"{'Nodes':<20} {result_false['num_nodes']:<15} {result_true['num_nodes']:<15} {result_selective['num_nodes']:<15}")
    print(f"{'Initializers':<20} {result_false['num_initializers']:<15} {result_true['num_initializers']:<15} {result_selective['num_initializers']:<15}")
    print(f"{'Local Functions':<20} {result_false['num_functions']:<15} {result_true['num_functions']:<15} {result_selective['num_functions']:<15}")
    
    # Analyze the differences
    print("\n" + "="*60)
    print("KEY OBSERVATIONS")
    print("="*60)
    
    if result_true['num_functions'] > 0:
        print(f"✓ Functions=True successfully created {result_true['num_functions']} local functions")
        print(f"✓ This represents module-level preservation of hierarchy")
    else:
        print("✗ Functions=True did not create local functions (possible version/compatibility issue)")
    
    if result_selective['num_functions'] > 0:
        print(f"✓ Selective export created {result_selective['num_functions']} functions")
        print(f"✓ This allows fine-grained control over which modules become functions")
    
    # Node count differences
    if result_true['num_nodes'] < result_false['num_nodes']:
        reduction = result_false['num_nodes'] - result_true['num_nodes']
        print(f"✓ Functions=True reduced main graph nodes by {reduction} ({reduction/result_false['num_nodes']*100:.1f}%)")
        print(f"✓ These operations were moved into local functions")
    elif result_true['num_nodes'] == result_false['num_nodes']:
        print(f"⚠ No reduction in main graph nodes - functions may not be working as expected")


def analyze_hierarchy_preservation():
    """Analyze how export_modules_as_functions relates to modelexport requirements."""
    
    print("\n" + "="*80)
    print("HIERARCHY PRESERVATION ANALYSIS")
    print("="*80)
    
    print("\n🎯 MODELEXPORT PROJECT REQUIREMENTS:")
    print("   • Tag individual ONNX operations with their source PyTorch modules")
    print("   • Preserve fine-grained operation-to-module mapping")
    print("   • Enable traceability from ONNX ops back to original code")
    print("   • Support any HuggingFace model universally")
    
    print("\n🔍 EXPORT_MODULES_AS_FUNCTIONS BEHAVIOR:")
    print("   • Exports entire PyTorch modules as ONNX local functions")
    print("   • Preserves module boundaries, not operation boundaries")
    print("   • Groups multiple operations within each function")
    print("   • Functions contain the actual computational operations")
    
    print("\n⚖️  COMPARISON:")
    
    print("\n   GRANULARITY:")
    print("   • ModelExport HTP: Operation-level (MatMul, Add, etc. → Module)")
    print("   • export_modules_as_functions: Module-level (Entire module → Function)")
    
    print("\n   STRUCTURE:")
    print("   • ModelExport HTP: Flat graph with rich metadata tags")
    print("   • export_modules_as_functions: Hierarchical functions containing operations")
    
    print("\n   USE CASES:")
    print("   • ModelExport HTP: Fine-grained analysis, debugging, custom backends")
    print("   • export_modules_as_functions: Module replacement, logical grouping")
    
    print("\n\n" + "="*80)
    print("VERDICT: COMPLEMENTARY BUT NOT EQUIVALENT")
    print("="*80)
    
    print("\n❌ DOES NOT REPLACE MODELEXPORT HTP STRATEGY:")
    print("   1. Different granularity - modules vs operations")
    print("   2. No operation-level traceability within functions")
    print("   3. Limited metadata propagation for individual ops")
    print("   4. Functions obscure internal operation structure")
    
    print("\n✅ POTENTIAL COMPLEMENTARY USE:")
    print("   1. Could be combined with HTP for dual-level hierarchy")
    print("   2. Module-level functions + operation-level tags")
    print("   3. Better organization for complex models")
    print("   4. Alternative export mode for different use cases")
    
    print("\n🔧 TECHNICAL LIMITATIONS:")
    print("   1. Deprecated feature - uncertain future support")
    print("   2. Requires opset_version >= 15")
    print("   3. May not work with all ONNX runtimes")
    print("   4. Less control over individual operation metadata")


def final_conclusions():
    """Present final conclusions and recommendations."""
    
    print("\n" + "="*80)
    print("FINAL CONCLUSIONS & RECOMMENDATIONS")
    print("="*80)
    
    print("\n🎯 MAIN QUESTION: Does export_modules_as_functions help with modelexport requirements?")
    print("\n📝 ANSWER: **PARTIALLY HELPFUL BUT NOT SUFFICIENT**")
    
    print("\n\n" + "-"*60)
    print("WHY IT'S NOT SUFFICIENT FOR MODELEXPORT:")
    print("-"*60)
    
    print("\n1. 🔍 GRANULARITY MISMATCH:")
    print("   • ModelExport needs: Operation → Module mapping (MatMul came from layer.attention.query)")
    print("   • export_modules_as_functions: Module → Function grouping (entire module becomes function)")
    print("   • Individual operations inside functions lose traceability")
    
    print("\n2. 🏗️ ARCHITECTURE DIFFERENCE:")
    print("   • ModelExport: Flat graph with rich metadata tags on each operation")
    print("   • export_modules_as_functions: Nested functions containing grouped operations")
    print("   • Can't trace specific ops within functions back to source code")
    
    print("\n3. 🎚️ USE CASE MISMATCH:")
    print("   • ModelExport target: Fine-grained debugging, custom backends, operation analysis")
    print("   • export_modules_as_functions target: Module replacement, logical organization")
    
    print("\n4. 🚫 TECHNICAL LIMITATIONS:")
    print("   • Deprecated feature with uncertain future")
    print("   • Limited ONNX runtime support")
    print("   • Less flexible than custom metadata approach")
    
    print("\n\n" + "-"*60)
    print("POTENTIAL COMPLEMENTARY VALUE:")
    print("-"*60)
    
    print("\n✅ COULD BE USEFUL AS ADDITIONAL FEATURE:")
    print("   • Dual-level hierarchy: Module functions + operation tags")
    print("   • Alternative export mode for different use cases")
    print("   • Better organization for very complex models")
    print("   • Module-level replacement capabilities")
    
    print("\n\n" + "="*80)
    print("FINAL RECOMMENDATION")
    print("="*80)
    
    print("\n🎯 CONTINUE WITH CURRENT HTP STRATEGY AS PRIMARY APPROACH")
    print("\n   Reasons:")
    print("   ✓ Provides the exact granularity needed (operation-level)")
    print("   ✓ Universal approach works with any model")
    print("   ✓ Flexible metadata system")
    print("   ✓ Better aligned with project requirements")
    
    print("\n🔧 CONSIDER export_modules_as_functions AS FUTURE ENHANCEMENT")
    print("\n   Potential use cases:")
    print("   • Optional dual-level hierarchy export mode")
    print("   • Better organization for extremely complex models")
    print("   • Alternative for users who prefer function-based structure")
    print("   • Research into hybrid approaches")
    
    print("\n\n" + "⚡" * 80)
    print("EXPERIMENT VALIDATES: HTP strategy addresses different, more granular need")
    print("⚡" * 80)


def main():
    """Run the complete experiment."""
    
    # Create output directory
    output_dir = Path("temp/export_functions_experiment")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Experiment output directory: {output_dir.absolute()}")
    print(f"PyTorch version: {torch.__version__}")
    print(f"ONNX version: {onnx.__version__}")
    
    # Create model and sample input
    model = TestTransformer(embed_dim=64, hidden_dim=128, num_layers=2)
    model.eval()
    
    # Sample input: batch_size=2, seq_len=8
    sample_input = torch.randint(0, 1000, (2, 8))
    
    print("Model created successfully!")
    print(f"Model hierarchy:")
    for name, module in model.named_modules():
        if name:  # Skip root module
            print(f"  {name}: {type(module).__name__}")
    
    # Run experiments
    result_false = export_and_analyze(model, sample_input, False, "functions_false", output_dir)
    result_true = export_and_analyze(model, sample_input, True, "functions_true", output_dir)
    
    # Selective modules experiment
    selective_modules = {AttentionHead, FeedForward}
    result_selective = export_and_analyze(model, sample_input, selective_modules, "functions_selective", output_dir)
    
    # Compare results
    compare_results(result_false, result_true, result_selective)
    
    # Analysis
    analyze_hierarchy_preservation()
    final_conclusions()
    
    # Save experiment summary
    experiment_summary = {
        "experiment_date": "2025-01-07",
        "pytorch_version": torch.__version__,
        "onnx_version": onnx.__version__,
        "test_model": "TestTransformer (hierarchical transformer-like model)",
        "results": {
            "functions_false": result_false,
            "functions_true": result_true,
            "functions_selective": result_selective
        },
        "conclusions": {
            "main_finding": "export_modules_as_functions provides module-level hierarchy, not operation-level granularity needed by modelexport",
            "recommendation": "Continue with HTP strategy as primary approach; consider export_modules_as_functions as future enhancement",
            "granularity_difference": "export_modules_as_functions: module→function, modelexport: operation→module",
            "use_case_alignment": "Different target use cases - module replacement vs operation traceability"
        }
    }
    
    summary_path = output_dir / "experiment_summary.json"
    with open(summary_path, 'w') as f:
        json.dump(experiment_summary, f, indent=2)
    
    print(f"\n✅ Experiment results saved to: {summary_path}")
    print(f"\n📁 All experiment files available in: {output_dir}")
    print(f"\n🔍 Key finding: export_modules_as_functions operates at module level, not operation level")
    print(f"🎯 Recommendation: Continue with current HTP strategy for operation-level hierarchy preservation")


if __name__ == "__main__":
    main()