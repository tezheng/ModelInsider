#!/usr/bin/env python3
"""
Examine the detailed ONNX structure differences between function=True/False exports.
"""

import onnx
from pathlib import Path


def examine_onnx_model(path: str, title: str):
    """Examine detailed ONNX model structure."""
    
    print(f"\n{'='*60}")
    print(f"{title}")
    print('='*60)
    
    model = onnx.load(path)
    graph = model.graph
    
    print(f"\nMain Graph Structure:")
    print(f"  Inputs: {len(graph.input)}")
    print(f"  Outputs: {len(graph.output)}")
    print(f"  Nodes: {len(graph.node)}")
    print(f"  Initializers: {len(graph.initializer)}")
    
    print(f"\nMain Graph Nodes:")
    for i, node in enumerate(graph.node):
        inputs_str = ", ".join(node.input[:3])
        if len(node.input) > 3:
            inputs_str += f", ... (+{len(node.input)-3})"
        print(f"  {i+1}. {node.op_type} ({inputs_str})")
    
    # Examine functions if present
    if hasattr(model, 'functions') and len(model.functions) > 0:
        print(f"\nLocal Functions ({len(model.functions)} total):")
        for i, func in enumerate(model.functions):
            print(f"\n  Function {i+1}: {func.name}")
            print(f"    Domain: {func.domain}")
            print(f"    Inputs: {len(func.input)}")  
            print(f"    Outputs: {len(func.output)}")
            print(f"    Internal Nodes: {len(func.node)}")
            
            print(f"    Internal Operations:")
            for j, node in enumerate(func.node):
                print(f"      {j+1}. {node.op_type}")
                
            # Show attributes if any
            if len(func.attribute) > 0:
                print(f"    Attributes: {len(func.attribute)}")
                for attr in func.attribute[:3]:  # Show first 3
                    print(f"      - {attr.name}: {attr.type}")
    else:
        print(f"\nNo local functions found.")


def main():
    """Examine both exported models."""
    
    base_dir = Path("temp/export_functions_simple")
    
    print("DETAILED ONNX STRUCTURE EXAMINATION")
    print("Comparing export_modules_as_functions=True vs False")
    
    examine_onnx_model(
        str(base_dir / "model_functions_false.onnx"),
        "FUNCTIONS=FALSE (Standard Export)"
    )
    
    examine_onnx_model(
        str(base_dir / "model_functions_true.onnx"), 
        "FUNCTIONS=TRUE (Modules as Functions)"
    )
    
    print(f"\n{'='*80}")
    print("STRUCTURE COMPARISON ANALYSIS")
    print('='*80)
    
    print(f"\n🔍 STRUCTURAL DIFFERENCES:")
    print(f"   • Functions=False: Flat graph with all operations visible")
    print(f"   • Functions=True: Hierarchical with operations grouped in functions")
    
    print(f"\n📊 GRANULARITY LEVELS:")
    print(f"   • Functions=False: Individual operations (Gemm, Add, Relu)")
    print(f"   • Functions=True: Module-level functions (Linear, ReLU, SimpleModel)")
    
    print(f"\n🎯 HIERARCHY PRESERVATION:")
    print(f"   • Functions=False: No structural hierarchy (flat operations)")
    print(f"   • Functions=True: Module boundaries preserved as function boundaries")
    
    print(f"\n⚖️  MODELEXPORT COMPARISON:")
    print(f"   • export_modules_as_functions: Groups operations into module functions")
    print(f"   • ModelExport HTP: Tags individual operations with module metadata")
    print(f"   • Key difference: Grouping vs Tagging")
    
    print(f"\n🔬 TRACEABILITY:")
    print(f"   • Functions=False: Direct operation visibility but no hierarchy")
    print(f"   • Functions=True: Module structure but ops hidden inside functions")
    print(f"   • ModelExport HTP: Both operation visibility AND hierarchy metadata")
    
    print(f"\n✅ FINAL VERDICT:")
    print(f"   • export_modules_as_functions provides module-level structure")
    print(f"   • ModelExport HTP provides operation-level traceability")
    print(f"   • Different use cases: structural organization vs fine-grained analysis")
    print(f"   • ModelExport approach better aligned with debugging/analysis needs")


if __name__ == "__main__":
    main()