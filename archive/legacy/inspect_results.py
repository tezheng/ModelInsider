#!/usr/bin/env python3
"""
Inspect Test Results - Show detailed analysis of extracted subgraphs
"""

import json
from pathlib import Path

import onnx


def inspect_extracted_models():
    """Inspect the extracted ONNX models in detail"""
    
    test_dir = Path("temp/simple_test")
    
    print("🔍 DETAILED INSPECTION OF EXTRACTED SUBGRAPHS")
    print("=" * 60)
    
    # Load test results
    with open(test_dir / "test_results.json") as f:
        results = json.load(f)
    
    print("📊 TEST RESULTS SUMMARY:")
    print(json.dumps(results, indent=2))
    
    # Inspect each extracted model
    extracted_files = [
        ("BertEmbeddings", "extracted__BertModel_BertEmbeddings_Embedding.onnx"),
        ("BertSelfOutput", "extracted__BertModel_BertEncoder_ModuleList.0_BertAttention_BertSelfOutput_Linear.onnx")
    ]
    
    for name, filename in extracted_files:
        print(f"\n{'='*50}")
        print(f"INSPECTING {name.upper()}")
        print(f"{'='*50}")
        
        model_path = test_dir / filename
        if not model_path.exists():
            print(f"❌ File not found: {model_path}")
            continue
        
        try:
            # Load ONNX model
            model = onnx.load(str(model_path))
            
            print(f"📁 File: {filename}")
            print(f"📏 Size: {model_path.stat().st_size / 1024:.1f} KB")
            
            # Basic model info
            print(f"\n📊 MODEL STRUCTURE:")
            print(f"   Nodes: {len(model.graph.node)}")
            print(f"   Inputs: {len(model.graph.input)}")
            print(f"   Outputs: {len(model.graph.output)}")
            print(f"   Initializers: {len(model.graph.initializer)}")
            
            # Show inputs
            print(f"\n📥 INPUTS:")
            for i, inp in enumerate(model.graph.input):
                shape = [dim.dim_value if dim.dim_value > 0 else 'dynamic' 
                        for dim in inp.type.tensor_type.shape.dim]
                print(f"   {i+1}. {inp.name}: {shape}")
            
            # Show outputs  
            print(f"\n📤 OUTPUTS:")
            for i, out in enumerate(model.graph.output):
                shape = [dim.dim_value if dim.dim_value > 0 else 'dynamic' 
                        for dim in out.type.tensor_type.shape.dim]
                print(f"   {i+1}. {out.name}: {shape}")
            
            # Show initializers (parameters)
            print(f"\n⚙️  PARAMETERS:")
            for i, init in enumerate(model.graph.initializer):
                print(f"   {i+1}. {init.name}: {list(init.dims)}")
            
            # Show operations
            print(f"\n🔧 OPERATIONS:")
            op_counts = {}
            for node in model.graph.node:
                op_counts[node.op_type] = op_counts.get(node.op_type, 0) + 1
            
            for op_type, count in sorted(op_counts.items()):
                print(f"   {op_type}: {count}")
            
            # Show first few nodes in detail
            print(f"\n🔍 FIRST 5 OPERATIONS:")
            for i, node in enumerate(model.graph.node[:5]):
                inputs_str = f"[{', '.join(node.input[:2])}{'...' if len(node.input) > 2 else ''}]"
                outputs_str = f"[{', '.join(node.output[:2])}{'...' if len(node.output) > 2 else ''}]"
                print(f"   {i+1}. {node.op_type}: {inputs_str} → {outputs_str}")
            
            # Validation
            print(f"\n✅ VALIDATION:")
            try:
                onnx.checker.check_model(model)
                print("   ✅ Model passes ONNX validation")
            except Exception as e:
                print(f"   ⚠️  Validation issue: {str(e)[:100]}...")
            
            # Check metadata
            print(f"\n📋 METADATA:")
            for prop in model.metadata_props:
                if len(prop.value) > 50:
                    value_preview = prop.value[:50] + "..."
                else:
                    value_preview = prop.value
                print(f"   {prop.key}: {value_preview}")
            
        except Exception as e:
            print(f"❌ Failed to inspect {filename}: {e}")
    
    # Compare with original
    print(f"\n{'='*60}")
    print("COMPARISON WITH ORIGINAL MODEL")
    print(f"{'='*60}")
    
    try:
        original_path = test_dir / "bert_enhanced_with_tags.onnx"
        original_model = onnx.load(str(original_path))
        
        print(f"📊 ORIGINAL MODEL:")
        print(f"   Nodes: {len(original_model.graph.node)}")
        print(f"   Size: {original_path.stat().st_size / 1024 / 1024:.1f} MB")
        
        print(f"\n📊 EXTRACTED MODELS:")
        total_extracted_nodes = 0
        total_extracted_size = 0
        
        for name, filename in extracted_files:
            model_path = test_dir / filename
            if model_path.exists():
                model = onnx.load(str(model_path))
                size_kb = model_path.stat().st_size / 1024
                total_extracted_nodes += len(model.graph.node)
                total_extracted_size += size_kb
                print(f"   {name}: {len(model.graph.node)} nodes, {size_kb:.1f} KB")
        
        print(f"\n📈 EXTRACTION EFFICIENCY:")
        coverage = total_extracted_nodes / len(original_model.graph.node) * 100
        print(f"   Node coverage: {total_extracted_nodes}/{len(original_model.graph.node)} ({coverage:.1f}%)")
        print(f"   Size reduction: {total_extracted_size:.1f} KB vs {original_path.stat().st_size / 1024:.1f} KB")
        
    except Exception as e:
        print(f"❌ Comparison failed: {e}")


if __name__ == "__main__":
    inspect_extracted_models()