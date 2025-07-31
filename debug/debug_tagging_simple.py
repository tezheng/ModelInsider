#!/usr/bin/env python3
"""
Simple debug script to test tagging logic without full model export.
"""

import tempfile

import torch
from transformers import AutoModel, AutoTokenizer

from modelexport.hierarchy_exporter import HierarchyExporter


def debug_tagging():
    print("🔍 Debug: Simple Tagging Test")
    print("=" * 50)
    
    # Use smaller sequence for faster testing
    model = AutoModel.from_pretrained('prajjwal1/bert-tiny')
    tokenizer = AutoTokenizer.from_pretrained('prajjwal1/bert-tiny')
    model.eval()
    
    # Short input for speed
    inputs = tokenizer("test", return_tensors='pt', max_length=8, padding='max_length', truncation=True)
    
    print(f"✅ Model loaded: {model.__class__.__name__}")
    print(f"✅ Input shapes: {[(k, v.shape) for k, v in inputs.items()]}")
    
    # Test the exporter step by step
    exporter = HierarchyExporter()
    exporter._reset_state()
    exporter._model = model  # Set model AFTER reset
    
    # Step 1: Register hooks and trace
    print("\n🪝 Step 1: Hook Registration")
    exporter._register_hooks(model)
    print(f"✅ Registered {len(exporter._hooks)} hooks")
    
    # Step 2: Trace execution
    print("\n🔄 Step 2: Trace Execution")
    with torch.no_grad():
        exporter._trace_model_execution(model, inputs)
    
    print(f"✅ Captured {len(exporter._operation_context)} operation contexts")
    
    # Show first few contexts
    print("\nFirst 5 operation contexts:")
    for i, (name, context) in enumerate(exporter._operation_context.items()):
        if i >= 5:
            break
        print(f"  {name}: {context['tag']}")
    
    # Step 3: Export to ONNX and build tag mapping
    print("\n📦 Step 3: ONNX Export and Tag Mapping")
    with tempfile.NamedTemporaryFile(suffix='.onnx', delete=False) as f:
        temp_path = f.name
    
    try:
        exporter._export_to_onnx(model, inputs, temp_path, opset_version=14)
        print(f"✅ ONNX exported to {temp_path}")
        
        exporter._build_tag_mapping_from_onnx(temp_path)
        print(f"✅ Tag mapping built: {len(exporter._tag_mapping)} operations")
        
        # Count tagged operations
        tagged_ops = len([op for op in exporter._tag_mapping.values() if op.get('tags', [])])
        print(f"✅ Tagged operations: {tagged_ops}")
        
        # Show tag distribution
        tag_stats = exporter._compute_tag_statistics()
        print(f"\nTag distribution (top 5):")
        for tag, count in sorted(tag_stats.items(), key=lambda x: x[1], reverse=True)[:5]:
            print(f"  {tag}: {count} operations")
        
        # Check specific patterns that tests are looking for
        bert_self_output_count = sum(1 for op_info in exporter._tag_mapping.values()
                                   if any('BertSelfOutput' in tag for tag in op_info.get('tags', [])))
        
        print(f"\n🎯 Specific Check:")
        print(f"   BertSelfOutput tagged operations: {bert_self_output_count}")
        
        if bert_self_output_count == 0:
            print("   ❌ This explains why tests are failing - no BertSelfOutput tags found")
        else:
            print("   ✅ BertSelfOutput tags found")
            
    finally:
        exporter._remove_hooks()
        import os
        os.unlink(temp_path)
    
    print("\n✅ Debug complete!")

if __name__ == "__main__":
    debug_tagging()