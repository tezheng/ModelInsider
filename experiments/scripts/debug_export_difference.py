#!/usr/bin/env python3
"""Debug the difference between baseline and hierarchy export."""

import torch
import torch.onnx
import onnx
from transformers import AutoModel, AutoTokenizer
from modelexport.hierarchy_exporter import HierarchyExporter

def debug_export_difference():
    """Debug why hierarchy export differs from baseline."""
    
    print("🔍 DEBUGGING EXPORT DIFFERENCE")
    print("=" * 40)
    
    # Load model
    model = AutoModel.from_pretrained('prajjwal1/bert-tiny')
    tokenizer = AutoTokenizer.from_pretrained('prajjwal1/bert-tiny')
    text = 'Fixed topology test'
    inputs = tokenizer(text, return_tensors='pt', padding=True, truncation=True)
    
    model.eval()
    
    print(f"Input text: '{text}'")
    print(f"Input shape: {inputs['input_ids'].shape}")
    print(f"Inputs: {list(inputs.keys())}")
    
    # Export 1: Direct torch.onnx.export
    print(f"\n📤 Direct torch.onnx.export...")
    tensor_inputs = tuple(inputs.data.values())
    
    torch.onnx.export(
        model,
        tensor_inputs,
        'temp/debug_direct.onnx',
        export_params=True,
        opset_version=14,
        do_constant_folding=True,
        input_names=['input_ids', 'attention_mask'],
        output_names=['last_hidden_state'],
        dynamic_axes={
            'input_ids': {0: 'batch_size', 1: 'sequence'},
            'attention_mask': {0: 'batch_size', 1: 'sequence'},
            'last_hidden_state': {0: 'batch_size', 1: 'sequence'}
        }
    )
    
    direct_model = onnx.load('temp/debug_direct.onnx')
    print(f"Direct export nodes: {len(direct_model.graph.node)}")
    
    # Export 2: Through our exporter's _export_to_onnx method only
    print(f"\n📤 Through our _export_to_onnx method...")
    exporter = HierarchyExporter()
    
    # Bypass all our logic, just call _export_to_onnx
    exporter._export_to_onnx(
        model,
        inputs,
        'temp/debug_our_method.onnx',
        export_params=True,
        opset_version=14,
        do_constant_folding=True,
        input_names=['input_ids', 'attention_mask'],
        output_names=['last_hidden_state'],
        dynamic_axes={
            'input_ids': {0: 'batch_size', 1: 'sequence'},
            'attention_mask': {0: 'batch_size', 1: 'sequence'},
            'last_hidden_state': {0: 'batch_size', 1: 'sequence'}
        }
    )
    
    our_method_model = onnx.load('temp/debug_our_method.onnx')
    print(f"Our method nodes: {len(our_method_model.graph.node)}")
    
    # Export 3: Full hierarchy export
    print(f"\n📤 Full hierarchy export...")
    result = exporter.export(
        model=model,
        example_inputs=inputs,
        output_path='temp/debug_full.onnx',
        export_params=True,
        opset_version=14,
        do_constant_folding=True,
        input_names=['input_ids', 'attention_mask'],
        output_names=['last_hidden_state'],
        dynamic_axes={
            'input_ids': {0: 'batch_size', 1: 'sequence'},
            'attention_mask': {0: 'batch_size', 1: 'sequence'},
            'last_hidden_state': {0: 'batch_size', 1: 'sequence'}
        }
    )
    
    full_model = onnx.load('temp/debug_full.onnx')
    print(f"Full export nodes: {len(full_model.graph.node)}")
    
    # Compare
    print(f"\n📊 COMPARISON:")
    print(f"   Direct torch.onnx.export: {len(direct_model.graph.node)} nodes")
    print(f"   Our _export_to_onnx method: {len(our_method_model.graph.node)} nodes")
    print(f"   Full hierarchy export: {len(full_model.graph.node)} nodes")
    
    if len(direct_model.graph.node) == len(our_method_model.graph.node) == len(full_model.graph.node):
        print("✅ All exports have same node count!")
    else:
        print("❌ Node counts differ - found the issue!")
        
        # Find where the difference occurs
        if len(direct_model.graph.node) != len(our_method_model.graph.node):
            print("   Issue is in our _export_to_onnx method")
        elif len(our_method_model.graph.node) != len(full_model.graph.node):
            print("   Issue is in our full export flow")

if __name__ == "__main__":
    debug_export_difference()