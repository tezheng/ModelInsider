#!/usr/bin/env python3
"""
Test V2 HTP Hierarchy Exporter with BERT-tiny
"""

import torch
from transformers import AutoModel, AutoTokenizer
import sys
from pathlib import Path

# Add current directory to path
sys.path.append(str(Path(__file__).parent))

from htp_hierarchy_exporter_v2 import HierarchyExporterV2

def test_bert_tiny():
    print("🚀 Testing V2 HTP Hierarchy Exporter with BERT-tiny")
    print("=" * 60)
    
    # Load BERT-tiny model
    print("📥 Loading prajjwal1/bert-tiny...")
    model = AutoModel.from_pretrained("prajjwal1/bert-tiny")
    tokenizer = AutoTokenizer.from_pretrained("prajjwal1/bert-tiny")
    
    # Prepare inputs
    text = "Hello world"
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
    example_inputs = (inputs['input_ids'], inputs['attention_mask'])
    
    # Export with V2
    print("\n🔄 Exporting with V2 implementation...")
    exporter = HierarchyExporterV2()
    
    try:
        result = exporter.export(
            model=model,
            example_inputs=example_inputs,
            output_path="temp/bert_tiny_v2.onnx",
            input_names=['input_ids', 'attention_mask'],
            output_names=['last_hidden_state'],
            dynamic_axes={
                'input_ids': {0: 'batch_size', 1: 'sequence_length'},
                'attention_mask': {0: 'batch_size', 1: 'sequence_length'},
                'last_hidden_state': {0: 'batch_size', 1: 'sequence_length'}
            },
            opset_version=17
        )
        
        print(f"\n✅ V2 Export successful!")
        print(f"Result: {result}")
        
        # Load and check the hierarchy metadata
        import json
        with open("temp/bert_tiny_v2_hierarchy.json", 'r') as f:
            metadata = json.load(f)
        
        print(f"\n📊 V2 Export Statistics:")
        print(f"   Total operations: {metadata['summary']['total_operations']}")
        print(f"   Tagged operations: {metadata['summary']['tagged_operations']}")
        print(f"   Coverage: {(metadata['summary']['tagged_operations']/metadata['summary']['total_operations'])*100:.1f}%")
        print(f"   Unique tags: {metadata['summary']['unique_tags']}")
        print(f"   Operation traces captured: {metadata['summary']['operation_trace_length']}")
        
        print(f"\n🏷️ Tag Statistics:")
        for tag, count in list(metadata['tag_statistics'].items())[:10]:
            print(f"   {tag}: {count}")
        
        # Debug: Check what traces were captured
        print(f"\n🔍 Analysis of Primary Operation Tagging:")
        primary_ops = []
        for node_name, node_info in metadata['node_tags'].items():
            if node_info['op_type'] in ['MatMul', 'Gemm'] and 'layer.' in node_name:
                primary_ops.append((node_name, node_info['tags']))
        
        print(f"Found {len(primary_ops)} primary operations (MatMul/Gemm in layers):")
        for node_name, tags in primary_ops[:5]:  # Show first 5
            print(f"   {node_name}: {tags}")
        
        # Check specific layer 1 auxiliary operations
        print(f"\n🔍 Checking Layer 1 auxiliary operations:")
        layer1_aux_ops = []
        for node_name, node_info in metadata['node_tags'].items():
            if 'layer.1' in node_name and node_info['op_type'] in ['Constant', 'Shape', 'Reshape', 'Transpose']:
                layer1_aux_ops.append((node_name, node_info['tags']))
        
        print(f"Found {len(layer1_aux_ops)} Layer 1 auxiliary operations:")
        for node_name, tags in layer1_aux_ops[:5]:  # Show first 5
            print(f"   {node_name}: {tags}")
        
    except Exception as e:
        print(f"❌ V2 Export failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True

if __name__ == "__main__":
    # Create temp directory
    Path("temp").mkdir(exist_ok=True)
    
    success = test_bert_tiny()
    sys.exit(0 if success else 1)