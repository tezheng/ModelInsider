#!/usr/bin/env python3
"""
Fix Module Tagging - Ensure operations are tagged at the correct module level
"""

import torch
from transformers import AutoModel
from enhanced_dag_extractor import EnhancedDAGExtractor
from input_generator import UniversalInputGenerator


def analyze_tagging_issue():
    """Analyze why BertSdpaSelfAttention operations aren't being tagged correctly"""
    print("🔍 ANALYZING TAGGING ISSUE")
    print("=" * 60)
    
    model = AutoModel.from_pretrained('google/bert_uncased_L-2_H-128_A-2')
    generator = UniversalInputGenerator()
    inputs = generator.generate_inputs(model, 'google/bert_uncased_L-2_H-128_A-2')
    
    extractor = EnhancedDAGExtractor()
    extractor.analyze_model_structure(model)
    
    print("📋 MODULE HIERARCHY FOR ATTENTION:")
    target_modules = {}
    for module_name, module_info in extractor.module_hierarchy.items():
        if 'attention.self' in module_name or 'BertSdpaSelfAttention' in module_info['hierarchy_path']:
            target_modules[module_name] = module_info
            print(f"  {module_name}")
            print(f"    → {module_info['hierarchy_path']}")
            print(f"    → Type: {module_info['type']}")
            print(f"    → Depth: {module_info['depth']}")
            print(f"    → Leaf: {module_info['is_leaf']}")
            print(f"    → Params: {module_info['parameter_count']:,}")
            print()
    
    # Check execution trace
    print("📋 EXECUTION TRACE FOR ATTENTION:")
    extractor.trace_execution_with_hooks(model, inputs)
    
    attention_executions = []
    for trace in extractor.execution_trace:
        if 'attention.self' in trace['module'] or 'BertSdpaSelfAttention' in trace['hierarchy_path']:
            attention_executions.append(trace)
            print(f"  {trace['module']} → {trace['hierarchy_path']}")
    
    print(f"\nFound {len(attention_executions)} attention-related executions")
    
    # Check parameter mapping
    print("📋 PARAMETER MAPPING FOR ATTENTION:")
    extractor.create_parameter_mapping(model)
    
    attention_params = {}
    for param_name, param_info in extractor.parameter_mapping.items():
        if 'attention.self' in param_name or 'BertSdpaSelfAttention' in param_info['hierarchy_path']:
            attention_params[param_name] = param_info
            print(f"  {param_name}")
            print(f"    → Module: {param_info['module']}")
            print(f"    → Hierarchy: {param_info['hierarchy_path']}")
    
    print(f"\nFound {len(attention_params)} attention parameters")
    
    # The issue is likely that operations get tagged to the LEAF modules (Linear)
    # instead of their parent module (BertSdpaSelfAttention)
    
    print("\n" + "="*60)
    print("🔧 PROPOSED FIX:")
    print("The issue is that our tagging assigns operations to the most specific (leaf) module.")
    print("For attention mechanisms, we want operations tagged to the parent BertSdpaSelfAttention module.")
    print("We need to modify the tagging strategy to assign attention operations to the correct level.")
    
    return target_modules, attention_executions, attention_params


def test_correct_extraction():
    """Test extraction with the correct tag for full BertSdpaSelfAttention"""
    print("\n🧪 TESTING CORRECT EXTRACTION")
    print("=" * 60)
    
    model = AutoModel.from_pretrained('google/bert_uncased_L-2_H-128_A-2')
    generator = UniversalInputGenerator()
    inputs = generator.generate_inputs(model, 'google/bert_uncased_L-2_H-128_A-2')
    
    # Create enhanced model
    extractor = EnhancedDAGExtractor()
    extractor.analyze_model_structure(model)
    extractor.trace_execution_with_hooks(model, inputs)
    extractor.create_parameter_mapping(model)
    
    import tempfile
    import os
    with tempfile.NamedTemporaryFile(suffix='.onnx', delete=False) as tmp:
        temp_path = tmp.name
    
    try:
        extractor.export_and_analyze_onnx(model, inputs, temp_path)
        enhanced_path = temp_path.replace('.onnx', '_with_tags.onnx')
        
        # Try to find ALL operations that should belong to BertSdpaSelfAttention
        print("🔍 Looking for operations that should belong to BertSdpaSelfAttention...")
        
        # Look for operations between attention input and output
        attention_ops = []
        for op_name, op_data in extractor.operation_metadata.items():
            # Check if operation name suggests it's part of attention
            if any(keyword in op_name.lower() for keyword in [
                'attention', 'query', 'key', 'value', 'matmul', 'softmax', 
                'encoder/layer.0/attention/self'
            ]):
                attention_ops.append((op_name, op_data))
        
        print(f"Found {len(attention_ops)} potential attention operations")
        
        # Group by current tags
        tag_groups = {}
        for op_name, op_data in attention_ops:
            for tag in op_data.get('tags', []):
                if tag not in tag_groups:
                    tag_groups[tag] = []
                tag_groups[tag].append(op_name)
        
        print(f"\nAttention operations grouped by current tags:")
        for tag, ops in tag_groups.items():
            print(f"  {tag}: {len(ops)} operations")
            if 'SdpaSelfAttention' in tag:
                print(f"    First 5: {ops[:5]}")
        
        # The real issue: we need to collect ALL operations that execute
        # during the BertSdpaSelfAttention forward pass, not just the ones
        # tagged to specific Linear sub-modules
        
    finally:
        if os.path.exists(temp_path):
            os.unlink(temp_path)
        enhanced_path = temp_path.replace('.onnx', '_with_tags.onnx')
        if os.path.exists(enhanced_path):
            os.unlink(enhanced_path)


if __name__ == "__main__":
    target_modules, attention_executions, attention_params = analyze_tagging_issue()
    test_correct_extraction()