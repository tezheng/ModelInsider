#!/usr/bin/env python3
"""
Analyze Untagged Nodes - Understand why tag coverage is not 100%
"""

import onnx
import json
from collections import defaultdict
from pathlib import Path


def analyze_untagged_nodes(model_name: str):
    """Analyze which nodes are not tagged and why"""
    print(f"\n{'='*60}")
    print(f"ANALYZING UNTAGGED NODES: {model_name.upper()}")
    print(f"{'='*60}")
    
    model_safe_name = model_name.replace('/', '_')
    
    # Load ONNX model with tags
    onnx_path = f"temp/onnx_models/{model_safe_name}_with_tags.onnx"
    onnx_model = onnx.load(onnx_path)
    
    # Load static metadata
    metadata_path = f"temp/test_outputs/{model_safe_name}_operation_metadata.json"
    with open(metadata_path) as f:
        static_metadata = json.load(f)
    
    # Categorize all nodes
    tagged_nodes = []
    untagged_nodes = []
    node_types = defaultdict(int)
    untagged_types = defaultdict(int)
    
    for node in onnx_model.graph.node:
        has_tags = any(attr.name in ["source_module", "hierarchy_tags"] for attr in node.attribute)
        
        node_types[node.op_type] += 1
        
        if has_tags:
            tagged_nodes.append({
                'name': node.name,
                'op_type': node.op_type,
                'inputs': list(node.input),
                'outputs': list(node.output)
            })
        else:
            untagged_types[node.op_type] += 1
            untagged_nodes.append({
                'name': node.name,
                'op_type': node.op_type,
                'inputs': list(node.input),
                'outputs': list(node.output)
            })
    
    # Analysis
    total_nodes = len(onnx_model.graph.node)
    tagged_count = len(tagged_nodes)
    untagged_count = len(untagged_nodes)
    
    print(f"📊 OVERALL STATISTICS:")
    print(f"   Total nodes: {total_nodes}")
    print(f"   Tagged nodes: {tagged_count} ({tagged_count/total_nodes:.1%})")
    print(f"   Untagged nodes: {untagged_count} ({untagged_count/total_nodes:.1%})")
    
    print(f"\n🏷️  NODE TYPES BREAKDOWN:")
    for op_type, count in sorted(node_types.items(), key=lambda x: x[1], reverse=True):
        untagged_this_type = untagged_types.get(op_type, 0)
        tagged_this_type = count - untagged_this_type
        tag_rate = tagged_this_type / count if count > 0 else 0
        print(f"   {op_type:15} Total: {count:3d} | Tagged: {tagged_this_type:3d} | Untagged: {untagged_this_type:3d} | Rate: {tag_rate:.1%}")
    
    print(f"\n❌ TOP UNTAGGED OPERATION TYPES:")
    for op_type, count in sorted(untagged_types.items(), key=lambda x: x[1], reverse=True)[:10]:
        print(f"   {op_type:15} {count:3d} untagged nodes")
    
    print(f"\n🔍 SAMPLE UNTAGGED NODES:")
    for i, node in enumerate(untagged_nodes[:10]):
        print(f"   {i+1}. {node['op_type']:12} | {node['name']}")
        if node['inputs']:
            print(f"      Inputs:  {node['inputs'][:3]}")  # Show first 3 inputs
        if node['outputs']:
            print(f"      Outputs: {node['outputs'][:3]}")  # Show first 3 outputs
    
    # Analyze why nodes are untagged
    print(f"\n🔬 ANALYSIS: Why nodes are untagged")
    
    # Check if untagged nodes use parameters
    param_names = set()
    for init in onnx_model.graph.initializer:
        param_names.add(init.name)
    
    untagged_using_params = 0
    untagged_no_params = 0
    
    for node in untagged_nodes:
        uses_params = any(inp in param_names for inp in node['inputs'])
        if uses_params:
            untagged_using_params += 1
        else:
            untagged_no_params += 1
    
    print(f"   Untagged nodes using parameters: {untagged_using_params}")
    print(f"   Untagged nodes NOT using parameters: {untagged_no_params}")
    
    # Check graph inputs/outputs
    model_inputs = {inp.name for inp in onnx_model.graph.input}
    model_outputs = {out.name for out in onnx_model.graph.output}
    
    untagged_input_related = 0
    untagged_output_related = 0
    untagged_internal = 0
    
    for node in untagged_nodes:
        is_input_related = any(inp in model_inputs for inp in node['inputs'])
        is_output_related = any(out in model_outputs for out in node['outputs'])
        
        if is_input_related:
            untagged_input_related += 1
        elif is_output_related:
            untagged_output_related += 1
        else:
            untagged_internal += 1
    
    print(f"   Untagged nodes related to model inputs: {untagged_input_related}")
    print(f"   Untagged nodes related to model outputs: {untagged_output_related}")
    print(f"   Untagged internal nodes: {untagged_internal}")
    
    return {
        'total_nodes': total_nodes,
        'tagged_count': tagged_count,
        'untagged_count': untagged_count,
        'untagged_types': dict(untagged_types),
        'untagged_using_params': untagged_using_params,
        'untagged_no_params': untagged_no_params,
        'untagged_input_related': untagged_input_related,
        'untagged_output_related': untagged_output_related,
        'untagged_internal': untagged_internal
    }


def main():
    """Analyze all tested models"""
    models = [
        'google_bert_uncased_L-2_H-128_A-2',
        'resnet18', 
        'google_vit-base-patch16-224'
    ]
    
    print("ANALYZING UNTAGGED NODES ACROSS ALL MODELS")
    print("=" * 80)
    
    results = {}
    
    for model in models:
        try:
            results[model] = analyze_untagged_nodes(model)
        except Exception as e:
            print(f"❌ Failed to analyze {model}: {e}")
    
    # Summary across models
    print(f"\n{'='*80}")
    print("SUMMARY ACROSS ALL MODELS")
    print(f"{'='*80}")
    
    for model, result in results.items():
        if result:
            tag_rate = result['tagged_count'] / result['total_nodes']
            print(f"\n📊 {model.upper()}:")
            print(f"   Tag Coverage: {tag_rate:.1%} ({result['tagged_count']}/{result['total_nodes']})")
            print(f"   Main untagged types: {list(result['untagged_types'].keys())[:5]}")
            print(f"   Untagged using params: {result['untagged_using_params']}")
            print(f"   Untagged input/output related: {result['untagged_input_related'] + result['untagged_output_related']}")
    
    print(f"\n🎯 INSIGHTS:")
    print("   • Lower tag coverage typically means more intermediate/reshape operations")
    print("   • Operations not using model parameters are harder to assign to modules")
    print("   • Input/output preprocessing operations are intentionally excluded")
    print("   • ResNet has higher coverage due to simpler operation patterns")
    print("   • Transformers have many intermediate tensor operations")


if __name__ == "__main__":
    main()