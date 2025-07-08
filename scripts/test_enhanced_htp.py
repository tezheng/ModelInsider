#!/usr/bin/env python3
"""
Test script to compare original HTP vs Enhanced HTP with proper trace capture.

This script:
1. Exports bert-tiny using both strategies
2. Compares the tagging quality and coverage
3. Analyzes cross-layer contamination reduction
"""

import torch
from transformers import AutoModel, AutoTokenizer
from pathlib import Path
import json
import onnx
from collections import defaultdict, Counter

# Import both exporters
from modelexport.strategies.htp.htp_hierarchy_exporter import HierarchyExporter
from modelexport.strategies.htp.htp_hierarchy_exporter_enhanced import EnhancedHTPExporter


def analyze_tag_distribution(onnx_path: str) -> dict:
    """Analyze tag distribution in an ONNX model."""
    
    model = onnx.load(onnx_path)
    
    # Extract tags from metadata
    tags_by_node = {}
    tag_counter = Counter()
    layer_mixing = defaultdict(set)
    
    for node in model.graph.node:
        node_name = node.name or f"{node.op_type}_unknown"
        
        # Look for hierarchy metadata
        hierarchy_tags = []
        for attr in node.attribute:
            if attr.name == "hierarchy":
                hierarchy_tags = [s.decode('utf-8') if isinstance(s, bytes) else s 
                                 for s in attr.strings]
                break
        
        tags_by_node[node_name] = hierarchy_tags
        
        # Count tags and detect layer mixing
        for tag in hierarchy_tags:
            tag_counter[tag] += 1
            
            # Extract layer number if present
            if "Layer." in tag:
                layer_match = tag.split("Layer.")[1].split("/")[0]
                layer_mixing[node_name].add(layer_match)
    
    # Calculate statistics
    total_nodes = len(model.graph.node)
    tagged_nodes = len([n for n in tags_by_node.values() if n])
    multi_layer_nodes = len([n for n, layers in layer_mixing.items() if len(layers) > 1])
    
    # Find most contaminated operations
    contaminated_ops = [
        (node, list(layers)) 
        for node, layers in layer_mixing.items() 
        if len(layers) > 1
    ]
    contaminated_ops.sort(key=lambda x: len(x[1]), reverse=True)
    
    return {
        'total_nodes': total_nodes,
        'tagged_nodes': tagged_nodes,
        'coverage_rate': tagged_nodes / total_nodes if total_nodes > 0 else 0,
        'unique_tags': len(tag_counter),
        'multi_layer_nodes': multi_layer_nodes,
        'contamination_rate': multi_layer_nodes / total_nodes if total_nodes > 0 else 0,
        'top_contaminated': contaminated_ops[:5],
        'tag_distribution': tag_counter.most_common(10)
    }


def main():
    """Run comparison test."""
    
    print("Enhanced HTP Trace Capture Test")
    print("=" * 60)
    
    # Setup
    output_dir = Path("temp/enhanced_htp_test")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load model
    print("\nLoading bert-tiny model...")
    model_name = "prajjwal1/bert-tiny"
    model = AutoModel.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # Prepare inputs
    inputs = tokenizer("Hello world!", return_tensors="pt")
    
    # Test 1: Original HTP
    print("\n1. Testing Original HTP Implementation...")
    original_path = output_dir / "bert_tiny_original_htp.onnx"
    
    original_exporter = HierarchyExporter(strategy="htp")
    original_result = original_exporter.export(
        model,
        inputs,
        str(original_path),
        opset_version=17,
        input_names=['input_ids', 'attention_mask'],
        output_names=['last_hidden_state'],
        dynamic_axes={
            'input_ids': {0: 'batch_size', 1: 'sequence'},
            'attention_mask': {0: 'batch_size', 1: 'sequence'},
            'last_hidden_state': {0: 'batch_size', 1: 'sequence'}
        }
    )
    
    print(f"Original result: {original_result}")
    
    # Test 2: Enhanced HTP
    print("\n2. Testing Enhanced HTP Implementation...")
    enhanced_path = output_dir / "bert_tiny_enhanced_htp.onnx"
    
    enhanced_exporter = EnhancedHTPExporter(strategy="htp_enhanced")
    enhanced_result = enhanced_exporter.export(
        model,
        inputs,
        str(enhanced_path),
        opset_version=17,
        input_names=['input_ids', 'attention_mask'],
        output_names=['last_hidden_state'],
        dynamic_axes={
            'input_ids': {0: 'batch_size', 1: 'sequence'},
            'attention_mask': {0: 'batch_size', 1: 'sequence'},
            'last_hidden_state': {0: 'batch_size', 1: 'sequence'}
        }
    )
    
    print(f"Enhanced result: {enhanced_result}")
    
    # Get trace statistics
    trace_stats = enhanced_exporter.get_trace_statistics()
    
    # Analyze both models
    print("\n3. Analyzing Tag Distributions...")
    original_analysis = analyze_tag_distribution(str(original_path))
    enhanced_analysis = analyze_tag_distribution(str(enhanced_path))
    
    # Compare results
    print("\n" + "=" * 60)
    print("COMPARISON RESULTS")
    print("=" * 60)
    
    print(f"\n{'Metric':<30} {'Original HTP':<20} {'Enhanced HTP':<20}")
    print("-" * 70)
    print(f"{'Total Operations':<30} {original_analysis['total_nodes']:<20} {enhanced_analysis['total_nodes']:<20}")
    print(f"{'Tagged Operations':<30} {original_analysis['tagged_nodes']:<20} {enhanced_analysis['tagged_nodes']:<20}")
    print(f"{'Coverage Rate':<30} {original_analysis['coverage_rate']:<20.1%} {enhanced_analysis['coverage_rate']:<20.1%}")
    print(f"{'Unique Tags':<30} {original_analysis['unique_tags']:<20} {enhanced_analysis['unique_tags']:<20}")
    print(f"{'Multi-Layer Operations':<30} {original_analysis['multi_layer_nodes']:<20} {enhanced_analysis['multi_layer_nodes']:<20}")
    print(f"{'Contamination Rate':<30} {original_analysis['contamination_rate']:<20.1%} {enhanced_analysis['contamination_rate']:<20.1%}")
    
    print("\n" + "-" * 70)
    print("ENHANCED HTP TRACE STATISTICS")
    print("-" * 70)
    print(f"{'Trace-Captured Operations':<30} {trace_stats['trace_captured_operations']:<20} ({trace_stats['trace_capture_rate']:.1%})")
    print(f"{'Parameter-Based Operations':<30} {trace_stats['parameter_based_operations']:<20}")
    print(f"{'Untagged Operations':<30} {trace_stats['untagged_operations']:<20}")
    
    # Show contamination reduction
    if original_analysis['multi_layer_nodes'] > 0:
        reduction = (original_analysis['multi_layer_nodes'] - enhanced_analysis['multi_layer_nodes']) / original_analysis['multi_layer_nodes']
        print(f"\n🎯 Cross-Layer Contamination Reduction: {reduction:.1%}")
    
    # Show top contaminated operations
    print("\n" + "-" * 70)
    print("TOP CONTAMINATED OPERATIONS")
    print("-" * 70)
    
    print("\nOriginal HTP:")
    for node, layers in original_analysis['top_contaminated'][:3]:
        print(f"  {node}: layers {layers}")
    
    print("\nEnhanced HTP:")
    if enhanced_analysis['top_contaminated']:
        for node, layers in enhanced_analysis['top_contaminated'][:3]:
            print(f"  {node}: layers {layers}")
    else:
        print("  No multi-layer contamination detected!")
    
    # Save detailed results
    results = {
        'original': original_analysis,
        'enhanced': enhanced_analysis,
        'trace_stats': trace_stats,
        'improvement': {
            'contamination_reduction': reduction if original_analysis['multi_layer_nodes'] > 0 else 0,
            'coverage_improvement': enhanced_analysis['coverage_rate'] - original_analysis['coverage_rate'],
        }
    }
    
    results_path = output_dir / "comparison_results.json"
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n✅ Detailed results saved to: {results_path}")
    print(f"📁 ONNX models saved to: {output_dir}")


if __name__ == "__main__":
    main()