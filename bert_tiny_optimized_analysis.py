#!/usr/bin/env python3
"""
BERT-tiny Analysis with Proper Export Configuration
==================================================

This script analyzes what ONNX operations we actually get when using proper
export configuration with optimizations enabled (should fuse GELU to avoid Erf ops).
"""

import torch
from transformers import AutoModel, AutoTokenizer
import onnx
import tempfile
from pathlib import Path
import json
from typing import Dict, List, Any


def export_with_optimization():
    """Export BERT-tiny with proper optimization configuration."""
    
    print("🚀 BERT-tiny Export with Optimization")
    print("=" * 50)
    
    # Load model
    model_name = "prajjwal1/bert-tiny"
    model = AutoModel.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # Prepare inputs
    text = "Hello world"
    inputs = tokenizer(text, return_tensors="pt", max_length=128, padding="max_length", truncation=True)
    input_ids = inputs["input_ids"]
    attention_mask = inputs["attention_mask"]
    
    print(f"Model: {model_name}")
    print(f"Input text: '{text}'")
    print(f"Input shapes: input_ids={input_ids.shape}, attention_mask={attention_mask.shape}")
    
    # Export configurations to test
    configs = {
        'basic': {
            'opset_version': 17,
            'do_constant_folding': False,
            'optimization_level': None
        },
        'optimized': {
            'opset_version': 17,
            'do_constant_folding': True,
            'optimization_level': None
        },
        'full_optimized': {
            'opset_version': 17,
            'do_constant_folding': True,
            'optimization_level': None,
            'custom_opsets': None
        }
    }
    
    results = {}
    
    for config_name, config in configs.items():
        print(f"\n📊 Testing configuration: {config_name}")
        
        with tempfile.NamedTemporaryFile(suffix='.onnx', delete=False) as tmp:
            temp_path = tmp.name
        
        try:
            # Export with current configuration
            torch.onnx.export(
                model,
                (input_ids, attention_mask),
                temp_path,
                input_names=['input_ids', 'attention_mask'],
                output_names=['last_hidden_state'],
                dynamic_axes={
                    'input_ids': {0: 'batch_size', 1: 'sequence'},
                    'attention_mask': {0: 'batch_size', 1: 'sequence'}
                },
                **config
            )
            
            # Analyze resulting operations
            onnx_model = onnx.load(temp_path)
            operations = analyze_operations(onnx_model)
            results[config_name] = operations
            
            print(f"   Total nodes: {operations['total_nodes']}")
            print(f"   Operation types: {len(operations['by_type'])}")
            
            # Check for optimization indicators
            has_erf = 'Erf' in operations['by_type']
            has_layer_norm = 'LayerNormalization' in operations['by_type']
            has_gelu = 'Gelu' in operations['by_type']
            
            print(f"   Has Erf ops: {has_erf} ({'⚠️ not fused' if has_erf else '✅ fused'})")
            print(f"   Has LayerNormalization: {has_layer_norm}")
            print(f"   Has GELU: {has_gelu}")
            
        except Exception as e:
            print(f"   ❌ Export failed: {e}")
            results[config_name] = {'error': str(e)}
        finally:
            Path(temp_path).unlink(missing_ok=True)
    
    return results


def analyze_operations(onnx_model) -> Dict[str, Any]:
    """Analyze ONNX operations and categorize them."""
    
    operations = {
        'total_nodes': len(onnx_model.graph.node),
        'by_type': {},
        'activation_ops': [],
        'attention_ops': [],
        'infrastructure_ops': []
    }
    
    for node in onnx_model.graph.node:
        op_type = node.op_type
        if op_type not in operations['by_type']:
            operations['by_type'][op_type] = 0
        operations['by_type'][op_type] += 1
        
        # Categorize operations
        if op_type in ['Erf', 'Gelu', 'Relu', 'Tanh', 'Sigmoid']:
            operations['activation_ops'].append(op_type)
        elif op_type in ['MatMul', 'Softmax', 'Div', 'Mul']:
            operations['attention_ops'].append(op_type)
        elif op_type in ['Shape', 'Cast', 'Constant', 'Unsqueeze']:
            operations['infrastructure_ops'].append(op_type)
    
    return operations


def create_optimized_config():
    """Create the proper export configuration for BERT models."""
    
    print("\n🔧 RECOMMENDED EXPORT CONFIGURATION")
    print("=" * 50)
    
    optimized_config = {
        "export_params": True,
        "opset_version": 17,
        "do_constant_folding": True,
        "input_names": ["input_ids", "attention_mask"],
        "output_names": ["last_hidden_state"],
        "dynamic_axes": {
            "input_ids": {"0": "batch_size", "1": "sequence_length"},
            "attention_mask": {"0": "batch_size", "1": "sequence_length"}
        },
        "input_specs": {
            "input_ids": {"dtype": "long", "range": [0, 30522]},  # BERT vocab size
            "attention_mask": {"dtype": "long", "range": [0, 1]}
        },
        "optimization": {
            "constant_folding": True,
            "operator_fusion": True,
            "gelu_fusion": True
        }
    }
    
    # Save optimized config
    output_path = "bert_tiny_optimized_config.json"
    with open(output_path, 'w') as f:
        json.dump(optimized_config, f, indent=2)
    
    print("✅ Optimized configuration created:")
    print("   - Constant folding enabled")
    print("   - GELU fusion should eliminate separate Erf operations")
    print("   - Proper input specifications for BERT")
    print(f"   - Saved to: {output_path}")
    
    return optimized_config


def update_ground_truth_with_optimization():
    """Update ground truth to reflect properly optimized ONNX operations."""
    
    print("\n🎯 UPDATED GROUND TRUTH WITH OPTIMIZATION")
    print("=" * 50)
    
    print("CORRECTED EXPECTED OPERATIONS:")
    print()
    
    optimized_expectations = {
        'critical_operations': {
            'MatMul': 'MUST be tagged (attention, dense layers)',
            'Add': 'MUST be tagged (residual connections, bias)',
            'Softmax': 'MUST be tagged (attention probabilities)',
            'Mul': 'MUST be tagged (attention masking, scaling)',
            'Div': 'MUST be tagged (attention scaling)',
        },
        'semantic_operations': {
            'Gather': 'SHOULD be tagged (embedding lookups)',
            'Sub': 'SHOULD be tagged (attention masking)',
            'LayerNormalization': 'SHOULD be tagged (layer normalization - fused)',
            'Sqrt': 'SHOULD be tagged (if not fused into LayerNorm)',
            'ReduceSum': 'SHOULD be tagged (aggregation operations)',
        },
        'activation_operations': {
            'Gelu': 'SHOULD be tagged (if GELU fusion available)',
            'Erf': '⚠️ Should NOT appear if properly optimized (unfused GELU)',
            'Tanh': 'MAY appear in pooler activation',
        },
        'structural_operations': {
            'Reshape': 'MAY be tagged (context-dependent)',
            'Transpose': 'MAY be tagged (attention head reshaping)',
            'Concat': 'MAY be tagged (multi-head concatenation)',
            'Slice': 'MAY be tagged (positional embeddings)',
            'Unsqueeze': 'MAY be tagged (dimension expansion)',
        },
        'support_operations': {
            'Shape': 'Empty tags acceptable (infrastructure)',
            'Cast': 'Empty tags acceptable (type conversion)',
            'Constant': 'Empty tags acceptable (unless parameter)',
            'Equal': 'Empty tags acceptable (mask generation)',
            'Where': 'Empty tags acceptable (conditional logic)',
        }
    }
    
    for category, ops in optimized_expectations.items():
        print(f"   {category.upper().replace('_', ' ')}:")
        for op_type, description in ops.items():
            print(f"      {op_type:20s}: {description}")
        print()
    
    print("🔍 KEY OPTIMIZATION INDICATORS:")
    print("   ✅ GELU should be fused (no separate Erf operations)")
    print("   ✅ LayerNormalization should be fused")
    print("   ✅ Constants should be folded")
    print("   ⚠️ If you see Erf operations, optimization is not working")
    
    return optimized_expectations


def main():
    """Main analysis with proper optimization configuration."""
    
    print("🎯 BERT-TINY OPTIMIZED ANALYSIS")
    print("=" * 80)
    print("Analyzing ONNX export with proper optimization to get accurate ground truth")
    print("=" * 80)
    
    # 1. Test different export configurations
    export_results = export_with_optimization()
    
    # 2. Create recommended configuration
    optimized_config = create_optimized_config()
    
    # 3. Update ground truth expectations
    updated_expectations = update_ground_truth_with_optimization()
    
    # 4. Save comprehensive results
    analysis_results = {
        'export_tests': export_results,
        'recommended_config': optimized_config,
        'updated_expectations': updated_expectations,
        'key_findings': {
            'optimization_needed': True,
            'config_file_usage': 'Use bert_tiny_optimized_config.json for accurate exports',
            'erf_indicator': 'Presence of Erf ops indicates missing GELU fusion',
            'ground_truth_update': 'Ground truth should reflect optimized operations'
        }
    }
    
    output_path = "temp/bert_tiny_optimized_analysis.json"
    Path("temp").mkdir(exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(analysis_results, f, indent=2)
    
    print(f"\n💾 Results saved to: {output_path}")
    
    print("\n" + "=" * 80)
    print("✅ ANALYSIS COMPLETE")
    print("=" * 80)
    print("\nKEY FINDINGS:")
    print("• Export configuration matters for accurate ground truth")
    print("• GELU fusion should eliminate separate Erf operations")
    print("• Use bert_tiny_optimized_config.json for proper exports")
    print("• Ground truth document should reflect optimized operations")


if __name__ == "__main__":
    main()