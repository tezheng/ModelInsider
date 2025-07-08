#!/usr/bin/env python3
"""Test Enhanced Semantic Exporter with BERT-tiny."""

import json
import torch
from transformers import AutoModel

from modelexport.core.enhanced_semantic_exporter import EnhancedSemanticExporter


def load_config(config_path="export_config_bertmodel.json"):
    """Load export configuration."""
    with open(config_path, 'r') as f:
        return json.load(f)


def generate_inputs_from_config(config, batch_size=1, sequence_length=16):
    """Generate input tensors based on config specifications."""
    inputs = {}
    input_specs = config.get('input_specs', {})
    
    for input_name, spec in input_specs.items():
        dtype_str = spec.get('dtype', 'int')
        dtype = torch.long if dtype_str in ['int', 'long'] else torch.float32
        
        # Generate values within specified range
        if 'range' in spec:
            min_val, max_val = spec['range']
            inputs[input_name] = torch.randint(
                min_val, max_val + 1, (batch_size, sequence_length), dtype=dtype
            )
        else:
            # Default generation
            inputs[input_name] = torch.randint(0, 1000, (batch_size, sequence_length), dtype=dtype)
    
    return inputs


def test_bert_tiny():
    """Test enhanced semantic exporter with BERT-tiny model."""
    print("Testing Enhanced Semantic Exporter with BERT-tiny...")
    
    # Load configuration
    config = load_config()
    print(f"📄 Loaded config with inputs: {config['input_names']}")
    
    # Load model
    model = AutoModel.from_pretrained("prajjwal1/bert-tiny")
    model.eval()
    
    # Generate inputs from config
    input_dict = generate_inputs_from_config(config)
    print(f"🎲 Generated inputs: {list(input_dict.keys())}")
    for name, tensor in input_dict.items():
        print(f"   {name}: {tensor.shape} {tensor.dtype} [{tensor.min():.0f}..{tensor.max():.0f}]")
    
    # Convert dict to tuple for enhanced semantic exporter
    input_args = tuple(input_dict.values())
    
    # Test export with config (temporarily remove dynamic_axes due to PyTorch ONNX issue)
    export_config = config.copy()
    export_config.pop('dynamic_axes', None)  # Remove dynamic axes to avoid export error
    
    exporter = EnhancedSemanticExporter(verbose=True)
    result = exporter.export(
        model=model,
        args=input_args,
        output_path="temp/bert_tiny_enhanced_semantic.onnx",
        **export_config
    )
    
    print(f"⚠️  Note: Removed dynamic_axes from config due to PyTorch ONNX export issue")
    
    print(f"✅ Export successful!")
    print(f"   Total ONNX nodes: {result['total_onnx_nodes']}")
    print(f"   HF module mappings: {result['hf_module_mappings']}")
    print(f"   Operation inferences: {result['operation_inferences']}")
    print(f"   Pattern fallbacks: {result['pattern_fallbacks']}")
    
    # Calculate coverage
    total_mapped = result['hf_module_mappings'] + result['operation_inferences'] + result['pattern_fallbacks']
    coverage = total_mapped / result['total_onnx_nodes'] * 100
    print(f"   Total coverage: {coverage:.1f}%")
    
    # Show confidence distribution
    print(f"   Confidence levels: {result['confidence_levels']}")
    
    # Show some sample mappings
    metadata = exporter.get_semantic_metadata()
    print("\n📊 Sample semantic mappings:")
    count = 0
    for node_name, tag_info in metadata['semantic_mappings'].items():
        if tag_info['hf_module_name'] and count < 10:
            print(f"   {tag_info['onnx_op_type']:12} -> {tag_info['semantic_tag']:50} (module: {tag_info['hf_module_name']})")
            count += 1
    
    return result


if __name__ == "__main__":
    test_bert_tiny()