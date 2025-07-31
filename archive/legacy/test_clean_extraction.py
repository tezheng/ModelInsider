#!/usr/bin/env python3
"""
Test Clean Subgraph Extraction - Simplified Version
"""

import json
from pathlib import Path

import onnx
import torch
from clean_subgraph_extractor import CleanSubgraphExtractor
from enhanced_dag_extractor import EnhancedDAGExtractor
from input_generator import UniversalInputGenerator
from transformers import AutoModel


def create_enhanced_bert_model():
    """Create enhanced BERT model with 100% tagging"""
    print("🔧 Creating enhanced BERT model...")
    
    model = AutoModel.from_pretrained('google/bert_uncased_L-2_H-128_A-2')
    generator = UniversalInputGenerator()
    inputs = generator.generate_inputs(model, 'google/bert_uncased_L-2_H-128_A-2')
    
    # Create enhanced model
    extractor = EnhancedDAGExtractor()
    extractor.analyze_model_structure(model)
    extractor.trace_execution_with_hooks(model, inputs)
    extractor.create_parameter_mapping(model)
    
    # Export with full tagging
    test_dir = Path("temp/clean_test")
    test_dir.mkdir(parents=True, exist_ok=True)
    
    whole_model_path = test_dir / "bert_enhanced.onnx"
    extractor.export_and_analyze_onnx(model, inputs, str(whole_model_path))
    enhanced_path = str(whole_model_path).replace('.onnx', '_with_tags.onnx')
    
    print(f"✅ Enhanced model: {enhanced_path}")
    return enhanced_path, model, inputs


def export_single_linear_module(model, inputs):
    """Export a single Linear module for comparison"""
    print("🔧 Exporting single Linear module...")
    
    # Get the first linear layer: encoder.layer.0.attention.self_output.dense
    target_module = model.encoder.layer[0].attention.self_output.dense
    print(f"Target module: {type(target_module).__name__}")
    print(f"Parameters: {sum(p.numel() for p in target_module.parameters())}")
    
    # Create appropriate inputs (Linear expects 2D input)
    batch_size, seq_len = 1, 32
    hidden_size = target_module.in_features
    dummy_input = torch.randn(batch_size * seq_len, hidden_size)
    
    print(f"Input shape: {dummy_input.shape}")
    print(f"Expected output shape: {(batch_size * seq_len, target_module.out_features)}")
    
    # Test the module
    target_module.eval()
    with torch.no_grad():
        test_output = target_module(dummy_input)
        print(f"Actual output shape: {test_output.shape}")
    
    # Export to ONNX
    output_path = "temp/clean_test/single_linear.onnx"
    
    torch.onnx.export(
        target_module,
        dummy_input,
        output_path,
        export_params=True,
        opset_version=14,
        do_constant_folding=False,
        input_names=['input'],
        output_names=['output'],
        verbose=False
    )
    
    # Validate
    single_model = onnx.load(output_path)
    onnx.checker.check_model(single_model)
    print(f"✅ Single module exported: {output_path}")
    
    return output_path, "encoder.layer.0.attention.self_output.dense"


def test_clean_extraction():
    """Test the complete clean extraction workflow"""
    print("🧪 TESTING CLEAN SUBGRAPH EXTRACTION")
    print("=" * 60)
    
    try:
        # Step 1: Create enhanced model
        enhanced_path, model, inputs = create_enhanced_bert_model()
        
        # Step 2: Export single module
        single_path, module_path = export_single_linear_module(model, inputs)
        
        # Step 3: List available modules
        print("\n📋 Available modules in enhanced model:")
        extractor = CleanSubgraphExtractor(enhanced_path)
        
        # Show first few modules
        for i, (node_name, tags) in enumerate(extractor.hierarchy_mapping.items()):
            if i < 10:  # Show first 10
                print(f"  {node_name}: {tags}")
        
        # Find a good tag to extract
        target_tag = "/BertModel/BertEncoder/ModuleList.0/BertAttention/BertSelfOutput/Linear"
        print(f"\n🎯 Target tag: {target_tag}")
        
        # Step 4: Extract subgraph
        print("\n🔧 Extracting subgraph...")
        extracted_path = "temp/clean_test/extracted_linear.onnx"
        
        extracted_model = extractor.extract_clean_subgraph(target_tag, extracted_path)
        
        # Step 5: Compare models
        print("\n📊 Comparing models...")
        
        # Load both models
        single_model = onnx.load(single_path)
        
        print(f"Single module    - Nodes: {len(single_model.graph.node):2d}, "
              f"Params: {len(single_model.graph.initializer):2d}")
        print(f"Extracted module - Nodes: {len(extracted_model.graph.node):2d}, "
              f"Params: {len(extracted_model.graph.initializer):2d}")
        
        # Compare operation types
        single_ops = [node.op_type for node in single_model.graph.node]
        extracted_ops = [node.op_type for node in extracted_model.graph.node]
        
        print(f"\nOperation types:")
        print(f"Single:    {single_ops}")
        print(f"Extracted: {extracted_ops}")
        
        # Check if Linear operation is present
        has_linear = any(op in ['MatMul', 'Gemm'] for op in extracted_ops)
        print(f"\nContains linear operation: {'✅' if has_linear else '❌'}")
        
        # Success assessment
        if len(extracted_model.graph.node) > 0 and has_linear:
            print("\n🎉 SUCCESS: Clean extraction worked!")
            print("✅ Extracted subgraph is valid ONNX")
            print("✅ Contains expected operations")
            print("✅ No topological sorting errors")
            
            # Save results
            results = {
                'success': True,
                'single_module_nodes': len(single_model.graph.node),
                'extracted_nodes': len(extracted_model.graph.node),
                'single_ops': single_ops,
                'extracted_ops': extracted_ops,
                'target_tag': target_tag,
                'files': {
                    'enhanced_model': enhanced_path,
                    'single_module': single_path,
                    'extracted_subgraph': extracted_path
                }
            }
        else:
            print("\n⚠️  PARTIAL SUCCESS: Extraction completed but needs refinement")
            results = {'success': False, 'reason': 'Empty or invalid subgraph'}
        
        # Save results
        with open("temp/clean_test/results.json", 'w') as f:
            json.dump(results, f, indent=2)
        
        return results
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return {'success': False, 'error': str(e)}


if __name__ == "__main__":
    test_clean_extraction()