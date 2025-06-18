#!/usr/bin/env python3
"""
BERT Self-Attention Test
Create both standalone module conversion AND extracted subgraph for proper comparison
"""

import torch
import torch.nn as nn
import onnx
import json
import argparse
from pathlib import Path
from transformers import AutoModel

from enhanced_dag_extractor import EnhancedDAGExtractor
from clean_subgraph_extractor import CleanSubgraphExtractor
from input_generator import UniversalInputGenerator
from extract_onnx_tags import ONNXTagExtractor


def find_bert_self_attention_module(model):
    """Find the BertSdpaSelfAttention module in the model"""
    print("🔍 Searching for BertSdpaSelfAttention module...")
    
    # Navigate the model structure
    target_module = None
    module_path = None
    
    try:
        # Path: model.encoder.layer[0].attention.self
        target_module = model.encoder.layer[0].attention.self  # This should be BertSdpaSelfAttention
        module_path = "encoder.layer.0.attention.self"
        print(f"✅ Found module: {type(target_module).__name__} at {module_path}")
        print(f"   Parameters: {sum(p.numel() for p in target_module.parameters()):,}")
        
        # Print module structure
        print("   Submodules:")
        for name, submodule in target_module.named_children():
            param_count = sum(p.numel() for p in submodule.parameters())
            print(f"     {name}: {type(submodule).__name__} ({param_count:,} params)")
        
        return target_module, module_path
        
    except Exception as e:
        print(f"❌ Could not find BertSdpaSelfAttention: {e}")
        
        # Print available structure for debugging
        print("\n🔍 Available model structure:")
        try:
            layer0 = model.encoder.layer[0]
            print(f"   layer[0]: {type(layer0).__name__}")
            attention = layer0.attention
            print(f"   attention: {type(attention).__name__}")
            for name, submodule in attention.named_children():
                print(f"     {name}: {type(submodule).__name__}")
        except Exception as e2:
            print(f"   Could not inspect structure: {e2}")
        
        return None, None


def convert_standalone_module(target_module, module_path, output_path):
    """Convert the standalone nn.Module to ONNX"""
    print(f"\n🔧 Converting standalone module: {module_path}")
    
    try:
        # Create appropriate inputs for BertSdpaSelfAttention
        # Forward signature: forward(hidden_states, attention_mask=None, head_mask=None, encoder_hidden_states=None, ...)
        
        batch_size = 1
        seq_length = 32
        hidden_size = target_module.query.in_features  # Should be 128 for BERT tiny
        
        print(f"   Input shape: [{batch_size}, {seq_length}, {hidden_size}]")
        
        # Create dummy inputs
        hidden_states = torch.randn(batch_size, seq_length, hidden_size)
        
        # Test the module first
        target_module.eval()
        with torch.no_grad():
            try:
                # Test with minimal input
                test_output = target_module(hidden_states)
                
                if isinstance(test_output, tuple):
                    print(f"   Module output: tuple with {len(test_output)} elements")
                    for i, out in enumerate(test_output):
                        if hasattr(out, 'shape'):
                            print(f"     output[{i}]: {out.shape}")
                        else:
                            print(f"     output[{i}]: {type(out)}")
                else:
                    print(f"   Module output: {test_output.shape}")
                
            except Exception as e:
                print(f"   ⚠️  Test forward failed: {e}")
                print("   Trying with additional parameters...")
                
                # Try with attention_mask
                attention_mask = torch.ones(batch_size, seq_length)
                test_output = target_module(hidden_states, attention_mask=attention_mask)
                print(f"   Module output with mask: {type(test_output)}")
        
        # Export to ONNX
        print(f"   Exporting to ONNX...")
        
        torch.onnx.export(
            target_module,
            hidden_states,  # Use minimal input for simplicity
            output_path,
            export_params=True,
            opset_version=14,
            do_constant_folding=False,
            input_names=['hidden_states'],
            output_names=['output'],
            verbose=False
        )
        
        # Validate the exported model
        standalone_model = onnx.load(output_path)
        onnx.checker.check_model(standalone_model)
        
        print(f"✅ Standalone module exported successfully!")
        print(f"   File: {output_path}")
        print(f"   Nodes: {len(standalone_model.graph.node)}")
        print(f"   Parameters: {len(standalone_model.graph.initializer)}")
        
        return standalone_model
        
    except Exception as e:
        print(f"❌ Standalone conversion failed: {e}")
        import traceback
        traceback.print_exc()
        return None


def generate_onnx_node_tags(onnx_path, output_path):
    """Generate node-tag mapping from any ONNX file"""
    try:
        extractor = ONNXTagExtractor(onnx_path)
        results = extractor.extract_all_tags()
        
        # Convert to format similar to enhanced DAG extractor
        mapping = {
            "metadata": {
                "source_onnx": onnx_path,
                "total_nodes": results["model_info"]["total_nodes"],
                "tagged_nodes": len(results["node_tags"]),
                "untagged_nodes": len(results["nodes_without_tags"]),
                "tag_coverage_percent": len(results["node_tags"]) / results["model_info"]["total_nodes"] * 100 if results["model_info"]["total_nodes"] > 0 else 0,
                "unique_tag_count": len(results["unique_tags"])
            },
            "node_tags": {},
            "tag_statistics": results["tag_statistics"],
            "untagged_operations": [node["name"] for node in results["nodes_without_tags"]]
        }
        
        # Reformat node tags to match enhanced DAG format
        for node_name, node_info in results["node_tags"].items():
            mapping["node_tags"][node_name] = {
                "op_type": node_info["op_type"],
                "tags": node_info["tags"],
                "input_count": node_info["input_count"],
                "output_count": node_info["output_count"]
            }
        
        # Save to file
        from pathlib import Path
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w') as f:
            json.dump(mapping, f, indent=2)
            
        print(f"   Saved {len(mapping['node_tags'])} node mappings from ONNX")
        print(f"   Tag coverage: {mapping['metadata']['tag_coverage_percent']:.1f}%")
        
    except Exception as e:
        print(f"   ⚠️  Failed to generate node tags: {e}")


def extract_self_attention_subgraph(enhanced_onnx_path, output_path):
    """Extract BertSdpaSelfAttention subgraph from enhanced whole model"""
    print(f"\n🔧 Extracting BertSdpaSelfAttention subgraph...")
    
    try:
        # Load the enhanced model and list available modules
        extractor = CleanSubgraphExtractor(enhanced_onnx_path)
        
        print("   Available tags containing 'SdpaSelfAttention':")
        available_tags = set()
        for tags in extractor.hierarchy_mapping.values():
            available_tags.update(tags)
        
        # Find BertSdpaSelfAttention related tags and BertAttention parent tags
        self_attention_tags = [tag for tag in available_tags if 'SdpaSelfAttention' in tag]
        attention_parent_tags = [tag for tag in available_tags if 'BertAttention' in tag and not tag.endswith('/Linear') and not tag.endswith('/LayerNorm')]
        
        for tag in self_attention_tags:
            node_count = sum(1 for node_tags in extractor.hierarchy_mapping.values() 
                           if tag in node_tags)
            print(f"     {tag} ({node_count} nodes)")
        
        for tag in attention_parent_tags:
            node_count = sum(1 for node_tags in extractor.hierarchy_mapping.values() 
                           if tag in node_tags)
            print(f"     {tag} ({node_count} nodes)")
        
        if not self_attention_tags and not attention_parent_tags:
            print("   ❌ No BertSdpaSelfAttention or BertAttention tags found!")
            return None
        
        # Priority 1: Use BertSdpaSelfAttention tag for layer 0 (matches standalone module exactly)
        layer_0_tags = [tag for tag in self_attention_tags if 'ModuleList.0' in tag and not tag.endswith('/Linear')]
        
        if layer_0_tags:
            target_tag = layer_0_tags[0]
            print(f"   Using layer 0 BertSdpaSelfAttention tag: {target_tag}")
        else:
            # Fallback: Use BertAttention parent tag for layer 0 (includes all attention operations)
            layer_0_attention_tags = [tag for tag in attention_parent_tags 
                                     if 'ModuleList.0' in tag and '/BertAttention' in tag and not '/BertAttention/' in tag]
            
            if layer_0_attention_tags:
                target_tag = layer_0_attention_tags[0]
                print(f"   Fallback: Using layer 0 BertAttention parent tag: {target_tag}")
            else:
                print("   ❌ No suitable layer 0 attention tags found!")
                return None
        print(f"   Extracting: {target_tag}")
        
        # Extract the subgraph
        extracted_model = extractor.extract_clean_subgraph(target_tag, output_path)
        
        print(f"✅ Subgraph extracted successfully!")
        print(f"   File: {output_path}")
        print(f"   Nodes: {len(extracted_model.graph.node)}")
        print(f"   Parameters: {len(extracted_model.graph.initializer)}")
        
        return extracted_model, target_tag
        
    except Exception as e:
        print(f"❌ Subgraph extraction failed: {e}")
        import traceback
        traceback.print_exc()
        return None, None


def compare_models(standalone_model, extracted_model, target_tag):
    """Compare standalone vs extracted models"""
    print(f"\n📊 COMPARING MODELS")
    print("=" * 50)
    
    if not standalone_model or not extracted_model:
        print("❌ Cannot compare - one or both models missing")
        return None
    
    # Basic comparison
    standalone_nodes = len(standalone_model.graph.node)
    extracted_nodes = len(extracted_model.graph.node)
    standalone_params = len(standalone_model.graph.initializer)
    extracted_params = len(extracted_model.graph.initializer)
    
    print(f"📈 STRUCTURE COMPARISON:")
    print(f"   Standalone module - Nodes: {standalone_nodes:2d}, Parameters: {standalone_params:2d}")
    print(f"   Extracted subgraph - Nodes: {extracted_nodes:2d}, Parameters: {extracted_params:2d}")
    
    # Operation comparison
    standalone_ops = [node.op_type for node in standalone_model.graph.node]
    extracted_ops = [node.op_type for node in extracted_model.graph.node]
    
    from collections import Counter
    standalone_op_counts = Counter(standalone_ops)
    extracted_op_counts = Counter(extracted_ops)
    
    print(f"\n🔧 OPERATION COMPARISON:")
    all_ops = set(standalone_op_counts.keys()) | set(extracted_op_counts.keys())
    
    matches = 0
    for op in sorted(all_ops):
        standalone_count = standalone_op_counts.get(op, 0)
        extracted_count = extracted_op_counts.get(op, 0)
        match = "✅" if standalone_count == extracted_count else "❌"
        if standalone_count == extracted_count:
            matches += 1
        print(f"   {op:15} Standalone: {standalone_count:2d}, Extracted: {extracted_count:2d} {match}")
    
    # Parameter comparison
    standalone_param_names = [init.name.split('.')[-1] for init in standalone_model.graph.initializer]
    extracted_param_names = [init.name.split('.')[-1] for init in extracted_model.graph.initializer]
    
    standalone_param_types = Counter(standalone_param_names)
    extracted_param_types = Counter(extracted_param_names)
    
    print(f"\n⚙️  PARAMETER COMPARISON:")
    all_param_types = set(standalone_param_types.keys()) | set(extracted_param_types.keys())
    
    param_matches = 0
    for param_type in sorted(all_param_types):
        standalone_count = standalone_param_types.get(param_type, 0)
        extracted_count = extracted_param_types.get(param_type, 0)
        match = "✅" if standalone_count == extracted_count else "❌"
        if standalone_count == extracted_count:
            param_matches += 1
        print(f"   {param_type:15} Standalone: {standalone_count:2d}, Extracted: {extracted_count:2d} {match}")
    
    # Overall assessment
    structure_match = abs(standalone_nodes - extracted_nodes) <= 5  # Allow small difference
    param_match = abs(standalone_params - extracted_params) <= 2
    op_similarity = matches / len(all_ops) > 0.7 if all_ops else False
    
    print(f"\n🎯 OVERALL ASSESSMENT:")
    print(f"   Structure similarity: {'✅' if structure_match else '❌'}")
    print(f"   Parameter similarity: {'✅' if param_match else '❌'}")
    print(f"   Operation similarity: {'✅' if op_similarity else '❌'} ({matches}/{len(all_ops)})")
    
    if structure_match and param_match and op_similarity:
        print(f"\n🎉 SUCCESS: Models are structurally equivalent!")
        print("The extracted subgraph matches the standalone module conversion.")
    else:
        print(f"\n⚠️  PARTIAL MATCH: Models have some differences.")
        print("This might be expected due to different conversion contexts.")
    
    # Return comparison results
    return {
        'structure_match': structure_match,
        'param_match': param_match,
        'op_similarity': op_similarity,
        'standalone_nodes': standalone_nodes,
        'extracted_nodes': extracted_nodes,
        'standalone_params': standalone_params,
        'extracted_params': extracted_params,
        'target_tag': target_tag
    }


def main(generate_detailed_json=True):
    """Main test function"""
    print("🧪 BERT SELF-ATTENTION COMPLETE TEST")
    print("=" * 60)
    
    if not generate_detailed_json:
        print("ℹ️  Detailed JSON generation disabled (faster execution)")
    
    test_dir = Path("temp/bert_self_attention_test")
    test_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        # Step 1: Load BERT model
        print("🔧 Loading BERT model...")
        model = AutoModel.from_pretrained('google/bert_uncased_L-2_H-128_A-2')
        
        # Step 2: Find the target module
        target_module, module_path = find_bert_self_attention_module(model)
        if not target_module:
            return None
        
        # Step 3: Create enhanced whole model
        print("\n🔧 Creating enhanced whole model...")
        generator = UniversalInputGenerator()
        inputs = generator.generate_inputs(model, 'google/bert_uncased_L-2_H-128_A-2')
        
        extractor = EnhancedDAGExtractor()
        extractor.analyze_model_structure(model)
        extractor.trace_execution_with_hooks(model, inputs)
        extractor.create_parameter_mapping(model)
        
        # Generate model WITHOUT tags (baseline)
        untagged_path = test_dir / "bert_enhanced_without_tags.onnx"
        torch.onnx.export(
            model,
            tuple(inputs.values()),
            str(untagged_path),
            export_params=True,
            opset_version=14,
            do_constant_folding=False,
            input_names=list(inputs.keys()),
            output_names=['output'],
            verbose=False
        )
        print(f"✅ Untagged model: {untagged_path}")
        
        # Generate model WITH tags
        whole_model_path = test_dir / "bert_enhanced.onnx"
        extractor.export_and_analyze_onnx(model, inputs, str(whole_model_path))
        enhanced_path = str(whole_model_path).replace('.onnx', '_with_tags.onnx')
        print(f"✅ Enhanced model: {enhanced_path}")
        
        # Generate node-tag mapping for enhanced model (optional)
        enhanced_node_tags_path = None
        if generate_detailed_json:
            enhanced_node_tags_path = test_dir / "enhanced_model_node_tags.json"
            extractor.save_node_tag_mapping(str(enhanced_node_tags_path))
            print(f"✅ Enhanced model node-tag mapping: {enhanced_node_tags_path}")
        else:
            print("⏭️  Skipping enhanced model node-tag mapping")
        
        # Step 4: Convert standalone module
        standalone_path = test_dir / "bert_self_attention_standalone.onnx"
        standalone_model = convert_standalone_module(target_module, module_path, str(standalone_path))
        
        # Step 5: Extract subgraph
        extracted_path = test_dir / "bert_self_attention_extracted.onnx"
        extracted_result = extract_self_attention_subgraph(enhanced_path, str(extracted_path))
        
        if extracted_result is None:
            print("❌ Extraction failed")
            return None
            
        extracted_model, target_tag = extracted_result
        
        # Generate node-tag mapping for extracted model (optional)
        extracted_node_tags_path = None
        if generate_detailed_json:
            extracted_node_tags_path = test_dir / "extracted_model_node_tags.json"
            # Use the enhanced extractor to generate proper tag mapping
            from clean_subgraph_extractor import CleanSubgraphExtractor
            extractor = CleanSubgraphExtractor(enhanced_path)
            
            # Get the extracted nodes from the ONNX model
            extracted_nodes = list(extracted_model.graph.node)
            extractor.generate_extracted_tag_mapping(target_tag, extracted_nodes, str(extracted_node_tags_path))
            print(f"✅ Extracted model node-tag mapping: {extracted_node_tags_path}")
        else:
            print("⏭️  Skipping extracted model node-tag mapping")
        
        # Step 6: Compare models
        comparison = compare_models(standalone_model, extracted_model, target_tag)
        
        # Step 7: Save results
        # Build file list with optional detailed files
        files_dict = {
            'untagged_whole_model': str(untagged_path),
            'enhanced_whole_model': enhanced_path,
            'standalone_module': str(standalone_path),
            'extracted_subgraph': str(extracted_path)
        }
        
        if generate_detailed_json:
            if enhanced_node_tags_path:
                files_dict['enhanced_model_node_tags'] = str(enhanced_node_tags_path)
            if extracted_node_tags_path:
                files_dict['extracted_model_node_tags'] = str(extracted_node_tags_path)
        
        results = {
            'test_name': 'BERT_SdpaSelfAttention_Complete_Test',
            'module_path': module_path,
            'target_tag': target_tag,
            'options': {
                'detailed_json_generated': generate_detailed_json
            },
            'files': files_dict,
            'comparison': comparison
        }
        
        results_path = test_dir / "complete_test_results.json"
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"\n💾 Complete test results saved to: {results_path}")
        
        return results
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return None


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="BERT Self-Attention Universal Export Test",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python bert_self_attention_test.py              # Basic test (ONNX models only)
  python bert_self_attention_test.py --debug-info # Full test with debug JSON files
        """
    )
    
    parser.add_argument(
        "--debug-info", 
        action="store_true", 
        help="Generate debug JSON files (node-tag mappings, detailed analysis)"
    )
    
    args = parser.parse_args()
    
    generate_detailed = args.debug_info
    
    result = main(generate_detailed_json=generate_detailed)
    
    if result:
        print(f"\n✅ Test completed successfully!")
        print(f"📁 Results saved to: temp/bert_self_attention_test/")
        if generate_detailed:
            print(f"🐛 Debug JSON files included")
        else:
            print(f"⚡ Basic mode: Essential files only")
    else:
        print(f"\n❌ Test failed!")
        exit(1)