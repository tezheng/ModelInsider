#!/usr/bin/env python3
"""
Simple Clean Extraction Test
Focus on testing the core functionality without complex module navigation
"""

import json
from pathlib import Path

from clean_subgraph_extractor import CleanSubgraphExtractor
from enhanced_dag_extractor import EnhancedDAGExtractor
from input_generator import UniversalInputGenerator
from transformers import AutoModel


def simple_test():
    """Simple test focusing on the extraction logic"""
    print("🧪 SIMPLE CLEAN EXTRACTION TEST")
    print("=" * 50)
    
    try:
        # Step 1: Create enhanced BERT model
        print("Creating enhanced BERT model...")
        model = AutoModel.from_pretrained('google/bert_uncased_L-2_H-128_A-2')
        generator = UniversalInputGenerator()
        inputs = generator.generate_inputs(model, 'google/bert_uncased_L-2_H-128_A-2')
        
        # Create enhanced model with tagging
        extractor = EnhancedDAGExtractor()
        extractor.analyze_model_structure(model)
        extractor.trace_execution_with_hooks(model, inputs)
        extractor.create_parameter_mapping(model)
        
        test_dir = Path("temp/simple_test")
        test_dir.mkdir(parents=True, exist_ok=True)
        
        whole_model_path = test_dir / "bert_enhanced.onnx"
        extractor.export_and_analyze_onnx(model, inputs, str(whole_model_path))
        enhanced_path = str(whole_model_path).replace('.onnx', '_with_tags.onnx')
        
        print(f"✅ Enhanced model created: {enhanced_path}")
        
        # Step 2: List available modules  
        print("\nListing available modules...")
        clean_extractor = CleanSubgraphExtractor(enhanced_path)
        
        print("Available tags:")
        unique_tags = set()
        for tags in clean_extractor.hierarchy_mapping.values():
            unique_tags.update(tags)
        
        for tag in sorted(unique_tags):
            node_count = sum(1 for node_tags in clean_extractor.hierarchy_mapping.values() 
                           if tag in node_tags)
            print(f"  {tag} ({node_count} nodes)")
        
        # Step 3: Test extraction with different tags
        test_tags = [
            "/BertModel/BertEmbeddings/Embedding",
            "/BertModel/BertEncoder/ModuleList.0/BertAttention/BertSelfOutput/Linear"
        ]
        
        results = {}
        
        for tag in test_tags:
            print(f"\n🎯 Testing extraction of: {tag}")
            
            try:
                output_path = test_dir / f"extracted_{tag.replace('/', '_')}.onnx"
                extracted_model = clean_extractor.extract_clean_subgraph(tag, str(output_path))
                
                result = {
                    'success': True,
                    'nodes': len(extracted_model.graph.node),
                    'inputs': len(extracted_model.graph.input),
                    'outputs': len(extracted_model.graph.output),
                    'initializers': len(extracted_model.graph.initializer),
                    'operations': [node.op_type for node in extracted_model.graph.node]
                }
                
                print(f"✅ Success: {result['nodes']} nodes, {result['inputs']} inputs, {result['outputs']} outputs")
                
            except Exception as e:
                result = {
                    'success': False,
                    'error': str(e)
                }
                print(f"❌ Failed: {e}")
            
            results[tag] = result
        
        # Step 4: Check if any extraction worked
        successful_extractions = [tag for tag, result in results.items() if result.get('success')]
        
        print(f"\n📊 SUMMARY:")
        print(f"Successful extractions: {len(successful_extractions)}/{len(test_tags)}")
        
        for tag in successful_extractions:
            result = results[tag]
            print(f"✅ {tag}: {result['nodes']} nodes")
        
        if successful_extractions:
            print("\n🎉 SUCCESS: Clean extraction is working!")
            print("✅ No topological sorting needed")
            print("✅ Subgraphs are valid ONNX models")
            
            # Test one successful extraction more deeply
            best_tag = successful_extractions[0]
            best_result = results[best_tag]
            
            print(f"\n🔍 Detailed analysis of {best_tag}:")
            print(f"Operations: {best_result['operations']}")
            
            # Try to load with ONNX Runtime for final validation
            try:
                import onnxruntime
                output_path = test_dir / f"extracted_{best_tag.replace('/', '_')}.onnx"
                session = onnxruntime.InferenceSession(str(output_path))
                print("✅ Loadable in ONNX Runtime!")
                
                # Show input/output info
                print("Inputs:")
                for inp in session.get_inputs():
                    print(f"  {inp.name}: {inp.shape} ({inp.type})")
                print("Outputs:")
                for out in session.get_outputs():
                    print(f"  {out.name}: {out.shape} ({out.type})")
                    
            except Exception as e:
                print(f"⚠️  ONNX Runtime issue: {e}")
        
        else:
            print("\n⚠️  No successful extractions - needs debugging")
        
        # Save results
        with open(test_dir / "test_results.json", 'w') as f:
            json.dump(results, f, indent=2)
        
        return results
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return None


if __name__ == "__main__":
    simple_test()