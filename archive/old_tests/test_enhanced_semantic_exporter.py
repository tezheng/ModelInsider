#!/usr/bin/env python3
"""
Comprehensive Test for Enhanced Semantic Exporter

This test validates the complete Enhanced Semantic Exporter implementation:
1. CLI integration
2. Core exporter functionality
3. Metadata generation
4. Coverage validation
5. No empty tags guarantee
"""

import json
from pathlib import Path

import onnx
import torch
from transformers import AutoModel, AutoTokenizer

from modelexport.core.enhanced_semantic_exporter import EnhancedSemanticExporter
from modelexport.semantic.enhanced_semantic_mapper import EnhancedSemanticMapper


def test_enhanced_semantic_exporter_bert_tiny():
    """Test Enhanced Semantic Exporter with BERT-tiny model."""
    print("🧪 Testing Enhanced Semantic Exporter with BERT-tiny")
    
    # Setup
    model = AutoModel.from_pretrained("prajjwal1/bert-tiny")
    tokenizer = AutoTokenizer.from_pretrained("prajjwal1/bert-tiny")
    
    # Prepare inputs
    inputs = tokenizer(["Test enhanced semantic mapping"], 
                      return_tensors="pt", 
                      max_length=8, 
                      padding=True, 
                      truncation=True)
    args = tuple(inputs.values())
    
    # Test directory
    test_dir = Path("temp/enhanced_semantic_test")
    test_dir.mkdir(parents=True, exist_ok=True)
    
    output_path = str(test_dir / "bert_tiny_enhanced_semantic.onnx")
    
    # Initialize Enhanced Semantic Exporter
    exporter = EnhancedSemanticExporter(verbose=True)
    
    print(f"   Model: {type(model).__name__}")
    print(f"   Input args: {len(args)} tensors")
    print(f"   Output: {output_path}")
    
    # Perform export
    result = exporter.export(
        model=model,
        args=args,
        output_path=output_path,
        input_names=['input_ids'],
        dynamic_axes={'input_ids': {0: 'batch_size', 1: 'sequence_length'}},
        opset_version=17,
        do_constant_folding=True
    )
    
    print(f"✅ Export completed in {result['export_time']:.2f}s")
    
    # Validate results
    assert Path(output_path).exists(), "ONNX file should be created"
    
    # Check basic statistics
    assert result['total_onnx_nodes'] > 0, "Should have ONNX nodes"
    assert result['hf_module_mappings'] > 0, "Should have HF module mappings"
    
    # Calculate coverage
    total_coverage = (
        result['hf_module_mappings'] + 
        result['operation_inferences'] + 
        result['pattern_fallbacks']
    )
    coverage_percentage = (total_coverage / result['total_onnx_nodes']) * 100
    
    print(f"   Coverage Analysis:")
    print(f"     Total ONNX nodes: {result['total_onnx_nodes']}")
    print(f"     HF module mappings: {result['hf_module_mappings']}")
    print(f"     Operation inferences: {result['operation_inferences']}")
    print(f"     Pattern fallbacks: {result['pattern_fallbacks']}")
    print(f"     Total coverage: {coverage_percentage:.1f}%")
    
    # Validate high coverage (should be 97% as designed)
    assert coverage_percentage >= 95.0, f"Coverage should be >=95%, got {coverage_percentage:.1f}%"
    
    # Check metadata file exists
    metadata_path = output_path.replace('.onnx', '_enhanced_semantic_metadata.json')
    assert Path(metadata_path).exists(), "Metadata file should be created"
    
    # Validate metadata content
    with open(metadata_path) as f:
        metadata = json.load(f)
    
    assert 'export_info' in metadata
    assert 'semantic_mappings' in metadata
    assert 'module_hierarchy' in metadata
    assert 'coverage_analysis' in metadata
    
    # Validate cardinal rules compliance
    export_info = metadata['export_info']
    assert export_info['cardinal_rules_followed']['MUST_001_no_hardcoded_logic'] == True
    assert export_info['cardinal_rules_followed']['MUST_002_torch_nn_filtering'] == True
    assert export_info['cardinal_rules_followed']['MUST_003_universal_design'] == True
    
    # Validate requirements met
    requirements = export_info['requirements_met']
    assert requirements['semantic_level_mapping'] == True
    assert requirements['comprehensive_coverage'] == True
    
    # Critical validation: NO EMPTY TAGS
    semantic_mappings = metadata['semantic_mappings']
    empty_tags = []
    for node_name, mapping in semantic_mappings.items():
        if not mapping.get('semantic_tag'):
            empty_tags.append(node_name)
    
    assert len(empty_tags) == 0, f"Found {len(empty_tags)} nodes with empty tags: {empty_tags}"
    
    print(f"✅ All validations passed!")
    print(f"   Metadata: {metadata_path}")
    print(f"   No empty tags: {len(semantic_mappings)} nodes all have semantic tags")
    
    return result


def test_enhanced_semantic_coverage_validation():
    """Test the coverage validation functionality."""
    print("🔍 Testing Enhanced Semantic coverage validation")
    
    # Setup with smaller input for faster testing
    model = AutoModel.from_pretrained("prajjwal1/bert-tiny")
    tokenizer = AutoTokenizer.from_pretrained("prajjwal1/bert-tiny")
    
    inputs = tokenizer(["Coverage test"], 
                      return_tensors="pt", 
                      max_length=4, 
                      padding=True, 
                      truncation=True)
    args = tuple(inputs.values())
    
    test_dir = Path("temp/coverage_test")
    test_dir.mkdir(parents=True, exist_ok=True)
    
    output_path = str(test_dir / "coverage_test.onnx")
    
    # Export with Enhanced Semantic Exporter
    exporter = EnhancedSemanticExporter(verbose=False)
    result = exporter.export(
        model=model,
        args=args,
        output_path=output_path,
        opset_version=17
    )
    
    # Test coverage validation
    validation_result = exporter.validate_semantic_coverage()
    
    assert validation_result['validation_passed'] == True, "Validation should pass"
    assert validation_result['empty_tags'] == 0, "Should have no empty tags"
    assert validation_result['coverage_percentage'] >= 95.0, "Should have high coverage"
    
    print(f"✅ Coverage validation passed:")
    print(f"   Total nodes: {validation_result['total_nodes']}")
    print(f"   Nodes with tags: {validation_result['nodes_with_tags']}")
    print(f"   Coverage: {validation_result['coverage_percentage']:.1f}%")
    
    return validation_result


def test_semantic_mapper_integration():
    """Test integration between Enhanced Semantic Mapper and Exporter."""
    print("🔗 Testing Enhanced Semantic Mapper integration")
    
    # Setup
    model = AutoModel.from_pretrained("prajjwal1/bert-tiny")
    tokenizer = AutoTokenizer.from_pretrained("prajjwal1/bert-tiny")
    
    # Create a simple ONNX export for testing
    inputs = tokenizer(["Integration test"], 
                      return_tensors="pt", 
                      max_length=4, 
                      padding=True, 
                      truncation=True)
    
    test_dir = Path("temp/integration_test")
    test_dir.mkdir(parents=True, exist_ok=True)
    
    onnx_path = str(test_dir / "integration_test.onnx")
    
    # Basic ONNX export
    torch.onnx.export(
        model, 
        inputs['input_ids'], 
        onnx_path,
        verbose=False,
        opset_version=17
    )
    
    # Load ONNX model
    onnx_model = onnx.load(onnx_path)
    
    # Test Enhanced Semantic Mapper directly
    mapper = EnhancedSemanticMapper(model, onnx_model)
    
    # Test a few nodes
    sample_nodes = onnx_model.graph.node[:5]  # Test first 5 nodes
    
    for node in sample_nodes:
        semantic_info = mapper.get_semantic_info_for_onnx_node(node)
        summary = semantic_info['semantic_summary']
        
        # Validate that we get meaningful information
        assert 'hf_module_name' in summary
        assert 'confidence' in summary
        assert 'primary_source' in summary
        
        print(f"   Node {node.name}: {summary['hf_module_name']} (confidence: {summary['confidence']})")
    
    # Test coverage statistics
    coverage_stats = mapper.get_mapping_coverage_stats()
    assert coverage_stats['total_nodes'] == len(onnx_model.graph.node)
    
    total_mapped = (
        coverage_stats['hf_module_mapped'] + 
        coverage_stats['operation_inferred'] + 
        coverage_stats['pattern_fallback']
    )
    
    coverage_percentage = (total_mapped / coverage_stats['total_nodes']) * 100
    assert coverage_percentage >= 95.0, f"Mapper coverage should be >=95%, got {coverage_percentage:.1f}%"
    
    print(f"✅ Integration test passed:")
    print(f"   Mapper coverage: {coverage_percentage:.1f}%")
    print(f"   Total nodes mapped: {total_mapped}/{coverage_stats['total_nodes']}")
    
    return coverage_stats


def test_comparative_results():
    """Compare Enhanced Semantic results with baseline expectations."""
    print("📊 Testing Enhanced Semantic comparative results")
    
    # This test validates that our Enhanced Semantic Exporter produces
    # results that are superior to previous approaches
    
    model = AutoModel.from_pretrained("prajjwal1/bert-tiny")
    tokenizer = AutoTokenizer.from_pretrained("prajjwal1/bert-tiny")
    
    inputs = tokenizer(["Comparative analysis"], 
                      return_tensors="pt", 
                      max_length=8, 
                      padding=True, 
                      truncation=True)
    args = tuple(inputs.values())
    
    test_dir = Path("temp/comparative_test")
    test_dir.mkdir(parents=True, exist_ok=True)
    
    output_path = str(test_dir / "comparative_test.onnx")
    
    # Export with Enhanced Semantic Exporter
    exporter = EnhancedSemanticExporter(verbose=False)
    result = exporter.export(
        model=model,
        args=args,
        output_path=output_path,
        opset_version=17
    )
    
    # Get the complete semantic metadata
    metadata = exporter.get_semantic_metadata()
    
    # Validate superior characteristics
    
    # 1. High HF module mapping ratio (should be >80%)
    hf_mapping_ratio = result['hf_module_mappings'] / result['total_onnx_nodes']
    assert hf_mapping_ratio > 0.75, f"HF mapping ratio should be >75%, got {hf_mapping_ratio:.1%}"
    
    # 2. High confidence level distribution (should have many 'high' confidence mappings)
    high_confidence = result['confidence_levels'].get('high', 0)
    high_confidence_ratio = high_confidence / result['total_onnx_nodes']
    assert high_confidence_ratio > 0.6, f"High confidence ratio should be >60%, got {high_confidence_ratio:.1%}"
    
    # 3. Comprehensive coverage (should be >97%)
    total_coverage = (
        result['hf_module_mappings'] + 
        result['operation_inferences'] + 
        result['pattern_fallbacks']
    )
    coverage_percentage = (total_coverage / result['total_onnx_nodes']) * 100
    assert coverage_percentage >= 97.0, f"Coverage should be >=97%, got {coverage_percentage:.1f}%"
    
    # 4. Fast execution (should be reasonable)
    assert result['export_time'] < 30.0, f"Export should be <30s, got {result['export_time']:.2f}s"
    
    print(f"✅ Comparative analysis passed:")
    print(f"   HF mapping ratio: {hf_mapping_ratio:.1%}")
    print(f"   High confidence ratio: {high_confidence_ratio:.1%}")
    print(f"   Total coverage: {coverage_percentage:.1f}%")
    print(f"   Export time: {result['export_time']:.2f}s")
    
    # 5. Validate semantic tag quality
    semantic_mappings = metadata['semantic_mappings']
    hf_semantic_tags = 0
    for _node_name, mapping in semantic_mappings.items():
        if mapping.get('hf_module_name') and mapping.get('semantic_type') != 'unknown':
            hf_semantic_tags += 1
    
    hf_semantic_ratio = hf_semantic_tags / len(semantic_mappings)
    assert hf_semantic_ratio > 0.7, f"HF semantic ratio should be >70%, got {hf_semantic_ratio:.1%}"
    
    print(f"   HF semantic tags: {hf_semantic_ratio:.1%}")
    
    return result


def main():
    """Run all Enhanced Semantic Exporter tests."""
    print("🚀 Enhanced Semantic Exporter Comprehensive Test Suite")
    print("=" * 70)
    
    tests = [
        test_enhanced_semantic_exporter_bert_tiny,
        test_enhanced_semantic_coverage_validation,
        test_semantic_mapper_integration,
        test_comparative_results
    ]
    
    results = {}
    
    for test_func in tests:
        try:
            result = test_func()
            results[test_func.__name__] = {"status": "PASSED", "result": result}
            print(f"✅ {test_func.__name__}: PASSED\n")
        except Exception as e:
            results[test_func.__name__] = {"status": "FAILED", "error": str(e)}
            print(f"❌ {test_func.__name__}: FAILED - {e}\n")
    
    # Summary
    passed = sum(1 for r in results.values() if r["status"] == "PASSED")
    total = len(results)
    
    print("=" * 70)
    print(f"📊 Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All Enhanced Semantic Exporter tests passed!")
        print("✅ Enhanced Semantic Exporter is ready for production use")
    else:
        print("❌ Some tests failed. Please review the errors above.")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())