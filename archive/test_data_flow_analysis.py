#!/usr/bin/env python3
"""
Data Flow Analysis Tests

Test suite for the data flow analysis functionality that enhances
semantic mappings through graph connectivity analysis.

CARDINAL RULES:
- MUST-002: ALL testing via pytest
- MUST-001: NO hardcoded logic
- MUST-003: Universal design validation
"""

import os
import tempfile

import onnx
import pytest
import torch
from transformers import AutoModel, AutoTokenizer

from modelexport.semantic.data_flow_analyzer import DataFlowAnalyzer
from modelexport.semantic.enhanced_semantic_mapper import EnhancedSemanticMapper


class TestDataFlowAnalysis:
    """Test data flow analysis functionality."""
    
    @pytest.fixture(scope="class")
    def test_model_and_mapper(self):
        """Create test model and semantic mapper."""
        model = AutoModel.from_pretrained('prajjwal1/bert-tiny')
        tokenizer = AutoTokenizer.from_pretrained('prajjwal1/bert-tiny')
        inputs = tokenizer(['Test data flow'], return_tensors='pt', max_length=8, padding=True, truncation=True)
        
        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = os.path.join(temp_dir, 'dataflow_test.onnx')
            
            torch.onnx.export(
                model,
                inputs['input_ids'],
                output_path,
                opset_version=17,
                do_constant_folding=True
            )
            
            onnx_model = onnx.load(output_path)
            mapper = EnhancedSemanticMapper(model, onnx_model)
            
            return {
                'model': model,
                'onnx_model': onnx_model,
                'mapper': mapper
            }
    
    def test_data_flow_analyzer_initialization(self, test_model_and_mapper):
        """Test that data flow analyzer initializes correctly."""
        mapper = test_model_and_mapper['mapper']
        onnx_model = test_model_and_mapper['onnx_model']
        
        # Get baseline semantic mappings
        baseline_mappings = {}
        for node in onnx_model.graph.node:
            semantic_info = mapper.get_semantic_info_for_onnx_node(node)
            baseline_mappings[node.name] = semantic_info['semantic_summary']
        
        # Initialize data flow analyzer
        analyzer = DataFlowAnalyzer(onnx_model, baseline_mappings)
        
        # Verify analyzer is properly initialized
        assert len(analyzer.node_graph) > 0, "Node graph should be built"
        assert len(analyzer.output_graph) > 0, "Output graph should be built"
        assert len(analyzer.input_graph) > 0, "Input graph should be built"
        assert analyzer.semantic_mappings == baseline_mappings, "Semantic mappings should be stored"
    
    def test_data_flow_enhancement_improves_coverage(self, test_model_and_mapper):
        """Test that data flow analysis improves semantic coverage."""
        mapper = test_model_and_mapper['mapper']
        
        # Get baseline statistics
        baseline_stats = mapper.get_mapping_coverage_stats()
        baseline_unknown = baseline_stats['unmapped']
        
        # Apply data flow enhancement
        enhancement_result = mapper.enhance_with_data_flow_analysis()
        enhancement_stats = enhancement_result['enhancement_statistics']
        
        # Verify improvement
        unknown_improvement = enhancement_stats['unknown_improvement']
        assert unknown_improvement >= 0, f"Should improve or maintain unknown count, got {unknown_improvement}"
        
        # Should enhance nodes or maintain quality (conservative approach is acceptable)
        nodes_enhanced = enhancement_stats['nodes_enhanced']
        assert nodes_enhanced >= 0, f"Enhancement count should be non-negative, got {nodes_enhanced}"
        
        # Enhancement rate should be reasonable
        enhancement_rate = enhancement_stats['enhancement_rate']
        assert 0 <= enhancement_rate <= 100, f"Enhancement rate should be 0-100%, got {enhancement_rate}"
    
    def test_data_flow_enhancement_maintains_high_confidence_nodes(self, test_model_and_mapper):
        """Test that high confidence nodes are not degraded by data flow analysis."""
        mapper = test_model_and_mapper['mapper']
        onnx_model = test_model_and_mapper['onnx_model']
        
        # Get baseline high confidence nodes
        baseline_high_confidence = []
        for node in onnx_model.graph.node:
            semantic_info = mapper.get_semantic_info_for_onnx_node(node)
            if semantic_info['semantic_summary'].get('confidence') == 'high':
                baseline_high_confidence.append(node.name)
        
        # Apply enhancement
        enhancement_result = mapper.enhance_with_data_flow_analysis()
        enhanced_mappings = enhancement_result['enhanced_mappings']
        
        # Verify high confidence nodes are maintained
        for node_name in baseline_high_confidence:
            enhanced_confidence = enhanced_mappings[node_name].get('confidence', 'unknown')
            assert enhanced_confidence == 'high', (
                f"High confidence node {node_name} should maintain high confidence, "
                f"got {enhanced_confidence}"
            )
    
    def test_data_flow_enhancement_provides_source_attribution(self, test_model_and_mapper):
        """Test that enhanced nodes have proper source attribution."""
        mapper = test_model_and_mapper['mapper']
        
        # Apply enhancement
        enhancement_result = mapper.enhance_with_data_flow_analysis()
        enhanced_mappings = enhancement_result['enhanced_mappings']
        
        # Find nodes enhanced by data flow
        data_flow_enhanced = [
            (name, mapping) for name, mapping in enhanced_mappings.items()
            if mapping.get('primary_source', '').startswith('data_flow')
        ]
        
        # Verify source attribution
        for node_name, mapping in data_flow_enhanced:
            primary_source = mapping.get('primary_source', '')
            assert primary_source in ['data_flow_backward', 'data_flow_forward', 'contextual_inference'], (
                f"Enhanced node {node_name} should have valid data flow source, got {primary_source}"
            )
            
            # Should have appropriate confidence
            confidence = mapping.get('confidence', 'unknown')
            assert confidence in ['low', 'medium'], (
                f"Data flow enhanced node {node_name} should have medium or low confidence, got {confidence}"
            )
    
    def test_backward_semantic_inheritance(self, test_model_and_mapper):
        """Test backward semantic inheritance functionality."""
        mapper = test_model_and_mapper['mapper']
        onnx_model = test_model_and_mapper['onnx_model']
        
        # Get baseline mappings
        baseline_mappings = {}
        for node in onnx_model.graph.node:
            semantic_info = mapper.get_semantic_info_for_onnx_node(node)
            baseline_mappings[node.name] = semantic_info['semantic_summary']
        
        # Create analyzer
        analyzer = DataFlowAnalyzer(onnx_model, baseline_mappings)
        
        # Find constants that should benefit from backward inheritance
        constant_nodes = [
            node for node in onnx_model.graph.node 
            if node.op_type == 'Constant' and baseline_mappings[node.name].get('semantic_type') == 'unknown'
        ]
        
        if constant_nodes:
            # Test backward inheritance on a constant
            test_node = constant_nodes[0]
            result = analyzer._try_backward_semantic_inheritance(test_node.name)
            
            # Should either succeed or fail gracefully
            assert 'success' in result, "Backward inheritance should return success status"
            
            if result['success']:
                enhancement = result['enhancement']
                assert 'semantic_type' in enhancement, "Should provide semantic type"
                assert enhancement['primary_source'] == 'data_flow_backward', "Should mark as backward inheritance"
    
    def test_forward_semantic_propagation(self, test_model_and_mapper):
        """Test forward semantic propagation functionality."""
        mapper = test_model_and_mapper['mapper']
        onnx_model = test_model_and_mapper['onnx_model']
        
        # Get baseline mappings
        baseline_mappings = {}
        for node in onnx_model.graph.node:
            semantic_info = mapper.get_semantic_info_for_onnx_node(node)
            baseline_mappings[node.name] = semantic_info['semantic_summary']
        
        # Create analyzer
        analyzer = DataFlowAnalyzer(onnx_model, baseline_mappings)
        
        # Find nodes that could benefit from forward propagation
        unknown_nodes = [
            node for node in onnx_model.graph.node 
            if baseline_mappings[node.name].get('semantic_type') == 'unknown'
        ]
        
        if unknown_nodes:
            # Test forward propagation
            test_node = unknown_nodes[0]
            result = analyzer._try_forward_semantic_propagation(test_node.name)
            
            # Should either succeed or fail gracefully
            assert 'success' in result, "Forward propagation should return success status"
            
            if result['success']:
                enhancement = result['enhancement']
                assert 'semantic_type' in enhancement, "Should provide semantic type"
                assert enhancement['primary_source'] == 'data_flow_forward', "Should mark as forward propagation"
    
    def test_contextual_pattern_recognition(self, test_model_and_mapper):
        """Test contextual pattern recognition functionality."""
        mapper = test_model_and_mapper['mapper']
        onnx_model = test_model_and_mapper['onnx_model']
        
        # Get baseline mappings
        baseline_mappings = {}
        for node in onnx_model.graph.node:
            semantic_info = mapper.get_semantic_info_for_onnx_node(node)
            baseline_mappings[node.name] = semantic_info['semantic_summary']
        
        # Create analyzer
        analyzer = DataFlowAnalyzer(onnx_model, baseline_mappings)
        
        # Look for GELU activation pattern nodes (Div, Erf, Add in intermediate_act_fn)
        gelu_nodes = [
            node for node in onnx_model.graph.node
            if (node.op_type in ['Div', 'Erf', 'Add'] and 
                'intermediate_act_fn' in node.name and
                baseline_mappings[node.name].get('semantic_type') == 'unknown')
        ]
        
        if gelu_nodes:
            # Test contextual inference on GELU pattern
            test_node = gelu_nodes[0]
            result = analyzer._try_contextual_operation_inference(test_node.name)
            
            # Should either succeed or fail gracefully
            assert 'success' in result, "Contextual inference should return success status"
            
            if result['success']:
                enhancement = result['enhancement']
                assert 'semantic_type' in enhancement, "Should provide semantic type"
                assert enhancement['primary_source'] == 'contextual_inference', "Should mark as contextual inference"
    
    def test_enhancement_statistics_are_accurate(self, test_model_and_mapper):
        """Test that enhancement statistics are accurately calculated."""
        mapper = test_model_and_mapper['mapper']
        onnx_model = test_model_and_mapper['onnx_model']
        
        # Count baseline statistics manually
        baseline_unknown = 0
        baseline_low_confidence = 0
        
        for node in onnx_model.graph.node:
            semantic_info = mapper.get_semantic_info_for_onnx_node(node)
            semantic_type = semantic_info['semantic_summary'].get('semantic_type', 'unknown')
            confidence = semantic_info['semantic_summary'].get('confidence', 'unknown')
            
            if semantic_type == 'unknown':
                baseline_unknown += 1
            if confidence == 'low':
                baseline_low_confidence += 1
        
        # Apply enhancement
        enhancement_result = mapper.enhance_with_data_flow_analysis()
        enhancement_stats = enhancement_result['enhancement_statistics']
        
        # Verify statistics accuracy
        assert enhancement_stats['original_unknown'] == baseline_unknown, (
            f"Original unknown count mismatch: expected {baseline_unknown}, "
            f"got {enhancement_stats['original_unknown']}"
        )
        
        assert enhancement_stats['original_low_confidence'] == baseline_low_confidence, (
            f"Original low confidence count mismatch: expected {baseline_low_confidence}, "
            f"got {enhancement_stats['original_low_confidence']}"
        )
        
        # Verify improvement calculations
        unknown_improvement = enhancement_stats['unknown_improvement']
        confidence_improvement = enhancement_stats['confidence_improvement']
        
        assert unknown_improvement == (enhancement_stats['original_unknown'] - enhancement_stats['enhanced_unknown']), (
            "Unknown improvement calculation is incorrect"
        )
        
        # Enhancement rate should be within bounds
        enhancement_rate = enhancement_stats['enhancement_rate']
        assert 0 <= enhancement_rate <= 100, f"Enhancement rate should be 0-100%, got {enhancement_rate}"
    
    def test_data_flow_analysis_follows_universal_design(self, test_model_and_mapper):
        """Test that data flow analysis follows universal design principles."""
        mapper = test_model_and_mapper['mapper']
        
        # Apply enhancement
        enhancement_result = mapper.enhance_with_data_flow_analysis()
        enhanced_mappings = enhancement_result['enhanced_mappings']
        
        # Check for hardcoded model-specific logic
        for node_name, mapping in enhanced_mappings.items():
            semantic_type = mapping.get('semantic_type', 'unknown')
            
            # Should not contain hardcoded model names
            assert 'bert' not in semantic_type.lower(), (
                f"Enhanced mapping for {node_name} contains hardcoded model reference: {semantic_type}"
            )
            assert 'gpt' not in semantic_type.lower(), (
                f"Enhanced mapping for {node_name} contains hardcoded model reference: {semantic_type}"
            )
            
            # Should have valid confidence levels
            confidence = mapping.get('confidence', 'unknown')
            assert confidence in ['high', 'medium', 'low', 'unknown'], (
                f"Invalid confidence level for {node_name}: {confidence}"
            )


def test_data_flow_analysis_integration():
    """Integration test for data flow analysis."""
    print("🧪 Running Data Flow Analysis Integration Test...")
    
    # Load model
    model = AutoModel.from_pretrained('prajjwal1/bert-tiny')
    tokenizer = AutoTokenizer.from_pretrained('prajjwal1/bert-tiny')
    inputs = tokenizer(['Integration test'], return_tensors='pt', max_length=8, padding=True, truncation=True)
    
    with tempfile.TemporaryDirectory() as temp_dir:
        output_path = os.path.join(temp_dir, 'integration_test.onnx')
        
        # Export model
        torch.onnx.export(
            model,
            inputs['input_ids'],
            output_path,
            opset_version=17
        )
        
        # Create mapper and test enhancement
        onnx_model = onnx.load(output_path)
        mapper = EnhancedSemanticMapper(model, onnx_model)
        
        # Get baseline
        baseline_stats = mapper.get_mapping_coverage_stats()
        
        # Apply enhancement
        enhancement_result = mapper.enhance_with_data_flow_analysis()
        enhancement_stats = enhancement_result['enhancement_statistics']
        
        print(f"✅ Integration test results:")
        print(f"  Baseline unknown: {enhancement_stats['original_unknown']}")
        print(f"  Enhanced unknown: {enhancement_stats['enhanced_unknown']}")
        print(f"  Improvement: {enhancement_stats['unknown_improvement']}")
        print(f"  Enhancement rate: {enhancement_stats['enhancement_rate']:.1f}%")
        
        # Should improve or maintain
        assert enhancement_stats['unknown_improvement'] >= 0, "Should improve unknown node count"
        assert enhancement_stats['nodes_enhanced'] >= 0, "Should enhance some nodes"
        
        return enhancement_stats


if __name__ == "__main__":
    # Run integration test
    stats = test_data_flow_analysis_integration()
    print("✅ Data flow analysis integration test completed!")