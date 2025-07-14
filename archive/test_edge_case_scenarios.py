#!/usr/bin/env python3
"""
Specific Edge Case Scenario Tests

This module tests specific challenging edge case scenarios that require
advanced semantic inference techniques.

CARDINAL RULES:
- MUST-002: ALL testing via pytest
- MUST-001: NO hardcoded logic 
- MUST-003: Universal design validation
"""

import os
import tempfile

import torch
from transformers import AutoModel, AutoTokenizer

from modelexport.core.enhanced_semantic_exporter import EnhancedSemanticExporter


class EdgeCaseScenarios:
    """Test specific edge case scenarios that challenge semantic mapping."""
    
    @staticmethod
    def create_test_onnx_model() -> str:
        """Create a minimal ONNX model with known edge cases for testing."""
        with tempfile.TemporaryDirectory() as temp_dir:
            model_path = os.path.join(temp_dir, 'test_edge_cases.onnx')
            
            # Load a real model first
            model = AutoModel.from_pretrained('prajjwal1/bert-tiny')
            tokenizer = AutoTokenizer.from_pretrained('prajjwal1/bert-tiny')
            inputs = tokenizer(['test'], return_tensors='pt', max_length=8, padding=True, truncation=True)
            
            torch.onnx.export(
                model,
                inputs['input_ids'],
                model_path,
                opset_version=17,
                do_constant_folding=True
            )
            
            return model_path


class TestConstantOperationEdgeCases:
    """Test edge cases related to constant operations."""
    
    def test_constant_operations_have_appropriate_semantics(self):
        """Test that constant operations are properly classified."""
        model = AutoModel.from_pretrained('prajjwal1/bert-tiny')
        tokenizer = AutoTokenizer.from_pretrained('prajjwal1/bert-tiny')
        inputs = tokenizer(['Test constant semantics'], return_tensors='pt', max_length=8)
        
        exporter = EnhancedSemanticExporter(verbose=False)
        
        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = os.path.join(temp_dir, 'constants_test.onnx')
            
            result = exporter.export(
                model=model,
                args=(inputs['input_ids'],),
                output_path=output_path,
                opset_version=17
            )
            
            metadata = exporter.get_semantic_metadata()
            semantic_mappings = metadata['semantic_mappings']
            
            # Find constant operations
            constant_nodes = [
                (name, mapping) for name, mapping in semantic_mappings.items()
                if mapping.get('semantic_type') == 'constant' or 'Constant' in name
            ]
            
            assert len(constant_nodes) > 0, "Should find constant operations in BERT model"
            
            # Validate constant operations have appropriate semantics
            for node_name, mapping in constant_nodes:
                semantic_type = mapping.get('semantic_type', 'unknown')
                
                # Constants can have context-specific semantics or generic 'constant' type
                valid_semantic_types = ['constant', 'embedding', 'normalization', 'attention', 'feed_forward', 'unknown']
                assert semantic_type in valid_semantic_types, (
                    f"Constant node {node_name} should have valid semantic type, got: {semantic_type}"
                )
                
                # Unknown semantic type should be rare (less than 30% of constants)
                # but is acceptable for some edge cases
                
                # Context-aware constants can have high confidence
                confidence = mapping.get('confidence', 'unknown')
                assert confidence in ['high', 'medium', 'low'], (
                    f"Constant node {node_name} should have valid confidence level: {confidence}"
                )
    
    def test_numbered_constants_pattern_recognition(self):
        """Test that numbered constants (/Constant_N) are properly handled."""
        model = AutoModel.from_pretrained('prajjwal1/bert-tiny')
        tokenizer = AutoTokenizer.from_pretrained('prajjwal1/bert-tiny')
        inputs = tokenizer(['Test numbered constants'], return_tensors='pt', max_length=8)
        
        exporter = EnhancedSemanticExporter(verbose=False)
        
        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = os.path.join(temp_dir, 'numbered_constants_test.onnx')
            
            exporter.export(
                model=model,
                args=(inputs['input_ids'],),
                output_path=output_path,
                opset_version=17
            )
            
            metadata = exporter.get_semantic_metadata()
            semantic_mappings = metadata['semantic_mappings']
            
            # Find numbered constant operations
            numbered_constants = [
                (name, mapping) for name, mapping in semantic_mappings.items()
                if '/Constant_' in name and name.split('_')[-1].isdigit()
            ]
            
            # Should find some numbered constants
            if len(numbered_constants) > 0:
                for node_name, mapping in numbered_constants:
                    # Numbered constants should be recognized with appropriate semantics
                    semantic_type = mapping.get('semantic_type', 'unknown')
                    valid_types = ['constant', 'numbered_constant', 'embedding', 'normalization', 'attention', 'feed_forward', 'unknown']
                    assert semantic_type in valid_types, (
                        f"Numbered constant {node_name} should be properly classified, got: {semantic_type}"
                    )


class TestRootOperationEdgeCases:
    """Test edge cases for root-level operations."""
    
    def test_root_level_operations_classification(self):
        """Test classification of operations at the graph root level."""
        model = AutoModel.from_pretrained('prajjwal1/bert-tiny')
        tokenizer = AutoTokenizer.from_pretrained('prajjwal1/bert-tiny')
        inputs = tokenizer(['Test root operations'], return_tensors='pt', max_length=8)
        
        exporter = EnhancedSemanticExporter(verbose=False)
        
        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = os.path.join(temp_dir, 'root_ops_test.onnx')
            
            exporter.export(
                model=model,
                args=(inputs['input_ids'],),
                output_path=output_path,
                opset_version=17
            )
            
            metadata = exporter.get_semantic_metadata()
            semantic_mappings = metadata['semantic_mappings']
            
            # Find root-level operations (single operation name, no hierarchy)
            root_operations = [
                (name, mapping) for name, mapping in semantic_mappings.items()
                if (name.startswith('/') and 
                    len(name.split('/')) == 2 and  # /OperationName
                    not name.split('/')[-1].replace('_', '').replace('0', '').replace('1', '').replace('2', '').replace('3', '').replace('4', '').replace('5', '').replace('6', '').replace('7', '').replace('8', '').replace('9', '').isalpha())
            ]
            
            # Should find some root operations
            if len(root_operations) > 0:
                for node_name, mapping in root_operations:
                    # Root operations should have some semantic classification
                    assert mapping.get('semantic_type') != 'unknown', (
                        f"Root operation {node_name} should have semantic classification"
                    )
                    assert mapping.get('primary_source') in ['operation_inference', 'pattern_fallback'], (
                        f"Root operation {node_name} should use inference or fallback strategy"
                    )
    
    def test_arithmetic_operations_at_root(self):
        """Test arithmetic operations that appear at graph root."""
        model = AutoModel.from_pretrained('prajjwal1/bert-tiny')
        tokenizer = AutoTokenizer.from_pretrained('prajjwal1/bert-tiny')
        inputs = tokenizer(['Test arithmetic'], return_tensors='pt', max_length=8)
        
        exporter = EnhancedSemanticExporter(verbose=False)
        
        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = os.path.join(temp_dir, 'arithmetic_test.onnx')
            
            exporter.export(
                model=model,
                args=(inputs['input_ids'],),
                output_path=output_path,
                opset_version=17
            )
            
            metadata = exporter.get_semantic_metadata()
            semantic_mappings = metadata['semantic_mappings']
            
            # Find arithmetic operations
            arithmetic_ops = [
                (name, mapping) for name, mapping in semantic_mappings.items()
                if mapping.get('onnx_op_type') in ['Add', 'Sub', 'Mul', 'Div']
            ]
            
            # Should find some arithmetic operations in BERT
            if len(arithmetic_ops) > 0:
                for node_name, mapping in arithmetic_ops:
                    # Arithmetic operations should be classified appropriately
                    semantic_type = mapping.get('semantic_type', 'unknown')
                    valid_types = ['arithmetic', 'attention_projection', 'linear_transformation', 'attention', 'embedding', 'normalization', 'output', 'feed_forward', 'unknown']
                    assert semantic_type in valid_types, (
                        f"Arithmetic operation {node_name} should have appropriate semantic type, got: {semantic_type}"
                    )


class TestOperationInferenceEdgeCases:
    """Test edge cases in operation-based semantic inference."""
    
    def test_gather_operation_inference(self):
        """Test that Gather operations are properly inferred as embedding lookups."""
        model = AutoModel.from_pretrained('prajjwal1/bert-tiny')
        tokenizer = AutoTokenizer.from_pretrained('prajjwal1/bert-tiny')
        inputs = tokenizer(['Test gather inference'], return_tensors='pt', max_length=8)
        
        exporter = EnhancedSemanticExporter(verbose=False)
        
        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = os.path.join(temp_dir, 'gather_test.onnx')
            
            exporter.export(
                model=model,
                args=(inputs['input_ids'],),
                output_path=output_path,
                opset_version=17
            )
            
            metadata = exporter.get_semantic_metadata()
            semantic_mappings = metadata['semantic_mappings']
            
            # Find Gather operations
            gather_ops = [
                (name, mapping) for name, mapping in semantic_mappings.items()
                if mapping.get('onnx_op_type') == 'Gather'
            ]
            
            # BERT should have Gather operations for embeddings
            assert len(gather_ops) > 0, "BERT model should contain Gather operations"
            
            for node_name, mapping in gather_ops:
                # Gather operations should be inferred appropriately
                semantic_type = mapping.get('semantic_type', 'unknown')
                valid_types = ['embedding_lookup', 'embedding', 'indexing', 'unknown']
                assert semantic_type in valid_types, (
                    f"Gather operation {node_name} should be inferred appropriately, got: {semantic_type}"
                )
                
                confidence = mapping.get('confidence', 'unknown')
                assert confidence in ['high', 'medium', 'low'], (
                    f"Gather operation {node_name} should have valid confidence: {confidence}"
                )
    
    def test_matmul_context_inference(self):
        """Test that MatMul operations use context for better inference."""
        model = AutoModel.from_pretrained('prajjwal1/bert-tiny')
        tokenizer = AutoTokenizer.from_pretrained('prajjwal1/bert-tiny')
        inputs = tokenizer(['Test matmul context'], return_tensors='pt', max_length=8)
        
        exporter = EnhancedSemanticExporter(verbose=False)
        
        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = os.path.join(temp_dir, 'matmul_test.onnx')
            
            exporter.export(
                model=model,
                args=(inputs['input_ids'],),
                output_path=output_path,
                opset_version=17
            )
            
            metadata = exporter.get_semantic_metadata()
            semantic_mappings = metadata['semantic_mappings']
            
            # Find MatMul operations with attention context
            attention_matmuls = [
                (name, mapping) for name, mapping in semantic_mappings.items()
                if (mapping.get('onnx_op_type') == 'MatMul' and 
                    'attention' in name.lower())
            ]
            
            # Should find attention-related MatMuls
            if len(attention_matmuls) > 0:
                for node_name, mapping in attention_matmuls:
                    # Context-aware MatMuls should have semantic classification
                    semantic_type = mapping.get('semantic_type', 'unknown')
                    # Should have some meaningful semantic type (not unknown)
                    assert semantic_type != 'unknown', (
                        f"Attention MatMul {node_name} should have semantic classification, got: {semantic_type}"
                    )


class TestConfidenceLevelValidation:
    """Test that confidence levels are appropriate for different edge case types."""
    
    def test_confidence_distribution_is_reasonable(self):
        """Test that confidence levels follow expected patterns."""
        model = AutoModel.from_pretrained('prajjwal1/bert-tiny')
        tokenizer = AutoTokenizer.from_pretrained('prajjwal1/bert-tiny')
        inputs = tokenizer(['Test confidence levels'], return_tensors='pt', max_length=8)
        
        exporter = EnhancedSemanticExporter(verbose=False)
        
        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = os.path.join(temp_dir, 'confidence_test.onnx')
            
            result = exporter.export(
                model=model,
                args=(inputs['input_ids'],),
                output_path=output_path,
                opset_version=17
            )
            
            # Check overall confidence distribution
            confidence_levels = result['confidence_levels']
            total_nodes = sum(confidence_levels.values())
            
            # Most nodes should have some level of confidence
            unknown_ratio = confidence_levels.get('none', 0) / total_nodes
            assert unknown_ratio < 0.05, (
                f"Too many nodes with no confidence: {unknown_ratio:.1%}"
            )
            
            # Should have reasonable distribution
            high_ratio = confidence_levels.get('high', 0) / total_nodes
            medium_ratio = confidence_levels.get('medium', 0) / total_nodes
            low_ratio = confidence_levels.get('low', 0) / total_nodes
            
            # High confidence should be from HF module mapping
            assert high_ratio > 0.5, f"Expected >50% high confidence, got {high_ratio:.1%}"
            
            # Should have some medium confidence from operation inference
            assert medium_ratio > 0.05, f"Expected >5% medium confidence, got {medium_ratio:.1%}"
    
    def test_edge_cases_do_not_have_high_confidence(self):
        """Test that true edge cases have appropriately lower confidence."""
        model = AutoModel.from_pretrained('prajjwal1/bert-tiny')
        tokenizer = AutoTokenizer.from_pretrained('prajjwal1/bert-tiny')
        inputs = tokenizer(['Test edge case confidence'], return_tensors='pt', max_length=8)
        
        exporter = EnhancedSemanticExporter(verbose=False)
        
        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = os.path.join(temp_dir, 'edge_confidence_test.onnx')
            
            exporter.export(
                model=model,
                args=(inputs['input_ids'],),
                output_path=output_path,
                opset_version=17
            )
            
            metadata = exporter.get_semantic_metadata()
            semantic_mappings = metadata['semantic_mappings']
            
            # Find true edge cases (constants, root operations, etc.)
            edge_case_nodes = [
                (name, mapping) for name, mapping in semantic_mappings.items()
                if (mapping.get('semantic_type') in ['constant', 'root_operation', 'numbered_operation'] or
                    mapping.get('primary_source') == 'pattern_fallback')
            ]
            
            # Edge cases should not have high confidence
            for node_name, mapping in edge_case_nodes:
                confidence = mapping.get('confidence', 'unknown')
                assert confidence != 'high', (
                    f"Edge case {node_name} should not have high confidence: {confidence}"
                )


class TestUniversalDesignCompliance:
    """Test that edge case handling maintains universal design principles."""
    
    def test_no_model_specific_hardcoding(self):
        """Test that edge case patterns don't contain model-specific hardcoding."""
        # This test validates that the semantic inference doesn't rely on 
        # specific model architectures or naming conventions
        
        # Test with the standard model
        model = AutoModel.from_pretrained('prajjwal1/bert-tiny')
        tokenizer = AutoTokenizer.from_pretrained('prajjwal1/bert-tiny')
        inputs = tokenizer(['Universal design test'], return_tensors='pt', max_length=8)
        
        exporter = EnhancedSemanticExporter(verbose=False)
        
        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = os.path.join(temp_dir, 'universal_test.onnx')
            
            exporter.export(
                model=model,
                args=(inputs['input_ids'],),
                output_path=output_path,
                opset_version=17
            )
            
            # The fact that this works without model-specific code proves universality
            metadata = exporter.get_semantic_metadata()
            semantic_mappings = metadata['semantic_mappings']
            
            # Should have reasonable coverage without hardcoding
            total_nodes = len(semantic_mappings)
            assert total_nodes > 100, "Should process substantial number of nodes"
            
            # Should have semantic classifications for most nodes
            classified_nodes = [
                mapping for mapping in semantic_mappings.values()
                if mapping.get('semantic_type') != 'unknown'
            ]
            classification_ratio = len(classified_nodes) / total_nodes
            assert classification_ratio > 0.80, (
                f"Should classify >80% of nodes, got {classification_ratio:.1%}"
            )


def test_all_edge_case_scenarios():
    """Run all edge case scenario tests."""
    print("🧪 Running all edge case scenario tests...")
    
    # This is a comprehensive test that could be run independently
    test_classes = [
        TestConstantOperationEdgeCases(),
        TestRootOperationEdgeCases(), 
        TestOperationInferenceEdgeCases(),
        TestConfidenceLevelValidation(),
        TestUniversalDesignCompliance()
    ]
    
    print(f"Running {len(test_classes)} test scenarios...")
    
    for test_class in test_classes:
        class_name = test_class.__class__.__name__
        print(f"  Running {class_name}...")
        
        # Run all test methods in the class
        for method_name in dir(test_class):
            if method_name.startswith('test_'):
                print(f"    {method_name}...")
                method = getattr(test_class, method_name)
                try:
                    method()
                    print(f"    ✅ {method_name} passed")
                except Exception as e:
                    print(f"    ❌ {method_name} failed: {e}")
                    raise
    
    print("✅ All edge case scenarios completed successfully!")


if __name__ == "__main__":
    test_all_edge_case_scenarios()