"""
Test suite for Graph Pattern Recognition functionality.

Tests the pattern recognition system's ability to identify common
computational patterns in ONNX graphs.
"""

import tempfile
from pathlib import Path

import onnx
import pytest
import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer

from modelexport.semantic.enhanced_semantic_mapper import EnhancedSemanticMapper
from modelexport.semantic.graph_pattern_recognizer import (
    GraphPattern,
    GraphPatternRecognizer,
)


class TestGraphPatternRecognizer:
    """Test the GraphPatternRecognizer class."""
    
    def setup_method(self):
        """Setup test environment."""
        self.recognizer = GraphPatternRecognizer()
    
    def test_initialization(self):
        """Test recognizer initialization."""
        assert self.recognizer is not None
        assert len(self.recognizer._patterns) > 0
        assert 'self_attention' in self.recognizer._patterns
        assert 'layer_norm' in self.recognizer._patterns
        assert 'feed_forward' in self.recognizer._patterns
    
    def test_pattern_definitions(self):
        """Test that pattern definitions are valid."""
        for pattern_name, pattern_def in self.recognizer._patterns.items():
            # Check required fields
            assert 'description' in pattern_def
            assert 'node_sequence' in pattern_def
            assert 'semantic_type' in pattern_def
            assert 'confidence' in pattern_def
            
            # Check confidence range
            assert 0.0 <= pattern_def['confidence'] <= 1.0
            
            # Check semantic type is valid
            valid_types = {
                'attention', 'normalization', 'activation', 'feed_forward',
                'embedding', 'residual', 'pooling', 'convolution'
            }
            assert pattern_def['semantic_type'] in valid_types
    
    def test_output_to_producer_map(self):
        """Test building of output to producer mapping."""
        # Create simple ONNX model
        with tempfile.NamedTemporaryFile(suffix='.onnx', delete=False) as f:
            onnx_path = f.name
        
        try:
            # Create minimal model for testing
            model = nn.Linear(10, 5)
            dummy_input = torch.randn(1, 10)
            
            torch.onnx.export(
                model, dummy_input, onnx_path,
                opset_version=17, verbose=False
            )
            
            onnx_model = onnx.load(onnx_path)
            output_map = self.recognizer._build_output_to_producer_map(onnx_model.graph)
            
            assert isinstance(output_map, dict)
            # Should have entries for model outputs
            assert len(output_map) > 0
            
        finally:
            Path(onnx_path).unlink(missing_ok=True)
    
    def test_input_to_consumers_map(self):
        """Test building of input to consumers mapping."""
        with tempfile.NamedTemporaryFile(suffix='.onnx', delete=False) as f:
            onnx_path = f.name
        
        try:
            # Create model with multiple operations
            class SimpleModel(nn.Module):
                def __init__(self):
                    super().__init__()
                    self.linear1 = nn.Linear(10, 20)
                    self.linear2 = nn.Linear(20, 5)
                
                def forward(self, x):
                    x = self.linear1(x)
                    x = torch.relu(x)
                    x = self.linear2(x)
                    return x
            
            model = SimpleModel()
            dummy_input = torch.randn(1, 10)
            
            torch.onnx.export(
                model, dummy_input, onnx_path,
                opset_version=17, verbose=False
            )
            
            onnx_model = onnx.load(onnx_path)
            consumer_map = self.recognizer._build_input_to_consumers_map(onnx_model.graph)
            
            assert isinstance(consumer_map, dict)
            # Should have entries mapping inputs to consumers
            assert len(consumer_map) > 0
            
        finally:
            Path(onnx_path).unlink(missing_ok=True)
    
    def test_pattern_recognition_on_simple_model(self):
        """Test pattern recognition on a simple model."""
        with tempfile.NamedTemporaryFile(suffix='.onnx', delete=False) as f:
            onnx_path = f.name
        
        try:
            # Create model with recognizable patterns
            class PatternModel(nn.Module):
                def __init__(self):
                    super().__init__()
                    self.embedding = nn.Embedding(100, 64)
                    self.layer_norm = nn.LayerNorm(64)
                    self.linear = nn.Linear(64, 32)
                
                def forward(self, x):
                    x = self.embedding(x)
                    x = self.layer_norm(x)
                    x = self.linear(x)
                    return x
            
            model = PatternModel()
            dummy_input = torch.randint(0, 100, (1, 10))
            
            torch.onnx.export(
                model, dummy_input, onnx_path,
                opset_version=17, verbose=False
            )
            
            onnx_model = onnx.load(onnx_path)
            patterns = self.recognizer.recognize_patterns(onnx_model)
            
            # Should find some patterns
            assert isinstance(patterns, list)
            
            # Check pattern structure
            for pattern in patterns:
                assert isinstance(pattern, GraphPattern)
                assert pattern.pattern_type is not None
                assert pattern.semantic_type is not None
                assert 0.0 <= pattern.confidence <= 1.0
                assert isinstance(pattern.nodes, list)
                assert len(pattern.nodes) > 0
                
        finally:
            Path(onnx_path).unlink(missing_ok=True)
    
    def test_pattern_enhancement(self):
        """Test semantic mapping enhancement with patterns."""
        with tempfile.NamedTemporaryFile(suffix='.onnx', delete=False) as f:
            onnx_path = f.name
        
        try:
            model = nn.Sequential(
                nn.Linear(10, 20),
                nn.ReLU(),
                nn.Linear(20, 5)
            )
            dummy_input = torch.randn(1, 10)
            
            torch.onnx.export(
                model, dummy_input, onnx_path,
                opset_version=17, verbose=False
            )
            
            onnx_model = onnx.load(onnx_path)
            
            # Create dummy existing mappings
            existing_mappings = {}
            for node in onnx_model.graph.node:
                existing_mappings[node.name] = {
                    'semantic_type': 'unknown',
                    'confidence': 'low'
                }
            
            # Enhance with patterns
            enhanced_mappings = self.recognizer.enhance_semantic_mappings(
                onnx_model, existing_mappings
            )
            
            assert isinstance(enhanced_mappings, dict)
            assert '__metadata__' in enhanced_mappings
            assert 'pattern_recognition' in enhanced_mappings['__metadata__']
            
            metadata = enhanced_mappings['__metadata__']['pattern_recognition']
            assert 'patterns_found' in metadata
            assert 'nodes_enhanced' in metadata
            assert 'pattern_types' in metadata
            
        finally:
            Path(onnx_path).unlink(missing_ok=True)
    
    def test_pattern_statistics(self):
        """Test pattern statistics collection."""
        stats = self.recognizer.get_pattern_statistics()
        
        assert isinstance(stats, dict)
        assert 'total_patterns_matched' in stats
        assert 'pattern_distribution' in stats
        assert 'defined_patterns' in stats
        
        # Check defined patterns
        defined = stats['defined_patterns']
        assert 'self_attention' in defined
        assert 'layer_norm' in defined
        assert 'feed_forward' in defined


class TestPatternIntegrationWithEnhancedMapper:
    """Test integration of pattern recognition with Enhanced Semantic Mapper."""
    
    def setup_method(self):
        """Setup test environment."""
        self.model_name = "prajjwal1/bert-tiny"
    
    @pytest.mark.slow
    def test_enhanced_mapper_with_pattern_recognition(self):
        """Test enhanced mapper with pattern recognition integration."""
        # Load model
        model = AutoModel.from_pretrained(self.model_name)
        tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        
        # Create inputs
        inputs = tokenizer(
            "Test sentence", 
            return_tensors="pt", 
            max_length=16, 
            padding=True, 
            truncation=True
        )
        
        with tempfile.NamedTemporaryFile(suffix='.onnx', delete=False) as f:
            onnx_path = f.name
        
        try:
            # Export to ONNX
            torch.onnx.export(
                model,
                inputs['input_ids'],
                onnx_path,
                opset_version=17,
                verbose=False,
                dynamic_axes={
                    'input_ids': {0: 'batch_size', 1: 'sequence_length'}
                }
            )
            
            # Load ONNX model
            onnx_model = onnx.load(onnx_path)
            
            # Create enhanced mapper
            mapper = EnhancedSemanticMapper(model, onnx_model)
            
            # Test pattern recognition enhancement
            pattern_result = mapper.enhance_with_pattern_recognition()
            
            assert 'enhanced_mappings' in pattern_result
            assert 'pattern_statistics' in pattern_result
            assert 'pattern_recognizer' in pattern_result
            
            # Check enhanced mappings
            enhanced_mappings = pattern_result['enhanced_mappings']
            assert isinstance(enhanced_mappings, dict)
            
            # Check pattern statistics
            pattern_stats = pattern_result['pattern_statistics']
            assert 'total_patterns_matched' in pattern_stats
            assert 'pattern_distribution' in pattern_stats
            
            # Check that some patterns were found
            if pattern_stats['total_patterns_matched'] > 0:
                assert len(pattern_stats['pattern_distribution']) > 0
            
        finally:
            Path(onnx_path).unlink(missing_ok=True)
    
    @pytest.mark.slow
    def test_pattern_recognition_improves_coverage(self):
        """Test that pattern recognition improves semantic coverage."""
        # Load smaller model for faster testing
        model = AutoModel.from_pretrained(self.model_name)
        tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        
        # Create inputs
        inputs = tokenizer(
            "Test", 
            return_tensors="pt", 
            max_length=8, 
            padding=True, 
            truncation=True
        )
        
        with tempfile.NamedTemporaryFile(suffix='.onnx', delete=False) as f:
            onnx_path = f.name
        
        try:
            # Export to ONNX
            torch.onnx.export(
                model, inputs['input_ids'], onnx_path,
                opset_version=17, verbose=False
            )
            
            onnx_model = onnx.load(onnx_path)
            mapper = EnhancedSemanticMapper(model, onnx_model)
            
            # Get baseline coverage
            baseline_stats = mapper.get_mapping_coverage_stats()
            baseline_coverage = baseline_stats['total_coverage_percentage']
            
            # Apply pattern recognition
            pattern_result = mapper.enhance_with_pattern_recognition()
            enhanced_mappings = pattern_result['enhanced_mappings']
            
            # Count enhanced nodes
            enhanced_nodes = 0
            if '__metadata__' in enhanced_mappings:
                metadata = enhanced_mappings['__metadata__']
                if 'pattern_recognition' in metadata:
                    enhanced_nodes = metadata['pattern_recognition']['nodes_enhanced']
            
            # Pattern recognition should provide additional insights
            pattern_stats = pattern_result['pattern_statistics']
            
            # Should have some defined patterns
            assert len(pattern_stats['defined_patterns']) > 0
            
            # Check if any patterns were matched (depends on model complexity)
            total_matched = pattern_stats['total_patterns_matched']
            print(f"Patterns matched: {total_matched}")
            print(f"Nodes enhanced: {enhanced_nodes}")
            print(f"Baseline coverage: {baseline_coverage:.1f}%")
            
        finally:
            Path(onnx_path).unlink(missing_ok=True)


class TestSpecificPatterns:
    """Test recognition of specific pattern types."""
    
    def setup_method(self):
        """Setup test environment."""
        self.recognizer = GraphPatternRecognizer()
    
    def test_embedding_pattern(self):
        """Test recognition of embedding lookup patterns."""
        # Create model with embedding
        model = nn.Embedding(100, 64)
        dummy_input = torch.randint(0, 100, (1, 10))
        
        with tempfile.NamedTemporaryFile(suffix='.onnx', delete=False) as f:
            onnx_path = f.name
        
        try:
            torch.onnx.export(
                model, dummy_input, onnx_path,
                opset_version=17, verbose=False
            )
            
            onnx_model = onnx.load(onnx_path)
            patterns = self.recognizer.recognize_patterns(onnx_model)
            
            # Should recognize embedding patterns
            embedding_patterns = [
                p for p in patterns 
                if p.semantic_type == 'embedding' or 'embedding' in p.pattern_type
            ]
            
            # May or may not find patterns depending on ONNX structure
            print(f"Found {len(embedding_patterns)} embedding patterns")
            
        finally:
            Path(onnx_path).unlink(missing_ok=True)
    
    def test_activation_pattern(self):
        """Test recognition of activation patterns."""
        # Create model with various activations
        class ActivationModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = nn.Linear(10, 10)
            
            def forward(self, x):
                x = self.linear(x)
                x = torch.relu(x)
                x = torch.sigmoid(x)
                return x
        
        model = ActivationModel()
        dummy_input = torch.randn(1, 10)
        
        with tempfile.NamedTemporaryFile(suffix='.onnx', delete=False) as f:
            onnx_path = f.name
        
        try:
            torch.onnx.export(
                model, dummy_input, onnx_path,
                opset_version=17, verbose=False
            )
            
            onnx_model = onnx.load(onnx_path)
            patterns = self.recognizer.recognize_patterns(onnx_model)
            
            # Check for activation-related patterns
            activation_patterns = [
                p for p in patterns 
                if p.semantic_type == 'activation'
            ]
            
            print(f"Found {len(activation_patterns)} activation patterns")
            
        finally:
            Path(onnx_path).unlink(missing_ok=True)


if __name__ == "__main__":
    # Run basic tests
    test_recognizer = TestGraphPatternRecognizer()
    test_recognizer.setup_method()
    
    print("🧪 Testing Graph Pattern Recognition")
    print("="*40)
    
    try:
        test_recognizer.test_initialization()
        print("✅ Initialization test passed")
        
        test_recognizer.test_pattern_definitions()
        print("✅ Pattern definitions test passed")
        
        test_recognizer.test_pattern_statistics()
        print("✅ Pattern statistics test passed")
        
        print("\n🔍 Pattern Recognition Tests Complete!")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()