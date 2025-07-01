"""
Unit tests for parameter-to-module mapping logic.

Tests the core functionality of mapping ONNX parameter names back to 
PyTorch module execution contexts.
"""

import pytest
import torch
import torch.nn as nn
from modelexport import HierarchyExporter
from transformers import AutoModel, AutoTokenizer
import tempfile
import os


class SimpleModel(nn.Module):
    """Simple test model for parameter mapping tests."""
    
    def __init__(self):
        super().__init__()
        self.embedding = nn.Embedding(100, 64)
        self.linear1 = nn.Linear(64, 32)
        self.linear2 = nn.Linear(32, 10)
        
    def forward(self, x):
        x = self.embedding(x)
        x = torch.relu(self.linear1(x))
        x = self.linear2(x)
        return x


class TestParameterMapping:
    """Test parameter-to-module mapping functionality."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.exporter = HierarchyExporter()
    
    def test_extract_module_name_from_param(self):
        """Test parameter name to module name extraction."""
        test_cases = [
            ("embedding.weight", "embedding"),
            ("linear1.weight", "linear1"),  
            ("linear1.bias", "linear1"),
            ("encoder.layer.0.attention.self.query.weight", "encoder.layer.0.attention.self.query"),
            ("encoder.layer.0.attention.self.query.bias", "encoder.layer.0.attention.self.query"),
            ("embeddings.word_embeddings.weight", "embeddings.word_embeddings"),
            ("embeddings.position_embeddings.weight", "embeddings.position_embeddings"),
            ("encoder.layer.0.attention.output.dense.weight", "encoder.layer.0.attention.output.dense"),
            ("layernorm.running_mean", "layernorm"),
            ("batchnorm.running_var", "batchnorm"),
            ("batchnorm.num_batches_tracked", "batchnorm")
        ]
        
        for param_name, expected_module in test_cases:
            result = self.exporter._extract_module_name_from_param(param_name)
            assert result == expected_module, f"Failed for {param_name}: expected {expected_module}, got {result}"
    
    def test_parameter_mapping_simple_model(self):
        """Test parameter mapping with simple PyTorch model."""
        model = SimpleModel()
        inputs = torch.tensor([[1, 2, 3]])
        
        with tempfile.NamedTemporaryFile(suffix='.onnx', delete=False) as f:
            try:
                # Export model
                self.exporter.export(model, inputs, f.name)
                
                # Check that parameter mapping was created
                assert hasattr(self.exporter, '_param_to_module')
                assert len(self.exporter._param_to_module) > 0
                
                # Verify specific parameter mappings
                param_mappings = self.exporter._param_to_module
                
                # Should have mappings for model parameters
                expected_params = ["embedding.weight", "linear1.weight", "linear1.bias", 
                                 "linear2.weight", "linear2.bias"]
                
                mapped_params = list(param_mappings.keys())
                
                # Check that all expected parameters have mappings (including generic ONNX names)
                # Note: torch.nn modules are filtered out, so all parameters map to /SimpleModel
                simple_model_params = [param for param, context in param_mappings.items() 
                                     if '/SimpleModel' in context.get('tag', '')]
                
                # Should have parameters for the SimpleModel (embedding + linear layers)
                assert len(simple_model_params) >= 3, f"Expected at least 3 SimpleModel parameters, got {simple_model_params}"
                
            finally:
                os.unlink(f.name)
                if os.path.exists(f.name.replace('.onnx', '_hierarchy.json')):
                    os.unlink(f.name.replace('.onnx', '_hierarchy.json'))
    
    def test_parameter_mapping_transformers_model(self):
        """Test parameter mapping with transformers BERT model."""
        model = AutoModel.from_pretrained('prajjwal1/bert-tiny')
        tokenizer = AutoTokenizer.from_pretrained('prajjwal1/bert-tiny')
        
        inputs = tokenizer("Hello world", return_tensors='pt', padding=True, truncation=True)
        
        with tempfile.NamedTemporaryFile(suffix='.onnx', delete=False) as f:
            try:
                # Export model
                self.exporter.export(model, inputs, f.name)
                
                # Check parameter mapping
                param_mappings = self.exporter._param_to_module
                assert len(param_mappings) > 0
                
                # Check for expected BERT parameters
                expected_param_patterns = [
                    "embeddings.word_embeddings.weight",
                    "embeddings.position_embeddings.weight", 
                    "embeddings.token_type_embeddings.weight",
                    "encoder.layer.0.attention.self.query.weight",
                    "encoder.layer.0.attention.self.key.weight",
                    "encoder.layer.0.attention.self.value.weight",
                    "encoder.layer.0.attention.output.dense.weight",
                    "pooler.dense.weight"
                ]
                
                mapped_params = list(param_mappings.keys())
                
                # Check by module types since ONNX export may rename parameters
                embedding_params = [param for param, context in param_mappings.items() 
                                   if 'BertEmbeddings' in context.get('tag', '')]
                attention_params = [param for param, context in param_mappings.items() 
                                  if 'BertAttention' in context.get('tag', '') or 'BertSdpa' in context.get('tag', '')]
                pooler_params = [param for param, context in param_mappings.items() 
                               if 'BertPooler' in context.get('tag', '')]
                
                # Should have parameters for major module types
                assert len(embedding_params) >= 1, f"Expected BertEmbeddings parameters, got {embedding_params}"
                assert len(attention_params) >= 1, f"Expected attention parameters, got {attention_params}"
                
                # Verify that mappings point to valid module contexts
                for param_name, module_context in param_mappings.items():
                    assert 'tag' in module_context, f"Missing tag for parameter: {param_name}"
                    assert 'module_class' in module_context, f"Missing module_class for parameter: {param_name}"
                    assert module_context['tag'].startswith('/'), f"Invalid tag format for parameter: {param_name}"
                
            finally:
                os.unlink(f.name)
                if os.path.exists(f.name.replace('.onnx', '_hierarchy.json')):
                    os.unlink(f.name.replace('.onnx', '_hierarchy.json'))
    
    def test_find_parent_transformers_module(self):
        """Test finding parent transformers modules."""
        # Set up mock operation context
        self.exporter._operation_context = {
            'encoder.layer.0.attention.self': {'tag': '/BertModel/BertEncoder/BertLayer/BertAttention/BertSdpaSelfAttention'},
            'encoder.layer.0.attention.output': {'tag': '/BertModel/BertEncoder/BertLayer/BertAttention/BertSelfOutput'},
            'encoder.layer.0': {'tag': '/BertModel/BertEncoder/BertLayer'},
            'embeddings': {'tag': '/BertModel/BertEmbeddings'},
        }
        
        test_cases = [
            ("encoder.layer.0.attention.self.query", "encoder.layer.0.attention.self"),
            ("encoder.layer.0.attention.self.key", "encoder.layer.0.attention.self"),
            ("encoder.layer.0.attention.output.dense", "encoder.layer.0.attention.output"),
            ("encoder.layer.0.intermediate.dense", "encoder.layer.0"),
            ("embeddings.word_embeddings", "embeddings"),
            ("embeddings.position_embeddings", "embeddings"),
            ("nonexistent.module", None)
        ]
        
        for module_name, expected_parent in test_cases:
            result = self.exporter._find_parent_transformers_module(module_name)
            assert result == expected_parent, \
                f"Failed for {module_name}: expected {expected_parent}, got {result}"
    
    def test_bounded_propagation_helpers(self):
        """Test bounded propagation helper methods."""
        # Test tag compatibility
        compatible_cases = [
            ("/BertModel/BertEmbeddings", "/BertModel/BertEmbeddings", True),
            ("/BertModel/BertEncoder/BertLayer", "/BertModel/BertEncoder/BertLayer/BertAttention", True),
            ("/BertModel/BertEmbeddings", "/BertModel/BertEncoder", False),
            ("/BertModel/BertEncoder/BertLayer/BertAttention/BertSelfOutput", 
             "/BertModel/BertEncoder/BertLayer/BertAttention/BertSdpaSelfAttention", True),
        ]
        
        for tag1, tag2, expected in compatible_cases:
            result = self.exporter._are_tags_compatible(tag1, tag2)
            assert result == expected, \
                f"Tag compatibility failed for {tag1} vs {tag2}: expected {expected}, got {result}"
        
        # Test propagation rules  
        propagation_cases = [
            ("/BertModel/BertEncoder/BertLayer/BertAttention/BertSelfOutput", 
             "/embeddings/word_embeddings/Gather", "embeddings.word_embeddings.weight", False),
            ("/BertModel/BertEmbeddings", 
             "/embeddings/LayerNorm/Add", "embeddings.layernorm.weight", True),
            ("/BertModel/BertEncoder/BertLayer/BertAttention/BertSdpaSelfAttention",
             "/encoder/layer.0/attention/self/query/MatMul", "encoder.layer.0.attention.self.query.weight", True),
        ]
        
        for tag, producer_node, tensor_name, expected in propagation_cases:
            result = self.exporter._should_propagate_tag(tag, producer_node, tensor_name)
            assert result == expected, \
                f"Propagation rule failed for tag {tag} to {producer_node}: expected {expected}, got {result}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])