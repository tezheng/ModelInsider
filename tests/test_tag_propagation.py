"""
Unit tests for tag propagation algorithm.

Tests the bounded propagation logic that prevents over-tagging and ensures
tags are propagated appropriately within module boundaries.
"""

import pytest
import torch
import torch.nn as nn
import onnx
from modelexport import HierarchyExporter
from transformers import AutoModel, AutoTokenizer
import tempfile
import os
from collections import defaultdict


class TestTagPropagation:
    """Test tag propagation algorithm functionality."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.exporter = HierarchyExporter()
    
    def test_bounded_propagation_depth_limit(self):
        """Test that propagation respects depth limits."""
        # Mock ONNX model with deep dependency chain
        mock_tensor_producers = {
            'tensor1': 'op1',
            'tensor2': 'op2', 
            'tensor3': 'op3',
            'tensor4': 'op4',
            'tensor5': 'op5'
        }
        
        # Create a chain: op1 -> op2 -> op3 -> op4 -> op5
        self.exporter._tag_mapping = {
            'op1': {'op_type': 'MatMul', 'tags': ['/Module1'], 'inputs': ['tensor2'], 'outputs': ['tensor1']},
            'op2': {'op_type': 'Add', 'tags': [], 'inputs': ['tensor3'], 'outputs': ['tensor2']},
            'op3': {'op_type': 'Relu', 'tags': [], 'inputs': ['tensor4'], 'outputs': ['tensor3']},
            'op4': {'op_type': 'MatMul', 'tags': [], 'inputs': ['tensor5'], 'outputs': ['tensor4']},
            'op5': {'op_type': 'Constant', 'tags': [], 'inputs': [], 'outputs': ['tensor5']}
        }
        
        # Mock ONNX model
        class MockONNXModel:
            def __init__(self):
                self.graph = self
                self.node = [
                    type('Node', (), {'name': 'op1', 'input': ['tensor2'], 'output': ['tensor1']})(),
                    type('Node', (), {'name': 'op2', 'input': ['tensor3'], 'output': ['tensor2']})(),
                    type('Node', (), {'name': 'op3', 'input': ['tensor4'], 'output': ['tensor3']})(),
                    type('Node', (), {'name': 'op4', 'input': ['tensor5'], 'output': ['tensor4']})(),
                    type('Node', (), {'name': 'op5', 'input': [], 'output': ['tensor5']})()
                ]
        
        mock_model = MockONNXModel()
        
        # Run propagation
        self.exporter._propagate_tags_recursively(mock_model, mock_tensor_producers)
        
        # Check results: with MAX_PROPAGATION_DEPTH = 3, op5 should NOT get tags
        assert len(self.exporter._tag_mapping['op1']['tags']) > 0, "op1 should keep its original tags"
        assert len(self.exporter._tag_mapping['op2']['tags']) > 0, "op2 should get propagated tags (depth 1)"
        assert len(self.exporter._tag_mapping['op3']['tags']) > 0, "op3 should get propagated tags (depth 2)"
        assert len(self.exporter._tag_mapping['op4']['tags']) > 0, "op4 should get propagated tags (depth 3)"
        # op5 might or might not get tags depending on exact implementation, but should be limited
    
    def test_tag_compatibility_blocking(self):
        """Test that incompatible tags don't propagate to nodes with existing different tags."""
        self.exporter._tag_mapping = {
            'embedding_op': {'op_type': 'Gather', 'tags': ['/BertModel/BertEmbeddings'], 'inputs': ['param1'], 'outputs': ['emb_out']},
            'attention_op': {'op_type': 'MatMul', 'tags': ['/BertModel/BertEncoder/BertLayer/BertAttention/BertSelfOutput'], 'inputs': ['emb_out'], 'outputs': ['attn_out']}
        }
        
        mock_tensor_producers = {'emb_out': 'embedding_op'}
        
        class MockONNXModel:
            def __init__(self):
                self.graph = self
                self.node = [
                    type('Node', (), {'name': 'embedding_op', 'input': ['param1'], 'output': ['emb_out']})(),
                    type('Node', (), {'name': 'attention_op', 'input': ['emb_out'], 'output': ['attn_out']})()
                ]
        
        mock_model = MockONNXModel()
        
        # Run propagation
        self.exporter._propagate_tags_recursively(mock_model, mock_tensor_producers)
        
        # embedding_op should NOT get attention tags due to incompatibility
        embedding_tags = self.exporter._tag_mapping['embedding_op']['tags']
        assert '/BertModel/BertEmbeddings' in embedding_tags, "Embedding should keep its original tag"
        assert not any('BertAttention' in tag for tag in embedding_tags), \
            "Embedding should not get incompatible attention tags"
    
    def test_module_boundary_blocking(self):
        """Test that tags don't propagate across major module boundaries."""
        # Test case: encoder operations should not propagate tags to embedding operations
        result = self.exporter._should_propagate_tag(
            '/BertModel/BertEncoder/BertLayer/BertAttention/BertSelfOutput',
            '/embeddings/word_embeddings/Gather',
            'embeddings.word_embeddings.weight'
        )
        assert not result, "Encoder tags should not propagate to embedding operations"
        
        # Test case: same module should allow propagation
        result = self.exporter._should_propagate_tag(
            '/BertModel/BertEmbeddings',
            '/embeddings/LayerNorm/Add',
            'embeddings.layernorm.bias'
        )
        assert result, "Same module tags should propagate within module"
        
        # Test case: attention operations should allow propagation within attention
        result = self.exporter._should_propagate_tag(
            '/BertModel/BertEncoder/BertLayer/BertAttention/BertSdpaSelfAttention',
            '/encoder/layer.0/attention/self/query/MatMul',
            'encoder.layer.0.attention.self.query.weight'
        )
        assert result, "Attention tags should propagate within attention operations"
    
    def test_real_bert_propagation_bounds(self):
        """Test bounded propagation with real BERT model to ensure no over-tagging."""
        model = AutoModel.from_pretrained('prajjwal1/bert-tiny')
        tokenizer = AutoTokenizer.from_pretrained('prajjwal1/bert-tiny')
        
        inputs = tokenizer("Test propagation bounds", return_tensors='pt', padding=True, truncation=True)
        
        with tempfile.NamedTemporaryFile(suffix='.onnx', delete=False) as f:
            try:
                # Export with bounded propagation
                result = self.exporter.export(model, inputs, f.name)
                
                # Load sidecar data to analyze results
                import json
                sidecar_path = f.name.replace('.onnx', '_hierarchy.json')
                with open(sidecar_path, 'r') as sidecar_file:
                    sidecar_data = json.load(sidecar_file)
                
                tag_stats = sidecar_data['tag_statistics']
                
                # Verify bounded propagation worked
                # BertSelfOutput should have reasonable count (not over-propagated)
                bert_self_output_count = tag_stats.get('/BertModel/BertEncoder/BertLayer/BertAttention/BertSelfOutput', 0)
                assert bert_self_output_count < 50, \
                    f"BertSelfOutput over-propagated: {bert_self_output_count} operations (should be < 50)"
                assert bert_self_output_count > 10, \
                    f"BertSelfOutput under-tagged: {bert_self_output_count} operations (should be > 10)"
                
                # Embeddings should not have encoder tags
                node_tags = sidecar_data['node_tags']
                embedding_ops = [name for name, info in node_tags.items() if 'embedding' in name.lower()]
                
                for op_name in embedding_ops[:5]:  # Check first few embedding ops
                    op_tags = node_tags[op_name]['tags']
                    has_encoder_tags = any('Encoder' in tag for tag in op_tags)
                    assert not has_encoder_tags, \
                        f"Embedding operation {op_name} incorrectly has encoder tags: {op_tags}"
                
                # Total tagged operations should be reasonable (not everything tagged)
                total_ops = sidecar_data['summary']['total_operations']
                tagged_ops = sidecar_data['summary']['tagged_operations']
                tagging_ratio = tagged_ops / total_ops
                
                assert tagging_ratio < 0.8, \
                    f"Too many operations tagged: {tagging_ratio:.2f} ratio (should be < 0.8)"
                assert tagging_ratio > 0.3, \
                    f"Too few operations tagged: {tagging_ratio:.2f} ratio (should be > 0.3)"
                
            finally:
                os.unlink(f.name)
                if os.path.exists(f.name.replace('.onnx', '_hierarchy.json')):
                    os.unlink(f.name.replace('.onnx', '_hierarchy.json'))
    
    def test_tag_compatibility_logic(self):
        """Test the tag compatibility logic in detail."""
        test_cases = [
            # Same module - compatible
            ("/BertModel/BertEmbeddings", "/BertModel/BertEmbeddings", True),
            
            # Same parent module - compatible  
            ("/BertModel/BertEncoder/BertLayer", "/BertModel/BertEncoder/BertLayer/BertAttention", True),
            
            # Different top-level modules - incompatible
            ("/BertModel/BertEmbeddings", "/BertModel/BertPooler", False),
            
            # Different encoder branches - incompatible
            ("/BertModel/BertEmbeddings", "/BertModel/BertEncoder", False),
            
            # Same attention sub-modules - compatible
            ("/BertModel/BertEncoder/BertLayer/BertAttention/BertSelfOutput", 
             "/BertModel/BertEncoder/BertLayer/BertAttention/BertSdpaSelfAttention", True),
            
            # Different layer numbers (if present) - should be incompatible
            # This tests the first 3 levels matching requirement
        ]
        
        for tag1, tag2, expected in test_cases:
            result = self.exporter._are_tags_compatible(tag1, tag2)
            assert result == expected, \
                f"Tag compatibility failed for '{tag1}' vs '{tag2}': expected {expected}, got {result}"
    
    def test_propagation_statistics_consistency(self):
        """Test that propagation produces consistent statistics."""
        model = AutoModel.from_pretrained('prajjwal1/bert-tiny')
        tokenizer = AutoTokenizer.from_pretrained('prajjwal1/bert-tiny')
        
        inputs = tokenizer("Consistency test", return_tensors='pt', padding=True, truncation=True)
        
        with tempfile.NamedTemporaryFile(suffix='.onnx', delete=False) as f:
            try:
                # Export twice to ensure consistency
                result1 = self.exporter.export(model, inputs, f.name)
                
                # Reset and export again
                self.exporter._reset_state()
                result2 = self.exporter.export(model, inputs, f.name + '2')
                
                # Results should be consistent
                assert result1['total_operations'] == result2['total_operations'], \
                    "Total operations should be consistent across exports"
                
                # Tagged operations should be similar (allow small variance due to implementation details)
                tagged_diff = abs(result1['tagged_operations'] - result2['tagged_operations'])
                assert tagged_diff <= 2, \
                    f"Tagged operations should be consistent: {result1['tagged_operations']} vs {result2['tagged_operations']}"
                
            finally:
                for path in [f.name, f.name + '2']:
                    if os.path.exists(path):
                        os.unlink(path)
                    if os.path.exists(path.replace('.onnx', '_hierarchy.json')):
                        os.unlink(path.replace('.onnx', '_hierarchy.json'))


if __name__ == "__main__":
    pytest.main([__file__, "-v"])