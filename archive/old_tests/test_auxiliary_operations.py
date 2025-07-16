"""
Auxiliary Operations Unit Tests

Tests for enhanced auxiliary operations tagging functionality implemented
in iterations 1-5.
"""

import json
import tempfile
from pathlib import Path

import onnx
import torch
import torch.nn as nn

from modelexport.strategies.htp.htp_hierarchy_exporter import HierarchyExporter


class SimpleTestModel(nn.Module):
    """Simple test model that generates auxiliary operations."""
    
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(10, 5)
        
    def forward(self, x):
        # This should generate auxiliary operations like Constant, MatMul, Add
        return self.linear(x)


class AuxiliaryOperationsTestModel(nn.Module):
    """Model designed to generate specific auxiliary operations."""
    
    def __init__(self):
        super().__init__()
        self.embedding = nn.Embedding(100, 20)
        self.linear = nn.Linear(200, 10)  # max_seq_len(10) * embed_dim(20) = 200
        self.layer_norm = nn.LayerNorm(10)
        
    def forward(self, input_ids):
        # Embedding will create auxiliary operations for parameter lookup
        x = self.embedding(input_ids)
        
        # Get shape for dynamic operations (creates Shape operations)
        batch_size = x.shape[0]
        seq_len = x.shape[1]
        
        # Create some constants (creates Constant operations)
        mask_value = torch.tensor(-1e9, dtype=x.dtype, device=x.device)
        
        # Reshape operations (creates Reshape/Transpose operations)
        x = x.reshape(batch_size, -1)  # Flatten to match linear layer input
        
        # Linear transformation (creates MatMul, Add operations)
        x = self.linear(x)
        
        # Layer normalization (creates additional auxiliary ops)
        x = self.layer_norm(x)
        
        return x


class TestAuxiliaryOperationsDetection:
    """Test auxiliary operations detection and classification."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.exporter = HierarchyExporter(strategy="htp")
        self.temp_dir = tempfile.mkdtemp()
        
    def test_auxiliary_operations_identification(self):
        """Test that auxiliary operations are correctly identified."""
        model = AuxiliaryOperationsTestModel()
        model.eval()
        
        inputs = torch.randint(0, 100, (2, 10))
        output_path = Path(self.temp_dir) / "aux_test.onnx"
        
        result = self.exporter.export(model, inputs, str(output_path))
        
        # Should achieve 100% coverage (no empty tags)
        assert result['total_operations'] > 0
        assert result['tagged_operations'] == result['total_operations']
        
        # Load ONNX model to analyze operations
        onnx_model = onnx.load(str(output_path))
        
        # Should contain auxiliary operations
        op_types = [node.op_type for node in onnx_model.graph.node]
        
        # Check for common auxiliary operations
        auxiliary_ops = ['Constant', 'MatMul', 'Add', 'Reshape', 'Transpose']
        found_auxiliary = [op for op in auxiliary_ops if op in op_types]
        
        assert len(found_auxiliary) > 0, f"No auxiliary operations found. Op types: {op_types}"
    
    def test_100_percent_coverage_guarantee(self):
        """Test that 100% operation coverage is guaranteed."""
        models = [
            SimpleTestModel(),
            AuxiliaryOperationsTestModel()
        ]
        
        for i, model in enumerate(models):
            model.eval()
            
            if i == 0:
                inputs = torch.randn(3, 10)
            else:
                inputs = torch.randint(0, 100, (2, 10))  # seq_len=10 to match linear layer (10*20=200)
                
            output_path = Path(self.temp_dir) / f"coverage_test_{i}.onnx"
            
            result = self.exporter.export(model, inputs, str(output_path))
            
            # Critical assertion: 100% coverage
            coverage_pct = (result['tagged_operations'] / result['total_operations']) * 100
            assert coverage_pct == 100.0, f"Coverage not 100%: {coverage_pct:.1f}%"
            
            # Verify no empty tags exist
            if 'node_tags' in result:
                for node_name, node_info in result['node_tags'].items():
                    tags = node_info.get('tags', [])
                    assert len(tags) > 0, f"Node {node_name} has empty tags"
    
    def test_auxiliary_operation_types_handled(self):
        """Test that all expected auxiliary operation types are handled."""
        model = AuxiliaryOperationsTestModel()
        model.eval()
        inputs = torch.randint(0, 100, (2, 10))  # seq_len=10 to match linear layer
        
        output_path = Path(self.temp_dir) / "aux_types_test.onnx"
        result = self.exporter.export(model, inputs, str(output_path))
        
        # Load and analyze the ONNX model
        onnx_model = onnx.load(str(output_path))
        
        # Expected auxiliary operation types
        expected_aux_ops = {
            'Constant', 'MatMul', 'Add', 'Gather', 'Transpose', 
            'Reshape', 'LayerNormalization'
        }
        
        found_ops = set(node.op_type for node in onnx_model.graph.node)
        auxiliary_ops_found = found_ops & expected_aux_ops
        
        # Should find several auxiliary operations
        assert len(auxiliary_ops_found) >= 3, f"Expected auxiliary ops, found: {auxiliary_ops_found}"
        
        # All operations should be tagged
        assert result['tagged_operations'] == result['total_operations']
    
    def test_fallback_strategy_behavior(self):
        """Test fallback strategy when context inheritance unavailable."""
        model = SimpleTestModel()
        model.eval()
        inputs = torch.randn(2, 10)
        
        output_path = Path(self.temp_dir) / "fallback_test.onnx"
        result = self.exporter.export(model, inputs, str(output_path))
        
        # Even with simple model, should achieve 100% coverage via fallback
        assert result['tagged_operations'] == result['total_operations']
        
        # Check that fallback strategy was used (some operations get default tags)
        if 'node_tags' in result:
            all_tags = []
            for node_info in result['node_tags'].values():
                all_tags.extend(node_info.get('tags', []))
            
            # Should have some tags (fallback strategy working)
            assert len(all_tags) > 0
            
            # Most tags should start with model name (fallback behavior)
            model_name_tags = [tag for tag in all_tags if 'SimpleTestModel' in tag]
            assert len(model_name_tags) > 0


class TestAuxiliaryOperationsIntegration:
    """Test auxiliary operations integration with HTP export."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.exporter = HierarchyExporter(strategy="htp")
        self.temp_dir = tempfile.mkdtemp()
    
    def test_auxiliary_operations_with_builtin_tracking(self):
        """Test auxiliary operations work with builtin tracking."""
        model = AuxiliaryOperationsTestModel()
        model.eval()
        inputs = torch.randint(0, 100, (2, 10))
        
        output_path = Path(self.temp_dir) / "builtin_aux_test.onnx"
        
        # Export should use builtin tracking by default
        result = self.exporter.export(model, inputs, str(output_path))
        
        # Verify builtin tracking was used
        assert result.get('builtin_tracking_enabled', False)
        
        # Should achieve 100% coverage
        assert result['tagged_operations'] == result['total_operations']
        
        # Should have strategy set correctly
        assert result['strategy'] in ['htp', 'htp_builtin']
    
    def test_auxiliary_operations_hierarchy_metadata(self):
        """Test that auxiliary operations are included in hierarchy metadata."""
        model = AuxiliaryOperationsTestModel()
        model.eval()
        inputs = torch.randint(0, 100, (2, 10))  # seq_len=10 to match linear layer
        
        output_path = Path(self.temp_dir) / "hierarchy_aux_test.onnx"
        result = self.exporter.export(model, inputs, str(output_path))
        
        # Check for hierarchy metadata file
        hierarchy_path = str(output_path).replace('.onnx', '_hierarchy.json')
        
        if Path(hierarchy_path).exists():
            with open(hierarchy_path) as f:
                hierarchy_data = json.load(f)
            
            # Should have node tags information
            assert 'node_tags' in hierarchy_data
            
            # All nodes should have tags
            for node_name, node_info in hierarchy_data['node_tags'].items():
                assert 'tags' in node_info
                assert len(node_info['tags']) > 0, f"Node {node_name} has empty tags"
                
                # Should have operation type information
                assert 'op_type' in node_info
    
    def test_auxiliary_operations_performance_impact(self):
        """Test that auxiliary operations don't significantly impact performance."""
        import time
        
        model = AuxiliaryOperationsTestModel()
        model.eval()
        inputs = torch.randint(0, 100, (2, 10))
        
        # Measure export time
        start_time = time.perf_counter()
        
        output_path = Path(self.temp_dir) / "perf_aux_test.onnx"
        result = self.exporter.export(model, inputs, str(output_path))
        
        end_time = time.perf_counter()
        export_time = end_time - start_time
        
        # Should complete within reasonable time (< 10 seconds)
        assert export_time < 10.0, f"Export took too long: {export_time:.2f}s"
        
        # Should still achieve 100% coverage
        assert result['tagged_operations'] == result['total_operations']


class TestAuxiliaryOperationsEdgeCases:
    """Test edge cases for auxiliary operations handling."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.exporter = HierarchyExporter(strategy="htp")
        self.temp_dir = tempfile.mkdtemp()
    
    def test_model_with_only_auxiliary_operations(self):
        """Test model that primarily generates auxiliary operations."""
        class AuxOnlyModel(nn.Module):
            def forward(self, x):
                # Operations that primarily generate auxiliary ops
                shape = x.shape[0]
                constant = torch.tensor(1.0)
                reshaped = x.reshape(-1)
                return reshaped + constant
        
        model = AuxOnlyModel()
        model.eval()
        inputs = torch.randn(3, 4)
        
        output_path = Path(self.temp_dir) / "aux_only_test.onnx"
        result = self.exporter.export(model, inputs, str(output_path))
        
        # Should handle auxiliary-heavy model correctly
        assert result['total_operations'] > 0
        assert result['tagged_operations'] == result['total_operations']
    
    def test_empty_model_handling(self):
        """Test handling of minimal models."""
        class MinimalModel(nn.Module):
            def forward(self, x):
                return x
        
        model = MinimalModel()
        model.eval()
        inputs = torch.randn(2, 3)
        
        output_path = Path(self.temp_dir) / "minimal_test.onnx"
        result = self.exporter.export(model, inputs, str(output_path))
        
        # Should handle minimal model gracefully
        if result['total_operations'] > 0:
            assert result['tagged_operations'] == result['total_operations']
    
    def test_complex_auxiliary_patterns(self):
        """Test complex patterns of auxiliary operations."""
        class ComplexAuxModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.linear1 = nn.Linear(10, 20)
                self.linear2 = nn.Linear(20, 5)
                
            def forward(self, x):
                # Multiple transformations creating auxiliary ops
                x = self.linear1(x)
                
                # Dynamic shape operations
                batch_size, features = x.shape
                
                # Multiple constants and casts
                scale = torch.tensor(2.0, dtype=x.dtype)
                offset = torch.tensor(1.0, dtype=x.dtype)
                
                # Complex reshaping
                x = x.view(batch_size, -1)
                x = x * scale + offset
                
                # Type conversions
                x = x.float()
                
                # Final transformation
                return self.linear2(x)
        
        model = ComplexAuxModel()
        model.eval()
        inputs = torch.randn(3, 10)
        
        output_path = Path(self.temp_dir) / "complex_aux_test.onnx"
        result = self.exporter.export(model, inputs, str(output_path))
        
        # Should handle complex auxiliary patterns
        assert result['total_operations'] > 0
        assert result['tagged_operations'] == result['total_operations']
        
        # Should have multiple operation types
        onnx_model = onnx.load(str(output_path))
        op_types = set(node.op_type for node in onnx_model.graph.node)
        assert len(op_types) >= 3, f"Expected multiple op types, got: {op_types}"


class TestAuxiliaryOperationsValidation:
    """Test validation of auxiliary operations functionality."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.exporter = HierarchyExporter(strategy="htp")
        self.temp_dir = tempfile.mkdtemp()
    
    def test_auxiliary_operations_onnx_validity(self):
        """Test that models with auxiliary operations produce valid ONNX."""
        model = AuxiliaryOperationsTestModel()
        model.eval()
        inputs = torch.randint(0, 100, (2, 10))  # seq_len=10 to match linear layer
        
        output_path = Path(self.temp_dir) / "validity_test.onnx"
        result = self.exporter.export(model, inputs, str(output_path))
        
        # Load and validate ONNX model
        onnx_model = onnx.load(str(output_path))
        
        # Should pass ONNX checker
        try:
            onnx.checker.check_model(onnx_model)
            onnx_valid = True
        except Exception as e:
            onnx_valid = False
            print(f"ONNX validation error: {e}")
        
        assert onnx_valid, "ONNX model should be valid"
        
        # Should have proper structure
        assert len(onnx_model.graph.node) > 0
        assert len(onnx_model.graph.input) > 0
        assert len(onnx_model.graph.output) > 0
    
    def test_auxiliary_operations_tag_consistency(self):
        """Test that auxiliary operations tags are consistent."""
        model = AuxiliaryOperationsTestModel()
        model.eval()
        inputs = torch.randint(0, 100, (2, 10))  # seq_len=10 to match linear layer
        
        output_path = Path(self.temp_dir) / "consistency_test.onnx"
        result = self.exporter.export(model, inputs, str(output_path))
        
        if 'node_tags' in result:
            all_tags = []
            for node_info in result['node_tags'].values():
                all_tags.extend(node_info.get('tags', []))
            
            # Tags should follow consistent format
            for tag in all_tags:
                assert isinstance(tag, str), f"Tag should be string: {tag}"
                assert len(tag) > 0, "Tag should not be empty"
                assert tag.startswith('/'), f"Tag should start with '/': {tag}"
                
                # Should not contain torch.nn modules
                torch_nn_modules = ['Linear', 'Embedding', 'LayerNorm', 'ReLU']
                for torch_module in torch_nn_modules:
                    assert torch_module not in tag, f"Tag contains torch.nn module: {tag}"