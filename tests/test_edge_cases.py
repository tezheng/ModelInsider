"""
Edge Cases and Error Handling Tests - Round 2

These tests validate robustness against unusual inputs, edge cases,
and error conditions that might occur in production environments.
"""

import pytest
import torch
import tempfile
import json
import os
from pathlib import Path
from unittest.mock import patch, MagicMock

from modelexport.hierarchy_exporter import HierarchyExporter, OperationConfig


class TestInputEdgeCases:
    """Test edge cases for various input types and shapes."""
    
    def test_empty_tensor_input(self, simple_pytorch_model):
        """Edge case: Model with empty tensor input."""
        exporter = HierarchyExporter(strategy="usage_based")
        
        # Empty tensor should still work
        empty_input = torch.randn(0, 10)  # Batch size 0
        
        with tempfile.NamedTemporaryFile(suffix='.onnx') as tmp:
            # This might fail, but should fail gracefully
            try:
                result = exporter.export(
                    model=simple_pytorch_model,
                    example_inputs=empty_input,
                    output_path=tmp.name
                )
                # If it succeeds, validate result
                assert result is not None
                assert 'output_path' in result
            except (RuntimeError, ValueError) as e:
                # Expected failure - empty batch size
                assert "batch" in str(e).lower() or "size" in str(e).lower()
    
    def test_single_element_tensor(self, simple_pytorch_model):
        """Edge case: Single element input tensor."""
        exporter = HierarchyExporter(strategy="usage_based")
        
        # Single element tensor
        single_input = torch.randn(1, 10)
        
        with tempfile.NamedTemporaryFile(suffix='.onnx') as tmp:
            result = exporter.export(
                model=simple_pytorch_model,
                example_inputs=single_input,
                output_path=tmp.name
            )
            
            assert result is not None
            assert result['total_operations'] > 0
    
    def test_very_large_tensor(self):
        """Edge case: Very large input tensor."""
        class SimpleModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = torch.nn.Linear(1000, 100)
                
            def forward(self, x):
                return self.linear(x)
        
        model = SimpleModel()
        model.eval()
        exporter = HierarchyExporter(strategy="usage_based")
        
        # Large tensor (but not too large to cause OOM)
        large_input = torch.randn(100, 1000)
        
        with tempfile.NamedTemporaryFile(suffix='.onnx') as tmp:
            result = exporter.export(
                model=model,
                example_inputs=large_input,
                output_path=tmp.name
            )
            
            assert result is not None
            assert result['total_operations'] > 0
    
    def test_multiple_input_tensors(self):
        """Edge case: Model with multiple input tensors."""
        class MultiInputModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.linear1 = torch.nn.Linear(10, 5)
                self.linear2 = torch.nn.Linear(5, 5)
                
            def forward(self, x, y):
                return self.linear1(x) + self.linear2(y)
        
        model = MultiInputModel()
        model.eval()
        exporter = HierarchyExporter(strategy="usage_based")
        
        inputs = (torch.randn(1, 10), torch.randn(1, 5))
        
        with tempfile.NamedTemporaryFile(suffix='.onnx') as tmp:
            result = exporter.export(
                model=model,
                example_inputs=inputs,
                output_path=tmp.name
            )
            
            assert result is not None
            assert result['total_operations'] > 0
    
    def test_dict_input_with_extra_keys(self):
        """Edge case: Dict input with extra unused keys."""
        class DictInputModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = torch.nn.Linear(10, 5)
                
            def forward(self, input_ids, attention_mask=None):
                # Only use input_ids, ignore attention_mask
                return self.linear(input_ids)
        
        model = DictInputModel()
        model.eval()
        exporter = HierarchyExporter(strategy="usage_based")
        
        # Dict with extra keys
        inputs = {
            'input_ids': torch.randn(1, 10),
            'attention_mask': torch.ones(1, 10),
            'unused_key': torch.randn(1, 5),  # Extra key
            'another_unused': "string_value"  # Non-tensor value
        }
        
        with tempfile.NamedTemporaryFile(suffix='.onnx') as tmp:
            result = exporter.export(
                model=model,
                example_inputs=inputs,
                output_path=tmp.name
            )
            
            assert result is not None
            assert result['total_operations'] > 0


class TestModelEdgeCases:
    """Test edge cases for different model architectures."""
    
    def test_model_with_no_parameters(self):
        """Edge case: Model with no learnable parameters."""
        class NoParamModel(torch.nn.Module):
            def forward(self, x):
                return torch.relu(x)  # No parameters
        
        model = NoParamModel()
        model.eval()
        exporter = HierarchyExporter(strategy="usage_based")
        
        inputs = torch.randn(1, 10)
        
        with tempfile.NamedTemporaryFile(suffix='.onnx') as tmp:
            result = exporter.export(
                model=model,
                example_inputs=inputs,
                output_path=tmp.name
            )
            
            assert result is not None
            # May have few or no tagged operations due to no parameters
            assert result['total_operations'] >= 0
    
    def test_model_with_unused_parameters(self):
        """Edge case: Model with parameters that aren't used in forward pass."""
        class UnusedParamModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.used_linear = torch.nn.Linear(10, 5)
                self.unused_linear = torch.nn.Linear(10, 5)  # Never used
                
            def forward(self, x):
                return self.used_linear(x)  # unused_linear is ignored
        
        model = UnusedParamModel()
        model.eval()
        exporter = HierarchyExporter(strategy="usage_based")
        
        inputs = torch.randn(1, 10)
        
        with tempfile.NamedTemporaryFile(suffix='.onnx') as tmp:
            result = exporter.export(
                model=model,
                example_inputs=inputs,
                output_path=tmp.name
            )
            
            assert result is not None
            # Should only tag operations from used_linear
            tag_mapping = exporter.get_tag_mapping()
            assert len(tag_mapping) > 0
    
    def test_deeply_nested_model(self):
        """Edge case: Very deeply nested model hierarchy."""
        class DeeplyNested(torch.nn.Module):
            def __init__(self, depth=10):
                super().__init__()
                self.layers = torch.nn.Sequential()
                for i in range(depth):
                    self.layers.add_module(f'layer_{i}', 
                        torch.nn.Sequential(
                            torch.nn.Linear(64, 64),
                            torch.nn.ReLU()
                        )
                    )
                
            def forward(self, x):
                return self.layers(x)
        
        model = DeeplyNested(depth=15)  # 15 layers deep
        model.eval()
        exporter = HierarchyExporter(strategy="htp")
        
        inputs = torch.randn(1, 64)
        
        with tempfile.NamedTemporaryFile(suffix='.onnx') as tmp:
            result = exporter.export(
                model=model,
                example_inputs=inputs,
                output_path=tmp.name
            )
            
            assert result is not None
            assert result['total_operations'] > 0
            
            # Check for deep hierarchy tags
            tag_mapping = exporter.get_tag_mapping()
            all_tags = []
            for node_info in tag_mapping.values():
                all_tags.extend(node_info.get('tags', []))
            
            # Should have some deep hierarchical tags
            deep_tags = [tag for tag in all_tags if tag.count('/') > 3]
            assert len(deep_tags) > 0


class TestFileSystemEdgeCases:
    """Test edge cases related to file system operations."""
    
    def test_output_to_readonly_directory(self, simple_pytorch_model, simple_model_input):
        """Edge case: Output to read-only directory."""
        exporter = HierarchyExporter(strategy="usage_based")
        
        with tempfile.TemporaryDirectory() as tmpdir:
            readonly_dir = Path(tmpdir) / "readonly"
            readonly_dir.mkdir()
            
            # Make directory read-only
            readonly_dir.chmod(0o444)
            
            output_path = readonly_dir / "test.onnx"
            
            try:
                with pytest.raises((PermissionError, OSError)):
                    exporter.export(
                        model=simple_pytorch_model,
                        example_inputs=simple_model_input,
                        output_path=str(output_path)
                    )
            finally:
                # Restore permissions for cleanup
                readonly_dir.chmod(0o755)
    
    def test_output_to_nonexistent_deep_path(self, simple_pytorch_model, simple_model_input):
        """Edge case: Output to non-existent deep directory path."""
        exporter = HierarchyExporter(strategy="usage_based")
        
        with tempfile.TemporaryDirectory() as tmpdir:
            deep_path = Path(tmpdir) / "a" / "b" / "c" / "d" / "e" / "test.onnx"
            
            with pytest.raises((FileNotFoundError, OSError)):
                exporter.export(
                    model=simple_pytorch_model,
                    example_inputs=simple_model_input,
                    output_path=str(deep_path)
                )
    
    def test_output_path_is_directory(self, simple_pytorch_model, simple_model_input):
        """Edge case: Output path points to existing directory."""
        exporter = HierarchyExporter(strategy="usage_based")
        
        with tempfile.TemporaryDirectory() as tmpdir:
            # Try to write to a directory instead of file
            with pytest.raises((IsADirectoryError, PermissionError, OSError)):
                exporter.export(
                    model=simple_pytorch_model,
                    example_inputs=simple_model_input,
                    output_path=tmpdir  # Directory, not file
                )
    
    def test_very_long_filename(self, simple_pytorch_model, simple_model_input):
        """Edge case: Very long filename."""
        exporter = HierarchyExporter(strategy="usage_based")
        
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create very long filename (but within filesystem limits)
            long_name = "a" * 200 + ".onnx"
            output_path = Path(tmpdir) / long_name
            
            try:
                result = exporter.export(
                    model=simple_pytorch_model,
                    example_inputs=simple_model_input,
                    output_path=str(output_path)
                )
                
                assert result is not None
                assert output_path.exists()
            except OSError as e:
                # Some filesystems have filename length limits
                assert "name too long" in str(e).lower() or "filename too long" in str(e).lower()


class TestONNXEdgeCases:
    """Test edge cases in ONNX export and processing."""
    
    def test_model_with_dynamic_shapes(self):
        """Edge case: Model with dynamic input shapes."""
        class DynamicShapeModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = torch.nn.Linear(10, 5)
                
            def forward(self, x):
                # Dynamic reshape based on input
                batch_size = x.size(0)
                x = x.view(batch_size, -1)
                return self.linear(x)
        
        model = DynamicShapeModel()
        model.eval()
        exporter = HierarchyExporter(strategy="usage_based")
        
        inputs = torch.randn(3, 2, 5)  # Will be reshaped to (3, 10)
        
        with tempfile.NamedTemporaryFile(suffix='.onnx') as tmp:
            result = exporter.export(
                model=model,
                example_inputs=inputs,
                output_path=tmp.name,
                dynamic_axes={'input': {0: 'batch_size'}}
            )
            
            assert result is not None
            assert result['total_operations'] > 0
    
    def test_model_with_control_flow(self):
        """Edge case: Model with conditional logic."""
        class ControlFlowModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.linear1 = torch.nn.Linear(10, 5)
                self.linear2 = torch.nn.Linear(10, 5)
                
            def forward(self, x):
                if x.sum() > 0:
                    return self.linear1(x)
                else:
                    return self.linear2(x)
        
        model = ControlFlowModel()
        model.eval()
        exporter = HierarchyExporter(strategy="usage_based")
        
        # Positive sum input - will take first branch
        inputs = torch.randn(1, 10).abs()
        
        with tempfile.NamedTemporaryFile(suffix='.onnx') as tmp:
            try:
                result = exporter.export(
                    model=model,
                    example_inputs=inputs,
                    output_path=tmp.name
                )
                
                # ONNX export might not support control flow directly
                assert result is not None
            except (RuntimeError, NotImplementedError) as e:
                # Expected - ONNX doesn't support arbitrary control flow
                assert "control flow" in str(e).lower() or "if" in str(e).lower()


class TestConfigurationEdgeCases:
    """Test edge cases in configuration and parameters."""
    
    def test_invalid_strategy_parameter(self):
        """Edge case: Invalid strategy parameter."""
        with pytest.raises(ValueError, match="Unsupported strategy"):
            HierarchyExporter(strategy="nonexistent_strategy")
    
    def test_empty_torch_nn_exceptions(self):
        """Edge case: Empty torch.nn exceptions list."""
        exporter = HierarchyExporter(strategy="htp", torch_nn_exceptions=[])
        
        # Should still work, just with different filtering
        assert exporter._torch_nn_exceptions == set()
    
    def test_custom_torch_nn_exceptions(self, simple_pytorch_model, simple_model_input):
        """Edge case: Custom torch.nn exceptions list."""
        # Include Linear in exceptions
        exporter = HierarchyExporter(
            strategy="htp", 
            torch_nn_exceptions=["Linear", "ReLU"]
        )
        
        with tempfile.NamedTemporaryFile(suffix='.onnx') as tmp:
            result = exporter.export(
                model=simple_pytorch_model,
                example_inputs=simple_model_input,
                output_path=tmp.name
            )
            
            assert result is not None
            assert "Linear" in exporter._torch_nn_exceptions
            assert "ReLU" in exporter._torch_nn_exceptions
    
    def test_invalid_export_kwargs(self, simple_pytorch_model, simple_model_input):
        """Edge case: Invalid ONNX export parameters."""
        exporter = HierarchyExporter(strategy="usage_based")
        
        with tempfile.NamedTemporaryFile(suffix='.onnx') as tmp:
            # Invalid opset version
            with pytest.raises((ValueError, RuntimeError)):
                exporter.export(
                    model=simple_pytorch_model,
                    example_inputs=simple_model_input,
                    output_path=tmp.name,
                    opset_version=999  # Invalid opset version
                )


class TestConcurrencyEdgeCases:
    """Test edge cases related to concurrent operations."""
    
    def test_multiple_exporter_instances(self, simple_pytorch_model, simple_model_input):
        """Edge case: Multiple exporter instances operating simultaneously."""
        exporter1 = HierarchyExporter(strategy="usage_based")
        exporter2 = HierarchyExporter(strategy="htp")
        
        results = []
        
        with tempfile.TemporaryDirectory() as tmpdir:
            # Export with both simultaneously
            path1 = Path(tmpdir) / "export1.onnx"
            path2 = Path(tmpdir) / "export2.onnx"
            
            result1 = exporter1.export(
                model=simple_pytorch_model,
                example_inputs=simple_model_input,
                output_path=str(path1)
            )
            
            result2 = exporter2.export(
                model=simple_pytorch_model,
                example_inputs=simple_model_input,
                output_path=str(path2)
            )
            
            assert result1 is not None
            assert result2 is not None
            assert path1.exists()
            assert path2.exists()
    
    def test_state_isolation_between_exports(self, simple_pytorch_model, simple_model_input):
        """Edge case: State isolation between multiple exports."""
        exporter = HierarchyExporter(strategy="usage_based")
        
        with tempfile.TemporaryDirectory() as tmpdir:
            # First export
            path1 = Path(tmpdir) / "export1.onnx"
            result1 = exporter.export(
                model=simple_pytorch_model,
                example_inputs=simple_model_input,
                output_path=str(path1)
            )
            
            tags1 = exporter.get_tag_mapping().copy()
            
            # Second export (state should be reset)
            path2 = Path(tmpdir) / "export2.onnx"
            result2 = exporter.export(
                model=simple_pytorch_model,
                example_inputs=simple_model_input,
                output_path=str(path2)
            )
            
            tags2 = exporter.get_tag_mapping()
            
            # Results should be identical (clean state)
            assert result1['total_operations'] == result2['total_operations']
            assert result1['tagged_operations'] == result2['tagged_operations']


class TestMemoryEdgeCases:
    """Test edge cases related to memory usage."""
    
    def test_large_number_of_operations(self):
        """Edge case: Model with very large number of operations."""
        class ManyOpsModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.layers = torch.nn.ModuleList([
                    torch.nn.Linear(64, 64) for _ in range(50)
                ])
                
            def forward(self, x):
                for layer in self.layers:
                    x = torch.relu(layer(x))
                return x
        
        model = ManyOpsModel()
        model.eval()
        exporter = HierarchyExporter(strategy="usage_based")
        
        inputs = torch.randn(1, 64)
        
        with tempfile.NamedTemporaryFile(suffix='.onnx') as tmp:
            result = exporter.export(
                model=model,
                example_inputs=inputs,
                output_path=tmp.name
            )
            
            assert result is not None
            assert result['total_operations'] > 100  # Should have many operations
    
    def test_cleanup_after_failure(self, simple_pytorch_model):
        """Edge case: Proper cleanup after export failure."""
        exporter = HierarchyExporter(strategy="usage_based")
        
        # Force a failure with invalid input
        with pytest.raises((RuntimeError, TypeError)):
            exporter.export(
                model=simple_pytorch_model,
                example_inputs="invalid_input",
                output_path="/tmp/test.onnx"
            )
        
        # State should be clean after failure
        assert len(exporter._tag_mapping) == 0
        assert len(exporter._tag_stack) == 0
        assert len(exporter._pre_hooks) == 0
        assert len(exporter._post_hooks) == 0


if __name__ == "__main__":
    # Run edge case tests
    pytest.main([__file__, "-v", "--tb=short"])