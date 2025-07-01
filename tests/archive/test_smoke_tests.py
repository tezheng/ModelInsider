"""
Smoke Tests - Round 1: Basic functionality validation

These tests provide quick validation that core functionality works
without diving into complex scenarios. Essential for CI/CD pipelines.
"""

import pytest
import torch
import tempfile
import json
from pathlib import Path

from modelexport.hierarchy_exporter import HierarchyExporter, OperationConfig


class TestCoreSmoke:
    """Core smoke tests for basic functionality."""
    
    def test_hierarchy_exporter_import(self):
        """Smoke test: Can import HierarchyExporter."""
        assert HierarchyExporter is not None
        
    def test_hierarchy_exporter_init_usage_based(self):
        """Smoke test: Can initialize with usage_based strategy."""
        exporter = HierarchyExporter(strategy="usage_based")
        assert exporter.strategy == "usage_based"
        
    def test_hierarchy_exporter_init_htp(self):
        """Smoke test: Can initialize with HTP strategy."""
        exporter = HierarchyExporter(strategy="htp")
        assert exporter.strategy == "htp"
        
    def test_hierarchy_exporter_invalid_strategy(self):
        """Smoke test: Rejects invalid strategy."""
        with pytest.raises(ValueError, match="Unsupported strategy"):
            HierarchyExporter(strategy="invalid_strategy")


class TestOperationConfigSmoke:
    """Smoke tests for OperationConfig functionality."""
    
    def test_operation_config_import(self):
        """Smoke test: Can import OperationConfig."""
        assert OperationConfig is not None
        
    def test_get_operations_to_patch(self):
        """Smoke test: Can get operations to patch."""
        ops = OperationConfig.get_operations_to_patch()
        assert isinstance(ops, list)
        assert len(ops) > 0
        
        # Check first operation has correct format
        assert len(ops[0]) == 2  # (module, op_name)
        
    def test_get_torch_to_onnx_mapping(self):
        """Smoke test: Can get ONNX mapping."""
        mapping = OperationConfig.get_torch_to_onnx_mapping()
        assert isinstance(mapping, dict)
        assert len(mapping) > 0
        
        # Check some basic operations exist
        assert 'matmul' in mapping
        assert 'add' in mapping
        
    def test_add_operation(self):
        """Smoke test: Can add new operation."""
        original_count = len(OperationConfig.OPERATION_REGISTRY)
        
        OperationConfig.add_operation(
            'test_smoke_op',
            [('torch', 'test_op')],
            ['TestOp'],
            priority=5
        )
        
        new_count = len(OperationConfig.OPERATION_REGISTRY)
        assert new_count == original_count + 1
        assert 'test_smoke_op' in OperationConfig.OPERATION_REGISTRY


class TestBasicExportSmoke:
    """Smoke tests for basic export functionality."""
    
    def test_simple_model_export_usage_based(self, simple_pytorch_model, simple_model_input):
        """Smoke test: Can export simple model with usage_based strategy."""
        exporter = HierarchyExporter(strategy="usage_based")
        
        with tempfile.NamedTemporaryFile(suffix='.onnx') as tmp:
            result = exporter.export(
                model=simple_pytorch_model,
                example_inputs=simple_model_input,
                output_path=tmp.name
            )
            
            assert result is not None
            assert 'output_path' in result
            assert 'strategy' in result
            assert result['strategy'] == 'usage_based'
            
            # Check ONNX file was created
            assert Path(tmp.name).exists()
            
    def test_simple_model_export_htp(self, simple_pytorch_model, simple_model_input):
        """Smoke test: Can export simple model with HTP strategy."""
        exporter = HierarchyExporter(strategy="htp")
        
        with tempfile.NamedTemporaryFile(suffix='.onnx') as tmp:
            result = exporter.export(
                model=simple_pytorch_model,
                example_inputs=simple_model_input,
                output_path=tmp.name
            )
            
            assert result is not None
            assert 'output_path' in result
            assert 'strategy' in result
            assert result['strategy'] == 'htp'
            
            # Check ONNX file was created
            assert Path(tmp.name).exists()
            
    def test_tag_mapping_populated(self, simple_pytorch_model, simple_model_input):
        """Smoke test: Export populates tag mapping."""
        exporter = HierarchyExporter(strategy="usage_based")
        
        with tempfile.NamedTemporaryFile(suffix='.onnx') as tmp:
            exporter.export(
                model=simple_pytorch_model,
                example_inputs=simple_model_input,
                output_path=tmp.name
            )
            
            tag_mapping = exporter.get_tag_mapping()
            assert isinstance(tag_mapping, dict)
            assert len(tag_mapping) > 0
            
            # Check some operations have tags
            tagged_ops = [op for op in tag_mapping.values() if op.get('tags', [])]
            assert len(tagged_ops) > 0


class TestUnifiedTagBuildingSmoke:
    """Smoke tests for unified tag building functionality."""
    
    def test_build_tag_usage_based_mode(self, simple_pytorch_model):
        """Smoke test: Unified tag building works in usage_based mode."""
        exporter = HierarchyExporter(strategy="usage_based")
        exporter._model = simple_pytorch_model
        
        tag = exporter._build_tag("linear1", simple_pytorch_model.linear1)
        assert isinstance(tag, str)
        assert tag.startswith("/")
        # In usage_based mode, torch.nn modules are filtered out
        assert "SimpleTestModel" in tag
        
    def test_build_tag_htp_mode(self, simple_pytorch_model):
        """Smoke test: Unified tag building works in HTP mode."""
        exporter = HierarchyExporter(strategy="htp")
        exporter._model = simple_pytorch_model
        
        tag = exporter._build_tag("linear1", simple_pytorch_model.linear1)
        assert isinstance(tag, str)
        assert tag.startswith("/")
        # In HTP mode, Linear is filtered out (not in exceptions), so we get SimpleTestModel
        assert "SimpleTestModel" in tag
        
    def test_build_tag_preserves_instances(self, simple_pytorch_model):
        """Smoke test: Tag building can preserve instance information."""
        exporter = HierarchyExporter(strategy="htp")
        exporter._model = simple_pytorch_model
        
        # Test explicit instance preservation
        tag_with_instances = exporter._build_tag("linear1", simple_pytorch_model.linear1, preserve_instances=True)
        tag_without_instances = exporter._build_tag("linear1", simple_pytorch_model.linear1, preserve_instances=False)
        
        assert isinstance(tag_with_instances, str)
        assert isinstance(tag_without_instances, str)
        assert tag_with_instances.startswith("/")
        assert tag_without_instances.startswith("/")


class TestRefactoredCodeSmoke:
    """Smoke tests specifically for refactored code paths."""
    
    def test_operation_registry_consistency(self):
        """Smoke test: Operation registry is internally consistent."""
        registry = OperationConfig.OPERATION_REGISTRY
        
        # Check all entries have required fields
        for op_name, op_data in registry.items():
            assert 'patch_targets' in op_data
            assert 'onnx_types' in op_data
            assert 'priority' in op_data
            
            assert isinstance(op_data['patch_targets'], list)
            assert isinstance(op_data['onnx_types'], list)
            assert isinstance(op_data['priority'], int)
            
    def test_operations_to_patch_valid(self):
        """Smoke test: All patch targets resolve to valid modules."""
        ops = OperationConfig.get_operations_to_patch()
        
        for module, op_name in ops:
            # Module should have the operation
            assert hasattr(module, op_name), f"Module {module} should have operation {op_name}"
            
    def test_both_strategies_produce_valid_results(self, simple_pytorch_model, simple_model_input):
        """Smoke test: Both strategies produce valid, different results."""
        results = {}
        
        for strategy in ['usage_based', 'htp']:
            exporter = HierarchyExporter(strategy=strategy)
            
            with tempfile.NamedTemporaryFile(suffix='.onnx') as tmp:
                result = exporter.export(
                    model=simple_pytorch_model,
                    example_inputs=simple_model_input,
                    output_path=tmp.name
                )
                
                results[strategy] = {
                    'total_operations': result['total_operations'],
                    'tagged_operations': result['tagged_operations'],
                    'tag_mapping': exporter.get_tag_mapping()
                }
        
        # Both should have operations
        assert results['usage_based']['total_operations'] > 0
        assert results['htp']['total_operations'] > 0
        
        # Total operations should be the same (same model/ONNX topology)
        assert results['usage_based']['total_operations'] == results['htp']['total_operations']
        
        # HTP typically tags more operations
        assert results['htp']['tagged_operations'] >= results['usage_based']['tagged_operations']


class TestErrorHandlingSmoke:
    """Smoke tests for basic error handling."""
    
    def test_export_invalid_model(self):
        """Smoke test: Graceful handling of invalid model."""
        exporter = HierarchyExporter(strategy="usage_based")
        
        with pytest.raises((TypeError, AttributeError)):
            exporter.export(
                model="not_a_model",  # Invalid model
                example_inputs=torch.randn(1, 10),
                output_path="/tmp/test.onnx"
            )
            
    def test_export_invalid_input(self, simple_pytorch_model):
        """Smoke test: Graceful handling of invalid input."""
        exporter = HierarchyExporter(strategy="usage_based")
        
        with pytest.raises((RuntimeError, TypeError)):
            exporter.export(
                model=simple_pytorch_model,
                example_inputs="not_a_tensor",  # Invalid input
                output_path="/tmp/test.onnx"
            )
            
    def test_export_invalid_path(self, simple_pytorch_model, simple_model_input):
        """Smoke test: Graceful handling of invalid output path."""
        exporter = HierarchyExporter(strategy="usage_based")
        
        with pytest.raises((FileNotFoundError, PermissionError, OSError)):
            exporter.export(
                model=simple_pytorch_model,
                example_inputs=simple_model_input,
                output_path="/invalid/path/test.onnx"  # Invalid path
            )


if __name__ == "__main__":
    # Run smoke tests quickly
    pytest.main([__file__, "-v", "--tb=short"])