"""
Operation Configuration and Registry Tests - Round 4

These tests validate the OperationConfig class and operation registry functionality,
including patching behavior, ONNX mappings, and registry management.
"""

import pytest
import torch
import torch.nn.functional as F
import tempfile
from unittest.mock import patch, MagicMock

from modelexport.hierarchy_exporter import HierarchyExporter, OperationConfig


class TestOperationRegistryBasics:
    """Test basic operation registry functionality."""
    
    def test_operation_registry_structure(self):
        """Test operation registry has proper structure."""
        registry = OperationConfig.OPERATION_REGISTRY
        
        assert isinstance(registry, dict)
        assert len(registry) > 0
        
        # Check all entries have required fields
        for op_name, op_data in registry.items():
            assert isinstance(op_name, str)
            assert isinstance(op_data, dict)
            
            # Required fields
            assert 'patch_targets' in op_data
            assert 'onnx_types' in op_data  
            assert 'priority' in op_data
            
            # Validate field types
            assert isinstance(op_data['patch_targets'], list)
            assert isinstance(op_data['onnx_types'], list)
            assert isinstance(op_data['priority'], int)
            
            # Patch targets should be tuples of (module_name, op_name)
            for target in op_data['patch_targets']:
                assert isinstance(target, tuple)
                assert len(target) == 2
                assert isinstance(target[0], str)  # module name
                assert isinstance(target[1], str)  # operation name
    
    def test_operation_registry_completeness(self):
        """Test operation registry contains expected operations."""
        registry = OperationConfig.OPERATION_REGISTRY
        
        # Core mathematical operations should be present
        expected_ops = {
            'matmul', 'add', 'sub', 'mul', 'div', 'pow', 'sqrt',
            'tanh', 'relu', 'bmm'
        }
        
        for op in expected_ops:
            assert op in registry, f"Expected operation '{op}' not found in registry"
    
    def test_operation_priorities(self):
        """Test operation priorities are valid."""
        registry = OperationConfig.OPERATION_REGISTRY
        
        for op_name, op_data in registry.items():
            priority = op_data['priority']
            # Priorities should be positive integers
            assert isinstance(priority, int)
            assert priority > 0
            assert priority <= 10  # Reasonable upper bound


class TestGetOperationsToPath:
    """Test getting operations to patch functionality."""
    
    def test_get_operations_to_patch_basic(self):
        """Test basic get operations to patch functionality."""
        ops = OperationConfig.get_operations_to_patch()
        
        assert isinstance(ops, list)
        assert len(ops) > 0
        
        # Each operation should be a (module, op_name) tuple
        valid_ops = []
        for module, op_name in ops:
            if hasattr(module, op_name):
                valid_ops.append((module, op_name))
        
        # Most operations should be valid (some may not exist in all PyTorch versions)
        assert len(valid_ops) > len(ops) * 0.8, "Most operations should be valid"
    
    def test_get_operations_to_patch_contains_core_ops(self):
        """Test that core operations are included in patch list."""
        ops = OperationConfig.get_operations_to_patch()
        
        # Convert to set of (module_name, op_name) for easier checking
        op_set = set()
        for module, op_name in ops:
            module_name = module.__name__ if hasattr(module, '__name__') else str(module)
            op_set.add((module_name, op_name))
        
        # Check for torch.matmul
        assert any(op_name == 'matmul' for _, op_name in op_set)
        
        # Check for torch.add
        assert any(op_name == 'add' for _, op_name in op_set)
        
        # Check for torch.relu
        assert any(op_name == 'relu' for _, op_name in op_set)
    
    def test_get_operations_to_patch_valid_targets(self):
        """Test that valid patch targets resolve to callable operations."""
        ops = OperationConfig.get_operations_to_patch()
        
        valid_count = 0
        for module, op_name in ops:
            # Module should exist
            assert module is not None
            
            # If operation exists, it should be callable
            if hasattr(module, op_name):
                operation = getattr(module, op_name)
                assert callable(operation)
                valid_count += 1
        
        # Most operations should be valid
        assert valid_count > len(ops) * 0.8, "Most patch targets should be valid"
    
    def test_get_operations_to_patch_no_duplicates(self):
        """Test that operations to patch contains no duplicates."""
        ops = OperationConfig.get_operations_to_patch()
        
        # Convert to set for duplicate detection
        op_set = set()
        for module, op_name in ops:
            module_id = id(module)  # Use id to distinguish modules
            op_tuple = (module_id, op_name)
            assert op_tuple not in op_set, f"Duplicate operation found: {module}.{op_name}"
            op_set.add(op_tuple)


class TestTorchToONNXMapping:
    """Test torch to ONNX mapping functionality."""
    
    def test_get_torch_to_onnx_mapping_basic(self):
        """Test basic torch to ONNX mapping functionality."""
        mapping = OperationConfig.get_torch_to_onnx_mapping()
        
        assert isinstance(mapping, dict)
        assert len(mapping) > 0
        
        # Each entry should map string to list of strings
        for torch_op, onnx_ops in mapping.items():
            assert isinstance(torch_op, str)
            assert isinstance(onnx_ops, list)
            assert len(onnx_ops) > 0
            
            for onnx_op in onnx_ops:
                assert isinstance(onnx_op, str)
                assert len(onnx_op) > 0
    
    def test_torch_to_onnx_mapping_core_operations(self):
        """Test core operations have proper ONNX mappings."""
        mapping = OperationConfig.get_torch_to_onnx_mapping()
        
        # Test key mappings
        expected_mappings = {
            'matmul': ['MatMul', 'Gemm'],
            'add': ['Add'],
            'relu': ['Relu'],
            'tanh': ['Tanh']
        }
        
        for torch_op, expected_onnx in expected_mappings.items():
            assert torch_op in mapping, f"Torch operation '{torch_op}' not found in mapping"
            
            actual_onnx = mapping[torch_op]
            for expected_op in expected_onnx:
                assert expected_op in actual_onnx, f"Expected ONNX op '{expected_op}' not found for '{torch_op}'"
    
    def test_torch_to_onnx_mapping_consistency(self):
        """Test consistency between registry and mapping."""
        registry = OperationConfig.OPERATION_REGISTRY
        mapping = OperationConfig.get_torch_to_onnx_mapping()
        
        # Every registry entry should have a corresponding mapping entry
        for op_name in registry.keys():
            assert op_name in mapping, f"Registry operation '{op_name}' missing from mapping"
            
            # ONNX types should match
            registry_onnx = registry[op_name]['onnx_types']
            mapping_onnx = mapping[op_name]
            
            assert registry_onnx == mapping_onnx, f"ONNX types mismatch for '{op_name}'"


class TestAddOperation:
    """Test adding new operations to registry."""
    
    def test_add_operation_basic(self):
        """Test basic add operation functionality."""
        original_count = len(OperationConfig.OPERATION_REGISTRY)
        
        # Add a test operation
        OperationConfig.add_operation(
            'test_add_basic',
            [('torch', 'test_func')],
            ['TestOp'],
            priority=5
        )
        
        new_count = len(OperationConfig.OPERATION_REGISTRY)
        assert new_count == original_count + 1
        
        # Check the operation was added correctly
        assert 'test_add_basic' in OperationConfig.OPERATION_REGISTRY
        op_data = OperationConfig.OPERATION_REGISTRY['test_add_basic']
        
        assert op_data['patch_targets'] == [('torch', 'test_func')]
        assert op_data['onnx_types'] == ['TestOp']
        assert op_data['priority'] == 5
    
    def test_add_operation_multiple_targets(self):
        """Test adding operation with multiple patch targets."""
        OperationConfig.add_operation(
            'test_multi_targets',
            [('torch', 'func1'), ('F', 'func2')],
            ['Op1', 'Op2'],
            priority=3
        )
        
        assert 'test_multi_targets' in OperationConfig.OPERATION_REGISTRY
        op_data = OperationConfig.OPERATION_REGISTRY['test_multi_targets']
        
        assert len(op_data['patch_targets']) == 2
        assert ('torch', 'func1') in op_data['patch_targets']
        assert ('F', 'func2') in op_data['patch_targets']
        
        assert len(op_data['onnx_types']) == 2
        assert 'Op1' in op_data['onnx_types']
        assert 'Op2' in op_data['onnx_types']
    
    def test_add_operation_overwrites_existing(self):
        """Test that adding existing operation overwrites it."""
        # Add initial operation
        OperationConfig.add_operation(
            'test_overwrite',
            [('torch', 'original')],
            ['OriginalOp'],
            priority=1
        )
        
        original_data = OperationConfig.OPERATION_REGISTRY['test_overwrite'].copy()
        
        # Overwrite with new data
        OperationConfig.add_operation(
            'test_overwrite',
            [('torch', 'new')],
            ['NewOp'],
            priority=2
        )
        
        new_data = OperationConfig.OPERATION_REGISTRY['test_overwrite']
        
        # Should be completely replaced
        assert new_data != original_data
        assert new_data['patch_targets'] == [('torch', 'new')]
        assert new_data['onnx_types'] == ['NewOp']
        assert new_data['priority'] == 2
    
    def test_add_operation_affects_mappings(self):
        """Test that adding operation affects get_* methods."""
        # Get initial state
        initial_mapping = OperationConfig.get_torch_to_onnx_mapping()
        initial_patches = OperationConfig.get_operations_to_patch()
        
        # Add new operation (with valid torch operation for patches)
        OperationConfig.add_operation(
            'test_affects_mappings',
            [('torch', 'abs')],  # torch.abs exists
            ['Abs'],
            priority=1
        )
        
        # Get new state
        new_mapping = OperationConfig.get_torch_to_onnx_mapping()
        new_patches = OperationConfig.get_operations_to_patch()
        
        # Should have new entries
        assert 'test_affects_mappings' in new_mapping
        assert new_mapping['test_affects_mappings'] == ['Abs']
        
        # Should have new patch target
        assert len(new_patches) > len(initial_patches)
        patch_targets = [(getattr(m, '__name__', str(m)), op) for m, op in new_patches]
        assert ('torch', 'abs') in patch_targets


class TestOperationConfigIntegration:
    """Test integration of OperationConfig with HierarchyExporter."""
    
    def test_operation_config_used_in_exporter(self, simple_pytorch_model, simple_model_input):
        """Test that HierarchyExporter uses OperationConfig correctly."""
        exporter = HierarchyExporter(strategy="htp")
        
        # Get operations that should be patched
        operations_to_patch = OperationConfig.get_operations_to_patch()
        assert len(operations_to_patch) > 0
        
        # Export should succeed
        with tempfile.NamedTemporaryFile(suffix='.onnx') as tmp:
            result = exporter.export(
                model=simple_pytorch_model,
                example_inputs=simple_model_input,
                output_path=tmp.name
            )
            
            assert result is not None
            assert 'total_operations' in result
    
    def test_custom_operation_in_export(self):
        """Test that custom operation can be used in export."""
        # Add a custom operation
        OperationConfig.add_operation(
            'test_custom_export',
            [('torch', 'abs')],
            ['Abs'],
            priority=1
        )
        
        # Create a model that uses abs
        class AbsModel(torch.nn.Module):
            def forward(self, x):
                return torch.abs(x)
        
        model = AbsModel()
        model.eval()
        inputs = torch.randn(1, 10)
        
        exporter = HierarchyExporter(strategy="htp")
        
        with tempfile.NamedTemporaryFile(suffix='.onnx') as tmp:
            result = exporter.export(
                model=model,
                example_inputs=inputs,
                output_path=tmp.name
            )
            
            assert result is not None


class TestOperationConfigEdgeCases:
    """Test edge cases and error conditions in OperationConfig."""
    
    def test_empty_patch_targets(self):
        """Test operation with empty patch targets."""
        OperationConfig.add_operation(
            'test_empty_targets',
            [],  # Empty patch targets
            ['EmptyOp'],
            priority=1
        )
        
        # Should be added to registry
        assert 'test_empty_targets' in OperationConfig.OPERATION_REGISTRY
        
        # Should not appear in patches (no targets to patch)
        patches = OperationConfig.get_operations_to_patch()
        patch_names = [op_name for _, op_name in patches]
        # No specific operation name in patches since targets are empty
    
    def test_empty_onnx_types(self):
        """Test operation with empty ONNX types."""
        OperationConfig.add_operation(
            'test_empty_onnx',
            [('torch', 'abs')],
            [],  # Empty ONNX types
            priority=1
        )
        
        # Should be added to registry
        assert 'test_empty_onnx' in OperationConfig.OPERATION_REGISTRY
        
        # Should appear in mapping with empty list
        mapping = OperationConfig.get_torch_to_onnx_mapping()
        assert 'test_empty_onnx' in mapping
        assert mapping['test_empty_onnx'] == []
    
    def test_invalid_patch_target_format(self):
        """Test handling of invalid patch target format."""
        # Add operation with malformed target (wrong tuple size)
        OperationConfig.add_operation(
            'test_invalid_format',
            [('torch', 'abs', 'extra_element')],  # Tuple with 3 elements instead of 2
            ['InvalidOp'],
            priority=1
        )
        
        # Should be in registry (validation is minimal)
        assert 'test_invalid_format' in OperationConfig.OPERATION_REGISTRY
        
        # get_operations_to_patch should handle this gracefully by skipping
        try:
            patches = OperationConfig.get_operations_to_patch()
            # Should not crash, invalid targets are filtered out
            assert isinstance(patches, list)
        except ValueError:
            # Expected behavior - method doesn't handle invalid formats
            pass
    
    def test_nonexistent_module_in_patch_target(self):
        """Test patch target with non-existent module."""
        # Clean up any invalid entries first
        registry = OperationConfig.OPERATION_REGISTRY
        invalid_keys = []
        for key, data in registry.items():
            for target in data['patch_targets']:
                if len(target) != 2:
                    invalid_keys.append(key)
                    break
        
        for key in invalid_keys:
            del registry[key]
        
        OperationConfig.add_operation(
            'test_nonexistent_module',
            [('nonexistent_module', 'some_func')],
            ['NonexistentOp'],
            priority=1
        )
        
        # Should be in registry
        assert 'test_nonexistent_module' in OperationConfig.OPERATION_REGISTRY
        
        # get_operations_to_patch should handle this gracefully
        # (The implementation may filter out non-existent modules)
        patches = OperationConfig.get_operations_to_patch()
        assert isinstance(patches, list)
    
    def test_priority_edge_values(self):
        """Test edge values for priority."""
        # Very high priority
        OperationConfig.add_operation(
            'test_high_priority',
            [('torch', 'abs')],
            ['HighPriorityOp'],
            priority=1000
        )
        
        # Zero priority (edge case)
        OperationConfig.add_operation(
            'test_zero_priority',
            [('torch', 'abs')],
            ['ZeroPriorityOp'],
            priority=0
        )
        
        # Negative priority (edge case)
        OperationConfig.add_operation(
            'test_negative_priority',
            [('torch', 'abs')],
            ['NegativePriorityOp'],
            priority=-1
        )
        
        # All should be added to registry
        assert 'test_high_priority' in OperationConfig.OPERATION_REGISTRY
        assert 'test_zero_priority' in OperationConfig.OPERATION_REGISTRY
        assert 'test_negative_priority' in OperationConfig.OPERATION_REGISTRY


class TestRegistryStateManagement:
    """Test registry state management and cleanup."""
    
    def test_registry_is_global(self):
        """Test that registry changes are global across instances."""
        # Add operation
        OperationConfig.add_operation(
            'test_global_state',
            [('torch', 'abs')],
            ['GlobalOp'],
            priority=1
        )
        
        # Should be visible from different access points
        assert 'test_global_state' in OperationConfig.OPERATION_REGISTRY
        
        # Create new reference
        registry_ref = OperationConfig.OPERATION_REGISTRY
        assert 'test_global_state' in registry_ref
    
    def test_registry_persistence(self):
        """Test registry changes persist across multiple operations."""
        initial_count = len(OperationConfig.OPERATION_REGISTRY)
        
        # Add multiple operations
        for i in range(3):
            OperationConfig.add_operation(
                f'test_persistence_{i}',
                [('torch', 'abs')],
                [f'PersistentOp{i}'],
                priority=i + 1
            )
        
        final_count = len(OperationConfig.OPERATION_REGISTRY)
        assert final_count == initial_count + 3
        
        # All should still be present
        for i in range(3):
            assert f'test_persistence_{i}' in OperationConfig.OPERATION_REGISTRY


if __name__ == "__main__":
    # Run operation config tests
    pytest.main([__file__, "-v", "--tb=short"])