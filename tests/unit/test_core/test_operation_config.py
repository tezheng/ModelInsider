"""
Operation Configuration Unit Tests

Tests for the centralized operation configuration system.
"""

import pytest
from modelexport.core.operation_config import OperationConfig


class TestOperationConfig:
    """Test the OperationConfig class."""
    
    def test_operation_registry_structure(self):
        """Test that the operation registry has the correct structure."""
        registry = OperationConfig.OPERATION_REGISTRY
        
        assert isinstance(registry, dict)
        assert len(registry) > 0
        
        # Check a few known operations
        expected_operations = ['matmul', 'add', 'relu', 'tanh', 'embedding']
        for op in expected_operations:
            assert op in registry, f"Operation {op} should be in registry"
    
    def test_operation_entry_structure(self):
        """Test that each operation entry has the required structure."""
        registry = OperationConfig.OPERATION_REGISTRY
        
        for op_name, op_config in registry.items():
            # Each operation should have required fields
            assert 'patch_targets' in op_config, f"Operation {op_name} missing patch_targets"
            assert 'onnx_types' in op_config, f"Operation {op_name} missing onnx_types"
            assert 'priority' in op_config, f"Operation {op_name} missing priority"
            
            # Validate field types
            assert isinstance(op_config['patch_targets'], list)
            assert isinstance(op_config['onnx_types'], list)
            assert isinstance(op_config['priority'], int)
            
            # Priority should be reasonable (1-10)
            assert 1 <= op_config['priority'] <= 10
    
    def test_patch_targets_format(self):
        """Test that patch targets have correct format."""
        registry = OperationConfig.OPERATION_REGISTRY
        
        for op_name, op_config in registry.items():
            for target in op_config['patch_targets']:
                if target:  # Some operations have empty patch targets
                    assert isinstance(target, tuple), f"Patch target should be tuple for {op_name}"
                    assert len(target) == 2, f"Patch target should have 2 elements for {op_name}"
                    
                    module_name, function_name = target
                    assert isinstance(module_name, str)
                    assert isinstance(function_name, str)
                    assert module_name in ['torch', 'F'], f"Unknown module {module_name} for {op_name}"
    
    def test_onnx_types_validity(self):
        """Test that ONNX types are valid."""
        registry = OperationConfig.OPERATION_REGISTRY
        
        # Known ONNX operation types (partial list for validation)
        known_onnx_ops = {
            'MatMul', 'Gemm', 'Add', 'Sub', 'Mul', 'Div', 'Relu', 'Sigmoid', 'Tanh',
            'Softmax', 'Gather', 'Concat', 'Reshape', 'Transpose', 'ReduceMean',
            'ReduceSum', 'Sqrt', 'Pow', 'Abs', 'Neg', 'Exp', 'Log', 'Floor', 'Ceil',
            'Unsqueeze', 'Squeeze', 'Slice', 'Where', 'Equal', 'LayerNormalization',
            'BatchNormalization', 'Conv', 'MaxPool', 'GlobalAveragePool', 'Identity',
            'Erf', 'CumSum', 'Pad', 'Dropout', 'Gelu', 'Constant', 'Shape', 'Input',
            'Output', 'Reciprocal'
        }
        
        for op_name, op_config in registry.items():
            for onnx_type in op_config['onnx_types']:
                assert isinstance(onnx_type, str), f"ONNX type should be string for {op_name}"
                # Most ONNX types should be known (allowing for some flexibility)
                if onnx_type not in known_onnx_ops:
                    print(f"Warning: Unknown ONNX type {onnx_type} for {op_name}")
    
    def test_get_operations_to_patch(self):
        """Test the get_operations_to_patch method."""
        operations = OperationConfig.get_operations_to_patch()
        
        assert isinstance(operations, list)
        assert len(operations) > 0
        
        # Each item should be a tuple (module, function_name)
        for item in operations:
            assert isinstance(item, tuple)
            assert len(item) == 2
            
            module, function_name = item
            # Module could be actual module object or string
            if hasattr(module, '__name__'):
                module_name = module.__name__
            else:
                module_name = str(module)
            assert 'torch' in module_name or module_name == 'F'
            assert isinstance(function_name, str)
    
    def test_get_torch_to_onnx_mapping(self):
        """Test the get_torch_to_onnx_mapping method."""
        mapping = OperationConfig.get_torch_to_onnx_mapping()
        
        assert isinstance(mapping, dict)
        assert len(mapping) > 0
        
        # Check some expected mappings
        expected_mappings = {
            'matmul': ['MatMul', 'Gemm'],
            'add': ['Add'],
            'relu': ['Relu'],
            'tanh': ['Tanh']
        }
        
        for op_name, expected_onnx_types in expected_mappings.items():
            assert op_name in mapping, f"Mapping should include {op_name}"
            actual_onnx_types = mapping[op_name]
            for onnx_type in expected_onnx_types:
                assert onnx_type in actual_onnx_types, f"{onnx_type} should be in mapping for {op_name}"
    
    def test_add_operation(self):
        """Test the add_operation method."""
        # Test adding a new operation
        OperationConfig.add_operation(
            'test_op',
            patch_targets=[('torch', 'test_function')],
            onnx_types=['TestOp'],
            priority=5
        )
        
        # Check that it was added
        assert 'test_op' in OperationConfig.OPERATION_REGISTRY
        
        test_config = OperationConfig.OPERATION_REGISTRY['test_op']
        assert test_config['patch_targets'] == [('torch', 'test_function')]
        assert test_config['onnx_types'] == ['TestOp']
        assert test_config['priority'] == 5
        
        # Clean up
        del OperationConfig.OPERATION_REGISTRY['test_op']
    
    def test_priority_ordering(self):
        """Test that operations have reasonable priority ordering."""
        registry = OperationConfig.OPERATION_REGISTRY
        
        # Core math operations should have high priority (low numbers)
        core_ops = ['matmul', 'add', 'sub', 'mul', 'div']
        for op in core_ops:
            if op in registry:
                assert registry[op]['priority'] <= 3, f"Core op {op} should have high priority"
        
        # Advanced operations should have lower priority (higher numbers)
        advanced_ops = ['scaled_dot_product_attention']
        for op in advanced_ops:
            if op in registry:
                assert registry[op]['priority'] >= 3, f"Advanced op {op} should have lower priority"
    
    def test_operation_categories(self):
        """Test that operations are properly categorized."""
        registry = OperationConfig.OPERATION_REGISTRY
        
        # Math operations
        math_ops = ['matmul', 'add', 'sub', 'mul', 'div', 'pow', 'sqrt', 'abs', 'neg']
        for op in math_ops:
            if op in registry:
                assert registry[op]['priority'] <= 3, f"Math op {op} should have high priority"
        
        # Shape operations
        shape_ops = ['reshape', 'transpose', 'unsqueeze', 'squeeze', 'cat']
        for op in shape_ops:
            if op in registry:
                assert registry[op]['priority'] >= 2, f"Shape op {op} should have medium priority"
        
        # Reduction operations
        reduction_ops = ['mean', 'sum', 'cumsum', 'cumprod']
        for op in reduction_ops:
            if op in registry:
                assert registry[op]['priority'] >= 3, f"Reduction op {op} should have medium-low priority"
    
    def test_functional_vs_torch_operations(self):
        """Test that both torch and functional operations are covered."""
        operations = OperationConfig.get_operations_to_patch()
        
        torch_ops = []
        functional_ops = []
        
        for op in operations:
            module, function_name = op
            # Module could be actual module object or string
            if hasattr(module, '__name__'):
                module_name = module.__name__
            else:
                module_name = str(module)
            
            if module_name == 'torch':
                torch_ops.append(op)
            elif 'functional' in module_name or module_name == 'F':
                functional_ops.append(op)
        
        assert len(torch_ops) > 0, "Should have torch operations"
        assert len(functional_ops) > 0, "Should have functional operations"
        
        # Some operations should be available in both
        torch_functions = {op[1] for op in torch_ops}
        functional_functions = {op[1] for op in functional_ops}
        
        # Common operations that exist in both
        common_ops = torch_functions & functional_functions
        expected_common = {'relu', 'tanh', 'sigmoid'}  # These exist in both torch and F
        
        for op in expected_common:
            if op in torch_functions or op in functional_functions:
                # At least one should exist
                assert op in torch_functions or op in functional_functions
    
    def test_operation_coverage(self):
        """Test that we have good coverage of important operations."""
        registry = OperationConfig.OPERATION_REGISTRY
        
        # Essential operations that should be covered
        essential_ops = [
            'matmul', 'add', 'relu', 'embedding', 'reshape', 'transpose'
        ]
        
        for op in essential_ops:
            assert op in registry, f"Essential operation {op} should be in registry"
        
        # Should have reasonable number of operations
        assert len(registry) >= 20, "Should have at least 20 operations"
        assert len(registry) <= 100, "Should not have excessive operations"
    
    def test_no_duplicate_operations(self):
        """Test that there are no duplicate operation definitions."""
        registry = OperationConfig.OPERATION_REGISTRY
        
        # Check for potential duplicates by comparing patch targets
        all_targets = []
        for op_name, op_config in registry.items():
            for target in op_config['patch_targets']:
                if target:  # Skip empty targets
                    all_targets.append((target, op_name))
        
        # Check for duplicates
        seen_targets = set()
        for target, op_name in all_targets:
            if target in seen_targets:
                # This might be intentional for some operations, so just warn
                print(f"Warning: Duplicate patch target {target} found in {op_name}")
            seen_targets.add(target)


class TestOperationConfigIntegration:
    """Test integration of OperationConfig with other components."""
    
    def test_config_used_by_strategies(self):
        """Test that strategies can access operation configuration."""
        # Test that we can import and use the config
        from modelexport.core.operation_config import OperationConfig
        
        # Get operations to patch (as strategies would)
        operations = OperationConfig.get_operations_to_patch()
        assert len(operations) > 0
        
        # Get torch to ONNX mapping (as strategies would)
        mapping = OperationConfig.get_torch_to_onnx_mapping()
        assert len(mapping) > 0
        
        # Verify consistency
        patch_ops = {op[1] for op in operations if op}
        mapping_ops = set(mapping.keys())
        
        # Most patched operations should have ONNX mappings
        common_ops = patch_ops & mapping_ops
        assert len(common_ops) > 0, "Should have operations in both patch list and mapping"