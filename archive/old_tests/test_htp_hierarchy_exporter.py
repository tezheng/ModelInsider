"""
HTP Hierarchy Exporter Unit Tests

Tests specific to the HTP (Hierarchical Trace-and-Project) export strategy.
"""

from pathlib import Path

import torch

from modelexport.strategies.htp import HTPHierarchyExporter
from tests.fixtures.base_test import BaseStrategyTest, StrategyCompatibilityTest
from tests.fixtures.test_models import TestModelFixtures


class TestHTPHierarchyExporter(BaseStrategyTest, StrategyCompatibilityTest):
    """Test HTP-based hierarchy exporter."""
    
    def get_exporter(self) -> HTPHierarchyExporter:
        """Get HTP exporter instance."""
        # Use HTP strategy specifically
        return HTPHierarchyExporter(strategy='htp')
    
    def get_strategy_name(self) -> str:
        """Get strategy name."""
        return "htp"
    
    def assert_htp_specific_results(self, result):
        """Assert HTP-specific result fields."""
        assert result['strategy'] in ['htp', 'htp_builtin']
        assert 'total_operations' in result
        assert 'tagged_operations' in result
        
        # Should have operation trace information
        if 'operation_trace' in result:
            assert isinstance(result['operation_trace'], (list, int))
    
    def test_htp_basic_export(self):
        """Test basic HTP export functionality."""
        model, inputs = self.fixtures.get_simple_cnn()
        output_path = self.get_output_path("htp_basic")
        
        result = self.exporter.export(model, inputs, output_path)
        
        self.assert_export_success(result)
        self.assert_htp_specific_results(result)
        
        # HTP should handle any model
        assert result['tagged_operations'] > 0
    
    def test_htp_with_control_flow(self):
        """Test HTP with control flow model (FX incompatible)."""
        model, inputs = self.fixtures.get_conditional_model()
        output_path = self.get_output_path("htp_control_flow")
        
        # HTP should handle control flow unlike FX
        result = self.exporter.export(model, inputs, output_path)
        
        self.assert_export_success(result)
        self.assert_htp_specific_results(result)
    
    def test_htp_builtin_strategy(self):
        """Test HTP with built-in module tracking."""
        exporter = HTPHierarchyExporter(strategy='htp')
        
        model, inputs = self.fixtures.get_complex_mlp()
        output_path = self.get_output_path("htp_builtin")
        
        result = exporter.export(model, inputs, output_path)
        
        self.assert_export_success(result)
        assert result['strategy'] == 'htp_builtin'
    
    def test_htp_attention_model(self):
        """Test HTP with attention model."""
        model, inputs = self.fixtures.get_attention_model()
        output_path = self.get_output_path("htp_attention")
        
        result = self.exporter.export(model, inputs, output_path)
        
        self.assert_export_success(result)
        self.assert_htp_specific_results(result)
        
        # HTP should handle attention mechanisms
        assert result['tagged_operations'] > 0
    
    def test_htp_operation_tracing(self):
        """Test that operation tracing works correctly."""
        model, inputs = self.fixtures.get_simple_cnn()
        output_path = self.get_output_path("htp_tracing")
        
        # Export model
        result = self.exporter.export(model, inputs, output_path)
        
        self.assert_export_success(result)
        
        # Check that operation tracing captured operations
        assert result['total_operations'] > 0
        assert result['tagged_operations'] >= 0
        assert result['tagged_operations'] <= result['total_operations']
    
    def test_htp_sidecar_file(self):
        """Test HTP sidecar file creation."""
        model, inputs = self.fixtures.get_simple_cnn()
        output_path = self.get_output_path("htp_sidecar")
        
        result = self.exporter.export(model, inputs, output_path)
        
        self.assert_export_success(result)
        
        # Check sidecar file
        sidecar_path = output_path.replace('.onnx', '_hierarchy.json')
        assert Path(sidecar_path).exists()
        
        # Load and validate sidecar content
        import json
        with open(sidecar_path) as f:
            sidecar_data = json.load(f)
        
        assert 'exporter' in sidecar_data
        assert 'strategy' in sidecar_data['exporter']
        assert sidecar_data['exporter']['strategy'] in ['htp', 'htp_builtin']
        assert 'node_tags' in sidecar_data


class TestHTPOperationConfig:
    """Test HTP operation configuration and patching."""
    
    def setup_method(self):
        """Set up test environment."""
        self.exporter = HTPHierarchyExporter(strategy='htp')
        self.fixtures = TestModelFixtures()
    
    def test_operation_patching(self):
        """Test that PyTorch operations are properly patched."""
        # This tests the patching mechanism indirectly through export
        model, inputs = self.fixtures.get_simple_cnn()
        output_path = f"/tmp/test_patching.onnx"
        
        result = self.exporter.export(model, inputs, output_path)
        
        # If operations were patched, we should get operation traces
        assert 'total_operations' in result
        assert result['total_operations'] > 0
    
    def test_operation_registry_access(self):
        """Test operation registry access."""
        from modelexport.core.operation_config import OperationConfig
        
        # Test that we can get operations to patch
        operations = OperationConfig.get_operations_to_patch()
        assert isinstance(operations, list)
        assert len(operations) > 0
        
        # Test torch to ONNX mapping
        mapping = OperationConfig.get_torch_to_onnx_mapping()
        assert isinstance(mapping, dict)
        assert len(mapping) > 0


class TestHTPHookManagement:
    """Test HTP hook registration and management."""
    
    def setup_method(self):
        """Set up test environment."""
        self.exporter = HTPHierarchyExporter(strategy='htp')
        self.fixtures = TestModelFixtures()
    
    def test_hook_registration_cleanup(self):
        """Test that hooks are properly registered and cleaned up."""
        model, inputs = self.fixtures.get_simple_cnn()
        
        # Count hooks before export
        initial_hooks = self._count_model_hooks(model)
        
        # Export (should register and clean up hooks)
        output_path = f"/tmp/test_hooks.onnx"
        result = self.exporter.export(model, inputs, output_path)
        
        # Count hooks after export (should be same as initial)
        final_hooks = self._count_model_hooks(model)
        
        assert final_hooks == initial_hooks, "Hooks were not properly cleaned up"
    
    def _count_model_hooks(self, model):
        """Count the number of hooks registered on a model."""
        hook_count = 0
        for module in model.modules():
            hook_count += len(module._forward_hooks)
        return hook_count


class TestHTPTagPropagation:
    """Test HTP tag propagation mechanisms."""
    
    def setup_method(self):
        """Set up test environment."""
        self.exporter = HTPHierarchyExporter(strategy='htp')
        self.fixtures = TestModelFixtures()
    
    def test_conservative_propagation(self):
        """Test conservative tag propagation."""
        model, inputs = self.fixtures.get_complex_mlp()
        output_path = f"/tmp/test_propagation.onnx"
        
        result = self.exporter.export(model, inputs, output_path)
        
        # HTP uses conservative propagation, so tagged operations
        # should be <= total operations
        assert result['tagged_operations'] <= result['total_operations']
    
    def test_tag_consistency(self):
        """Test that tags are consistently applied."""
        model, inputs = self.fixtures.get_simple_cnn()
        output_path = f"/tmp/test_consistency.onnx"
        
        # Export same model multiple times
        results = []
        for i in range(3):
            path = f"/tmp/test_consistency_{i}.onnx"
            result = self.exporter.export(model, inputs, path)
            results.append(result)
        
        # Results should be consistent
        first_result = results[0]
        for result in results[1:]:
            assert result['total_operations'] == first_result['total_operations']
            # Tagged operations might vary slightly due to execution order
            # but should be in similar range
            assert abs(result['tagged_operations'] - first_result['tagged_operations']) <= 2


class TestHTPMemoryManagement:
    """Test HTP memory management and cleanup."""
    
    def setup_method(self):
        """Set up test environment."""
        self.exporter = HTPHierarchyExporter(strategy='htp')
        self.fixtures = TestModelFixtures()
    
    def test_large_model_handling(self):
        """Test that HTP can handle larger models without memory issues."""
        # Create a larger version of the CNN
        class LargeCNN(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.features = torch.nn.Sequential(
                    torch.nn.Conv2d(3, 64, 3, padding=1),
                    torch.nn.ReLU(),
                    torch.nn.Conv2d(64, 128, 3, padding=1),
                    torch.nn.ReLU(),
                    torch.nn.MaxPool2d(2),
                    torch.nn.Conv2d(128, 256, 3, padding=1),
                    torch.nn.ReLU(),
                    torch.nn.Conv2d(256, 256, 3, padding=1),
                    torch.nn.ReLU(),
                    torch.nn.MaxPool2d(2),
                )
                self.classifier = torch.nn.Sequential(
                    torch.nn.Dropout(),
                    torch.nn.Linear(256 * 8 * 8, 512),
                    torch.nn.ReLU(),
                    torch.nn.Dropout(),
                    torch.nn.Linear(512, 10)
                )
            
            def forward(self, x):
                x = self.features(x)
                x = x.view(x.size(0), -1)
                x = self.classifier(x)
                return x
        
        model = LargeCNN()
        model.eval()
        inputs = torch.randn(1, 3, 32, 32)
        
        output_path = f"/tmp/test_large_model.onnx"
        result = self.exporter.export(model, inputs, output_path)
        
        self.assert_export_success(result)
        assert result['total_operations'] > 0
    
    def assert_export_success(self, result):
        """Helper method for export success assertion."""
        assert 'onnx_path' in result
        assert 'strategy' in result
        assert Path(result['onnx_path']).exists()