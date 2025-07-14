"""
Usage-based Exporter Unit Tests

Tests specific to the usage-based (legacy) export strategy.
"""

from pathlib import Path

from modelexport.strategies.usage_based import UsageBasedExporter
from tests.fixtures.base_test import BaseStrategyTest, StrategyCompatibilityTest
from tests.fixtures.test_models import TestModelFixtures


class TestUsageBasedExporter(BaseStrategyTest, StrategyCompatibilityTest):
    """Test usage-based hierarchy exporter."""
    
    def get_exporter(self) -> UsageBasedExporter:
        """Get usage-based exporter instance."""
        return UsageBasedExporter()
    
    def get_strategy_name(self) -> str:
        """Get strategy name."""
        return "usage_based"
    
    def assert_usage_specific_results(self, result):
        """Assert usage-based specific result fields."""
        assert result['strategy'] == 'usage_based'
        assert 'hierarchy_nodes' in result
        assert 'unique_modules' in result
        assert 'usage_stats' in result
    
    def test_usage_basic_export(self):
        """Test basic usage-based export functionality."""
        model, inputs = self.fixtures.get_simple_cnn()
        output_path = self.get_output_path("usage_basic")
        
        result = self.exporter.export(model, inputs, output_path)
        
        self.assert_export_success(result)
        self.assert_usage_specific_results(result)
        
        # MUST-002 compliance: torch.nn modules filtered out, only custom modules remain
        assert result['hierarchy_nodes'] == 0  # MUST-002: torch.nn modules filtered out
        assert result['unique_modules'] == 1   # Only SimpleCNN root (non-torch.nn) remains
    
    def test_usage_module_tracking(self):
        """Test module usage tracking functionality."""
        model, inputs = self.fixtures.get_complex_mlp()
        output_path = self.get_output_path("usage_tracking")
        
        result = self.exporter.export(model, inputs, output_path)
        
        self.assert_export_success(result)
        self.assert_usage_specific_results(result)
        
        # Check usage statistics
        usage_stats = result['usage_stats']
        assert isinstance(usage_stats, dict)
        assert len(usage_stats) > 0
        
        # All usage counts should be positive
        for module_name, count in usage_stats.items():
            assert count > 0
    
    def test_usage_custom_exceptions(self):
        """Test usage-based export with custom torch.nn exceptions."""
        custom_exceptions = ["LayerNorm", "Linear", "ReLU", "Dropout"]
        exporter = UsageBasedExporter(torch_nn_exceptions=custom_exceptions)
        
        model, inputs = self.fixtures.get_complex_mlp()
        output_path = self.get_output_path("usage_custom_exceptions")
        
        result = exporter.export(model, inputs, output_path)
        
        self.assert_export_success(result)
        self.assert_usage_specific_results(result)
        
        # With more exceptions, should get more modules tracked
        assert result['unique_modules'] > 0
    
    def test_usage_sidecar_file(self):
        """Test usage-based sidecar file creation."""
        model, inputs = self.fixtures.get_simple_cnn()
        output_path = self.get_output_path("usage_sidecar")
        
        result = self.exporter.export(model, inputs, output_path)
        
        self.assert_export_success(result)
        
        # Check sidecar file
        sidecar_path = Path(result['sidecar_path'])
        assert sidecar_path.exists()
        
        # Load and validate sidecar content
        import json
        with open(sidecar_path) as f:
            sidecar_data = json.load(f)
        
        assert 'metadata' in sidecar_data
        assert sidecar_data['metadata']['strategy'] == 'usage_based'
        assert 'node_tags' in sidecar_data
        assert 'metadata' in sidecar_data
        
        # Check usage-specific metadata
        metadata = sidecar_data['metadata']
        assert 'usage_stats' in metadata
        assert 'torch_nn_exceptions' in metadata
    
    def test_usage_with_attention_model(self):
        """Test usage-based export with attention model."""
        model, inputs = self.fixtures.get_attention_model()
        output_path = self.get_output_path("usage_attention")
        
        result = self.exporter.export(model, inputs, output_path)
        
        self.assert_export_success(result)
        self.assert_usage_specific_results(result)
        
        # Should track attention-related modules
        usage_stats = result['usage_stats']
        assert len(usage_stats) > 0
    
    def test_usage_with_control_flow(self):
        """Test usage-based export with control flow model."""
        model, inputs = self.fixtures.get_conditional_model()
        output_path = self.get_output_path("usage_control_flow")
        
        # Usage-based should handle control flow (unlike FX)
        result = self.exporter.export(model, inputs, output_path)
        
        self.assert_export_success(result)
        self.assert_usage_specific_results(result)
    
    def test_usage_subgraph_extraction_limitation(self):
        """Test subgraph extraction with usage-based strategy (limited functionality)."""
        model, inputs = self.fixtures.get_simple_cnn()
        output_path = self.get_output_path("usage_subgraph")
        
        # First export the model
        export_result = self.exporter.export(model, inputs, output_path)
        self.assert_export_success(export_result)
        
        # Test subgraph extraction (should have limited functionality)
        subgraph_result = self.exporter.extract_subgraph(output_path, "conv1")
        
        assert 'target_module' in subgraph_result
        assert subgraph_result['target_module'] == "conv1"
        assert subgraph_result['strategy'] == 'usage_based'
        assert 'note' in subgraph_result or 'matching_nodes' in subgraph_result


class TestUsageBasedHookManagement:
    """Test usage-based hook management."""
    
    def setup_method(self):
        """Set up test environment."""
        self.exporter = UsageBasedExporter()
        self.fixtures = TestModelFixtures()
    
    def test_hook_cleanup(self):
        """Test that hooks are properly cleaned up after export."""
        model, inputs = self.fixtures.get_simple_cnn()
        
        # Count initial hooks
        initial_hooks = self._count_forward_hooks(model)
        
        # Export model (registers and removes hooks)
        output_path = f"/tmp/test_usage_hooks.onnx"
        result = self.exporter.export(model, inputs, output_path)
        
        # Count final hooks (should be same as initial)
        final_hooks = self._count_forward_hooks(model)
        
        assert final_hooks == initial_hooks, "Hooks were not properly cleaned up"
    
    def _count_forward_hooks(self, model):
        """Count forward hooks registered on model."""
        hook_count = 0
        for module in model.modules():
            hook_count += len(module._forward_hooks)
        return hook_count
    
    def test_module_usage_counting(self):
        """Test that module usage is counted correctly."""
        model, inputs = self.fixtures.get_simple_cnn()
        output_path = f"/tmp/test_usage_counting.onnx"
        
        # Export and check usage stats
        result = self.exporter.export(model, inputs, output_path)
        
        usage_stats = result['usage_stats']
        
        # Each tracked module should be used at least once
        for module_name, count in usage_stats.items():
            assert count >= 1, f"Module {module_name} should have usage count >= 1"


class TestUsageBasedModuleFiltering:
    """Test usage-based module filtering logic."""
    
    def setup_method(self):
        """Set up test environment."""
        self.exporter = UsageBasedExporter()
        self.fixtures = TestModelFixtures()
    
    def test_torch_nn_filtering(self):
        """Test that torch.nn modules are properly filtered."""
        model, inputs = self.fixtures.get_complex_mlp()
        
        # Test filtering logic directly
        from modelexport.core.base import should_tag_module
        
        for name, module in model.named_modules():
            should_include = should_tag_module(module, self.exporter._torch_nn_exceptions)
            
            module_class = module.__class__.__name__
            module_path = module.__class__.__module__
            
            if module_path.startswith('torch.nn'):
                # Should only include if in exceptions list
                expected = module_class in self.exporter._torch_nn_exceptions
                assert should_include == expected
            else:
                # Non-torch.nn modules should be included
                assert should_include
    
    def test_custom_exceptions_effect(self):
        """Test effect of custom torch.nn exceptions."""
        # Create exporter with minimal exceptions
        minimal_exporter = UsageBasedExporter(torch_nn_exceptions=["LayerNorm"])
        
        # Create exporter with many exceptions
        extensive_exporter = UsageBasedExporter(
            torch_nn_exceptions=["LayerNorm", "Linear", "ReLU", "Dropout", "Sequential"]
        )
        
        model, inputs = self.fixtures.get_complex_mlp()
        
        # Export with minimal exceptions
        result_minimal = minimal_exporter.export(model, inputs, "/tmp/test_minimal.onnx")
        
        # Export with extensive exceptions
        result_extensive = extensive_exporter.export(model, inputs, "/tmp/test_extensive.onnx")
        
        # Extensive exceptions should capture more modules
        assert result_extensive['unique_modules'] >= result_minimal['unique_modules']


class TestUsageBasedPerformance:
    """Test usage-based export performance characteristics."""
    
    def setup_method(self):
        """Set up test environment."""
        self.exporter = UsageBasedExporter()
        self.fixtures = TestModelFixtures()
    
    def test_export_speed(self):
        """Test that usage-based export completes in reasonable time."""
        import time
        
        model, inputs = self.fixtures.get_simple_cnn()
        output_path = f"/tmp/test_speed.onnx"
        
        start_time = time.time()
        result = self.exporter.export(model, inputs, output_path)
        end_time = time.time()
        
        export_time = end_time - start_time
        
        # Should complete within reasonable time (10 seconds for simple model)
        assert export_time < 10.0, f"Export took too long: {export_time}s"
        
        # Result should indicate successful export
        assert 'onnx_path' in result
        assert Path(result['onnx_path']).exists()
    
    def test_memory_usage(self):
        """Test that memory usage is reasonable."""
        # This is a basic test - in production you might use memory profilers
        model, inputs = self.fixtures.get_complex_mlp()
        output_path = f"/tmp/test_memory.onnx"
        
        # Export model
        result = self.exporter.export(model, inputs, output_path)
        
        # Should complete without memory errors
        assert 'onnx_path' in result
        
        # Multiple exports should not accumulate memory
        for i in range(3):
            path = f"/tmp/test_memory_{i}.onnx"
            result = self.exporter.export(model, inputs, path)
            assert 'onnx_path' in result