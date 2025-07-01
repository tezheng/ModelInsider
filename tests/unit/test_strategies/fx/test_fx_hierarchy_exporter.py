"""
FX Hierarchy Exporter Unit Tests

Tests specific to the FX graph-based export strategy.
"""

import pytest
import torch
import torch.fx
from pathlib import Path

from modelexport.strategies.fx import FXHierarchyExporter
from tests.fixtures.base_test import BaseStrategyTest, StrategyCompatibilityTest
from tests.fixtures.test_models import TestModelFixtures


class TestFXHierarchyExporter(BaseStrategyTest, StrategyCompatibilityTest):
    """Test FX-based hierarchy exporter."""
    
    def get_exporter(self) -> FXHierarchyExporter:
        """Get FX exporter instance."""
        return FXHierarchyExporter()
    
    def get_strategy_name(self) -> str:
        """Get strategy name."""
        return "fx"
    
    def assert_fx_specific_results(self, result):
        """Assert FX-specific result fields."""
        assert 'fx_graph_stats' in result
        assert 'sidecar_path' in result
        assert 'module_info_path' in result
        
        fx_stats = result['fx_graph_stats']
        assert 'total_fx_nodes' in fx_stats
        assert 'coverage_ratio' in fx_stats
        assert 'module_types_found' in fx_stats
    
    def test_fx_graph_creation(self):
        """Test that FX graph is created successfully."""
        model, inputs = self.fixtures.get_simple_cnn()
        output_path = self.get_output_path("fx_graph_test")
        
        result = self.exporter.export(model, inputs, output_path)
        
        self.assert_export_success(result)
        self.assert_fx_specific_results(result)
        
        # FX should achieve high coverage on simple CNN
        assert result['fx_graph_stats']['coverage_ratio'] > 0.8
    
    def test_fx_node_hierarchy_mapping(self):
        """Test that FX nodes are properly mapped to hierarchy."""
        model, inputs = self.fixtures.get_complex_mlp()
        output_path = self.get_output_path("hierarchy_mapping")
        
        result = self.exporter.export(model, inputs, output_path)
        
        self.assert_export_success(result)
        self.assert_fx_specific_results(result)
        
        # Check that hierarchy nodes were created
        assert result['hierarchy_nodes'] > 0
        assert result['unique_modules'] > 0
    
    def test_fx_exception_handling(self):
        """Test custom torch.nn exception handling."""
        # Create exporter with custom exceptions
        custom_exceptions = {"LayerNorm", "Linear", "ReLU"}
        exporter = FXHierarchyExporter(torch_nn_exceptions=custom_exceptions)
        
        model, inputs = self.fixtures.get_complex_mlp()
        output_path = self.get_output_path("custom_exceptions")
        
        result = exporter.export(model, inputs, output_path)
        
        self.assert_export_success(result)
        self.assert_fx_specific_results(result)
    
    def test_fx_auto_fallback_disabled(self):
        """Test FX export with auto-fallback disabled."""
        exporter = FXHierarchyExporter(auto_fallback=False)
        
        # Try with conditional model (should fail)
        model, inputs = self.fixtures.get_conditional_model()
        output_path = self.get_output_path("no_fallback")
        
        with pytest.raises(RuntimeError, match="FX symbolic tracing failed"):
            exporter.export(model, inputs, output_path)
    
    def test_fx_auto_fallback_enabled(self):
        """Test FX export with auto-fallback enabled."""
        exporter = FXHierarchyExporter(auto_fallback=True)
        
        # This would require HTP exporter to be available for fallback
        # For now, we test that the compatibility analysis works
        model, inputs = self.fixtures.get_conditional_model()
        
        # The compatibility analysis should detect this as incompatible
        compatibility = exporter._analyze_model_compatibility(model, inputs)
        assert not compatibility['fx_compatible']
    
    def test_fx_attention_model(self):
        """Test FX export with attention model."""
        model, inputs = self.fixtures.get_attention_model()
        output_path = self.get_output_path("attention_test")
        
        result = self.exporter.export(model, inputs, output_path)
        
        self.assert_export_success(result)
        self.assert_fx_specific_results(result)
        
        # Attention models should still get reasonable coverage
        assert result['fx_graph_stats']['coverage_ratio'] > 0.5
    
    def test_fx_module_filtering(self):
        """Test that torch.nn modules are properly filtered."""
        model, inputs = self.fixtures.get_simple_cnn()
        
        # Test the module filtering logic directly
        for name, module in model.named_modules():
            should_include = self.exporter._should_include_module(module)
            
            # torch.nn modules should be filtered except for exceptions
            module_class = module.__class__.__name__
            if module.__class__.__module__.startswith('torch.nn'):
                expected = module_class in self.exporter._torch_nn_exceptions
                assert should_include == expected, f"Module {name} ({module_class}) filtering incorrect"
    
    def test_fx_hierarchy_path_building(self):
        """Test hierarchy path building logic."""
        model, inputs = self.fixtures.get_simple_cnn()
        
        # Test hierarchy path building
        all_modules = dict(model.named_modules())
        
        for name, module in all_modules.items():
            if name and self.exporter._should_include_module(module):
                hierarchy_path = self.exporter._build_fx_hierarchy_path(name, module)
                
                # Hierarchy path should start with root model class
                assert hierarchy_path.startswith("/SimpleCNN")
                
                # Should contain the module class name
                assert module.__class__.__name__ in hierarchy_path
    
    def test_fx_sidecar_files(self):
        """Test that sidecar files are created correctly."""
        model, inputs = self.fixtures.get_simple_cnn()
        output_path = self.get_output_path("sidecar_test")
        
        result = self.exporter.export(model, inputs, output_path)
        
        self.assert_export_success(result)
        
        # Check sidecar files exist
        sidecar_path = Path(result['sidecar_path'])
        module_info_path = Path(result['module_info_path'])
        
        assert sidecar_path.exists()
        assert module_info_path.exists()
        
        # Check sidecar file content
        import json
        with open(sidecar_path) as f:
            sidecar_data = json.load(f)
        
        assert 'export_method' in sidecar_data
        assert sidecar_data['export_method'] == 'fx_graph'
        assert 'hierarchy_mapping' in sidecar_data
        assert 'statistics' in sidecar_data
    
    def test_fx_topology_preservation(self):
        """Test that topology is preserved in ONNX export."""
        model, inputs = self.fixtures.get_simple_cnn()
        output_path = self.get_output_path("topology_test")
        
        result = self.exporter.export(model, inputs, output_path)
        
        self.assert_export_success(result)
        
        # Check topology preservation flag
        assert result.get('topology_preserved', False)
        
        # Load and validate ONNX model
        import onnx
        onnx_model = onnx.load(result['onnx_path'])
        onnx.checker.check_model(onnx_model)
    
    def test_fx_coverage_statistics(self):
        """Test coverage statistics calculation."""
        model, inputs = self.fixtures.get_complex_mlp()
        output_path = self.get_output_path("coverage_stats")
        
        result = self.exporter.export(model, inputs, output_path)
        
        self.assert_export_success(result)
        
        fx_stats = result['fx_graph_stats']
        
        # Validate statistics
        assert fx_stats['total_fx_nodes'] > 0
        assert fx_stats['hierarchy_nodes'] >= 0
        assert fx_stats['hierarchy_nodes'] <= fx_stats['total_fx_nodes']
        assert 0 <= fx_stats['coverage_ratio'] <= 1
        
        # Check detailed statistics
        assert 'node_type_distribution' in fx_stats
        assert 'confidence_distribution' in fx_stats
        assert 'hierarchy_categories' in fx_stats


class TestFXCompatibilityAnalysis:
    """Test FX compatibility analysis functionality."""
    
    def setup_method(self):
        """Set up test environment."""
        self.exporter = FXHierarchyExporter()
        self.fixtures = TestModelFixtures()
    
    def test_architecture_pattern_detection(self):
        """Test architecture pattern detection."""
        models = self.fixtures.get_all_models()
        metadata = self.fixtures.get_model_metadata()
        
        for model_name, (model, inputs) in models.items():
            compatibility = self.exporter._analyze_model_compatibility(model, inputs)
            expected_compatible = metadata[model_name]['fx_compatible']
            
            # Verify compatibility detection
            if model_name == 'conditional_model':
                # Should detect as incompatible due to control flow
                assert not compatibility['fx_compatible']
                assert 'control_flow' in compatibility.get('reason', '').lower() or \
                       compatibility['confidence'] < 0.5
            else:
                # Simple models should be detected as compatible
                assert compatibility['fx_compatible'] or compatibility['confidence'] > 0.5
    
    def test_model_signature_caching(self):
        """Test that model signatures are cached for performance."""
        model, inputs = self.fixtures.get_simple_cnn()
        
        # First analysis
        sig1 = self.exporter._get_model_signature(model)
        compat1 = self.exporter._analyze_model_compatibility(model, inputs)
        
        # Second analysis should use cache
        sig2 = self.exporter._get_model_signature(model)
        compat2 = self.exporter._analyze_model_compatibility(model, inputs)
        
        assert sig1 == sig2
        assert compat1 == compat2
        
        # Cache should contain the signature
        assert sig1 in self.exporter._compatibility_cache


class TestFXGraphAnalysis:
    """Test FX graph analysis and mapping functionality."""
    
    def setup_method(self):
        """Set up test environment."""
        self.exporter = FXHierarchyExporter()
        self.fixtures = TestModelFixtures()
    
    def test_fx_node_signature_generation(self):
        """Test FX node signature generation."""
        model, inputs = self.fixtures.get_simple_cnn()
        
        # Create FX graph
        fx_graph = torch.fx.symbolic_trace(model)
        
        # Test signature generation for different node types
        for node in fx_graph.graph.nodes:
            self.exporter._fx_result = type('MockResult', (), {'fx_graph': fx_graph})()
            signature = self.exporter._get_fx_node_signature(node)
            
            # Signature should be a string
            assert isinstance(signature, str)
            
            # Should contain the operation type
            assert node.op in signature or str(node.target) in signature
    
    def test_fx_execution_order_analysis(self):
        """Test FX execution order analysis."""
        model, inputs = self.fixtures.get_simple_cnn()
        
        # Create FX graph
        fx_graph = torch.fx.symbolic_trace(model)
        
        # Analyze execution order
        execution_order = self.exporter._analyze_fx_execution_order(fx_graph)
        
        # Should return a dictionary mapping node names to execution indices
        assert isinstance(execution_order, dict)
        
        # Indices should be in ascending order
        if execution_order:
            indices = list(execution_order.values())
            assert indices == sorted(indices)