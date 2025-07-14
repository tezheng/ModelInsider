"""
Strategy Comparison Integration Tests

Tests that compare different export strategies against each other
to validate consistency and identify strategy-specific advantages.
"""

import shutil
import tempfile
from pathlib import Path

import pytest

from modelexport.strategies.htp import HTPHierarchyExporter
from modelexport.strategies.usage_based import UsageBasedExporter

from ..fixtures.base_test import IntegrationTestBase
from ..fixtures.test_models import TestModelFixtures


class TestStrategyComparison(IntegrationTestBase):
    """Compare all export strategies on the same models."""
    
    def setup_method(self):
        """Set up test environment."""
        super().setup_method()
        
        # Initialize all exporters
        self.htp_exporter = HTPHierarchyExporter(strategy='htp')
        self.usage_exporter = UsageBasedExporter()
        
        self.exporters = {
            'htp': self.htp_exporter,
            'usage_based': self.usage_exporter
        }
    
    def test_control_flow_models(self):
        """Test strategies on models with control flow (FX incompatible)."""
        model, inputs = self.fixtures.get_conditional_model()
        results = {}
        
        for strategy_name, exporter in self.exporters.items():
            output_path = self.get_output_path(strategy_name, "conditional_model")
            
            try:
                result = exporter.export(model, inputs, output_path)
                results[strategy_name] = result
                
            except Exception as e:
                # HTP and usage-based should handle control flow
                pytest.fail(f"{strategy_name} should handle control flow but failed: {e}")
        
        # HTP and usage-based should succeed
        assert results['htp'] is not None, "HTP should handle control flow"
        assert results['usage_based'] is not None, "Usage-based should handle control flow"
    
    def test_coverage_comparison(self):
        """Compare coverage rates across strategies."""
        model, inputs = self.fixtures.get_complex_mlp()
        coverage_results = {}
        
        for strategy_name, exporter in self.exporters.items():
            output_path = self.get_output_path(strategy_name, "coverage_comparison")
            
            try:
                result = exporter.export(model, inputs, output_path)
                
                # Extract coverage information
                if 'tagged_operations' in result and 'total_operations' in result:
                    coverage = result['tagged_operations'] / max(result['total_operations'], 1)
                else:
                    coverage = None
                
                coverage_results[strategy_name] = {
                    'coverage': coverage,
                    'hierarchy_nodes': result.get('hierarchy_nodes', 0),
                    'strategy': strategy_name
                }
                
            except Exception as e:
                print(f"Coverage test failed for {strategy_name}: {e}")
                coverage_results[strategy_name] = None
        
        # Analyze coverage differences
        valid_results = {k: v for k, v in coverage_results.items() if v is not None}
        
        if len(valid_results) >= 2:
            strategies = list(valid_results.keys())
            print(f"Coverage comparison for {strategies}:")
            
            for strategy, data in valid_results.items():
                coverage = data['coverage']
                nodes = data['hierarchy_nodes']
                print(f"  {strategy}: {coverage:.2%} coverage, {nodes} hierarchy nodes")
    
    def test_export_consistency(self):
        """Test that multiple exports of the same model are consistent."""
        model, inputs = self.fixtures.get_simple_cnn()
        
        for strategy_name, exporter in self.exporters.items():
            # Export same model multiple times
            results = []
            for i in range(3):
                output_path = self.get_output_path(strategy_name, f"consistency_{i}")
                result = exporter.export(model, inputs, output_path)
                results.append(result)
            
            # Check consistency
            first_result = results[0]
            for i, result in enumerate(results[1:], 1):
                # Strategy should be same
                assert result['strategy'] == first_result['strategy']
                
                # File should be created
                assert Path(result['onnx_path']).exists()
                
                # Results should be reasonably similar
                if 'hierarchy_nodes' in result and 'hierarchy_nodes' in first_result:
                    # Allow small variations
                    diff = abs(result['hierarchy_nodes'] - first_result['hierarchy_nodes'])
                    assert diff <= 2, f"Inconsistent hierarchy nodes in {strategy_name}: {diff}"
    
    def test_output_file_structure(self):
        """Test that all strategies create proper output file structure."""
        model, inputs = self.fixtures.get_simple_cnn()
        
        for strategy_name, exporter in self.exporters.items():
            output_path = self.get_output_path(strategy_name, "file_structure")
            
            result = exporter.export(model, inputs, output_path)
            
            # Main ONNX file should exist
            onnx_path = Path(result['onnx_path'])
            assert onnx_path.exists()
            assert onnx_path.suffix == '.onnx'
            assert onnx_path.stat().st_size > 0
            
            # Sidecar file should exist
            sidecar_path = output_path.replace('.onnx', '_hierarchy.json')
            assert Path(sidecar_path).exists()
            
            # Load and validate sidecar
            import json
            with open(sidecar_path) as f:
                sidecar_data = json.load(f)
            
            assert 'strategy' in sidecar_data
            assert 'node_tags' in sidecar_data
            
    
    def _analyze_coverage_differences(self, results, model_name):
        """Analyze coverage differences between strategies."""
        coverage_data = {}
        
        for strategy, result in results.items():
            if result is None:
                continue
                
            if 'tagged_operations' in result and 'total_operations' in result:
                coverage = result['tagged_operations'] / max(result['total_operations'], 1)
                nodes = result.get('hierarchy_nodes', result['tagged_operations'])
            else:
                continue
            
            coverage_data[strategy] = {'coverage': coverage, 'nodes': nodes}
        
        if len(coverage_data) >= 2:
            print(f"\nCoverage analysis for {model_name}:")
            for strategy, data in coverage_data.items():
                print(f"  {strategy}: {data['coverage']:.1%} coverage, {data['nodes']} nodes")


class TestStrategyInteroperability:
    """Test that outputs from different strategies are interoperable."""
    
    def setup_method(self):
        """Set up test environment."""
        self.temp_dir = tempfile.mkdtemp()
        self.temp_path = Path(self.temp_dir)
        self.fixtures = TestModelFixtures()
    
    def teardown_method(self):
        """Clean up test environment."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_tag_utils_compatibility(self):
        """Test that tag utilities work with all strategy outputs."""
        from modelexport.core import tag_utils
        
        model, inputs = self.fixtures.get_simple_cnn()
        
        # Export with different strategies
        strategies = {
            'htp': HTPHierarchyExporter(strategy='htp'),
            'usage_based': UsageBasedExporter()
        }
        
        onnx_files = {}
        
        for strategy_name, exporter in strategies.items():
            output_path = str(self.temp_path / f"interop_{strategy_name}.onnx")
            result = exporter.export(model, inputs, output_path)
            onnx_files[strategy_name] = output_path
        
        # Test tag utilities on all outputs
        for strategy_name, onnx_path in onnx_files.items():
            # Load tags from ONNX
            onnx_tags = tag_utils.load_tags_from_onnx(onnx_path)
            assert isinstance(onnx_tags, dict)
            
            # Load tags from sidecar
            try:
                sidecar_data = tag_utils.load_tags_from_sidecar(onnx_path)
                assert isinstance(sidecar_data, dict)
                assert 'strategy' in sidecar_data
            except FileNotFoundError:
                pytest.skip(f"No sidecar file for {strategy_name}")
            
            # Get tag statistics
            stats = tag_utils.get_tag_statistics(onnx_path)
            assert isinstance(stats, dict)
            
            # Validate consistency
            validation = tag_utils.validate_tag_consistency(onnx_path)
            assert isinstance(validation, dict)
            assert 'consistent' in validation
    
    def test_onnx_model_validity(self):
        """Test that all strategies produce valid ONNX models."""
        import onnx
        
        model, inputs = self.fixtures.get_simple_cnn()
        
        strategies = {
            'htp': HTPHierarchyExporter(strategy='htp'),
            'usage_based': UsageBasedExporter()
        }
        
        for strategy_name, exporter in strategies.items():
            output_path = str(self.temp_path / f"validity_{strategy_name}.onnx")
            result = exporter.export(model, inputs, output_path)
            
            # Load and validate ONNX model
            onnx_model = onnx.load(output_path)
            
            # ONNX checker should pass
            try:
                onnx.checker.check_model(onnx_model)
            except Exception as e:
                pytest.fail(f"ONNX validation failed for {strategy_name}: {e}")
            
            # Model should have basic structure
            assert len(onnx_model.graph.node) > 0, f"No nodes in {strategy_name} output"
            assert len(onnx_model.graph.input) > 0, f"No inputs in {strategy_name} output"
            assert len(onnx_model.graph.output) > 0, f"No outputs in {strategy_name} output"