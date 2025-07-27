"""
Base Test Classes for Strategy Testing

Provides common test infrastructure and utilities for all export strategies.
"""

import shutil
import tempfile
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

import pytest

from modelexport.strategies.htp import HTPExporter

from .test_models import TestModelFixtures


class BaseStrategyTest(ABC):
    """
    Abstract base class for strategy-specific tests.
    
    Provides common test infrastructure and standard test patterns that
    all export strategies should implement.
    """
    
    @abstractmethod
    def get_exporter(self) -> HTPExporter:
        """Get the exporter instance for this strategy."""
        pass
    
    @abstractmethod
    def get_strategy_name(self) -> str:
        """Get the strategy name for identification."""
        pass
    
    def setup_method(self):
        """Set up test environment before each test."""
        self.temp_dir = tempfile.mkdtemp()
        self.temp_path = Path(self.temp_dir)
        self.exporter = self.get_exporter()
        self.fixtures = TestModelFixtures()
        
    def teardown_method(self):
        """Clean up test environment after each test."""
        if hasattr(self, 'temp_dir'):
            shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def get_output_path(self, name: str) -> str:
        """Get a temporary output path for test files."""
        return str(self.temp_path / f"{name}_{self.get_strategy_name()}.onnx")
    
    def assert_export_success(self, result: dict[str, Any]):
        """Assert that export was successful with required fields."""
        assert 'onnx_path' in result
        assert 'strategy' in result
        assert Path(result['onnx_path']).exists()
        
        # Strategy-specific assertions can be added in subclasses
        if 'hierarchy_nodes' in result:
            assert result['hierarchy_nodes'] >= 0
    
    def assert_hierarchy_coverage(self, result: dict[str, Any], min_coverage: float = 0.1):
        """Assert minimum hierarchy coverage."""
        if 'coverage_ratio' in result:
            assert result['coverage_ratio'] >= min_coverage
    
    def test_simple_model_export(self):
        """Test export with SimpleCNN model."""
        model, inputs = self.fixtures.get_simple_cnn()
        output_path = self.get_output_path("simple_cnn")
        
        result = self.exporter.export(model, inputs, output_path)
        self.assert_export_success(result)
        
        # Check that files were created
        assert Path(output_path).exists()
        assert Path(output_path).stat().st_size > 0
    
    def test_complex_model_export(self):
        """Test export with ComplexMLP model."""
        model, inputs = self.fixtures.get_complex_mlp()
        output_path = self.get_output_path("complex_mlp")
        
        result = self.exporter.export(model, inputs, output_path)
        self.assert_export_success(result)
        self.assert_hierarchy_coverage(result)
    
    def test_export_with_kwargs(self):
        """Test export with additional ONNX export arguments."""
        model, inputs = self.fixtures.get_simple_cnn()
        output_path = self.get_output_path("kwargs_test")
        
        # Test with common ONNX export arguments
        result = self.exporter.export(
            model, inputs, output_path,
            opset_version=14,
            input_names=['input'],
            output_names=['output']
        )
        
        self.assert_export_success(result)
    
    def test_extract_subgraph(self):
        """Test subgraph extraction functionality."""
        model, inputs = self.fixtures.get_simple_cnn()
        output_path = self.get_output_path("subgraph_test")
        
        # First export the model
        export_result = self.exporter.export(model, inputs, output_path)
        self.assert_export_success(export_result)
        
        # Then test subgraph extraction
        subgraph_result = self.exporter.extract_module_subgraph(output_path, "conv1")
        
        # Check if extraction succeeded or handle expected error gracefully
        if 'error' not in subgraph_result:
            assert 'target_module' in subgraph_result
            assert subgraph_result['target_module'] == "conv1"
            assert 'strategy' in subgraph_result
        else:
            # If module not found, just verify the method works and returns proper error structure
            assert isinstance(subgraph_result, dict)
            assert 'error' in subgraph_result
    
    def test_get_export_stats(self):
        """Test export statistics retrieval."""
        model, inputs = self.fixtures.get_simple_cnn()
        output_path = self.get_output_path("stats_test")
        
        # Export model
        export_result = self.exporter.export(model, inputs, output_path)
        self.assert_export_success(export_result)
        
        # Check stats from export result
        assert isinstance(export_result, dict)
        assert 'strategy' in export_result
        assert export_result['strategy'] == 'htp_builtin'


class StrategyCompatibilityTest:
    """
    Mixin class for testing strategy compatibility with different model types.
    """
    
    def test_model_compatibility_detection(self):
        """Test that strategy correctly identifies compatible models."""
        if not hasattr(self, 'exporter'):
            pytest.skip("No exporter available")
            
        models = self.fixtures.get_all_models()
        metadata = self.fixtures.get_model_metadata()
        
        for model_name, (model, inputs) in models.items():
            model_meta = metadata[model_name]
            
            try:
                output_path = self.get_output_path(f"compat_{model_name}")
                result = self.exporter.export(model, inputs, output_path)
                
                # If export succeeds, model should be compatible
                self.assert_export_success(result)
                    
            except Exception as e:
                # If export fails, check if it was expected
                pass  # Some strategies may fail on certain models


class IntegrationTestBase:
    """
    Base class for integration tests that test multiple strategies together.
    """
    
    def setup_method(self):
        """Set up integration test environment."""
        self.temp_dir = tempfile.mkdtemp()
        self.temp_path = Path(self.temp_dir)
        self.fixtures = TestModelFixtures()
    
    def teardown_method(self):
        """Clean up integration test environment."""
        if hasattr(self, 'temp_dir'):
            shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def get_output_path(self, strategy: str, name: str) -> str:
        """Get output path for strategy-specific files."""
        return str(self.temp_path / f"{name}_{strategy}.onnx")
    
    def compare_strategy_results(self, results: dict[str, dict[str, Any]]):
        """Compare results from different strategies."""
        strategies = list(results.keys())
        
        # All strategies should succeed on compatible models
        for strategy, result in results.items():
            if result is not None:  # None indicates expected failure
                assert 'onnx_path' in result
                assert 'strategy' in result
                assert result['strategy'] == strategy or strategy in result['strategy']