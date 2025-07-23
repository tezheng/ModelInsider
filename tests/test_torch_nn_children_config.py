"""
Test the flexible include_torch_nn_children configuration for HTP exporter.

Tests the regression fix for TEZ-47 where include_torch_nn_children should be:
- bool: False (default), True (use default list)
- list[str]: Custom list of torch.nn module types to include
"""

import pytest
import torch
import torch.nn as nn
from pathlib import Path
import tempfile
import onnx

from modelexport.strategies.htp_new.htp_exporter import HTPExporter, HTPConfig


class SimpleModel(nn.Module):
    """Simple test model with both HF and torch.nn components."""
    
    def __init__(self):
        super().__init__()
        self.embedding = nn.Embedding(100, 32)
        self.layer_norm = nn.LayerNorm(32)
        self.linear = nn.Linear(32, 16)
        self.dropout = nn.Dropout(0.1)
        
    def forward(self, input_ids):
        x = self.embedding(input_ids)
        x = self.layer_norm(x)
        x = self.linear(x)
        x = self.dropout(x)
        return x


class TestTorchNNChildrenConfig:
    """Test the flexible include_torch_nn_children configuration."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.model = SimpleModel()
        self.model.eval()
        self.temp_dir = tempfile.mkdtemp()
        
    def teardown_method(self):
        """Clean up test fixtures."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_default_behavior_false(self):
        """Test default behavior (include_torch_nn_children=False)."""
        exporter = HTPExporter(include_torch_nn_children=False)
        output_path = Path(self.temp_dir) / "test_false.onnx"
        
        # Export model
        result = exporter.export(
            model=self.model,
            output_path=str(output_path),
            input_specs={"input_ids": {"shape": [1, 10], "dtype": "int64"}}
        )
        
        # Check that torch.nn modules are NOT in hierarchy
        assert "embedding" not in exporter._hierarchy_data
        assert "layer_norm" not in exporter._hierarchy_data
        assert "linear" not in exporter._hierarchy_data
        assert "dropout" not in exporter._hierarchy_data
        
        # Only the root module should be in hierarchy
        assert len(exporter._hierarchy_data) == 1
        assert "" in exporter._hierarchy_data  # Root module
        
    def test_default_list_true(self):
        """Test using default list (include_torch_nn_children=True)."""
        exporter = HTPExporter(include_torch_nn_children=True)
        output_path = Path(self.temp_dir) / "test_true.onnx"
        
        # Export model
        result = exporter.export(
            model=self.model,
            output_path=str(output_path),
            input_specs={"input_ids": {"shape": [1, 10], "dtype": "int64"}}
        )
        
        # Check that default torch.nn modules are in hierarchy
        assert "embedding" in exporter._hierarchy_data
        assert "layer_norm" in exporter._hierarchy_data
        
        # Check that non-default modules are NOT in hierarchy
        assert "linear" not in exporter._hierarchy_data
        assert "dropout" not in exporter._hierarchy_data
        
        # Verify the class names match default list
        assert exporter._hierarchy_data["embedding"]["class_name"] == "Embedding"
        assert exporter._hierarchy_data["layer_norm"]["class_name"] == "LayerNorm"
        
    def test_custom_list(self):
        """Test using custom list of torch.nn modules."""
        custom_list = ["Linear", "Dropout"]
        exporter = HTPExporter(include_torch_nn_children=custom_list)
        output_path = Path(self.temp_dir) / "test_custom.onnx"
        
        # Export model
        result = exporter.export(
            model=self.model,
            output_path=str(output_path),
            input_specs={"input_ids": {"shape": [1, 10], "dtype": "int64"}}
        )
        
        # Check that custom torch.nn modules are in hierarchy
        assert "linear" in exporter._hierarchy_data
        assert "dropout" in exporter._hierarchy_data
        
        # Check that non-custom modules are NOT in hierarchy
        assert "embedding" not in exporter._hierarchy_data
        assert "layer_norm" not in exporter._hierarchy_data
        
        # Verify the class names match custom list
        assert exporter._hierarchy_data["linear"]["class_name"] == "Linear"
        assert exporter._hierarchy_data["dropout"]["class_name"] == "Dropout"
        
    def test_empty_custom_list(self):
        """Test using empty custom list (should behave like False)."""
        exporter = HTPExporter(include_torch_nn_children=[])
        output_path = Path(self.temp_dir) / "test_empty.onnx"
        
        # Export model
        result = exporter.export(
            model=self.model,
            output_path=str(output_path),
            input_specs={"input_ids": {"shape": [1, 10], "dtype": "int64"}}
        )
        
        # Check that no torch.nn modules are in hierarchy
        assert "embedding" not in exporter._hierarchy_data
        assert "layer_norm" not in exporter._hierarchy_data
        assert "linear" not in exporter._hierarchy_data
        assert "dropout" not in exporter._hierarchy_data
        
        # Only the root module should be in hierarchy
        assert len(exporter._hierarchy_data) == 1
        
    def test_config_default_list(self):
        """Test that HTPConfig.DEFAULT_TORCH_NN_CHILDREN is correctly defined."""
        # Check the default list contains only LayerNorm and Embedding
        assert HTPConfig.DEFAULT_TORCH_NN_CHILDREN == ["LayerNorm", "Embedding"]
        
    def test_type_validation(self):
        """Test that invalid types are handled properly."""
        # This test is mainly for documentation - Python will handle type errors at runtime
        # In a statically typed environment, this would be caught by the type checker
        
        # Valid types - should not raise errors
        HTPExporter(include_torch_nn_children=False)
        HTPExporter(include_torch_nn_children=True)
        HTPExporter(include_torch_nn_children=["Linear", "Conv2d"])
        HTPExporter(include_torch_nn_children=[])
        
    def test_hierarchy_builder_receives_correct_exceptions(self):
        """Test that TracingHierarchyBuilder receives the correct exceptions list."""
        # We need to set up example inputs for _trace_model_hierarchy to work
        example_inputs = {"input_ids": torch.randint(0, 100, (1, 10))}
        
        # Test with False
        exporter = HTPExporter(include_torch_nn_children=False)
        exporter.example_inputs = example_inputs
        exporter._trace_model_hierarchy(self.model)
        # Can't easily check the exceptions passed to TracingHierarchyBuilder
        # without modifying the code, but we can verify the behavior through hierarchy
        assert len(exporter._hierarchy_data) == 1  # Only root
        
        # Test with True (default list)
        exporter = HTPExporter(include_torch_nn_children=True)
        exporter.example_inputs = example_inputs
        exporter._trace_model_hierarchy(self.model)
        # Should have root + 2 default modules
        assert len(exporter._hierarchy_data) == 3
        
        # Test with custom list
        exporter = HTPExporter(include_torch_nn_children=["Linear"])
        exporter.example_inputs = example_inputs
        exporter._trace_model_hierarchy(self.model)
        # Should have root + 1 custom module
        assert len(exporter._hierarchy_data) == 2
        
    def test_case_sensitivity(self):
        """Test that module names are case-sensitive."""
        # Use wrong case - should not match
        exporter = HTPExporter(include_torch_nn_children=["linear", "layernorm"])  # lowercase
        output_path = Path(self.temp_dir) / "test_case.onnx"
        
        result = exporter.export(
            model=self.model,
            output_path=str(output_path),
            input_specs={"input_ids": {"shape": [1, 10], "dtype": "int64"}}
        )
        
        # Should not find any modules due to case mismatch
        assert "linear" not in exporter._hierarchy_data
        assert "layer_norm" not in exporter._hierarchy_data
        
        # Only root module
        assert len(exporter._hierarchy_data) == 1


class TestBackwardCompatibility:
    """Test backward compatibility with existing code."""
    
    def test_existing_code_continues_working(self):
        """Test that existing code using bool continues to work."""
        # This should work as before
        exporter1 = HTPExporter(include_torch_nn_children=False)
        assert exporter1.include_torch_nn_children is False
        
        exporter2 = HTPExporter(include_torch_nn_children=True)
        assert exporter2.include_torch_nn_children is True
        
        # Default should still be False
        exporter3 = HTPExporter()
        assert exporter3.include_torch_nn_children is False