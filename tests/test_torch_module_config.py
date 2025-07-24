"""
Test the flexible torch_module configuration for HTP exporter.

Tests the regression fix for TEZ-47 where torch_module should be:
- bool: False (default), True (use default list)
- list[str]: Custom list of torch.nn module types to include

NOTE: After TEZ-24 fix, ALL modules are included in hierarchy for complete reports.
The torch_module parameter is preserved for API compatibility but currently has no effect.
"""

import tempfile
from pathlib import Path

import torch
import torch.nn as nn

from modelexport.strategies.htp_new.htp_exporter import HTPConfig, HTPExporter


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


class TestTorchModuleConfig:
    """Test the flexible torch_module configuration."""
    
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
        """Test default behavior (torch_module=False) - MUST-002: torch.nn modules excluded."""
        exporter = HTPExporter(torch_module=False)
        output_path = Path(self.temp_dir) / "test_false.onnx"
        
        # Export model
        exporter.export(
            model=self.model,
            output_path=str(output_path),
            input_specs={"input_ids": {"shape": [1, 10], "dtype": "int64"}}
        )
        
        # MUST-002: torch.nn modules should NOT be included when torch_module=False
        assert "embedding" not in exporter._hierarchy_data
        assert "layer_norm" not in exporter._hierarchy_data
        assert "linear" not in exporter._hierarchy_data
        assert "dropout" not in exporter._hierarchy_data
        
        # Should only have root module
        assert len(exporter._hierarchy_data) == 1
        assert "" in exporter._hierarchy_data  # Root module
        
    def test_default_list_true(self):
        """Test using default list (torch_module=True) - includes default torch.nn exceptions."""
        exporter = HTPExporter(torch_module=True)
        output_path = Path(self.temp_dir) / "test_true.onnx"
        
        # Export model
        exporter.export(
            model=self.model,
            output_path=str(output_path),
            input_specs={"input_ids": {"shape": [1, 10], "dtype": "int64"}}
        )
        
        # When torch_module=True, should include modules from DEFAULT_TORCH_MODULES
        # Default list is ["LayerNorm", "Embedding"]
        assert "embedding" in exporter._hierarchy_data  # Embedding is in default list
        assert "layer_norm" in exporter._hierarchy_data  # LayerNorm is in default list
        
        # But Linear and Dropout are NOT in default list
        assert "linear" not in exporter._hierarchy_data
        assert "dropout" not in exporter._hierarchy_data
        
        # Verify the class names for included modules
        assert exporter._hierarchy_data["embedding"]["class_name"] == "Embedding"
        assert exporter._hierarchy_data["layer_norm"]["class_name"] == "LayerNorm"
        
    def test_custom_list(self):
        """Test using custom list of torch.nn modules - only specified modules included."""
        custom_list = ["Linear", "Dropout"]
        exporter = HTPExporter(torch_module=custom_list)
        output_path = Path(self.temp_dir) / "test_custom.onnx"
        
        # Export model
        exporter.export(
            model=self.model,
            output_path=str(output_path),
            input_specs={"input_ids": {"shape": [1, 10], "dtype": "int64"}}
        )
        
        # Only modules in custom list should be included
        assert "linear" in exporter._hierarchy_data
        assert "dropout" in exporter._hierarchy_data
        # These are NOT in the custom list, so should NOT be included
        assert "embedding" not in exporter._hierarchy_data
        assert "layer_norm" not in exporter._hierarchy_data
        
        # Verify the class names for included modules
        assert exporter._hierarchy_data["linear"]["class_name"] == "Linear"
        assert exporter._hierarchy_data["dropout"]["class_name"] == "Dropout"
        
        # Should have root + 2 specified torch.nn modules
        assert len(exporter._hierarchy_data) == 3
        
    def test_empty_custom_list(self):
        """Test using empty custom list - no torch.nn modules included."""
        exporter = HTPExporter(torch_module=[])
        output_path = Path(self.temp_dir) / "test_empty.onnx"
        
        # Export model
        exporter.export(
            model=self.model,
            output_path=str(output_path),
            input_specs={"input_ids": {"shape": [1, 10], "dtype": "int64"}}
        )
        
        # Empty list means NO torch.nn modules should be included
        assert "embedding" not in exporter._hierarchy_data
        assert "layer_norm" not in exporter._hierarchy_data
        assert "linear" not in exporter._hierarchy_data
        assert "dropout" not in exporter._hierarchy_data
        
        # Should have only root module
        assert len(exporter._hierarchy_data) == 1
        
    def test_config_default_list(self):
        """Test that HTPConfig.DEFAULT_TORCH_MODULES is correctly defined."""
        # Check the default list contains only LayerNorm and Embedding
        assert HTPConfig.DEFAULT_TORCH_MODULES == ["LayerNorm", "Embedding"]
        
    def test_type_validation(self):
        """Test that invalid types are handled properly."""
        # This test is mainly for documentation - Python will handle type errors at runtime
        # In a statically typed environment, this would be caught by the type checker
        
        # Valid types - should not raise errors
        HTPExporter(torch_module=False)
        HTPExporter(torch_module=True)
        HTPExporter(torch_module=["Linear", "Conv2d"])
        HTPExporter(torch_module=[])
        
    def test_hierarchy_builder_receives_correct_exceptions(self):
        """Test that TracingHierarchyBuilder receives correct exceptions parameter."""
        # We need to set up example inputs for _trace_model_hierarchy to work
        example_inputs = {"input_ids": torch.randint(0, 100, (1, 10))}
        
        # Test with False (no torch.nn modules)
        exporter = HTPExporter(torch_module=False)
        exporter.example_inputs = example_inputs
        exporter._trace_model_hierarchy(self.model)
        # Should have only root module
        assert len(exporter._hierarchy_data) == 1
        
        # Test with True (default torch.nn modules: LayerNorm, Embedding)
        exporter = HTPExporter(torch_module=True)
        exporter.example_inputs = example_inputs
        exporter._trace_model_hierarchy(self.model)
        # Should have root + 2 default modules (LayerNorm, Embedding)
        assert len(exporter._hierarchy_data) == 3
        assert "embedding" in exporter._hierarchy_data
        assert "layer_norm" in exporter._hierarchy_data
        assert "linear" not in exporter._hierarchy_data
        assert "dropout" not in exporter._hierarchy_data
        
        # Test with custom list (only Linear)
        exporter = HTPExporter(torch_module=["Linear"])
        exporter.example_inputs = example_inputs
        exporter._trace_model_hierarchy(self.model)
        # Should have root + 1 specified module
        assert len(exporter._hierarchy_data) == 2
        assert "linear" in exporter._hierarchy_data
        assert "embedding" not in exporter._hierarchy_data
        assert "layer_norm" not in exporter._hierarchy_data
        assert "dropout" not in exporter._hierarchy_data
        
    def test_case_sensitivity(self):
        """Test module names are case-sensitive."""
        # Use wrong case - should NOT match
        exporter = HTPExporter(torch_module=["linear", "layernorm"])  # lowercase
        output_path = Path(self.temp_dir) / "test_case.onnx"
        
        exporter.export(
            model=self.model,
            output_path=str(output_path),
            input_specs={"input_ids": {"shape": [1, 10], "dtype": "int64"}}
        )
        
        # Module names are case-sensitive, so lowercase won't match
        assert "linear" not in exporter._hierarchy_data  # "linear" != "Linear"
        assert "layer_norm" not in exporter._hierarchy_data  # "layernorm" != "LayerNorm"
        assert "embedding" not in exporter._hierarchy_data
        assert "dropout" not in exporter._hierarchy_data
        
        # Should have only root module (no torch.nn matched)
        assert len(exporter._hierarchy_data) == 1


class TestBackwardCompatibility:
    """Test backward compatibility with existing code."""
    
    def test_existing_code_continues_working(self):
        """Test that existing code using bool continues to work."""
        # This should work as before
        exporter1 = HTPExporter(torch_module=False)
        assert exporter1.torch_module is False
        
        exporter2 = HTPExporter(torch_module=True)
        assert exporter2.torch_module is True
        
        # Default should still be False
        exporter3 = HTPExporter()
        assert exporter3.torch_module is False