"""
Test runtime schema validation for HTP metadata.

This test demonstrates that we need runtime validation to catch
schema violations before writing metadata to disk.
"""

import json
from pathlib import Path

import jsonschema
import pytest

from modelexport.strategies.htp.metadata_builder import (
    HTPMetadataBuilder,
    ModelInfo,
    OnnxModelOutput,
    OutputFiles,
)


class TestMetadataSchemaValidation:
    """Test that metadata conforms to schema."""
    
    @pytest.fixture
    def schema(self):
        """Load the HTP metadata schema."""
        schema_path = Path(__file__).parent.parent / "modelexport/strategies/htp/htp_metadata_schema.json"
        with open(schema_path) as f:
            return json.load(f)
    
    def test_timestamp_format_validation(self, schema):
        """Test that timestamps must be in ISO format."""
        builder = HTPMetadataBuilder()
        
        # This should fail validation because timestamp is not ISO format
        builder._export_context.timestamp = "2025-07-22 12:00:00"  # Wrong format!
        
        # Build minimal metadata
        builder._model_info = ModelInfo(
            name_or_path="test",
            class_name="TestModel", 
            total_modules=1,
            total_parameters=100
        )
        
        # Add required output files
        builder._output_files = OutputFiles(
            onnx_model=OnnxModelOutput(path="test.onnx")
        )
        
        # This metadata has invalid timestamp format
        # But currently, there's no validation to catch this!
        metadata = builder.build()
        
        # This test shows we NEED validation
        with pytest.raises(jsonschema.ValidationError) as exc_info:
            jsonschema.validate(metadata, schema)
        
        # The error should be about timestamp format
        assert "timestamp" in str(exc_info.value)
    
    def test_missing_required_fields(self, schema):
        """Test that required fields are enforced."""
        # Create metadata missing required fields
        incomplete_metadata = {
            "export_context": {
                "strategy": "htp"
                # Missing required "timestamp"!
            },
            "model": {
                "name_or_path": "test",
                "class_name": "TestModel"
                # Missing required fields!
            }
        }
        
        with pytest.raises(jsonschema.ValidationError) as exc_info:
            jsonschema.validate(incomplete_metadata, schema)
        
        assert "required" in str(exc_info.value).lower()
    
    def test_numeric_constraints(self, schema):
        """Test that numeric constraints are enforced."""
        builder = HTPMetadataBuilder()
        
        # Set invalid numeric values
        builder.with_statistics(
            export_time=-1.0,  # Negative time!
            hierarchy_modules=-5,  # Negative count!
            traced_modules=10,  # Add required parameter
            onnx_nodes=100,
            tagged_nodes=150,  # More than total!
            empty_tags=10,  # Should always be 0!
            coverage_percentage=150.0,  # >100%!
            module_types=[]
        )
        
        # Build partial metadata
        builder._model_info = ModelInfo(
            name_or_path="test",
            class_name="TestModel",
            total_modules=1,
            total_parameters=100
        )
        
        # Add required output files
        builder._output_files = OutputFiles(
            onnx_model=OnnxModelOutput(path="test.onnx")
        )
        
        # This shows we need validation to catch these errors
        # Currently these invalid values would be written to disk!
        metadata = builder.build()
        
        # Test that schema validation would catch these issues
        with pytest.raises(jsonschema.ValidationError):
            jsonschema.validate(metadata, schema)
    
    def test_cross_field_validation_needed(self):
        """Demonstrate need for cross-field validation."""
        # This test shows relationships that should be validated:
        # 1. tagged_nodes <= total_onnx_nodes
        # 2. coverage_percentage = (tagged_nodes / total_onnx_nodes) * 100
        # 3. empty_tags should always be 0 for HTP
        
        invalid_relationships = {
            "statistics": {
                "onnx_nodes": 100,
                "tagged_nodes": 150,  # More than total!
                "coverage_percentage": 50.0,  # Doesn't match calculation!
                "empty_tags": 5  # Should be 0!
            }
        }
        
        # Currently, this would pass basic schema validation
        # but violates logical constraints!


def validate_metadata_before_write(metadata: dict, schema: dict) -> None:
    """
    Example of what we should add to production code.
    
    This function should be called in MetadataWriter.flush()
    before writing JSON to disk.
    """
    try:
        jsonschema.validate(metadata, schema)
    except jsonschema.ValidationError as e:
        # Provide user-friendly error message
        path = " -> ".join(str(p) for p in e.path)
        raise ValueError(
            f"Invalid metadata at {path}: {e.message}\n"
            f"Please check your export configuration."
        ) from e
    
    # Additional cross-field validations
    if "statistics" in metadata:
        stats = metadata["statistics"]
        if stats.get("tagged_nodes", 0) > stats.get("onnx_nodes", 0):
            raise ValueError(
                "Invalid metadata: tagged_nodes cannot exceed total onnx_nodes"
            )
        
        # Verify coverage calculation
        if stats.get("onnx_nodes", 0) > 0:
            expected_coverage = (stats.get("tagged_nodes", 0) / stats["onnx_nodes"]) * 100
            actual_coverage = stats.get("coverage_percentage", 0)
            if abs(expected_coverage - actual_coverage) > 0.1:  # Allow small float errors
                raise ValueError(
                    f"Invalid metadata: coverage_percentage {actual_coverage} "
                    f"doesn't match calculation {expected_coverage}"
                )