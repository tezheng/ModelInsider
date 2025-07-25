"""
Test runtime validation of HTP metadata.

This test verifies that the runtime validation catches errors
before metadata is written to disk.
"""

import pytest
from modelexport.strategies.htp_new.validation_utils import (
    validate_metadata,
    validate_before_write,
    MetadataValidationError
)


class TestRuntimeValidation:
    """Test runtime validation functionality."""
    
    def test_timestamp_validation(self):
        """Test that invalid timestamp formats are caught."""
        invalid_metadata = {
            "export_context": {
                "timestamp": "2025-07-22 12:00:00",  # Invalid format (missing T and Z)
                "strategy": "htp"
            },
            "model": {
                "name_or_path": "test",
                "class_name": "TestModel",
                "total_modules": 1,
                "total_parameters": 100
            },
            "tracing": {
                "builder": "TracingHierarchyBuilder",
                "modules_traced": 1,
                "execution_steps": 5
            },
            "modules": {
                "class_name": "TestModel",
                "traced_tag": "/TestModel",
                "scope": ""
            },
            "nodes": {},
            "outputs": {
                "onnx_model": {
                    "path": "test.onnx",
                    "size_mb": 1.0,
                    "opset_version": 17
                },
                "metadata": {
                    "path": "test_metadata.json"
                }
            },
            "report": {
                "steps": {}
            },
            "statistics": {
                "export_time": 1.0,
                "hierarchy_modules": 1,
                "traced_modules": 1,
                "onnx_nodes": 10,
                "tagged_nodes": 8,
                "empty_tags": 0,
                "coverage_percentage": 80.0
            }
        }
        
        with pytest.raises(MetadataValidationError) as exc_info:
            validate_before_write(invalid_metadata)
        
        assert "timestamp" in str(exc_info.value)
    
    def test_cross_field_validation(self):
        """Test that cross-field validation catches logical errors."""
        invalid_metadata = {
            "export_context": {
                "timestamp": "2025-07-22T12:00:00Z",
                "strategy": "htp"
            },
            "model": {
                "name_or_path": "test",
                "class_name": "TestModel",
                "total_modules": 1,
                "total_parameters": 100
            },
            "tracing": {
                "builder": "TracingHierarchyBuilder",
                "modules_traced": 1,
                "execution_steps": 5
            },
            "modules": {
                "class_name": "TestModel",
                "traced_tag": "/TestModel",
                "scope": ""
            },
            "nodes": {},
            "outputs": {
                "onnx_model": {
                    "path": "test.onnx",
                    "size_mb": 1.0,
                    "opset_version": 17
                },
                "metadata": {
                    "path": "test_metadata.json"
                }
            },
            "report": {
                "steps": {}
            },
            "statistics": {
                "export_time": 1.0,
                "hierarchy_modules": 1,
                "traced_modules": 1,
                "onnx_nodes": 100,
                "tagged_nodes": 150,  # More than total!
                "empty_tags": 0,
                "coverage_percentage": 80.0  # Should be 150%!
            }
        }
        
        with pytest.raises(MetadataValidationError) as exc_info:
            validate_before_write(invalid_metadata)
        
        error_msg = str(exc_info.value)
        assert "tagged_nodes" in error_msg and "exceed" in error_msg
    
    def test_empty_tags_validation(self):
        """Test that empty_tags must be 0 for HTP strategy."""
        invalid_metadata = {
            "export_context": {
                "timestamp": "2025-07-22T12:00:00Z",
                "strategy": "htp"
            },
            "model": {
                "name_or_path": "test",
                "class_name": "TestModel",
                "total_modules": 1,
                "total_parameters": 100
            },
            "tracing": {
                "builder": "TracingHierarchyBuilder",
                "modules_traced": 1,
                "execution_steps": 5
            },
            "modules": {
                "class_name": "TestModel",
                "traced_tag": "/TestModel",
                "scope": ""
            },
            "nodes": {},
            "outputs": {
                "onnx_model": {
                    "path": "test.onnx",
                    "size_mb": 1.0,
                    "opset_version": 17
                },
                "metadata": {
                    "path": "test_metadata.json"
                }
            },
            "report": {
                "steps": {}
            },
            "statistics": {
                "export_time": 1.0,
                "hierarchy_modules": 1,
                "traced_modules": 1,
                "onnx_nodes": 100,
                "tagged_nodes": 80,
                "empty_tags": 5,  # Should be 0!
                "coverage_percentage": 80.0
            }
        }
        
        with pytest.raises(MetadataValidationError) as exc_info:
            validate_before_write(invalid_metadata)
        
        error_msg = str(exc_info.value)
        assert "empty_tags" in error_msg and ("must be 0" in error_msg or "maximum of 0" in error_msg)
    
    def test_valid_metadata_passes(self):
        """Test that valid metadata passes validation."""
        valid_metadata = {
            "export_context": {
                "timestamp": "2025-07-22T12:00:00Z",
                "strategy": "htp"
            },
            "model": {
                "name_or_path": "test",
                "class_name": "TestModel",
                "total_modules": 1,
                "total_parameters": 100
            },
            "tracing": {
                "builder": "TracingHierarchyBuilder",
                "modules_traced": 1,
                "execution_steps": 5
            },
            "modules": {
                "class_name": "TestModel",
                "traced_tag": "/TestModel",
                "scope": ""
            },
            "nodes": {},
            "outputs": {
                "onnx_model": {
                    "path": "test.onnx",
                    "size_mb": 1.0,
                    "opset_version": 17
                },
                "metadata": {
                    "path": "test_metadata.json"
                }
            },
            "report": {
                "steps": {}
            },
            "statistics": {
                "export_time": 1.0,
                "hierarchy_modules": 1,
                "traced_modules": 1,
                "onnx_nodes": 100,
                "tagged_nodes": 80,
                "empty_tags": 0,
                "coverage_percentage": 80.0
            }
        }
        
        # Should not raise any exception
        validate_before_write(valid_metadata)