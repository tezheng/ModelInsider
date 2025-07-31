"""
Validation utilities for HTP metadata.

This module provides runtime validation to ensure all generated
metadata conforms to the schema before writing to disk.
"""

import json
from pathlib import Path
from typing import Any

try:
    import jsonschema
except ImportError:
    jsonschema = None


class MetadataValidationError(ValueError):
    """Raised when metadata validation fails."""
    pass


def load_schema() -> dict[str, Any]:
    """Load the HTP metadata schema."""
    schema_path = Path(__file__).parent / "htp_metadata_schema.json"
    with open(schema_path) as f:
        return json.load(f)


def validate_metadata(metadata: dict[str, Any], schema: dict[str, Any] | None = None) -> None:
    """
    Validate metadata against the HTP schema.
    
    Args:
        metadata: The metadata dictionary to validate
        schema: Optional schema dict. If not provided, loads default schema.
        
    Raises:
        MetadataValidationError: If validation fails
        ImportError: If jsonschema is not installed
    """
    if jsonschema is None:
        # For now, just log a warning and return
        # In production, we might want to make this a hard requirement
        import warnings
        warnings.warn(
            "jsonschema not installed. Skipping metadata validation. "
            "Install with: pip install jsonschema", stacklevel=2
        )
        return
    
    if schema is None:
        schema = load_schema()
    
    try:
        jsonschema.validate(metadata, schema)
    except jsonschema.ValidationError as e:
        # Provide user-friendly error message
        path = " -> ".join(str(p) for p in e.path) if e.path else "root"
        raise MetadataValidationError(
            f"Invalid metadata at {path}: {e.message}\n"
            f"Please check your export configuration."
        ) from e
    
    # Additional cross-field validations
    _validate_cross_fields(metadata)


def _validate_cross_fields(metadata: dict[str, Any]) -> None:
    """
    Validate relationships between fields.
    
    Args:
        metadata: The metadata dictionary to validate
        
    Raises:
        MetadataValidationError: If cross-field validation fails
    """
    # Validate statistics relationships
    if "statistics" in metadata:
        stats = metadata["statistics"]
        
        # Check tagged_nodes <= total_onnx_nodes
        tagged = stats.get("tagged_nodes", 0)
        total = stats.get("onnx_nodes", 0)
        if tagged > total:
            raise MetadataValidationError(
                f"Invalid statistics: tagged_nodes ({tagged}) cannot exceed "
                f"total onnx_nodes ({total})"
            )
        
        # Verify coverage calculation
        if total > 0:
            expected_coverage = (tagged / total) * 100
            actual_coverage = stats.get("coverage_percentage", 0)
            if abs(expected_coverage - actual_coverage) > 0.1:  # Allow small float errors
                raise MetadataValidationError(
                    f"Invalid statistics: coverage_percentage {actual_coverage:.2f}% "
                    f"doesn't match calculation {expected_coverage:.2f}%"
                )
        
        # Verify empty_tags is 0 for HTP
        empty_tags = stats.get("empty_tags", 0)
        if empty_tags != 0:
            raise MetadataValidationError(
                f"Invalid statistics: empty_tags must be 0 for HTP strategy, "
                f"but got {empty_tags}"
            )
    
    # Validate node tagging statistics if present
    if "report" in metadata and "steps" in metadata["report"]:
        steps = metadata["report"]["steps"]
        if "node_tagging" in steps:
            node_tagging = steps["node_tagging"]
            
            # Check coverage consistency
            if "coverage" in node_tagging:
                coverage = node_tagging["coverage"]
                tagged = coverage.get("tagged_nodes", 0)
                total = coverage.get("total_onnx_nodes", 0)
                percentage = coverage.get("percentage", 0)
                
                if total > 0:
                    expected = (tagged / total) * 100
                    if abs(expected - percentage) > 0.1:
                        raise MetadataValidationError(
                            f"Invalid node_tagging coverage: percentage {percentage:.2f}% "
                            f"doesn't match calculation {expected:.2f}%"
                        )
    
    # Validate traced modules count
    if "modules" in metadata and "statistics" in metadata:
        # Count modules in hierarchical structure (these are traced modules)
        module_count = _count_modules(metadata["modules"])
        traced_count = metadata["statistics"].get("traced_modules", 0)
        
        if module_count != traced_count:
            raise MetadataValidationError(
                f"Invalid statistics: traced_modules ({traced_count}) "
                f"doesn't match actual module count ({module_count})"
            )
        
        # Validate that traced_modules <= hierarchy_modules
        hierarchy_count = metadata["statistics"].get("hierarchy_modules", 0)
        if traced_count > hierarchy_count:
            raise MetadataValidationError(
                f"Invalid statistics: traced_modules ({traced_count}) "
                f"cannot exceed hierarchy_modules ({hierarchy_count})"
            )


def _count_modules(module: dict[str, Any]) -> int:
    """Count total modules in hierarchical structure."""
    if not module or not isinstance(module, dict):
        return 0
    
    count = 1  # Count current module
    
    # Count children recursively
    if "children" in module and isinstance(module["children"], dict):
        for child in module["children"].values():
            count += _count_modules(child)
    
    return count


def validate_before_write(metadata: dict[str, Any]) -> None:
    """
    Convenience function to validate metadata before writing.
    
    This should be called in MetadataWriter.flush() before json.dump().
    
    Args:
        metadata: The metadata dictionary to validate
        
    Raises:
        MetadataValidationError: If validation fails
    """
    validate_metadata(metadata)