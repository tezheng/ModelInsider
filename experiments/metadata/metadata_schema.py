"""
JSON Schema generation for HTP metadata.

This module provides JSON schema for validation and documentation.
"""

from typing import Any

try:
    from .metadata_models import HTPMetadataModel
except ImportError:
    # For standalone execution
    from metadata_models import HTPMetadataModel


def get_metadata_schema() -> dict[str, Any]:
    """
    Get the JSON schema for HTP metadata.
    
    Returns:
        JSON schema dictionary that can be used for validation.
    """
    return HTPMetadataModel.model_json_schema()


def save_metadata_schema(output_path: str) -> None:
    """
    Save the metadata schema to a file.
    
    Args:
        output_path: Path to save the schema JSON file.
    """
    import json
    
    schema = get_metadata_schema()
    
    # Add custom metadata
    schema["$id"] = "https://github.com/user/modelexport/schemas/htp-metadata-v1.0.json"
    schema["$comment"] = "Schema for HTP (Hierarchy-preserving Tags Protocol) export metadata"
    
    with open(output_path, "w") as f:
        json.dump(schema, f, indent=2)


def validate_metadata_file(metadata_path: str) -> tuple[bool, list[str]]:
    """
    Validate a metadata file against the schema.
    
    Args:
        metadata_path: Path to the metadata JSON file.
        
    Returns:
        Tuple of (is_valid, error_messages)
    """
    import json
    from pydantic import ValidationError
    
    try:
        with open(metadata_path) as f:
            data = json.load(f)
        
        # Validate against schema
        HTPMetadataModel.model_validate(data)
        return True, []
        
    except ValidationError as e:
        errors = []
        for error in e.errors():
            loc = " -> ".join(str(l) for l in error["loc"])
            errors.append(f"{loc}: {error['msg']}")
        return False, errors
        
    except Exception as e:
        return False, [str(e)]


if __name__ == "__main__":
    # Generate schema file
    save_metadata_schema("htp_metadata_schema.json")
    print("Schema saved to htp_metadata_schema.json")