"""
Test JSON Schema generation with Pydantic models.
"""

import json
from pydantic import BaseModel, Field


# Simple example model
class ExportContext(BaseModel):
    """Export context with Pydantic."""
    timestamp: str = Field(description="Export timestamp in ISO format")
    strategy: str = Field(default="htp", pattern="^htp$", description="Must be 'htp'")
    version: str = Field(default="1.0", pattern=r"^\d+\.\d+$", description="Version X.Y format")
    embed_hierarchy_attributes: bool = Field(default=True, description="Embed tags in ONNX")


class ModelInfo(BaseModel):
    """Model information."""
    name_or_path: str = Field(description="Model name or HuggingFace path")
    class_: str = Field(alias="class", description="Model class name")
    total_modules: int = Field(ge=0, description="Total number of modules")
    total_parameters: int = Field(ge=0, description="Total parameter count")


class TaggingCoverage(BaseModel):
    """Coverage statistics with validation."""
    total_onnx_nodes: int = Field(ge=0, description="Total ONNX nodes")
    tagged_nodes: int = Field(ge=0, description="Number of tagged nodes")
    coverage_percentage: float = Field(ge=0.0, le=100.0, description="Coverage percentage")
    empty_tags: int = Field(ge=0, description="Number of empty tags")
    
    @property
    def is_complete(self) -> bool:
        """Check if coverage is complete."""
        return self.coverage_percentage >= 95.0 and self.empty_tags == 0


class HTPMetadata(BaseModel):
    """Complete HTP metadata model."""
    export_context: ExportContext
    model: ModelInfo
    coverage: TaggingCoverage


def demonstrate_json_schema():
    """Show JSON Schema generation capabilities."""
    print("=== JSON SCHEMA GENERATION ===\n")
    
    # Generate schema for the complete model
    schema = HTPMetadata.model_json_schema()
    
    print("Generated JSON Schema:")
    print(json.dumps(schema, indent=2))
    
    # Save schema to file
    schema_path = "experiments/metadata/htp_metadata_schema.json"
    with open(schema_path, "w") as f:
        json.dump(schema, f, indent=2)
    print(f"\nSchema saved to: {schema_path}")
    
    return schema


def demonstrate_validation():
    """Show validation capabilities."""
    print("\n\n=== VALIDATION EXAMPLES ===\n")
    
    # Valid example
    print("1. Valid metadata:")
    try:
        valid_metadata = HTPMetadata(
            export_context=ExportContext(
                timestamp="2024-01-01T00:00:00Z",
                strategy="htp",
                version="2.0"
            ),
            model=ModelInfo(
                name_or_path="prajjwal1/bert-tiny",
                **{"class": "BertModel"},  # Using alias
                total_modules=42,
                total_parameters=4365312
            ),
            coverage=TaggingCoverage(
                total_onnx_nodes=1000,
                tagged_nodes=950,
                coverage_percentage=95.0,
                empty_tags=0
            )
        )
        print("✅ Valid! Coverage complete:", valid_metadata.coverage.is_complete)
    except Exception as e:
        print(f"❌ Validation error: {e}")
    
    # Invalid examples
    print("\n2. Invalid version format:")
    try:
        ExportContext(version="1.2.3")  # Should be X.Y
    except Exception as e:
        print(f"✅ Caught: {e}")
    
    print("\n3. Invalid coverage percentage:")
    try:
        TaggingCoverage(
            total_onnx_nodes=100,
            tagged_nodes=50,
            coverage_percentage=150.0,  # > 100!
            empty_tags=0
        )
    except Exception as e:
        print(f"✅ Caught: {e}")
    
    print("\n4. Invalid strategy:")
    try:
        ExportContext(strategy="wrong")  # Must be 'htp'
    except Exception as e:
        print(f"✅ Caught: {e}")


def demonstrate_serialization():
    """Show serialization options."""
    print("\n\n=== SERIALIZATION OPTIONS ===\n")
    
    metadata = HTPMetadata(
        export_context=ExportContext(timestamp="2024-01-01T00:00:00Z"),
        model=ModelInfo(
            name_or_path="model",
            **{"class": "TestModel"},
            total_modules=10,
            total_parameters=1000
        ),
        coverage=TaggingCoverage(
            total_onnx_nodes=100,
            tagged_nodes=95,
            coverage_percentage=95.0,
            empty_tags=0
        )
    )
    
    # Different serialization options
    print("1. Standard JSON:")
    print(metadata.model_dump_json(indent=2))
    
    print("\n2. With aliases (note 'class' field):")
    print(metadata.model_dump_json(indent=2, by_alias=True))
    
    print("\n3. Exclude defaults:")
    print(metadata.model_dump_json(indent=2, exclude_defaults=True))


if __name__ == "__main__":
    # Generate JSON Schema
    schema = demonstrate_json_schema()
    
    # Show validation
    demonstrate_validation()
    
    # Show serialization
    demonstrate_serialization()
    
    print("\n\n=== BENEFITS SUMMARY ===")
    print("1. ✅ Automatic JSON Schema generation")
    print("2. ✅ Built-in validation with clear errors")
    print("3. ✅ Field constraints (min/max, patterns)")
    print("4. ✅ Property support (@property)")
    print("5. ✅ Alias support for reserved words")
    print("6. ✅ Multiple serialization options")