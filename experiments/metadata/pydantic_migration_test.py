"""
Test migrating from dataclass builder to Pydantic models.

This shows a side-by-side comparison and validates the output is identical.
"""

import json
import sys
from pathlib import Path

# Add parent paths to import the actual modules
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from modelexport.strategies.htp.metadata_builder import HTPMetadataBuilder
from experiments.metadata.pydantic_builder import HTPMetadataBuilder as PydanticBuilder


def test_dataclass_builder():
    """Test current dataclass approach."""
    print("=== DATACLASS BUILDER (Current) ===")
    
    builder = HTPMetadataBuilder()
    metadata = (
        builder
        .with_export_context(strategy="htp", version="1.0")
        .with_model_info(
            name_or_path="prajjwal1/bert-tiny",
            class_name="BertModel",
            total_modules=42,
            total_parameters=4365312
        )
        .with_tracing_info(
            modules_traced=42,
            execution_steps=100,
            inputs={
                "input_ids": {"shape": [1, 128], "dtype": "torch.int64"},
                "attention_mask": {"shape": [1, 128], "dtype": "torch.int64"}
            },
            outputs=["last_hidden_state"]
        )
        .with_modules({
            "embeddings": {
                "name": "embeddings",
                "class_name": "BertEmbeddings",
                "traced_tag": "/BertModel/BertEmbeddings"
            }
        })
        .with_tagging_info(
            tagged_nodes={"node1": "/BertModel/BertEmbeddings"},
            statistics={},
            total_onnx_nodes=1000,
            tagged_nodes_count=950,
            coverage_percentage=95.0,
            empty_tags=0
        )
        .with_output_files(
            onnx_path="model.onnx",
            onnx_size_mb=16.5,
            metadata_path="model_metadata.json"
        )
        .build()
    )
    
    print("Generated metadata:")
    print(json.dumps(metadata, indent=2)[:500] + "...")
    print(f"\nTotal size: {len(json.dumps(metadata))} bytes")
    return metadata


def test_pydantic_builder():
    """Test Pydantic approach."""
    print("\n=== PYDANTIC BUILDER (New) ===")
    
    builder = PydanticBuilder()
    metadata = (
        builder
        .with_export_context(strategy="htp", version="1.0")
        .with_model_info(
            name_or_path="prajjwal1/bert-tiny",
            class_name="BertModel",
            total_modules=42,
            total_parameters=4365312
        )
        .with_tracing_info(
            modules_traced=42,
            execution_steps=100,
            inputs={
                "input_ids": {"shape": [1, 128], "dtype": "torch.int64"},
                "attention_mask": {"shape": [1, 128], "dtype": "torch.int64"}
            },
            outputs=["last_hidden_state"]
        )
        .with_modules({
            "embeddings": {
                "name": "embeddings",
                "class_name": "BertEmbeddings",
                "traced_tag": "/BertModel/BertEmbeddings"
            }
        })
        .with_tagging_info(
            tagged_nodes={"node1": "/BertModel/BertEmbeddings"},
            statistics={},
            total_onnx_nodes=1000,
            tagged_nodes_count=950,
            coverage_percentage=95.0,
            empty_tags=0
        )
        .with_output_files(
            onnx_path="model.onnx",
            onnx_size_mb=16.5,
            metadata_path="model_metadata.json"
        )
        .build()
    )
    
    print("Generated metadata:")
    print(json.dumps(metadata, indent=2)[:500] + "...")
    print(f"\nTotal size: {len(json.dumps(metadata))} bytes")
    
    # Show JSON Schema capability
    print("\n--- JSON SCHEMA GENERATION ---")
    schema = builder.get_json_schema()
    print(json.dumps(schema, indent=2)[:300] + "...")
    
    return metadata


def compare_outputs(dataclass_meta, pydantic_meta):
    """Compare the two outputs."""
    print("\n=== COMPARISON ===")
    
    # Convert to JSON for comparison
    dc_json = json.dumps(dataclass_meta, sort_keys=True, indent=2)
    pd_json = json.dumps(pydantic_meta, sort_keys=True, indent=2)
    
    if dc_json == pd_json:
        print("✅ Outputs are IDENTICAL!")
    else:
        print("❌ Outputs differ")
        # Find differences
        dc_lines = dc_json.split('\n')
        pd_lines = pd_json.split('\n')
        for i, (dc, pd) in enumerate(zip(dc_lines, pd_lines)):
            if dc != pd:
                print(f"Line {i}: {dc} != {pd}")
                break
    
    print("\n=== PYDANTIC ADVANTAGES ===")
    print("1. ✅ Automatic validation during build")
    print("2. ✅ JSON Schema generation")
    print("3. ✅ Better field names ('class' instead of 'class_name')")
    print("4. ✅ Type checking and constraints")
    print("5. ✅ Serialization control")


def test_validation():
    """Test Pydantic validation."""
    print("\n=== VALIDATION EXAMPLE ===")
    
    builder = PydanticBuilder()
    
    # Try invalid coverage
    try:
        builder.with_tagging_info(
            tagged_nodes={},
            statistics={},
            total_onnx_nodes=100,
            tagged_nodes_count=50,
            coverage_percentage=150.0,  # Invalid!
            empty_tags=0
        )
    except Exception as e:
        print(f"✅ Validation caught invalid coverage: {e}")
    
    # Try invalid version pattern
    try:
        builder.with_export_context(version="1.2.3")  # Should be X.Y
    except Exception as e:
        print(f"✅ Validation caught invalid version: {e}")


if __name__ == "__main__":
    # Test both approaches
    dataclass_metadata = test_dataclass_builder()
    pydantic_metadata = test_pydantic_builder()
    
    # Compare outputs
    compare_outputs(dataclass_metadata, pydantic_metadata)
    
    # Test validation
    test_validation()