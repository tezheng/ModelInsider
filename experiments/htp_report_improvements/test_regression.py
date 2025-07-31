"""
Test that our changes don't break existing HTP functionality.
"""

import json
import sys
import tempfile
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from modelexport.strategies.htp.htp_exporter import HTPExporter
from modelexport.strategies.htp.metadata_builder import HTPMetadataBuilder


def test_existing_functionality():
    """Test that existing HTP export still works."""
    print("Testing existing HTP functionality...")
    
    # Test 1: HTPExporter can be instantiated
    exporter = HTPExporter(verbose=False)
    assert exporter is not None
    print("✅ HTPExporter instantiation works")
    
    # Test 2: Metadata builder still works
    builder = HTPMetadataBuilder()
    metadata = (
        builder
        .with_export_context(strategy="htp", version="1.0")
        .with_model_info(
            name_or_path="test-model",
            class_name="TestModel", 
            total_modules=10,
            total_parameters=1000
        )
        .with_modules({})
        .with_tagging_info(
            tagged_nodes={"node1": "/tag1"},
            statistics={},
            total_onnx_nodes=100,
            tagged_nodes_count=90,
            coverage_percentage=90.0,
            empty_tags=0
        )
        .with_output_files(
            onnx_path="test.onnx",
            onnx_size_mb=1.0,
            metadata_path="test_metadata.json"
        )
        .build()
    )
    
    # Verify old structure is still generated
    assert "tagging" in metadata
    assert "tagged_nodes" in metadata["tagging"]
    assert metadata["tagging"]["coverage"]["coverage_percentage"] == 90.0
    print("✅ Metadata builder produces expected structure")
    
    # Test 3: Verify we can read the metadata
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        json.dump(metadata, f)
        temp_path = f.name
    
    with open(temp_path) as f:
        loaded = json.load(f)
    
    assert loaded["tagging"]["tagged_nodes"]["node1"] == "/tag1"
    Path(temp_path).unlink()
    print("✅ Metadata can be saved and loaded")
    
    # Test 4: Test _output_message exists
    exporter._output_message("Test message")
    print("✅ _output_message method exists")
    
    # Test 5: Check required attributes exist
    assert hasattr(exporter, 'report_buffer')
    assert hasattr(exporter, 'verbose')
    assert hasattr(exporter, 'embed_hierarchy_attributes')
    print("✅ Required attributes exist")
    
    print("\nAll regression tests passed! ✅")
    print("Safe to proceed with modifications.")


def test_proposed_changes():
    """Test that proposed changes maintain compatibility."""
    print("\nTesting proposed changes compatibility...")
    
    # Simulate the new metadata structure
    old_metadata = {
        "tagging": {
            "tagged_nodes": {"node1": "/tag1"},
            "statistics": {"total": 100},
            "coverage": {"coverage_percentage": 90.0}
        }
    }
    
    # Function that would handle both old and new formats
    def migrate_metadata(metadata):
        """Migrate old format to new format if needed."""
        if "tagging" in metadata and "nodes" not in metadata:
            # Old format - migrate
            new_metadata = metadata.copy()
            new_metadata["nodes"] = metadata["tagging"]["tagged_nodes"]
            
            if "report" not in new_metadata:
                new_metadata["report"] = {}
            
            new_metadata["report"]["node_tagging"] = {
                "statistics": metadata["tagging"].get("statistics", {}),
                "coverage": metadata["tagging"].get("coverage", {})
            }
            
            del new_metadata["tagging"]
            return new_metadata
        return metadata
    
    new_metadata = migrate_metadata(old_metadata)
    
    # Verify migration worked
    assert "nodes" in new_metadata
    assert new_metadata["nodes"]["node1"] == "/tag1"
    assert new_metadata["report"]["node_tagging"]["coverage"]["coverage_percentage"] == 90.0
    assert "tagging" not in new_metadata
    
    print("✅ Metadata migration function works")
    
    # Test reverse compatibility - reading new format with old code expectations
    def read_nodes_compatible(metadata):
        """Read nodes from either format."""
        if "nodes" in metadata:
            return metadata["nodes"]
        elif "tagging" in metadata:
            return metadata["tagging"].get("tagged_nodes", {})
        return {}
    
    # Test with both formats
    assert read_nodes_compatible(old_metadata) == {"node1": "/tag1"}
    assert read_nodes_compatible(new_metadata) == {"node1": "/tag1"}
    
    print("✅ Compatible reading function works")
    print("\nCompatibility tests passed! ✅")


if __name__ == "__main__":
    test_existing_functionality()
    test_proposed_changes()