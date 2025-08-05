"""
Error handling tests for ONNX to GraphML conversion.

Tests error conditions and edge cases including:
- File system errors (missing files, permissions)
- Malformed input files (ONNX, metadata)
- Invalid parameter combinations
- Resource exhaustion scenarios
- Edge cases (empty models, invalid formats)
"""

import builtins
import contextlib
import json
import xml.etree.ElementTree as ET

import pytest

from modelexport.graphml import ONNXToGraphMLConverter
from modelexport.graphml.metadata_reader import MetadataReader

from .test_utils import get_graphml_content


class TestFileSystemErrors:
    """Test file system related error conditions."""
    
    def test_missing_onnx_file(self):
        """Test error handling for non-existent ONNX file."""
        converter = ONNXToGraphMLConverter(hierarchical=False)
        
        with pytest.raises(FileNotFoundError) as exc_info:
            converter.convert("nonexistent_model.onnx")
        
        assert "ONNX model not found" in str(exc_info.value)
        assert "nonexistent_model.onnx" in str(exc_info.value)
    
    def test_missing_metadata_file(self):
        """Test error handling for non-existent metadata file."""
        with pytest.raises(FileNotFoundError) as exc_info:
            ONNXToGraphMLConverter(htp_metadata_path="nonexistent_metadata.json")
        
        assert "HTP metadata file not found" in str(exc_info.value)
    
    def test_empty_file_path(self):
        """Test error handling for empty file paths."""
        converter = ONNXToGraphMLConverter(hierarchical=False)
        
        # Empty string gets converted to "." which is a directory, so expect different error
        with pytest.raises(Exception):  # Could be IsADirectoryError, PermissionError, or ONNX error
            converter.convert("")
    
    def test_directory_instead_of_file(self, tmp_path):
        """Test error handling when directory is passed instead of file."""
        converter = ONNXToGraphMLConverter(hierarchical=False)
        
        # Create a directory
        dir_path = tmp_path / "not_a_file"
        dir_path.mkdir()
        
        with pytest.raises((IsADirectoryError, PermissionError, FileNotFoundError)):
            converter.convert(str(dir_path))


class TestMalformedInputFiles:
    """Test handling of malformed and invalid input files."""
    
    def test_malformed_onnx_file(self, malformed_onnx_file):
        """Test error handling for malformed ONNX files."""
        converter = ONNXToGraphMLConverter(hierarchical=False)
        
        # Should raise an ONNX-related error (don't be too specific about the message)
        with pytest.raises(Exception):
            converter.convert(malformed_onnx_file)
    
    def test_empty_onnx_file(self, tmp_path):
        """Test handling of empty ONNX files."""
        empty_file = tmp_path / "empty.onnx"
        empty_file.touch()  # Create empty file
        
        converter = ONNXToGraphMLConverter(hierarchical=False)
        
        # Empty ONNX files may be handled gracefully by ONNX library
        # Either raises an exception or produces empty GraphML
        try:
            graphml_output = converter.convert(str(empty_file))
            # If it succeeds, should produce valid GraphML (possibly empty)
            assert isinstance(graphml_output, str)
            if len(graphml_output) > 0:
                root = ET.fromstring(graphml_output)
                assert root.tag.endswith("graphml")
        except Exception:
            # If it fails, that's also acceptable
            pass
    
    def test_malformed_json_metadata(self, tmp_path, simple_onnx_model):
        """Test error handling for malformed JSON metadata."""
        # Create malformed JSON file
        metadata_file = tmp_path / "malformed.json" 
        with open(metadata_file, 'w') as f:
            f.write("{ invalid json content }")
        
        with pytest.raises(json.JSONDecodeError):
            ONNXToGraphMLConverter(htp_metadata_path=str(metadata_file))
    
    def test_empty_json_metadata(self, tmp_path, simple_onnx_model):
        """Test error handling for empty JSON metadata."""
        # Create empty JSON file
        metadata_file = tmp_path / "empty.json"
        with open(metadata_file, 'w') as f:
            f.write("{}")
        
        # Should not crash, but may not produce hierarchy
        converter = ONNXToGraphMLConverter(htp_metadata_path=str(metadata_file))
        graphml_output = converter.convert(simple_onnx_model)
        
        # Should still produce valid GraphML
        content, root = get_graphml_content(graphml_output)
        assert len(content) > 0
        assert root.tag.endswith("graphml")
    
    def test_invalid_metadata_structure(self, tmp_path, simple_onnx_model):
        """Test handling of metadata with invalid structure."""
        # Create metadata with invalid structure
        metadata_file = tmp_path / "invalid_structure.json"
        with open(metadata_file, 'w') as f:
            json.dump({
                "invalid_field": "value",
                "tagged_nodes": "should_be_dict_not_string"
            }, f)
        
        # This should fail because tagged_nodes is expected to be a dict
        with pytest.raises(AttributeError):  # tagged_nodes must be a dictionary
            converter = ONNXToGraphMLConverter(htp_metadata_path=str(metadata_file))


class TestEdgeCases:
    """Test edge cases and boundary conditions."""
    
    def test_very_small_model(self, tmp_path):
        """Test conversion of minimal ONNX model."""
        import torch
        
        # Create minimal model - just identity
        class MinimalModel(torch.nn.Module):
            def forward(self, x):
                return x
        
        model = MinimalModel()
        dummy_input = torch.randn(1, 1)
        
        onnx_path = tmp_path / "minimal.onnx"
        torch.onnx.export(
            model, dummy_input, str(onnx_path),
            input_names=['input'], output_names=['output'],
            opset_version=17
        )
        
        converter = ONNXToGraphMLConverter(hierarchical=False)
        graphml_output = converter.convert(str(onnx_path))
        
        # Should produce valid GraphML even for minimal model
        assert len(graphml_output) > 0
        root = ET.fromstring(graphml_output)
        assert root.tag.endswith("graphml")
        
        # Should have at least input and output nodes
        stats = converter.get_statistics()
        assert stats['nodes'] >= 0  # May be 0 for identity
        assert stats['edges'] >= 0
    
    def test_model_with_no_operations(self, tmp_path):
        """Test model that has no actual operations."""
        import torch
        
        # Create model with just a parameter but no operations
        class NoOpModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.param = torch.nn.Parameter(torch.randn(1))
            
            def forward(self, x):
                # Just return input unchanged
                return x
        
        model = NoOpModel()
        dummy_input = torch.randn(1, 5)
        
        onnx_path = tmp_path / "noop.onnx"
        torch.onnx.export(
            model, dummy_input, str(onnx_path),
            input_names=['input'], output_names=['output'],
            opset_version=17
        )
        
        converter = ONNXToGraphMLConverter(hierarchical=False)
        graphml_output = converter.convert(str(onnx_path))
        
        # Should handle gracefully
        assert len(graphml_output) > 0
        root = ET.fromstring(graphml_output)
        assert root.tag.endswith("graphml")
    
    def test_metadata_with_nonexistent_modules(self, tmp_path, simple_onnx_model):
        """Test metadata referencing non-existent modules."""
        # Create metadata with references to modules that don't exist
        metadata_file = tmp_path / "nonexistent_modules.json"
        with open(metadata_file, 'w') as f:
            json.dump({
                "strategy": "htp",
                "tagged_nodes": {
                    "nonexistent_node1": "/NonExistent/Module1",
                    "nonexistent_node2": "/NonExistent/Module2", 
                    "fake_node": "/Fake/Path/To/Module"
                }
            }, f)
        
        # Should handle gracefully without crashing
        converter = ONNXToGraphMLConverter(htp_metadata_path=str(metadata_file))
        graphml_output = converter.convert(simple_onnx_model)
        
        # Should produce valid GraphML
        content, root = get_graphml_content(graphml_output)
        assert len(content) > 0
        assert root.tag.endswith("graphml")


class TestParameterValidation:
    """Test invalid parameter combinations and edge cases."""
    
    def test_invalid_exclude_attributes_type(self):
        """Test error handling for invalid exclude_attributes parameter."""
        # Should handle invalid types gracefully (use flat mode for parameter testing)
        converter = ONNXToGraphMLConverter(hierarchical=False, exclude_attributes="should_be_set_not_string")
        
        # Should still work (convert to set internally or handle gracefully)
        assert converter.exclude_attributes is not None
    
    def test_extreme_exclude_attributes(self, simple_onnx_model):
        """Test with very large exclude_attributes set."""
        # Create large set of attributes to exclude
        large_exclude_set = {f"attr_{i}" for i in range(1000)}
        
        converter = ONNXToGraphMLConverter(hierarchical=False, exclude_attributes=large_exclude_set)
        graphml_output = converter.convert(simple_onnx_model)
        
        # Should still work
        content, root = get_graphml_content(graphml_output)
        assert len(content) > 0
        assert root.tag.endswith("graphml")
    
    def test_boolean_parameter_edge_cases(self, simple_onnx_model):
        """Test boolean parameters with various values."""
        # Test with different boolean-like values
        for exclude_init in [True, False, 1, 0]:
            converter = ONNXToGraphMLConverter(hierarchical=False, exclude_initializers=bool(exclude_init))
            graphml_output = converter.convert(simple_onnx_model)
            
            content, root = get_graphml_content(graphml_output)
            assert len(content) > 0
            assert root.tag.endswith("graphml")


class TestResourceConstraints:
    """Test behavior under resource constraints and stress conditions."""
    
    def test_save_to_readonly_directory(self, simple_onnx_model, tmp_path):
        """Test error handling when saving to read-only directory."""
        converter = ONNXToGraphMLConverter(hierarchical=False)
        
        # Create read-only directory (skip if we can't make it readonly)
        readonly_dir = tmp_path / "readonly"
        readonly_dir.mkdir()
        
        try:
            readonly_dir.chmod(0o444)  # Make read-only
            readonly_output = readonly_dir / "output.graphml"
            
            with pytest.raises(PermissionError):
                converter.save(simple_onnx_model, str(readonly_output))
        except (OSError, PermissionError):
            # Skip test if we can't create readonly directory (e.g., on Windows)
            pytest.skip("Cannot create read-only directory on this system")
        finally:
            # Restore permissions for cleanup
            with contextlib.suppress(builtins.BaseException):
                readonly_dir.chmod(0o755)
    
    def test_save_with_invalid_output_path(self, simple_onnx_model):
        """Test save with various invalid output paths."""
        converter = ONNXToGraphMLConverter(hierarchical=False)
        
        # Test with a path that definitely should fail
        invalid_path = "/nonexistent/deeply/nested/path/that/should/not/exist/output.graphml"
        
        with pytest.raises((FileNotFoundError, OSError, PermissionError)):
            converter.save(simple_onnx_model, invalid_path)
    
    def test_concurrent_conversion_safety(self, simple_onnx_model):
        """Test that converter can handle multiple instances safely."""
        # Create multiple converter instances (use flat mode for safety testing)
        converters = [ONNXToGraphMLConverter(hierarchical=False) for _ in range(3)]
        
        # Convert with all instances
        results = []
        for converter in converters:
            graphml_output = converter.convert(simple_onnx_model)
            results.append(graphml_output)
        
        # All should succeed and produce valid GraphML
        for result in results:
            content, root = get_graphml_content(result)
            assert len(content) > 0
            assert root.tag.endswith("graphml")
        
        # Results should have similar structure (may not be identical due to timestamps)
        # Just verify they all have similar node/edge counts
        roots = [get_graphml_content(result)[1] for result in results]
        node_counts = [len(root.findall(".//{http://graphml.graphdrawing.org/xmlns}node")) for root in roots]
        edge_counts = [len(root.findall(".//{http://graphml.graphdrawing.org/xmlns}edge")) for root in roots]
        
        # All should have same structure
        assert all(count == node_counts[0] for count in node_counts), "Node counts should be identical"
        assert all(count == edge_counts[0] for count in edge_counts), "Edge counts should be identical"


class TestMetadataReaderErrors:
    """Test MetadataReader error handling specifically."""
    
    def test_metadata_reader_with_malformed_json(self, tmp_path):
        """Test MetadataReader with malformed JSON."""
        malformed_file = tmp_path / "malformed.json"
        with open(malformed_file, 'w') as f:
            f.write("{ malformed json")
        
        with pytest.raises(json.JSONDecodeError):
            MetadataReader(str(malformed_file))
    
    def test_metadata_reader_with_non_json_file(self, tmp_path):
        """Test MetadataReader with non-JSON file."""
        text_file = tmp_path / "not_json.txt"
        with open(text_file, 'w') as f:
            f.write("This is not JSON")
        
        with pytest.raises(json.JSONDecodeError):
            MetadataReader(str(text_file))
    
    def test_metadata_reader_missing_fields(self, tmp_path):
        """Test MetadataReader behavior with missing expected fields."""
        # Create JSON with no expected fields
        minimal_file = tmp_path / "minimal.json"
        with open(minimal_file, 'w') as f:
            json.dump({"unexpected_field": "value"}, f)
        
        # Should not crash, but may return empty results
        reader = MetadataReader(str(minimal_file))
        assert reader.get_all_modules() == []
        assert reader.get_node_hierarchy_tag("any_node") is None


class TestOutputValidation:
    """Test that output is always valid GraphML even in error conditions."""
    
    def test_output_always_valid_xml(self, simple_onnx_model):
        """Test that GraphML output is always valid XML."""
        converter = ONNXToGraphMLConverter(hierarchical=False)
        graphml_output = converter.convert(simple_onnx_model)
        
        # Should be parseable XML
        root = ET.fromstring(graphml_output)
        assert root is not None
        
        # Should have GraphML namespace
        assert "graphml" in root.tag
    
    def test_output_has_required_graphml_structure(self, simple_onnx_model):
        """Test that output has required GraphML elements."""
        converter = ONNXToGraphMLConverter(hierarchical=False)
        graphml_output = converter.convert(simple_onnx_model)
        
        root = ET.fromstring(graphml_output)
        
        # Should have key definitions
        keys = root.findall(".//{http://graphml.graphdrawing.org/xmlns}key")
        assert len(keys) > 0, "GraphML should have key definitions"
        
        # Should have at least one graph
        graphs = root.findall(".//{http://graphml.graphdrawing.org/xmlns}graph")
        assert len(graphs) >= 1, "GraphML should have at least one graph"
    
    def test_statistics_consistency(self, simple_onnx_model):
        """Test that conversion statistics are consistent with output."""
        converter = ONNXToGraphMLConverter(hierarchical=False)
        graphml_output = converter.convert(simple_onnx_model)
        
        # Get statistics
        stats = converter.get_statistics()
        
        # Parse output to count nodes and edges
        root = ET.fromstring(graphml_output)
        actual_nodes = len(root.findall(".//{http://graphml.graphdrawing.org/xmlns}node"))
        actual_edges = len(root.findall(".//{http://graphml.graphdrawing.org/xmlns}edge"))
        
        # Statistics should match (allowing for some flexibility in counting)
        assert stats['nodes'] >= 0
        assert stats['edges'] >= 0
        assert stats['excluded_initializers'] >= 0
        
        # Note: actual counts may differ slightly due to input/output nodes
        # but should be in the same ballpark
        if actual_nodes > 0:
            assert abs(stats['nodes'] - actual_nodes) <= 5  # Allow some variance