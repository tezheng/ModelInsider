"""
Test suite for ONNX to GraphML converter.

Tests the base converter functionality including:
- Basic conversion from ONNX to GraphML
- Node and edge extraction
- Metadata handling
- Error cases
"""

import xml.etree.ElementTree as ET

import pytest

from modelexport.graphml import ONNXToGraphMLConverter


class TestONNXToGraphMLConverter:
    """Test cases for the base ONNX to GraphML converter."""
    
    def test_converter_initialization(self):
        """Test converter can be initialized with default settings."""
        converter = ONNXToGraphMLConverter(hierarchical=False)
        assert converter.exclude_initializers is True
        assert converter.exclude_attributes == set()
    
    def test_converter_with_options(self):
        """Test converter initialization with custom options."""
        converter = ONNXToGraphMLConverter(
            hierarchical=False,
            exclude_initializers=False,
            exclude_attributes={"custom_attr"}
        )
        assert converter.exclude_initializers is False
        assert "custom_attr" in converter.exclude_attributes
    
    def test_convert_simple_model(self, simple_onnx_model):
        """Test conversion of a simple ONNX model."""
        converter = ONNXToGraphMLConverter(hierarchical=False)
        graphml = converter.convert(simple_onnx_model)
        
        # Parse result
        root = ET.fromstring(graphml)
        
        # Verify GraphML structure
        assert root.tag.endswith("graphml")
        graphs = root.findall(".//{http://graphml.graphdrawing.org/xmlns}graph")
        assert len(graphs) == 1
    
    def test_missing_file_error(self):
        """Test error handling for missing ONNX file."""
        converter = ONNXToGraphMLConverter(hierarchical=False)
        
        with pytest.raises(FileNotFoundError):
            converter.convert("nonexistent_model.onnx")
    
    def test_save_to_file(self, simple_onnx_model, tmp_path):
        """Test saving GraphML to file."""
        converter = ONNXToGraphMLConverter(hierarchical=False)
        output_path = tmp_path / "output.graphml"
        
        # Convert and save
        graphml_str = converter.convert(simple_onnx_model)
        with open(output_path, 'w') as f:
            f.write(graphml_str)
        
        # Verify file exists and has content
        assert output_path.exists()
        assert output_path.stat().st_size > 0
        
        # Verify it's valid XML
        root = ET.parse(output_path).getroot()
        assert root.tag.endswith("graphml")
    
    def test_statistics_tracking(self, simple_onnx_model):
        """Test that conversion statistics are tracked correctly."""
        converter = ONNXToGraphMLConverter(hierarchical=False)
        
        # Convert model
        graphml_str = converter.convert(simple_onnx_model)
        
        # Check that statistics are tracked in parser
        assert hasattr(converter.parser, 'last_node_count')
        assert hasattr(converter.parser, 'last_edge_count')
        assert hasattr(converter.parser, 'last_initializer_count')
        
        # Verify statistics are reasonable
        assert converter.parser.last_node_count > 0
        assert converter.parser.last_edge_count > 0
        assert converter.parser.last_initializer_count >= 0