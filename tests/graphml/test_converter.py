"""
Test suite for ONNX to GraphML converter.

Tests the base converter functionality including:
- Basic conversion from ONNX to GraphML
- Node and edge extraction
- Metadata handling
- Error cases
"""

import json
import xml.etree.ElementTree as ET
from pathlib import Path

import onnx
import pytest
from onnx import TensorProto, helper

from modelexport.graphml.onnx_to_graphml_converter import ONNXToGraphMLConverter


@pytest.fixture
def simple_onnx_model(tmp_path):
    """Create a simple ONNX model for testing."""
    # Create a simple model: Input -> Add -> Output
    input_tensor = helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 3])
    output_tensor = helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, 3])
    
    # Create bias initializer
    bias_init = helper.make_tensor(
        name="bias",
        data_type=TensorProto.FLOAT,
        dims=[3],
        vals=[0.1, 0.2, 0.3]
    )
    
    # Create Add node
    add_node = helper.make_node(
        "Add",
        inputs=["input", "bias"],
        outputs=["output"],
        name="Add_1"
    )
    
    # Create graph
    graph = helper.make_graph(
        nodes=[add_node],
        name="SimpleModel",
        inputs=[input_tensor],
        outputs=[output_tensor],
        initializer=[bias_init]
    )
    
    # Create model
    model = helper.make_model(graph)
    model.opset_import[0].version = 17
    
    # Save model
    model_path = tmp_path / "simple_model.onnx"
    onnx.save(model, str(model_path))
    
    return str(model_path)


@pytest.fixture  
def simple_htp_metadata(tmp_path):
    """Create simple HTP metadata for testing."""
    metadata = {
        "modules": {
            "scope": "/SimpleModel",
            "class_name": "SimpleModel", 
            "execution_order": 0,
            "traced_tag": "/SimpleModel"
        },
        "tagged_nodes": {
            "Add_1": {
                "hierarchy_tag": "/SimpleModel",
                "op_type": "Add"
            }
        }
    }
    
    metadata_path = tmp_path / "simple_metadata.json"
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)
    
    return str(metadata_path)


@pytest.mark.graphml
@pytest.mark.unit
class TestONNXToGraphMLConverter:
    """Test cases for the base ONNX to GraphML converter."""
    
    @pytest.mark.smoke
    def test_converter_initialization(self, simple_htp_metadata):
        """Test converter can be initialized with default settings."""
        converter = ONNXToGraphMLConverter(htp_metadata_path=simple_htp_metadata)
        assert converter.exclude_initializers is True
        assert converter.exclude_attributes == set()
        assert converter.hierarchical is True
    
    @pytest.mark.smoke
    def test_converter_with_options(self, simple_htp_metadata):
        """Test converter initialization with custom options."""
        converter = ONNXToGraphMLConverter(
            htp_metadata_path=simple_htp_metadata,
            exclude_initializers=False,
            exclude_attributes={"custom_attr"}
        )
        assert converter.exclude_initializers is False
        assert "custom_attr" in converter.exclude_attributes
    
    @pytest.mark.sanity
    @pytest.mark.integration
    def test_convert_simple_model(self, simple_onnx_model, simple_htp_metadata):
        """Test conversion of a simple ONNX model."""
        converter = ONNXToGraphMLConverter(
            htp_metadata_path=simple_htp_metadata,
            parameter_strategy="embedded"
        )
        result = converter.convert(simple_onnx_model)
        
        # Should get a result dict with GraphML content
        assert isinstance(result, dict)
        assert "graphml" in result
        
        # Read and parse the GraphML file
        with open(result["graphml"]) as f:
            graphml_content = f.read()
        
        root = ET.fromstring(graphml_content)
        
        # Verify GraphML structure
        assert root.tag.endswith("graphml")
        graphs = root.findall(".//{http://graphml.graphdrawing.org/xmlns}graph")
        assert len(graphs) >= 1
    
    @pytest.mark.error_recovery
    def test_missing_file_error(self, simple_htp_metadata):
        """Test error handling for missing ONNX file."""
        converter = ONNXToGraphMLConverter(htp_metadata_path=simple_htp_metadata)
        
        with pytest.raises(FileNotFoundError):
            converter.convert("nonexistent_model.onnx")
    
    @pytest.mark.integration
    def test_save_to_file(self, simple_onnx_model, simple_htp_metadata, tmp_path):
        """Test saving GraphML to file."""
        converter = ONNXToGraphMLConverter(
            htp_metadata_path=simple_htp_metadata,
            parameter_strategy="embedded"
        )
        
        output_path = str(tmp_path / "test_output")
        result = converter.convert(simple_onnx_model, output_path)
        
        # Verify GraphML file was created
        assert "graphml" in result
        graphml_path = Path(result["graphml"])
        assert graphml_path.exists()
        assert graphml_path.suffix == ".graphml"
        
        # Verify content is valid GraphML
        with open(graphml_path) as f:
            content = f.read()
        root = ET.fromstring(content)
        assert root.tag.endswith("graphml")
    
    @pytest.mark.integration
    def test_statistics_tracking(self, simple_onnx_model, simple_htp_metadata):
        """Test that conversion statistics are tracked correctly."""
        converter = ONNXToGraphMLConverter(
            htp_metadata_path=simple_htp_metadata,
            parameter_strategy="embedded"
        )
        
        result = converter.convert(simple_onnx_model)
        
        # Verify we get result with format info
        assert isinstance(result, dict)
        assert "format_version" in result
        assert result["format_version"] == "1.1"
        
        # Verify GraphML file exists and contains nodes
        with open(result["graphml"]) as f:
            content = f.read()
        root = ET.fromstring(content)
        
        # Should have nodes in the GraphML
        nodes = root.findall(".//{http://graphml.graphdrawing.org/xmlns}node")
        assert len(nodes) > 0