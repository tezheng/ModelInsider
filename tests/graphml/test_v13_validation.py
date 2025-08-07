"""
Comprehensive test suite for GraphML v1.3 validation.

This module tests REAL validation, not documentation theater.
Tests all three layers of validation with code-generated results.

CARDINAL RULE #2: All testing via pytest with code-generated results.
NO hardcoded expectations, NO LLM-generated test data.

Linear Task: TEZ-137 (Pillar 2: Multi-Layer Validation System)
"""

import json
import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path

import onnx
import pytest
from onnx import helper

from modelexport.graphml.constants import GRAPHML_FORMAT_VERSION, GRAPHML_V13_KEYS
from modelexport.graphml.constants import GRAPHML_FORMAT_VERSION as GRAPHML_VERSION
from modelexport.graphml.graphml_to_onnx_converter import GraphMLToONNXConverter
from modelexport.graphml.onnx_to_graphml_converter import ONNXToGraphMLConverter
from modelexport.graphml.validators import (
    GraphMLV13Validator,
    SchemaValidator,
    SemanticValidator,
    RoundTripValidator,
    ValidationStatus,
)


@pytest.fixture
def simple_onnx_model():
    """Generate a simple ONNX model for testing."""
    # Create a simple Add operation
    input1 = helper.make_tensor_value_info("input1", onnx.TensorProto.FLOAT, [1, 3])
    input2 = helper.make_tensor_value_info("input2", onnx.TensorProto.FLOAT, [1, 3])
    output = helper.make_tensor_value_info("output", onnx.TensorProto.FLOAT, [1, 3])
    
    add_node = helper.make_node(
        "Add",
        inputs=["input1", "input2"],
        outputs=["output"],
        name="add_node"
    )
    
    graph = helper.make_graph(
        [add_node],
        "simple_graph",
        [input1, input2],
        [output],
    )
    
    model = helper.make_model(graph, producer_name="test")
    model.model_version = 1
    model.doc_string = "Test model"
    
    # Add opset
    opset = model.opset_import.add()
    opset.version = 17
    
    return model


@pytest.fixture
def valid_v13_graphml():
    """Generate a valid GraphML v1.3 file."""
    graphml = ET.Element("graphml", attrib={
        "xmlns": "http://graphml.graphdrawing.org/xmlns",
        "xmlns:xsi": "http://www.w3.org/2001/XMLSchema-instance",
        "xsi:schemaLocation": (
            "http://graphml.graphdrawing.org/xmlns "
            "http://graphml.graphdrawing.org/xmlns/1.0/graphml.xsd"
        )
    })
    
    # Add all required keys (31 total for v1.3)
    keys = [
        # Graph keys
        ("d0", "graph", "class_name", "string"),
        ("d1", "graph", "module_type", "string"),
        ("d2", "graph", "execution_order", "int"),
        ("d3", "graph", "traced_tag", "string"),
        # Node keys
        ("n0", "node", "op_type", "string"),
        ("n1", "node", "hierarchy_tag", "string"),
        ("n2", "node", "onnx_attributes", "string"),
        ("n3", "node", "name", "string"),
        ("n4", "node", "input_names", "string"),
        ("n5", "node", "output_names", "string"),
        ("n6", "node", "domain", "string"),
        # Edge keys (v1.3 naming)
        ("e0", "edge", "tensor_name", "string"),
        ("e1", "edge", "tensor_type", "string"),
        ("e2", "edge", "tensor_shape", "string"),
        ("e3", "edge", "tensor_data_ref", "string"),
        # Metadata keys (v1.3 naming)
        ("meta0", "graph", "source_onnx_file", "string"),
        ("meta1", "graph", "source_htp_file", "string"),
        ("meta2", "graph", "format_version", "string"),
        ("meta3", "graph", "export_timestamp", "string"),
        ("meta4", "graph", "opset_imports", "string"),
        ("meta5", "graph", "producer_name", "string"),
        ("meta6", "graph", "producer_version", "string"),
        ("meta7", "graph", "model_version", "string"),
        ("meta8", "graph", "doc_string", "string"),
        # Parameter keys (v1.3 naming)
        ("param0", "graph", "parameter_strategy", "string"),
        ("param1", "graph", "parameter_file", "string"),
        ("param2", "graph", "parameter_checksum", "string"),
        # I/O keys (v1.3 naming)
        ("io0", "graph", "graph_inputs", "string"),
        ("io1", "graph", "graph_outputs", "string"),
        ("io2", "graph", "value_info", "string"),
        ("io3", "graph", "initializers_ref", "string"),
    ]
    
    for key_id, for_type, attr_name, attr_type in keys:
        key_elem = ET.SubElement(graphml, "key", attrib={
            "id": key_id,
            "for": for_type,
            "attr.name": attr_name,
            "attr.type": attr_type
        })
    
    # Add graph
    graph = ET.SubElement(graphml, "graph", attrib={
        "id": "TestModel",
        "edgedefault": "directed"
    })
    
    # Add required metadata
    ET.SubElement(graph, "data", attrib={"key": "meta2"}).text = GRAPHML_FORMAT_VERSION
    ET.SubElement(graph, "data", attrib={"key": "meta3"}).text = "2025-08-04T12:00:00"
    ET.SubElement(graph, "data", attrib={"key": "param0"}).text = "sidecar"
    ET.SubElement(graph, "data", attrib={"key": "param1"}).text = "model.onnxdata"
    ET.SubElement(graph, "data", attrib={"key": "meta5"}).text = "test"
    ET.SubElement(graph, "data", attrib={"key": "io0"}).text = json.dumps([])
    ET.SubElement(graph, "data", attrib={"key": "io1"}).text = json.dumps([])
    
    # Add a node
    node = ET.SubElement(graph, "node", attrib={"id": "add_node"})
    ET.SubElement(node, "data", attrib={"key": "n0"}).text = "Add"
    ET.SubElement(node, "data", attrib={"key": "n1"}).text = "/Model"
    ET.SubElement(node, "data", attrib={"key": "n2"}).text = "{}"
    ET.SubElement(node, "data", attrib={"key": "n3"}).text = "add_node"
    
    return graphml


class TestSchemaValidation:
    """Test Layer 1: XSD Schema Compliance."""
    
    def test_valid_v13_schema(self, valid_v13_graphml, tmp_path):
        """Test that valid v1.3 GraphML passes schema validation."""
        # Save GraphML to file
        graphml_file = tmp_path / "valid.graphml"
        ET.ElementTree(valid_v13_graphml).write(
            graphml_file, encoding="utf-8", xml_declaration=True
        )
        
        # Validate
        validator = SchemaValidator()
        result = validator.validate(str(graphml_file))
        
        assert result.status == ValidationStatus.PASS
        assert f"Valid GraphML v{GRAPHML_VERSION} structure" in result.message
    
    def test_missing_required_keys(self, tmp_path):
        """Test that missing required keys fails validation."""
        # Create GraphML with missing keys
        graphml = ET.Element("graphml", attrib={
            "xmlns": "http://graphml.graphdrawing.org/xmlns"
        })
        
        # Add only some keys (not all 35)
        for i in range(10):
            ET.SubElement(graphml, "key", attrib={
                "id": f"key{i}",
                "for": "node",
                "attr.name": f"attr{i}",
                "attr.type": "string"
            })
        
        # Save and validate
        graphml_file = tmp_path / "invalid.graphml"
        ET.ElementTree(graphml).write(
            graphml_file, encoding="utf-8", xml_declaration=True
        )
        
        validator = SchemaValidator()
        result = validator.validate(str(graphml_file))
        
        assert result.status == ValidationStatus.FAIL
        assert "Schema validation failed" in result.message
    
    def test_old_keys_rejected(self, tmp_path):
        """Test that old v1.1/v1.2 keys are rejected."""
        graphml = ET.Element("graphml", attrib={
            "xmlns": "http://graphml.graphdrawing.org/xmlns"
        })
        
        # Add old conflicting keys
        old_keys = [
            ("m5", "graph", "producer_name", "string"),  # Should be meta5
            ("p0", "graph", "parameter_strategy", "string"),  # Should be param0
            ("g0", "graph", "graph_inputs", "string"),  # Should be io0
            ("t0", "edge", "tensor_type", "string"),  # Should be e1
        ]
        
        for key_id, for_type, attr_name, attr_type in old_keys:
            ET.SubElement(graphml, "key", attrib={
                "id": key_id,
                "for": for_type,
                "attr.name": attr_name,
                "attr.type": attr_type
            })
        
        # Save and validate
        graphml_file = tmp_path / "old_keys.graphml"
        ET.ElementTree(graphml).write(
            graphml_file, encoding="utf-8", xml_declaration=True
        )
        
        validator = SchemaValidator()
        result = validator.validate(str(graphml_file))
        
        assert result.status == ValidationStatus.FAIL


class TestSemanticValidation:
    """Test Layer 2: Semantic Consistency."""
    
    @pytest.mark.version
    @pytest.mark.graphml
    @pytest.mark.unit
    def test_format_version_required(self, valid_v13_graphml, tmp_path):
        """Test that format version must be exactly 1.3."""
        # Remove format version
        graph = valid_v13_graphml.find(".//graph")
        for data in graph.findall("data[@key='meta2']"):
            graph.remove(data)
        
        # Save and validate
        graphml_file = tmp_path / "no_version.graphml"
        ET.ElementTree(valid_v13_graphml).write(
            graphml_file, encoding="utf-8", xml_declaration=True
        )
        
        validator = SemanticValidator()
        result = validator.validate(str(graphml_file))
        
        assert result.status == ValidationStatus.FAIL
        assert "Missing format version" in result.message
    
    @pytest.mark.version
    @pytest.mark.graphml  
    @pytest.mark.unit
    def test_wrong_format_version(self, valid_v13_graphml, tmp_path):
        """Test that wrong format version fails."""
        # Change format version to 1.2
        graph = valid_v13_graphml.find(".//graph")
        version_elem = graph.find("data[@key='meta2']")
        version_elem.text = "1.2"
        
        # Save and validate
        graphml_file = tmp_path / "wrong_version.graphml"
        ET.ElementTree(valid_v13_graphml).write(
            graphml_file, encoding="utf-8", xml_declaration=True
        )
        
        validator = SemanticValidator()
        result = validator.validate(str(graphml_file))
        
        assert result.status == ValidationStatus.FAIL
        assert "Invalid format version" in result.message
    
    def test_edge_connectivity(self, valid_v13_graphml, tmp_path):
        """Test that edges must connect to valid nodes."""
        graph = valid_v13_graphml.find(".//graph")
        
        # Add edge with invalid source/target
        ET.SubElement(graph, "edge", attrib={
            "source": "nonexistent_node",
            "target": "add_node"
        })
        
        # Save and validate
        graphml_file = tmp_path / "bad_edge.graphml"
        ET.ElementTree(valid_v13_graphml).write(
            graphml_file, encoding="utf-8", xml_declaration=True
        )
        
        validator = SemanticValidator()
        result = validator.validate(str(graphml_file))
        
        assert result.status == ValidationStatus.FAIL
        assert "not found in nodes" in result.message
    
    def test_json_field_validation(self, valid_v13_graphml, tmp_path):
        """Test that JSON fields must be valid JSON."""
        graph = valid_v13_graphml.find(".//graph")
        
        # Add invalid JSON to io0
        io0_elem = graph.find("data[@key='io0']")
        io0_elem.text = "not valid json {"
        
        # Save and validate
        graphml_file = tmp_path / "bad_json.graphml"
        ET.ElementTree(valid_v13_graphml).write(
            graphml_file, encoding="utf-8", xml_declaration=True
        )
        
        validator = SemanticValidator()
        result = validator.validate(str(graphml_file))
        
        assert result.status == ValidationStatus.FAIL
        assert "Invalid JSON" in result.message
    
    def test_parameter_strategy_consistency(self, valid_v13_graphml, tmp_path):
        """Test that sidecar strategy requires parameter file."""
        graph = valid_v13_graphml.find(".//graph")
        
        # Remove parameter file but keep sidecar strategy
        for data in graph.findall("data[@key='param1']"):
            graph.remove(data)
        
        # Save and validate
        graphml_file = tmp_path / "no_param_file.graphml"
        ET.ElementTree(valid_v13_graphml).write(
            graphml_file, encoding="utf-8", xml_declaration=True
        )
        
        validator = SemanticValidator()
        result = validator.validate(str(graphml_file))
        
        assert result.status == ValidationStatus.FAIL
        assert "Sidecar strategy requires param1" in result.message


class TestRoundTripValidation:
    """Test Layer 3: Round-Trip Accuracy."""
    
    def test_round_trip_accuracy(self, simple_onnx_model, tmp_path):
        """Test round-trip conversion accuracy."""
        # Save ONNX model
        onnx_file = tmp_path / "model.onnx"
        onnx.save(simple_onnx_model, str(onnx_file))
        
        # Create simple HTP metadata
        htp_metadata = {
            "model": {
                "class_name": "TestModel",
                "name_or_path": "test"
            },
            "modules": {},
            "nodes": {}
        }
        htp_file = tmp_path / "metadata.json"
        htp_file.write_text(json.dumps(htp_metadata))
        
        # Convert to GraphML (without validation to test validator separately)
        converter = ONNXToGraphMLConverter(
            htp_metadata_path=str(htp_file),
            validate_output=False
        )
        result = converter.convert(str(onnx_file), str(tmp_path / "model"))
        graphml_file = result["graphml"]
        
        # Validate round-trip
        validator = RoundTripValidator()
        result = validator.validate(str(onnx_file), graphml_file)
        
        assert result.status == ValidationStatus.PASS
        assert result.metrics["node_preservation"] >= 0.85
    
    def test_parameter_preservation(self, simple_onnx_model, tmp_path):
        """Test that parameters are preserved 100%."""
        # Add an initializer (parameter)
        import numpy as np
        weight = onnx.numpy_helper.from_array(
            np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32),
            name="weight"
        )
        simple_onnx_model.graph.initializer.append(weight)
        
        # Save model
        onnx_file = tmp_path / "model_with_params.onnx"
        onnx.save(simple_onnx_model, str(onnx_file))
        
        # Create HTP metadata
        htp_metadata = {
            "model": {
                "class_name": "TestModel",
                "name_or_path": "test"
            },
            "modules": {},
            "nodes": {}
        }
        htp_file = tmp_path / "metadata.json"
        htp_file.write_text(json.dumps(htp_metadata))
        
        # Convert to GraphML
        converter = ONNXToGraphMLConverter(
            htp_metadata_path=str(htp_file),
            validate_output=False
        )
        result = converter.convert(str(onnx_file), str(tmp_path / "model"))
        graphml_file = result["graphml"]
        
        # Validate round-trip
        validator = RoundTripValidator()
        result = validator.validate(str(onnx_file), graphml_file)
        
        assert result.metrics["param_preservation"] == 1.0


class TestCompleteValidation:
    """Test complete three-layer validation."""
    
    def test_all_layers_pass(self, simple_onnx_model, tmp_path):
        """Test that a properly converted model passes all layers."""
        # Save ONNX model
        onnx_file = tmp_path / "model.onnx"
        onnx.save(simple_onnx_model, str(onnx_file))
        
        # Create HTP metadata
        htp_metadata = {
            "model": {
                "class_name": "TestModel",
                "name_or_path": "test"
            },
            "modules": {},
            "nodes": {}
        }
        htp_file = tmp_path / "metadata.json"
        htp_file.write_text(json.dumps(htp_metadata))
        
        # Convert to GraphML with validation
        converter = ONNXToGraphMLConverter(
            htp_metadata_path=str(htp_file),
            validate_output=True  # Enable validation
        )
        
        # Should not raise if validation passes
        result = converter.convert(str(onnx_file), str(tmp_path / "model"))
        assert result["format_version"] == GRAPHML_FORMAT_VERSION
    
    def test_strict_validation(self, valid_v13_graphml, tmp_path):
        """Test strict validation mode."""
        # Save valid GraphML
        graphml_file = tmp_path / "valid.graphml"
        ET.ElementTree(valid_v13_graphml).write(
            graphml_file, encoding="utf-8", xml_declaration=True
        )
        
        # Test strict validation
        validator = GraphMLV13Validator()
        assert validator.validate_strict(str(graphml_file)) == True
        
        # Break something
        graph = valid_v13_graphml.find(".//graph")
        version_elem = graph.find("data[@key='meta2']")
        version_elem.text = "1.2"  # Wrong version
        
        # Save and test again
        invalid_file = tmp_path / "invalid.graphml"
        ET.ElementTree(valid_v13_graphml).write(
            invalid_file, encoding="utf-8", xml_declaration=True
        )
        
        assert validator.validate_strict(str(invalid_file)) == False
    
    def test_converter_rejects_old_format(self, tmp_path):
        """Test that GraphML to ONNX converter rejects old formats."""
        # Create v1.2 GraphML
        graphml = ET.Element("graphml", attrib={
            "xmlns": "http://graphml.graphdrawing.org/xmlns"
        })
        
        graph = ET.SubElement(graphml, "graph", attrib={
            "id": "TestModel",
            "edgedefault": "directed"
        })
        
        # Use old key naming (m2 instead of meta2)
        ET.SubElement(graph, "data", attrib={"key": "m2"}).text = "1.2"
        
        # Save
        graphml_file = tmp_path / "v12.graphml"
        ET.ElementTree(graphml).write(
            graphml_file, encoding="utf-8", xml_declaration=True
        )
        
        # Try to convert - should fail
        converter = GraphMLToONNXConverter()
        
        with pytest.raises(ValueError, match="not a valid v1.3 key"):
            converter.convert(str(graphml_file), str(tmp_path / "output.onnx"))