"""
Tests for Hierarchical GraphML Converter

Tests the HierarchicalGraphMLConverter class functionality for creating
GraphML with compound nodes from ONNX models and HTP metadata.
"""

import json
import xml.etree.ElementTree as ET

import onnx
import pytest
from onnx import TensorProto, helper

from modelexport.graphml.hierarchical_converter import HierarchicalGraphMLConverter


@pytest.fixture
def sample_onnx_model(tmp_path):
    """Create a sample ONNX model for testing."""
    # Create a simple model: Input -> Add -> MatMul -> Output
    
    # Define tensors
    input_tensor = helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 3])
    output_tensor = helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, 4])
    
    # Create initializers
    bias_init = helper.make_tensor(
        name="bias",
        data_type=TensorProto.FLOAT,
        dims=[3],
        vals=[0.1, 0.2, 0.3]
    )
    
    weight_init = helper.make_tensor(
        name="weight",
        data_type=TensorProto.FLOAT,
        dims=[3, 4],
        vals=[0.1] * 12
    )
    
    # Create nodes
    add_node = helper.make_node(
        "Add",
        inputs=["input", "bias"],
        outputs=["add_out"],
        name="Add_1"
    )
    
    matmul_node = helper.make_node(
        "MatMul",
        inputs=["add_out", "weight"],
        outputs=["output"],
        name="MatMul_2"
    )
    
    # Create graph
    graph = helper.make_graph(
        nodes=[add_node, matmul_node],
        name="test_graph",
        inputs=[input_tensor],
        outputs=[output_tensor],
        initializer=[bias_init, weight_init]
    )
    
    # Create model
    model = helper.make_model(graph)
    model.opset_import[0].version = 17
    
    # Save model
    model_path = tmp_path / "test_model.onnx"
    onnx.save(model, str(model_path))
    
    return model_path


@pytest.fixture
def hierarchical_htp_metadata(tmp_path):
    """Create HTP metadata with hierarchy for the test model."""
    metadata = {
        "modules": {
            "scope": "/TestModel",
            "class_name": "TestModel",
            "execution_order": 0,
            "traced_tag": "/TestModel",
            "children": {
                "layer1": {
                    "scope": "/TestModel/layer1",
                    "class_name": "AddLayer",
                    "execution_order": 1,
                    "traced_tag": "/TestModel/layer1"
                },
                "layer2": {
                    "scope": "/TestModel/layer2",
                    "class_name": "MatMulLayer",
                    "execution_order": 2,
                    "traced_tag": "/TestModel/layer2"
                }
            }
        },
        "tagged_nodes": {
            "Add_1": {
                "hierarchy_tag": "/TestModel/layer1",
                "op_type": "Add"
            },
            "MatMul_2": {
                "hierarchy_tag": "/TestModel/layer2",
                "op_type": "MatMul"
            }
        }
    }
    
    metadata_path = tmp_path / "test_metadata.json"
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)
    
    return metadata_path


@pytest.fixture
def deep_hierarchy_metadata(tmp_path):
    """Create HTP metadata with deep nested hierarchy."""
    metadata = {
        "modules": {
            "scope": "/DeepModel",
            "class_name": "DeepModel",
            "execution_order": 0,
            "traced_tag": "/DeepModel",
            "children": {
                "encoder": {
                    "scope": "/DeepModel/encoder",
                    "class_name": "Encoder",
                    "execution_order": 1,
                    "traced_tag": "/DeepModel/encoder",
                    "children": {
                        "layer": {
                            "scope": "/DeepModel/encoder/layer",
                            "class_name": "LayerList",
                            "execution_order": 2,
                            "traced_tag": "/DeepModel/encoder/layer",
                            "children": {
                                "0": {
                                    "scope": "/DeepModel/encoder/layer/0",
                                    "class_name": "Layer",
                                    "execution_order": 3,
                                    "traced_tag": "/DeepModel/encoder/layer/0",
                                    "children": {
                                        "attention": {
                                            "scope": "/DeepModel/encoder/layer/0/attention",
                                            "class_name": "Attention",
                                            "execution_order": 4,
                                            "traced_tag": "/DeepModel/encoder/layer/0/attention"
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        },
        "tagged_nodes": {
            "Add_1": {
                "hierarchy_tag": "/DeepModel/encoder/layer/0",
                "op_type": "Add"
            },
            "MatMul_2": {
                "hierarchy_tag": "/DeepModel/encoder/layer/0/attention",
                "op_type": "MatMul"
            }
        }
    }
    
    metadata_path = tmp_path / "deep_metadata.json"
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)
    
    return metadata_path


class TestHierarchicalGraphMLConverter:
    """Test HierarchicalGraphMLConverter functionality."""
    
    def test_init_with_valid_metadata(self, hierarchical_htp_metadata):
        """Test initialization with valid metadata file."""
        converter = HierarchicalGraphMLConverter(
            str(hierarchical_htp_metadata),
            exclude_initializers=True
        )
        assert converter.htp_metadata_path == str(hierarchical_htp_metadata)
        assert converter.metadata_reader is not None
        assert converter.exclude_initializers is True
    
    def test_init_with_missing_metadata(self, tmp_path):
        """Test initialization with non-existent metadata file."""
        missing_path = tmp_path / "missing.json"
        with pytest.raises(FileNotFoundError):
            HierarchicalGraphMLConverter(str(missing_path))
    
    def test_convert_basic_hierarchy(self, sample_onnx_model, hierarchical_htp_metadata):
        """Test basic hierarchical conversion."""
        converter = HierarchicalGraphMLConverter(
            str(hierarchical_htp_metadata),
            exclude_initializers=True
        )
        
        graphml_str = converter.convert(str(sample_onnx_model))
        assert isinstance(graphml_str, str)
        assert len(graphml_str) > 0
        
        # Parse generated GraphML
        root = ET.fromstring(graphml_str)
        assert root.tag == "{http://graphml.graphdrawing.org/xmlns}graphml"
        
        # Find main graph
        graphs = root.findall(".//{http://graphml.graphdrawing.org/xmlns}graph")
        assert len(graphs) >= 1
        
        # Check for compound nodes (nodes with nested graphs)
        nodes = root.findall(".//{http://graphml.graphdrawing.org/xmlns}node")
        compound_nodes = []
        for node in nodes:
            # Check if this node contains a nested graph element (compound node)
            nested_graph = node.find("./{http://graphml.graphdrawing.org/xmlns}graph")
            if nested_graph is not None:
                compound_nodes.append(node)
        assert len(compound_nodes) > 0
        
        # Verify hierarchical structure exists
        assert len(compound_nodes) >= 2  # Should have at least 2 modules in hierarchy
    
    def test_hierarchy_tag_assignment(self, sample_onnx_model, hierarchical_htp_metadata):
        """Test that hierarchy tags are properly assigned to nodes in GraphML output."""
        converter = HierarchicalGraphMLConverter(
            str(hierarchical_htp_metadata),
            exclude_initializers=True
        )
        
        # Convert to GraphML
        graphml_str = converter.convert(str(sample_onnx_model))
        root = ET.fromstring(graphml_str)
        
        # Find nodes with hierarchy tags
        all_nodes = root.findall(".//{http://graphml.graphdrawing.org/xmlns}node")
        nodes_with_hierarchy = []
        
        for node in all_nodes:
            # Check if node has hierarchy tag data
            data_elements = node.findall(".//{http://graphml.graphdrawing.org/xmlns}data")
            for data in data_elements:
                if data.get("key") == "n1":  # hierarchy tag key
                    nodes_with_hierarchy.append((node, data.text))
                    break
        
        # Should have nodes with hierarchy tags
        assert len(nodes_with_hierarchy) > 0
        
        # Verify nodes are properly placed in hierarchical structure
        compound_nodes = []
        for node in all_nodes:
            nested_graph = node.find("./{http://graphml.graphdrawing.org/xmlns}graph")
            if nested_graph is not None:
                compound_nodes.append(node)
        
        assert len(compound_nodes) >= 2  # Should have hierarchical structure
    
    def test_compound_node_structure(self, sample_onnx_model, hierarchical_htp_metadata):
        """Test compound node hierarchy is properly built in GraphML output."""
        converter = HierarchicalGraphMLConverter(
            str(hierarchical_htp_metadata),
            exclude_initializers=True
        )
        
        # Convert to GraphML
        graphml_str = converter.convert(str(sample_onnx_model))
        root = ET.fromstring(graphml_str)
        
        # Find compound nodes (nodes with nested graphs)
        all_nodes = root.findall(".//{http://graphml.graphdrawing.org/xmlns}node")
        compound_nodes = []
        for node in all_nodes:
            nested_graph = node.find("./{http://graphml.graphdrawing.org/xmlns}graph")
            if nested_graph is not None:
                compound_nodes.append(node)
        
        # Should have multiple compound nodes for hierarchy
        assert len(compound_nodes) >= 2
        
        # Verify nested structure exists
        nested_graphs = root.findall(".//{http://graphml.graphdrawing.org/xmlns}graph")
        assert len(nested_graphs) > 1  # Main graph plus nested graphs
        
        # Verify we have both compound nodes (modules) and operation nodes
        # The main graph should contain operation nodes at the root level
        main_graph = root.find("./{http://graphml.graphdrawing.org/xmlns}graph")
        main_graph_nodes = main_graph.findall("./{http://graphml.graphdrawing.org/xmlns}node")
        
        # Should have nodes in the main graph (compound nodes or operation nodes)
        assert len(main_graph_nodes) > 0
    
    def test_deep_hierarchy(self, sample_onnx_model, deep_hierarchy_metadata):
        """Test deep nested hierarchy conversion."""
        converter = HierarchicalGraphMLConverter(
            str(deep_hierarchy_metadata),
            exclude_initializers=True
        )
        
        graphml_str = converter.convert(str(sample_onnx_model))
        root = ET.fromstring(graphml_str)
        
        # Check deep compound nodes exist
        nodes = root.findall(".//{http://graphml.graphdrawing.org/xmlns}node")
        compound_nodes = []
        for node in nodes:
            # Check if this node contains a nested graph element (compound node)
            nested_graph = node.find("./{http://graphml.graphdrawing.org/xmlns}graph")
            if nested_graph is not None:
                compound_nodes.append(node)
        
        # Verify deep hierarchy exists (should have multiple nested levels)
        assert len(compound_nodes) >= 4  # DeepModel > encoder > layer > 0 > attention
    
    def test_graphml_metadata_inclusion(self, sample_onnx_model, hierarchical_htp_metadata):
        """Test that metadata is included in GraphML output."""
        converter = HierarchicalGraphMLConverter(
            str(hierarchical_htp_metadata),
            exclude_initializers=True
        )
        
        graphml_str = converter.convert(str(sample_onnx_model))
        root = ET.fromstring(graphml_str)
        
        # Find the main graph element (first graph is the main one)
        graphs = root.findall(".//{http://graphml.graphdrawing.org/xmlns}graph") 
        assert len(graphs) >= 1
        main_graph = graphs[0]
        
        # Check for metadata data elements in the main graph
        data_elements = main_graph.findall("./{http://graphml.graphdrawing.org/xmlns}data")
        metadata_keys = [d.get("key") for d in data_elements]
        
        # Check that metadata is present (should have some metadata keys)
        assert len(metadata_keys) > 0
        
        # Check for format version metadata (should be one of the m keys)
        format_version_keys = [k for k in metadata_keys if k.startswith("m")]
        assert len(format_version_keys) > 0
        
        # Verify that data elements have content
        non_empty_data = [d for d in data_elements if d.text and len(d.text.strip()) > 0]
        assert len(non_empty_data) > 0
    
    def test_nested_subgraphs(self, sample_onnx_model, hierarchical_htp_metadata):
        """Test that nested subgraphs are created for compound nodes."""
        converter = HierarchicalGraphMLConverter(
            str(hierarchical_htp_metadata),
            exclude_initializers=True
        )
        
        graphml_str = converter.convert(str(sample_onnx_model))
        root = ET.fromstring(graphml_str)
        
        # Find all compound nodes (nodes containing nested graphs)
        all_nodes = root.findall(".//{http://graphml.graphdrawing.org/xmlns}node")
        compound_nodes = []
        for node in all_nodes:
            nested_graph = node.find("./{http://graphml.graphdrawing.org/xmlns}graph")
            if nested_graph is not None:
                compound_nodes.append(node)
        
        # Should have compound nodes
        assert len(compound_nodes) > 0
        
        # Check that we have nested graphs
        all_graphs = root.findall(".//{http://graphml.graphdrawing.org/xmlns}graph")
        assert len(all_graphs) > 1  # Main graph plus nested graphs
        
        # Verify the nested structure exists
        assert len(compound_nodes) >= 2  # Should have hierarchical structure
    
    def test_edge_preservation(self, sample_onnx_model, hierarchical_htp_metadata):
        """Test that edges are preserved in hierarchical structure."""
        converter = HierarchicalGraphMLConverter(
            str(hierarchical_htp_metadata),
            exclude_initializers=True
        )
        
        graphml_str = converter.convert(str(sample_onnx_model))
        root = ET.fromstring(graphml_str)
        
        # Find all edges
        edges = root.findall(".//{http://graphml.graphdrawing.org/xmlns}edge")
        assert len(edges) > 0
        
        # Edges should connect nodes across hierarchy
        edge_sources = [e.get("source") for e in edges]
        edge_targets = [e.get("target") for e in edges]
        
        # Should have connections between input, Add, MatMul, and output
        assert any("input" in src for src in edge_sources)
        assert any("output" in tgt for tgt in edge_targets)
    
    def test_fallback_to_flat_structure(self, sample_onnx_model, tmp_path):
        """Test fallback to flat structure when no hierarchy is present."""
        # Create minimal metadata without hierarchy
        minimal_metadata = {"export_data": {}}
        metadata_path = tmp_path / "minimal.json"
        with open(metadata_path, "w") as f:
            json.dump(minimal_metadata, f)
        
        converter = HierarchicalGraphMLConverter(
            str(metadata_path),
            exclude_initializers=True
        )
        
        graphml_str = converter.convert(str(sample_onnx_model))
        root = ET.fromstring(graphml_str)
        
        # Should still generate valid GraphML
        assert root.tag == "{http://graphml.graphdrawing.org/xmlns}graphml"
        
        # But no compound nodes (since no valid modules metadata)
        nodes = root.findall(".//{http://graphml.graphdrawing.org/xmlns}node")
        compound_nodes = []
        for node in nodes:
            # Check if this node contains a nested graph element (compound node)
            nested_graph = node.find("./{http://graphml.graphdrawing.org/xmlns}graph")
            if nested_graph is not None:
                compound_nodes.append(node)
        assert len(compound_nodes) == 0
        
        # Should have regular nodes
        assert len(nodes) > 0
    
    def test_module_class_name_in_compounds(self, sample_onnx_model, hierarchical_htp_metadata):
        """Test that compound nodes include module class names."""
        converter = HierarchicalGraphMLConverter(
            str(hierarchical_htp_metadata),
            exclude_initializers=True
        )
        
        graphml_str = converter.convert(str(sample_onnx_model))
        root = ET.fromstring(graphml_str)
        
        # Find compound nodes
        nodes = root.findall(".//{http://graphml.graphdrawing.org/xmlns}node")
        compound_nodes = []
        for node in nodes:
            # Check if this node contains a nested graph element (compound node)
            nested_graph = node.find("./{http://graphml.graphdrawing.org/xmlns}graph")
            if nested_graph is not None:
                compound_nodes.append(node)
        
        assert len(compound_nodes) > 0
        
        # Check that compound nodes have class name data
        found_class_names = False
        for compound_node in compound_nodes:
            data_elements = compound_node.findall(".//{http://graphml.graphdrawing.org/xmlns}data")
            op_types = [d.text for d in data_elements if d.get("key") == "d0"]  # op_type key
            if "AddLayer" in op_types or "TestModel" in op_types:
                found_class_names = True
                break
        assert found_class_names