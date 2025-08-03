"""
Test suite for ONNX graph parser.

Tests the ONNX parsing functionality including:
- Node extraction
- Edge creation
- Input/output handling
- Initializer filtering
"""

import numpy as np
import pytest
from onnx import TensorProto, helper

from modelexport.graphml.onnx_parser import ONNXGraphParser


@pytest.fixture
def simple_onnx_model():
    """Create a simple ONNX model for testing."""
    # Create a simple model: Input -> MatMul -> Add -> Output
    input_tensor = helper.make_tensor_value_info("input", TensorProto.FLOAT, [2, 3])
    output_tensor = helper.make_tensor_value_info("output", TensorProto.FLOAT, [2, 4])
    
    # Create weight initializer
    weight_init = helper.make_tensor(
        name="weight",
        data_type=TensorProto.FLOAT,
        dims=[3, 4],
        vals=np.random.rand(12).astype(np.float32).tolist()
    )
    
    # Create bias initializer
    bias_init = helper.make_tensor(
        name="bias",
        data_type=TensorProto.FLOAT,
        dims=[4],
        vals=[0.1, 0.2, 0.3, 0.4]
    )
    
    # Create MatMul node
    matmul_node = helper.make_node(
        "MatMul",
        inputs=["input", "weight"],
        outputs=["matmul_output"],
        name="MatMul_1"
    )
    
    # Create Add node
    add_node = helper.make_node(
        "Add",
        inputs=["matmul_output", "bias"],
        outputs=["output"],
        name="Add_1"
    )
    
    # Create graph
    graph = helper.make_graph(
        nodes=[matmul_node, add_node],
        name="SimpleModel",
        inputs=[input_tensor],
        outputs=[output_tensor],
        initializer=[weight_init, bias_init]
    )
    
    # Create model
    model = helper.make_model(graph)
    model.opset_import[0].version = 17
    
    return model


@pytest.mark.graphml
class TestONNXGraphParser:
    """Test cases for ONNX graph parsing."""
    
    def test_parser_initialization(self):
        """Test parser can be initialized with default settings."""
        parser = ONNXGraphParser()
        assert parser.exclude_initializers is True
        assert parser.exclude_attributes == set()
        assert parser.last_node_count == 0
        assert parser.last_edge_count == 0
        assert parser.last_initializer_count == 0
    
    def test_parser_with_options(self):
        """Test parser initialization with custom options."""
        parser = ONNXGraphParser(
            exclude_initializers=False,
            exclude_attributes={"axis", "perm"}
        )
        assert parser.exclude_initializers is False
        assert "axis" in parser.exclude_attributes
        assert "perm" in parser.exclude_attributes
    
    def test_parse_simple_model(self, simple_onnx_model):
        """Test parsing a simple ONNX model."""
        parser = ONNXGraphParser()
        graph_data = parser.parse(simple_onnx_model)
        
        # Verify graph data structure
        assert len(graph_data.nodes) > 0
        assert len(graph_data.edges) > 0
        assert parser.last_node_count == len(graph_data.nodes)
        assert parser.last_edge_count == len(graph_data.edges)
        
        # Check that we have the expected nodes (2 operation nodes)
        op_nodes = [n for n in graph_data.nodes if hasattr(n, 'op_type')]
        assert len(op_nodes) == 2
        
        # Check node types
        op_types = [n.op_type for n in op_nodes]
        assert "MatMul" in op_types
        assert "Add" in op_types
    
    def test_node_extraction(self, simple_onnx_model):
        """Test node data extraction from ONNX nodes."""
        parser = ONNXGraphParser()
        graph_data = parser.parse(simple_onnx_model)
        
        # Find the MatMul node
        matmul_nodes = [n for n in graph_data.nodes if hasattr(n, 'op_type') and n.op_type == "MatMul"]
        assert len(matmul_nodes) == 1
        
        matmul_node = matmul_nodes[0]
        assert matmul_node.name == "MatMul_1"
        assert matmul_node.id == "MatMul_1"
        assert matmul_node.op_type == "MatMul"
        
        # Find the Add node  
        add_nodes = [n for n in graph_data.nodes if hasattr(n, 'op_type') and n.op_type == "Add"]
        assert len(add_nodes) == 1
        
        add_node = add_nodes[0]
        assert add_node.name == "Add_1"
        assert add_node.id == "Add_1"
        assert add_node.op_type == "Add"
    
    def test_edge_creation(self, simple_onnx_model):
        """Test edge creation between nodes."""
        parser = ONNXGraphParser()
        graph_data = parser.parse(simple_onnx_model)
        
        # Should have edges connecting nodes
        assert len(graph_data.edges) > 0
        
        # Find edge from MatMul to Add (through matmul_output tensor)
        matmul_to_add_edges = [
            e for e in graph_data.edges 
            if e.source_id == "MatMul_1" and e.target_id == "Add_1"
        ]
        assert len(matmul_to_add_edges) == 1
        assert matmul_to_add_edges[0].tensor_name == "matmul_output"
        
        # Check edges from inputs
        input_edges = [e for e in graph_data.edges if "input" in e.source_id.lower()]
        assert len(input_edges) > 0  # Should have edge from input to MatMul
        
        # Check edges to outputs
        output_edges = [e for e in graph_data.edges if "output" in e.target_id.lower()]
        assert len(output_edges) > 0  # Should have edge from Add to output
    
    def test_initializer_filtering(self, simple_onnx_model):
        """Test that initializers are properly filtered when requested."""
        # Test with initializers excluded (default)
        parser_exclude = ONNXGraphParser(exclude_initializers=True)
        graph_data_excluded = parser_exclude.parse(simple_onnx_model)
        
        # Count edges that reference initializers (weight and bias)
        # When excluded, these edges should not exist
        initializer_edges_excluded = [
            e for e in graph_data_excluded.edges 
            if e.tensor_name in ["weight", "bias"]
        ]
        assert len(initializer_edges_excluded) == 0, "Initializer edges should be excluded"
        
        # Test with initializers included
        parser_include = ONNXGraphParser(exclude_initializers=False)
        graph_data_included = parser_include.parse(simple_onnx_model)
        
        # Count edges that reference initializers
        # When included, these should still not exist as edges because
        # initializers don't create nodes - they're just filtered when creating edges
        initializer_edges_included = [
            e for e in graph_data_included.edges 
            if e.tensor_name in ["weight", "bias"]
        ]
        assert len(initializer_edges_included) == 0, "Initializers don't create edges"
        
        # Verify parser stats (both parsers see 2 initializers in the model)
        assert parser_exclude.last_initializer_count == 2
        assert parser_include.last_initializer_count == 2
        
        # The key difference is that exclude_initializers affects whether
        # node inputs that reference initializers are skipped when creating edges.
        # In our model, MatMul has inputs ["input", "weight"] and Add has inputs ["matmul_output", "bias"]
        # Since weight and bias are initializers without producer nodes, no edges are created for them
        # regardless of the exclude_initializers setting.
        
        # Both parsers should have the same edges because initializers don't produce edges
        assert len(graph_data_excluded.edges) == len(graph_data_included.edges)