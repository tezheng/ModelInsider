"""
Test suite for GraphML XML writer.

Tests the GraphML generation functionality including:
- XML structure creation
- Attribute key definitions
- Node and edge elements
- Metadata handling
"""

import xml.etree.ElementTree as ET

import pytest

from modelexport.graphml.graphml_writer import GraphMLWriter
from modelexport.graphml.utils import EdgeData, GraphData, NodeData, NodeType


@pytest.mark.graphml
@pytest.mark.unit
class TestGraphMLWriter:
    """Test cases for GraphML XML generation."""
    
    @pytest.mark.smoke
    def test_writer_initialization(self):
        """Test writer can be initialized."""
        writer = GraphMLWriter()
        assert writer is not None
    
    @pytest.mark.smoke
    def test_empty_graph_generation(self):
        """Test generating GraphML for empty graph."""
        writer = GraphMLWriter()
        graph_data = GraphData()
        
        element = writer.write(graph_data)
        xml_str = writer.to_string(element)
        
        # Parse and verify
        root = ET.fromstring(xml_str)
        assert root.tag.endswith("graphml")
        
        # Check for graph element
        graphs = root.findall(".//{http://graphml.graphdrawing.org/xmlns}graph")
        assert len(graphs) == 1
        assert graphs[0].get("edgedefault") == "directed"
    
    @pytest.mark.sanity
    def test_node_creation(self):
        """Test creating node elements."""
        writer = GraphMLWriter()
        graph_data = GraphData()
        
        # Add a test node
        node = NodeData(
            id="test_node",
            name="TestOp",
            op_type="Add",
            node_type=NodeType.OPERATION
        )
        graph_data.nodes.append(node)
        
        element = writer.write(graph_data)
        xml_str = writer.to_string(element)
        
        # Verify node in output
        root = ET.fromstring(xml_str)
        nodes = root.findall(".//{http://graphml.graphdrawing.org/xmlns}node")
        assert len(nodes) == 1
        assert nodes[0].get("id") == "test_node"
    
    @pytest.mark.sanity
    def test_edge_creation(self):
        """Test creating edge elements."""
        writer = GraphMLWriter()
        graph_data = GraphData()
        
        # Add two nodes and an edge
        node1 = NodeData(id="n1", name="Op1", op_type="Input")
        node2 = NodeData(id="n2", name="Op2", op_type="Add")
        edge = EdgeData(
            source_id="n1",
            target_id="n2",
            tensor_name="tensor_x"
        )
        
        graph_data.nodes.extend([node1, node2])
        graph_data.edges.append(edge)
        
        element = writer.write(graph_data)
        xml_str = writer.to_string(element)
        
        # Verify edge in output
        root = ET.fromstring(xml_str)
        edges = root.findall(".//{http://graphml.graphdrawing.org/xmlns}edge")
        assert len(edges) == 1
        assert edges[0].get("source") == "n1"
        assert edges[0].get("target") == "n2"
    
    @pytest.mark.sanity
    def test_attribute_keys_defined(self):
        """Test that attribute keys are properly defined."""
        writer = GraphMLWriter()
        graph_data = GraphData()
        
        element = writer.write(graph_data)
        xml_str = writer.to_string(element)
        
        # Check for key definitions
        root = ET.fromstring(xml_str)
        keys = root.findall(".//{http://graphml.graphdrawing.org/xmlns}key")
        
        # Should have multiple keys defined
        assert len(keys) > 0
        
        # Check for specific keys  
        key_ids = [key.get("id") for key in keys]
        assert "n0" in key_ids  # op_type
        assert "n1" in key_ids  # hierarchy_tag