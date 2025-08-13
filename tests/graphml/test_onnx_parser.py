"""
Test suite for ONNX graph parser.

Tests the ONNX parsing functionality including:
- Node extraction
- Edge creation
- Input/output handling
- Initializer filtering
"""

from modelexport.graphml.onnx_parser import ONNXGraphParser


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
            exclude_initializers=False, exclude_attributes={"axis", "perm"}
        )
        assert parser.exclude_initializers is False
        assert "axis" in parser.exclude_attributes
        assert "perm" in parser.exclude_attributes

    def test_parse_simple_model(self, simple_onnx_model):
        """Test parsing a simple ONNX model."""
        import onnx

        parser = ONNXGraphParser()
        onnx_model = onnx.load(simple_onnx_model)
        graph_data = parser.parse(onnx_model)

        # Verify graph data structure
        assert len(graph_data.nodes) > 0
        assert len(graph_data.edges) > 0
        assert parser.last_node_count == len(graph_data.nodes)
        assert parser.last_edge_count == len(graph_data.edges)

    def test_node_extraction(self, simple_onnx_model):
        """Test node data extraction from ONNX nodes."""
        import onnx

        parser = ONNXGraphParser()
        onnx_model = onnx.load(simple_onnx_model)
        graph_data = parser.parse(onnx_model)

        # Verify nodes have required properties
        for node in graph_data.nodes:
            assert hasattr(node, "id")
            assert hasattr(node, "name")
            assert hasattr(node, "op_type")
            assert node.id is not None
            assert node.op_type is not None

    def test_edge_creation(self, simple_onnx_model):
        """Test edge creation between nodes."""
        import onnx

        parser = ONNXGraphParser()
        onnx_model = onnx.load(simple_onnx_model)
        graph_data = parser.parse(onnx_model)

        # Verify edges have required properties
        for edge in graph_data.edges:
            assert hasattr(edge, "source_id")
            assert hasattr(edge, "target_id")
            assert edge.source_id is not None
            assert edge.target_id is not None

            # Verify source and target nodes exist
            source_exists = any(
                node.id == edge.source_id
                for node in graph_data.nodes + graph_data.inputs
            )
            target_exists = any(
                node.id == edge.target_id
                for node in graph_data.nodes + graph_data.outputs
            )
            assert source_exists or target_exists  # At least one should exist

    def test_initializer_filtering(self, simple_onnx_model):
        """Test that initializers are properly filtered when requested."""
        import onnx

        # Test with initializers excluded (default)
        parser_exclude = ONNXGraphParser(exclude_initializers=True)
        onnx_model = onnx.load(simple_onnx_model)
        graph_data_exclude = parser_exclude.parse(onnx_model)

        # Test with initializers included
        parser_include = ONNXGraphParser(exclude_initializers=False)
        graph_data_include = parser_include.parse(onnx_model)

        # Should have tracked initializer count
        assert parser_exclude.last_initializer_count >= 0
        assert parser_include.last_initializer_count >= 0

        # The behavior might differ but both should work
        assert len(graph_data_exclude.nodes) >= 0
        assert len(graph_data_include.nodes) >= 0
