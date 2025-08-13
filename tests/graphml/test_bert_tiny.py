"""
Unit test for GraphML converter with mocked ONNX model.

This test verifies that the GraphML converter works correctly
without requiring actual model files.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import onnx
from onnx import helper, numpy_helper

from modelexport.graphml import ONNXToGraphMLConverter
from tests.constants import (
    BERT_VOCAB_SIZE,
    DEFAULT_BATCH_SIZE,
    DEFAULT_SEQUENCE_LENGTH,
    NODE_BERT_ADD,
    NODE_BERT_EMBEDDINGS,
    NODE_BERT_ENCODER,
)


def create_mock_bert_tiny_onnx():
    """Create a minimal mock BERT-tiny ONNX model for testing."""
    # Create input tensor
    input_ids = helper.make_tensor_value_info(
        "input_ids",
        onnx.TensorProto.INT64,
        [DEFAULT_BATCH_SIZE, DEFAULT_SEQUENCE_LENGTH],
    )

    # Note: intermediate tensors are created by operations, not declared

    # Create output tensor
    output = helper.make_tensor_value_info(
        "output",
        onnx.TensorProto.FLOAT,
        [DEFAULT_BATCH_SIZE, DEFAULT_SEQUENCE_LENGTH, DEFAULT_SEQUENCE_LENGTH],
    )

    # Create weight initializers (small for testing)
    embedding_weight = numpy_helper.from_array(
        np.random.randn(BERT_VOCAB_SIZE, DEFAULT_SEQUENCE_LENGTH).astype(np.float32),
        name="embedding.weight",
    )

    encoder_weight = numpy_helper.from_array(
        np.random.randn(DEFAULT_SEQUENCE_LENGTH, DEFAULT_SEQUENCE_LENGTH).astype(
            np.float32
        ),
        name="encoder.weight",
    )

    # Create nodes
    embedding_node = helper.make_node(
        "Gather",
        inputs=["embedding.weight", "input_ids"],
        outputs=["embeddings"],
        name=NODE_BERT_EMBEDDINGS,
    )

    encoder_node = helper.make_node(
        "MatMul",
        inputs=["embeddings", "encoder.weight"],
        outputs=["encoder_output"],
        name=NODE_BERT_ENCODER,
    )

    output_node = helper.make_node(
        "Add",
        inputs=["encoder_output", "embeddings"],
        outputs=["output"],
        name=NODE_BERT_ADD,
    )

    # Create the graph
    graph = helper.make_graph(
        [embedding_node, encoder_node, output_node],
        "mock_bert_tiny",
        [input_ids],
        [output],
        [embedding_weight, encoder_weight],
    )

    # Create the model
    model = helper.make_model(graph)

    # Add metadata
    model.metadata_props.append(
        onnx.StringStringEntryProto(key="model_type", value="bert-tiny")
    )

    return model


def test_bert_tiny_conversion():
    """Test converting mock bert-tiny to GraphML."""
    # Create mock ONNX model
    mock_model = create_mock_bert_tiny_onnx()

    # Use temporary files for testing
    with tempfile.TemporaryDirectory() as tmpdir:
        model_path = Path(tmpdir) / "mock_bert.onnx"
        output_path = Path(tmpdir) / "mock_bert.graphml"

        # Save mock model
        onnx.save(mock_model, str(model_path))

        # Convert to GraphML
        converter = ONNXToGraphMLConverter(hierarchical=False)
        converter.save(str(model_path), str(output_path))

        # Verify output file exists
        assert output_path.exists()

        # Get statistics
        stats = converter.get_statistics()
        assert stats["nodes"] == 3  # We created 3 nodes
        assert stats["edges"] > 0  # Should have edges between nodes

        # Verify GraphML content
        with open(output_path) as f:
            content = f.read()
            assert "graphml" in content
            assert "bert/embeddings" in content
            assert "bert/encoder" in content


def test_bert_tiny_mock():
    """Test with mock data when bert-tiny is not available."""
    from modelexport.graphml.graphml_writer import GraphMLWriter
    from modelexport.graphml.utils import EdgeData, GraphData, NodeData

    # Create mock graph data
    graph_data = GraphData()

    # Add some nodes
    graph_data.nodes.append(
        NodeData(
            id="bert_embeddings", name="/bert/embeddings", op_type="BertEmbeddings"
        )
    )
    graph_data.nodes.append(
        NodeData(id="bert_encoder", name="/bert/encoder", op_type="BertEncoder")
    )

    # Add an edge
    graph_data.edges.append(
        EdgeData(
            source_id="bert_embeddings",
            target_id="bert_encoder",
            tensor_name="hidden_states",
        )
    )

    # Write GraphML
    writer = GraphMLWriter()
    element = writer.write(graph_data)
    xml_str = writer.to_string(element)

    # Basic validation
    assert "graphml" in xml_str
    assert "bert_embeddings" in xml_str
    assert "bert_encoder" in xml_str


# Tests should be run using pytest, not directly as scripts
# Usage: pytest tests/graphml/test_bert_tiny.py -v
