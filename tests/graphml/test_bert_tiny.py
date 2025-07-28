"""
Integration test with bert-tiny model.

This test verifies that the GraphML converter works with a real model.
"""

import pytest
from pathlib import Path

from modelexport.graphml.converter import ONNXToGraphMLConverter


@pytest.mark.skipif(
    not Path("temp/bert-tiny/model.onnx").exists(),
    reason="bert-tiny ONNX model not available"
)
def test_bert_tiny_conversion():
    """Test converting bert-tiny to GraphML."""
    model_path = "temp/bert-tiny/model.onnx"
    output_path = "temp/bert-tiny/model.graphml"
    
    converter = ONNXToGraphMLConverter()
    converter.save(model_path, output_path)
    
    # Verify output file exists
    assert Path(output_path).exists()
    
    # Get statistics
    stats = converter.get_statistics()
    assert stats["nodes"] > 0
    assert stats["edges"] > 0
    
    print(f"Conversion successful: {stats}")


def test_bert_tiny_mock():
    """Test with mock data when bert-tiny is not available."""
    from modelexport.graphml.utils import GraphData, NodeData, EdgeData
    from modelexport.graphml.graphml_writer import GraphMLWriter
    
    # Create mock graph data
    graph_data = GraphData()
    
    # Add some nodes
    graph_data.nodes.append(NodeData(
        id="bert_embeddings",
        name="/bert/embeddings",
        op_type="BertEmbeddings"
    ))
    graph_data.nodes.append(NodeData(
        id="bert_encoder",
        name="/bert/encoder",
        op_type="BertEncoder"
    ))
    
    # Add an edge
    graph_data.edges.append(EdgeData(
        source_id="bert_embeddings",
        target_id="bert_encoder",
        tensor_name="hidden_states"
    ))
    
    # Write GraphML
    writer = GraphMLWriter()
    element = writer.write(graph_data)
    xml_str = writer.to_string(element)
    
    # Basic validation
    assert "graphml" in xml_str
    assert "bert_embeddings" in xml_str
    assert "bert_encoder" in xml_str


if __name__ == "__main__":
    # Try real conversion if model exists
    if Path("temp/bert-tiny/model.onnx").exists():
        test_bert_tiny_conversion()
    else:
        print("bert-tiny model not found, running mock test")
        test_bert_tiny_mock()
        print("Mock test passed!")