"""Test module hierarchy format."""

import io
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from rich.console import Console

from modelexport.strategies.htp_new.export_monitor import HTPExportMonitor, HTPExportMonitor
from modelexport.strategies.htp_new.base_writer import ExportStep as HTPExportStep


def test_module_hierarchy_format():
    """Test that module hierarchy displays path: ClassName format."""
    # Mock output path
    output_path = Path("temp/test_model.onnx")
    
    # Initialize monitor
    monitor = HTPExportMonitor(
        output_path=str(output_path),
        model_name="test/model",
        verbose=True,
        enable_report=True,
        embed_hierarchy=True,
    )
    
    # Create test console with capture
    string_buffer = io.StringIO()
    test_console = Console(
        file=string_buffer,
        width=120,
        force_terminal=True,
        legacy_windows=False,
        highlight=False,
    )
    
    # Replace monitor's console
    monitor.console = test_console
    
    # Create test hierarchy data
    hierarchy_data = {
        "": {"class_name": "BertModel"},
        "embeddings": {"class_name": "BertEmbeddings"},
        "encoder": {"class_name": "BertEncoder"},
        "encoder.layer.0": {"class_name": "BertLayer"},
        "encoder.layer.0.attention": {"class_name": "BertAttention"},
        "encoder.layer.0.attention.self": {"class_name": "BertSdpaSelfAttention"},
        "encoder.layer.0.attention.output": {"class_name": "BertSelfOutput"},
        "pooler": {"class_name": "BertPooler"},
    }
    
    # Update monitor with hierarchy data
    monitor.update(
        HTPExportStep.HIERARCHY,
        hierarchy=hierarchy_data,
        execution_steps=10,
    )
    
    # Get captured output
    output = string_buffer.getvalue()
    
    # Check that the format is ClassName: path
    # Need to check with ANSI codes since force_terminal=True
    # The actual text will have ANSI codes for styling
    import re
    
    # Strip ANSI codes for easier checking
    plain_output = re.sub(r'\x1b\[[0-9;]*m', '', output)
    
    assert "BertEmbeddings: embeddings" in plain_output
    assert "BertEncoder: encoder" in plain_output
    assert "BertLayer: encoder.layer.0" in plain_output
    assert "BertAttention: encoder.layer.0.attention" in plain_output
    assert "BertSdpaSelfAttention: encoder.layer.0.attention.self" in plain_output
    assert "BertSelfOutput: encoder.layer.0.attention.output" in plain_output
    assert "BertPooler: pooler" in plain_output
    
    # Check that the wrong format is NOT present
    assert "embeddings: BertEmbeddings" not in plain_output
    assert "encoder: BertEncoder" not in plain_output
    assert "encoder.layer.0: BertLayer" not in plain_output
    
    # Check styling (path should be dim, class name should be bold)
    # Path (dim): \x1b[2m
    # Bold: \x1b[1m
    assert "\\x1b[2membeddings\\x1b[0m" in repr(output)  # Path is dim
    assert "\\x1b[1mBertEmbeddings\\x1b[0m" in repr(output)  # Class name is bold


def test_module_hierarchy_with_counts_format():
    """Test that module hierarchy with counts displays path: ClassName format."""
    # Mock output path
    output_path = Path("temp/test_model.onnx")
    
    # Initialize monitor
    monitor = HTPExportMonitor(
        output_path=str(output_path),
        model_name="test/model",
        verbose=True,
        enable_report=True,
        embed_hierarchy=True,
    )
    
    # Create test console with capture
    string_buffer = io.StringIO()
    test_console = Console(
        file=string_buffer,
        width=120,
        force_terminal=True,
        legacy_windows=False,
        highlight=False,
    )
    
    # Replace monitor's console
    monitor.console = test_console
    
    # Create test hierarchy data with traced tags
    hierarchy_data = {
        "": {"class_name": "BertModel", "traced_tag": "/BertModel"},
        "embeddings": {"class_name": "BertEmbeddings", "traced_tag": "/BertModel/embeddings"},
        "encoder": {"class_name": "BertEncoder", "traced_tag": "/BertModel/encoder"},
        "encoder.layer.0": {"class_name": "BertLayer", "traced_tag": "/BertModel/encoder/layer.0"},
        "encoder.layer.0.attention": {"class_name": "BertAttention", "traced_tag": "/BertModel/encoder/layer.0/attention"},
    }
    
    # Create tagged nodes
    tagged_nodes = {
        "Add_0": "/BertModel/embeddings",
        "Add_1": "/BertModel/embeddings",
        "LayerNormalization_0": "/BertModel/encoder/layer.0",
        "MatMul_0": "/BertModel/encoder/layer.0/attention",
        "MatMul_1": "/BertModel/encoder/layer.0/attention",
        "MatMul_2": "/BertModel/encoder/layer.0/attention",
    }
    
    # Update monitor with node tagging data
    monitor.update(
        HTPExportStep.NODE_TAGGING,
        total_nodes=6,
        tagged_nodes=tagged_nodes,
        tagging_stats={},
        coverage=100.0,
        op_counts={"Add": 2, "LayerNormalization": 1, "MatMul": 3},
        hierarchy=hierarchy_data,
    )
    
    # Get captured output
    output = string_buffer.getvalue()
    
    # Strip ANSI codes for easier checking
    import re
    plain_output = re.sub(r'\x1b\[[0-9;]*m', '', output)
    
    # Check that the format is ClassName: path
    assert "BertEmbeddings: embeddings (2 nodes)" in plain_output
    assert "BertEncoder: encoder (4 nodes)" in plain_output
    assert "BertLayer: encoder.layer.0 (4 nodes)" in plain_output
    assert "BertAttention: encoder.layer.0.attention (3 nodes)" in plain_output
    
    # Check that the wrong format is NOT present
    assert "embeddings: BertEmbeddings" not in plain_output
    assert "encoder: BertEncoder" not in plain_output
    assert "encoder.layer.0: BertLayer" not in plain_output
    assert "encoder.layer.0.attention: BertAttention" not in plain_output


if __name__ == "__main__":
    pytest.main([__file__, "-v"])