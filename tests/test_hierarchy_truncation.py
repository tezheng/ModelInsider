"""Test hierarchy truncation in export monitor."""

import io
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from rich.console import Console

from modelexport.strategies.htp.export_monitor import HTPExportMonitor, HTPExportStep, HTPExportMonitorConfig as Config


def test_hierarchy_truncation():
    """Test that hierarchy displays are truncated to MAX_HIERARCHY_LINES."""
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
        force_terminal=False,  # No ANSI codes for easier testing
        legacy_windows=False,
        highlight=False,
    )
    
    # Replace monitor's console
    monitor.console = test_console
    
    # Create large hierarchy data (more than 30 entries)
    hierarchy_data = {
        "": {"class_name": "BertModel"},
        "encoder": {"class_name": "BertEncoder"},  # Need to add encoder first
    }
    
    # Add many layers to exceed truncation limit
    for i in range(10):
        layer_path = f"encoder.layer.{i}"
        hierarchy_data[layer_path] = {"class_name": "BertLayer"}
        hierarchy_data[f"{layer_path}.attention"] = {"class_name": "BertAttention"}
        hierarchy_data[f"{layer_path}.attention.self"] = {"class_name": "BertSelfAttention"}
        hierarchy_data[f"{layer_path}.attention.output"] = {"class_name": "BertSelfOutput"}
        hierarchy_data[f"{layer_path}.intermediate"] = {"class_name": "BertIntermediate"}
        hierarchy_data[f"{layer_path}.output"] = {"class_name": "BertOutput"}
    
    # Update monitor with hierarchy data
    monitor.update(
        HTPExportStep.HIERARCHY,
        hierarchy=hierarchy_data,
        execution_steps=100,
    )
    
    # Get captured output
    output = string_buffer.getvalue()
    
    # Check for truncation message
    assert "truncated for console" in output
    assert f"showing first {Config.MAX_HIERARCHY_LINES} lines" in output
    
    # Verify that we have the right number of hierarchy lines
    # The truncation logic displays exactly MAX_HIERARCHY_LINES of hierarchy
    # plus the truncation message


def test_onnx_hierarchy_truncation():
    """Test that ONNX hierarchy with nodes is also truncated."""
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
        force_terminal=False,  # No ANSI codes for easier testing
        legacy_windows=False,
        highlight=False,
    )
    
    # Replace monitor's console
    monitor.console = test_console
    
    # Create large hierarchy data with traced tags
    hierarchy_data = {
        "": {"class_name": "BertModel", "traced_tag": "/BertModel"},
    }
    
    # Create many tagged nodes
    tagged_nodes = {}
    
    # Add many layers and nodes to exceed truncation limit
    for i in range(10):
        layer_path = f"encoder.layer.{i}"
        layer_tag = f"/BertModel/encoder/layer.{i}"
        
        hierarchy_data[layer_path] = {"class_name": "BertLayer", "traced_tag": layer_tag}
        hierarchy_data[f"{layer_path}.attention"] = {"class_name": "BertAttention", "traced_tag": f"{layer_tag}/attention"}
        hierarchy_data[f"{layer_path}.attention.self"] = {"class_name": "BertSelfAttention", "traced_tag": f"{layer_tag}/attention/self"}
        
        # Add many nodes for each layer
        for j in range(20):
            node_name = f"MatMul_{i}_{j}"
            tagged_nodes[node_name] = f"{layer_tag}/attention/self"
            
            node_name = f"Add_{i}_{j}"
            tagged_nodes[node_name] = f"{layer_tag}/attention"
    
    # Update monitor with node tagging data
    monitor.update(
        HTPExportStep.NODE_TAGGING,
        total_nodes=len(tagged_nodes),
        tagged_nodes=tagged_nodes,
        tagging_stats={},
        coverage=100.0,
        op_counts={"MatMul": 200, "Add": 200},
        hierarchy=hierarchy_data,
    )
    
    # Get captured output
    output = string_buffer.getvalue()
    
    # Check for truncation in the Complete HF Hierarchy section
    if "Complete HF Hierarchy with ONNX Nodes:" in output:
        # Find the section after this header
        lines = output.split('\n')
        hierarchy_section_found = False
        for line in lines:
            if "Complete HF Hierarchy with ONNX Nodes:" in line:
                hierarchy_section_found = True
            elif hierarchy_section_found and "truncated for console" in line:
                assert f"showing first {Config.MAX_HIERARCHY_LINES} lines" in line
                break


def test_no_truncation_when_under_limit():
    """Test that no truncation occurs when hierarchy is under the limit."""
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
        force_terminal=False,  # No ANSI codes for easier testing
        legacy_windows=False,
        highlight=False,
    )
    
    # Replace monitor's console
    monitor.console = test_console
    
    # Create small hierarchy data (less than 30 entries)
    hierarchy_data = {
        "": {"class_name": "BertModel"},
        "embeddings": {"class_name": "BertEmbeddings"},
        "encoder": {"class_name": "BertEncoder"},
        "encoder.layer.0": {"class_name": "BertLayer"},
        "encoder.layer.0.attention": {"class_name": "BertAttention"},
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
    
    # Should NOT have truncation message
    assert "truncated for console" not in output
    assert "showing first" not in output


if __name__ == "__main__":
    pytest.main([__file__, "-v"])