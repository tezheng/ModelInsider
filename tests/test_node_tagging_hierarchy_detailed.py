"""
Detailed test case for NODE TAGGING hierarchy display to check for missing nodes and styling.
"""

import pytest
from io import StringIO
from rich.console import Console

# Force reload to ensure we get the latest code
import importlib
import modelexport.strategies.htp.export_monitor
importlib.reload(modelexport.strategies.htp.export_monitor)

from modelexport.strategies.htp.export_monitor import HTPExportMonitor, HTPExportStep, HTPExportMonitorConfig as Config


def test_node_tagging_hierarchy_missing_nodes(tmp_path):
    """Test that NODE TAGGING hierarchy tree shows ALL nodes including operations."""
    output_path = str(tmp_path / "test.onnx")
    
    # Capture console output
    string_buffer = StringIO()
    test_console = Console(file=string_buffer, width=120, force_terminal=True, highlight=False)
    
    monitor = HTPExportMonitor(
        output_path=output_path,
        model_name="test-model",
        verbose=True,
        embed_hierarchy=True
    )
    
    # Replace monitor's console with our test console
    monitor.console = test_console
    
    # Create test data with hierarchy and tagged nodes
    hierarchy = {
        "": {"class_name": "BertModel", "traced_tag": "/BertModel"},
        "embeddings": {"class_name": "BertEmbeddings", "traced_tag": "/BertModel/BertEmbeddings"},
        "encoder": {"class_name": "BertEncoder", "traced_tag": "/BertModel/BertEncoder"},
        "encoder.layer": {"class_name": "ModuleList", "traced_tag": "/BertModel/BertEncoder/ModuleList"},
        "encoder.layer.0": {"class_name": "BertLayer", "traced_tag": "/BertModel/BertEncoder/BertLayer.0"},
        "encoder.layer.0.attention": {"class_name": "BertAttention", "traced_tag": "/BertModel/BertEncoder/BertLayer.0/BertAttention"},
        "encoder.layer.0.attention.self": {"class_name": "BertSdpaSelfAttention", "traced_tag": "/BertModel/BertEncoder/BertLayer.0/BertAttention/BertSdpaSelfAttention"},
        "encoder.layer.0.attention.output": {"class_name": "BertSelfOutput", "traced_tag": "/BertModel/BertEncoder/BertLayer.0/BertAttention/BertSelfOutput"},
        "encoder.layer.0.intermediate": {"class_name": "BertIntermediate", "traced_tag": "/BertModel/BertEncoder/BertLayer.0/BertIntermediate"},
        "encoder.layer.0.intermediate.intermediate_act_fn": {"class_name": "GELUActivation", "traced_tag": "/BertModel/BertEncoder/BertLayer.0/BertIntermediate/GELUActivation"},
        "encoder.layer.0.output": {"class_name": "BertOutput", "traced_tag": "/BertModel/BertEncoder/BertLayer.0/BertOutput"},
        "pooler": {"class_name": "BertPooler", "traced_tag": "/BertModel/BertPooler"}
    }
    
    # Tagged nodes including operations
    tagged_nodes = {
        # Embeddings operations
        "Constant_0": "/BertModel/BertEmbeddings",
        "Constant_1": "/BertModel/BertEmbeddings",
        "Add_0": "/BertModel/BertEmbeddings",
        "Add_1": "/BertModel/BertEmbeddings",
        "Gather_0": "/BertModel/BertEmbeddings",
        "Gather_1": "/BertModel/BertEmbeddings",
        "Gather_2": "/BertModel/BertEmbeddings",
        "LayerNormalization_0": "/BertModel/BertEmbeddings",
        
        # Attention operations
        "MatMul_0": "/BertModel/BertEncoder/BertLayer.0/BertAttention/BertSdpaSelfAttention",
        "Add_2": "/BertModel/BertEncoder/BertLayer.0/BertAttention/BertSdpaSelfAttention",
        "Reshape_0": "/BertModel/BertEncoder/BertLayer.0/BertAttention/BertSdpaSelfAttention",
        "Softmax_0": "/BertModel/BertEncoder/BertLayer.0/BertAttention/BertSdpaSelfAttention",
        
        # Root level operations
        "Constant_2": "/BertModel",
        "Unsqueeze_0": "/BertModel",
    }
    
    # Update the monitor with NODE_TAGGING data
    monitor.update(
        HTPExportStep.NODE_TAGGING,
        total_nodes=14,
        tagged_nodes=tagged_nodes,
        coverage=100.0,
        tagging_stats={
            "direct_matches": 10,
            "parent_matches": 2,
            "root_fallbacks": 2
        },
        hierarchy=hierarchy,
        op_counts={
            "Add": 3,
            "MatMul": 1,
            "Gather": 3,
            "LayerNormalization": 1,
            "Softmax": 1,
            "Constant": 3,
            "Reshape": 1,
            "Unsqueeze": 1
        }
    )
    
    # Get console output
    output = string_buffer.getvalue()
    
    # Print output for debugging
    print("\n=== ACTUAL OUTPUT ===")
    print(output)
    print("=== END OUTPUT ===\n")
    
    # Check for hierarchy tree
    assert "Complete HF Hierarchy with ONNX Nodes:" in output
    
    # Check that the hierarchy tree includes ONNX operations
    # The baseline shows operations under each module
    print("\nChecking for ONNX operations in hierarchy tree...")
    
    # Should show operations like:
    # ├── Add (2 ops)
    # ├── Gather (3 ops)
    # └── LayerNormalization: /embeddings/LayerNorm/LayerNormalization
    
    # Check for specific operations that should be shown
    operations_found = {
        "Add": "Add" in output and "ops)" in output,
        "Gather": "Gather" in output and "ops)" in output,
        "LayerNormalization": "LayerNormalization" in output,
        "Softmax": "Softmax" in output,
        "MatMul": "MatMul" in output,
    }
    
    print("Operations found in hierarchy tree:")
    for op, found in operations_found.items():
        print(f"  {op}: {'YES' if found else 'NO'}")
    
    # These assertions should FAIL if operations are missing
    missing_operations = not all(operations_found.values())
    if missing_operations:
        print("\n!!! FAILURE: ONNX operations are missing from the hierarchy tree !!!")
        print("The hierarchy tree should show operations grouped by type under each module.")
    
    # Check for node names in single operations (e.g., LayerNormalization_0)
    has_operation_names = "LayerNormalization_0" in output or "Softmax_0" in output or "MatMul_0" in output
    
    print(f"\nHas operation names (e.g., LayerNormalization_0, Softmax_0): {has_operation_names}")
    
    # Assert to make test fail if operations are missing
    assert not missing_operations, "ONNX operations are missing from hierarchy tree"
    assert has_operation_names, "Operation names (e.g., LayerNormalization_0) are missing"


def test_node_tagging_hierarchy_styling(tmp_path):
    """Test that NODE TAGGING hierarchy tree has correct styling."""
    output_path = str(tmp_path / "test.onnx")
    
    # Capture console output with ANSI codes
    string_buffer = StringIO()
    test_console = Console(file=string_buffer, width=120, force_terminal=True, highlight=False)
    
    monitor = HTPExportMonitor(
        output_path=output_path,
        model_name="test-model",
        verbose=True,
        embed_hierarchy=True
    )
    
    monitor.console = test_console
    
    # Simple test data
    hierarchy = {
        "": {"class_name": "BertModel", "traced_tag": "/BertModel"},
        "encoder": {"class_name": "BertEncoder", "traced_tag": "/BertModel/BertEncoder"},
        "encoder.layer.0": {"class_name": "BertLayer", "traced_tag": "/BertModel/BertEncoder/BertLayer.0"},
        "encoder.layer.0.attention": {"class_name": "BertAttention", "traced_tag": "/BertModel/BertEncoder/BertLayer.0/BertAttention"},
    }
    
    tagged_nodes = {
        "node_0": "/BertModel/BertEncoder/BertLayer.0/BertAttention",
        "node_1": "/BertModel/BertEncoder/BertLayer.0/BertAttention",
    }
    
    monitor.update(
        HTPExportStep.NODE_TAGGING,
        total_nodes=2,
        tagged_nodes=tagged_nodes,
        coverage=100.0,
        hierarchy=hierarchy,
    )
    
    output = string_buffer.getvalue()
    
    print("\n=== STYLED OUTPUT ===")
    print(repr(output))
    print("=== END OUTPUT ===\n")
    
    # Check for gray/dim style on text after colon
    # In the hierarchy tree, text after colon should be dim (gray)
    # Example: "BertAttention: encoder.layer.0.attention (39 nodes)"
    #          ^bold         ^dim                       ^normal
    
    # Check if "encoder.layer.0.attention" has dim style
    # ANSI code for dim is \x1b[2m or similar
    has_dim_style = "\\x1b[2m" in repr(output) or "dim" in output.lower()
    
    print(f"Has dim/gray style for text after colon: {has_dim_style}")
    
    # Check for cyan style on node counts
    # "(2 nodes)" should have cyan color for the number
    # ANSI code for cyan is \x1b[36m or \x1b[1;36m (bold cyan)
    has_cyan_counts = "\\x1b[36m" in repr(output) or "\\x1b[1;36m" in repr(output)
    
    print(f"Has cyan style for node counts: {has_cyan_counts}")
    
    # These assertions should FAIL if styling is incorrect
    assert has_dim_style, "Text after colon should have dim/gray style"
    assert has_cyan_counts, "Node counts should have cyan style"


def test_export_config_styling(tmp_path):
    """Test that export configuration values have correct green styling."""
    output_path = str(tmp_path / "test.onnx")
    
    string_buffer = StringIO()
    test_console = Console(file=string_buffer, width=120, force_terminal=True, highlight=False)
    
    monitor = HTPExportMonitor(
        output_path=output_path,
        model_name="prajjwal1/bert-tiny",
        verbose=True,
    )
    
    monitor.console = test_console
    
    # Test ONNX export step
    monitor.update(
        HTPExportStep.ONNX_EXPORT,
        opset_version=17,
        do_constant_folding=True,
        input_names=["input_ids", "attention_mask"],
        onnx_size_mb=16.5
    )
    
    output = string_buffer.getvalue()
    
    print("\n=== ONNX EXPORT OUTPUT ===")
    print(output)
    print(repr(output))
    print("=== END OUTPUT ===\n")
    
    # Check that opset version number is green
    # Green ANSI code is \x1b[32m or \x1b[1;32m
    has_green_opset = "\\x1b[32m17\\x1b" in repr(output) or "\\x1b[1;32m17\\x1b" in repr(output)
    
    print(f"Opset version (17) has green style: {has_green_opset}")
    
    # These should FAIL if styling is missing
    assert has_green_opset, "Opset version number should be green"


def test_input_generation_styling(tmp_path):
    """Test that input generation step has correct green styling for values."""
    output_path = str(tmp_path / "test.onnx")
    
    string_buffer = StringIO()
    test_console = Console(file=string_buffer, width=120, force_terminal=True, highlight=False)
    
    monitor = HTPExportMonitor(
        output_path=output_path,
        model_name="prajjwal1/bert-tiny",
        verbose=True,
    )
    
    monitor.console = test_console
    
    # Test input generation step
    monitor.update(
        HTPExportStep.INPUT_GEN,
        method="auto_generated",
        model_type="bert",
        task="feature-extraction",
        inputs={
            "input_ids": {"shape": [2, 16], "dtype": "torch.int64"},
            "attention_mask": {"shape": [2, 16], "dtype": "torch.int64"},
        }
    )
    
    output = string_buffer.getvalue()
    
    print("\n=== INPUT GEN OUTPUT ===")
    print(output)
    print(repr(output))
    print("=== END OUTPUT ===\n")
    
    # Check that config values are green
    # "bert" and "feature-extraction" should be green
    has_green_model_type = "\\x1b[32mbert\\x1b" in repr(output) or "\\x1b[1;32mbert\\x1b" in repr(output)
    has_green_task = "\\x1b[32mfeature-extraction\\x1b" in repr(output) or "\\x1b[1;32mfeature-extraction\\x1b" in repr(output)
    
    print(f"Model type (bert) has green style: {has_green_model_type}")
    print(f"Task (feature-extraction) has green style: {has_green_task}")
    
    # These should FAIL if styling is missing
    assert has_green_model_type, "Model type value should be green"
    assert has_green_task, "Task value should be green"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])