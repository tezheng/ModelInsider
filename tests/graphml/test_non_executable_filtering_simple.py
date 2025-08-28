"""
Simple test for verifying non-executable module filtering in GraphML export.

Tests the fix for Linear issue TEZ-184.
"""

import json
import xml.etree.ElementTree as ET
from pathlib import Path

import pytest
import torch
import torch.nn as nn

from modelexport.graphml import ONNXToGraphMLConverter


class TestNonExecutableFilteringSimple:
    """Simple test for non-executable module filtering."""
    
    def test_filters_modules_with_execution_order_negative_one(self, tmp_path):
        """Test that modules with execution_order == -1 are filtered from GraphML."""
        # Create a simple ONNX model
        model = nn.Sequential(
            nn.Linear(10, 20),
            nn.ReLU(),
            nn.Linear(20, 5)
        )
        
        dummy_input = torch.randn(1, 10)
        onnx_path = tmp_path / "test.onnx"
        
        torch.onnx.export(
            model,
            dummy_input,
            str(onnx_path),
            input_names=["input"],
            output_names=["output"],
            opset_version=17
        )
        
        # Create HTP metadata with some modules having execution_order == -1
        htp_metadata = {
            "model": {
                "name_or_path": "test_model",
                "total_modules": 6
            },
            "modules": {
                "": {
                    "class_name": "Sequential",
                    "scope": "",
                    "traced_tag": "/Sequential",
                    "execution_order": 0,
                    "children": {
                        "0": {
                            "class_name": "Linear",
                            "scope": "0",
                            "traced_tag": "/Sequential/Linear.0",
                            "execution_order": 1  # Executed
                        },
                        "1": {
                            "class_name": "ReLU",
                            "scope": "1",
                            "traced_tag": "/Sequential/ReLU",
                            "execution_order": 2  # Executed
                        },
                        "2": {
                            "class_name": "Linear",
                            "scope": "2",
                            "traced_tag": "/Sequential/Linear.1",
                            "execution_order": 3  # Executed
                        },
                        "unused_embedding": {
                            "class_name": "Embedding",
                            "scope": "unused_embedding",
                            "traced_tag": "/Sequential/Embedding",
                            "execution_order": -1  # NOT executed - should be filtered
                        },
                        "unused_param": {
                            "class_name": "Parameter",
                            "scope": "unused_param",
                            "traced_tag": "/Sequential/Parameter",
                            "execution_order": -1  # NOT executed - should be filtered
                        }
                    }
                }
            }
        }
        
        # Save metadata
        metadata_path = tmp_path / "test_htp_metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(htp_metadata, f)
        
        # Convert to GraphML with the metadata
        converter = ONNXToGraphMLConverter(
            hierarchical=True,
            htp_metadata_path=str(metadata_path)
        )
        
        # Disable structural discovery to avoid model loading
        # Return the root module data (empty key)
        converter._enhance_with_structural_hierarchy = lambda x: x.get("", x) if isinstance(x, dict) else x.get("", x) if isinstance(x, dict) else x
        
        result = converter.convert(
            str(onnx_path),
            str(tmp_path / "output")
        )
        
        # Parse the GraphML
        graphml_path = result['graphml']
        tree = ET.parse(graphml_path)
        root = tree.getroot()
        
        # Define namespace for GraphML
        ns = {"": "http://graphml.graphdrawing.org/xmlns"}
        
        # Find all compound nodes (nodes with nested graphs)
        compound_nodes = []
        for node in root.findall(".//node", ns):
            nested_graph = node.find("graph", ns)
            if nested_graph is not None:
                compound_nodes.append(node.get("id"))
        
        print(f"Found compound nodes: {compound_nodes}")
        
        # Verify that modules with execution_order == -1 are NOT present
        assert "unused_embedding" not in compound_nodes, \
            "Module with execution_order=-1 should not be a compound node"
        assert "unused_param" not in compound_nodes, \
            "Module with execution_order=-1 should not be a compound node"
        
        # Verify that modules with positive execution_order ARE present
        assert any("0" in node or "Linear" in node for node in compound_nodes), \
            "First Linear module should be present"
        assert any("1" in node or "ReLU" in node for node in compound_nodes), \
            "ReLU module should be present"
        assert any("2" in node or "Linear" in node for node in compound_nodes), \
            "Second Linear module should be present"
    
    def test_graphml_has_no_execution_order_negative_one_in_attributes(self, tmp_path):
        """Test that no compound nodes have execution_order: -1 in their attributes."""
        # Create a simple ONNX model
        model = nn.Linear(5, 3)
        dummy_input = torch.randn(1, 5)
        onnx_path = tmp_path / "simple.onnx"
        
        torch.onnx.export(
            model,
            dummy_input,
            str(onnx_path),
            opset_version=17
        )
        
        # Create metadata with mixed execution orders
        # Add a wrapper module to ensure we have compound nodes
        htp_metadata = {
            "model": {"name_or_path": "simple", "total_modules": 4},
            "modules": {
                "": {
                    "class_name": "Model",
                    "scope": "",
                    "traced_tag": "/Model",
                    "execution_order": 0,
                    "children": {
                        "linear": {
                            "class_name": "Linear",
                            "scope": "linear",
                            "traced_tag": "/Model/Linear",
                            "execution_order": 1,
                            "children": {
                                "weight": {
                                    "class_name": "Parameter",
                                    "scope": "linear.weight",
                                    "traced_tag": "/Model/Linear/weight",
                                    "execution_order": -1  # Should be filtered
                                },
                                "bias": {
                                    "class_name": "Parameter",
                                    "scope": "linear.bias",
                                    "traced_tag": "/Model/Linear/bias",
                                    "execution_order": -1  # Should be filtered
                                }
                            }
                        }
                    }
                }
            }
        }
        
        # Save metadata
        metadata_path = tmp_path / "simple_htp_metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(htp_metadata, f)
        
        # Convert to GraphML
        converter = ONNXToGraphMLConverter(
            hierarchical=True,
            htp_metadata_path=str(metadata_path)
        )
        converter._enhance_with_structural_hierarchy = lambda x: x.get("", x) if isinstance(x, dict) else x
        
        result = converter.convert(
            str(onnx_path),
            str(tmp_path / "simple_output")
        )
        
        # Read GraphML content as text
        with open(result['graphml'], 'r') as f:
            graphml_content = f.read()
        
        # Check that execution_order: -1 doesn't appear anywhere
        assert '"execution_order": -1' not in graphml_content, \
            "GraphML should not contain execution_order: -1"
        assert '"execution_order":-1' not in graphml_content, \
            "GraphML should not contain execution_order: -1 (no space)"
        
        # Parse and verify structure
        tree = ET.parse(result['graphml'])
        root = tree.getroot()
        ns = {"": "http://graphml.graphdrawing.org/xmlns"}
        
        # Count compound nodes
        compound_count = 0
        for node in root.findall(".//node", ns):
            if node.find("graph", ns) is not None:
                compound_count += 1
        
        # Should have at least the linear module
        assert compound_count >= 1, "Should have at least one compound node"
        
        # But should not have weight or bias as compound nodes
        compound_ids = [node.get("id") for node in root.findall(".//node", ns) 
                       if node.find("graph", ns) is not None]
        assert not any("weight" in id for id in compound_ids), \
            f"weight parameter should not be a compound node, found: {compound_ids}"
        assert not any("bias" in id for id in compound_ids), \
            f"bias parameter should not be a compound node, found: {compound_ids}"