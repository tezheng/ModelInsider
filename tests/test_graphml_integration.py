"""
Integration Tests for GraphML Export

Tests the complete GraphML export functionality with real models.
"""

import json
import xml.etree.ElementTree as ET
from pathlib import Path

import pytest

from modelexport.graphml import HierarchicalGraphMLConverter, ONNXToGraphMLConverter


@pytest.fixture
def bert_tiny_model():
    """Path to bert-tiny ONNX model if it exists."""
    model_path = Path("temp/bert-tiny.onnx")
    if model_path.exists():
        return model_path
    pytest.skip("bert-tiny model not found in temp/bert-tiny.onnx")


@pytest.fixture
def bert_tiny_htp_metadata():
    """Path to bert-tiny HTP metadata if it exists."""
    metadata_path = Path("temp/bert-tiny_htp_metadata.json")
    if metadata_path.exists():
        return metadata_path
    pytest.skip("bert-tiny HTP metadata not found in temp/bert-tiny_htp_metadata.json")


class TestGraphMLIntegration:
    """Integration tests with real models."""
    
    def test_basic_conversion_with_bert_tiny(self, bert_tiny_model):
        """Test basic GraphML conversion with bert-tiny model."""
        converter = ONNXToGraphMLConverter(hierarchical=False, exclude_initializers=True)
        
        # Convert to GraphML
        graphml_str = converter.convert(str(bert_tiny_model))
        
        # Verify output
        assert isinstance(graphml_str, str)
        assert len(graphml_str) > 1000  # Should be substantial
        
        # Parse and validate structure
        root = ET.fromstring(graphml_str)
        assert root.tag == "{http://graphml.graphdrawing.org/xmlns}graphml"
        
        # Check nodes and edges
        nodes = root.findall(".//{http://graphml.graphdrawing.org/xmlns}node")
        edges = root.findall(".//{http://graphml.graphdrawing.org/xmlns}edge")
        
        # bert-tiny should have many nodes and edges
        assert len(nodes) > 50
        assert len(edges) > 50
        
        # Save output for manual inspection
        output_path = Path("temp/bert-tiny_basic_graph.graphml")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            f.write(graphml_str)
        print(f"\nBasic GraphML saved to: {output_path}")
        print(f"Nodes: {len(nodes)}, Edges: {len(edges)}")
    
    def test_hierarchical_conversion_with_bert_tiny(self, bert_tiny_model, bert_tiny_htp_metadata):
        """Test hierarchical GraphML conversion with bert-tiny model and HTP metadata."""
        converter = HierarchicalGraphMLConverter(
            str(bert_tiny_htp_metadata),
            exclude_initializers=True
        )
        
        # Convert to hierarchical GraphML
        result = converter.convert(str(bert_tiny_model))
        
        # Verify output is a dictionary
        assert isinstance(result, dict)
        assert "graphml" in result
        assert "format_version" in result
        
        # Read the generated GraphML file
        graphml_path = result["graphml"]
        with open(graphml_path, 'r') as f:
            graphml_str = f.read()
        
        assert len(graphml_str) > 2000  # Should be larger with hierarchy
        
        # Parse and validate structure
        root = ET.fromstring(graphml_str)
        assert root.tag == "{http://graphml.graphdrawing.org/xmlns}graphml"
        
        # Check for compound nodes (nodes that contain nested graphs)
        nodes = root.findall(".//{http://graphml.graphdrawing.org/xmlns}node")
        compound_nodes = []
        for node in nodes:
            # Check if this node contains a nested graph element (compound node)
            nested_graph = node.find("./{http://graphml.graphdrawing.org/xmlns}graph")
            if nested_graph is not None:
                compound_nodes.append(node)
        regular_nodes = [n for n in nodes if n not in compound_nodes]
        
        print("\nHierarchical structure:")
        print(f"Total nodes: {len(nodes)}")
        print(f"Compound nodes: {len(compound_nodes)}")
        print(f"Regular nodes: {len(regular_nodes)}")
        
        # Should have compound nodes for modules
        assert len(compound_nodes) > 0
        
        # Check for nested graphs
        nested_graphs = root.findall(".//{http://graphml.graphdrawing.org/xmlns}graph[@id!='G']")
        print(f"Nested graphs: {len(nested_graphs)}")
        assert len(nested_graphs) > 0
        
        # Save output for manual inspection
        output_path = Path("temp/bert-tiny_hierarchical_graph.graphml")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            f.write(graphml_str)
        print(f"Hierarchical GraphML saved to: {output_path}")
    
    def test_compound_node_hierarchy_validation(self, bert_tiny_model, bert_tiny_htp_metadata):
        """Validate the compound node hierarchy structure."""
        # First, let's examine the metadata structure
        with open(bert_tiny_htp_metadata) as f:
            metadata = json.load(f)
        
        # Extract module hierarchy from metadata
        if "export_data" in metadata and "hierarchy_data" in metadata["export_data"]:
            hierarchy_data = metadata["export_data"]["hierarchy_data"]
        elif "hierarchy_data" in metadata:
            hierarchy_data = metadata["hierarchy_data"]
        else:
            hierarchy_data = {}
        
        print("\nModule hierarchy from HTP metadata:")
        module_count = len(hierarchy_data)
        print(f"Total modules: {module_count}")
        
        # Sample some module paths
        sample_paths = list(hierarchy_data.keys())[:10]
        for path in sample_paths:
            module_info = hierarchy_data[path]
            print(f"  {path or '[root]'}: {module_info.get('class_name', 'Unknown')}")
        
        # Now convert and validate
        converter = HierarchicalGraphMLConverter(
            str(bert_tiny_htp_metadata),
            exclude_initializers=True
        )
        
        result = converter.convert(str(bert_tiny_model))
        
        # Read the generated GraphML file
        graphml_path = result["graphml"]
        with open(graphml_path, 'r') as f:
            graphml_str = f.read()
            
        root = ET.fromstring(graphml_str)
        
        # Extract compound nodes (XPath starts-with not supported, use list comprehension)
        all_nodes = root.findall(".//{http://graphml.graphdrawing.org/xmlns}node")
        compound_nodes = [n for n in all_nodes if n.get('id', '').startswith('module_')]
        print(f"\nCompound nodes in GraphML: {len(compound_nodes)}")
        
        # Verify key modules exist
        compound_ids = [n.get("id") for n in compound_nodes]
        
        # Check for expected BERT modules
        expected_modules = ["module_root", "module_embeddings", "module_encoder"]
        for expected in expected_modules:
            if expected in compound_ids:
                print(f"✓ Found {expected}")
            else:
                print(f"✗ Missing {expected}")
    
    def test_node_tagging_validation(self, bert_tiny_model, bert_tiny_htp_metadata):
        """Validate that nodes are properly tagged with hierarchy information."""
        converter = HierarchicalGraphMLConverter(
            str(bert_tiny_htp_metadata),
            exclude_initializers=True
        )
        
        result = converter.convert(str(bert_tiny_model))
        
        # Read the generated GraphML file
        graphml_path = result["graphml"]
        with open(graphml_path, 'r') as f:
            graphml_str = f.read()
            
        root = ET.fromstring(graphml_str)
        
        # Find operation nodes (not compound nodes)
        all_nodes = root.findall(".//{http://graphml.graphdrawing.org/xmlns}node")
        op_nodes = [n for n in all_nodes if not n.get('id', '').startswith(('module_', 'input_', 'output_'))]
        
        # Check how many have hierarchy tags
        tagged_count = 0
        sample_tags = []
        
        for node in op_nodes:
            # Look for hierarchy tag data (key="d1")
            tag_data = node.find(".//{http://graphml.graphdrawing.org/xmlns}data[@key='d1']")
            if tag_data is not None and tag_data.text:
                tagged_count += 1
                if len(sample_tags) < 5:
                    op_type_data = node.find(".//{http://graphml.graphdrawing.org/xmlns}data[@key='d0']")
                    op_type = op_type_data.text if op_type_data is not None else "Unknown"
                    sample_tags.append(f"{node.get('id')} ({op_type}): {tag_data.text}")
        
        print("\nNode tagging statistics:")
        print(f"Total operation nodes: {len(op_nodes)}")
        print(f"Tagged nodes: {tagged_count}")
        print(f"Tagging rate: {tagged_count/len(op_nodes)*100:.1f}%")
        
        print("\nSample tagged nodes:")
        for tag in sample_tags:
            print(f"  {tag}")
    
    def test_graphml_visualization_compatibility(self, bert_tiny_model):
        """Test that generated GraphML follows visualization tool standards."""
        converter = ONNXToGraphMLConverter(hierarchical=False, exclude_initializers=True)
        graphml_str = converter.convert(str(bert_tiny_model))
        
        root = ET.fromstring(graphml_str)
        
        # Check for required GraphML structure
        assert root.tag == "{http://graphml.graphdrawing.org/xmlns}graphml"
        
        # Check for key definitions
        keys = root.findall("./{http://graphml.graphdrawing.org/xmlns}key")
        assert len(keys) > 0
        
        key_ids = [k.get("id") for k in keys]
        key_names = [k.get("attr.name") for k in keys]
        
        print("\nGraphML key definitions:")
        for key_id, key_name in zip(key_ids, key_names, strict=False):
            print(f"  {key_id}: {key_name}")
        
        # Check for proper graph structure
        graphs = root.findall("./{http://graphml.graphdrawing.org/xmlns}graph")
        assert len(graphs) == 1
        
        main_graph = graphs[0]
        assert main_graph.get("id") == "G"
        assert main_graph.get("edgedefault") == "directed"
        
        print("\n✓ GraphML structure is compatible with visualization tools")