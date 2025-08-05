#!/usr/bin/env python3
"""Test custom attribute handling in GraphML conversion."""

import json
import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path

import onnx
import pytest

from modelexport.graphml.graphml_to_onnx_converter import GraphMLToONNXConverter
from modelexport.graphml.onnx_to_graphml_converter import ONNXToGraphMLConverter


@pytest.mark.graphml
@pytest.mark.sanity
class TestCustomAttributeHandling:
    """Test that custom GraphML attributes are properly handled."""
    
    def test_custom_attributes_preserved_in_graphml(self, sample_htp_data):
        """Test that custom attributes are preserved in GraphML but not passed to ONNX."""
        htp_data, sample_onnx_path = sample_htp_data
        
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            
            # Create metadata file FIRST
            metadata_path = temp_path / "metadata.json"
            with open(metadata_path, 'w') as f:
                json.dump(htp_data, f)
            
            # Convert ONNX to GraphML with custom attributes
            converter = ONNXToGraphMLConverter(
                htp_metadata_path=str(metadata_path)
            )
            
            # Convert to GraphML
            result = converter.convert(str(sample_onnx_path), str(temp_path / "test"))
            graphml_path = result["graphml"]
            
            # Parse GraphML and verify custom attributes are present
            tree = ET.parse(graphml_path)
            root = tree.getroot()
            
            # Find nodes with hierarchy_tag attributes
            namespaces = {'gml': 'http://graphml.graphdrawing.org/xmlns'}
            nodes_with_hierarchy = []
            
            for node in root.findall('.//gml:node', namespaces):
                # Look for hierarchy_tag data
                hierarchy_elem = node.find('./gml:data[@key="n1"]', namespaces)
                if hierarchy_elem is None:
                    hierarchy_elem = node.find('./data[@key="n1"]')
                
                if hierarchy_elem is not None and hierarchy_elem.text:
                    nodes_with_hierarchy.append({
                        'id': node.get('id'),
                        'hierarchy_tag': hierarchy_elem.text
                    })
            
            # Verify we found nodes with custom hierarchy attributes
            assert len(nodes_with_hierarchy) > 0, "No nodes with hierarchy_tag found in GraphML"
            
            # Check for specific expected hierarchy tags
            hierarchy_tags = [node['hierarchy_tag'] for node in nodes_with_hierarchy]
            
            # Should have module-level hierarchy tags
            module_tags = [tag for tag in hierarchy_tags if tag.startswith('/BertModel/')]
            assert len(module_tags) > 0, f"No module hierarchy tags found. Got: {hierarchy_tags[:5]}"
            
            print(f"✅ Found {len(nodes_with_hierarchy)} nodes with hierarchy_tag custom attributes")
            print(f"✅ Sample hierarchy tags: {hierarchy_tags[:3]}")
    
    def test_custom_attributes_filtered_from_onnx_conversion(self, sample_htp_data):
        """Test that custom attributes are filtered out during GraphML→ONNX conversion."""
        htp_data, sample_onnx_path = sample_htp_data
        
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            
            # Create metadata file FIRST
            metadata_path = temp_path / "metadata.json"
            with open(metadata_path, 'w') as f:
                json.dump(htp_data, f)
            
            # Step 1: Convert ONNX to GraphML (includes custom attributes)
            onnx_to_graphml = ONNXToGraphMLConverter(
                htp_metadata_path=str(metadata_path)
            )
            
            # Convert to GraphML
            result = onnx_to_graphml.convert(str(sample_onnx_path), str(temp_path / "test"))
            graphml_path = result["graphml"]
            
            # Step 2: Convert GraphML back to ONNX (should filter custom attributes)
            graphml_to_onnx = GraphMLToONNXConverter()
            reconstructed_onnx_path = temp_path / "reconstructed.onnx"
            
            # Skip validation since this test is about custom attributes, not model validity
            graphml_to_onnx.convert(graphml_path, str(reconstructed_onnx_path), validate=False)
            
            # Step 3: Load reconstructed ONNX and verify no custom attributes
            reconstructed_model = onnx.load(str(reconstructed_onnx_path))
            
            # Check that no nodes have custom attributes in the reconstructed ONNX
            custom_attr_names = {
                "hierarchy_tag", "module_type", "execution_order", 
                "scope", "traced_tag", "class_name"
            }
            
            nodes_with_custom_attrs = []
            for node in reconstructed_model.graph.node:
                for attr in node.attribute:
                    if attr.name in custom_attr_names:
                        nodes_with_custom_attrs.append({
                            'node_name': node.name,
                            'attr_name': attr.name
                        })
            
            # Verify no custom attributes made it to ONNX
            assert len(nodes_with_custom_attrs) == 0, \
                f"Custom attributes found in reconstructed ONNX: {nodes_with_custom_attrs}"
            
            print("✅ Custom attributes correctly filtered out during GraphML→ONNX conversion")
    
    def test_valid_onnx_attributes_preserved(self, sample_htp_data):
        """Test that valid ONNX attributes are preserved during round-trip conversion."""
        htp_data, sample_onnx_path = sample_htp_data
        
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            
            # Load original ONNX to check for valid attributes
            original_model = onnx.load(str(sample_onnx_path))
            
            # Find nodes with valid ONNX attributes
            original_attrs = {}
            for node in original_model.graph.node:
                if node.attribute:
                    original_attrs[node.name] = {attr.name: attr for attr in node.attribute}
            
            if not original_attrs:
                pytest.skip("No ONNX attributes found in sample model to test")
            
            # Create metadata file FIRST
            metadata_path = temp_path / "metadata.json"
            with open(metadata_path, 'w') as f:
                json.dump(htp_data, f)
            
            # Step 1: ONNX → GraphML
            onnx_to_graphml = ONNXToGraphMLConverter(
                htp_metadata_path=str(metadata_path)
            )
            
            result = onnx_to_graphml.convert(str(sample_onnx_path), str(temp_path / "test"))
            graphml_path = result["graphml"]
            
            # Step 2: GraphML → ONNX
            graphml_to_onnx = GraphMLToONNXConverter()
            reconstructed_onnx_path = temp_path / "reconstructed.onnx"
            
            # Skip validation since this test is about preserving attributes
            graphml_to_onnx.convert(graphml_path, str(reconstructed_onnx_path), validate=False)
            
            # Step 3: Compare valid ONNX attributes
            reconstructed_model = onnx.load(str(reconstructed_onnx_path))
            
            reconstructed_attrs = {}
            for node in reconstructed_model.graph.node:
                if node.attribute:
                    reconstructed_attrs[node.name] = {attr.name: attr for attr in node.attribute}
            
            # Check that valid ONNX attributes are preserved
            preserved_count = 0
            for node_name, attrs in original_attrs.items():
                if node_name in reconstructed_attrs:
                    for attr_name in attrs:
                        if attr_name in reconstructed_attrs[node_name]:
                            preserved_count += 1
            
            print(f"✅ {preserved_count} valid ONNX attributes preserved during round-trip")
            
            # Should preserve at least some valid attributes
            if len(original_attrs) > 0:
                assert preserved_count > 0, "No valid ONNX attributes were preserved"

    def test_custom_attribute_filtering_logic(self):
        """Test the _is_custom_attribute method directly."""
        converter = GraphMLToONNXConverter()
        
        # Test custom GraphML metadata attributes (should be filtered out)
        custom_attrs = [
            "hierarchy_tag", "module_type", "execution_order", 
            "scope", "traced_tag", "class_name"
        ]
        
        for attr in custom_attrs:
            is_custom = converter._is_custom_attribute(attr)
            assert is_custom, f"Custom attribute '{attr}' should be recognized as custom"
        
        # Test valid ONNX attributes (should NOT be marked as custom)
        valid_onnx_attrs = [
            "kernel_shape",
            "strides", 
            "pads",
            "dilations",
            "group",
        ]
        
        for attr_name in valid_onnx_attrs:
            is_custom = converter._is_custom_attribute(attr_name)
            assert not is_custom, f"Valid ONNX attribute '{attr_name}' should not be marked as custom"
        
        # Test custom/vendor-specific attributes (should NOT be marked as custom - we preserve them)
        # This is the key change - we now preserve ALL attributes that aren't GraphML metadata
        custom_vendor_attrs = [
            "custom_attr",
            "vendor_specific_attr",
            "proprietary_param",
        ]
        
        for attr_name in custom_vendor_attrs:
            is_custom = converter._is_custom_attribute(attr_name)
            assert not is_custom, f"Vendor attribute '{attr_name}' should be preserved (not marked as custom)"
        
        print("✅ Custom attribute filtering logic works correctly")


@pytest.fixture
def sample_htp_data():
    """Provide sample HTP metadata and ONNX model for testing."""
    import tempfile

    from click.testing import CliRunner

    from modelexport.cli import cli
    
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        onnx_path = temp_path / "test_model.onnx"
        
        # Export a simple model with HTP metadata
        runner = CliRunner()
        result = runner.invoke(cli, [
            'export',
            '--model', 'prajjwal1/bert-tiny',
            '--output', str(onnx_path)
        ])
        
        assert result.exit_code == 0, f"Model export failed: {result.output}"
        
        # Load the HTP metadata
        metadata_path = temp_path / f"{onnx_path.stem}_htp_metadata.json"
        assert metadata_path.exists(), "HTP metadata file not created"
        
        with open(metadata_path) as f:
            htp_data = json.load(f)
        
        yield htp_data, str(onnx_path)