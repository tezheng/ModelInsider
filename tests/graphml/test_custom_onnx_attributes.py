#!/usr/bin/env python3
"""
Test custom ONNX attribute preservation in GraphML round-trip conversion.

This test ensures that:
1. ALL original ONNX attributes (standard and custom) are preserved
2. GraphML-specific metadata doesn't leak into reconstructed ONNX
3. No attributes appear from nowhere
"""

import pytest
import tempfile
import json
from pathlib import Path
import onnx
from onnx import helper, TensorProto, AttributeProto
import numpy as np

from modelexport.graphml.onnx_to_graphml_converter import ONNXToGraphMLConverter
from modelexport.graphml.graphml_to_onnx_converter import GraphMLToONNXConverter


@pytest.mark.graphml
class TestCustomONNXAttributes:
    """Test that custom ONNX attributes are properly preserved during round-trip."""
    
    @pytest.fixture
    def onnx_with_custom_attrs(self):
        """Create an ONNX model with both standard and custom attributes."""
        # Create a simple model with custom attributes
        X = helper.make_tensor_value_info('X', TensorProto.FLOAT, [1, 3, 224, 224])
        Y = helper.make_tensor_value_info('Y', TensorProto.FLOAT, [1, 10])
        
        # Create nodes without initializers to test pure attribute preservation
        # Cast node with standard and custom attributes
        cast_node = helper.make_node(
            'Cast',
            inputs=['X'],
            outputs=['cast_out'],
            name='MyCast',
            # Standard ONNX attribute
            to=TensorProto.FLOAT,
            # Custom attributes (not in ONNX spec)
            custom_attr_int=42,
            custom_attr_string="my_custom_value",
            custom_attr_float=3.14,
            vendor_specific_flag=True
        )
        
        # Custom domain operator
        custom_op = helper.make_node(
            'CustomOp',
            inputs=['cast_out'],
            outputs=['custom_out'],
            name='MyCustomOp',
            domain='com.example',
            # All attributes for custom ops should be preserved
            proprietary_param="secret_sauce",
            magic_number=12345,
            enable_optimization=False
        )
        
        # Standard ReLU
        relu_node = helper.make_node(
            'Relu',
            inputs=['custom_out'],
            outputs=['Y'],
            name='MyRelu'
            # ReLU has no attributes
        )
        
        graph = helper.make_graph(
            [cast_node, custom_op, relu_node],
            'test_graph',
            [X],
            [Y],
            []  # No initializers
        )
        
        model = helper.make_model(graph, producer_name='test_producer')
        model.opset_import[0].version = 17
        
        # Add custom domain
        domain = model.opset_import.add()
        domain.domain = 'com.example'
        domain.version = 1
        
        return model
    
    def test_standard_attrs_preserved(self, onnx_with_custom_attrs):
        """Test that standard ONNX attributes are preserved exactly."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            
            # Save original model
            original_path = temp_path / "original.onnx"
            onnx.save(onnx_with_custom_attrs, str(original_path))
            
            # Convert ONNX → GraphML
            converter = ONNXToGraphMLConverter(hierarchical=False)
            graphml_str = converter.convert(str(original_path))
            graphml_path = temp_path / "model.graphml"
            with open(graphml_path, 'w') as f:
                f.write(graphml_str)
            
            # Convert GraphML → ONNX
            graphml_converter = GraphMLToONNXConverter()
            reconstructed_path = temp_path / "reconstructed.onnx"
            graphml_converter.convert(str(graphml_path), str(reconstructed_path))
            
            # Load and compare
            reconstructed = onnx.load(str(reconstructed_path))
            
            # Find Cast node in both models
            orig_cast = next(n for n in onnx_with_custom_attrs.graph.node if n.op_type == 'Cast')
            recon_cast = next(n for n in reconstructed.graph.node if n.op_type == 'Cast')
            
            # Check standard attributes
            standard_attrs = ['to']
            for attr_name in standard_attrs:
                orig_attr = next(a for a in orig_cast.attribute if a.name == attr_name)
                recon_attr = next(a for a in recon_cast.attribute if a.name == attr_name)
                
                assert orig_attr.type == recon_attr.type, f"Type mismatch for {attr_name}"
                
                if orig_attr.type == AttributeProto.INTS:
                    assert list(orig_attr.ints) == list(recon_attr.ints), f"Value mismatch for {attr_name}"
    
    def test_custom_attrs_preserved(self, onnx_with_custom_attrs):
        """Test that custom/vendor-specific attributes are preserved."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            
            # Save original model
            original_path = temp_path / "original.onnx"
            onnx.save(onnx_with_custom_attrs, str(original_path))
            
            # Convert ONNX → GraphML → ONNX
            converter = ONNXToGraphMLConverter(hierarchical=False)
            graphml_str = converter.convert(str(original_path))
            graphml_path = temp_path / "model.graphml"
            with open(graphml_path, 'w') as f:
                f.write(graphml_str)
            
            graphml_converter = GraphMLToONNXConverter()
            reconstructed_path = temp_path / "reconstructed.onnx"
            graphml_converter.convert(str(graphml_path), str(reconstructed_path))
            
            # Load reconstructed
            reconstructed = onnx.load(str(reconstructed_path))
            
            # Check Cast node custom attributes
            orig_cast = next(n for n in onnx_with_custom_attrs.graph.node if n.op_type == 'Cast')
            recon_cast = next(n for n in reconstructed.graph.node if n.op_type == 'Cast')
            
            custom_attrs = ['custom_attr_int', 'custom_attr_string', 'custom_attr_float', 'vendor_specific_flag']
            
            orig_attr_names = {a.name for a in orig_cast.attribute}
            recon_attr_names = {a.name for a in recon_cast.attribute}
            
            # All custom attributes should be preserved
            for attr_name in custom_attrs:
                assert attr_name in orig_attr_names, f"Test setup error: {attr_name} not in original"
                assert attr_name in recon_attr_names, f"Custom attribute {attr_name} was lost"
            
            # Check custom domain operator
            orig_custom = next(n for n in onnx_with_custom_attrs.graph.node if n.op_type == 'CustomOp')
            recon_custom = next(n for n in reconstructed.graph.node if n.op_type == 'CustomOp')
            
            # All attributes of custom ops should be preserved
            assert len(orig_custom.attribute) == len(recon_custom.attribute), \
                "Custom op lost attributes"
            
            for orig_attr in orig_custom.attribute:
                recon_attr = next((a for a in recon_custom.attribute if a.name == orig_attr.name), None)
                assert recon_attr is not None, f"Custom op attribute {orig_attr.name} was lost"
    
    def test_no_graphml_metadata_leaks(self, onnx_with_custom_attrs):
        """Test that GraphML-specific metadata doesn't appear in reconstructed ONNX."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            
            # Save original model
            original_path = temp_path / "original.onnx"
            onnx.save(onnx_with_custom_attrs, str(original_path))
            
            # Create HTP metadata to add hierarchy info
            htp_metadata = {
                "modules": {
                    "class_name": "TestModel",
                    "traced_tag": "/TestModel"
                }
            }
            metadata_path = temp_path / "metadata.json"
            with open(metadata_path, 'w') as f:
                json.dump(htp_metadata, f)
            
            # Convert with hierarchical mode to add metadata
            converter = ONNXToGraphMLConverter(
                hierarchical=True,
                htp_metadata_path=str(metadata_path)
            )
            result = converter.convert(str(original_path), str(temp_path / "test"))
            graphml_path = result["graphml"]
            
            # Convert back
            graphml_converter = GraphMLToONNXConverter()
            reconstructed_path = temp_path / "reconstructed.onnx"
            graphml_converter.convert(graphml_path, str(reconstructed_path))
            
            # Load reconstructed
            reconstructed = onnx.load(str(reconstructed_path))
            
            # GraphML metadata that should NOT appear in ONNX
            forbidden_attrs = {
                'hierarchy_tag', 'module_type', 'execution_order',
                'scope', 'traced_tag', 'class_name'
            }
            
            # Check all nodes
            for node in reconstructed.graph.node:
                attr_names = {a.name for a in node.attribute}
                leaked = attr_names & forbidden_attrs
                assert not leaked, f"GraphML metadata leaked into ONNX node {node.name}: {leaked}"
    
    def test_no_phantom_attributes(self, onnx_with_custom_attrs):
        """Test that no attributes appear from nowhere."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            
            # Save original model
            original_path = temp_path / "original.onnx"
            onnx.save(onnx_with_custom_attrs, str(original_path))
            
            # Round-trip conversion
            converter = ONNXToGraphMLConverter(hierarchical=False)
            graphml_str = converter.convert(str(original_path))
            graphml_path = temp_path / "model.graphml"
            with open(graphml_path, 'w') as f:
                f.write(graphml_str)
            
            graphml_converter = GraphMLToONNXConverter()
            reconstructed_path = temp_path / "reconstructed.onnx"
            graphml_converter.convert(str(graphml_path), str(reconstructed_path))
            
            # Load both models
            reconstructed = onnx.load(str(reconstructed_path))
            
            # Map nodes by name for comparison
            orig_nodes = {n.name: n for n in onnx_with_custom_attrs.graph.node}
            recon_nodes = {n.name: n for n in reconstructed.graph.node}
            
            # Check each reconstructed node
            for name, recon_node in recon_nodes.items():
                orig_node = orig_nodes.get(name)
                assert orig_node is not None, f"Phantom node appeared: {name}"
                
                # Get attribute sets
                orig_attrs = {a.name for a in orig_node.attribute}
                recon_attrs = {a.name for a in recon_node.attribute}
                
                # No phantom attributes
                phantom = recon_attrs - orig_attrs
                assert not phantom, f"Phantom attributes in {name}: {phantom}"
    
    def test_attribute_types_preserved(self, onnx_with_custom_attrs):
        """Test that attribute types and values are preserved exactly."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            
            # Save and round-trip
            original_path = temp_path / "original.onnx"
            onnx.save(onnx_with_custom_attrs, str(original_path))
            
            converter = ONNXToGraphMLConverter(hierarchical=False)
            graphml_str = converter.convert(str(original_path))
            graphml_path = temp_path / "model.graphml"
            with open(graphml_path, 'w') as f:
                f.write(graphml_str)
            
            graphml_converter = GraphMLToONNXConverter()
            reconstructed_path = temp_path / "reconstructed.onnx"
            graphml_converter.convert(str(graphml_path), str(reconstructed_path))
            
            reconstructed = onnx.load(str(reconstructed_path))
            
            # Check Cast node attributes in detail
            orig_cast = next(n for n in onnx_with_custom_attrs.graph.node if n.op_type == 'Cast')
            recon_cast = next(n for n in reconstructed.graph.node if n.op_type == 'Cast')
            
            # Build attribute maps
            orig_attrs = {a.name: a for a in orig_cast.attribute}
            recon_attrs = {a.name: a for a in recon_cast.attribute}
            
            # Check each attribute
            for name, orig_attr in orig_attrs.items():
                recon_attr = recon_attrs.get(name)
                assert recon_attr is not None, f"Attribute {name} missing"
                
                # Type must match
                assert orig_attr.type == recon_attr.type, \
                    f"Type mismatch for {name}: {orig_attr.type} vs {recon_attr.type}"
                
                # Value must match based on type
                if orig_attr.type == AttributeProto.INT:
                    assert orig_attr.i == recon_attr.i
                elif orig_attr.type == AttributeProto.FLOAT:
                    assert abs(orig_attr.f - recon_attr.f) < 1e-6
                elif orig_attr.type == AttributeProto.STRING:
                    assert orig_attr.s == recon_attr.s
                elif orig_attr.type == AttributeProto.INTS:
                    assert list(orig_attr.ints) == list(recon_attr.ints)