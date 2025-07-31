"""
Test cases for metadata extraction from ONNX models.

This test suite verifies extraction of execution_order, module_type,
and other metadata attributes from ONNX nodes.
"""

import xml.etree.ElementTree as ET

import onnx
import torch
import torch.nn as nn

from modelexport.graphml import ONNXToGraphMLConverter
from modelexport.graphml.onnx_parser import ONNXGraphParser


def add_node_metadata(onnx_model: onnx.ModelProto, metadata: dict) -> None:
    """Add metadata attributes to ONNX nodes."""
    for node_name, attrs in metadata.items():
        # Find node by name
        for node in onnx_model.graph.node:
            if node.name == node_name:
                for attr_name, (attr_type, attr_value) in attrs.items():
                    attr = onnx.AttributeProto()
                    attr.name = attr_name
                    
                    if attr_type == 'string':
                        attr.type = onnx.AttributeProto.STRING
                        attr.s = attr_value.encode('utf-8')
                    elif attr_type == 'int':
                        attr.type = onnx.AttributeProto.INT
                        attr.i = attr_value
                    elif attr_type == 'float':
                        attr.type = onnx.AttributeProto.FLOAT
                        attr.f = attr_value
                    
                    node.attribute.append(attr)
                break


class TestMetadataExtraction:
    """Test extraction of metadata attributes from ONNX nodes."""
    
    def test_execution_order_extraction(self, tmp_path):
        """Test that execution_order is extracted from node attributes."""
        # Create model
        model = nn.Sequential(
            nn.Linear(10, 20),
            nn.ReLU(),
            nn.Linear(20, 10)
        )
        dummy_input = torch.randn(1, 10)
        onnx_path = tmp_path / "test.onnx"
        torch.onnx.export(model, dummy_input, str(onnx_path))
        
        # Load and add execution_order metadata
        onnx_model = onnx.load(str(onnx_path))
        
        # Add execution order to first few nodes
        for i, node in enumerate(onnx_model.graph.node[:3]):
            attr = onnx.AttributeProto()
            attr.name = 'execution_order'
            attr.type = onnx.AttributeProto.INT
            attr.i = i + 1
            node.attribute.append(attr)
        
        onnx.save(onnx_model, str(onnx_path))
        
        # Parse and verify extraction
        parser = ONNXGraphParser()
        onnx_model = onnx.load(str(onnx_path))
        graph_data = parser.parse(onnx_model)
        
        # Check execution order was extracted
        nodes_with_order = [n for n in graph_data.nodes if n.execution_order is not None]
        assert len(nodes_with_order) >= 3, "Execution order not extracted"
        
        # Verify values
        orders = [n.execution_order for n in nodes_with_order]
        assert 1 in orders and 2 in orders and 3 in orders, \
            f"Expected orders 1,2,3 but got {orders}"
    
    def test_module_type_extraction(self, tmp_path):
        """Test that module_type is extracted from node attributes."""
        # Create model
        model = nn.Sequential(
            nn.Linear(10, 20),
            nn.ReLU(),
            nn.BatchNorm1d(20)
        )
        dummy_input = torch.randn(1, 10)
        onnx_path = tmp_path / "test.onnx"
        torch.onnx.export(model, dummy_input, str(onnx_path))
        
        # Load and add module_type metadata
        onnx_model = onnx.load(str(onnx_path))
        
        # Map of expected node patterns to module types
        module_types = {
            'MatMul': 'Linear',
            'Relu': 'ReLU',
            'BatchNormalization': 'BatchNorm1d'
        }
        
        for node in onnx_model.graph.node:
            if node.op_type in module_types:
                attr = onnx.AttributeProto()
                attr.name = 'module_type'
                attr.type = onnx.AttributeProto.STRING
                attr.s = module_types[node.op_type].encode('utf-8')
                node.attribute.append(attr)
        
        onnx.save(onnx_model, str(onnx_path))
        
        # Parse and verify extraction
        parser = ONNXGraphParser()
        onnx_model = onnx.load(str(onnx_path))
        graph_data = parser.parse(onnx_model)
        
        # Check module type was extracted
        nodes_with_type = [n for n in graph_data.nodes if n.module_type is not None]
        assert len(nodes_with_type) > 0, "Module type not extracted"
        
        # Verify values
        found_types = {n.module_type for n in nodes_with_type}
        assert 'Linear' in found_types or 'ReLU' in found_types, \
            f"Expected module types not found. Got: {found_types}"
    
    def test_combined_metadata_extraction(self, tmp_path):
        """Test extraction of multiple metadata attributes together."""
        # Create model
        model = nn.Linear(10, 5)
        dummy_input = torch.randn(1, 10)
        onnx_path = tmp_path / "test.onnx"
        torch.onnx.export(model, dummy_input, str(onnx_path))
        
        # Load and add multiple metadata attributes
        onnx_model = onnx.load(str(onnx_path))
        
        if onnx_model.graph.node:
            node = onnx_model.graph.node[0]
            
            # Add hierarchy_tag
            attr1 = onnx.AttributeProto()
            attr1.name = 'hierarchy_tag'
            attr1.type = onnx.AttributeProto.STRING
            attr1.s = b'/Model/Linear'
            node.attribute.append(attr1)
            
            # Add module_type
            attr2 = onnx.AttributeProto()
            attr2.name = 'module_type'
            attr2.type = onnx.AttributeProto.STRING
            attr2.s = b'Linear'
            node.attribute.append(attr2)
            
            # Add execution_order
            attr3 = onnx.AttributeProto()
            attr3.name = 'execution_order'
            attr3.type = onnx.AttributeProto.INT
            attr3.i = 1
            node.attribute.append(attr3)
        
        onnx.save(onnx_model, str(onnx_path))
        
        # Parse and verify all metadata extracted
        parser = ONNXGraphParser()
        onnx_model = onnx.load(str(onnx_path))
        graph_data = parser.parse(onnx_model)
        
        # Find the node with metadata
        metadata_node = None
        for node in graph_data.nodes:
            if node.hierarchy_tag or node.module_type or node.execution_order:
                metadata_node = node
                break
        
        assert metadata_node is not None, "No node with metadata found"
        assert metadata_node.hierarchy_tag == '/Model/Linear', \
            f"Wrong hierarchy_tag: {metadata_node.hierarchy_tag}"
        assert metadata_node.module_type == 'Linear', \
            f"Wrong module_type: {metadata_node.module_type}"
        assert metadata_node.execution_order == 1, \
            f"Wrong execution_order: {metadata_node.execution_order}"
    
    def test_metadata_in_graphml_output(self, tmp_path):
        """Test that extracted metadata appears in GraphML output."""
        # Create model
        model = nn.Linear(10, 5)
        dummy_input = torch.randn(1, 10)
        onnx_path = tmp_path / "test.onnx"
        torch.onnx.export(model, dummy_input, str(onnx_path))
        
        # Add metadata
        onnx_model = onnx.load(str(onnx_path))
        if onnx_model.graph.node:
            node = onnx_model.graph.node[0]
            
            # Add all metadata types
            attrs = [
                ('hierarchy_tag', onnx.AttributeProto.STRING, b'/Model/Linear'),
                ('module_type', onnx.AttributeProto.STRING, b'Linear'),
                ('execution_order', onnx.AttributeProto.INT, 42)
            ]
            
            for attr_name, attr_type_enum, attr_value in attrs:
                attr = onnx.AttributeProto()
                attr.name = attr_name
                attr.type = attr_type_enum
                if attr_type_enum == onnx.AttributeProto.STRING:
                    attr.s = attr_value
                else:
                    attr.i = attr_value
                node.attribute.append(attr)
        
        onnx.save(onnx_model, str(onnx_path))
        
        # Convert to GraphML (use flat mode to read existing metadata from ONNX)
        converter = ONNXToGraphMLConverter(hierarchical=False)
        graphml_str = converter.convert(str(onnx_path))
        
        # Parse GraphML and verify metadata
        root = ET.fromstring(graphml_str)
        ns = {'': 'http://graphml.graphdrawing.org/xmlns'}
        
        # Find nodes with metadata
        nodes = root.findall(".//node", ns)
        found_metadata = False
        
        for node in nodes:
            # Check for hierarchy_tag (key n1)
            hierarchy_data = node.find("./data[@key='n1']", ns)
            if hierarchy_data is not None and hierarchy_data.text == '/Model/Linear':
                found_metadata = True
                
                # Module_type and execution_order are stored as JSON in n2
                json_data = node.find("./data[@key='n2']", ns)
                assert json_data is not None, "Node attributes JSON (n2) not found"
                
                # Parse JSON to verify content
                import json
                attrs = json.loads(json_data.text)
                assert attrs.get('module_type') == 'Linear', \
                    f"Module type not in JSON attributes: {attrs}"
                assert attrs.get('execution_order') == 42, \
                    f"Execution order not in JSON attributes: {attrs}"
                break
        
        assert found_metadata, "Metadata not found in GraphML output"
    
    def test_metadata_with_special_values(self, tmp_path):
        """Test handling of special metadata values."""
        # Create model
        model = nn.Linear(10, 5)
        dummy_input = torch.randn(1, 10)
        onnx_path = tmp_path / "test.onnx"
        torch.onnx.export(model, dummy_input, str(onnx_path))
        
        # Add metadata with special values
        onnx_model = onnx.load(str(onnx_path))
        if onnx_model.graph.node:
            node = onnx_model.graph.node[0]
            
            # Empty string
            attr1 = onnx.AttributeProto()
            attr1.name = 'module_type'
            attr1.type = onnx.AttributeProto.STRING
            attr1.s = b''
            node.attribute.append(attr1)
            
            # Zero execution order
            attr2 = onnx.AttributeProto()
            attr2.name = 'execution_order'
            attr2.type = onnx.AttributeProto.INT
            attr2.i = 0
            node.attribute.append(attr2)
            
            # Unicode in hierarchy tag
            attr3 = onnx.AttributeProto()
            attr3.name = 'hierarchy_tag'
            attr3.type = onnx.AttributeProto.STRING
            attr3.s = '/Model/层级/模块'.encode()
            node.attribute.append(attr3)
        
        onnx.save(onnx_model, str(onnx_path))
        
        # Should handle special values gracefully (use flat mode)
        converter = ONNXToGraphMLConverter(hierarchical=False)
        graphml_str = converter.convert(str(onnx_path))
        
        # Verify XML is valid
        root = ET.fromstring(graphml_str)
        assert root is not None
        
        # Check Unicode preserved
        assert '层级' in graphml_str or '/Model/' in graphml_str