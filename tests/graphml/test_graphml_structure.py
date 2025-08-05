"""
Test cases for GraphML v1.3 structural compliance.

This test suite verifies GraphML output structure matches v1.3 format
with proper key definitions, nested graphs, and metadata.
"""

import json
import xml.etree.ElementTree as ET

import onnx
import pytest
import torch
import torch.nn as nn

from modelexport.graphml import ONNXToGraphMLConverter
from modelexport.version import GRAPHML_VERSION
from modelexport.graphml.constants import (
    GRAPHML_V13_KEYS,
    GRAPHML_FORMAT_VERSION,
    GRAPHML_V13_KEY_MAPPINGS
)


class TestGraphMLStructure:
    """Test GraphML structure and format compliance."""
    
    def test_key_definitions_match_baseline(self, tmp_path):
        """Test that GraphML key definitions match baseline format."""
        # Create simple model
        model = nn.Linear(10, 5)
        dummy_input = torch.randn(1, 10)
        onnx_path = tmp_path / "test.onnx"
        torch.onnx.export(model, dummy_input, str(onnx_path))
        
        # Convert to GraphML
        converter = ONNXToGraphMLConverter(hierarchical=False)
        graphml_str = converter.convert(str(onnx_path))
        root = ET.fromstring(graphml_str)
        
        # Check key definitions
        ns = {'': 'http://graphml.graphdrawing.org/xmlns'}
        keys = root.findall(".//key", ns)
        
        # Build key map
        key_map = {}
        for key in keys:
            key_id = key.get('id')
            for_type = key.get('for')
            attr_name = key.get('attr.name')
            key_map[key_id] = (for_type, attr_name)
        
        # Verify graph keys (for compound nodes)
        assert 'd0' in key_map, "Missing graph class_name key (d0)"
        assert key_map['d0'] == ('graph', 'class_name'), "Wrong definition for d0"
        
        assert 'd1' in key_map, "Missing graph module_type key (d1)"
        assert key_map['d1'] == ('graph', 'module_type'), "Wrong definition for d1"
        
        assert 'd2' in key_map, "Missing graph execution_order key (d2)"
        assert key_map['d2'] == ('graph', 'execution_order'), "Wrong definition for d2"
        
        assert 'd3' in key_map, "Missing graph traced_tag key (d3)"
        assert key_map['d3'] == ('graph', 'traced_tag'), "Wrong definition for d3"
        
        # Verify node keys
        assert 'n0' in key_map, "Missing node op_type key (n0)"
        assert key_map['n0'] == ('node', 'op_type'), "Wrong definition for n0"
        
        assert 'n1' in key_map, "Missing node hierarchy_tag key (n1)"
        assert key_map['n1'] == ('node', 'hierarchy_tag'), "Wrong definition for n1"
        
        assert 'n2' in key_map, "Missing node onnx_attributes key (n2)"
        assert key_map['n2'] == ('node', 'onnx_attributes'), "Wrong definition for n2"
        
        assert 'n3' in key_map, "Missing node name key (n3)"
        assert key_map['n3'] == ('node', 'name'), "Wrong definition for n3"
        
        # Verify edge keys
        assert 'e0' in key_map, "Missing edge tensor_name key (e0)"
        assert key_map['e0'] == ('edge', 'tensor_name'), "Wrong definition for e0"
        
        # Verify metadata keys
        assert 'meta0' in key_map, "Missing graph source_onnx_file key (meta0)"
        assert key_map['meta0'] == ('graph', 'source_onnx_file'), "Wrong definition for meta0"
        
        assert 'meta2' in key_map, "Missing graph format_version key (meta2)"
        assert key_map['meta2'] == ('graph', 'format_version'), "Wrong definition for meta2"
        
        assert 'meta3' in key_map, "Missing graph export_timestamp key (meta3)"
        assert key_map['meta3'] == ('graph', 'export_timestamp'), "Wrong definition for meta3"
    
    def test_node_name_attribute(self, tmp_path):
        """Test that nodes have a name attribute in addition to id."""
        # Create model
        model = nn.Sequential(
            nn.Linear(10, 20),
            nn.ReLU(),
            nn.Linear(20, 5)
        )
        dummy_input = torch.randn(1, 10)
        onnx_path = tmp_path / "test.onnx"
        torch.onnx.export(model, dummy_input, str(onnx_path))
        
        # Convert to GraphML
        converter = ONNXToGraphMLConverter(hierarchical=False)
        graphml_str = converter.convert(str(onnx_path))
        root = ET.fromstring(graphml_str)
        
        # Check nodes have both id and name
        ns = {'': 'http://graphml.graphdrawing.org/xmlns'}
        nodes = root.findall(".//node", ns)
        
        for node in nodes[:3]:  # Check first few nodes
            node_id = node.get('id')
            assert node_id is not None, "Node missing id attribute"
            
            # Should have a data element with node name (key 'n3')
            name_data = node.find("./data[@key='n3']", ns)
            assert name_data is not None, f"Node {node_id} missing name data (key n3)"
            
            # Should also have op_type (key 'n0')
            op_type_data = node.find("./data[@key='n0']", ns)
            assert op_type_data is not None, f"Node {node_id} missing op_type data (key n0)"
    
    def test_nested_graph_support(self, tmp_path):
        """Test support for nested graphs (compound nodes)."""
        # This tests hierarchical converter
        # Create model with hierarchy
        class NestedModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.encoder = nn.Sequential(
                    nn.Linear(10, 20),
                    nn.ReLU()
                )
                self.decoder = nn.Linear(20, 10)
            
            def forward(self, x):
                x = self.encoder(x)
                return self.decoder(x)
        
        model = NestedModel()
        dummy_input = torch.randn(1, 10)
        onnx_path = tmp_path / "nested.onnx"
        torch.onnx.export(model, dummy_input, str(onnx_path))
        
        # Create mock HTP metadata with hierarchical structure
        htp_metadata = {
            "model": {
                "class_name": "NestedModel"
            },
            "modules": {
                "scope": "/NestedModel",
                "class_name": "NestedModel", 
                "execution_order": 0,
                "traced_tag": "/NestedModel",
                "children": {
                    "encoder": {
                        "scope": "/NestedModel/encoder", 
                        "class_name": "Sequential", 
                        "execution_order": 1,
                        "traced_tag": "/NestedModel/encoder",
                        "children": {
                            "0": {
                                "scope": "/NestedModel/encoder/0",
                                "class_name": "Linear",
                                "execution_order": 2,
                                "traced_tag": "/NestedModel/encoder/0"
                            }
                        }
                    },
                    "decoder": {
                        "scope": "/NestedModel/decoder",
                        "class_name": "Linear", 
                        "execution_order": 3,
                        "traced_tag": "/NestedModel/decoder"
                    }
                }
            },
            "node_mappings": {}
        }
        htp_path = tmp_path / "htp.json"
        htp_path.write_text(json.dumps(htp_metadata))
        
        # Convert with hierarchical converter
        converter = ONNXToGraphMLConverter(htp_metadata_path=str(htp_path))
        result = converter.convert(str(onnx_path))
        
        # Hierarchical mode returns a dict with file paths
        if isinstance(result, dict):
            graphml_path = result['graphml']
            with open(graphml_path) as f:
                graphml_str = f.read()
        else:
            graphml_str = result
            
        root = ET.fromstring(graphml_str)
        
        # Check for nested graphs
        ns = {'': 'http://graphml.graphdrawing.org/xmlns'}
        graphs = root.findall(".//graph", ns)
        assert len(graphs) > 1, "No nested graphs found in hierarchical output"
    
    def test_graph_metadata_attributes(self, tmp_path):
        """Test that graphs have proper metadata attributes."""
        # Create model
        model = nn.Linear(10, 5)
        dummy_input = torch.randn(1, 10)
        onnx_path = tmp_path / "test.onnx"
        torch.onnx.export(model, dummy_input, str(onnx_path))
        
        # Convert to GraphML
        converter = ONNXToGraphMLConverter(hierarchical=False)
        graphml_str = converter.convert(str(onnx_path))
        root = ET.fromstring(graphml_str)
        
        # Check main graph has metadata
        ns = {'': 'http://graphml.graphdrawing.org/xmlns'}
        main_graph = root.find("./graph", ns)
        assert main_graph is not None, "No main graph found"
        
        # Check for metadata elements
        source_file = main_graph.find("./data[@key='meta0']", ns)
        assert source_file is not None, "Missing source_file metadata"
        assert source_file.text == "test.onnx", f"Wrong source file: {source_file.text}"
        
        # Check for format version
        format_version = main_graph.find("./data[@key='meta2']", ns)
        assert format_version is not None, "Missing format_version metadata"
        assert format_version.text == GRAPHML_VERSION, f"Wrong format version: {format_version.text}"
        
        # Check for timestamp
        timestamp = main_graph.find("./data[@key='meta3']", ns)
        assert timestamp is not None, "Missing export_timestamp metadata"
        assert timestamp.text is not None, "Empty timestamp"


class TestMissingFeatures:
    """Test cases for features missing in current implementation."""
    
    def test_node_attributes_json_storage(self, tmp_path):
        """Test that node attributes are stored as JSON string."""
        # In baseline, node attributes are stored under key 'n2' as JSON
        model = nn.Linear(10, 5)
        dummy_input = torch.randn(1, 10)
        onnx_path = tmp_path / "test.onnx"
        
        # Add custom attributes to ONNX
        torch.onnx.export(model, dummy_input, str(onnx_path))
        onnx_model = onnx.load(str(onnx_path))
        
        # Add module_type and execution_order attributes to first node
        if onnx_model.graph.node:
            node = onnx_model.graph.node[0]
            
            # Add module_type
            attr1 = onnx.AttributeProto()
            attr1.name = 'module_type'
            attr1.type = onnx.AttributeProto.STRING
            attr1.s = b'Linear'
            node.attribute.append(attr1)
            
            # Add execution_order
            attr2 = onnx.AttributeProto()
            attr2.name = 'execution_order'
            attr2.type = onnx.AttributeProto.INT
            attr2.i = 42
            node.attribute.append(attr2)
        
        onnx.save(onnx_model, str(onnx_path))
        
        # Convert to GraphML
        converter = ONNXToGraphMLConverter(hierarchical=False)
        graphml_str = converter.convert(str(onnx_path))
        root = ET.fromstring(graphml_str)
        
        # Check that node attributes are stored as JSON under key 'n2'
        ns = {'': 'http://graphml.graphdrawing.org/xmlns'}
        nodes = root.findall(".//node", ns)
        
        # Find the first operation node (skip input nodes)
        op_node = None
        for node in nodes:
            if not node.get('id').startswith('input_') and not node.get('id').startswith('output_'):
                op_node = node
                break
        
        assert op_node is not None, "No operation node found"
        
        # Check for JSON attributes under key 'n2'
        json_data = op_node.find("./data[@key='n2']", ns)
        assert json_data is not None, "Missing node_attributes JSON data"
        
        # Parse and verify JSON content
        import json
        attrs = json.loads(json_data.text)
        assert isinstance(attrs, dict), "node_attributes should be a JSON object"
        assert attrs.get('module_type') == 'Linear', f"Expected module_type 'Linear', got {attrs.get('module_type')}"
        assert attrs.get('execution_order') == 42, f"Expected execution_order 42, got {attrs.get('execution_order')}"
    
    def test_execution_order_extraction(self, tmp_path):
        """Test extraction of execution order from ONNX metadata."""
        # Create model with execution order metadata
        model = nn.Sequential(
            nn.Linear(10, 20),
            nn.ReLU(),
            nn.Linear(20, 10)
        )
        dummy_input = torch.randn(1, 10)
        onnx_path = tmp_path / "test.onnx"
        torch.onnx.export(model, dummy_input, str(onnx_path))
        
        # Add execution_order attributes
        onnx_model = onnx.load(str(onnx_path))
        for i, node in enumerate(onnx_model.graph.node[:3]):
            attr = onnx.AttributeProto()
            attr.name = 'execution_order'
            attr.type = onnx.AttributeProto.INT
            attr.i = i
            node.attribute.append(attr)
        
        onnx.save(onnx_model, str(onnx_path))
        
        # Parse with updated parser
        from modelexport.graphml.onnx_parser import ONNXGraphParser
        parser = ONNXGraphParser()
        graph_data = parser.parse(onnx_model)
        
        # Check if execution order was extracted
        nodes_with_order = [n for n in graph_data.nodes if n.execution_order is not None]
        assert len(nodes_with_order) == 3, \
            f"Expected 3 nodes with execution_order, got {len(nodes_with_order)}"
        
        # Verify correct order values
        orders = [n.execution_order for n in nodes_with_order]
        assert sorted(orders) == [0, 1, 2], f"Expected orders [0, 1, 2], got {sorted(orders)}"
    
    def test_module_type_extraction(self, tmp_path):
        """Test extraction of module type from ONNX metadata."""
        # Similar to execution order, module_type should be extracted
        model = nn.Linear(10, 5)
        dummy_input = torch.randn(1, 10)
        onnx_path = tmp_path / "test.onnx"
        torch.onnx.export(model, dummy_input, str(onnx_path))
        
        # Add module_type attribute
        onnx_model = onnx.load(str(onnx_path))
        if onnx_model.graph.node:
            node = onnx_model.graph.node[0]
            attr = onnx.AttributeProto()
            attr.name = 'module_type'
            attr.type = onnx.AttributeProto.STRING
            attr.s = b'Linear'
            node.attribute.append(attr)
        
        onnx.save(onnx_model, str(onnx_path))
        
        # Parse with parser
        from modelexport.graphml.onnx_parser import ONNXGraphParser
        parser = ONNXGraphParser()
        graph_data = parser.parse(onnx_model)
        
        # Check if module type was extracted
        nodes_with_type = [n for n in graph_data.nodes if n.module_type is not None]
        assert len(nodes_with_type) >= 1, \
            f"Expected at least 1 node with module_type, got {len(nodes_with_type)}"
        
        # Verify the module type value
        assert nodes_with_type[0].module_type == 'Linear', \
            f"Expected module_type 'Linear', got {nodes_with_type[0].module_type}"


class TestEdgeCasesAndErrors:
    """Test edge cases and error handling."""
    
    def test_empty_hierarchy_tag(self, tmp_path):
        """Test handling of empty hierarchy tags."""
        model = nn.Linear(10, 5)
        dummy_input = torch.randn(1, 10)
        onnx_path = tmp_path / "test.onnx"
        torch.onnx.export(model, dummy_input, str(onnx_path))
        
        # Add empty hierarchy tag
        onnx_model = onnx.load(str(onnx_path))
        if onnx_model.graph.node:
            node = onnx_model.graph.node[0]
            attr = onnx.AttributeProto()
            attr.name = 'hierarchy_tag'
            attr.type = onnx.AttributeProto.STRING
            attr.s = b''  # Empty string
            node.attribute.append(attr)
        
        onnx.save(onnx_model, str(onnx_path))
        
        # Should handle gracefully
        converter = ONNXToGraphMLConverter(hierarchical=False)
        graphml_str = converter.convert(str(onnx_path))
        assert graphml_str is not None
    
    def test_special_characters_in_hierarchy_tag(self, tmp_path):
        """Test handling of special characters in hierarchy tags."""
        model = nn.Linear(10, 5)
        dummy_input = torch.randn(1, 10)
        onnx_path = tmp_path / "test.onnx"
        torch.onnx.export(model, dummy_input, str(onnx_path))
        
        # Add hierarchy tag with special characters
        onnx_model = onnx.load(str(onnx_path))
        if onnx_model.graph.node:
            node = onnx_model.graph.node[0]
            attr = onnx.AttributeProto()
            attr.name = 'hierarchy_tag'
            attr.type = onnx.AttributeProto.STRING
            attr.s = b'/Model<>Layer[0]/Sub-Module'
            node.attribute.append(attr)
        
        onnx.save(onnx_model, str(onnx_path))
        
        # Should handle special characters
        converter = ONNXToGraphMLConverter(hierarchical=False)
        graphml_str = converter.convert(str(onnx_path))
        
        # Parse to verify XML is valid
        root = ET.fromstring(graphml_str)
        assert root is not None
    
    def test_very_deep_hierarchy(self, tmp_path):
        """Test performance with very deep hierarchy tags."""
        model = nn.Linear(10, 5)
        dummy_input = torch.randn(1, 10)
        onnx_path = tmp_path / "test.onnx"
        torch.onnx.export(model, dummy_input, str(onnx_path))
        
        # Add very deep hierarchy tag
        onnx_model = onnx.load(str(onnx_path))
        if onnx_model.graph.node:
            node = onnx_model.graph.node[0]
            # Create a very deep path
            deep_path = '/'.join([f'Layer{i}' for i in range(50)])
            attr = onnx.AttributeProto()
            attr.name = 'hierarchy_tag'
            attr.type = onnx.AttributeProto.STRING
            attr.s = deep_path.encode('utf-8')
            node.attribute.append(attr)
        
        onnx.save(onnx_model, str(onnx_path))
        
        # Should handle deep hierarchies
        converter = ONNXToGraphMLConverter(hierarchical=False)
        graphml_str = converter.convert(str(onnx_path))
        assert deep_path in graphml_str
    
    def test_duplicate_hierarchy_tags(self, tmp_path):
        """Test handling of duplicate hierarchy tags across nodes."""
        model = nn.Sequential(
            nn.Linear(10, 20),
            nn.Linear(20, 10)
        )
        dummy_input = torch.randn(1, 10)
        onnx_path = tmp_path / "test.onnx"
        torch.onnx.export(model, dummy_input, str(onnx_path))
        
        # Add same hierarchy tag to multiple nodes
        onnx_model = onnx.load(str(onnx_path))
        for node in onnx_model.graph.node[:2]:
            attr = onnx.AttributeProto()
            attr.name = 'hierarchy_tag'
            attr.type = onnx.AttributeProto.STRING
            attr.s = b'/Model/SharedModule'
            node.attribute.append(attr)
        
        onnx.save(onnx_model, str(onnx_path))
        
        # Should handle duplicate tags
        converter = ONNXToGraphMLConverter(hierarchical=False)
        graphml_str = converter.convert(str(onnx_path))
        
        # Count occurrences
        count = graphml_str.count('/Model/SharedModule')
        assert count >= 2, "Duplicate tags not preserved"


class TestQAPerspective:
    """Test cases from QA/PM perspective for real-world scenarios."""
    
    def test_large_model_performance(self, tmp_path):
        """Test conversion performance with large models."""
        # Create a reasonably large model
        layers = []
        for _i in range(20):
            layers.extend([
                nn.Linear(100, 100),
                nn.ReLU(),
                nn.BatchNorm1d(100)
            ])
        model = nn.Sequential(*layers)
        
        dummy_input = torch.randn(1, 100)
        onnx_path = tmp_path / "large_model.onnx"
        torch.onnx.export(model, dummy_input, str(onnx_path))
        
        # Time the conversion
        import time
        converter = ONNXToGraphMLConverter(hierarchical=False)
        start = time.time()
        graphml_str = converter.convert(str(onnx_path))
        duration = time.time() - start
        
        # Should complete in reasonable time
        assert duration < 10.0, f"Conversion too slow: {duration}s"
        assert len(graphml_str) > 1000, "Output too small for large model"
    
    def test_model_with_dynamic_shapes(self, tmp_path):
        """Test handling of models with dynamic shapes."""
        model = nn.Linear(10, 5)
        dummy_input = torch.randn(1, 10)
        onnx_path = tmp_path / "dynamic.onnx"
        
        # Export with dynamic axes
        torch.onnx.export(
            model,
            dummy_input,
            str(onnx_path),
            dynamic_axes={
                'input': {0: 'batch_size'},
                'output': {0: 'batch_size'}
            }
        )
        
        # Should handle dynamic shapes
        converter = ONNXToGraphMLConverter(hierarchical=False)
        graphml_str = converter.convert(str(onnx_path))
        assert graphml_str is not None
    
    def test_model_with_custom_ops(self, tmp_path):
        """Test handling of models with custom operations."""
        # Create model with less common ops
        class CustomModel(nn.Module):
            def forward(self, x):
                # Use some less common operations
                x = torch.clamp(x, min=0, max=1)
                x = torch.where(x > 0.5, x * 2, x / 2)
                return x
        
        model = CustomModel()
        dummy_input = torch.randn(1, 10)
        onnx_path = tmp_path / "custom_ops.onnx"
        
        torch.onnx.export(model, dummy_input, str(onnx_path))
        
        # Should handle custom ops
        converter = ONNXToGraphMLConverter(hierarchical=False)
        graphml_str = converter.convert(str(onnx_path))
        
        # Check that ops are preserved
        assert 'Clip' in graphml_str or 'Clamp' in graphml_str  # Clamp -> Clip in ONNX
        assert 'Where' in graphml_str
    
    def test_incremental_conversion(self, tmp_path):
        """Test converting multiple models in sequence (memory leaks)."""
        # Convert multiple models to check for memory issues
        converter = ONNXToGraphMLConverter(hierarchical=False)
        
        for i in range(5):
            model = nn.Linear(10 + i, 5 + i)
            dummy_input = torch.randn(1, 10 + i)
            onnx_path = tmp_path / f"model_{i}.onnx"
            torch.onnx.export(model, dummy_input, str(onnx_path))
            
            # Convert
            graphml_str = converter.convert(str(onnx_path))
            assert graphml_str is not None
            
            # Save to file to test save method too
            output_path = tmp_path / f"model_{i}.graphml"
            converter.save(str(onnx_path), str(output_path))
            assert output_path.exists()
    
    def test_error_recovery(self, tmp_path):
        """Test error handling and recovery."""
        converter = ONNXToGraphMLConverter(hierarchical=False)
        
        # Test with non-existent file
        with pytest.raises(FileNotFoundError):
            converter.convert("non_existent.onnx")
        
        # Test with invalid ONNX file
        invalid_path = tmp_path / "invalid.onnx"
        invalid_path.write_text("This is not an ONNX file")
        
        with pytest.raises(Exception):  # Could be various exceptions
            converter.convert(str(invalid_path))
        
        # Should still work after errors
        model = nn.Linear(10, 5)
        dummy_input = torch.randn(1, 10)
        valid_path = tmp_path / "valid.onnx"
        torch.onnx.export(model, dummy_input, str(valid_path))
        
        graphml_str = converter.convert(str(valid_path))
        assert graphml_str is not None