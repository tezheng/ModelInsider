"""
Test cases for hierarchy tag extraction from ONNX models.

This test suite verifies that the GraphML converter correctly extracts
hierarchy tags and other metadata directly from ONNX models.
"""

import json
import xml.etree.ElementTree as ET

import onnx
import torch
import torch.nn as nn

from modelexport.graphml import ONNXToGraphMLConverter
from modelexport.graphml.onnx_parser import ONNXGraphParser


class HierarchicalTestModel(nn.Module):
    """Test model with deep hierarchy for testing hierarchy tag extraction."""
    
    def __init__(self):
        super().__init__()
        self.encoder = nn.ModuleDict({
            'embeddings': nn.Sequential(
                nn.Embedding(100, 64),
                nn.LayerNorm(64)
            ),
            'layers': nn.ModuleList([
                nn.TransformerEncoderLayer(64, 2, dim_feedforward=128) 
                for _ in range(2)
            ])
        })
        self.decoder = nn.Linear(64, 10)
    
    def forward(self, x):
        # Embeddings
        x = self.encoder['embeddings'](x)
        
        # Transformer layers
        for layer in self.encoder['layers']:
            x = layer(x)
        
        # Output
        x = x.mean(dim=1)  # Pool over sequence
        return self.decoder(x)


def create_onnx_with_hierarchy_tags(output_path: str) -> onnx.ModelProto:
    """Create an ONNX model with hierarchy_tag attributes on nodes."""
    # Create a simple model
    model = HierarchicalTestModel()
    dummy_input = torch.randint(0, 100, (1, 10))
    
    # Export to ONNX with hierarchy preservation
    torch.onnx.export(
        model,
        dummy_input,
        output_path,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={'input': {0: 'batch', 1: 'seq'}},
        opset_version=17
    )
    
    # Load and modify ONNX model to add hierarchy tags
    onnx_model = onnx.load(output_path)
    
    # Debug: Print actual node names
    print("Actual node names in ONNX model:")
    for i, node in enumerate(onnx_model.graph.node):
        print(f"  {i}: '{node.name}' (op_type: {node.op_type})")
    
    # Add hierarchy_tag attributes to first few nodes for testing
    test_tags = {
        0: '/HierarchicalTestModel/ModuleDict/Sequential/Embedding',
        1: '/HierarchicalTestModel/ModuleDict/Sequential/LayerNorm',
        2: '/HierarchicalTestModel/Linear',
    }
    
    for idx, tag in test_tags.items():
        if idx < len(onnx_model.graph.node):
            node = onnx_model.graph.node[idx]
            # Add hierarchy_tag attribute
            attr = onnx.AttributeProto()
            attr.name = 'hierarchy_tag'
            attr.type = onnx.AttributeProto.STRING
            attr.s = tag.encode('utf-8')
            node.attribute.append(attr)
    
    # Save modified model
    onnx.save(onnx_model, output_path)
    return onnx_model


class TestHierarchyExtraction:
    """Test suite for hierarchy tag extraction."""
    
    def test_extract_hierarchy_tag_from_onnx_attributes(self, tmp_path):
        """Test that hierarchy_tag attributes are extracted from ONNX nodes."""
        # Create ONNX model with hierarchy tags
        onnx_path = tmp_path / "test_hierarchy.onnx"
        create_onnx_with_hierarchy_tags(str(onnx_path))
        
        # Parse the model
        parser = ONNXGraphParser()
        onnx_model = onnx.load(str(onnx_path))
        graph_data = parser.parse(onnx_model)
        
        # Check that hierarchy tags were extracted
        nodes_with_tags = [n for n in graph_data.nodes if n.hierarchy_tag]
        assert len(nodes_with_tags) > 0, "No hierarchy tags extracted from ONNX model"
        
        # Verify that we have the expected tags
        found_tags = {n.hierarchy_tag for n in graph_data.nodes if n.hierarchy_tag}
        print(f"Found hierarchy tags: {found_tags}")
        
        # Check that we have some expected patterns
        assert any('/HierarchicalTestModel' in tag for tag in found_tags), \
            f"No HierarchicalTestModel tags found. Found: {found_tags}"
    
    def test_graphml_contains_hierarchy_tags(self, tmp_path):
        """Test that GraphML output contains hierarchy tag attributes."""
        # Create ONNX model with hierarchy tags
        onnx_path = tmp_path / "test_hierarchy.onnx"
        create_onnx_with_hierarchy_tags(str(onnx_path))
        
        # Convert to GraphML (use flat mode to read existing hierarchy tags from ONNX)
        converter = ONNXToGraphMLConverter(hierarchical=False)
        graphml_str = converter.convert(str(onnx_path))
        
        # Parse GraphML and check for hierarchy tags
        root = ET.fromstring(graphml_str)
        ns = {'': 'http://graphml.graphdrawing.org/xmlns'}
        
        # Find nodes with hierarchy tags (now using key 'n1' for node hierarchy_tag)
        hierarchy_tags = root.findall(".//data[@key='n1']", ns)
        assert len(hierarchy_tags) > 0, "No hierarchy tags found in GraphML output"
        
        # Check specific values
        tag_values = [tag.text for tag in hierarchy_tags if tag.text]
        assert any('/HierarchicalTestModel' in v for v in tag_values), \
            "Expected hierarchy tag pattern not found"
    
    def test_baseline_compatibility(self, tmp_path):
        """Test that output matches baseline GraphML structure."""
        # Create test model
        onnx_path = tmp_path / "test_model.onnx"
        create_onnx_with_hierarchy_tags(str(onnx_path))
        
        # Convert to GraphML (use flat mode to read existing tags)
        converter = ONNXToGraphMLConverter(hierarchical=False)
        graphml_str = converter.convert(str(onnx_path))
        root = ET.fromstring(graphml_str)
        
        # Check for required baseline elements
        ns = {'': 'http://graphml.graphdrawing.org/xmlns'}
        
        # 1. Check key definitions match baseline
        required_keys = {
            # Graph attributes
            'd0': 'class_name',
            'd1': 'module_type',
            'd2': 'execution_order',
            'd3': 'traced_tag',
            # Node attributes
            'n0': 'op_type',
            'n1': 'hierarchy_tag',
            'n2': 'onnx_attributes',
            'n3': 'name',
            # Edge attributes
            'e0': 'tensor_name',
        }
        
        keys = root.findall(".//key", ns)
        key_map = {k.get('id'): k.get('attr.name') for k in keys}
        
        for key_id, attr_name in required_keys.items():
            assert key_id in key_map, f"Missing key definition: {key_id}"
            assert key_map[key_id] == attr_name, \
                f"Key {key_id} has wrong attr.name: {key_map[key_id]} != {attr_name}"
        
        # 2. Check nodes have proper attributes
        nodes = root.findall(".//node", ns)
        assert len(nodes) > 0, "No nodes found in GraphML"
        
        # Each node should have at least op_type (now under key 'n0')
        for node in nodes:
            op_type_data = node.find("./data[@key='n0']", ns)
            assert op_type_data is not None, f"Node {node.get('id')} missing op_type"


class TestCompoundNodeGeneration:
    """Test cases for compound node generation from hierarchy."""
    
    def test_compound_nodes_created_from_hierarchy(self, tmp_path):
        """Test that compound nodes are created for module hierarchy."""
        # Create ONNX model with hierarchy tags
        onnx_path = tmp_path / "test_hierarchy.onnx"
        create_onnx_with_hierarchy_tags(str(onnx_path))
        
        # Create HTP metadata for hierarchical converter with proper hierarchical structure
        htp_metadata = {
            "modules": {
                "scope": "/HierarchicalTestModel",
                "class_name": "HierarchicalTestModel",
                "execution_order": 0,
                "traced_tag": "/HierarchicalTestModel",
                "children": {
                    "ModuleDict": {
                        "scope": "/HierarchicalTestModel/ModuleDict",
                        "class_name": "ModuleDict",
                        "execution_order": 1,
                        "traced_tag": "/HierarchicalTestModel/ModuleDict",
                        "children": {
                            "Sequential": {
                                "scope": "/HierarchicalTestModel/ModuleDict/Sequential",
                                "class_name": "Sequential",
                                "execution_order": 2,
                                "traced_tag": "/HierarchicalTestModel/ModuleDict/Sequential",
                                "children": {
                                    "Embedding": {
                                        "scope": "/HierarchicalTestModel/ModuleDict/Sequential/Embedding",
                                        "class_name": "Embedding",
                                        "execution_order": 3,
                                        "traced_tag": "/HierarchicalTestModel/ModuleDict/Sequential/Embedding"
                                    },
                                    "LayerNorm": {
                                        "scope": "/HierarchicalTestModel/ModuleDict/Sequential/LayerNorm",
                                        "class_name": "LayerNorm",
                                        "execution_order": 4,
                                        "traced_tag": "/HierarchicalTestModel/ModuleDict/Sequential/LayerNorm"
                                    }
                                }
                            }
                        }
                    },
                    "Linear": {
                        "scope": "/HierarchicalTestModel/Linear",
                        "class_name": "Linear",
                        "execution_order": 5,
                        "traced_tag": "/HierarchicalTestModel/Linear"
                    }
                }
            },
            "node_mappings": {
                "/encoder/embeddings/0/Gather": "/HierarchicalTestModel/ModuleDict/Sequential/Embedding",
                "/encoder/embeddings/1/ReduceMean": "/HierarchicalTestModel/ModuleDict/Sequential/LayerNorm",
                "/decoder/MatMul": "/HierarchicalTestModel/Linear"
            }
        }
        
        htp_path = tmp_path / "test_htp.json"
        htp_path.write_text(json.dumps(htp_metadata))
        
        # Convert with hierarchical converter
        converter = ONNXToGraphMLConverter(hierarchical=True, htp_metadata_path=str(htp_path))
        result = converter.convert(str(onnx_path))
        
        # Parse and verify compound structure (hierarchical mode returns dict)
        if isinstance(result, dict):
            graphml_path = result["graphml"]
            root = ET.parse(graphml_path).getroot()
        else:
            root = ET.fromstring(result)
        ns = {'': 'http://graphml.graphdrawing.org/xmlns'}
        
        # Check for nested graph elements (compound nodes)
        graphs = root.findall(".//graph", ns)
        assert len(graphs) > 1, "No compound nodes (nested graphs) found"
        
        # Verify graph hierarchy
        main_graph = root.find("./graph", ns)
        assert main_graph is not None
        
        # Check for specific compound nodes
        compound_ids = [g.get('id') for g in graphs if g.get('id')]
        assert any('Sequential' in id for id in compound_ids), \
            "Sequential compound node not found"


class TestEdgeCases:
    """Test cases for edge cases and error conditions."""
    
    def test_missing_hierarchy_tags(self, tmp_path):
        """Test graceful handling when hierarchy tags are missing."""
        # Create simple model without hierarchy tags
        model = nn.Linear(10, 5)
        dummy_input = torch.randn(1, 10)
        onnx_path = tmp_path / "simple.onnx"
        
        torch.onnx.export(model, dummy_input, str(onnx_path))
        
        # Should convert without errors (use flat mode)
        converter = ONNXToGraphMLConverter(hierarchical=False)
        graphml_str = converter.convert(str(onnx_path))
        
        # Verify basic structure
        root = ET.fromstring(graphml_str)
        assert root.tag.endswith('graphml')
    
    def test_malformed_hierarchy_tags(self, tmp_path):
        """Test handling of malformed hierarchy tag values."""
        # Create ONNX with malformed tags
        model = nn.Linear(10, 5)
        dummy_input = torch.randn(1, 10)
        onnx_path = tmp_path / "malformed.onnx"
        
        torch.onnx.export(model, dummy_input, str(onnx_path))
        
        # Add malformed hierarchy tags
        onnx_model = onnx.load(str(onnx_path))
        for node in onnx_model.graph.node[:1]:
            attr = onnx.AttributeProto()
            attr.name = 'hierarchy_tag'
            attr.type = onnx.AttributeProto.STRING
            attr.s = b''  # Empty tag
            node.attribute.append(attr)
        
        onnx.save(onnx_model, str(onnx_path))
        
        # Should handle gracefully (use flat mode)
        converter = ONNXToGraphMLConverter(hierarchical=False)
        graphml_str = converter.convert(str(onnx_path))
        assert graphml_str is not None
    
    def test_deep_hierarchy_performance(self, tmp_path):
        """Test performance with deeply nested hierarchy."""
        # Create model with deep nesting
        class DeepModel(nn.Module):
            def __init__(self, depth=10):
                super().__init__()
                current = nn.Linear(10, 10)
                for i in range(depth):
                    container = nn.Sequential()
                    container.add_module(f'layer_{i}', current)
                    current = container
                self.deep = current
            
            def forward(self, x):
                return self.deep(x)
        
        model = DeepModel(depth=20)
        dummy_input = torch.randn(1, 10)
        onnx_path = tmp_path / "deep.onnx"
        
        torch.onnx.export(model, dummy_input, str(onnx_path))
        
        # Should complete in reasonable time (use flat mode)
        import time
        start = time.time()
        converter = ONNXToGraphMLConverter(hierarchical=False)
        graphml_str = converter.convert(str(onnx_path))
        duration = time.time() - start
        
        assert duration < 5.0, f"Conversion took too long: {duration}s"
        assert graphml_str is not None


class TestRealWorldScenarios:
    """Test cases based on real-world usage patterns."""
    
    def test_bert_model_structure(self, tmp_path):
        """Test with BERT-like model structure."""
        # This would test actual BERT model conversion
        # For now, create a simplified version
        class SimpleBERT(nn.Module):
            def __init__(self):
                super().__init__()
                self.embeddings = nn.ModuleDict({
                    'word_embeddings': nn.Embedding(1000, 128),
                    'position_embeddings': nn.Embedding(512, 128),
                    'token_type_embeddings': nn.Embedding(2, 128),
                    'LayerNorm': nn.LayerNorm(128)
                })
                self.encoder = nn.ModuleList([
                    nn.TransformerEncoderLayer(128, 4, 512)
                    for _ in range(2)
                ])
            
            def forward(self, input_ids, token_type_ids=None):
                word_emb = self.embeddings['word_embeddings'](input_ids)
                pos_ids = torch.arange(input_ids.size(1), device=input_ids.device)
                pos_emb = self.embeddings['position_embeddings'](pos_ids)
                
                if token_type_ids is None:
                    token_type_ids = torch.zeros_like(input_ids)
                type_emb = self.embeddings['token_type_embeddings'](token_type_ids)
                
                embeddings = word_emb + pos_emb + type_emb
                embeddings = self.embeddings['LayerNorm'](embeddings)
                
                for layer in self.encoder:
                    embeddings = layer(embeddings)
                
                return embeddings
        
        model = SimpleBERT()
        input_ids = torch.randint(0, 1000, (1, 20))
        onnx_path = tmp_path / "simple_bert.onnx"
        
        torch.onnx.export(
            model,
            (input_ids,),
            str(onnx_path),
            input_names=['input_ids'],
            output_names=['output']
        )
        
        # Convert and verify structure (use flat mode)
        converter = ONNXToGraphMLConverter(hierarchical=False)
        graphml_str = converter.convert(str(onnx_path))
        
        # Check for expected patterns
        assert 'Gather' in graphml_str  # Embedding operations
        assert 'LayerNorm' in graphml_str or 'ReduceMean' in graphml_str  # LayerNorm ops
    
    def test_concurrent_conversion(self, tmp_path):
        """Test thread safety with concurrent conversions."""
        import concurrent.futures
        
        # Create multiple models
        models = []
        for i in range(5):
            model = nn.Sequential(
                nn.Linear(10, 20),
                nn.ReLU(),
                nn.Linear(20, 10)
            )
            onnx_path = tmp_path / f"model_{i}.onnx"
            torch.onnx.export(
                model,
                torch.randn(1, 10),
                str(onnx_path)
            )
            models.append(onnx_path)
        
        # Convert concurrently (use flat mode)
        converter = ONNXToGraphMLConverter(hierarchical=False)
        with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
            futures = [
                executor.submit(converter.convert, str(model_path))
                for model_path in models
            ]
            
            results = []
            for future in concurrent.futures.as_completed(futures):
                result = future.result()
                results.append(result)
        
        # All conversions should succeed
        assert len(results) == 5
        for result in results:
            assert result is not None
            assert '<graphml' in result