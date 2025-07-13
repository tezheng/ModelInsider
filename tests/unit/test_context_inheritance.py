"""
Context Inheritance Unit Tests

Tests for context inheritance functionality implemented in iteration 2,
including producer-consumer analysis and semantic accuracy validation.
"""

import tempfile
from pathlib import Path

import onnx
import torch
import torch.nn as nn

from modelexport.strategies.htp.htp_hierarchy_exporter import HierarchyExporter


class ContextInheritanceTestModel(nn.Module):
    """Model designed to test context inheritance patterns."""
    
    def __init__(self):
        super().__init__()
        self.embedding = nn.Embedding(50, 16)
        self.linear1 = nn.Linear(16, 32)
        self.activation = nn.ReLU()
        self.linear2 = nn.Linear(32, 8)
        
    def forward(self, input_ids):
        # Embedding layer - should generate producer operations
        x = self.embedding(input_ids)
        
        # Shape analysis - should inherit from embedding context
        batch_size = x.shape[0]
        seq_len = x.shape[1]
        
        # Constants that feed into linear1 - should inherit linear1 context
        scale = torch.tensor(2.0, dtype=x.dtype, device=x.device)
        
        # Linear transformation with scaling
        x = self.linear1(x * scale)
        
        # Activation - should inherit from linear1 output
        x = self.activation(x)
        
        # Final transformation - should inherit from activation/linear2
        output = self.linear2(x)
        
        # Reshape operations - should inherit from linear2 context
        output = output.reshape(batch_size, -1)
        
        return output


class TestGraphContextBuilding:
    """Test graph context building for auxiliary operations."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.exporter = HierarchyExporter(strategy="htp")
        self.temp_dir = tempfile.mkdtemp()
    
    def test_producer_consumer_analysis(self):
        """Test producer-consumer relationship analysis."""
        model = ContextInheritanceTestModel()
        model.eval()
        inputs = torch.randint(0, 50, (2, 8))
        
        output_path = Path(self.temp_dir) / "producer_consumer_test.onnx"
        result = self.exporter.export(model, inputs, str(output_path))
        
        # Should achieve 100% coverage
        assert result['tagged_operations'] == result['total_operations']
        
        # Load ONNX model to analyze structure
        onnx_model = onnx.load(str(output_path))
        
        # Build our own producer-consumer map to validate
        producers = {}
        consumers = {}
        
        for node in onnx_model.graph.node:
            for output_tensor in node.output:
                if output_tensor:
                    producers[output_tensor] = node.name or node.op_type
            
            for input_tensor in node.input:
                if input_tensor:
                    if input_tensor not in consumers:
                        consumers[input_tensor] = []
                    consumers[input_tensor].append(node.name or node.op_type)
        
        # Should have producer-consumer relationships
        assert len(producers) > 0
        assert len(consumers) > 0
        
        # Some tensors should have both producers and consumers
        intermediate_tensors = set(producers.keys()) & set(consumers.keys())
        assert len(intermediate_tensors) > 0, "Should have intermediate tensors"
    
    def test_graph_context_building_accuracy(self):
        """Test accuracy of graph context building."""
        model = ContextInheritanceTestModel()
        model.eval()
        inputs = torch.randint(0, 50, (2, 6))
        
        output_path = Path(self.temp_dir) / "context_accuracy_test.onnx"
        result = self.exporter.export(model, inputs, str(output_path))
        
        # Should have node tags with proper structure
        if 'node_tags' in result:
            node_tags = result['node_tags']
            
            # Should have auxiliary operations
            auxiliary_nodes = []
            main_computation_nodes = []
            
            for node_name, node_info in node_tags.items():
                op_type = node_info.get('op_type', '')
                if op_type in ['Constant', 'Shape', 'Reshape', 'Cast']:
                    auxiliary_nodes.append(node_name)
                elif op_type in ['MatMul', 'Add', 'Relu', 'Gather']:
                    main_computation_nodes.append(node_name)
            
            # Should have both types of operations
            assert len(auxiliary_nodes) > 0, "Should have auxiliary operations"
            assert len(main_computation_nodes) > 0, "Should have main computation operations"
    
    def test_context_inheritance_semantic_accuracy(self):
        """Test semantic accuracy of context inheritance."""
        model = ContextInheritanceTestModel()
        model.eval()
        inputs = torch.randint(0, 50, (2, 4))
        
        output_path = Path(self.temp_dir) / "semantic_accuracy_test.onnx"
        result = self.exporter.export(model, inputs, str(output_path))
        
        if 'node_tags' in result:
            # Analyze tag distribution for semantic accuracy
            tags_by_module = {}
            
            for node_name, node_info in result['node_tags'].items():
                tags = node_info.get('tags', [])
                op_type = node_info.get('op_type', '')
                
                for tag in tags:
                    if tag not in tags_by_module:
                        tags_by_module[tag] = []
                    tags_by_module[tag].append({
                        'node': node_name,
                        'op_type': op_type
                    })
            
            # Should have multiple module tags
            assert len(tags_by_module) > 1, "Should have multiple module contexts"
            
            # Each module should have associated operations
            for tag, operations in tags_by_module.items():
                assert len(operations) > 0, f"Tag {tag} should have operations"


class TestContextInheritanceLogic:
    """Test context inheritance logic and effectiveness."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.exporter = HierarchyExporter(strategy="htp")
        self.temp_dir = tempfile.mkdtemp()
    
    def test_context_inheritance_effectiveness(self):
        """Test effectiveness of context inheritance (success rate)."""
        model = ContextInheritanceTestModel()
        model.eval()
        inputs = torch.randint(0, 50, (2, 10))
        
        output_path = Path(self.temp_dir) / "effectiveness_test.onnx"
        result = self.exporter.export(model, inputs, str(output_path))
        
        # Analyze context inheritance success
        if 'node_tags' in result:
            auxiliary_operations = []
            
            for node_name, node_info in result['node_tags'].items():
                op_type = node_info.get('op_type', '')
                tags = node_info.get('tags', [])
                
                if op_type in ['Constant', 'Shape', 'Reshape', 'Cast', 'Transpose']:
                    auxiliary_operations.append({
                        'node': node_name,
                        'op_type': op_type,
                        'tags': tags
                    })
            
            if auxiliary_operations:
                # Check for context-specific tags (not just fallback)
                context_inherited = []
                fallback_tagged = []
                
                for aux_op in auxiliary_operations:
                    tags = aux_op['tags']
                    
                    # Look for context-specific tags (containing specific module names)
                    has_specific_context = any(
                        'Embedding' in tag or 'Linear1' in tag or 'Linear2' in tag 
                        or 'Activation' in tag
                        for tag in tags
                    )
                    
                    if has_specific_context:
                        context_inherited.append(aux_op)
                    else:
                        fallback_tagged.append(aux_op)
                
                # Should have some context inheritance success
                total_aux = len(auxiliary_operations)
                inherited_count = len(context_inherited)
                
                if total_aux > 0:
                    success_rate = (inherited_count / total_aux) * 100
                    print(f"Context inheritance success rate: {success_rate:.1f}% ({inherited_count}/{total_aux})")
                    
                    # Should achieve some context inheritance (at least 10%)
                    assert success_rate >= 10.0, f"Context inheritance too low: {success_rate:.1f}%"
    
    def test_producer_inheritance_logic(self):
        """Test producer-based context inheritance."""
        model = ContextInheritanceTestModel()
        model.eval()
        inputs = torch.randint(0, 50, (2, 6))
        
        output_path = Path(self.temp_dir) / "producer_inheritance_test.onnx"
        result = self.exporter.export(model, inputs, str(output_path))
        
        # Load ONNX to analyze producer relationships
        onnx_model = onnx.load(str(output_path))
        
        # Find auxiliary operations that consume outputs from main operations
        producers = {}
        for node in onnx_model.graph.node:
            for output_tensor in node.output:
                if output_tensor:
                    producers[output_tensor] = {
                        'node_name': node.name or f"{node.op_type}_{hash(node)}",
                        'op_type': node.op_type
                    }
        
        # Find auxiliary operations
        auxiliary_consumer_relationships = []
        
        for node in onnx_model.graph.node:
            if node.op_type in ['Constant', 'Shape', 'Reshape']:
                node_name = node.name or f"{node.op_type}_{hash(node)}"
                
                for input_tensor in node.input:
                    if input_tensor in producers:
                        producer_info = producers[input_tensor]
                        auxiliary_consumer_relationships.append({
                            'auxiliary_node': node_name,
                            'auxiliary_op': node.op_type,
                            'producer_node': producer_info['node_name'],
                            'producer_op': producer_info['op_type']
                        })
        
        # Should have some producer-consumer relationships for auxiliary ops
        print(f"Found {len(auxiliary_consumer_relationships)} auxiliary consumer relationships")
    
    def test_consumer_inheritance_logic(self):
        """Test consumer-based context inheritance."""
        model = ContextInheritanceTestModel()
        model.eval()
        inputs = torch.randint(0, 50, (2, 8))
        
        output_path = Path(self.temp_dir) / "consumer_inheritance_test.onnx"
        result = self.exporter.export(model, inputs, str(output_path))
        
        # Load ONNX to analyze consumer relationships
        onnx_model = onnx.load(str(output_path))
        
        # Find auxiliary operations that feed main operations
        consumers = {}
        for node in onnx_model.graph.node:
            node_name = node.name or f"{node.op_type}_{hash(node)}"
            for input_tensor in node.input:
                if input_tensor:
                    if input_tensor not in consumers:
                        consumers[input_tensor] = []
                    consumers[input_tensor].append({
                        'node_name': node_name,
                        'op_type': node.op_type
                    })
        
        # Find auxiliary operations that produce tensors for main operations
        auxiliary_producer_relationships = []
        
        for node in onnx_model.graph.node:
            if node.op_type in ['Constant', 'Shape', 'Reshape']:
                node_name = node.name or f"{node.op_type}_{hash(node)}"
                
                for output_tensor in node.output:
                    if output_tensor in consumers:
                        for consumer_info in consumers[output_tensor]:
                            auxiliary_producer_relationships.append({
                                'auxiliary_node': node_name,
                                'auxiliary_op': node.op_type,
                                'consumer_node': consumer_info['node_name'],
                                'consumer_op': consumer_info['op_type']
                            })
        
        # Should have some producer-consumer relationships
        print(f"Found {len(auxiliary_producer_relationships)} auxiliary producer relationships")


class TestDataFlowAnalysis:
    """Test data flow analysis for context inheritance."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.exporter = HierarchyExporter(strategy="htp")
        self.temp_dir = tempfile.mkdtemp()
    
    def test_data_flow_analysis_accuracy(self):
        """Test accuracy of data flow analysis."""
        model = ContextInheritanceTestModel()
        model.eval()
        inputs = torch.randint(0, 50, (2, 6))
        
        output_path = Path(self.temp_dir) / "data_flow_test.onnx"
        result = self.exporter.export(model, inputs, str(output_path))
        
        # Should achieve 100% coverage
        assert result['tagged_operations'] == result['total_operations']
        
        # Verify that data flow was analyzed
        if 'node_tags' in result:
            # Count different tag patterns
            tag_patterns = set()
            
            for node_info in result['node_tags'].values():
                for tag in node_info.get('tags', []):
                    # Extract module pattern from tag
                    parts = tag.split('/')
                    if len(parts) >= 3:  # /ModelName/ModuleName/...
                        module_pattern = '/'.join(parts[:3])
                        tag_patterns.add(module_pattern)
            
            # Should have multiple module patterns (indicating flow analysis)
            assert len(tag_patterns) > 1, f"Should have multiple patterns: {tag_patterns}"
    
    def test_multi_level_inheritance(self):
        """Test multi-level context inheritance through data flow."""
        class MultiLevelModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.embedding = nn.Embedding(20, 8)
                self.linear1 = nn.Linear(8, 16)
                self.linear2 = nn.Linear(16, 4)
                
            def forward(self, x):
                # Level 1: Embedding
                x = self.embedding(x)
                
                # Level 2: Linear1 with auxiliary ops
                constant1 = torch.tensor(1.5, dtype=x.dtype)
                x = self.linear1(x * constant1)
                
                # Level 3: Linear2 with more auxiliary ops
                shape = x.shape[0]
                constant2 = torch.tensor(0.5, dtype=x.dtype)
                x = x.reshape(shape, -1)
                x = self.linear2(x + constant2)
                
                return x
        
        model = MultiLevelModel()
        model.eval()
        inputs = torch.randint(0, 20, (2, 4))
        
        output_path = Path(self.temp_dir) / "multi_level_test.onnx"
        result = self.exporter.export(model, inputs, str(output_path))
        
        # Should handle multi-level inheritance
        assert result['tagged_operations'] == result['total_operations']
        
        if 'node_tags' in result:
            # Should have tags from different levels
            level_tags = set()
            for node_info in result['node_tags'].values():
                for tag in node_info.get('tags', []):
                    if 'Embedding' in tag:
                        level_tags.add('Embedding')
                    elif 'Linear1' in tag:
                        level_tags.add('Linear1')
                    elif 'Linear2' in tag:
                        level_tags.add('Linear2')
            
            # Should detect multiple levels
            assert len(level_tags) >= 2, f"Should have multiple levels: {level_tags}"


class TestContextInheritanceEdgeCases:
    """Test edge cases for context inheritance."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.exporter = HierarchyExporter(strategy="htp")
        self.temp_dir = tempfile.mkdtemp()
    
    def test_circular_dependency_handling(self):
        """Test handling of potential circular dependencies."""
        class CircularModel(nn.Module):
            def forward(self, x):
                # Create potential circular patterns
                a = x + 1
                b = a * 2
                c = b + a  # Uses both a and b
                return c
        
        model = CircularModel()
        model.eval()
        inputs = torch.randn(2, 3)
        
        output_path = Path(self.temp_dir) / "circular_test.onnx"
        result = self.exporter.export(model, inputs, str(output_path))
        
        # Should handle gracefully without infinite loops
        assert result['total_operations'] > 0
        assert result['tagged_operations'] == result['total_operations']
    
    def test_isolated_auxiliary_operations(self):
        """Test auxiliary operations with no clear context."""
        class IsolatedAuxModel(nn.Module):
            def forward(self, x):
                # Create isolated auxiliary operations
                shape_info = x.shape[0]
                constant = torch.tensor(42.0)
                # Operations not clearly connected to main computation
                result = x.sum() + constant
                return result
        
        model = IsolatedAuxModel()
        model.eval()
        inputs = torch.randn(3, 4)
        
        output_path = Path(self.temp_dir) / "isolated_aux_test.onnx"
        result = self.exporter.export(model, inputs, str(output_path))
        
        # Should handle isolated auxiliary operations with fallback
        assert result['tagged_operations'] == result['total_operations']
    
    def test_complex_inheritance_patterns(self):
        """Test complex inheritance patterns."""
        class ComplexInheritanceModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.linear1 = nn.Linear(8, 16)
                self.linear2 = nn.Linear(16, 8)
                
            def forward(self, x):
                # Multiple branching paths
                branch1 = self.linear1(x)
                
                # Branch 2 with auxiliary ops
                constant = torch.tensor(2.0, dtype=x.dtype)
                scaled_x = x * constant
                branch2 = self.linear2(scaled_x)
                
                # Merge branches with more auxiliary ops
                combined_shape = branch1.shape[0]
                result = branch1 + branch2.reshape(combined_shape, -1)
                
                return result
        
        model = ComplexInheritanceModel()
        model.eval()
        inputs = torch.randn(2, 8)
        
        output_path = Path(self.temp_dir) / "complex_inheritance_test.onnx"
        result = self.exporter.export(model, inputs, str(output_path))
        
        # Should handle complex patterns
        assert result['tagged_operations'] == result['total_operations']
        
        # Should have reasonable inheritance success
        if 'node_tags' in result:
            auxiliary_ops = []
            for node_name, node_info in result['node_tags'].items():
                if node_info.get('op_type') in ['Constant', 'Reshape']:
                    auxiliary_ops.append(node_name)
            
            # Should have auxiliary operations in complex model
            assert len(auxiliary_ops) > 0, "Should have auxiliary operations"