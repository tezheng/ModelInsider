"""
ONNX Graph Filtering Utilities with Safety Validation

This module provides safe graph filtering capabilities for ONNX models exported with
hierarchy-preserving tags. Ensures that filtered subgraphs maintain integrity and 
remain executable.

Key Features:
1. Tag-based filtering with dependency resolution
2. Graph connectivity validation
3. Auxiliary operations handling
4. Execution safety verification
"""

import onnx
import re
from typing import Dict, List, Set, Tuple, Optional, Any, Union
from collections import defaultdict, deque
import json
from pathlib import Path


class GraphFilteringError(Exception):
    """Exception raised when graph filtering would create invalid results."""
    pass


class ONNXGraphFilter:
    """
    Safe ONNX graph filtering with connectivity validation and dependency resolution.
    """
    
    def __init__(self, onnx_model_path: str, hierarchy_json_path: Optional[str] = None):
        """
        Initialize graph filter with ONNX model and hierarchy information.
        
        Args:
            onnx_model_path: Path to ONNX model file
            hierarchy_json_path: Optional path to hierarchy JSON file
        """
        self.onnx_model = onnx.load(onnx_model_path)
        self.hierarchy_data = {}
        
        if hierarchy_json_path and Path(hierarchy_json_path).exists():
            with open(hierarchy_json_path, 'r') as f:
                self.hierarchy_data = json.load(f)
        
        # Build graph analysis structures
        self.node_map = {}  # node_name -> node
        self.tensor_producers = {}  # tensor_name -> node_name  
        self.tensor_consumers = defaultdict(list)  # tensor_name -> [node_names]
        self.node_tags = {}  # node_name -> [tags]
        
        self._analyze_graph_structure()
    
    def _analyze_graph_structure(self):
        """Analyze ONNX graph structure for filtering operations."""
        # Build node mapping
        for node in self.onnx_model.graph.node:
            node_name = node.name or f"{node.op_type}_{len(self.node_map)}"
            self.node_map[node_name] = node
            
            # Build tensor producer-consumer relationships
            for output_tensor in node.output:
                if output_tensor:  # Skip empty output names
                    self.tensor_producers[output_tensor] = node_name
            
            for input_tensor in node.input:
                if input_tensor:  # Skip empty input names
                    self.tensor_consumers[input_tensor].append(node_name)
        
        # Extract node tags from hierarchy data
        if 'node_tags' in self.hierarchy_data:
            for node_name, node_info in self.hierarchy_data['node_tags'].items():
                self.node_tags[node_name] = node_info.get('tags', [])
        else:
            # Fallback: create empty tags for all nodes
            for node_name in self.node_map.keys():
                self.node_tags[node_name] = []
    
    def filter_by_tags(
        self, 
        tag_patterns: Union[str, List[str]], 
        include_dependencies: bool = True,
        include_auxiliary: bool = True,
        validate_safety: bool = True
    ) -> onnx.ModelProto:
        """
        Filter ONNX graph by tag patterns with safety validation.
        
        Args:
            tag_patterns: Tag pattern(s) to match (supports regex)
            include_dependencies: Whether to include dependency operations
            include_auxiliary: Whether to include auxiliary operations
            validate_safety: Whether to validate filtered graph safety
        
        Returns:
            Filtered ONNX model
        
        Raises:
            GraphFilteringError: If filtering would create invalid graph
        """
        if isinstance(tag_patterns, str):
            tag_patterns = [tag_patterns]
        
        print(f"🔍 Filtering graph with tag patterns: {tag_patterns}")
        
        # Step 1: Find nodes matching tag patterns
        matching_nodes = self._find_nodes_by_tags(tag_patterns)
        print(f"📋 Found {len(matching_nodes)} nodes matching tag patterns")
        
        # Step 2: Include dependencies if requested
        if include_dependencies:
            matching_nodes = self._include_dependencies(matching_nodes)
            print(f"🔗 Expanded to {len(matching_nodes)} nodes with dependencies")
        
        # Step 3: Include auxiliary operations if requested
        if include_auxiliary:
            matching_nodes = self._include_auxiliary_operations(matching_nodes)
            print(f"⚙️ Expanded to {len(matching_nodes)} nodes with auxiliary operations")
        
        # Step 4: Validate graph safety
        if validate_safety:
            safety_issues = self._validate_subgraph_safety(matching_nodes)
            if safety_issues:
                raise GraphFilteringError(f"Graph filtering safety issues: {safety_issues}")
        
        # Step 5: Create filtered ONNX model
        filtered_model = self._create_filtered_model(matching_nodes)
        print(f"✅ Created filtered model with {len(matching_nodes)} operations")
        
        return filtered_model
    
    def _find_nodes_by_tags(self, tag_patterns: List[str]) -> Set[str]:
        """Find nodes matching tag patterns."""
        matching_nodes = set()
        
        for node_name, tags in self.node_tags.items():
            if self._tags_match_patterns(tags, tag_patterns):
                matching_nodes.add(node_name)
        
        return matching_nodes
    
    def _tags_match_patterns(self, tags: List[str], patterns: List[str]) -> bool:
        """Check if any tag matches any pattern."""
        for tag in tags:
            for pattern in patterns:
                if re.search(pattern, tag, re.IGNORECASE):
                    return True
        return False
    
    def _include_dependencies(self, nodes: Set[str]) -> Set[str]:
        """Include dependency operations for the given nodes."""
        expanded_nodes = set(nodes)
        nodes_to_process = deque(nodes)
        
        while nodes_to_process:
            current_node = nodes_to_process.popleft()
            
            if current_node not in self.node_map:
                continue
            
            node = self.node_map[current_node]
            
            # Include producers of input tensors
            for input_tensor in node.input:
                if input_tensor in self.tensor_producers:
                    producer_node = self.tensor_producers[input_tensor]
                    if producer_node not in expanded_nodes:
                        expanded_nodes.add(producer_node)
                        nodes_to_process.append(producer_node)
        
        return expanded_nodes
    
    def _include_auxiliary_operations(self, nodes: Set[str]) -> Set[str]:
        """Include auxiliary operations that support the given nodes."""
        auxiliary_ops = {'Shape', 'Constant', 'Cast', 'Reshape', 'Transpose', 
                        'Unsqueeze', 'Squeeze', 'Where', 'Gather', 'ReduceMean',
                        'Slice', 'Concat', 'Add', 'Sub', 'Mul', 'Div'}
        
        expanded_nodes = set(nodes)
        
        # Find auxiliary operations connected to our nodes
        for node_name in list(expanded_nodes):
            if node_name not in self.node_map:
                continue
                
            node = self.node_map[node_name]
            
            # Check input tensors for auxiliary operations
            for input_tensor in node.input:
                if input_tensor in self.tensor_producers:
                    producer_node_name = self.tensor_producers[input_tensor]
                    producer_node = self.node_map.get(producer_node_name)
                    
                    if (producer_node and 
                        producer_node.op_type in auxiliary_ops and 
                        producer_node_name not in expanded_nodes):
                        expanded_nodes.add(producer_node_name)
            
            # Check output tensors for auxiliary operations
            for output_tensor in node.output:
                if output_tensor in self.tensor_consumers:
                    for consumer_node_name in self.tensor_consumers[output_tensor]:
                        consumer_node = self.node_map.get(consumer_node_name)
                        
                        if (consumer_node and 
                            consumer_node.op_type in auxiliary_ops and 
                            consumer_node_name not in expanded_nodes):
                            expanded_nodes.add(consumer_node_name)
        
        return expanded_nodes
    
    def _validate_subgraph_safety(self, nodes: Set[str]) -> List[str]:
        """Validate that the subgraph is safe and executable."""
        issues = []
        
        # Check 1: All input dependencies are satisfied
        external_inputs = set()
        for node_name in nodes:
            if node_name not in self.node_map:
                continue
                
            node = self.node_map[node_name]
            for input_tensor in node.input:
                if input_tensor not in self.tensor_producers:
                    # This is a model input
                    external_inputs.add(input_tensor)
                elif self.tensor_producers[input_tensor] not in nodes:
                    # This input comes from outside the filtered subgraph
                    issues.append(f"Node {node_name} depends on external tensor {input_tensor}")
        
        # Check 2: Subgraph has valid outputs
        subgraph_outputs = set()
        for node_name in nodes:
            if node_name not in self.node_map:
                continue
                
            node = self.node_map[node_name]
            for output_tensor in node.output:
                # Check if this output is consumed outside the subgraph
                external_consumers = [
                    consumer for consumer in self.tensor_consumers.get(output_tensor, [])
                    if consumer not in nodes
                ]
                if not external_consumers:
                    # This output is only used within the subgraph or not used at all
                    subgraph_outputs.add(output_tensor)
        
        if not subgraph_outputs and len(nodes) > 1:
            issues.append("Subgraph has no valid outputs")
        
        # Check 3: No isolated nodes
        for node_name in nodes:
            if node_name not in self.node_map:
                continue
                
            node = self.node_map[node_name]
            has_valid_inputs = any(
                input_tensor in self.tensor_producers and 
                self.tensor_producers[input_tensor] in nodes
                for input_tensor in node.input
            )
            has_external_inputs = any(
                input_tensor not in self.tensor_producers
                for input_tensor in node.input
            )
            
            if not has_valid_inputs and not has_external_inputs and len(node.input) > 0:
                issues.append(f"Node {node_name} appears to be isolated")
        
        return issues
    
    def _create_filtered_model(self, nodes: Set[str]) -> onnx.ModelProto:
        """Create a new ONNX model with only the specified nodes."""
        # Create a new model based on the original
        filtered_model = onnx.ModelProto()
        filtered_model.CopyFrom(self.onnx_model)
        
        # Clear the graph and rebuild with filtered nodes in topological order
        filtered_model.graph.ClearField('node')
        
        # Sort nodes in topological order to maintain dependencies
        sorted_nodes = self._topological_sort_nodes(nodes)
        
        # Add filtered nodes in correct order
        for node_name in sorted_nodes:
            if node_name in self.node_map:
                new_node = onnx.NodeProto()
                new_node.CopyFrom(self.node_map[node_name])
                filtered_model.graph.node.append(new_node)
        
        # Update graph inputs and outputs to match filtered subgraph
        self._update_graph_inputs_outputs(filtered_model, nodes)
        
        return filtered_model
    
    def _topological_sort_nodes(self, nodes: Set[str]) -> List[str]:
        """Sort nodes in topological order to maintain dependencies."""
        # Build dependency graph for the filtered nodes
        in_degree = {node: 0 for node in nodes}
        dependencies = defaultdict(list)
        
        for node_name in nodes:
            if node_name not in self.node_map:
                continue
                
            node = self.node_map[node_name]
            
            # Count dependencies within the filtered set
            for input_tensor in node.input:
                if input_tensor in self.tensor_producers:
                    producer = self.tensor_producers[input_tensor]
                    if producer in nodes and producer != node_name:
                        dependencies[producer].append(node_name)
                        in_degree[node_name] += 1
        
        # Topological sort using Kahn's algorithm
        queue = deque([node for node in nodes if in_degree[node] == 0])
        sorted_nodes = []
        
        while queue:
            current = queue.popleft()
            sorted_nodes.append(current)
            
            for dependent in dependencies[current]:
                in_degree[dependent] -= 1
                if in_degree[dependent] == 0:
                    queue.append(dependent)
        
        # Handle any remaining nodes (cycles or disconnected nodes)
        remaining = set(nodes) - set(sorted_nodes)
        sorted_nodes.extend(remaining)
        
        return sorted_nodes
    
    def _update_graph_inputs_outputs(self, model: onnx.ModelProto, nodes: Set[str]):
        """Update graph inputs and outputs for filtered model."""
        # Find actual inputs and outputs of the filtered subgraph
        subgraph_inputs = set()
        subgraph_outputs = set()
        internal_tensors = set()
        
        # Collect all tensors produced by the subgraph
        for node_name in nodes:
            if node_name not in self.node_map:
                continue
                
            node = self.node_map[node_name]
            for output_tensor in node.output:
                internal_tensors.add(output_tensor)
        
        # Find inputs (tensors consumed but not produced within subgraph)
        for node_name in nodes:
            if node_name not in self.node_map:
                continue
                
            node = self.node_map[node_name]
            for input_tensor in node.input:
                if input_tensor and input_tensor not in internal_tensors:
                    subgraph_inputs.add(input_tensor)
        
        # Find outputs (tensors produced but consumed outside subgraph or final outputs)
        for node_name in nodes:
            if node_name not in self.node_map:
                continue
                
            node = self.node_map[node_name]
            for output_tensor in node.output:
                if output_tensor:
                    # Check if this tensor is consumed outside the subgraph
                    external_consumers = [
                        consumer for consumer in self.tensor_consumers.get(output_tensor, [])
                        if consumer not in nodes
                    ]
                    # Include as output if consumed externally or if it's a terminal output
                    if external_consumers or output_tensor not in self.tensor_consumers:
                        subgraph_outputs.add(output_tensor)
        
        # Update model inputs (preserve original input information where available)
        model.graph.ClearField('input')
        for input_tensor in subgraph_inputs:
            # Try to find original input info
            original_input = None
            for orig_input in self.onnx_model.graph.input:
                if orig_input.name == input_tensor:
                    original_input = orig_input
                    break
            
            if original_input:
                new_input = onnx.ValueInfoProto()
                new_input.CopyFrom(original_input)
                model.graph.input.append(new_input)
            else:
                # Create a basic input specification
                new_input = onnx.helper.make_tensor_value_info(
                    input_tensor, onnx.TensorProto.FLOAT, []
                )
                model.graph.input.append(new_input)
        
        # Update model outputs (preserve original output information where available)
        model.graph.ClearField('output')
        for output_tensor in subgraph_outputs:
            # Try to find original output info
            original_output = None
            for orig_output in self.onnx_model.graph.output:
                if orig_output.name == output_tensor:
                    original_output = orig_output
                    break
            
            if original_output:
                new_output = onnx.ValueInfoProto()
                new_output.CopyFrom(original_output)
                model.graph.output.append(new_output)
            else:
                # Create a basic output specification
                new_output = onnx.helper.make_tensor_value_info(
                    output_tensor, onnx.TensorProto.FLOAT, []
                )
                model.graph.output.append(new_output)
    
    def analyze_graph_structure(self) -> Dict[str, Any]:
        """Analyze and return graph structure information."""
        total_nodes = len(self.node_map)
        tagged_nodes = sum(1 for tags in self.node_tags.values() if tags)
        
        # Count operation types
        op_type_counts = defaultdict(int)
        for node in self.node_map.values():
            op_type_counts[node.op_type] += 1
        
        # Analyze tag distribution
        tag_counts = defaultdict(int)
        for tags in self.node_tags.values():
            for tag in tags:
                tag_counts[tag] += 1
        
        return {
            'total_nodes': total_nodes,
            'tagged_nodes': tagged_nodes,
            'tag_coverage': (tagged_nodes / total_nodes * 100) if total_nodes > 0 else 0,
            'operation_types': dict(op_type_counts),
            'tag_distribution': dict(tag_counts),
            'unique_tags': len(tag_counts),
            'tensor_count': len(self.tensor_producers),
        }
    
    def validate_model_integrity(self, model: Optional[onnx.ModelProto] = None) -> List[str]:
        """Validate ONNX model integrity."""
        if model is None:
            model = self.onnx_model
        
        issues = []
        
        try:
            # Basic ONNX validation
            onnx.checker.check_model(model)
        except Exception as e:
            issues.append(f"ONNX validation failed: {str(e)}")
        
        # Check for common issues
        node_names = {node.name for node in model.graph.node if node.name}
        if len(node_names) != len(model.graph.node):
            issues.append("Some nodes have missing or duplicate names")
        
        # Check tensor connectivity
        produced_tensors = set()
        consumed_tensors = set()
        
        for node in model.graph.node:
            for output in node.output:
                if output:
                    produced_tensors.add(output)
            for input_tensor in node.input:
                if input_tensor:
                    consumed_tensors.add(input_tensor)
        
        # Add model inputs to produced tensors
        for input_info in model.graph.input:
            produced_tensors.add(input_info.name)
        
        # Check for consumed but not produced tensors (excluding initializers)
        initializer_names = {init.name for init in model.graph.initializer}
        orphaned_tensors = consumed_tensors - produced_tensors - initializer_names
        
        if orphaned_tensors:
            issues.append(f"Orphaned tensors (consumed but not produced): {list(orphaned_tensors)[:5]}")
        
        return issues


def create_filtering_test_suite():
    """Create a comprehensive test suite for graph filtering safety."""
    return [
        {
            'name': 'Single Tag Filtering',
            'patterns': ['/CustomAuxiliaryTestModel/Embedding'],
            'description': 'Filter operations with specific tag',
            'expected_safety': True
        },
        {
            'name': 'Hierarchical Filtering', 
            'patterns': ['/CustomAuxiliaryTestModel/.*'],
            'description': 'Filter by tag prefix pattern',
            'expected_safety': True
        },
        {
            'name': 'Multi-Tag Filtering',
            'patterns': ['/CustomAuxiliaryTestModel/Embedding', '/CustomAuxiliaryTestModel/Linear1'],
            'description': 'Filter multiple specific tags',
            'expected_safety': True
        },
        {
            'name': 'Auxiliary Operations Test',
            'patterns': ['Constant', 'MatMul'],
            'description': 'Filter auxiliary and computation operations',
            'expected_safety': True
        },
        {
            'name': 'Empty Pattern Test',
            'patterns': ['NonExistentTag'],
            'description': 'Filter with non-matching pattern',
            'expected_safety': False
        }
    ]