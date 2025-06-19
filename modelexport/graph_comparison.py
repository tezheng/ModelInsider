"""
DAG comparison utilities for ONNX models.

Implements mature algorithms for comparing ONNX computational graphs to determine
if models are structurally and semantically equivalent.
"""

import onnx
import networkx as nx
from typing import Dict, List, Tuple, Set, Any, Optional
from pathlib import Path
import json


def onnx_to_networkx(onnx_model: onnx.ModelProto, include_attributes: bool = False) -> nx.DiGraph:
    """
    Convert ONNX model to NetworkX directed graph.
    
    Args:
        onnx_model: ONNX model to convert
        include_attributes: Whether to include node attributes in comparison
        
    Returns:
        NetworkX DiGraph representing the ONNX computational graph
    """
    G = nx.DiGraph()
    
    # Add nodes with metadata
    for i, node in enumerate(onnx_model.graph.node):
        node_id = node.name or f"{node.op_type}_{i}"
        
        node_data = {
            'op_type': node.op_type,
            'index': i,
            'inputs': list(node.input),
            'outputs': list(node.output)
        }
        
        if include_attributes:
            # Include non-hierarchy attributes for semantic comparison
            attrs = {}
            for attr in node.attribute:
                if not attr.name.startswith('hierarchy_'):
                    attrs[attr.name] = _extract_attribute_value(attr)
            node_data['attributes'] = attrs
        
        G.add_node(node_id, **node_data)
    
    # Add edges based on tensor flow
    tensor_producers = {}
    for node in onnx_model.graph.node:
        node_id = node.name or f"{node.op_type}_{onnx_model.graph.node.index(node)}"
        for output in node.output:
            tensor_producers[output] = node_id
    
    # Create edges from tensor producers to consumers
    for node in onnx_model.graph.node:
        node_id = node.name or f"{node.op_type}_{onnx_model.graph.node.index(node)}"
        for input_tensor in node.input:
            if input_tensor in tensor_producers:
                producer_id = tensor_producers[input_tensor]
                G.add_edge(producer_id, node_id, tensor=input_tensor)
    
    return G


def _extract_attribute_value(attr: onnx.AttributeProto) -> Any:
    """Extract value from ONNX attribute."""
    if attr.type == onnx.AttributeProto.FLOAT:
        return attr.f
    elif attr.type == onnx.AttributeProto.INT:
        return attr.i
    elif attr.type == onnx.AttributeProto.STRING:
        return attr.s.decode('utf-8')
    elif attr.type == onnx.AttributeProto.FLOATS:
        return list(attr.floats)
    elif attr.type == onnx.AttributeProto.INTS:
        return list(attr.ints)
    elif attr.type == onnx.AttributeProto.STRINGS:
        return [s.decode('utf-8') for s in attr.strings]
    else:
        return str(attr)


def compare_graphs_structural(graph1: nx.DiGraph, graph2: nx.DiGraph) -> Dict[str, Any]:
    """
    Compare two graphs for structural equivalence using graph isomorphism.
    
    Args:
        graph1: First graph to compare
        graph2: Second graph to compare
        
    Returns:
        Dictionary with comparison results
    """
    # Basic structural metrics
    comparison = {
        'nodes_count_match': len(graph1.nodes) == len(graph2.nodes),
        'edges_count_match': len(graph1.edges) == len(graph2.edges),
        'node_count_diff': len(graph1.nodes) - len(graph2.nodes),
        'edge_count_diff': len(graph1.edges) - len(graph2.edges)
    }
    
    # Topological comparison
    comparison['is_structurally_isomorphic'] = nx.is_isomorphic(graph1, graph2)
    
    # Operation type distribution comparison
    ops1 = [data['op_type'] for _, data in graph1.nodes(data=True)]
    ops2 = [data['op_type'] for _, data in graph2.nodes(data=True)]
    
    from collections import Counter
    op_count1 = Counter(ops1)
    op_count2 = Counter(ops2)
    
    comparison['operation_distribution_match'] = op_count1 == op_count2
    comparison['operation_differences'] = {
        op: op_count1.get(op, 0) - op_count2.get(op, 0)
        for op in set(op_count1.keys()) | set(op_count2.keys())
        if op_count1.get(op, 0) != op_count2.get(op, 0)
    }
    
    return comparison


def compare_graphs_semantic(graph1: nx.DiGraph, graph2: nx.DiGraph) -> Dict[str, Any]:
    """
    Compare two graphs for semantic equivalence with node/edge matching.
    
    Args:
        graph1: First graph to compare  
        graph2: Second graph to compare
        
    Returns:
        Dictionary with semantic comparison results
    """
    def node_match(n1_data, n2_data):
        """Check if two nodes are semantically equivalent."""
        # Must have same operation type
        if n1_data['op_type'] != n2_data['op_type']:
            return False
        
        # Compare non-hierarchy attributes if present
        if 'attributes' in n1_data and 'attributes' in n2_data:
            return n1_data['attributes'] == n2_data['attributes']
        
        return True
    
    def edge_match(e1_data, e2_data):
        """Check if two edges are semantically equivalent."""
        # Edges represent tensor flow, tensor names may differ
        # For semantic equivalence, just check that connection exists
        return True
    
    # Semantic isomorphism with node and edge matching
    is_semantic_isomorphic = nx.is_isomorphic(
        graph1, graph2,
        node_match=node_match,
        edge_match=edge_match
    )
    
    return {
        'is_semantically_isomorphic': is_semantic_isomorphic,
        'semantic_equivalence': is_semantic_isomorphic
    }


def compare_onnx_models(model1_path: str, model2_path: str) -> Dict[str, Any]:
    """
    Comprehensive comparison of two ONNX models.
    
    Args:
        model1_path: Path to first ONNX model
        model2_path: Path to second ONNX model
        
    Returns:
        Comprehensive comparison report
    """
    # Load models
    model1 = onnx.load(model1_path)
    model2 = onnx.load(model2_path)
    
    # Convert to NetworkX graphs
    graph1_structural = onnx_to_networkx(model1, include_attributes=False)
    graph2_structural = onnx_to_networkx(model2, include_attributes=False)
    
    graph1_semantic = onnx_to_networkx(model1, include_attributes=True)
    graph2_semantic = onnx_to_networkx(model2, include_attributes=True)
    
    # Perform comparisons
    structural_comparison = compare_graphs_structural(graph1_structural, graph2_structural)
    semantic_comparison = compare_graphs_semantic(graph1_semantic, graph2_semantic)
    
    # I/O comparison
    io_comparison = {
        'input_count_match': len(model1.graph.input) == len(model2.graph.input),
        'output_count_match': len(model1.graph.output) == len(model2.graph.output),
        'input_names_match': [inp.name for inp in model1.graph.input] == [inp.name for inp in model2.graph.input],
        'output_names_match': [out.name for out in model1.graph.output] == [out.name for out in model2.graph.output]
    }
    
    # Hierarchy attribute analysis
    hierarchy_comparison = _compare_hierarchy_attributes(model1, model2)
    
    # Overall assessment
    overall_equivalent = (
        structural_comparison.get('nodes_count_match', False) and
        structural_comparison.get('edges_count_match', False) and
        semantic_comparison.get('semantic_equivalence', False) and
        io_comparison.get('input_count_match', False) and
        io_comparison.get('output_count_match', False)
    )
    
    return {
        'model1_path': model1_path,
        'model2_path': model2_path,
        'structural_comparison': structural_comparison,
        'semantic_comparison': semantic_comparison,
        'io_comparison': io_comparison,
        'hierarchy_comparison': hierarchy_comparison,
        'overall_equivalent': overall_equivalent,
        'summary': {
            'structurally_isomorphic': structural_comparison.get('is_structurally_isomorphic', False),
            'semantically_equivalent': semantic_comparison.get('semantic_equivalence', False),
            'io_compatible': io_comparison.get('input_count_match', False) and io_comparison.get('output_count_match', False),
            'hierarchy_preserved': hierarchy_comparison.get('hierarchy_only_in_model2', False)
        }
    }


def _compare_hierarchy_attributes(model1: onnx.ModelProto, model2: onnx.ModelProto) -> Dict[str, Any]:
    """Compare hierarchy attributes between two models."""
    hierarchy_attrs_1 = 0
    hierarchy_attrs_2 = 0
    
    for node in model1.graph.node:
        for attr in node.attribute:
            if attr.name.startswith('hierarchy_'):
                hierarchy_attrs_1 += 1
                break
    
    for node in model2.graph.node:
        for attr in node.attribute:
            if attr.name.startswith('hierarchy_'):
                hierarchy_attrs_2 += 1
                break
    
    return {
        'model1_hierarchy_nodes': hierarchy_attrs_1,
        'model2_hierarchy_nodes': hierarchy_attrs_2,
        'hierarchy_only_in_model2': hierarchy_attrs_1 == 0 and hierarchy_attrs_2 > 0,
        'hierarchy_difference': hierarchy_attrs_2 - hierarchy_attrs_1
    }


def diagnose_graph_differences(model1_path: str, model2_path: str, output_path: Optional[str] = None) -> Dict[str, Any]:
    """
    Diagnose specific differences between two ONNX models.
    
    Args:
        model1_path: Path to first model (baseline)
        model2_path: Path to second model (tagged)
        output_path: Optional path to save detailed diagnosis
        
    Returns:
        Detailed diagnosis of differences
    """
    comparison = compare_onnx_models(model1_path, model2_path)
    
    # Additional detailed analysis
    model1 = onnx.load(model1_path)
    model2 = onnx.load(model2_path)
    
    # Node-by-node analysis
    ops1 = [(i, node.op_type, node.name or f"{node.op_type}_{i}") 
            for i, node in enumerate(model1.graph.node)]
    ops2 = [(i, node.op_type, node.name or f"{node.op_type}_{i}") 
            for i, node in enumerate(model2.graph.node)]
    
    diagnosis = {
        'comparison_summary': comparison['summary'],
        'node_count_analysis': {
            'model1_nodes': len(model1.graph.node),
            'model2_nodes': len(model2.graph.node),
            'difference': len(model1.graph.node) - len(model2.graph.node),
            'percentage_difference': (len(model1.graph.node) - len(model2.graph.node)) / len(model1.graph.node) * 100
        },
        'operation_type_analysis': comparison['structural_comparison']['operation_differences'],
        'likely_causes': []
    }
    
    # Analyze likely causes of differences
    node_diff = diagnosis['node_count_analysis']['difference']
    if abs(node_diff) > 50:
        diagnosis['likely_causes'].append({
            'cause': 'Major structural difference',
            'description': f'Large node count difference ({node_diff}) suggests different export parameters or optimization'
        })
    
    if not comparison['structural_comparison']['operation_distribution_match']:
        diagnosis['likely_causes'].append({
            'cause': 'Operation distribution mismatch',
            'description': 'Different operation types/counts suggest graph transformation during export'
        })
    
    if comparison['hierarchy_comparison']['hierarchy_only_in_model2']:
        diagnosis['likely_causes'].append({
            'cause': 'Hierarchy attributes added',
            'description': f"Model2 has {comparison['hierarchy_comparison']['model2_hierarchy_nodes']} nodes with hierarchy attributes"
        })
    
    # Save detailed report if requested
    if output_path:
        with open(output_path, 'w') as f:
            json.dump(diagnosis, f, indent=2)
    
    return diagnosis