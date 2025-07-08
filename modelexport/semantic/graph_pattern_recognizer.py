"""
Graph Pattern Recognition for ONNX Semantic Mapping.

This module implements pattern recognition to identify common computational
subgraphs in ONNX models, enhancing semantic understanding through structural
analysis.
"""

import onnx
from typing import Dict, List, Tuple, Set, Optional, Any
from collections import defaultdict
import numpy as np
from dataclasses import dataclass
import json


@dataclass
class GraphPattern:
    """Represents a recognized graph pattern."""
    pattern_type: str
    nodes: List[str]
    semantic_type: str
    confidence: float
    metadata: Dict[str, Any]


class GraphPatternRecognizer:
    """Recognizes common computational patterns in ONNX graphs."""
    
    def __init__(self):
        self._patterns = self._define_patterns()
        self._pattern_stats = defaultdict(int)
        
    def _define_patterns(self) -> Dict[str, Dict[str, Any]]:
        """Define common ONNX graph patterns and their semantic meanings."""
        return {
            # Attention patterns
            'self_attention': {
                'description': 'Self-attention mechanism',
                'node_sequence': ['MatMul', 'Add', 'Reshape', 'Transpose', 'MatMul', 'Softmax', 'MatMul'],
                'semantic_type': 'attention',
                'confidence': 0.9,
                'variations': [
                    ['MatMul', 'Add', 'Div', 'Softmax', 'MatMul'],  # Scaled dot-product
                    ['MatMul', 'Mul', 'Add', 'Softmax', 'MatMul'],  # With scaling
                ]
            },
            
            # Layer normalization patterns
            'layer_norm': {
                'description': 'Layer normalization',
                'node_sequence': ['ReduceMean', 'Sub', 'Pow', 'ReduceMean', 'Add', 'Sqrt', 'Div', 'Mul', 'Add'],
                'semantic_type': 'normalization',
                'confidence': 0.85,
                'variations': [
                    ['ReduceMean', 'Sub', 'Mul', 'ReduceMean', 'Add', 'Sqrt', 'Div'],  # Simplified
                    ['Sub', 'Pow', 'Mean', 'Add', 'Sqrt', 'Div', 'Mul', 'Add'],  # Alternative ops
                ]
            },
            
            # Activation patterns
            'gelu': {
                'description': 'GELU activation',
                'node_sequence': ['Mul', 'Pow', 'Mul', 'Add', 'Mul', 'Tanh', 'Add', 'Mul', 'Mul'],
                'semantic_type': 'activation',
                'confidence': 0.9,
                'variations': [
                    ['Erf', 'Add', 'Mul', 'Mul'],  # Erf-based GELU
                    ['Div', 'Erf', 'Add', 'Mul'],  # Normalized GELU
                ]
            },
            
            # Feed-forward patterns
            'feed_forward': {
                'description': 'Feed-forward network',
                'node_sequence': ['MatMul', 'Add', 'activation', 'MatMul', 'Add'],
                'semantic_type': 'feed_forward',
                'confidence': 0.8,
                'activation_ops': ['Relu', 'Gelu', 'Tanh', 'Sigmoid'],
                'variations': [
                    ['MatMul', 'BiasAdd', 'activation', 'MatMul', 'BiasAdd'],
                    ['Gemm', 'activation', 'Gemm'],  # Using Gemm ops
                ]
            },
            
            # Embedding lookup patterns
            'embedding_lookup': {
                'description': 'Embedding lookup operation',
                'node_sequence': ['Gather', 'Add'],
                'semantic_type': 'embedding',
                'confidence': 0.85,
                'variations': [
                    ['Gather'],  # Simple lookup
                    ['Gather', 'Add', 'LayerNorm'],  # With layer norm
                ]
            },
            
            # Residual connection patterns
            'residual_connection': {
                'description': 'Residual/skip connection',
                'node_sequence': ['Add'],
                'semantic_type': 'residual',
                'confidence': 0.7,
                'constraints': {
                    'inputs_from_different_branches': True,
                    'one_input_is_identity': True
                }
            },
            
            # Pooling patterns
            'global_pooling': {
                'description': 'Global pooling operation',
                'node_sequence': ['ReduceMean'],
                'semantic_type': 'pooling',
                'confidence': 0.8,
                'constraints': {
                    'reduces_spatial_dims': True
                },
                'variations': [
                    ['ReduceMax'],  # Global max pooling
                    ['GlobalAveragePool'],  # Explicit global pool op
                ]
            },
            
            # Batch normalization patterns
            'batch_norm': {
                'description': 'Batch normalization',
                'node_sequence': ['Sub', 'Div', 'Mul', 'Add'],
                'semantic_type': 'normalization',
                'confidence': 0.85,
                'variations': [
                    ['BatchNormalization'],  # Direct BN op
                    ['InstanceNormalization'],  # Instance norm
                ]
            },
            
            # Convolution patterns
            'conv_block': {
                'description': 'Convolutional block',
                'node_sequence': ['Conv', 'Add', 'activation'],
                'semantic_type': 'convolution',
                'confidence': 0.85,
                'activation_ops': ['Relu', 'LeakyRelu', 'PRelu'],
                'variations': [
                    ['Conv', 'BatchNormalization', 'activation'],
                    ['Conv', 'GroupNormalization', 'activation'],
                ]
            },
            
            # Squeeze-and-excitation patterns
            'squeeze_excitation': {
                'description': 'Squeeze-and-excitation block',
                'node_sequence': ['GlobalAveragePool', 'MatMul', 'activation', 'MatMul', 'Sigmoid', 'Mul'],
                'semantic_type': 'attention',
                'confidence': 0.85,
                'activation_ops': ['Relu', 'Gelu']
            }
        }
    
    def recognize_patterns(self, onnx_model: onnx.ModelProto) -> List[GraphPattern]:
        """Recognize patterns in the ONNX model graph."""
        recognized_patterns = []
        graph = onnx_model.graph
        
        # Build node connectivity map
        node_map = {node.name: node for node in graph.node}
        output_to_producer = self._build_output_to_producer_map(graph)
        input_to_consumers = self._build_input_to_consumers_map(graph)
        
        # Try to match each pattern
        for pattern_name, pattern_def in self._patterns.items():
            matches = self._find_pattern_matches(
                graph, node_map, output_to_producer, input_to_consumers, 
                pattern_name, pattern_def
            )
            
            for match in matches:
                recognized_patterns.append(match)
                self._pattern_stats[pattern_name] += 1
        
        # Sort by confidence and remove overlapping patterns
        recognized_patterns = self._filter_overlapping_patterns(recognized_patterns)
        
        return recognized_patterns
    
    def _build_output_to_producer_map(self, graph: onnx.GraphProto) -> Dict[str, str]:
        """Build mapping from output names to producer node names."""
        output_to_producer = {}
        for node in graph.node:
            for output in node.output:
                output_to_producer[output] = node.name
        return output_to_producer
    
    def _build_input_to_consumers_map(self, graph: onnx.GraphProto) -> Dict[str, List[str]]:
        """Build mapping from input names to consumer node names."""
        input_to_consumers = defaultdict(list)
        for node in graph.node:
            for input_name in node.input:
                input_to_consumers[input_name].append(node.name)
        return input_to_consumers
    
    def _find_pattern_matches(
        self, 
        graph: onnx.GraphProto,
        node_map: Dict[str, onnx.NodeProto],
        output_to_producer: Dict[str, str],
        input_to_consumers: Dict[str, List[str]],
        pattern_name: str,
        pattern_def: Dict[str, Any]
    ) -> List[GraphPattern]:
        """Find all matches of a specific pattern in the graph."""
        matches = []
        visited = set()
        
        # Try main pattern
        for node in graph.node:
            if node.name not in visited:
                match = self._match_sequence(
                    node, node_map, output_to_producer, input_to_consumers,
                    pattern_def['node_sequence'], pattern_def, visited
                )
                if match:
                    matches.append(GraphPattern(
                        pattern_type=pattern_name,
                        nodes=match,
                        semantic_type=pattern_def['semantic_type'],
                        confidence=pattern_def['confidence'],
                        metadata={
                            'description': pattern_def['description'],
                            'start_node': match[0] if match else None
                        }
                    ))
        
        # Try variations
        if 'variations' in pattern_def:
            for variation in pattern_def['variations']:
                visited_var = set()
                for node in graph.node:
                    if node.name not in visited_var:
                        match = self._match_sequence(
                            node, node_map, output_to_producer, input_to_consumers,
                            variation, pattern_def, visited_var
                        )
                        if match:
                            matches.append(GraphPattern(
                                pattern_type=pattern_name,
                                nodes=match,
                                semantic_type=pattern_def['semantic_type'],
                                confidence=pattern_def['confidence'] * 0.9,  # Slightly lower for variations
                                metadata={
                                    'description': pattern_def['description'],
                                    'is_variation': True,
                                    'start_node': match[0] if match else None
                                }
                            ))
        
        return matches
    
    def _match_sequence(
        self,
        start_node: onnx.NodeProto,
        node_map: Dict[str, onnx.NodeProto],
        output_to_producer: Dict[str, str],
        input_to_consumers: Dict[str, List[str]],
        sequence: List[str],
        pattern_def: Dict[str, Any],
        visited: Set[str]
    ) -> Optional[List[str]]:
        """Match a sequence pattern starting from a node."""
        if not sequence:
            return []
        
        matched_nodes = []
        current_node = start_node
        
        for i, expected_op in enumerate(sequence):
            if not current_node or current_node.name in visited:
                return None
            
            # Handle wildcard activation ops
            if expected_op == 'activation' and 'activation_ops' in pattern_def:
                if current_node.op_type not in pattern_def['activation_ops']:
                    return None
            elif current_node.op_type != expected_op:
                return None
            
            matched_nodes.append(current_node.name)
            visited.add(current_node.name)
            
            # Check constraints if any
            if 'constraints' in pattern_def:
                if not self._check_constraints(
                    current_node, node_map, output_to_producer, 
                    input_to_consumers, pattern_def['constraints']
                ):
                    return None
            
            # Move to next node in sequence
            if i < len(sequence) - 1:
                # Find connected node
                next_nodes = []
                for output in current_node.output:
                    if output in input_to_consumers:
                        for consumer in input_to_consumers[output]:
                            if consumer not in visited:
                                next_nodes.append(node_map.get(consumer))
                
                if not next_nodes:
                    return None
                
                # For simplicity, take first valid next node
                current_node = next_nodes[0] if next_nodes[0] else None
        
        return matched_nodes
    
    def _check_constraints(
        self,
        node: onnx.NodeProto,
        node_map: Dict[str, onnx.NodeProto],
        output_to_producer: Dict[str, str],
        input_to_consumers: Dict[str, List[str]],
        constraints: Dict[str, Any]
    ) -> bool:
        """Check if node satisfies pattern constraints."""
        if 'inputs_from_different_branches' in constraints:
            if len(node.input) < 2:
                return False
            
            # Check if inputs come from different computation branches
            producers = []
            for input_name in node.input[:2]:  # Check first two inputs
                if input_name in output_to_producer:
                    producers.append(output_to_producer[input_name])
            
            if len(set(producers)) < 2:
                return False
        
        if 'one_input_is_identity' in constraints:
            # Simple heuristic: one input should have significantly fewer ops in its history
            # This is a simplified check - in practice would trace back further
            pass
        
        if 'reduces_spatial_dims' in constraints:
            # Check if this is a reduction operation on spatial dimensions
            if 'axes' in [attr.name for attr in node.attribute]:
                return True
        
        return True
    
    def _filter_overlapping_patterns(self, patterns: List[GraphPattern]) -> List[GraphPattern]:
        """Remove overlapping patterns, keeping higher confidence ones."""
        # Sort by confidence (descending)
        patterns.sort(key=lambda p: p.confidence, reverse=True)
        
        filtered = []
        used_nodes = set()
        
        for pattern in patterns:
            # Check if any nodes are already used
            if not any(node in used_nodes for node in pattern.nodes):
                filtered.append(pattern)
                used_nodes.update(pattern.nodes)
        
        return filtered
    
    def enhance_semantic_mappings(
        self, 
        onnx_model: onnx.ModelProto,
        existing_mappings: Dict[str, Dict[str, Any]]
    ) -> Dict[str, Dict[str, Any]]:
        """Enhance existing semantic mappings with pattern recognition."""
        patterns = self.recognize_patterns(onnx_model)
        enhanced_mappings = existing_mappings.copy()
        
        # Track enhancements
        enhancements = {
            'total_patterns': len(patterns),
            'enhanced_nodes': 0,
            'pattern_distribution': defaultdict(int)
        }
        
        for pattern in patterns:
            enhancements['pattern_distribution'][pattern.pattern_type] += 1
            
            for node_name in pattern.nodes:
                if node_name in enhanced_mappings:
                    current = enhanced_mappings[node_name]
                    
                    # Only enhance if current confidence is not high
                    if current.get('confidence', 'low') != 'high':
                        # Update with pattern information
                        enhanced_mappings[node_name]['pattern_type'] = pattern.pattern_type
                        enhanced_mappings[node_name]['pattern_semantic_type'] = pattern.semantic_type
                        enhanced_mappings[node_name]['pattern_confidence'] = pattern.confidence
                        
                        # Upgrade confidence if pattern confidence is high
                        if pattern.confidence > 0.85 and current.get('confidence', 'low') == 'low':
                            enhanced_mappings[node_name]['confidence'] = 'medium'
                            enhanced_mappings[node_name]['enhancement_source'] = 'pattern_recognition'
                        
                        enhancements['enhanced_nodes'] += 1
        
        # Add enhancement statistics to result
        if '__metadata__' not in enhanced_mappings:
            enhanced_mappings['__metadata__'] = {}
        
        enhanced_mappings['__metadata__']['pattern_recognition'] = {
            'patterns_found': enhancements['total_patterns'],
            'nodes_enhanced': enhancements['enhanced_nodes'],
            'pattern_types': dict(enhancements['pattern_distribution']),
            'recognizer_stats': dict(self._pattern_stats)
        }
        
        return enhanced_mappings
    
    def get_pattern_statistics(self) -> Dict[str, Any]:
        """Get statistics about recognized patterns."""
        return {
            'total_patterns_matched': sum(self._pattern_stats.values()),
            'pattern_distribution': dict(self._pattern_stats),
            'defined_patterns': list(self._patterns.keys())
        }