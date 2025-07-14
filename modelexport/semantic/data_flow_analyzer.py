#!/usr/bin/env python3
"""
Data Flow Analysis for Semantic Inheritance

This module implements data flow analysis to help nodes inherit semantic context
from their input/output connections in the ONNX graph.

Key capabilities:
1. Backward semantic tracing - inherit context from inputs
2. Forward semantic propagation - propagate context to outputs  
3. Multi-path context resolution - handle nodes with multiple semantic sources
4. Confidence scoring for inherited semantics

CARDINAL RULES:
- MUST-001: NO HARDCODED LOGIC - Universal graph analysis only
- MUST-002: UNIVERSAL DESIGN - Works with any ONNX model
- MUST-003: CONFIDENCE AWARE - All inherited semantics have confidence scores
"""

import logging
from collections import defaultdict, deque
from typing import Any

import onnx

logger = logging.getLogger(__name__)


class DataFlowAnalyzer:
    """
    Analyzes ONNX model data flow to enable semantic inheritance.
    
    This class builds a graph representation of the ONNX model and provides
    methods to trace semantic context through data dependencies.
    """
    
    def __init__(self, onnx_model: onnx.ModelProto, semantic_mappings: dict[str, Any]):
        """
        Initialize data flow analyzer.
        
        Args:
            onnx_model: ONNX model proto
            semantic_mappings: Existing semantic mappings from enhanced semantic mapper
        """
        self.onnx_model = onnx_model
        self.semantic_mappings = semantic_mappings
        self.node_graph = {}
        self.input_graph = {}  # tensor_name -> nodes that consume it
        self.output_graph = {}  # tensor_name -> node that produces it
        self.enhanced_mappings = {}
        
        self._build_data_flow_graph()
    
    def _build_data_flow_graph(self) -> None:
        """Build internal data flow graph representation."""
        # Build node index
        for node in self.onnx_model.graph.node:
            self.node_graph[node.name] = {
                'node': node,
                'inputs': list(node.input),
                'outputs': list(node.output),
                'op_type': node.op_type
            }
        
        # Build tensor-to-node mappings
        for node in self.onnx_model.graph.node:
            # Map outputs to producing node
            for output_tensor in node.output:
                self.output_graph[output_tensor] = node.name
            
            # Map inputs to consuming nodes
            for input_tensor in node.input:
                if input_tensor not in self.input_graph:
                    self.input_graph[input_tensor] = []
                self.input_graph[input_tensor].append(node.name)
        
        logger.info(f"Built data flow graph: {len(self.node_graph)} nodes, "
                   f"{len(self.output_graph)} tensors")
    
    def enhance_semantic_mappings(self) -> dict[str, Any]:
        """
        Enhance existing semantic mappings using data flow analysis.
        
        Returns:
            Enhanced semantic mappings with improved coverage and confidence
        """
        # Start with existing mappings
        self.enhanced_mappings = self.semantic_mappings.copy()
        
        # Identify nodes that need enhancement
        enhancement_candidates = self._identify_enhancement_candidates()
        
        logger.info(f"Found {len(enhancement_candidates)} candidates for semantic enhancement")
        
        # Apply enhancement strategies
        improved_count = 0
        
        for node_name in enhancement_candidates:
            original_mapping = self.enhanced_mappings[node_name]
            enhanced_mapping = self._enhance_node_semantics(node_name, original_mapping)
            
            if enhanced_mapping != original_mapping:
                self.enhanced_mappings[node_name] = enhanced_mapping
                improved_count += 1
        
        logger.info(f"Enhanced semantics for {improved_count} nodes through data flow analysis")
        
        return self.enhanced_mappings
    
    def _identify_enhancement_candidates(self) -> list[str]:
        """Identify nodes that could benefit from data flow enhancement."""
        candidates = []
        
        for node_name, mapping in self.semantic_mappings.items():
            semantic_type = mapping.get('semantic_type', 'unknown')
            confidence = mapping.get('confidence', 'unknown')
            primary_source = mapping.get('primary_source', 'unknown')
            
            # Candidates for enhancement:
            # 1. Unknown semantic type
            # 2. Low confidence  
            # 3. Pattern fallback (could be improved with context)
            # 4. Medium confidence nodes that could potentially be improved
            # EXCLUDE high confidence nodes to preserve their quality
            if (semantic_type == 'unknown' or 
                confidence in ['low', 'medium'] or 
                primary_source == 'pattern_fallback'):
                candidates.append(node_name)
        
        return candidates
    
    def _enhance_node_semantics(self, node_name: str, original_mapping: dict[str, Any]) -> dict[str, Any]:
        """
        Enhance semantics for a specific node using data flow analysis.
        
        Args:
            node_name: Name of node to enhance
            original_mapping: Original semantic mapping
            
        Returns:
            Enhanced semantic mapping (only if improvement is made)
        """
        original_semantic_type = original_mapping.get('semantic_type', 'unknown')
        original_confidence = original_mapping.get('confidence', 'unknown')
        
        # Don't modify high confidence nodes
        if original_confidence == 'high':
            return original_mapping
        
        best_enhancement = None
        best_score = self._calculate_enhancement_score(original_mapping)
        
        # Strategy 1: Backward semantic inheritance
        backward_inheritance = self._try_backward_semantic_inheritance(node_name)
        if backward_inheritance['success']:
            temp_mapping = original_mapping.copy()
            temp_mapping.update(backward_inheritance['enhancement'])
            score = self._calculate_enhancement_score(temp_mapping)
            if score > best_score:
                best_enhancement = temp_mapping
                best_score = score
        
        # Strategy 2: Forward semantic propagation  
        forward_propagation = self._try_forward_semantic_propagation(node_name)
        if forward_propagation['success']:
            temp_mapping = original_mapping.copy()
            temp_mapping.update(forward_propagation['enhancement'])
            score = self._calculate_enhancement_score(temp_mapping)
            if score > best_score:
                best_enhancement = temp_mapping
                best_score = score
        
        # Strategy 3: Contextual operation inference
        contextual_inference = self._try_contextual_operation_inference(node_name)
        if contextual_inference['success']:
            temp_mapping = original_mapping.copy()
            temp_mapping.update(contextual_inference['enhancement'])
            score = self._calculate_enhancement_score(temp_mapping)
            if score > best_score:
                best_enhancement = temp_mapping
                best_score = score
        
        # Return best enhancement or original if no improvement
        return best_enhancement if best_enhancement else original_mapping
    
    def _calculate_enhancement_score(self, mapping: dict[str, Any]) -> float:
        """Calculate enhancement score for a mapping (higher is better)."""
        semantic_type = mapping.get('semantic_type', 'unknown')
        confidence = mapping.get('confidence', 'unknown')
        
        # Base score from semantic type
        semantic_score = 0.0 if semantic_type == 'unknown' else 1.0
        
        # Confidence score
        confidence_scores = {'high': 3.0, 'medium': 2.0, 'low': 1.0, 'unknown': 0.0}
        confidence_score = confidence_scores.get(confidence, 0.0)
        
        return semantic_score + confidence_score
    
    def _try_backward_semantic_inheritance(self, node_name: str) -> dict[str, Any]:
        """
        Try to inherit semantics from input nodes (backward tracing).
        
        This is particularly useful for constants and operations that don't
        have inherent semantic meaning but get meaning from their consumers.
        """
        if node_name not in self.node_graph:
            return {'success': False}
        
        node_info = self.node_graph[node_name]
        input_semantics = []
        
        # Collect semantics from input producers
        for input_tensor in node_info['inputs']:
            if input_tensor in self.output_graph:
                producer_node = self.output_graph[input_tensor]
                if producer_node in self.semantic_mappings:
                    producer_mapping = self.semantic_mappings[producer_node]
                    if producer_mapping.get('confidence') in ['high', 'medium']:
                        input_semantics.append(producer_mapping)
        
        if not input_semantics:
            return {'success': False}
        
        # Aggregate input semantics
        inherited_semantics = self._aggregate_semantic_contexts(input_semantics)
        
        # Apply inheritance rules based on operation type
        op_type = node_info['op_type']
        enhancement = self._apply_inheritance_rules(inherited_semantics, op_type)
        
        if enhancement:
            return {
                'success': True,
                'enhancement': {
                    'semantic_type': enhancement['semantic_type'],
                    'confidence': 'medium',  # Inherited confidence is medium
                    'primary_source': 'data_flow_backward',
                    'inheritance_context': inherited_semantics
                }
            }
        
        return {'success': False}
    
    def _try_forward_semantic_propagation(self, node_name: str) -> dict[str, Any]:
        """
        Try to infer semantics from output consumers (forward tracing).
        
        Useful when a node's meaning can be inferred from how it's used.
        """
        if node_name not in self.node_graph:
            return {'success': False}
        
        node_info = self.node_graph[node_name]
        consumer_semantics = []
        
        # Collect semantics from output consumers
        for output_tensor in node_info['outputs']:
            if output_tensor in self.input_graph:
                for consumer_node in self.input_graph[output_tensor]:
                    if consumer_node in self.semantic_mappings:
                        consumer_mapping = self.semantic_mappings[consumer_node]
                        if consumer_mapping.get('confidence') in ['high', 'medium']:
                            consumer_semantics.append(consumer_mapping)
        
        if not consumer_semantics:
            return {'success': False}
        
        # Aggregate consumer semantics
        forward_semantics = self._aggregate_semantic_contexts(consumer_semantics)
        
        # Apply forward propagation rules
        op_type = node_info['op_type']
        enhancement = self._apply_forward_propagation_rules(forward_semantics, op_type)
        
        if enhancement:
            return {
                'success': True,
                'enhancement': {
                    'semantic_type': enhancement['semantic_type'],
                    'confidence': 'medium',
                    'primary_source': 'data_flow_forward',
                    'propagation_context': forward_semantics
                }
            }
        
        return {'success': False}
    
    def _try_contextual_operation_inference(self, node_name: str) -> dict[str, Any]:
        """
        Try to infer semantics using broader graph context.
        
        Looks at the neighborhood of operations to understand the computational pattern.
        """
        if node_name not in self.node_graph:
            return {'success': False}
        
        node_info = self.node_graph[node_name]
        op_type = node_info['op_type']
        
        # Get neighborhood context
        neighborhood = self._get_node_neighborhood(node_name, depth=2)
        neighborhood_ops = [self.node_graph[n]['op_type'] for n in neighborhood if n in self.node_graph]
        
        # Apply contextual inference patterns
        enhancement = self._apply_contextual_patterns(op_type, neighborhood_ops)
        
        if enhancement:
            return {
                'success': True,
                'enhancement': {
                    'semantic_type': enhancement['semantic_type'],
                    'confidence': 'low',  # Contextual inference has lower confidence
                    'primary_source': 'contextual_inference',
                    'context_pattern': enhancement.get('pattern', 'unknown')
                }
            }
        
        return {'success': False}
    
    def _aggregate_semantic_contexts(self, semantic_contexts: list[dict[str, Any]]) -> dict[str, Any]:
        """Aggregate multiple semantic contexts into a unified context."""
        if not semantic_contexts:
            return {}
        
        # Count semantic types
        semantic_type_counts = defaultdict(int)
        layer_ids = []
        components = []
        hf_modules = []
        
        for context in semantic_contexts:
            semantic_type = context.get('semantic_type')
            if semantic_type:
                semantic_type_counts[semantic_type] += 1
            
            layer_id = context.get('layer_id')
            if layer_id is not None:
                layer_ids.append(layer_id)
            
            component = context.get('component')
            if component:
                components.append(component)
            
            hf_module = context.get('hf_module_name')
            if hf_module:
                hf_modules.append(hf_module)
        
        # Find most common semantic type
        most_common_semantic_type = max(semantic_type_counts, key=semantic_type_counts.get) if semantic_type_counts else None
        
        # Aggregate other attributes
        aggregated = {
            'semantic_type': most_common_semantic_type,
            'semantic_confidence': len(semantic_contexts) / 10.0,  # More contexts = higher confidence
            'layer_ids': list(set(layer_ids)),
            'components': list(set(components)),
            'hf_modules': list(set(hf_modules))
        }
        
        return aggregated
    
    def _apply_inheritance_rules(self, inherited_semantics: dict[str, Any], op_type: str) -> dict[str, Any] | None:
        """Apply semantic inheritance rules based on operation type."""
        inherited_type = inherited_semantics.get('semantic_type')
        
        if not inherited_type:
            return None
        
        # Operation-specific inheritance rules
        if op_type == 'Constant':
            # Constants inherit semantic context from their usage
            return {
                'semantic_type': inherited_type,
                'pattern': 'constant_inheritance'
            }
        
        elif op_type in ['Add', 'Sub', 'Mul', 'Div']:
            # Arithmetic operations in specific contexts
            if inherited_type == 'attention':
                return {
                    'semantic_type': 'attention_arithmetic',
                    'pattern': 'attention_computation'
                }
            elif inherited_type == 'embedding':
                return {
                    'semantic_type': 'embedding_arithmetic',
                    'pattern': 'embedding_computation'
                }
            elif inherited_type == 'normalization':
                return {
                    'semantic_type': 'normalization_arithmetic', 
                    'pattern': 'normalization_computation'
                }
        
        elif op_type in ['Reshape', 'Transpose', 'Squeeze', 'Unsqueeze']:
            # Tensor manipulation inherits semantic context
            return {
                'semantic_type': f'{inherited_type}_manipulation',
                'pattern': 'tensor_manipulation_inheritance'
            }
        
        elif op_type in ['Cast', 'Convert']:
            # Type conversions maintain semantic context
            return {
                'semantic_type': inherited_type,
                'pattern': 'type_conversion_inheritance'
            }
        
        return None
    
    def _apply_forward_propagation_rules(self, forward_semantics: dict[str, Any], op_type: str) -> dict[str, Any] | None:
        """Apply forward semantic propagation rules."""
        forward_type = forward_semantics.get('semantic_type')
        
        if not forward_type:
            return None
        
        # Forward propagation rules
        if op_type == 'Constant':
            # Constants used in specific contexts
            return {
                'semantic_type': f'{forward_type}_constant',
                'pattern': 'constant_forward_propagation'
            }
        
        elif op_type in ['Shape', 'Size', 'Slice']:
            # Shape operations used in specific contexts
            return {
                'semantic_type': f'{forward_type}_introspection',
                'pattern': 'shape_forward_propagation'
            }
        
        return None
    
    def _apply_contextual_patterns(self, op_type: str, neighborhood_ops: list[str]) -> dict[str, Any] | None:
        """Apply contextual pattern recognition."""
        # Pattern: GELU activation (Div, Erf, Add pattern)
        if (op_type in ['Div', 'Erf', 'Add'] and 
            all(op in neighborhood_ops for op in ['Div', 'Erf', 'Add'])):
            return {
                'semantic_type': 'activation',
                'pattern': 'gelu_activation_pattern'
            }
        
        # Pattern: Layer normalization (ReduceMean, Sub, Pow, Add, Sqrt, Div)
        normalization_ops = ['ReduceMean', 'Sub', 'Pow', 'Add', 'Sqrt', 'Div']
        if op_type in normalization_ops and len(set(normalization_ops) & set(neighborhood_ops)) >= 3:
            return {
                'semantic_type': 'normalization',
                'pattern': 'layer_norm_pattern'
            }
        
        # Pattern: Attention masking (Expand, Sub, Mul, Add)
        attention_masking_ops = ['Expand', 'Sub', 'Mul', 'Add']
        if op_type in attention_masking_ops and len(set(attention_masking_ops) & set(neighborhood_ops)) >= 2:
            return {
                'semantic_type': 'attention_masking',
                'pattern': 'attention_mask_pattern'
            }
        
        return None
    
    def _get_node_neighborhood(self, node_name: str, depth: int = 1) -> set[str]:
        """Get neighborhood of nodes within specified depth."""
        if node_name not in self.node_graph:
            return set()
        
        visited = set()
        queue = deque([(node_name, 0)])
        neighborhood = set()
        
        while queue:
            current_node, current_depth = queue.popleft()
            
            if current_node in visited or current_depth > depth:
                continue
            
            visited.add(current_node)
            neighborhood.add(current_node)
            
            if current_node in self.node_graph:
                node_info = self.node_graph[current_node]
                
                # Add input producers
                for input_tensor in node_info['inputs']:
                    if input_tensor in self.output_graph:
                        producer = self.output_graph[input_tensor]
                        if producer not in visited:
                            queue.append((producer, current_depth + 1))
                
                # Add output consumers
                for output_tensor in node_info['outputs']:
                    if output_tensor in self.input_graph:
                        for consumer in self.input_graph[output_tensor]:
                            if consumer not in visited:
                                queue.append((consumer, current_depth + 1))
        
        return neighborhood
    
    def get_enhancement_statistics(self) -> dict[str, Any]:
        """Get statistics about semantic enhancements made."""
        original_unknown = sum(1 for mapping in self.semantic_mappings.values() 
                              if mapping.get('semantic_type') == 'unknown')
        enhanced_unknown = sum(1 for mapping in self.enhanced_mappings.values() 
                              if mapping.get('semantic_type') == 'unknown')
        
        original_low_confidence = sum(1 for mapping in self.semantic_mappings.values() 
                                     if mapping.get('confidence') == 'low')
        enhanced_low_confidence = sum(1 for mapping in self.enhanced_mappings.values() 
                                     if mapping.get('confidence') == 'low')
        
        enhanced_count = sum(1 for name, mapping in self.enhanced_mappings.items()
                            if mapping.get('primary_source', '').startswith('data_flow'))
        
        return {
            'total_nodes': len(self.semantic_mappings),
            'original_unknown': original_unknown,
            'enhanced_unknown': enhanced_unknown,
            'unknown_improvement': original_unknown - enhanced_unknown,
            'original_low_confidence': original_low_confidence,
            'enhanced_low_confidence': enhanced_low_confidence,
            'confidence_improvement': original_low_confidence - enhanced_low_confidence,
            'nodes_enhanced': enhanced_count,
            'enhancement_rate': enhanced_count / len(self.semantic_mappings) * 100
        }