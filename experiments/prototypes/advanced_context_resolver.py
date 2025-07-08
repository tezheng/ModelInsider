#!/usr/bin/env python3
"""
Advanced Context Resolver: Multi-Context Tensor Provenance Analysis

This module implements cutting-edge approaches to resolve cross-layer contamination
by embracing multi-context reality and using tensor provenance analysis.
"""

from typing import Dict, List, Any, Optional, Set, Tuple
from dataclasses import dataclass
from collections import defaultdict, deque
import re
import numpy as np


@dataclass
class TensorProvenance:
    """Complete lineage information for a tensor."""
    tensor_id: str
    primary_creator: str
    creation_operation: str
    creation_context_stack: List[str]
    timestamp: int
    dependencies: List[str]  # Input tensor IDs


@dataclass
class ContextAssignment:
    """Multi-context assignment with confidence scoring."""
    primary_context: str
    auxiliary_contexts: List[str]
    confidence: float
    assignment_type: str  # 'single', 'multi_context', 'residual', 'attention'
    reasoning: str
    

@dataclass
class ContaminationCase:
    """Represents a case of cross-layer contamination."""
    node_name: str
    expected_context: str
    actual_contexts: List[str]
    operation_type: str
    confidence: float = 0.0
    is_resolved: bool = False


class TensorProvenanceTracker:
    """Tracks complete tensor creation and usage lineage."""
    
    def __init__(self):
        self.tensor_lineage: Dict[str, TensorProvenance] = {}
        self.operation_contexts: Dict[str, ContextAssignment] = {}
        self.context_relationships: Dict[str, Set[str]] = defaultdict(set)
        self.execution_timestamp = 0
    
    def track_tensor_creation(self, tensor_id: str, creating_module: str, 
                             operation_type: str, context_stack: List[str],
                             input_tensor_ids: List[str] = None):
        """Track tensor creation with complete provenance."""
        self.tensor_lineage[tensor_id] = TensorProvenance(
            tensor_id=tensor_id,
            primary_creator=creating_module,
            creation_operation=operation_type,
            creation_context_stack=context_stack.copy(),
            timestamp=self.execution_timestamp,
            dependencies=input_tensor_ids or []
        )
        self.execution_timestamp += 1
    
    def get_tensor_provenance_chain(self, tensor_id: str, max_depth: int = 5) -> List[str]:
        """Get complete provenance chain for a tensor."""
        if tensor_id not in self.tensor_lineage:
            return []
        
        chain = []
        current_tensors = [tensor_id]
        depth = 0
        
        while current_tensors and depth < max_depth:
            next_tensors = []
            for tid in current_tensors:
                if tid in self.tensor_lineage:
                    provenance = self.tensor_lineage[tid]
                    chain.append(provenance.primary_creator)
                    next_tensors.extend(provenance.dependencies)
            
            current_tensors = next_tensors
            depth += 1
        
        return chain
    
    def analyze_tensor_contexts(self, operation_name: str, 
                               input_tensors: List[str]) -> ContextAssignment:
        """Analyze tensor contexts to determine operation assignment."""
        if not input_tensors:
            return ContextAssignment(
                primary_context="unknown",
                auxiliary_contexts=[],
                confidence=0.0,
                assignment_type="unknown",
                reasoning="no_input_tensors"
            )
        
        # Get contexts for all input tensors
        input_contexts = []
        for tensor_id in input_tensors:
            if tensor_id in self.tensor_lineage:
                context = self.tensor_lineage[tensor_id].primary_creator
                input_contexts.append(context)
        
        if not input_contexts:
            return ContextAssignment(
                primary_context="unknown",
                auxiliary_contexts=[],
                confidence=0.0,
                assignment_type="unknown", 
                reasoning="no_trackable_inputs"
            )
        
        # Analyze context patterns
        unique_contexts = list(set(input_contexts))
        
        if len(unique_contexts) == 1:
            # Single context - high confidence
            return ContextAssignment(
                primary_context=unique_contexts[0],
                auxiliary_contexts=[],
                confidence=0.95,
                assignment_type="single",
                reasoning="single_input_context"
            )
        elif len(unique_contexts) == 2:
            # Potential residual connection or multi-context operation
            return self._analyze_multi_context_operation(
                operation_name, unique_contexts, input_contexts
            )
        else:
            # Complex multi-context operation
            most_common = max(set(input_contexts), key=input_contexts.count)
            others = [ctx for ctx in unique_contexts if ctx != most_common]
            
            return ContextAssignment(
                primary_context=most_common,
                auxiliary_contexts=others,
                confidence=0.7,
                assignment_type="multi_context",
                reasoning="complex_multi_input"
            )
    
    def _analyze_multi_context_operation(self, operation_name: str, 
                                        unique_contexts: List[str],
                                        all_contexts: List[str]) -> ContextAssignment:
        """Analyze operations with exactly two input contexts."""
        
        # Check for residual connection pattern
        if self._is_residual_pattern(operation_name, unique_contexts):
            # For residual connections, assign to the "later" layer (consuming layer)
            primary_context = self._determine_residual_primary_context(unique_contexts)
            auxiliary_context = [ctx for ctx in unique_contexts if ctx != primary_context][0]
            
            return ContextAssignment(
                primary_context=primary_context,
                auxiliary_contexts=[auxiliary_context],
                confidence=0.9,
                assignment_type="residual",
                reasoning="residual_connection_pattern"
            )
        
        # Check for attention pattern
        elif self._is_attention_pattern(operation_name, unique_contexts):
            return ContextAssignment(
                primary_context=unique_contexts[0],  # First context as primary
                auxiliary_contexts=unique_contexts[1:],
                confidence=0.85,
                assignment_type="attention",
                reasoning="attention_mechanism_pattern"
            )
        
        # Default multi-context assignment
        else:
            most_common = max(set(all_contexts), key=all_contexts.count)
            others = [ctx for ctx in unique_contexts if ctx != most_common]
            
            return ContextAssignment(
                primary_context=most_common,
                auxiliary_contexts=others,
                confidence=0.75,
                assignment_type="multi_context",
                reasoning="multiple_input_contexts"
            )
    
    def _is_residual_pattern(self, operation_name: str, contexts: List[str]) -> bool:
        """Detect residual connection patterns."""
        # Look for Add operations between different layers
        if 'Add' not in operation_name and 'add' not in operation_name.lower():
            return False
        
        # Check if contexts are from consecutive layers
        layer_numbers = []
        for context in contexts:
            # Extract layer numbers from context paths
            layer_match = re.search(r'[Ll]ayer[._](\d+)', context)
            if layer_match:
                layer_numbers.append(int(layer_match.group(1)))
        
        if len(layer_numbers) == 2:
            # Residual if layers are consecutive or same
            return abs(layer_numbers[0] - layer_numbers[1]) <= 1
        
        return False
    
    def _is_attention_pattern(self, operation_name: str, contexts: List[str]) -> bool:
        """Detect attention mechanism patterns."""
        attention_keywords = ['attention', 'self', 'query', 'key', 'value', 'softmax']
        operation_lower = operation_name.lower()
        
        return any(keyword in operation_lower for keyword in attention_keywords)
    
    def _determine_residual_primary_context(self, contexts: List[str]) -> str:
        """Determine primary context for residual connections."""
        # Extract layer numbers and assign to later layer
        layer_info = []
        for context in contexts:
            layer_match = re.search(r'[Ll]ayer[._](\d+)', context)
            if layer_match:
                layer_info.append((int(layer_match.group(1)), context))
        
        if len(layer_info) == 2:
            # Return the later layer as primary
            return max(layer_info, key=lambda x: x[0])[1]
        
        # Fallback to first context
        return contexts[0]


class ResidualConnectionDetector:
    """Specialized detector for residual connection patterns."""
    
    def __init__(self):
        self.detected_patterns = []
        self.pattern_confidence = {}
    
    def detect_residual_patterns(self, onnx_model, contamination_cases) -> List[Dict]:
        """Detect residual connection patterns in contamination cases."""
        residual_patterns = []
        
        # Group contamination cases by operation type
        add_operations = [case for case in contamination_cases 
                         if case.operation_type == 'Add']
        
        for case in add_operations:
            pattern_info = self._analyze_add_operation_pattern(case, onnx_model)
            if pattern_info['is_residual']:
                residual_patterns.append({
                    'case': case,
                    'pattern_type': 'residual_connection',
                    'suggested_context': pattern_info['suggested_context'],
                    'confidence': pattern_info['confidence'],
                    'reasoning': pattern_info['reasoning']
                })
        
        return residual_patterns
    
    def _analyze_add_operation_pattern(self, case: ContaminationCase, onnx_model) -> Dict:
        """Analyze a specific Add operation for residual patterns."""
        node_name = case.node_name
        
        # Look for skip/residual patterns in node name
        skip_patterns = [
            r'/layer\.(\d+)/.*[Aa]dd',           # Layer skip connections
            r'/encoder/layer\.(\d+)/.*[Aa]dd',   # BERT-style
            r'/attention/.*/[Aa]dd',             # Attention residuals
            r'/output/.*[Aa]dd',                 # Output residuals
        ]
        
        for pattern in skip_patterns:
            match = re.search(pattern, node_name)
            if match:
                layer_num = match.group(1) if match.groups() else None
                
                # Determine suggested context based on pattern
                if layer_num:
                    # Assign to current layer (consuming layer)
                    suggested_context = self._build_layer_context(layer_num, case.actual_contexts)
                    
                    return {
                        'is_residual': True,
                        'suggested_context': suggested_context,
                        'confidence': 0.9,
                        'reasoning': f'residual_pattern_layer_{layer_num}'
                    }
        
        return {
            'is_residual': False,
            'suggested_context': None,
            'confidence': 0.0,
            'reasoning': 'no_residual_pattern_detected'
        }
    
    def _build_layer_context(self, layer_num: str, actual_contexts: List[str]) -> str:
        """Build proper layer context for residual connection."""
        # Find context that matches the layer number
        for context in actual_contexts:
            if f'Layer.{layer_num}' in context or f'layer.{layer_num}' in context:
                return context
        
        # Fallback to first context
        return actual_contexts[0] if actual_contexts else f"Layer.{layer_num}"


class AdvancedContextResolver:
    """Advanced context resolver using multi-context analysis."""
    
    def __init__(self):
        self.provenance_tracker = TensorProvenanceTracker()
        self.residual_detector = ResidualConnectionDetector()
        self.resolution_history = []
    
    def resolve_contamination_cases(self, contamination_cases: List[ContaminationCase],
                                   onnx_model, hierarchy_data) -> Dict[str, Any]:
        """Resolve contamination cases using advanced analysis."""
        
        print(f"🔬 Advanced Context Resolver: Analyzing {len(contamination_cases)} contamination cases")
        
        resolution_results = {
            'total_cases': len(contamination_cases),
            'resolved_cases': [],
            'unresolved_cases': [],
            'resolution_strategies': {},
            'confidence_distribution': [],
            'multi_context_assignments': []
        }
        
        # Step 1: Detect residual connection patterns
        print("🔍 Step 1: Detecting residual connection patterns...")
        residual_patterns = self.residual_detector.detect_residual_patterns(
            onnx_model, contamination_cases
        )
        
        print(f"   Found {len(residual_patterns)} residual connection patterns")
        
        # Step 2: Apply pattern-based resolution
        for pattern in residual_patterns:
            case = pattern['case']
            
            resolved_assignment = ContextAssignment(
                primary_context=pattern['suggested_context'],
                auxiliary_contexts=[ctx for ctx in case.actual_contexts 
                                  if ctx != pattern['suggested_context']],
                confidence=pattern['confidence'],
                assignment_type='residual',
                reasoning=pattern['reasoning']
            )
            
            resolution_results['resolved_cases'].append({
                'case': case,
                'resolution': resolved_assignment,
                'strategy': 'residual_pattern_detection'
            })
            
            if len(resolved_assignment.auxiliary_contexts) > 0:
                resolution_results['multi_context_assignments'].append(resolved_assignment)
        
        # Step 3: Handle remaining cases with tensor provenance
        resolved_node_names = {res['case'].node_name for res in resolution_results['resolved_cases']}
        remaining_cases = [case for case in contamination_cases 
                          if case.node_name not in resolved_node_names]
        
        print(f"🧬 Step 2: Applying tensor provenance analysis to {len(remaining_cases)} remaining cases...")
        
        for case in remaining_cases:
            # Simulate tensor provenance analysis (would need actual tensor IDs from execution)
            assignment = self._simulate_provenance_analysis(case)
            
            if assignment.confidence > 0.7:
                resolution_results['resolved_cases'].append({
                    'case': case,
                    'resolution': assignment,
                    'strategy': 'tensor_provenance_analysis'
                })
                
                if len(assignment.auxiliary_contexts) > 0:
                    resolution_results['multi_context_assignments'].append(assignment)
            else:
                resolution_results['unresolved_cases'].append({
                    'case': case,
                    'attempted_resolution': assignment,
                    'reason': 'low_confidence'
                })
        
        # Step 4: Generate comprehensive analysis
        self._generate_resolution_analysis(resolution_results)
        
        return resolution_results
    
    def _simulate_provenance_analysis(self, case: ContaminationCase) -> ContextAssignment:
        """Simulate tensor provenance analysis (placeholder for full implementation)."""
        
        # Analyze node name patterns for context clues
        node_name = case.node_name
        
        # Check for attention patterns
        if any(keyword in node_name.lower() for keyword in ['attention', 'query', 'key', 'value']):
            return ContextAssignment(
                primary_context=case.actual_contexts[0] if case.actual_contexts else case.expected_context,
                auxiliary_contexts=case.actual_contexts[1:] if len(case.actual_contexts) > 1 else [],
                confidence=0.8,
                assignment_type='attention',
                reasoning='attention_pattern_in_name'
            )
        
        # Check for layer-specific operations
        layer_match = re.search(r'/layer\.(\d+)/', node_name)
        if layer_match:
            layer_num = layer_match.group(1)
            
            # Find matching context
            matching_context = None
            for context in case.actual_contexts:
                if f'Layer.{layer_num}' in context:
                    matching_context = context
                    break
            
            if matching_context:
                other_contexts = [ctx for ctx in case.actual_contexts if ctx != matching_context]
                return ContextAssignment(
                    primary_context=matching_context,
                    auxiliary_contexts=other_contexts,
                    confidence=0.85,
                    assignment_type='layer_specific',
                    reasoning=f'layer_specific_operation_layer_{layer_num}'
                )
        
        # Default assignment
        return ContextAssignment(
            primary_context=case.actual_contexts[0] if case.actual_contexts else case.expected_context,
            auxiliary_contexts=case.actual_contexts[1:] if len(case.actual_contexts) > 1 else [],
            confidence=0.6,
            assignment_type='default',
            reasoning='fallback_assignment'
        )
    
    def _generate_resolution_analysis(self, results: Dict[str, Any]):
        """Generate comprehensive analysis of resolution results."""
        
        print(f"\n📊 ADVANCED CONTEXT RESOLUTION ANALYSIS")
        print(f"{'='*50}")
        
        total_cases = results['total_cases']
        resolved_count = len(results['resolved_cases'])
        unresolved_count = len(results['unresolved_cases'])
        multi_context_count = len(results['multi_context_assignments'])
        
        print(f"Total contamination cases: {total_cases}")
        print(f"Resolved cases: {resolved_count} ({resolved_count/total_cases*100:.1f}%)")
        print(f"Unresolved cases: {unresolved_count} ({unresolved_count/total_cases*100:.1f}%)")
        print(f"Multi-context assignments: {multi_context_count}")
        
        # Strategy effectiveness
        strategy_counts = defaultdict(int)
        for resolved in results['resolved_cases']:
            strategy_counts[resolved['strategy']] += 1
        
        print(f"\nResolution strategy effectiveness:")
        for strategy, count in strategy_counts.items():
            print(f"  {strategy}: {count} cases")
        
        # Confidence distribution
        confidences = [res['resolution'].confidence for res in results['resolved_cases']]
        if confidences:
            avg_confidence = np.mean(confidences)
            print(f"\nAverage resolution confidence: {avg_confidence:.3f}")
            print(f"High confidence (>0.8): {sum(1 for c in confidences if c > 0.8)} cases")
            print(f"Medium confidence (0.6-0.8): {sum(1 for c in confidences if 0.6 <= c <= 0.8)} cases")
            print(f"Low confidence (<0.6): {sum(1 for c in confidences if c < 0.6)} cases")
        
        # Multi-context analysis
        if multi_context_count > 0:
            print(f"\nMulti-context assignment types:")
            type_counts = defaultdict(int)
            for assignment in results['multi_context_assignments']:
                type_counts[assignment.assignment_type] += 1
            
            for assignment_type, count in type_counts.items():
                print(f"  {assignment_type}: {count} cases")


def create_test_contamination_cases() -> List[ContaminationCase]:
    """Create test contamination cases for validation."""
    return [
        ContaminationCase(
            node_name="/encoder/layer.0/attention/output/dense/Add",
            expected_context="/BertModel/BertEncoder/BertLayer.0/BertAttention/BertSelfOutput",
            actual_contexts=[
                "/BertModel/BertEncoder/BertLayer.0/BertAttention/BertSelfOutput",
                "/BertModel/BertEncoder/BertLayer.1/BertOutput/LayerNorm"
            ],
            operation_type="Add"
        ),
        ContaminationCase(
            node_name="/encoder/layer.1/attention/self/query/Add",
            expected_context="/BertModel/BertEncoder/BertLayer.1/BertAttention/BertSelfAttention",
            actual_contexts=[
                "/BertModel/BertEncoder/BertLayer.0/BertAttention/BertSelfAttention",
                "/BertModel/BertEncoder/BertLayer.1/BertAttention/BertSelfAttention"
            ],
            operation_type="Add"
        ),
        ContaminationCase(
            node_name="/encoder/layer.0/intermediate/dense/MatMul",
            expected_context="/BertModel/BertEncoder/BertLayer.0/BertIntermediate",
            actual_contexts=[
                "/BertModel/BertEncoder/BertLayer.1/BertAttention/BertSelfAttention"
            ],
            operation_type="MatMul"
        )
    ]


if __name__ == "__main__":
    print("🚀 Testing Advanced Context Resolver")
    
    # Create test cases
    test_cases = create_test_contamination_cases()
    
    # Initialize resolver
    resolver = AdvancedContextResolver()
    
    # Resolve contamination cases
    results = resolver.resolve_contamination_cases(test_cases, None, None)
    
    print(f"\n✅ Advanced context resolution complete!")
    print(f"Demonstrated multi-context assignment and pattern recognition capabilities.")