"""
Enhanced HTP Hierarchy Exporter with Proper Trace Map Utilization

This enhanced version properly captures and uses PyTorch's built-in trace mapping
to achieve better module-to-operation mapping with reduced cross-layer contamination.

Key improvements:
1. Captures PyTorch's internal trace map during ONNX export
2. Uses the captured trace for direct operation-to-module mapping
3. Reduces reliance on parameter-based inference
4. Better handles auxiliary operations with proper spatial context
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.onnx
import onnx
from typing import Dict, List, Any, Optional, Tuple, Set
from collections import defaultdict, deque
from dataclasses import dataclass
import weakref
import warnings

# Import base HTP exporter
from .htp_hierarchy_exporter import HierarchyExporter, OperationConfig


class EnhancedHTPExporter(HierarchyExporter):
    """Enhanced HTP exporter that properly utilizes PyTorch's trace mapping."""
    
    def __init__(self, strategy: str = "htp_enhanced"):
        super().__init__(strategy)
        self._captured_trace_map = {}
        self._operation_to_module_map = {}
        self._trace_capture_enabled = False
        self._original_trace_fn = None
        
    def export(
        self,
        model: torch.nn.Module,
        example_inputs,
        output_path: str,
        input_specs: Optional[List[tuple]] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        """Enhanced export with proper trace capture."""
        
        # Reset state
        self._reset_state()
        self._model = model
        
        # Step 1: Setup builtin module tracking
        self._setup_builtin_module_tracking(model)
        
        # Step 2: Register forward hooks for stack-based context
        self._register_hooks(model)
        
        # Step 3: Setup enhanced trace capture
        self._setup_enhanced_trace_capture()
        
        try:
            # Step 4: Trace model execution to populate operation context
            with torch.no_grad():
                self._trace_model_execution(model, example_inputs)
            
            # Step 5: Export to ONNX with trace capture enabled
            self._trace_capture_enabled = True
            self._export_to_onnx(model, example_inputs, output_path, **kwargs)
            self._trace_capture_enabled = False
            
            # Step 6: Load and analyze ONNX model
            onnx_model = onnx.load(output_path)
            
            # Step 7: Build enhanced tag mapping using captured traces
            self._build_enhanced_tag_mapping(onnx_model)
            
            # Step 8: Forward propagate tags
            tensor_producers = self._build_tensor_producer_map(onnx_model)
            self._forward_propagate_tags_htp(onnx_model, tensor_producers)
            
            # Step 9: Build tensor tagging
            self._build_tensor_tags(onnx_model)
            
            # Step 10: Ensure complete coverage
            self._ensure_complete_coverage(onnx_model)
            
            # Step 11: Inject tags into ONNX
            self._inject_htp_tags_into_onnx(output_path, onnx_model)
            
            return {
                "output_path": output_path,
                "strategy": self.strategy,
                "total_operations": len(self._tag_mapping),
                "tagged_operations": len([op for op in self._tag_mapping.values() if op.get("tags", [])]),
                "captured_traces": len(self._captured_trace_map),
                "direct_mappings": len(self._operation_to_module_map),
            }
            
        finally:
            # Cleanup
            self._cleanup_enhanced_trace_capture()
            self._cleanup_builtin_module_tracking()
            self._remove_hooks()
    
    def _setup_enhanced_trace_capture(self):
        """Setup enhanced trace capture to intercept PyTorch's internal tracing."""
        
        # Monkey-patch PyTorch's trace recording to capture module context
        import torch.jit._trace
        
        # Store original function
        self._original_trace_fn = torch.jit._trace._record_trace_map
        
        # Create our enhanced version
        def enhanced_record_trace_map(graph, module_map, *args, **kwargs):
            """Enhanced trace recording that captures module context."""
            
            # Call original function
            result = self._original_trace_fn(graph, module_map, *args, **kwargs) if self._original_trace_fn else None
            
            # Capture the trace map if we're in export mode
            if self._trace_capture_enabled and hasattr(graph, '_raw_module_map'):
                self._captured_trace_map = graph._raw_module_map.copy()
                
                # Also process the map to create direct operation mappings
                self._process_captured_trace_map(graph)
            
            return result
        
        # Replace with our version
        torch.jit._trace._record_trace_map = enhanced_record_trace_map
        
        # Also setup hook for operation recording
        self._setup_operation_recording_hook()
    
    def _setup_operation_recording_hook(self):
        """Setup hook to record operations as they're created in the graph."""
        
        # This is a more direct approach - hook into the graph building process
        import torch._C
        
        # Store original graph op creation
        if hasattr(torch._C, '_create_graph_op'):
            self._original_create_op = torch._C._create_graph_op
            
            def enhanced_create_op(graph, op_name, *args, **kwargs):
                """Enhanced operation creation that records module context."""
                
                # Create the operation
                op = self._original_create_op(graph, op_name, *args, **kwargs)
                
                # If we're capturing and have module context, record it
                if self._trace_capture_enabled and self._current_module_context:
                    op_id = id(op)
                    module_name = self._builtin_module_map.get(self._current_module_context, "")
                    if module_name:
                        self._operation_to_module_map[op_id] = {
                            'module_name': module_name,
                            'module_tag': self._build_module_tag_from_name(module_name),
                            'op_type': op_name
                        }
                
                return op
            
            torch._C._create_graph_op = enhanced_create_op
    
    def _process_captured_trace_map(self, graph):
        """Process the captured trace map to extract module mappings."""
        
        if not hasattr(graph, 'nodes'):
            return
            
        # Iterate through graph nodes and map them to modules
        for node in graph.nodes():
            if hasattr(node, 'sourceRange') and hasattr(node, 'kind'):
                # Try to get module context from source range
                source_range = node.sourceRange()
                if source_range and hasattr(source_range, 'module'):
                    module = source_range.module
                    if module in self._builtin_module_map:
                        module_name = self._builtin_module_map[module]
                        node_id = str(node)  # or node.debugName() if available
                        
                        self._operation_to_module_map[node_id] = {
                            'module_name': module_name,
                            'module_tag': self._build_module_tag_from_name(module_name),
                            'op_type': str(node.kind())
                        }
    
    def _build_module_tag_from_name(self, module_name: str) -> str:
        """Build hierarchical tag from module name."""
        
        # Convert dot notation to hierarchical path
        # e.g., "encoder.layer.0.attention.self.query" -> "/BertModel/Encoder/Layer.0/Attention/Self/Query"
        
        if not module_name:
            return ""
            
        parts = module_name.split('.')
        tag_parts = []
        
        for part in parts:
            # Handle numeric indices
            if part.isdigit():
                if tag_parts:
                    tag_parts[-1] = f"{tag_parts[-1]}.{part}"
            else:
                # Capitalize first letter
                tag_parts.append(part.capitalize())
        
        # Get root model class
        root_class = self._model.__class__.__name__ if self._model else "Model"
        
        return f"/{root_class}/" + "/".join(tag_parts)
    
    def _build_enhanced_tag_mapping(self, onnx_model):
        """Build enhanced tag mapping using captured trace information."""
        
        # First, build basic mapping
        super()._build_tag_mapping_from_onnx(onnx_model)
        
        # Now enhance with captured trace information
        if self._operation_to_module_map:
            print(f"Enhanced mapping: Using {len(self._operation_to_module_map)} captured traces")
            
            # Try to match ONNX nodes to captured operations
            for node in onnx_model.graph.node:
                node_name = node.name or f"{node.op_type}_{len(self._tag_mapping)}"
                
                # Try multiple matching strategies
                
                # Strategy 1: Direct node name match
                if node_name in self._operation_to_module_map:
                    mapping = self._operation_to_module_map[node_name]
                    self._tag_mapping[node_name]['tags'] = [mapping['module_tag']]
                    self._tag_mapping[node_name]['trace_source'] = 'direct_match'
                    continue
                
                # Strategy 2: Match by operation type and order
                op_type_matches = [
                    (k, v) for k, v in self._operation_to_module_map.items()
                    if v['op_type'] == node.op_type
                ]
                
                if op_type_matches:
                    # Use order-based matching
                    op_index = len([n for n in onnx_model.graph.node[:onnx_model.graph.node.index(node)] 
                                   if n.op_type == node.op_type])
                    
                    if op_index < len(op_type_matches):
                        _, mapping = op_type_matches[op_index]
                        self._tag_mapping[node_name]['tags'] = [mapping['module_tag']]
                        self._tag_mapping[node_name]['trace_source'] = 'type_order_match'
                        continue
                
                # Strategy 3: Use spatial locality from captured traces
                # (This would use producer/consumer relationships to infer module context)
                
        # If no captured traces or no matches, fall back to original parameter-based approach
        # But with lower confidence
        untagged_count = len([n for n in self._tag_mapping.values() if not n.get('tags')])
        if untagged_count > 0:
            print(f"Enhanced mapping: {untagged_count} operations still need parameter-based tagging")
    
    def _cleanup_enhanced_trace_capture(self):
        """Cleanup enhanced trace capture."""
        
        import torch.jit._trace
        
        # Restore original function
        if self._original_trace_fn:
            torch.jit._trace._record_trace_map = self._original_trace_fn
            self._original_trace_fn = None
        
        # Restore original op creation if we hooked it
        if hasattr(self, '_original_create_op'):
            import torch._C
            torch._C._create_graph_op = self._original_create_op
        
        # Clear captured data
        self._captured_trace_map.clear()
        self._operation_to_module_map.clear()
    
    def _build_tensor_producer_map(self, onnx_model) -> Dict[str, str]:
        """Build mapping from tensor names to their producer operations."""
        
        tensor_producers = {}
        for node in onnx_model.graph.node:
            node_name = node.name or f"{node.op_type}_{len(self._tag_mapping)}"
            for output in node.output:
                tensor_producers[output] = node_name
        
        return tensor_producers
    
    def get_trace_statistics(self) -> Dict[str, Any]:
        """Get statistics about trace capture effectiveness."""
        
        total_ops = len(self._tag_mapping)
        traced_ops = len([op for op in self._tag_mapping.values() if op.get('trace_source')])
        param_based_ops = len([op for op in self._tag_mapping.values() 
                              if op.get('tags') and not op.get('trace_source')])
        untagged_ops = len([op for op in self._tag_mapping.values() if not op.get('tags')])
        
        return {
            'total_operations': total_ops,
            'trace_captured_operations': traced_ops,
            'parameter_based_operations': param_based_ops,
            'untagged_operations': untagged_ops,
            'trace_capture_rate': traced_ops / total_ops if total_ops > 0 else 0,
            'coverage_rate': (total_ops - untagged_ops) / total_ops if total_ops > 0 else 0,
        }


def export_with_enhanced_htp(
    model: torch.nn.Module,
    example_inputs: Any,
    output_path: str,
    **kwargs
) -> Dict[str, Any]:
    """Convenience function to export using enhanced HTP strategy."""
    
    exporter = EnhancedHTPExporter()
    result = exporter.export(model, example_inputs, output_path, **kwargs)
    
    # Print statistics
    stats = exporter.get_trace_statistics()
    print(f"\nEnhanced HTP Export Statistics:")
    print(f"  Total operations: {stats['total_operations']}")
    print(f"  Trace-captured: {stats['trace_captured_operations']} ({stats['trace_capture_rate']:.1%})")
    print(f"  Parameter-based: {stats['parameter_based_operations']}")
    print(f"  Untagged: {stats['untagged_operations']}")
    print(f"  Coverage: {stats['coverage_rate']:.1%}")
    
    return result