#!/usr/bin/env python3
"""
Fixed Universal Hierarchy Exporter that properly uses _captured_trace_map

This fixed version actually uses the captured trace module map from PyTorch
to improve operation tagging accuracy.
"""

import torch
import torch.nn as nn
from typing import Dict, List, Any, Optional, Tuple, Union
import json
from pathlib import Path
import onnx
import tempfile
import time
import logging
from collections import defaultdict

# Import the original exporter
from .universal_hierarchy_exporter import UniversalHierarchyExporter

logger = logging.getLogger(__name__)


class UniversalHierarchyExporterFixed(UniversalHierarchyExporter):
    """
    Fixed version that properly utilizes the captured trace map for better tagging accuracy.
    """
    
    def _find_best_tag_for_operation(self, node) -> str:
        """
        Enhanced version that uses _captured_trace_map for better accuracy.
        
        This method now:
        1. First tries to use the captured trace map for direct module mapping
        2. Falls back to path-based matching if trace map doesn't help
        3. Uses operation context as final fallback
        """
        node_name = node.name or f"{node.op_type}_{id(node)}"
        
        # NEW: Strategy 0 - Use captured trace map for direct mapping
        if hasattr(self, '_captured_trace_map') and self._captured_trace_map:
            # The trace map contains enhanced scope names in format:
            # ClassName::__module.path.to.module
            
            # Extract the operation path from node name
            op_path = node_name.lstrip('/')
            
            # Try to find a matching module in the captured trace map
            best_match = None
            best_score = 0
            
            for module, trace_name in self._captured_trace_map.items():
                if not trace_name:
                    continue
                
                # Parse the trace name (e.g., "BertSelfAttention::__module.bert.encoder.layer.0.attention.self")
                if '::' in trace_name:
                    class_name, module_path = trace_name.split('::', 1)
                    module_path = module_path.replace('__module.', '')
                    
                    # Convert module path to operation path format
                    # e.g., bert.encoder.layer.0.attention.self -> encoder/layer.0/attention/self
                    module_op_path = module_path.replace('.', '/')
                    
                    # Check if this module path matches the operation path
                    if module_op_path in op_path.lower():
                        score = len(module_op_path)
                        
                        # Bonus points for exact class name match
                        if class_name.lower() in op_path.lower():
                            score += len(class_name)
                        
                        if score > best_score:
                            best_score = score
                            # Build the hierarchy tag from the module path
                            best_match = self._build_tag_from_trace_name(trace_name)
            
            if best_match:
                if self.verbose:
                    logger.debug(f"Found direct trace map match for {node_name}: {best_match}")
                return best_match
        
        # Original Strategy 1: Match operation path with module paths
        op_path = node_name.lstrip('/')
        
        # Remove operation type from the end
        path_parts = op_path.split('/')
        if path_parts and path_parts[-1] in ['Gather', 'MatMul', 'Add', 'LayerNormalization', 
                                              'Gemm', 'Tanh', 'Softmax', 'Div', 'Mul', 'Sub',
                                              'Transpose', 'Reshape', 'Constant', 'Shape', 
                                              'Unsqueeze', 'Concat', 'Slice', 'Where', 'Cast',
                                              'Expand', 'Equal', 'ConstantOfShape', 'Sqrt', 'Erf']:
            op_path = '/'.join(path_parts[:-1])
        
        # Try to find the best matching module based on path similarity
        best_match = None
        best_score = 0
        
        for module_name, context in self._operation_context.items():
            if not context.get("tag"):
                continue
                
            # Calculate match score based on common path components
            module_path = module_name.lower().replace('.', '/')
            op_path_lower = op_path.lower()
            
            # Check if operation path contains module path components
            if module_path in op_path_lower:
                score = len(module_path)
                if score > best_score:
                    best_score = score
                    best_match = context["tag"]
            
            # Also check individual components
            module_parts = module_path.split('/')
            op_parts = op_path_lower.split('/')
            common_parts = sum(1 for mp in module_parts if mp in op_parts)
            if common_parts > best_score:
                best_score = common_parts
                best_match = context["tag"]
        
        if best_match:
            return best_match
        
        # Strategy 2: Use operation context if no path match
        for context in reversed(list(self._operation_context.values())):
            if context.get("tag"):
                return context["tag"]
        
        # Final fallback to root tag
        return f"/{self._get_root_class_name()}" if self._module_hierarchy else ""
    
    def _build_tag_from_trace_name(self, trace_name: str) -> str:
        """
        Build a hierarchy tag from PyTorch's trace name format.
        
        Example:
        Input: "BertSelfAttention::__module.bert.encoder.layer.0.attention.self"
        Output: "/BertForSequenceClassification/BertModel/BertEncoder/BertLayer.0/BertAttention/BertSelfAttention"
        """
        if '::' not in trace_name:
            return ""
        
        class_name, module_path = trace_name.split('::', 1)
        module_path = module_path.replace('__module.', '')
        
        # Split the path and build hierarchy
        parts = module_path.split('.')
        
        # Try to find this module in our hierarchy to get the proper tag
        full_path = f"__module.{module_path}"
        if full_path in self._module_hierarchy:
            return self._module_hierarchy[full_path].get('expected_tag', '')
        
        # If not found in hierarchy, build a tag from the trace name
        # This is a fallback that constructs a reasonable tag
        tag_parts = []
        
        # Add root class name
        root_class = self._get_root_class_name()
        tag_parts.append(root_class)
        
        # Process each part of the path
        for i, part in enumerate(parts):
            if part.isdigit():
                # Instance number - append to previous part
                if tag_parts:
                    tag_parts[-1] = f"{tag_parts[-1]}.{part}"
            else:
                # Convert to proper case
                tag_part = ''.join(word.capitalize() for word in part.split('_'))
                tag_parts.append(tag_part)
        
        return '/' + '/'.join(tag_parts)
    
    def _map_operations_post_export(self, output_path: str) -> None:
        """
        Enhanced version that reports trace map usage statistics.
        """
        # Call parent implementation
        super()._map_operations_post_export(output_path)
        
        # Add trace map statistics
        if hasattr(self, '_captured_trace_map') and self._captured_trace_map:
            if self.verbose:
                logger.info(f"Captured trace map contains {len(self._captured_trace_map)} module mappings")
                
                # Show a few examples of the trace map
                examples = list(self._captured_trace_map.items())[:3]
                for module, trace_name in examples:
                    logger.debug(f"  {trace_name}")
    
    def get_trace_map_stats(self) -> Dict[str, Any]:
        """Get statistics about trace map usage."""
        stats = {
            'trace_map_size': len(self._captured_trace_map) if hasattr(self, '_captured_trace_map') else 0,
            'operation_context_size': len(self._operation_context),
            'tagged_operations': len(self._operation_tags),
            'has_trace_map': hasattr(self, '_captured_trace_map') and bool(self._captured_trace_map)
        }
        
        # Sample of trace map entries
        if hasattr(self, '_captured_trace_map') and self._captured_trace_map:
            stats['trace_map_sample'] = [
                trace_name for _, trace_name in list(self._captured_trace_map.items())[:5]
            ]
        
        return stats


def export_with_fixed_trace_map(
    model: nn.Module,
    args: Tuple[torch.Tensor, ...],
    output_path: str,
    **kwargs
) -> Dict[str, Any]:
    """
    Convenience function to export using the fixed exporter.
    """
    exporter = UniversalHierarchyExporterFixed(
        torch_nn_exceptions=kwargs.pop('torch_nn_exceptions', ['LayerNorm', 'Embedding']),
        verbose=kwargs.pop('verbose', True)
    )
    
    result = exporter.export(model, args, output_path, **kwargs)
    
    # Add trace map statistics to result
    trace_stats = exporter.get_trace_map_stats()
    result['trace_map_stats'] = trace_stats
    
    if exporter.verbose:
        print(f"\nTrace Map Statistics:")
        print(f"  Captured modules: {trace_stats['trace_map_size']}")
        print(f"  Has trace map: {trace_stats['has_trace_map']}")
        print(f"  Tagged operations: {trace_stats['tagged_operations']}")
    
    return result