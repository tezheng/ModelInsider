"""
Root cause analysis and fix for slice operation context tagging issue.

The issue: Slice operations in BERT attention layers get tagged with wrong contexts
(embeddings/pooler) instead of the attention submodule context because the slice
operations are executed with delayed/deferred timing.

The fix: Implement context-aware slice tagging that uses both execution context
and ONNX graph structure to correctly map slice operations to their source modules.
"""

import torch
import torch.nn as nn
from typing import Dict, List, Any, Optional, Tuple
from collections import defaultdict


class SliceContextFixer:
    """
    Fixes slice operation context tagging by implementing a multi-stage approach:
    
    1. Capture slice operations with immediate context (may be wrong due to timing)
    2. Use ONNX graph structure to infer correct context from node names/paths
    3. Apply heuristics to map slice operations to correct attention modules
    4. Fallback to propagation-based tagging when direct mapping fails
    """
    
    def __init__(self, exporter):
        self.exporter = exporter
        self.slice_context_hints = {}  # Additional context hints for slice operations
        self.attention_module_names = set()  # Track attention module names
    
    def patch_slice_tracking(self):
        """Enhanced slice operation tracking with context fixing."""
        original_patch_getitem = self.exporter._patch_tensor_getitem
        
        def enhanced_patch_tensor_getitem():
            """Enhanced __getitem__ patch with better context tracking."""
            if self.exporter._original_getitem is None:
                self.exporter._original_getitem = torch.Tensor.__getitem__
                
                def enhanced_context_aware_getitem(tensor_self, key):
                    is_slice = self.exporter._is_slice_operation(key)
                    
                    if is_slice:
                        # Capture immediate context (may be wrong due to timing)
                        immediate_context = self.exporter.get_current_tag()
                        
                        # Try to infer correct context using additional heuristics
                        corrected_context = self._infer_correct_slice_context(
                            tensor_self, key, immediate_context
                        )
                        
                        # Record slice operation with both contexts
                        if corrected_context:
                            slice_info = {
                                'tensor_id': id(tensor_self),
                                'key': str(key),
                                'context': corrected_context,
                                'immediate_context': immediate_context,
                                'order': len(self.exporter._slice_operations),
                                'type': 'slice',
                                'tensor_shape': list(tensor_self.shape),
                                'context_corrected': corrected_context != immediate_context
                            }
                            self.exporter._slice_operations.append(slice_info)
                    
                    return self.exporter._original_getitem(tensor_self, key)
                
                torch.Tensor.__getitem__ = enhanced_context_aware_getitem
        
        # Replace the patch method
        self.exporter._patch_tensor_getitem = enhanced_patch_tensor_getitem
    
    def _infer_correct_slice_context(self, tensor, key, immediate_context):
        """
        Infer the correct context for a slice operation using multiple heuristics.
        
        Args:
            tensor: The tensor being sliced
            key: The slice key/index
            immediate_context: The context captured immediately (may be wrong)
            
        Returns:
            Corrected context string or None
        """
        # Heuristic 1: If immediate context is already an attention module, trust it
        if immediate_context and 'attention' in immediate_context.lower():
            return immediate_context
        
        # Heuristic 2: Check if tensor shape/pattern suggests attention operation
        if self._is_attention_slice_pattern(tensor, key):
            # Try to find the most recent attention context
            attention_context = self._find_most_recent_attention_context()
            if attention_context:
                return attention_context
        
        # Heuristic 3: Use immediate context as fallback
        return immediate_context
    
    def _is_attention_slice_pattern(self, tensor, key):
        """
        Determine if a slice operation pattern suggests it's from attention.
        
        Common attention slice patterns:
        - Multi-dimensional tensors with head/sequence dimensions
        - Slicing on sequence dimension (often dimension 1 or 2)
        - Tensor shapes that match attention computation patterns
        """
        if not isinstance(key, tuple):
            return False
        
        shape = tensor.shape
        
        # Pattern 1: 4D tensor slicing (typical for multi-head attention)
        if len(shape) == 4 and len(key) >= 3:
            # Check for sequence dimension slicing
            for i, k in enumerate(key):
                if isinstance(k, slice) and k.start is not None and k.stop is not None:
                    # Likely attention sequence slicing
                    return True
        
        # Pattern 2: 3D tensor slicing with attention-like dimensions
        if len(shape) == 3 and len(key) >= 2:
            # Check for sequence dimension slicing
            for i, k in enumerate(key):
                if isinstance(k, slice) and k.start is not None:
                    return True
        
        return False
    
    def _find_most_recent_attention_context(self):
        """Find the most recently executed attention module context."""
        # Search through operation context for attention modules
        attention_contexts = []
        
        for module_name, context_info in self.exporter._operation_context.items():
            if 'attention' in context_info['tag'].lower():
                attention_contexts.append(context_info['tag'])
        
        # Return the most specific attention context (longest path)
        if attention_contexts:
            return max(attention_contexts, key=len)
        
        return None
    
    def enhanced_tag_slice_operations(self, onnx_model, onnx_nodes_by_type):
        """
        Enhanced slice operation tagging with context fixing.
        
        This method replaces the original _tag_slice_operations with:
        1. Better context mapping using ONNX node paths
        2. Attention-specific heuristics
        3. Fallback propagation when direct mapping fails
        """
        if 'Slice' not in onnx_nodes_by_type or not self.exporter._slice_operations:
            return
        
        slice_nodes = onnx_nodes_by_type['Slice']
        print(f"[SLICE FIX] Processing {len(slice_nodes)} ONNX Slice nodes")
        print(f"[SLICE FIX] Have {len(self.exporter._slice_operations)} captured slice operations")
        
        # Build mapping from ONNX node paths to expected contexts
        node_path_contexts = self._build_node_path_context_mapping(onnx_model)
        
        for i, node in enumerate(slice_nodes):
            node_name = node.name or f"{node.op_type}_{len(self.exporter._tag_mapping)}"
            
            print(f"[SLICE FIX] Processing ONNX node: {node_name}")
            
            # Skip if already tagged
            if self.exporter._tag_mapping[node_name]["tags"]:
                print(f"  Already tagged: {self.exporter._tag_mapping[node_name]['tags']}")
                continue
            
            # Method 1: Use ONNX node path to infer correct context
            inferred_context = self._infer_context_from_onnx_path(node_name, node_path_contexts)
            
            if inferred_context:
                print(f"  Inferred from path: {inferred_context}")
                self.exporter._tag_mapping[node_name]["tags"] = [inferred_context]
            else:
                # Method 2: Fallback to captured slice operation (with potential correction)
                if i < len(self.exporter._slice_operations):
                    slice_op = self.exporter._slice_operations[i]
                    context = slice_op['context']
                    was_corrected = slice_op.get('context_corrected', False)
                    
                    print(f"  Using captured context: {context} (corrected: {was_corrected})")
                    self.exporter._tag_mapping[node_name]["tags"] = [context]
                else:
                    # Method 3: Default attention context for attention paths
                    if 'attention' in node_name.lower():
                        default_context = self._find_most_recent_attention_context()
                        if default_context:
                            print(f"  Using default attention context: {default_context}")
                            self.exporter._tag_mapping[node_name]["tags"] = [default_context]
    
    def _build_node_path_context_mapping(self, onnx_model):
        """Build mapping from ONNX node paths to expected module contexts."""
        mapping = {}
        
        # Use operation context to build path -> context mapping
        for module_name, context_info in self.exporter._operation_context.items():
            # Convert module name to potential ONNX path patterns
            onnx_path_variants = self._module_name_to_onnx_path_variants(module_name)
            
            for path_variant in onnx_path_variants:
                mapping[path_variant] = context_info['tag']
        
        return mapping
    
    def _module_name_to_onnx_path_variants(self, module_name):
        """Convert module name to potential ONNX path variants."""
        variants = []
        
        # Direct mapping: encoder.layer.0.attention.self -> /encoder/layer.0/attention/self
        onnx_path = '/' + module_name.replace('.', '/')
        variants.append(onnx_path)
        
        # Partial paths for matching
        parts = module_name.split('.')
        for i in range(1, len(parts) + 1):
            partial_path = '/' + '/'.join(parts[:i])
            variants.append(partial_path)
        
        return variants
    
    def _infer_context_from_onnx_path(self, node_name, path_contexts):
        """Infer correct context from ONNX node path."""
        # Try exact matches first
        if node_name in path_contexts:
            return path_contexts[node_name]
        
        # Try partial matches (find longest matching prefix)
        best_match = None
        best_length = 0
        
        for path, context in path_contexts.items():
            if node_name.startswith(path) and len(path) > best_length:
                best_match = context
                best_length = len(path)
        
        # Special handling for attention paths
        if 'attention' in node_name.lower() and best_match:
            # Ensure we get the most specific attention context
            if 'attention' in best_match.lower():
                return best_match
        
        return best_match


def apply_slice_context_fix(exporter):
    """Apply the slice context fix to an existing exporter."""
    
    # Create the fixer
    fixer = SliceContextFixer(exporter)
    
    # Replace the slice tracking method
    fixer.patch_slice_tracking()
    
    # Replace the slice tagging method
    original_tag_slice_operations = exporter._tag_slice_operations
    exporter._tag_slice_operations = fixer.enhanced_tag_slice_operations
    
    return fixer


# Example usage with the hierarchy exporter
if __name__ == "__main__":
    from modelexport.hierarchy_exporter import HierarchyExporter
    
    print("Slice Context Fix - Root Cause and Solution")
    print("=" * 50)
    print()
    print("ROOT CAUSE:")
    print("- Slice operations in BERT attention layers execute with delayed timing")
    print("- They get captured when embeddings/pooler modules are on the execution stack")
    print("- This causes incorrect context tagging (root-level instead of attention)")
    print()
    print("SOLUTION:")
    print("1. Enhanced slice tracking with context correction heuristics")
    print("2. ONNX node path analysis to infer correct module context")  
    print("3. Attention-specific pattern recognition for slice operations")
    print("4. Fallback mechanisms for robust tagging")
    print()
    print("To apply this fix to your exporter:")
    print("  from slice_context_fix import apply_slice_context_fix")
    print("  exporter = HierarchyExporter(strategy='htp')")
    print("  fixer = apply_slice_context_fix(exporter)")
    print("  # Now export as normal - slices will be correctly tagged")