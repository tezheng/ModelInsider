#!/usr/bin/env python3
"""Solution for tagging native operations that decompose into multiple ONNX nodes."""

import json
from collections import OrderedDict

import onnx
import torch
import torch.nn.functional as F


class NativeOpAwareExporter:
    def __init__(self):
        self.tag_stack = []
        self.native_op_boundaries = []  # Track where native ops start/end
        self.op_trace = OrderedDict()
        self.op_counter = 0
        
    def export_with_native_op_support(self, model, example_input, output_path):
        """Export with support for native operations."""
        
        # 1. Register hooks
        self._register_hooks(model)
        
        # 2. Patch native operations
        self._patch_native_operations()
        
        try:
            # 3. Export to ONNX
            print("Exporting to ONNX with native op tracking...")
            
            # First pass: trace to capture native op boundaries
            with torch.no_grad():
                _ = model(example_input)
            
            # Reset for actual export
            self.native_op_boundaries.clear()
            self.op_counter = 0
            
            # Export
            torch.onnx.export(
                model,
                example_input,
                output_path,
                opset_version=14,
                verbose=False
            )
            
            # 4. Post-process to tag native op decompositions
            self._tag_native_op_decompositions(output_path)
            
        finally:
            self._unpatch_native_operations()
            self._remove_hooks()
    
    def _patch_native_operations(self):
        """Patch native operations to track their boundaries."""
        self.original_ops = {}
        
        # Patch scaled_dot_product_attention
        self.original_ops['sdpa'] = F.scaled_dot_product_attention
        F.scaled_dot_product_attention = self._create_native_op_wrapper(
            'scaled_dot_product_attention',
            F.scaled_dot_product_attention
        )
        
        # Could patch other native ops here...
        
    def _create_native_op_wrapper(self, op_name, original_op):
        """Create wrapper that tracks native op execution."""
        def wrapper(*args, **kwargs):
            # Record start of native op
            current_tag = self.tag_stack[-1] if self.tag_stack else None
            
            start_marker = {
                'type': 'native_op_start',
                'op_name': op_name,
                'tag': current_tag,
                'op_id': self.op_counter
            }
            self.native_op_boundaries.append(start_marker)
            self.op_counter += 1
            
            print(f"  [Native Op Start] {op_name} in context: {current_tag}")
            
            # Execute original
            result = original_op(*args, **kwargs)
            
            # Record end of native op
            end_marker = {
                'type': 'native_op_end',
                'op_name': op_name,
                'tag': current_tag,
                'op_id': self.op_counter
            }
            self.native_op_boundaries.append(end_marker)
            self.op_counter += 1
            
            print(f"  [Native Op End] {op_name}")
            
            return result
        
        return wrapper
    
    def _tag_native_op_decompositions(self, onnx_path):
        """Tag ONNX nodes that came from native operations."""
        onnx_model = onnx.load(onnx_path)
        
        print("\n=== Tagging Native Operation Decompositions ===")
        
        # Strategy: Identify patterns that match native op decompositions
        # For scaled_dot_product_attention, we look for:
        # MatMul -> Div/Mul -> (optional Add) -> Softmax -> MatMul
        
        nodes = onnx_model.graph.node
        tagged_count = 0
        
        for i in range(len(nodes)):
            node = nodes[i]
            
            # Pattern detection for scaled_dot_product_attention
            if self._is_attention_pattern_start(nodes, i):
                # Tag the entire attention pattern
                pattern_length = self._get_attention_pattern_length(nodes, i)
                module_tag = self._infer_module_tag(node)
                
                print(f"\nFound attention pattern starting at node {i}:")
                for j in range(i, min(i + pattern_length, len(nodes))):
                    tag_info = {
                        'module_tag': module_tag,
                        'native_op': 'scaled_dot_product_attention',
                        'pattern_position': j - i
                    }
                    nodes[j].doc_string = json.dumps(tag_info)
                    print(f"  Tagged {nodes[j].name} ({nodes[j].op_type})")
                    tagged_count += 1
        
        # Save tagged model
        output_path = onnx_path.replace('.onnx', '_tagged.onnx')
        onnx.save(onnx_model, output_path)
        print(f"\nTagged {tagged_count} nodes")
        print(f"Saved to: {output_path}")
    
    def _is_attention_pattern_start(self, nodes, idx):
        """Check if this position starts an attention pattern."""
        if idx + 4 >= len(nodes):  # Need at least 5 nodes
            return False
        
        # Simple pattern: look for MatMul followed by scaling operations
        if nodes[idx].op_type == 'MatMul':
            # Check next few operations
            next_ops = [nodes[idx + i].op_type for i in range(1, min(5, len(nodes) - idx))]
            
            # Attention patterns often have Div/Mul for scaling
            if 'Div' in next_ops or 'Mul' in next_ops:
                # And eventually a Softmax
                for j in range(idx, min(idx + 10, len(nodes))):
                    if nodes[j].op_type == 'Softmax':
                        return True
        
        return False
    
    def _get_attention_pattern_length(self, nodes, start_idx):
        """Get the length of the attention pattern."""
        # Find the Softmax
        softmax_idx = None
        for i in range(start_idx, min(start_idx + 10, len(nodes))):
            if nodes[i].op_type == 'Softmax':
                softmax_idx = i
                break
        
        if softmax_idx:
            # Attention pattern typically ends with MatMul after Softmax
            for i in range(softmax_idx + 1, min(softmax_idx + 5, len(nodes))):
                if nodes[i].op_type == 'MatMul':
                    return i - start_idx + 1
        
        return 5  # Default length
    
    def _infer_module_tag(self, node):
        """Infer module tag from node name."""
        # This is simplified - in practice would use our trace
        if 'attention' in node.name.lower():
            return '/BertAttention/scaled_dot_product_attention'
        return '/UnknownModule'
    
    def _register_hooks(self, model):
        """Register hooks to track module execution."""
        self.hooks = []
        
        def create_pre_hook(name, module):
            def hook(module, inputs):
                tag = f"/{module.__class__.__name__}"
                if name:
                    tag = f"{tag}/{name}"
                self.tag_stack.append(tag)
                print(f"Entering: {tag}")
            return hook
        
        def create_post_hook():
            def hook(module, inputs, outputs):
                if self.tag_stack:
                    tag = self.tag_stack.pop()
                    print(f"Exiting: {tag}")
            return hook
        
        for name, module in model.named_modules():
            if name:
                pre = module.register_forward_pre_hook(create_pre_hook(name, module))
                post = module.register_forward_hook(create_post_hook())
                self.hooks.extend([pre, post])
    
    def _remove_hooks(self):
        for hook in getattr(self, 'hooks', []):
            hook.remove()
    
    def _unpatch_native_operations(self):
        F.scaled_dot_product_attention = self.original_ops['sdpa']


# Test with attention model
class TestAttentionModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.attention = torch.nn.MultiheadAttention(64, 4, batch_first=True)
        self.pooler = torch.nn.Linear(64, 32)
    
    def forward(self, x):
        # MultiheadAttention uses scaled_dot_product_attention internally
        attn_out, _ = self.attention(x, x, x)
        pooled = self.pooler(attn_out.mean(dim=1))
        return pooled


def test_native_op_tagging():
    """Test tagging of native operations."""
    model = TestAttentionModel()
    model.eval()
    
    example_input = torch.randn(1, 10, 64)
    
    exporter = NativeOpAwareExporter()
    exporter.export_with_native_op_support(model, example_input, "native_test.onnx")
    
    # Verify the result
    print("\n=== Verification ===")
    tagged_model = onnx.load("native_test_tagged.onnx")
    
    native_op_nodes = []
    for node in tagged_model.graph.node:
        if node.doc_string:
            try:
                info = json.loads(node.doc_string)
                if 'native_op' in info:
                    native_op_nodes.append((node.name, node.op_type, info))
            except:
                pass
    
    print(f"\nFound {len(native_op_nodes)} nodes from native operations:")
    for name, op_type, info in native_op_nodes[:5]:
        print(f"  {name} ({op_type}): {info['native_op']}")


if __name__ == "__main__":
    test_native_op_tagging()