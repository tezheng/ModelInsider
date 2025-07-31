#!/usr/bin/env python3
"""Trace-aware ONNX export that captures module context."""

import json
from collections import OrderedDict

import onnx
import torch
import torch.onnx


class TraceAwareExporter:
    def __init__(self):
        self.tag_stack = []
        self.operation_trace = OrderedDict()  # Maps operation_id -> module_tag
        self.op_counter = 0
        
    def export_with_tags(self, model, example_input, output_path):
        """Export model with module tagging."""
        
        # Step 1: Trace model execution with hooks
        print("=== Step 1: Tracing Model Execution ===")
        self._register_execution_hooks(model)
        
        # Monkey-patch torch operations to capture context
        self._patch_operations()
        
        try:
            # Step 2: Export to ONNX (our patches capture context)
            print("\n=== Step 2: Exporting to ONNX ===")
            torch.onnx.export(
                model,
                example_input,
                output_path,
                verbose=False,
                opset_version=11,
                do_constant_folding=True,
                input_names=['input'],
                output_names=['output'],
            )
            
            # Step 3: Post-process ONNX model with tags
            print("\n=== Step 3: Injecting Tags into ONNX ===")
            self._inject_tags(output_path)
            
        finally:
            # Cleanup
            self._unpatch_operations()
            self._remove_hooks()
    
    def _register_execution_hooks(self, model):
        """Register hooks to track module execution."""
        self.hooks = []
        
        def create_pre_hook(module_name, module):
            def pre_hook(module, inputs):
                tag = f"/{module.__class__.__name__}/{module_name}" if module_name else f"/{module.__class__.__name__}"
                self.tag_stack.append(tag)
                print(f"  Entering: {tag}")
            return pre_hook
        
        def create_post_hook(module_name):
            def post_hook(module, inputs, outputs):
                if self.tag_stack:
                    tag = self.tag_stack.pop()
                    print(f"  Exiting: {tag}")
            return post_hook
        
        for name, module in model.named_modules():
            if name:  # Skip root
                pre = module.register_forward_pre_hook(create_pre_hook(name, module))
                post = module.register_forward_hook(create_post_hook(name))
                self.hooks.extend([pre, post])
    
    def _remove_hooks(self):
        """Remove all hooks."""
        for hook in getattr(self, 'hooks', []):
            hook.remove()
        self.hooks = []
    
    def _patch_operations(self):
        """Monkey-patch key torch operations to capture context."""
        self.original_ops = {}
        
        # Patch common operations
        operations = {
            'matmul': torch.matmul,
            'addmm': torch.addmm,
            'tanh': torch.tanh,
            'add': torch.add,
        }
        
        for op_name, op_func in operations.items():
            self.original_ops[op_name] = op_func
            setattr(torch, op_name, self._create_traced_op(op_name, op_func))
        
        # Also patch functional versions
        import torch.nn.functional as F
        if hasattr(F, 'linear'):
            self.original_ops['F.linear'] = F.linear
            F.linear = self._create_traced_op('linear', F.linear)
    
    def _create_traced_op(self, op_name, original_op):
        """Create a traced version of an operation."""
        def traced_op(*args, **kwargs):
            # Capture current module context
            current_tag = self.tag_stack[-1] if self.tag_stack else None
            
            # Execute operation
            result = original_op(*args, **kwargs)
            
            # Record the operation with its context
            if current_tag:
                self.op_counter += 1
                op_id = f"{op_name}_{self.op_counter}"
                self.operation_trace[op_id] = current_tag
                print(f"    Operation {op_id} in context: {current_tag}")
            
            return result
        
        return traced_op
    
    def _unpatch_operations(self):
        """Restore original operations."""
        for op_name, original_op in self.original_ops.items():
            if '.' in op_name:
                module_name, func_name = op_name.split('.', 1)
                if module_name == 'F':
                    import torch.nn.functional as F
                    setattr(F, func_name, original_op)
            else:
                setattr(torch, op_name, original_op)
    
    def _inject_tags(self, onnx_path):
        """Post-process ONNX model to inject tags."""
        onnx_model = onnx.load(onnx_path)
        
        # Simple heuristic: match operations by order and type
        op_trace_items = list(self.operation_trace.items())
        trace_idx = 0
        
        print("\nMatching ONNX nodes to execution trace:")
        for node in onnx_model.graph.node:
            # Try to match with our trace
            if trace_idx < len(op_trace_items):
                op_id, module_tag = op_trace_items[trace_idx]
                op_type = op_id.split('_')[0]
                
                # Simple matching by operation type
                if self._matches_op_type(node.op_type, op_type):
                    # Add tag as doc_string
                    tag_info = {
                        'module_tag': module_tag,
                        'op_trace_id': op_id
                    }
                    node.doc_string = json.dumps(tag_info)
                    print(f"  Tagged {node.name} ({node.op_type}) -> {module_tag}")
                    trace_idx += 1
        
        # Save modified model
        onnx.save(onnx_model, onnx_path.replace('.onnx', '_tagged.onnx'))
        print(f"\nTagged model saved to: {onnx_path.replace('.onnx', '_tagged.onnx')}")
    
    def _matches_op_type(self, onnx_op, torch_op):
        """Check if ONNX op type matches torch op."""
        mappings = {
            'Gemm': ['linear', 'addmm', 'matmul'],
            'MatMul': ['matmul'],
            'Tanh': ['tanh'],
            'Add': ['add'],
        }
        
        for onnx_type, torch_types in mappings.items():
            if onnx_op == onnx_type and torch_op in torch_types:
                return True
        return False


# Test model
class TestModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = torch.nn.Sequential(
            torch.nn.Linear(10, 8),
            torch.nn.Tanh(),
        )
        self.pooler = torch.nn.Sequential(
            torch.nn.Linear(8, 4),
            torch.nn.Tanh(),
        )
    
    def forward(self, x):
        encoded = self.encoder(x)
        pooled = self.pooler(encoded)
        return pooled


def test_trace_aware_export():
    """Test trace-aware ONNX export."""
    model = TestModel()
    example_input = torch.randn(1, 10)
    
    exporter = TraceAwareExporter()
    exporter.export_with_tags(model, example_input, "test_model.onnx")
    
    # Verify the result
    print("\n=== Verifying Tagged ONNX Model ===")
    tagged_model = onnx.load("test_model_tagged.onnx")
    
    for node in tagged_model.graph.node:
        if node.doc_string:
            tag_info = json.loads(node.doc_string)
            print(f"Node {node.name} ({node.op_type}): {tag_info['module_tag']}")


if __name__ == "__main__":
    test_trace_aware_export()