#!/usr/bin/env python3
"""Test custom symbolic functions for ONNX export with tagging."""

import torch
import torch.onnx
import onnx
import json

class TaggingExporter:
    def __init__(self):
        self.tag_stack = []
        self.current_tag = None
        self.original_symbolic_fns = {}
        
    def push_tag(self, tag):
        """Push a module tag onto the stack."""
        self.tag_stack.append(tag)
        self.current_tag = tag
        
    def pop_tag(self):
        """Pop a tag from the stack."""
        if self.tag_stack:
            self.tag_stack.pop()
            self.current_tag = self.tag_stack[-1] if self.tag_stack else None
    
    def create_tagged_symbolic(self, op_name, original_symbolic):
        """Create a symbolic function that adds tagging."""
        def tagged_symbolic(g, *args, **kwargs):
            # Call original symbolic function
            outputs = original_symbolic(g, *args, **kwargs)
            
            # Add our module tag as metadata
            if self.current_tag and hasattr(outputs, 's_'):
                # s_ sets string attributes on the node
                outputs.s_('module_tag', self.current_tag)
                print(f"Tagged {op_name} with {self.current_tag}")
            
            return outputs
        
        return tagged_symbolic
    
    def register_symbolic_hooks(self):
        """Register custom symbolic functions for common ops."""
        ops_to_hook = [
            'aten::add',
            'aten::matmul', 
            'aten::linear',
            'aten::tanh',
        ]
        
        for op in ops_to_hook:
            try:
                # Get original symbolic function
                original = torch.onnx.symbolic_registry.get_registered_op(op, '', 9)
                if original:
                    self.original_symbolic_fns[op] = original
                    # Register our wrapped version
                    tagged_fn = self.create_tagged_symbolic(op, original)
                    torch.onnx.register_custom_op_symbolic(op, tagged_fn, 9)
                    print(f"Registered hook for {op}")
            except Exception as e:
                print(f"Could not hook {op}: {e}")
    
    def unregister_symbolic_hooks(self):
        """Restore original symbolic functions."""
        for op, original in self.original_symbolic_fns.items():
            torch.onnx.register_custom_op_symbolic(op, original, 9)


# Test model
class SimpleModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = torch.nn.Linear(10, 5)
        self.layer2 = torch.nn.Linear(5, 2)
        
    def forward(self, x):
        # layer1 computation
        x = self.layer1(x)
        x = torch.tanh(x)
        
        # layer2 computation  
        x = self.layer2(x)
        x = torch.tanh(x)
        
        return x


def test_symbolic_tagging():
    """Test if we can tag ONNX nodes via symbolic functions."""
    print("=== Testing Symbolic Function Tagging ===\n")
    
    model = SimpleModel()
    example_input = torch.randn(1, 10)
    
    # Create exporter
    exporter = TaggingExporter()
    
    # Register hooks on the model to track module execution
    def register_module_hooks(model, exporter):
        def create_pre_hook(module_name):
            def hook(module, input):
                tag = f"/{module.__class__.__name__}/{module_name}"
                exporter.push_tag(tag)
                print(f"Entering {module_name}, tag: {tag}")
            return hook
            
        def create_post_hook(module_name):
            def hook(module, input, output):
                exporter.pop_tag()
                print(f"Exiting {module_name}")
            return hook
        
        for name, module in model.named_modules():
            if name and not isinstance(module, torch.nn.Sequential):
                module.register_forward_pre_hook(create_pre_hook(name))
                module.register_forward_hook(create_post_hook(name))
    
    # Setup
    register_module_hooks(model, exporter)
    exporter.register_symbolic_hooks()
    
    try:
        # Export with our hooks active
        print("\nExporting to ONNX...")
        torch.onnx.export(
            model,
            example_input,
            "test_tagged.onnx",
            verbose=True,
            opset_version=9
        )
        
        # Check the result
        print("\n=== Checking ONNX Model ===")
        onnx_model = onnx.load("test_tagged.onnx")
        
        for node in onnx_model.graph.node:
            # Look for our custom attribute
            for attr in node.attribute:
                if attr.name == "module_tag":
                    print(f"Node {node.name} ({node.op_type}) has tag: {attr.s.decode()}")
        
    finally:
        # Cleanup
        exporter.unregister_symbolic_hooks()


if __name__ == "__main__":
    test_symbolic_tagging()