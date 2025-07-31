#!/usr/bin/env python3
"""Direct test of native operation handling."""


import onnx
import torch
import torch.nn.functional as F


class DirectAttentionModel(torch.nn.Module):
    """Model that directly uses scaled_dot_product_attention."""
    def __init__(self):
        super().__init__()
        self.q_proj = torch.nn.Linear(64, 64)
        self.k_proj = torch.nn.Linear(64, 64)
        self.v_proj = torch.nn.Linear(64, 64)
        
    def forward(self, x):
        # Direct use of native function
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)
        
        # This is the native C++ function
        output = F.scaled_dot_product_attention(q, k, v)
        
        return output


def analyze_native_decomposition():
    """Analyze how native ops decompose in ONNX."""
    model = DirectAttentionModel()
    model.eval()
    
    x = torch.randn(1, 10, 64)
    
    # Track operations
    op_sequence = []
    
    # Patch operations to track execution
    original_ops = {}
    
    # Patch the native function
    original_sdpa = F.scaled_dot_product_attention
    def tracked_sdpa(*args, **kwargs):
        op_sequence.append(('NATIVE_START', 'scaled_dot_product_attention'))
        result = original_sdpa(*args, **kwargs)
        op_sequence.append(('NATIVE_END', 'scaled_dot_product_attention'))
        return result
    
    # Patch common ops that might appear in decomposition
    ops_to_track = {
        'matmul': torch.matmul,
        'bmm': torch.bmm,
        'softmax': torch.softmax,
        'div': torch.div,
        'mul': torch.mul,
        'add': torch.add,
    }
    
    for op_name, op_func in ops_to_track.items():
        original_ops[op_name] = op_func
        
        def make_tracked_op(name, original):
            def tracked(*args, **kwargs):
                op_sequence.append(('OP', name))
                return original(*args, **kwargs)
            return tracked
        
        setattr(torch, op_name, make_tracked_op(op_name, op_func))
    
    # Patch F.scaled_dot_product_attention
    F.scaled_dot_product_attention = tracked_sdpa
    
    try:
        # Export to ONNX
        print("=== Exporting to ONNX ===")
        torch.onnx.export(
            model,
            x,
            "direct_native.onnx",
            opset_version=14,
            verbose=False
        )
        
        print("\nOperation sequence during export:")
        for op_type, op_name in op_sequence:
            print(f"  {op_type}: {op_name}")
        
        # Analyze ONNX
        onnx_model = onnx.load("direct_native.onnx")
        
        print(f"\n=== ONNX Graph Analysis ===")
        print(f"Total nodes: {len(onnx_model.graph.node)}")
        
        # Find the decomposition pattern
        print("\nONNX operations (in order):")
        native_region = False
        for i, node in enumerate(onnx_model.graph.node):
            # Look for patterns that indicate attention computation
            if node.op_type in ['MatMul', 'Div', 'Softmax', 'Mul']:
                if not native_region and i > 3:  # After the projections
                    print("\n--- Likely start of scaled_dot_product_attention decomposition ---")
                    native_region = True
                    
            print(f"  {i}: {node.op_type} ({node.name})")
            
            # End of attention pattern
            if native_region and node.op_type == 'MatMul' and i > 10:
                print("--- Likely end of scaled_dot_product_attention decomposition ---\n")
                native_region = False
        
    finally:
        # Restore originals
        F.scaled_dot_product_attention = original_sdpa
        for op_name, original in original_ops.items():
            setattr(torch, op_name, original)


def demonstrate_solution():
    """Demonstrate the complete solution."""
    print("\n\n=== SOLUTION SUMMARY ===\n")
    
    print("For native C++ operations like scaled_dot_product_attention:")
    print()
    print("1. **Detection**: Patch the Python wrapper to mark boundaries")
    print("   - Know when native op starts and ends")
    print("   - Track the module context at that time")
    print()
    print("2. **Pattern Matching**: Native ops decompose predictably")
    print("   - scaled_dot_product_attention → MatMul, Scale, Softmax, MatMul")
    print("   - flash_attention → Similar pattern with optimizations")
    print()
    print("3. **Tagging Strategy**:")
    print("   - Tag all operations in the pattern with the same module tag")
    print("   - Use the module context from when native op was called")
    print()
    print("4. **Implementation in HierarchyExporter**:")
    print("   - Maintain a 'native_op_regions' list during export")
    print("   - Post-process ONNX to tag operations in these regions")
    print("   - Each native op type has its known decomposition pattern")


if __name__ == "__main__":
    analyze_native_decomposition()
    demonstrate_solution()