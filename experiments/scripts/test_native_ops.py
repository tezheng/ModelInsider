#!/usr/bin/env python3
"""Test how native C++ operations like scaled_dot_product_attention behave in ONNX export."""

import torch
import torch.nn.functional as F
import onnx

class AttentionModel(torch.nn.Module):
    def __init__(self, hidden_size=64, num_heads=4):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        
        # Standard transformer components
        self.q_proj = torch.nn.Linear(hidden_size, hidden_size)
        self.k_proj = torch.nn.Linear(hidden_size, hidden_size)
        self.v_proj = torch.nn.Linear(hidden_size, hidden_size)
        self.out_proj = torch.nn.Linear(hidden_size, hidden_size)
        
    def forward(self, hidden_states, attention_mask=None):
        batch_size, seq_len, _ = hidden_states.shape
        
        # Project to Q, K, V
        query = self.q_proj(hidden_states)
        key = self.k_proj(hidden_states)
        value = self.v_proj(hidden_states)
        
        # Reshape for multi-head attention
        query = query.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        key = key.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        value = value.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        # This is the native C++ function!
        attn_output = F.scaled_dot_product_attention(
            query, key, value,
            attn_mask=attention_mask,
            dropout_p=0.0,
            is_causal=False
        )
        
        # Reshape back
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(batch_size, seq_len, self.hidden_size)
        
        # Output projection
        output = self.out_proj(attn_output)
        
        return output


def analyze_native_ops():
    """Analyze how native ops appear in ONNX."""
    print("=== Native Operation Analysis ===\n")
    
    # Create model and example input
    model = AttentionModel()
    model.eval()
    
    batch_size, seq_len = 1, 10
    hidden_states = torch.randn(batch_size, seq_len, 64)
    
    # Export to ONNX
    print("Exporting model with scaled_dot_product_attention...")
    torch.onnx.export(
        model,
        hidden_states,
        "attention_model.onnx",
        input_names=['hidden_states'],
        output_names=['output'],
        opset_version=14,
        verbose=False
    )
    
    # Load and analyze ONNX model
    onnx_model = onnx.load("attention_model.onnx")
    
    print("\n=== ONNX Graph Analysis ===")
    print(f"Total nodes: {len(onnx_model.graph.node)}")
    
    # Count operation types
    op_counts = {}
    for node in onnx_model.graph.node:
        op_type = node.op_type
        op_counts[op_type] = op_counts.get(op_type, 0) + 1
    
    print("\nOperation types:")
    for op_type, count in sorted(op_counts.items()):
        print(f"  {op_type}: {count}")
    
    # Look for attention-related operations
    print("\n=== Attention-Related Operations ===")
    attention_ops = []
    for i, node in enumerate(onnx_model.graph.node):
        # scaled_dot_product_attention typically decomposes into:
        # MatMul (Q@K), Div (scale), Softmax, MatMul (attention@V)
        if node.op_type in ['MatMul', 'Div', 'Softmax', 'Mul', 'Add']:
            # Check if it's between the projection and output projection
            if i > 5 and i < len(onnx_model.graph.node) - 5:  # Rough heuristic
                attention_ops.append((i, node.op_type, node.name))
    
    print(f"\nPotential attention decomposition ({len(attention_ops)} ops):")
    for idx, op_type, name in attention_ops[:10]:  # Show first 10
        print(f"  Node {idx}: {op_type} - {name}")
    
    # Trace execution with hooks
    print("\n=== Execution Trace with Hooks ===")
    trace = []
    
    def trace_hook(name):
        def hook(module, inputs, outputs):
            trace.append(f"{name} ({type(module).__name__})")
        return hook
    
    # Register hooks
    handles = []
    for name, module in model.named_modules():
        if name:
            handle = module.register_forward_hook(trace_hook(name))
            handles.append(handle)
    
    # Trace execution
    with torch.no_grad():
        _ = model(hidden_states)
    
    print("\nModule execution order:")
    for t in trace:
        print(f"  {t}")
    
    # The challenge: scaled_dot_product_attention executes atomically
    # but produces multiple ONNX nodes. How do we tag them all?
    
    print("\n=== The Challenge ===")
    print("scaled_dot_product_attention is atomic in PyTorch but creates multiple ONNX nodes:")
    print("  1. MatMul (Q @ K^T)")
    print("  2. Div or Mul (scaling by sqrt(d_k))")
    print("  3. Add (attention mask if provided)")
    print("  4. Softmax")
    print("  5. MatMul (attention @ V)")
    print("\nWe need to identify and tag all these operations together!")
    
    # Clean up
    for handle in handles:
        handle.remove()


def test_native_op_patching():
    """Test if we can intercept native operations."""
    print("\n\n=== Testing Native Operation Patching ===\n")
    
    # Try to patch scaled_dot_product_attention
    original_sdpa = F.scaled_dot_product_attention
    calls = []
    
    def patched_sdpa(query, key, value, **kwargs):
        calls.append({
            'query_shape': query.shape,
            'key_shape': key.shape,
            'value_shape': value.shape,
            'kwargs': kwargs
        })
        print(f"Intercepted scaled_dot_product_attention call!")
        return original_sdpa(query, key, value, **kwargs)
    
    # Patch it
    F.scaled_dot_product_attention = patched_sdpa
    
    try:
        # Test execution
        model = AttentionModel()
        hidden_states = torch.randn(1, 10, 64)
        
        with torch.no_grad():
            output = model(hidden_states)
        
        print(f"\nIntercepted {len(calls)} calls to scaled_dot_product_attention")
        for i, call in enumerate(calls):
            print(f"  Call {i}: Q{call['query_shape']}, K{call['key_shape']}, V{call['value_shape']}")
        
        # Export to ONNX with patching
        print("\nExporting with patched operation...")
        torch.onnx.export(
            model,
            hidden_states,
            "attention_patched.onnx",
            verbose=False
        )
        
        print("✓ Patching works! We can intercept native operations")
        
    finally:
        # Restore original
        F.scaled_dot_product_attention = original_sdpa


if __name__ == "__main__":
    analyze_native_ops()
    test_native_op_patching()