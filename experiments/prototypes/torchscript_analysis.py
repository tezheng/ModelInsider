"""
Explore TorchScript/IR approach for more explicit operation tagging.
"""

import torch
import torch.nn as nn


class SimpleSliceModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(10, 5)
        
    def forward(self, x):
        # Various operations
        sliced = x[1:4]           # aten::slice
        reshaped = sliced.view(-1, 10)  # aten::view
        processed = self.linear(reshaped)  # aten::linear
        return processed


def analyze_torchscript_ir():
    """Analyze TorchScript IR to see explicit operations."""
    
    model = SimpleSliceModel()
    model.eval()
    
    # Convert to TorchScript
    example_input = torch.randn(5, 10)
    script_model = torch.jit.trace(model, example_input)
    
    print("=== TorchScript Graph ===")
    print(script_model.graph)
    
    print("\n=== Graph Analysis ===")
    
    # Analyze the graph
    graph = script_model.graph
    
    # Get all nodes in the graph
    nodes = list(graph.nodes())
    
    print(f"Total nodes in graph: {len(nodes)}")
    
    for i, node in enumerate(nodes):
        print(f"Node {i}: {node.kind()}")
        print(f"  Inputs: {[inp.debugName() for inp in node.inputs()]}")
        print(f"  Outputs: {[out.debugName() for out in node.outputs()]}")
        print(f"  Schema: {node.schema()}")
        print()
    
    # Look for specific operations
    slice_nodes = [node for node in nodes if 'slice' in node.kind()]
    linear_nodes = [node for node in nodes if 'linear' in node.kind()]
    
    print(f"Slice operations: {len(slice_nodes)}")
    print(f"Linear operations: {len(linear_nodes)}")
    
    return script_model, graph


def map_ir_to_modules(model, graph):
    """Attempt to map IR operations back to modules."""
    
    print("\n=== Module Mapping Analysis ===")
    
    # Get named modules
    named_modules = dict(model.named_modules())
    print("Named modules:")
    for name, module in named_modules.items():
        print(f"  {name}: {type(module).__name__}")
    
    # Analyze parameters in graph
    print("\nParameters in graph:")
    for inp in graph.inputs():
        print(f"  {inp.debugName()}: {inp.type()}")
    
    # This is where we'd need to implement the mapping logic
    # IR nodes -> module contexts
    

class HFStyleModel(nn.Module):
    """HuggingFace-style model for IR analysis."""
    
    def __init__(self):
        super().__init__()
        self.embeddings = nn.Embedding(100, 64)
        self.layer_norm = nn.LayerNorm(64)
        self.attention = nn.MultiheadAttention(64, 8, batch_first=True)
        
    def forward(self, input_ids):
        # Embedding
        x = self.embeddings(input_ids)
        
        # Slice for attention (common pattern)
        seq_len = x.size(1)
        x = x[:, :seq_len-1]  # Remove last token
        
        # Attention
        attn_out, _ = self.attention(x, x, x)
        
        # Normalize
        return self.layer_norm(attn_out)


def analyze_hf_style_ir():
    """Analyze HF-style model IR."""
    
    print("\n" + "="*50)
    print("=== HuggingFace-Style Model IR Analysis ===")
    
    model = HFStyleModel()
    model.eval()
    
    # Create realistic input
    input_ids = torch.randint(0, 100, (2, 10))  # batch=2, seq_len=10
    
    try:
        # Trace the model
        script_model = torch.jit.trace(model, input_ids)
        
        print("HF-Style TorchScript Graph:")
        print(script_model.graph)
        
        # Count operations
        nodes = list(script_model.graph.nodes())
        op_types = [node.kind() for node in nodes]
        
        print(f"\nOperation types found:")
        from collections import Counter
        op_counts = Counter(op_types)
        for op, count in op_counts.most_common():
            print(f"  {op}: {count}")
            
        # Look for slice operations
        slice_ops = [node for node in nodes if 'slice' in node.kind()]
        print(f"\nSlice operations: {len(slice_ops)}")
        for node in slice_ops:
            print(f"  {node.kind()}: {node.schema()}")
            
    except Exception as e:
        print(f"Error tracing HF-style model: {e}")
        print("This might be due to dynamic operations")


if __name__ == "__main__":
    print("Analyzing TorchScript IR for explicit operation mapping...")
    
    # Analyze simple model
    script_model, graph = analyze_torchscript_ir()
    map_ir_to_modules(script_model, graph)
    
    # Analyze HF-style model
    analyze_hf_style_ir()
    
    print("\n=== Conclusion ===")
    print("TorchScript IR provides explicit operation representation")
    print("This could enable more comprehensive tagging than hook-based approach")
    print("Key advantage: Captures ALL operations, including tensor methods")