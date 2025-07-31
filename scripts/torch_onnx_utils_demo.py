#!/usr/bin/env python3
"""
Demonstration of key torch.onnx.utils functions discovered in our analysis.

This script demonstrates the critical functions from torch.onnx.utils that are
relevant to our hierarchy preservation work, particularly _setup_trace_module_map.
"""

import inspect
import tempfile
from pathlib import Path

import torch
import torch.nn as nn
import torch.onnx
from torch.onnx.utils import (
    is_in_onnx_export,
    model_signature,
    unconvertible_ops,
    unpack_quantized_tensor,
)


class HierarchicalBertLikeModel(nn.Module):
    """A BERT-like model to demonstrate hierarchy tracing."""
    
    def __init__(self, vocab_size=1000, hidden_size=128, num_layers=2):
        super().__init__()
        self.embeddings = nn.ModuleDict({
            'word_embeddings': nn.Embedding(vocab_size, hidden_size),
            'position_embeddings': nn.Embedding(512, hidden_size),
            'LayerNorm': nn.LayerNorm(hidden_size)
        })
        
        self.encoder = nn.ModuleList([
            self._create_layer(hidden_size) for _ in range(num_layers)
        ])
        
        self.pooler = nn.Linear(hidden_size, hidden_size)
    
    def _create_layer(self, hidden_size):
        return nn.ModuleDict({
            'attention': nn.ModuleDict({
                'self': nn.MultiheadAttention(hidden_size, num_heads=8, batch_first=True),
                'output': nn.ModuleDict({
                    'dense': nn.Linear(hidden_size, hidden_size),
                    'LayerNorm': nn.LayerNorm(hidden_size)
                })
            }),
            'intermediate': nn.ModuleDict({
                'dense': nn.Linear(hidden_size, hidden_size * 4),
                'activation': nn.GELU()
            }),
            'output': nn.ModuleDict({
                'dense': nn.Linear(hidden_size * 4, hidden_size),
                'LayerNorm': nn.LayerNorm(hidden_size)
            })
        })
    
    def forward(self, input_ids, position_ids=None):
        # Context-aware behavior demonstration
        if is_in_onnx_export():
            print("🔍 Model is executing within ONNX export context")
        
        seq_length = input_ids.size(1)
        if position_ids is None:
            position_ids = torch.arange(seq_length, dtype=torch.long, device=input_ids.device)
            position_ids = position_ids.unsqueeze(0).expand_as(input_ids)
        
        # Embeddings
        word_embeds = self.embeddings['word_embeddings'](input_ids)
        pos_embeds = self.embeddings['position_embeddings'](position_ids)
        embeddings = self.embeddings['LayerNorm'](word_embeds + pos_embeds)
        
        # Encoder layers
        hidden_states = embeddings
        for _i, layer in enumerate(self.encoder):
            # Self attention
            attn_output, _ = layer['attention']['self'](
                hidden_states, hidden_states, hidden_states
            )
            attn_output = layer['attention']['output']['dense'](attn_output)
            attn_output = layer['attention']['output']['LayerNorm'](attn_output + hidden_states)
            
            # Feed forward
            intermediate = layer['intermediate']['dense'](attn_output)
            intermediate = layer['intermediate']['activation'](intermediate)
            layer_output = layer['output']['dense'](intermediate)
            hidden_states = layer['output']['LayerNorm'](layer_output + attn_output)
        
        # Pooler
        pooled_output = self.pooler(hidden_states[:, 0])  # Use [CLS] token
        
        return hidden_states, pooled_output


def demonstrate_model_signature():
    """Demonstrate model_signature function."""
    print("🔍 Model Signature Analysis")
    print("=" * 50)
    
    model = HierarchicalBertLikeModel()
    sig = model_signature(model)
    
    print(f"Model signature: {sig}")
    print(f"Parameters:")
    for name, param in sig.parameters.items():
        default = f", default={param.default}" if param.default != inspect.Parameter.empty else ""
        print(f"  - {name}: {param.annotation}{default}")
    print()


def demonstrate_is_in_onnx_export():
    """Demonstrate is_in_onnx_export context checking."""
    print("🔍 ONNX Export Context Detection")
    print("=" * 50)
    
    model = HierarchicalBertLikeModel()
    input_ids = torch.randint(0, 1000, (1, 16))
    
    print("Normal execution:")
    _ = model(input_ids)
    
    print("\nDuring ONNX export:")
    with tempfile.NamedTemporaryFile(suffix='.onnx', delete=False) as f:
        torch.onnx.export(
            model,
            (input_ids,),
            f.name,
            verbose=False,
            opset_version=17
        )
        print(f"✅ Model exported to {f.name}")
    print()


def demonstrate_unconvertible_ops():
    """Demonstrate unconvertible operations analysis."""
    print("🔍 Unconvertible Operations Analysis")
    print("=" * 50)
    
    model = HierarchicalBertLikeModel()
    input_ids = torch.randint(0, 1000, (1, 16))
    
    try:
        unconvertible = unconvertible_ops(model, (input_ids,))
        if unconvertible:
            print(f"⚠️  Found {len(unconvertible)} unconvertible operations:")
            for op in unconvertible:
                print(f"   - {op}")
        else:
            print("✅ All operations are convertible to ONNX!")
    except Exception as e:
        print(f"Analysis completed with note: {e}")
    print()


def demonstrate_quantized_tensor_handling():
    """Demonstrate quantized tensor unpacking."""
    print("🔍 Quantized Tensor Handling")
    print("=" * 50)
    
    # Create a quantized tensor
    x = torch.randn(2, 3)
    quantized_tensor = torch.quantize_per_tensor(x, scale=0.1, zero_point=10, dtype=torch.quint8)
    
    print(f"Original tensor: {x}")
    print(f"Quantized tensor: {quantized_tensor}")
    print(f"Quantized dtype: {quantized_tensor.dtype}")
    
    # Unpack for ONNX export
    unpacked = unpack_quantized_tensor(quantized_tensor)
    print(f"Unpacked result: {unpacked}")
    print(f"Unpacked type: {type(unpacked)}")
    print()


def demonstrate_trace_module_map_concepts():
    """Demonstrate concepts related to trace module map."""
    print("🔍 Trace Module Map Concepts")
    print("=" * 50)
    
    model = HierarchicalBertLikeModel()
    
    print("Model hierarchy that _setup_trace_module_map will process:")
    for name, module in model.named_modules():
        if name:  # Skip root module
            module_type = type(module).__name__
            # Simulate what _setup_trace_module_map creates
            scope_name = f"{module_type}::{name}"
            print(f"  {name:40} -> {scope_name}")
    
    print(f"\n📊 Total modules: {len(list(model.named_modules())) - 1}")  # -1 for root
    print("This hierarchy mapping is exactly what _setup_trace_module_map creates!")
    print()


def demonstrate_export_with_hierarchy_analysis():
    """Demonstrate ONNX export with hierarchy analysis."""
    print("🔍 ONNX Export with Hierarchy Analysis")
    print("=" * 50)
    
    model = HierarchicalBertLikeModel()
    input_ids = torch.randint(0, 1000, (1, 16))
    
    # Create temp directory
    temp_dir = Path("temp")
    temp_dir.mkdir(exist_ok=True)
    output_path = temp_dir / "hierarchical_model.onnx"
    
    print(f"Exporting model to {output_path}")
    
    # Export with detailed configuration
    torch.onnx.export(
        model,
        (input_ids,),
        str(output_path),
        input_names=['input_ids'],
        output_names=['hidden_states', 'pooled_output'],
        dynamic_axes={
            'input_ids': {0: 'batch_size', 1: 'sequence_length'},
            'hidden_states': {0: 'batch_size', 1: 'sequence_length'},
            'pooled_output': {0: 'batch_size'}
        },
        opset_version=17,
        do_constant_folding=True,
        verbose=False,
        export_modules_as_functions=False  # Standard export mode
    )
    
    print("✅ Export completed!")
    print(f"📁 File size: {output_path.stat().st_size / (1024*1024):.2f} MB")
    
    # Load and analyze the exported model
    import onnx
    onnx_model = onnx.load(str(output_path))
    print(f"📊 ONNX nodes: {len(onnx_model.graph.node)}")
    print(f"📊 Inputs: {len(onnx_model.graph.input)}")
    print(f"📊 Outputs: {len(onnx_model.graph.output)}")
    
    # Show some sample node names (these would have hierarchy info in a full implementation)
    print(f"\n🏷️  Sample ONNX node names:")
    for i, node in enumerate(onnx_model.graph.node[:10]):
        print(f"   {node.name or f'node_{i}'} ({node.op_type})")
    
    print()


def main():
    """Run all demonstrations."""
    print("🎯 PyTorch ONNX Utils Demonstration")
    print("=" * 70)
    print("This script demonstrates key functions from torch.onnx.utils")
    print("that are relevant to our hierarchy preservation work.\n")
    
    demonstrate_model_signature()
    demonstrate_is_in_onnx_export()
    demonstrate_unconvertible_ops()
    demonstrate_quantized_tensor_handling()
    demonstrate_trace_module_map_concepts()
    demonstrate_export_with_hierarchy_analysis()
    
    print("🎉 All demonstrations completed!")
    print("\n🔍 Key Insights:")
    print("- _setup_trace_module_map() is the critical function for hierarchy preservation")
    print("- is_in_onnx_export() allows context-aware model behavior")
    print("- model_signature() helps understand model requirements")
    print("- unconvertible_ops() provides pre-export validation")
    print("- The trace module map creates scope names like 'ModuleType::module.path'")


if __name__ == "__main__":
    main()