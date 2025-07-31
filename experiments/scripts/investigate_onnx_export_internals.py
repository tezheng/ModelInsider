#!/usr/bin/env python3
"""Investigate ONNX export internals for BERT-tiny."""

import warnings

import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer

warnings.filterwarnings("ignore")

def trace_module_annotations():
    """Trace what happens during ONNX export with module annotations."""
    
    print("🔍 Investigating ONNX export internals...\n")
    
    # Load BERT-tiny
    model = AutoModel.from_pretrained("prajjwal1/bert-tiny")
    tokenizer = AutoTokenizer.from_pretrained("prajjwal1/bert-tiny")
    model.eval()
    
    # Create sample input
    inputs = tokenizer(["Hello world"], return_tensors="pt", padding=True, truncation=True, max_length=32)
    sample_input = inputs['input_ids']
    
    # 1. Check if modules have torch.jit annotations
    print("📝 Checking for JIT annotations...\n")
    
    annotated_modules = []
    for name, module in model.named_modules():
        # Check for JIT-related attributes
        jit_attrs = ['_jit_function_counter', '_jit_is_script_module', 
                     '_jit_override_qualname', '__annotations__']
        
        for attr in jit_attrs:
            if hasattr(module, attr):
                value = getattr(module, attr)
                if value:  # Not empty
                    annotated_modules.append((name, module.__class__.__name__, attr, value))
                    print(f"{name} ({module.__class__.__name__}): {attr} = {value}")
    
    if not annotated_modules:
        print("No JIT annotations found on modules")
    
    # 2. Test simplified models
    print("\n" + "="*60)
    print("\n🧪 Testing simplified models...\n")
    
    # Test 1: Simple duplicate modules
    class SimpleModelWithDuplicates(nn.Module):
        def __init__(self):
            super().__init__()
            self.layer1 = nn.Linear(10, 10)
            self.layer2 = nn.Linear(10, 10)
            
        def forward(self, x):
            x = self.layer1(x)
            x = self.layer2(x)
            return x
    
    simple_model = SimpleModelWithDuplicates()
    simple_input = torch.randn(1, 10)
    
    try:
        torch.onnx.export(simple_model, simple_input, "temp/simple_test.onnx", 
                         export_modules_as_functions=True, verbose=False)
        print("✅ Simple duplicate modules: export_modules_as_functions works")
    except Exception as e:
        print(f"❌ Simple duplicate modules failed: {e}")
    
    # Test 2: Modules with dynamic attributes
    class ModelWithDynamicAttrs(nn.Module):
        def __init__(self):
            super().__init__()
            self.layer1 = nn.Linear(10, 10)
            self.layer2 = nn.Linear(10, 10)
            # Add dynamic attribute to only one layer
            self.layer1.custom_attr = "test"
            
        def forward(self, x):
            x = self.layer1(x)
            x = self.layer2(x)
            return x
    
    dynamic_model = ModelWithDynamicAttrs()
    
    try:
        torch.onnx.export(dynamic_model, simple_input, "temp/dynamic_test.onnx", 
                         export_modules_as_functions=True, verbose=False)
        print("✅ Dynamic attributes: export_modules_as_functions works")
    except Exception as e:
        print(f"❌ Dynamic attributes failed: {e}")
    
    # 3. Check BERT-specific issues
    print("\n" + "="*60)
    print("\n🔬 Analyzing BERT-specific architecture...\n")
    
    # Check for BertSdpaSelfAttention - this is BERT's special attention
    sdpa_modules = []
    for name, module in model.named_modules():
        if 'SdpaSelfAttention' in module.__class__.__name__:
            sdpa_modules.append((name, module))
    
    print(f"Found {len(sdpa_modules)} SdpaSelfAttention modules")
    if sdpa_modules:
        print("This is using Scaled Dot Product Attention (SDPA) optimization!")
        print("SDPA modules often have special handling that may conflict with export_modules_as_functions")
    
    # Check for gradient checkpointing
    if hasattr(model.config, 'gradient_checkpointing'):
        print(f"\nGradient checkpointing: {model.config.gradient_checkpointing}")
    
    # Check for special model attributes
    special_attrs = ['_enable_nested_tensor', '_use_flash_attention_2', 
                     '_use_sdpa', 'gradient_checkpointing']
    
    print("\nSpecial model attributes:")
    for attr in special_attrs:
        if hasattr(model, attr):
            print(f"  {attr}: {getattr(model, attr)}")
    
    # 4. Try to reproduce with minimal BERT-like structure
    print("\n" + "="*60)
    print("\n🔨 Testing minimal BERT-like structure...\n")
    
    class MinimalBertLayer(nn.Module):
        def __init__(self, hidden_size=128):
            super().__init__()
            self.attention = nn.MultiheadAttention(hidden_size, num_heads=2, batch_first=True)
            self.norm1 = nn.LayerNorm(hidden_size)
            self.ffn = nn.Sequential(
                nn.Linear(hidden_size, hidden_size * 4),
                nn.GELU(),
                nn.Linear(hidden_size * 4, hidden_size)
            )
            self.norm2 = nn.LayerNorm(hidden_size)
            
        def forward(self, x):
            # Self-attention
            attn_out, _ = self.attention(x, x, x)
            x = self.norm1(x + attn_out)
            # FFN
            ffn_out = self.ffn(x)
            x = self.norm2(x + ffn_out)
            return x
    
    class MinimalBert(nn.Module):
        def __init__(self):
            super().__init__()
            self.embeddings = nn.Embedding(1000, 128)
            self.layer1 = MinimalBertLayer()
            self.layer2 = MinimalBertLayer()
            
        def forward(self, input_ids):
            x = self.embeddings(input_ids)
            x = self.layer1(x)
            x = self.layer2(x)
            return x
    
    minimal_bert = MinimalBert()
    minimal_input = torch.randint(0, 1000, (1, 10))
    
    try:
        torch.onnx.export(minimal_bert, minimal_input, "temp/minimal_bert_test.onnx", 
                         export_modules_as_functions=True, verbose=False)
        print("✅ Minimal BERT-like model: export_modules_as_functions works")
    except Exception as e:
        print(f"❌ Minimal BERT-like model failed: {e}")
        
        # Try without MultiheadAttention
        print("\nTrying without MultiheadAttention...")
        
        class SimpleBertLayer(nn.Module):
            def __init__(self, hidden_size=128):
                super().__init__()
                # Replace MultiheadAttention with simple linear layers
                self.query = nn.Linear(hidden_size, hidden_size)
                self.key = nn.Linear(hidden_size, hidden_size)
                self.value = nn.Linear(hidden_size, hidden_size)
                self.output = nn.Linear(hidden_size, hidden_size)
                self.norm1 = nn.LayerNorm(hidden_size)
                self.ffn = nn.Sequential(
                    nn.Linear(hidden_size, hidden_size * 4),
                    nn.GELU(),
                    nn.Linear(hidden_size * 4, hidden_size)
                )
                self.norm2 = nn.LayerNorm(hidden_size)
                
            def forward(self, x):
                # Simplified attention
                q = self.query(x)
                k = self.key(x)
                v = self.value(x)
                # Simple attention (not accurate, just for testing)
                attn = torch.softmax(q @ k.transpose(-2, -1) / 128**0.5, dim=-1)
                attn_out = self.output(attn @ v)
                x = self.norm1(x + attn_out)
                # FFN
                ffn_out = self.ffn(x)
                x = self.norm2(x + ffn_out)
                return x
        
        class SimpleBert(nn.Module):
            def __init__(self):
                super().__init__()
                self.embeddings = nn.Embedding(1000, 128)
                self.layer1 = SimpleBertLayer()
                self.layer2 = SimpleBertLayer()
                
            def forward(self, input_ids):
                x = self.embeddings(input_ids)
                x = self.layer1(x)
                x = self.layer2(x)
                return x
        
        simple_bert = SimpleBert()
        
        try:
            torch.onnx.export(simple_bert, minimal_input, "temp/simple_bert_test.onnx", 
                             export_modules_as_functions=True, verbose=False)
            print("✅ Simple BERT (no MultiheadAttention): export_modules_as_functions works")
        except Exception as e:
            print(f"❌ Simple BERT also failed: {e}")

if __name__ == "__main__":
    trace_module_annotations()