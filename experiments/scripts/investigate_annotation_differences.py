#!/usr/bin/env python3
"""Investigate the specific annotation differences causing the export failure."""

import torch
from transformers import AutoModel
import warnings
warnings.filterwarnings("ignore")

def investigate_annotation_differences():
    """Deep dive into annotation differences between module instances."""
    
    print("🔍 Investigating type annotation differences in BERT-tiny...\n")
    
    # Load BERT-tiny
    model = AutoModel.from_pretrained("prajjwal1/bert-tiny")
    model.eval()
    
    # Focus on Linear layers which have many instances
    linear_modules = []
    for name, module in model.named_modules():
        if module.__class__.__name__ == 'Linear':
            linear_modules.append((name, module))
    
    print(f"Found {len(linear_modules)} Linear modules")
    
    # Compare annotations
    print("\n📝 Comparing Linear module annotations:\n")
    
    # Get the first Linear module as reference
    if linear_modules:
        ref_name, ref_module = linear_modules[0]
        ref_annotations = ref_module.__annotations__ if hasattr(ref_module, '__annotations__') else {}
        
        print(f"Reference: {ref_name}")
        print(f"Annotations: {list(ref_annotations.keys())}\n")
        
        # Check each Linear module for differences
        for name, module in linear_modules[1:]:
            if hasattr(module, '__annotations__'):
                current_annotations = module.__annotations__
                
                # Check for differences
                ref_keys = set(ref_annotations.keys())
                current_keys = set(current_annotations.keys())
                
                if ref_keys != current_keys:
                    print(f"⚠️  DIFFERENCE FOUND in {name}:")
                    print(f"   Missing: {ref_keys - current_keys}")
                    print(f"   Extra: {current_keys - ref_keys}")
                
                # Check for value differences
                for key in ref_keys & current_keys:
                    if ref_annotations[key] != current_annotations[key]:
                        print(f"⚠️  TYPE DIFFERENCE in {name}.{key}:")
                        print(f"   Reference: {ref_annotations[key]}")
                        print(f"   Current: {current_annotations[key]}")
    
    # Check for the 'bias' attribute specifically
    print("\n🔎 Checking 'bias' attribute (common issue):\n")
    
    for name, module in linear_modules:
        has_bias = hasattr(module, 'bias') and module.bias is not None
        bias_in_annotations = 'bias' in module.__annotations__ if hasattr(module, '__annotations__') else False
        
        print(f"{name}:")
        print(f"  Has bias parameter: {has_bias}")
        print(f"  'bias' in annotations: {bias_in_annotations}")
        
        if has_bias and not bias_in_annotations:
            print(f"  ⚠️  MISMATCH: Has bias but not in annotations!")
        elif not has_bias and bias_in_annotations:
            print(f"  ⚠️  MISMATCH: No bias but in annotations!")
    
    # Check Embedding modules for 'freeze' attribute
    print("\n🔎 Checking Embedding 'freeze' attribute:\n")
    
    embedding_modules = []
    for name, module in model.named_modules():
        if module.__class__.__name__ == 'Embedding':
            embedding_modules.append((name, module))
            
    for name, module in embedding_modules:
        print(f"{name}:")
        if hasattr(module, '__annotations__'):
            print(f"  Annotations: {list(module.__annotations__.keys())}")
        if hasattr(module, 'freeze'):
            print(f"  Has 'freeze' attribute: {module.freeze}")
        
    # Test theory: Remove problematic annotations
    print("\n🧪 Testing theory: Can we fix by modifying annotations?\n")
    
    # Create a test model
    test_model = AutoModel.from_pretrained("prajjwal1/bert-tiny")
    test_model.eval()
    
    # Try to standardize annotations
    for name, module in test_model.named_modules():
        if hasattr(module, '__annotations__'):
            # Remove module-level annotations that might differ
            if 'freeze' in module.__annotations__:
                print(f"Removing 'freeze' annotation from {name}")
                del module.__annotations__['freeze']
    
    # Try export
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained("prajjwal1/bert-tiny")
    inputs = tokenizer(["Hello world"], return_tensors="pt", padding=True, truncation=True, max_length=32)
    sample_input = inputs['input_ids']
    
    try:
        torch.onnx.export(test_model, sample_input, "temp/test_fixed.onnx", 
                         export_modules_as_functions=True, verbose=False)
        print("✅ Export succeeded after removing 'freeze' annotations!")
    except Exception as e:
        print(f"❌ Export still failed: {str(e)[:200]}...")

if __name__ == "__main__":
    investigate_annotation_differences()