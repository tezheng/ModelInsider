#!/usr/bin/env python3
"""
Test selective hook registration to avoid hanging during ONNX export
"""

import sys
sys.path.append('/mnt/d/BYOM/modelexport')

from transformers import AutoModel, AutoTokenizer
import torch
import torch.nn as nn
from pathlib import Path
import time

def test_selective_hooks():
    """Test different hook registration strategies"""
    
    print("🔍 Testing Selective Hook Registration Strategies")
    print("=" * 60)
    
    # Load model
    print("\n1. Loading BERT-tiny...")
    model = AutoModel.from_pretrained("prajjwal1/bert-tiny")
    model.eval()
    
    # Prepare input
    tokenizer = AutoTokenizer.from_pretrained("prajjwal1/bert-tiny")
    inputs = tokenizer("Hello world", return_tensors="pt", max_length=128, padding="max_length")
    input_ids = inputs["input_ids"]
    attention_mask = inputs["attention_mask"]
    
    # Test 1: No hooks (baseline)
    print("\n2. Testing ONNX export without hooks...")
    start_time = time.time()
    try:
        torch.onnx.export(
            model,
            (input_ids, attention_mask),
            "temp/test_no_hooks.onnx",
            input_names=['input_ids', 'attention_mask'],
            output_names=['last_hidden_state'],
            opset_version=17,
            verbose=False
        )
        print(f"   ✅ Success in {time.time() - start_time:.2f}s")
    except Exception as e:
        print(f"   ❌ Failed: {e}")
    
    # Test 2: Hooks only on HuggingFace modules
    print("\n3. Testing with hooks on HuggingFace modules only...")
    hooks = []
    hf_module_count = 0
    
    def dummy_hook(module, inputs, outputs):
        return outputs
    
    for name, module in model.named_modules():
        if name and not module.__class__.__module__.startswith('torch.nn'):
            hf_module_count += 1
            hook = module.register_forward_hook(dummy_hook)
            hooks.append(hook)
    
    print(f"   Registered {hf_module_count} hooks on HF modules")
    
    start_time = time.time()
    try:
        torch.onnx.export(
            model,
            (input_ids, attention_mask),
            "temp/test_hf_hooks.onnx",
            input_names=['input_ids', 'attention_mask'],
            output_names=['last_hidden_state'],
            opset_version=17,
            verbose=False
        )
        print(f"   ✅ Success in {time.time() - start_time:.2f}s")
    except Exception as e:
        print(f"   ❌ Failed: {e}")
    
    # Clean up hooks
    for hook in hooks:
        hook.remove()
    hooks.clear()
    
    # Test 3: Pre and post hooks on selected modules
    print("\n4. Testing with pre/post hooks on selected modules...")
    pre_hooks = []
    post_hooks = []
    selected_count = 0
    
    def pre_hook(module, inputs):
        pass  # Just a dummy pre-hook
    
    def post_hook(module, inputs, outputs):
        return outputs
    
    # Only hook major components
    major_components = ['embeddings', 'encoder', 'pooler', 'encoder.layer.0', 'encoder.layer.1']
    
    for name, module in model.named_modules():
        if name in major_components:
            selected_count += 1
            pre_h = module.register_forward_pre_hook(pre_hook)
            post_h = module.register_forward_hook(post_hook)
            pre_hooks.append(pre_h)
            post_hooks.append(post_h)
    
    print(f"   Registered {selected_count} pre/post hook pairs")
    
    start_time = time.time()
    try:
        torch.onnx.export(
            model,
            (input_ids, attention_mask),
            "temp/test_selected_hooks.onnx",
            input_names=['input_ids', 'attention_mask'],
            output_names=['last_hidden_state'],
            opset_version=17,
            verbose=False
        )
        print(f"   ✅ Success in {time.time() - start_time:.2f}s")
    except Exception as e:
        print(f"   ❌ Failed: {e}")
    
    # Clean up
    for hook in pre_hooks + post_hooks:
        hook.remove()
    
    # Test 4: Hook with stack operations (similar to our implementation)
    print("\n5. Testing with stack-based hooks...")
    tag_stack = ["/BertModel"]
    hook_count = 0
    hooks = []
    
    def stack_pre_hook(tag):
        def hook(module, inputs):
            tag_stack.append(tag)
        return hook
    
    def stack_post_hook(tag):
        def hook(module, inputs, outputs):
            if tag_stack and tag_stack[-1] == tag:
                tag_stack.pop()
            return outputs
        return hook
    
    # Register on a few modules only
    for name, module in model.named_modules():
        if name and not module.__class__.__module__.startswith('torch.nn'):
            if hook_count < 10:  # Limit to 10 modules
                tag = f"/BertModel/{module.__class__.__name__}"
                pre_h = module.register_forward_pre_hook(stack_pre_hook(tag))
                post_h = module.register_forward_hook(stack_post_hook(tag))
                hooks.extend([pre_h, post_h])
                hook_count += 1
    
    print(f"   Registered {hook_count} stack-based hook pairs")
    
    start_time = time.time()
    try:
        torch.onnx.export(
            model,
            (input_ids, attention_mask),
            "temp/test_stack_hooks.onnx",
            input_names=['input_ids', 'attention_mask'],
            output_names=['last_hidden_state'],
            opset_version=17,
            verbose=False
        )
        print(f"   ✅ Success in {time.time() - start_time:.2f}s")
    except Exception as e:
        print(f"   ❌ Failed: {e}")
    
    # Clean up
    for hook in hooks:
        hook.remove()
    
    print("\n✨ Testing complete!")

if __name__ == "__main__":
    # Create temp directory
    Path("temp").mkdir(exist_ok=True)
    test_selective_hooks()