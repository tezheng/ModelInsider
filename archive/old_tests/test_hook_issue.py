#!/usr/bin/env python3
"""
Debug the hanging issue with universal hierarchy exporter hooks
"""

import sys
sys.path.append('/mnt/d/BYOM/modelexport')

from transformers import AutoModel, AutoTokenizer
import torch
import torch.nn as nn
from pathlib import Path
import time

def test_problematic_pattern():
    """Test the specific pattern that might be causing issues"""
    
    print("🐛 Debugging Hook Hanging Issue")
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
    
    # Test: Hooks with module path lookup (similar to our _get_module_by_path)
    print("\n2. Testing with module path lookup pattern...")
    
    def get_module_by_path(model, path):
        """Get module by its dotted path."""
        if not path:
            return model
            
        parts = path.split('.')
        current = model
        
        for part in parts:
            if hasattr(current, part):
                current = getattr(current, part)
            else:
                return None
        
        return current
    
    # Build module hierarchy first
    module_hierarchy = {}
    for name, module in model.named_modules():
        if name:
            module_hierarchy[name] = {
                'module': module,
                'class_name': module.__class__.__name__
            }
    
    print(f"   Found {len(module_hierarchy)} modules")
    
    # Register hooks using the hierarchy
    hooks = []
    hook_count = 0
    
    def create_hook(module_name):
        def hook(module, inputs, outputs):
            # Just access the name, don't do anything complex
            _ = module_name
            return outputs
        return hook
    
    for path, data in module_hierarchy.items():
        # Try to get module by path (this might be the issue)
        module = get_module_by_path(model, path)
        if module is not None:
            hook = module.register_forward_hook(create_hook(path))
            hooks.append(hook)
            hook_count += 1
            
            # Limit hooks to test
            if hook_count >= 20:
                break
    
    print(f"   Registered {hook_count} hooks")
    
    # Try export
    start_time = time.time()
    try:
        torch.onnx.export(
            model,
            (input_ids, attention_mask),
            "temp/test_hierarchy_hooks.onnx",
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
    
    # Test: The exact pattern from universal exporter
    print("\n3. Testing exact universal exporter pattern...")
    
    # Simulate the _module_hierarchy structure
    _module_hierarchy = {}
    _module_hierarchy['__module'] = {
        'name': '',
        'class_name': 'BertModel',
        'should_filter': False,
        'expected_tag': '/BertModel'
    }
    
    for name, module in model.named_modules():
        if name:
            full_path = f"__module.{name}"
            _module_hierarchy[full_path] = {
                'name': name,
                'class_name': module.__class__.__name__,
                'should_filter': module.__class__.__module__.startswith('torch.nn'),
                'expected_tag': f"/BertModel/{module.__class__.__name__}"
            }
    
    # Register hooks like universal exporter
    _pre_hooks = []
    _post_hooks = []
    _tag_stack = ["/BertModel"]
    _operation_context = {}
    
    def _create_pre_hook(module_name, expected_tag):
        def pre_hook(module, inputs):
            _tag_stack.append(expected_tag)
            _operation_context[module_name] = {
                "tag": expected_tag,
                "creates_hierarchy": True,
                "stack_depth": len(_tag_stack),
                "module_class": module.__class__.__name__
            }
        return pre_hook
    
    def _create_post_hook(module_name, expected_tag):
        def post_hook(module, inputs, outputs):
            if _tag_stack and _tag_stack[-1] == expected_tag:
                _tag_stack.pop()
            return outputs
        return post_hook
    
    hook_count = 0
    for full_path, module_data in _module_hierarchy.items():
        if full_path == "__module":
            continue
            
        # Get the actual module
        path = full_path.replace("__module.", "")
        module = get_module_by_path(model, path)
        if module is None:
            continue
        
        module_name = module_data['name']
        should_filter = module_data['should_filter']
        expected_tag = module_data['expected_tag']
        
        if not should_filter and hook_count < 10:  # Limit for testing
            pre_hook = module.register_forward_pre_hook(
                _create_pre_hook(module_name, expected_tag)
            )
            _pre_hooks.append(pre_hook)
            
            post_hook = module.register_forward_hook(
                _create_post_hook(module_name, expected_tag)
            )
            _post_hooks.append(post_hook)
            hook_count += 1
    
    print(f"   Registered {hook_count} pre/post hook pairs")
    
    # Try export
    start_time = time.time()
    try:
        torch.onnx.export(
            model,
            (input_ids, attention_mask),
            "temp/test_exact_pattern.onnx",
            input_names=['input_ids', 'attention_mask'],
            output_names=['last_hidden_state'],
            opset_version=17,
            verbose=False
        )
        print(f"   ✅ Success in {time.time() - start_time:.2f}s")
    except Exception as e:
        print(f"   ❌ Failed: {e}")
    
    # Clean up
    for hook in _pre_hooks + _post_hooks:
        hook.remove()
    
    print("\n✨ Debugging complete!")

if __name__ == "__main__":
    Path("temp").mkdir(exist_ok=True)
    test_problematic_pattern()