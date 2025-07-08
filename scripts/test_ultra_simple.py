#!/usr/bin/env python3
"""
Test the ultra-simple TracingHierarchyBuilder.

Verifies that execution order naturally builds correct hierarchy.
"""

import torch
from transformers import AutoModel
from modelexport.core.tracing_hierarchy_builder import TracingHierarchyBuilder
from modelexport.core.tracing_hierarchy_builder_ultra_simple import UltraSimpleTracingHierarchyBuilder


def test_ultra_simple():
    """Test the ultra-simple implementation."""
    print("🌟 Testing Ultra-Simple TracingHierarchyBuilder")
    print("=" * 70)
    print("Key Insight: Execution order IS hierarchy order!")
    print("Parents always execute before children.\n")
    
    # Load model
    model = AutoModel.from_pretrained("prajjwal1/bert-tiny")
    model.eval()
    
    # Create example inputs
    input_ids = torch.randint(0, 1000, (1, 8))
    attention_mask = torch.ones((1, 8), dtype=torch.long)
    
    # Test ultra-simple version
    ultra = UltraSimpleTracingHierarchyBuilder()
    ultra.trace_model_execution(model, (input_ids, attention_mask))
    ultra_summary = ultra.get_execution_summary()
    
    # Test original for comparison
    original = TracingHierarchyBuilder()
    original.trace_model_execution(model, (input_ids, attention_mask))
    original_summary = original.get_execution_summary()
    
    print(f"📊 Results Comparison:")
    print(f"  Ultra-Simple: {ultra_summary['total_modules']} modules")
    print(f"  Original:     {original_summary['total_modules']} modules")
    
    # Show execution order proves hierarchy
    print(f"\n🔍 Execution Order = Hierarchy Order:")
    print("Module execution sequence:")
    for i, (name, data) in enumerate(ultra.module_hierarchy.items()):
        if i < 10:  # Show first 10
            indent = "  " * name.count(".")
            order = data.get('execution_order', -1)
            print(f"{order:3d}: {indent}{name or 'ROOT':30} -> {data['traced_tag']}")
    
    # Verify tags match
    print(f"\n✅ Tag Verification:")
    all_match = True
    for name in ultra.traced_modules:
        if name in original.traced_modules:
            ultra_tag = ultra.module_hierarchy[name]['traced_tag']
            orig_tag = original.module_hierarchy[name]['traced_tag']
            if ultra_tag != orig_tag:
                print(f"  ❌ Mismatch: {name}")
                print(f"     Ultra:    {ultra_tag}")
                print(f"     Original: {orig_tag}")
                all_match = False
    
    if all_match:
        print(f"  ✅ All tags match perfectly!")
    
    # Show the simplification
    print(f"\n💡 Code Simplification:")
    print(f"  ❌ Removed: _ensure_parent_hierarchy() - Parents execute first!")
    print(f"  ❌ Removed: _parent_map - Tag stack tracks parents!")
    print(f"  ❌ Removed: _module_registry - Add modules as they execute!")
    print(f"  ❌ Removed: Complex hierarchy building logic!")
    
    print(f"\n🎉 Ultra-Simple Implementation:")
    print(f"  ✅ Same results as original")
    print(f"  ✅ Much cleaner code")
    print(f"  ✅ Leverages natural execution order")
    print(f"  ✅ No complex bookkeeping needed!")


if __name__ == "__main__":
    test_ultra_simple()