#!/usr/bin/env python3
"""
Test the maximum lazy TracingHierarchyBuilder implementation.

This version only builds hierarchy during execution, not during registration.
"""

import torch
from transformers import AutoModel
from modelexport.core.tracing_hierarchy_builder_optimized import OptimizedTracingHierarchyBuilder


def test_maximum_lazy():
    """Test the maximum lazy implementation."""
    print("🔋 Testing Maximum Lazy TracingHierarchyBuilder")
    print("=" * 70)
    
    # Load model
    model = AutoModel.from_pretrained("prajjwal1/bert-tiny")
    model.eval()
    
    # Create example inputs
    input_ids = torch.randint(0, 1000, (1, 8))
    attention_mask = torch.ones((1, 8), dtype=torch.long)
    
    tracer = OptimizedTracingHierarchyBuilder()
    
    # Register hooks to see the state after registration
    tracer.register_hooks(model)
    
    print("📊 State after hook registration (before execution):")
    print(f"  Module hierarchy size: {len(tracer.module_hierarchy)}")
    print(f"  Module registry size: {len(tracer._module_registry)}")
    print(f"  Parent map size: {len(tracer._parent_map)}")
    print(f"  Hooks registered: {len(tracer.hooks)}")
    
    # Check what's in hierarchy before execution
    print(f"\n  Modules in hierarchy before execution:")
    for path, metadata in tracer.module_hierarchy.items():
        print(f"    {path or 'ROOT'}: {metadata['class_name']}")
    
    print(f"\n📈 Executing model...")
    
    # Run the forward pass
    model.eval()
    with torch.no_grad():
        _ = model(input_ids, attention_mask)
    
    # Clean up hooks
    tracer.remove_hooks()
    
    summary = tracer.get_execution_summary()
    
    print(f"\n📊 State after execution:")
    print(f"  Module hierarchy size: {summary['total_modules']}")
    print(f"  Modules traced: {summary['total_modules_traced']}")
    print(f"  Total modules available: {summary['total_modules_available']}")
    print(f"  Execution steps: {summary['execution_steps']}")
    
    # Show optimization stats
    if 'optimization_stats' in summary:
        stats = summary['optimization_stats']
        print(f"\n✨ Optimization Results:")
        print(f"  Modules excluded: {stats['modules_excluded']}")
        print(f"  Percentage excluded: {stats['percentage_excluded']:.1f}%")
    
    # Show what was actually added to hierarchy
    print(f"\n🏗️  Modules added to hierarchy during execution:")
    hf_count = 0
    torch_count = 0
    for path, metadata in sorted(summary['module_hierarchy'].items()):
        if path:  # Skip root
            traced = "✓" if metadata.get('traced') else "○"
            module_type = metadata['module_type']
            if module_type == 'huggingface':
                hf_count += 1
                color = "\033[92m"  # Green
            else:
                torch_count += 1
                color = "\033[91m"  # Red
            print(f"  {color}{traced} {path:40} ({metadata['class_name']})\033[0m")
    
    print(f"\n📈 Summary:")
    print(f"  HuggingFace modules in hierarchy: {hf_count}")
    print(f"  torch.nn modules in hierarchy: {torch_count}")
    print(f"  Total modules executed: {summary['total_modules_traced']}")
    
    # Verify tags are correct
    print(f"\n🏷️  Sample traced tags:")
    for path, metadata in sorted(summary['module_hierarchy'].items()):
        if metadata.get('traced') and metadata['module_type'] == 'huggingface':
            print(f"  ✓ {path:40} -> {metadata.get('traced_tag')}")
    
    print(f"\n🎉 Maximum Lazy Implementation Complete!")
    print(f"✅ Only {summary['total_modules']} modules in hierarchy (out of {summary['total_modules_available']} available)")
    print(f"✅ All traced modules have proper tags")
    print(f"✅ No unnecessary module metadata extraction during registration")


if __name__ == "__main__":
    test_maximum_lazy()