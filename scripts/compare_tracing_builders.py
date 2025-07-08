#!/usr/bin/env python3
"""
Compare original vs optimized TracingHierarchyBuilder implementations.

Shows the difference in module hierarchy size and efficiency.
"""

import torch
from transformers import AutoModel
from modelexport.core.tracing_hierarchy_builder import TracingHierarchyBuilder
from modelexport.core.tracing_hierarchy_builder_optimized import OptimizedTracingHierarchyBuilder


def compare_builders():
    """Compare the two implementations."""
    print("🔬 Comparing TracingHierarchyBuilder Implementations")
    print("=" * 70)
    
    # Load model
    model = AutoModel.from_pretrained("prajjwal1/bert-tiny")
    model.eval()
    
    # Create example inputs
    input_ids = torch.randint(0, 1000, (1, 8))
    attention_mask = torch.ones((1, 8), dtype=torch.long)
    
    # Test original implementation
    print("\n📊 Original TracingHierarchyBuilder:")
    print("-" * 50)
    
    original_tracer = TracingHierarchyBuilder()
    original_tracer.trace_model_execution(model, (input_ids, attention_mask))
    original_summary = original_tracer.get_execution_summary()
    
    print(f"Total modules in hierarchy: {original_summary['total_modules']}")
    print(f"Modules traced: {original_summary['total_modules_traced']}")
    print(f"Execution steps: {original_summary['execution_steps']}")
    
    # Count module types in hierarchy
    hf_count = 0
    torch_nn_count = 0
    for path, metadata in original_summary['module_hierarchy'].items():
        if metadata['module_type'] == 'huggingface':
            hf_count += 1
        elif metadata['module_type'] == 'torch.nn':
            torch_nn_count += 1
    
    print(f"\nModule types in hierarchy:")
    print(f"  HuggingFace modules: {hf_count}")
    print(f"  torch.nn modules: {torch_nn_count}")
    
    # Show some unused modules
    unused_modules = []
    for path, metadata in original_summary['module_hierarchy'].items():
        if not metadata.get('traced') and metadata['module_type'] == 'torch.nn':
            unused_modules.append((path, metadata['class_name']))
    
    print(f"\n⚠️  Unused torch.nn modules in hierarchy: {len(unused_modules)}")
    if unused_modules:
        print("  Examples:")
        for path, class_name in unused_modules[:5]:
            print(f"    {path:50} ({class_name})")
        if len(unused_modules) > 5:
            print(f"    ... and {len(unused_modules) - 5} more")
    
    # Test optimized implementation
    print("\n\n📊 Optimized TracingHierarchyBuilder:")
    print("-" * 50)
    
    optimized_tracer = OptimizedTracingHierarchyBuilder()
    optimized_tracer.trace_model_execution(model, (input_ids, attention_mask))
    optimized_summary = optimized_tracer.get_execution_summary()
    
    print(f"Total modules in hierarchy: {optimized_summary['total_modules']}")
    print(f"Modules traced: {optimized_summary['total_modules_traced']}")
    print(f"Total modules available: {optimized_summary['total_modules_available']}")
    print(f"Execution steps: {optimized_summary['execution_steps']}")
    
    # Count module types in optimized hierarchy
    opt_hf_count = 0
    opt_torch_nn_count = 0
    for path, metadata in optimized_summary['module_hierarchy'].items():
        if metadata['module_type'] == 'huggingface':
            opt_hf_count += 1
        elif metadata['module_type'] == 'torch.nn':
            opt_torch_nn_count += 1
    
    print(f"\nModule types in hierarchy:")
    print(f"  HuggingFace modules: {opt_hf_count}")
    print(f"  torch.nn modules: {opt_torch_nn_count}")
    
    print(f"\nOptimization stats:")
    stats = optimized_summary['optimization_stats']
    print(f"  Modules excluded: {stats['modules_excluded']}")
    print(f"  Percentage excluded: {stats['percentage_excluded']:.1f}%")
    
    # Compare results
    print("\n\n📊 Comparison Summary:")
    print("=" * 70)
    
    print(f"Original hierarchy size: {original_summary['total_modules']} modules")
    print(f"Optimized hierarchy size: {optimized_summary['total_modules']} modules")
    print(f"Reduction: {original_summary['total_modules'] - optimized_summary['total_modules']} modules "
          f"({(1 - optimized_summary['total_modules'] / original_summary['total_modules']) * 100:.1f}%)")
    
    print(f"\ntorch.nn modules:")
    print(f"  Original: {torch_nn_count}")
    print(f"  Optimized: {opt_torch_nn_count}")
    print(f"  Removed: {torch_nn_count - opt_torch_nn_count} unnecessary modules")
    
    # Verify same HF modules are included
    print(f"\n✅ Verification:")
    print(f"  Same number of HF modules: {hf_count == opt_hf_count}")
    print(f"  Same execution trace length: {original_summary['execution_steps'] == optimized_summary['execution_steps']}")
    
    # Check that all traced modules have the same tags
    tag_mismatches = []
    for path in original_tracer.traced_modules:
        if path in optimized_tracer.traced_modules:
            orig_tag = original_summary['module_hierarchy'].get(path, {}).get('traced_tag')
            opt_tag = optimized_summary['module_hierarchy'].get(path, {}).get('traced_tag')
            if orig_tag != opt_tag:
                tag_mismatches.append((path, orig_tag, opt_tag))
    
    if tag_mismatches:
        print(f"\n❌ Tag mismatches found: {len(tag_mismatches)}")
        for path, orig, opt in tag_mismatches[:3]:
            print(f"  {path}: '{orig}' vs '{opt}'")
    else:
        print(f"  All traced modules have identical tags: ✓")
    
    # Show example of cleaned hierarchy
    print(f"\n🧹 Example of cleaned hierarchy (first 10 HF modules):")
    count = 0
    for path, metadata in sorted(optimized_summary['module_hierarchy'].items()):
        if metadata['module_type'] == 'huggingface' and count < 10:
            traced = '✓' if metadata.get('traced') else '✗'
            print(f"  {traced} {path:40} -> {metadata.get('traced_tag', 'N/A')}")
            count += 1
    
    print("\n🎉 Optimization Complete!")
    print("The optimized version only includes modules that are:")
    print("  1. Actually executed during forward pass")
    print("  2. Parents of executed modules (needed for tag generation)")
    print("This results in a cleaner, more efficient hierarchy!")


if __name__ == "__main__":
    compare_builders()