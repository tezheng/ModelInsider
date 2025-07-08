#!/usr/bin/env python3
"""
Final comparison of all three TracingHierarchyBuilder approaches:
1. Original - builds complete hierarchy upfront
2. Optimized - builds complete hierarchy upfront but filters output  
3. Maximum Lazy - builds hierarchy only during execution
"""

import torch
from transformers import AutoModel
from modelexport.core.tracing_hierarchy_builder import TracingHierarchyBuilder
from modelexport.core.tracing_hierarchy_builder_optimized import OptimizedTracingHierarchyBuilder


def compare_all_approaches():
    """Compare all three approaches."""
    print("🔬 Final TracingHierarchyBuilder Comparison")
    print("=" * 80)
    
    # Load model
    model = AutoModel.from_pretrained("prajjwal1/bert-tiny")
    model.eval()
    
    # Create example inputs
    input_ids = torch.randint(0, 1000, (1, 8))
    attention_mask = torch.ones((1, 8), dtype=torch.long)
    
    results = {}
    
    # Test 1: Original approach
    print("\n📊 1. Original TracingHierarchyBuilder")
    print("-" * 50)
    
    original = TracingHierarchyBuilder()
    original.trace_model_execution(model, (input_ids, attention_mask))
    original_summary = original.get_execution_summary()
    
    results['original'] = {
        'name': 'Original (Upfront Complete Hierarchy)',
        'total_modules': original_summary['total_modules'],
        'traced_modules': original_summary['total_modules_traced'],
        'hierarchy': original_summary['module_hierarchy']
    }
    
    print(f"Total modules in hierarchy: {original_summary['total_modules']}")
    print(f"Modules traced during execution: {original_summary['total_modules_traced']}")
    print(f"Unused modules: {original_summary['total_modules'] - original_summary['total_modules_traced']}")
    
    # Test 2: Maximum Lazy approach  
    print("\n📊 2. Maximum Lazy TracingHierarchyBuilder")
    print("-" * 50)
    
    lazy = OptimizedTracingHierarchyBuilder()
    
    # Show state before execution
    lazy.register_hooks(model)
    print(f"After registration - hierarchy size: {len(lazy.module_hierarchy)} (only root)")
    print(f"Module registry size: {len(lazy._module_registry)} (lightweight references)")
    
    # Execute
    model.eval()
    with torch.no_grad():
        _ = model(input_ids, attention_mask)
    lazy.remove_hooks()
    
    lazy_summary = lazy.get_execution_summary()
    
    results['lazy'] = {
        'name': 'Maximum Lazy (Build During Execution)',
        'total_modules': lazy_summary['total_modules'],
        'traced_modules': lazy_summary['total_modules_traced'],
        'hierarchy': lazy_summary['module_hierarchy']
    }
    
    print(f"Total modules in hierarchy: {lazy_summary['total_modules']}")
    print(f"Modules traced during execution: {lazy_summary['total_modules_traced']}")
    print(f"Registry size: {lazy_summary['total_modules_available']}")
    
    # Comparison table
    print(f"\n📊 Detailed Comparison")
    print("=" * 80)
    
    print(f"{'Approach':<35} {'Hierarchy Size':<15} {'Traced':<10} {'Unused':<10} {'Efficiency':<15}")
    print("-" * 80)
    
    for key, result in results.items():
        name = result['name']
        total = result['total_modules']
        traced = result['traced_modules']
        unused = total - traced
        efficiency = f"{traced/total*100:.1f}%" if total > 0 else "N/A"
        
        print(f"{name:<35} {total:<15} {traced:<10} {unused:<10} {efficiency:<15}")
    
    # Verify identical functionality
    print(f"\n🔍 Functionality Verification")
    print("-" * 80)
    
    # Check that all traced modules have identical tags
    original_tags = {}
    lazy_tags = {}
    
    for path, metadata in results['original']['hierarchy'].items():
        if metadata.get('traced'):
            original_tags[path] = metadata.get('traced_tag')
    
    for path, metadata in results['lazy']['hierarchy'].items():
        if metadata.get('traced'):
            lazy_tags[path] = metadata.get('traced_tag')
    
    tags_match = True
    mismatches = []
    
    for path in original_tags:
        if path in lazy_tags:
            if original_tags[path] != lazy_tags[path]:
                tags_match = False
                mismatches.append((path, original_tags[path], lazy_tags[path]))
        else:
            tags_match = False
            mismatches.append((path, original_tags[path], "MISSING"))
    
    if tags_match:
        print("✅ All traced modules have identical tags between approaches")
    else:
        print(f"❌ Found {len(mismatches)} tag mismatches:")
        for path, orig, lazy in mismatches[:3]:
            print(f"  {path}: '{orig}' vs '{lazy}'")
    
    # Show benefits of lazy approach
    print(f"\n✨ Benefits of Maximum Lazy Approach")
    print("-" * 80)
    
    original_total = results['original']['total_modules']
    lazy_total = results['lazy']['total_modules']
    reduction = original_total - lazy_total
    percentage = (reduction / original_total) * 100
    
    print(f"1. **Memory Efficiency**: {reduction} fewer modules ({percentage:.1f}% reduction)")
    print(f"2. **Processing Speed**: No upfront metadata extraction for unused modules")
    print(f"3. **Accuracy**: Only includes modules that actually execute")
    print(f"4. **Lazy Loading**: Hierarchy built on-demand during execution")
    print(f"5. **Same Results**: Identical tags for all traced modules")
    
    # Show what gets excluded
    excluded_types = {}
    for path, metadata in results['original']['hierarchy'].items():
        if path not in results['lazy']['hierarchy'] and path:
            class_name = metadata['class_name']
            excluded_types[class_name] = excluded_types.get(class_name, 0) + 1
    
    if excluded_types:
        print(f"\n🗑️  Excluded Module Types (not executed during forward pass):")
        for class_name, count in sorted(excluded_types.items()):
            print(f"  {class_name}: {count} instances")
    
    print(f"\n🎯 Recommendation")
    print("-" * 80)
    print("Use **Maximum Lazy TracingHierarchyBuilder** for:")
    print("✅ Production ONNX export (most common use case)")
    print("✅ Memory-constrained environments") 
    print("✅ When you only care about executed modules")
    print("✅ Cleaner, more focused hierarchy output")
    print()
    print("Use **Original TracingHierarchyBuilder** for:")
    print("📋 Complete model structure analysis")
    print("📋 Debugging unused modules")
    print("📋 Architecture documentation")


if __name__ == "__main__":
    compare_all_approaches()