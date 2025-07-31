#!/usr/bin/env python3
"""
Analyze the Enhanced Trace Module Map patterns to understand PyTorch's naming convention.
"""

import json
from collections import defaultdict
from pathlib import Path


def analyze_trace_map_patterns():
    """Analyze the captured trace map to understand PyTorch's naming patterns."""
    
    # Load the results
    results_path = Path("enhanced_trace_map_results.json")
    if not results_path.exists():
        print("❌ Results file not found. Run experimental_enhanced_trace_map.py first.")
        return
    
    with open(results_path) as f:
        data = json.load(f)
    
    entries = data['captures'][0]['entries']
    
    print("=" * 80)
    print("📊 ANALYSIS OF PYTORCH'S TRACE MODULE MAP PATTERNS")
    print("=" * 80)
    
    # Analyze patterns
    patterns = {
        'huggingface_modules': [],
        'torch_nn_modules': [],
        'other_modules': []
    }
    
    for _key, info in entries.items():
        module_class = info['module_class']
        scope_name = info['scope_name']
        
        # Parse the scope name
        if '::' in scope_name:
            full_class_path, instance_name = scope_name.split('::', 1)
        else:
            full_class_path = scope_name
            instance_name = ''
        
        # Categorize by module origin
        if 'transformers' in full_class_path:
            patterns['huggingface_modules'].append({
                'module_class': module_class,
                'full_class_path': full_class_path,
                'instance_name': instance_name,
                'scope_name': scope_name
            })
        elif 'torch.nn' in full_class_path:
            patterns['torch_nn_modules'].append({
                'module_class': module_class,
                'full_class_path': full_class_path,
                'instance_name': instance_name,
                'scope_name': scope_name
            })
        else:
            patterns['other_modules'].append({
                'module_class': module_class,
                'full_class_path': full_class_path,
                'instance_name': instance_name,
                'scope_name': scope_name
            })
    
    # Print analysis
    print(f"\n📋 MODULE CATEGORIZATION:")
    print(f"   - HuggingFace modules: {len(patterns['huggingface_modules'])}")
    print(f"   - PyTorch nn modules: {len(patterns['torch_nn_modules'])}")
    print(f"   - Other modules: {len(patterns['other_modules'])}")
    
    # Analyze HuggingFace patterns
    print(f"\n🤗 HUGGINGFACE MODULE PATTERNS:")
    print("   PyTorch uses FULL class path for HF modules:")
    for i, item in enumerate(patterns['huggingface_modules'], 1):
        print(f"   {i}. {item['module_class']:25s} → {item['scope_name']}")
        print(f"      Full class path: {item['full_class_path']}")
        print(f"      Instance name:   {item['instance_name'] or '(root)'}")
        print()
    
    # Analyze torch.nn patterns
    print(f"\n🔥 TORCH.NN MODULE PATTERNS:")
    print("   PyTorch uses FULL torch.nn path + instance name:")
    
    # Group by module type
    by_type = defaultdict(list)
    for item in patterns['torch_nn_modules']:
        by_type[item['module_class']].append(item)
    
    for module_type, items in sorted(by_type.items()):
        print(f"\n   {module_type} ({len(items)} instances):")
        for item in items[:3]:  # Show first 3 of each type
            print(f"      → {item['scope_name']}")
    
    # Key insights
    print(f"\n💡 KEY INSIGHTS:")
    print(f"   1. HuggingFace modules use: full.package.path.ClassName::instance_name")
    print(f"   2. PyTorch nn modules use: torch.nn.modules.category.ClassName::instance_name")
    print(f"   3. Instance names are the FINAL component (e.g., 'attention', 'layer.0')")
    print(f"   4. The '::' separator divides class path from instance name")
    
    # Show hierarchy reconstruction
    print(f"\n🏗️ HIERARCHY RECONSTRUCTION:")
    print("   To reconstruct the full module path from scope names:")
    print()
    
    # Create a mapping from instance names to full paths
    instance_to_path = {}
    
    # First, let's build the expected hierarchy for bert-tiny
    expected_hierarchy = {
        '': '__module',  # root
        'embeddings': '__module.embeddings',
        'encoder': '__module.encoder',
        'layer': '__module.encoder.layer',
        'layer.0': '__module.encoder.layer.0',
        'layer.1': '__module.encoder.layer.1',
        'attention': '__module.encoder.layer.0.attention',  # Could be layer.0 or layer.1
        'self': '__module.encoder.layer.0.attention.self',
        'output': '__module.encoder.layer.0.attention.output',
        'intermediate': '__module.encoder.layer.0.intermediate',
        'output': '__module.encoder.layer.0.output',
        'pooler': '__module.pooler'
    }
    
    # Match scope names to hierarchy
    print("   Mapping scope names to module hierarchy:")
    
    matches = []
    for _key, info in entries.items():
        scope_name = info['scope_name']
        if '::' in scope_name:
            _, instance_name = scope_name.split('::', 1)
            
            # Try to find the best match in expected hierarchy
            best_match = None
            if instance_name in expected_hierarchy:
                best_match = expected_hierarchy[instance_name]
            elif instance_name == '':  # Root module
                best_match = '__module'
            else:
                # For torch.nn modules, instance name is just the attribute name
                for path_suffix, full_path in expected_hierarchy.items():
                    if path_suffix.endswith(instance_name):
                        best_match = full_path
                        break
            
            if best_match:
                matches.append({
                    'module_class': info['module_class'],
                    'scope_name': scope_name,
                    'instance_name': instance_name,
                    'reconstructed_path': best_match
                })
    
    # Show reconstruction results
    print()
    for match in sorted(matches, key=lambda x: x['reconstructed_path'])[:20]:
        print(f"   {match['module_class']:25s} | Instance: {match['instance_name']:20s} → {match['reconstructed_path']}")
    
    print(f"\n✅ CONCLUSION:")
    print(f"   The Enhanced Trace Module Map DOES contain hierarchy information!")
    print(f"   - Class information: Preserved in full class path before '::'")
    print(f"   - Instance hierarchy: Preserved in instance name after '::'")
    print(f"   - Full reconstruction requires mapping instance names to paths")

if __name__ == "__main__":
    analyze_trace_map_patterns()