#!/usr/bin/env python3
"""
Deep analysis of cross-layer contamination patterns.
"""

import json

def analyze_contamination_patterns():
    """Analyze the specific patterns of cross-layer contamination."""
    
    print("=== DEEP CONTAMINATION PATTERN ANALYSIS ===\n")
    
    # Load both hierarchy files
    try:
        with open('temp/real_bert_old_hierarchy.json', 'r') as f:
            old_hierarchy = json.load(f)
    except FileNotFoundError:
        print("Old hierarchy file not found")
        return
        
    try:
        with open('temp/real_bert_new_hierarchy.json', 'r') as f:
            new_hierarchy = json.load(f)
    except FileNotFoundError:
        print("New hierarchy file not found")
        return
    
    def analyze_approach(hierarchy, name):
        print(f"=== {name.upper()} APPROACH CONTAMINATION PATTERNS ===")
        
        contamination_patterns = {
            'layer0_with_layer1_tags': [],
            'layer1_with_layer0_tags': [],
            'operation_types': {},
            'tag_mismatches': {}
        }
        
        node_tags = hierarchy.get('node_tags', {})
        
        for node_name, node_info in node_tags.items():
            tags = node_info.get('tags', [])
            op_type = node_info.get('op_type', 'Unknown')
            
            # Identify layer context from node name
            node_layer = None
            if '/layer.0/' in node_name or 'layer.0' in node_name:
                node_layer = '0'
            elif '/layer.1/' in node_name or 'layer.1' in node_name:
                node_layer = '1'
            
            if node_layer is None:
                continue
                
            # Check tags for opposite layer
            for tag in tags:
                tag_layer = None
                if any(part in tag for part in ['/BertLayer.0/', 'Layer.0', '/0/']):
                    tag_layer = '0'
                elif any(part in tag for part in ['/BertLayer.1/', 'Layer.1', '/1/']):
                    tag_layer = '1'
                
                if tag_layer is None:
                    continue
                    
                # Found contamination
                if node_layer != tag_layer:
                    contamination_info = {
                        'node': node_name,
                        'node_layer': node_layer,
                        'tag': tag,
                        'tag_layer': tag_layer,
                        'op_type': op_type
                    }
                    
                    if node_layer == '0' and tag_layer == '1':
                        contamination_patterns['layer0_with_layer1_tags'].append(contamination_info)
                    elif node_layer == '1' and tag_layer == '0':
                        contamination_patterns['layer1_with_layer0_tags'].append(contamination_info)
                    
                    # Track operation types
                    if op_type not in contamination_patterns['operation_types']:
                        contamination_patterns['operation_types'][op_type] = 0
                    contamination_patterns['operation_types'][op_type] += 1
        
        # Report findings
        print(f"Layer 0 operations with Layer 1 tags: {len(contamination_patterns['layer0_with_layer1_tags'])}")
        for item in contamination_patterns['layer0_with_layer1_tags'][:3]:
            print(f"  {item['op_type']}: {item['node']} → {item['tag']}")
        
        print(f"Layer 1 operations with Layer 0 tags: {len(contamination_patterns['layer1_with_layer0_tags'])}")
        for item in contamination_patterns['layer1_with_layer0_tags'][:3]:
            print(f"  {item['op_type']}: {item['node']} → {item['tag']}")
        
        print(f"Operation types involved in contamination:")
        for op_type, count in sorted(contamination_patterns['operation_types'].items()):
            print(f"  {op_type}: {count} cases")
        
        return contamination_patterns
    
    old_patterns = analyze_approach(old_hierarchy, "old")
    print()
    new_patterns = analyze_approach(new_hierarchy, "new")
    
    print(f"\n=== PATTERN COMPARISON ===")
    
    # Compare operation types
    old_ops = set(old_patterns['operation_types'].keys())
    new_ops = set(new_patterns['operation_types'].keys())
    
    print(f"Operation types in old approach: {old_ops}")
    print(f"Operation types in new approach: {new_ops}")
    print(f"Common contaminated operations: {old_ops & new_ops}")
    print(f"Old-only contaminated operations: {old_ops - new_ops}")
    print(f"New-only contaminated operations: {new_ops - old_ops}")
    
    # Look for specific patterns
    print(f"\n=== SPECIFIC CONTAMINATION ANALYSIS ===")
    
    def find_common_patterns(patterns, name):
        print(f"{name} patterns:")
        
        # Group by node path patterns
        path_patterns = {}
        for item in patterns['layer0_with_layer1_tags'] + patterns['layer1_with_layer0_tags']:
            # Extract path pattern
            parts = item['node'].split('/')
            if len(parts) >= 4:
                pattern = '/'.join(parts[:4])  # e.g., "/encoder/layer.0/attention"
                if pattern not in path_patterns:
                    path_patterns[pattern] = []
                path_patterns[pattern].append(item)
        
        for pattern, items in sorted(path_patterns.items()):
            print(f"  {pattern}: {len(items)} contamination cases")
            if len(items) > 0:
                example = items[0]
                print(f"    Example: {example['op_type']} operation")
    
    find_common_patterns(old_patterns, "Old approach")
    find_common_patterns(new_patterns, "New approach")
    
    print(f"\n=== ROOT CAUSE HYPOTHESIS ===")
    print("Based on the patterns, the contamination appears to involve:")
    print("1. Operations that create tensors used across layers (residual connections)")
    print("2. Shared operations between layers (like LayerNorm, dropout)")  
    print("3. Operations whose execution context spans multiple modules")
    print("4. Tensor operations that get traced during different layer executions")
    
    print(f"\nThe new approach provides better granularity but doesn't fully solve")
    print(f"the fundamental issue of operations that genuinely span multiple contexts.")

if __name__ == "__main__":
    analyze_contamination_patterns()