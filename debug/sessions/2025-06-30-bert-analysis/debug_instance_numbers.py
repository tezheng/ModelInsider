#!/usr/bin/env python3
"""
Debug why instance numbers aren't being detected correctly.
"""

def debug_tag_generation(full_path: str, module_hierarchy: dict):
    """Debug the tag generation for a specific path."""
    print(f"\n🔍 Debugging: {full_path}")
    
    # Check if this module should be filtered
    module_data = module_hierarchy.get(full_path)
    if not module_data:
        print(f"❌ No module data found for {full_path}")
        return ""
    
    print(f"📝 Module data: {module_data['class_name']}, filtered={module_data['should_filter']}")
    
    if module_data['should_filter']:
        print("🚫 Module is filtered - returning empty tag")
        return ""  # Empty tag for filtered torch.nn modules
    
    # Build hierarchy by walking path segments from root to current
    path_segments = full_path.split('.')
    hierarchy_parts = []
    
    print(f"📋 Path segments: {path_segments}")
    
    # Walk each segment and build cumulative path
    i = 0
    while i < len(path_segments):
        segment = path_segments[i]
        print(f"\n  Step {i}: Processing segment '{segment}'")
        
        # Check if this segment is a digit (instance number) FIRST
        if segment.isdigit():
            # This is an instance number - append to the previous class name
            print(f"    🔢 Detected digit: {segment}")
            if hierarchy_parts:
                old_name = hierarchy_parts[-1]
                hierarchy_parts[-1] += f".{segment}"
                print(f"    ✅ Updated '{old_name}' to '{hierarchy_parts[-1]}'")
            else:
                print(f"    ❌ No previous class name to append digit to")
        else:
            # Build cumulative path to this point for non-digit segments
            current_path = '.'.join(path_segments[:i+1])
            current_module_data = module_hierarchy.get(current_path)
            
            print(f"    Current path: {current_path}")
            print(f"    Module data exists: {current_module_data is not None}")
            
            if current_module_data:
                print(f"    Class name: {current_module_data['class_name']}")
                print(f"    Should filter: {current_module_data['should_filter']}")
            
            if current_module_data and not current_module_data['should_filter']:
                class_name = current_module_data['class_name']
                hierarchy_parts.append(class_name)
                print(f"    ✅ Added '{class_name}' to hierarchy")
            else:
                print(f"    ⏭️  Skipping segment (filtered or no data)")
        
        i += 1
    
    # Return full hierarchy path from root to leaf
    result = "/" + "/".join(hierarchy_parts) if hierarchy_parts else ""
    print(f"\n🎯 Final result: '{result}'")
    return result


if __name__ == "__main__":
    # Load the actual hierarchy metadata
    import json
    with open('/mnt/d/BYOM/modelexport/temp/bert_tiny_universal_export_hierarchy_metadata.json', 'r') as f:
        metadata = json.load(f)
    
    module_hierarchy = metadata['module_hierarchy']
    
    # Test the problematic cases
    test_cases = [
        "__module.encoder.layer.0",
        "__module.encoder.layer.1",
        "__module.encoder.layer.0.attention",
        "__module.encoder.layer.1.attention"
    ]
    
    for test_case in test_cases:
        debug_tag_generation(test_case, module_hierarchy)
    
    print("\n" + "="*60)
    print("🔍 ANALYSIS COMPLETE")