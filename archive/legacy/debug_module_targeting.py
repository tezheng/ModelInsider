#!/usr/bin/env python3
"""
Debug Module Targeting - Confirm exactly which nn.Module we're targeting
"""

from enhanced_dag_extractor import EnhancedDAGExtractor
from input_generator import UniversalInputGenerator
from transformers import AutoModel


def debug_module_targeting():
    """Debug exactly which modules we're targeting"""
    print("🔍 DEBUGGING MODULE TARGETING")
    print("=" * 60)
    
    # Load BERT model
    model = AutoModel.from_pretrained('google/bert_uncased_L-2_H-128_A-2')
    
    print("📍 STANDALONE MODULE TARGET:")
    print("Module path: encoder.layer.0.attention.self")
    
    # Get the actual module
    standalone_module = model.encoder.layer[0].attention.self
    print(f"Actual module type: {type(standalone_module).__name__}")
    print(f"Module parameters: {sum(p.numel() for p in standalone_module.parameters()):,}")
    
    print("\nModule structure:")
    for name, submodule in standalone_module.named_children():
        param_count = sum(p.numel() for p in submodule.parameters())
        print(f"  {name}: {type(submodule).__name__} ({param_count:,} params)")
    
    print("\nModule parameters:")
    for name, param in standalone_module.named_parameters():
        print(f"  {name}: {list(param.shape)}")
    
    print("\n" + "="*60)
    print("📍 EXTRACTED SUBGRAPH TARGET:")
    
    # Check what tags we actually have for this module
    generator = UniversalInputGenerator()
    inputs = generator.generate_inputs(model, 'google/bert_uncased_L-2_H-128_A-2')
    
    extractor = EnhancedDAGExtractor()
    extractor.analyze_model_structure(model)
    
    # Find hierarchy path for encoder.layer.0.attention.self
    target_module_hierarchy = None
    
    print("Looking for hierarchy path for 'encoder.layer.0.attention.self'...")
    for module_name, module_info in extractor.module_hierarchy.items():
        if module_name == "encoder.layer.0.attention.self":
            target_module_hierarchy = module_info['hierarchy_path']
            print(f"Found hierarchy path: {target_module_hierarchy}")
            break
    
    if not target_module_hierarchy:
        print("❌ Could not find hierarchy path!")
        print("\nAvailable modules containing 'attention.self':")
        for module_name, module_info in extractor.module_hierarchy.items():
            if 'attention' in module_name and 'self' in module_name:
                print(f"  {module_name} → {module_info['hierarchy_path']}")
    
    # Now run full analysis to see what operations get tagged
    print("\n" + "="*60)
    print("📍 OPERATION TAGGING ANALYSIS:")
    
    extractor.trace_execution_with_hooks(model, inputs)
    extractor.create_parameter_mapping(model)
    
    # Create temporary ONNX to analyze tagging
    import os
    import tempfile
    with tempfile.NamedTemporaryFile(suffix='.onnx', delete=False) as tmp:
        temp_path = tmp.name
    
    try:
        extractor.export_and_analyze_onnx(model, inputs, temp_path)
        
        # Look for operations tagged with our target module
        if target_module_hierarchy:
            print(f"\nOperations tagged with '{target_module_hierarchy}':")
            
            count = 0
            for op_name, op_data in extractor.operation_metadata.items():
                if target_module_hierarchy in op_data.get('tags', []):
                    count += 1
                    if count <= 10:  # Show first 10
                        print(f"  {op_name}: {op_data['op_type']}")
                    elif count == 11:
                        print(f"  ... and {len([op for op, data in extractor.operation_metadata.items() if target_module_hierarchy in data.get('tags', [])]) - 10} more")
            
            print(f"Total operations tagged: {count}")
        
        # Look for operations with 'SdpaSelfAttention' in the tag
        print(f"\nOperations with 'SdpaSelfAttention' in tag:")
        sdpa_ops = []
        for op_name, op_data in extractor.operation_metadata.items():
            for tag in op_data.get('tags', []):
                if 'SdpaSelfAttention' in tag:
                    sdpa_ops.append((op_name, op_data['op_type'], tag))
                    break
        
        print(f"Found {len(sdpa_ops)} operations with SdpaSelfAttention tags")
        for _i, (op_name, op_type, tag) in enumerate(sdpa_ops[:10]):
            print(f"  {op_name}: {op_type} → {tag}")
        
        if len(sdpa_ops) > 10:
            print(f"  ... and {len(sdpa_ops) - 10} more")
        
        # Show unique SdpaSelfAttention tags
        sdpa_tags = set()
        for _, _, tag in sdpa_ops:
            sdpa_tags.add(tag)
        
        print(f"\nUnique SdpaSelfAttention tags:")
        for tag in sorted(sdpa_tags):
            op_count = sum(1 for _, _, t in sdpa_ops if t == tag)
            print(f"  {tag} ({op_count} operations)")
    
    finally:
        # Clean up temp file
        if os.path.exists(temp_path):
            os.unlink(temp_path)
        enhanced_path = temp_path.replace('.onnx', '_with_tags.onnx')
        if os.path.exists(enhanced_path):
            os.unlink(enhanced_path)
    
    print("\n" + "="*60)
    print("📍 CONCLUSION:")
    print("The standalone module should contain ALL operations from BertSdpaSelfAttention")
    print("The extracted subgraph should only contain operations tagged with the specific hierarchy path")
    print("If they differ significantly, we might be extracting a sub-component rather than the full module")


if __name__ == "__main__":
    debug_module_targeting()