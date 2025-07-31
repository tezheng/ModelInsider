#!/usr/bin/env python3
"""
Test script for Universal Hierarchy Exporter with BERT-tiny
"""

import sys

sys.path.append('/mnt/d/BYOM/modelexport')

import json
import time
from pathlib import Path

from transformers import AutoModel, AutoTokenizer

from modelexport.core.universal_hierarchy_exporter import UniversalHierarchyExporter


def test_bert_tiny_export():
    """Test universal hierarchy export with BERT-tiny"""
    
    print("🎯 Testing Universal Hierarchy Exporter with BERT-tiny")
    print("=" * 60)
    
    # Load model and tokenizer
    print("\n1. Loading BERT-tiny model...")
    model_name = "prajjwal1/bert-tiny"
    model = AutoModel.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model.eval()  # Important: set to eval mode
    
    print(f"   Model: {model.__class__.__name__}")
    print(f"   Parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Prepare sample input
    print("\n2. Preparing sample input...")
    text = "Hello, world! This is a test."
    inputs = tokenizer(text, return_tensors="pt", max_length=128, padding="max_length", truncation=True)
    input_ids = inputs["input_ids"]
    attention_mask = inputs["attention_mask"]
    print(f"   Input shape: {input_ids.shape}")
    
    # Create exporter
    print("\n3. Creating Universal Hierarchy Exporter...")
    exporter = UniversalHierarchyExporter(
        torch_nn_exceptions=['LayerNorm', 'Embedding'],
        verbose=True  # Enable to see hook registration
    )
    
    # Create output directory
    output_dir = Path("temp/universal_export_test")
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = str(output_dir / "bert_tiny_universal.onnx")
    
    # Export
    print(f"\n4. Exporting to ONNX (output: {output_path})...")
    start_time = time.time()
    
    try:
        export_result = exporter.export(
            model=model,
            args=(input_ids, attention_mask),
            output_path=output_path,
            input_names=['input_ids', 'attention_mask'],
            output_names=['last_hidden_state'],
            dynamic_axes={
                'input_ids': {0: 'batch_size', 1: 'sequence'},
                'attention_mask': {0: 'batch_size', 1: 'sequence'},
                'last_hidden_state': {0: 'batch_size', 1: 'sequence'}
            },
            opset_version=17,
            do_constant_folding=True
        )
        
        export_time = time.time() - start_time
        print(f"\n✅ Export completed in {export_time:.2f}s")
        
        # Display results
        print(f"\n5. Export Results:")
        print(f"   Total modules: {export_result['total_modules']}")
        print(f"   Tagged operations: {export_result['tagged_operations']}")
        print(f"   Filtered modules: {export_result['filtered_modules']}")
        
        # Check hierarchy metadata
        metadata_path = output_path.replace('.onnx', '_hierarchy_metadata.json')
        if Path(metadata_path).exists():
            with open(metadata_path) as f:
                metadata = json.load(f)
            
            # The metadata structure has changed - look for module_hierarchy
            hierarchy_info = metadata.get('hierarchy_info', {})
            if not hierarchy_info:
                # Try the actual structure returned by get_hierarchy_metadata
                module_hierarchy = exporter.get_hierarchy_metadata().get('module_hierarchy', {})
            else:
                module_hierarchy = hierarchy_info
                
            modules_with_tags = sum(1 for m in module_hierarchy.values() if m.get('expected_tag'))
            
            print(f"\n6. Hierarchy Metadata:")
            print(f"   Total modules in hierarchy: {len(module_hierarchy)}")
            print(f"   Modules with tags: {modules_with_tags}")
            
            # Show sample tags
            print(f"\n   Sample hierarchy tags:")
            count = 0
            for path, info in module_hierarchy.items():
                if info.get('expected_tag') and count < 5:
                    print(f"     {path} -> {info['expected_tag']}")
                    count += 1
        
        # Validate the exported ONNX model
        print(f"\n7. Validating ONNX model...")
        try:
            import onnx
            onnx_model = onnx.load(output_path)
            onnx.checker.check_model(onnx_model)
            print("   ✅ ONNX model is valid")
            
            # Check file size
            file_size = Path(output_path).stat().st_size / (1024 * 1024)
            print(f"   File size: {file_size:.2f} MB")
            
        except Exception as e:
            print(f"   ❌ ONNX validation failed: {e}")
        
        print(f"\n✨ Success! Model exported to: {output_path}")
        
    except Exception as e:
        print(f"\n❌ Export failed: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(test_bert_tiny_export())