#!/usr/bin/env python3
"""
Debug script specifically for hierarchical path building.
"""

from transformers import AutoModel

from modelexport.hierarchy_exporter import HierarchyExporter


def debug_path_building():
    print("🔍 Debug: Path Building Logic")
    print("=" * 50)
    
    model = AutoModel.from_pretrained('prajjwal1/bert-tiny')
    exporter = HierarchyExporter()
    exporter._model = model
    
    # Test path building for specific modules
    test_modules = [
        'embeddings',
        'embeddings.word_embeddings',
        'encoder.layer.0.attention.self.query',
        'encoder.layer.0.attention.output.dense',
        'pooler.dense'
    ]
    
    print(f"✅ Root model class: {model.__class__.__name__}")
    print(f"✅ Root model module: {model.__class__.__module__}")
    
    for module_name in test_modules:
        print(f"\n🔍 Testing: {module_name}")
        
        # Manually trace the path building
        current_module = model
        name_parts = module_name.split('.')
        path_segments = [model.__class__.__name__]
        
        print(f"  Starting with: {path_segments}")
        
        for i, part in enumerate(name_parts):
            if hasattr(current_module, part):
                current_module = getattr(current_module, part)
                class_module = current_module.__class__.__module__
                class_name = current_module.__class__.__name__
                
                should_include = not class_module.startswith('torch._C') and not class_module.startswith('torch.nn')
                
                print(f"    Part {i+1}: {part}")
                print(f"      Class: {class_name}")
                print(f"      Module: {class_module}")
                print(f"      Include: {should_include}")
                
                if should_include:
                    path_segments.append(class_name)
            else:
                print(f"    Part {i+1}: {part} - NOT FOUND")
                break
        
        # Build final path
        final_path = "/" + "/".join(path_segments)
        print(f"  Final path: {final_path}")
        
        # Compare with exporter method
        exporter_path = exporter._build_hierarchical_tag(module_name, current_module)
        print(f"  Exporter path: {exporter_path}")
        print(f"  Match: {final_path == exporter_path}")

if __name__ == "__main__":
    debug_path_building()