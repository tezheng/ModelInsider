#!/usr/bin/env python3
"""
Debug script to check what happens during hook execution.
"""

import torch
from transformers import AutoModel, AutoTokenizer

from modelexport.hierarchy_exporter import HierarchyExporter


def debug_hook_execution():
    print("🔍 Debug: Hook Execution")
    print("=" * 50)
    
    model = AutoModel.from_pretrained('prajjwal1/bert-tiny')
    tokenizer = AutoTokenizer.from_pretrained('prajjwal1/bert-tiny')
    model.eval()
    
    inputs = tokenizer("test", return_tensors='pt', max_length=8, padding='max_length', truncation=True)
    
    exporter = HierarchyExporter()
    exporter._reset_state()
    exporter._model = model  # Set model reference
    
    print(f"✅ Model set: {exporter._model.__class__.__name__}")
    
    # Override the hook to debug what's happening
    original_build_tag = exporter._build_hierarchical_tag
    
    def debug_build_tag(module_name, module):
        print(f"🪝 Hook fired for: {module_name}")
        print(f"   Module class: {module.__class__.__name__}")
        print(f"   Model set: {exporter._model is not None}")
        
        if exporter._model:
            print(f"   Model class: {exporter._model.__class__.__name__}")
        
        tag = original_build_tag(module_name, module)
        print(f"   Generated tag: {tag}")
        print()
        return tag
    
    exporter._build_hierarchical_tag = debug_build_tag
    
    # Register hooks
    exporter._register_hooks(model)
    print(f"✅ Registered {len(exporter._hooks)} hooks")
    
    # Trace execution
    print("\n🔄 Starting trace execution:")
    with torch.no_grad():
        exporter._trace_model_execution(model, inputs)
    
    print(f"\n✅ Captured {len(exporter._operation_context)} operation contexts")
    
    # Show what's actually stored
    print("\nFirst 5 stored contexts:")
    for i, (name, context) in enumerate(exporter._operation_context.items()):
        if i >= 5:
            break
        print(f"  {name}: {context['tag']}")
    
    # Clean up
    exporter._remove_hooks()

if __name__ == "__main__":
    debug_hook_execution()