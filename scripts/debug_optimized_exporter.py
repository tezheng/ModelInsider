#!/usr/bin/env python3
"""
Debug the optimized enhanced semantic exporter.
"""

import torch
from transformers import AutoModel

from modelexport.core.enhanced_semantic_exporter import EnhancedSemanticExporter
from modelexport.core.tracing_hierarchy_builder_optimized import (
    OptimizedTracingHierarchyBuilder,
)


def debug_optimized_exporter():
    """Debug the optimized exporter."""
    print("🐛 Debugging Optimized Enhanced Semantic Exporter")
    print("=" * 70)
    
    # Load model
    model = AutoModel.from_pretrained("prajjwal1/bert-tiny")
    model.eval()
    
    # Create inputs
    input_ids = torch.randint(0, 1000, (1, 8))
    attention_mask = torch.ones((1, 8), dtype=torch.long)
    
    # Test the tracer directly first
    print("\n🔍 Testing OptimizedTracingHierarchyBuilder directly:")
    tracer = OptimizedTracingHierarchyBuilder()
    tracer.trace_model_execution(model, (input_ids, attention_mask))
    summary = tracer.get_execution_summary()
    
    print(f"  Total modules in hierarchy: {summary['total_modules']}")
    print(f"  Modules traced: {summary['total_modules_traced']}")
    print(f"  Module hierarchy keys: {list(summary['module_hierarchy'].keys())[:5]}...")
    
    # Test the enhanced semantic exporter
    print("\n🔍 Testing Enhanced Semantic Exporter:")
    exporter = EnhancedSemanticExporter(verbose=True)
    
    # Check state before export
    print(f"  Exporter module hierarchy before export: {len(exporter._module_hierarchy)}")
    
    try:
        result = exporter.export(
            model=model,
            args=(input_ids, attention_mask),
            output_path="temp/debug_test.onnx",
            input_names=['input_ids', 'attention_mask'],
            output_names=['last_hidden_state', 'pooler_output'],
            opset_version=17
        )
        
        print(f"  Export successful!")
        print(f"  Exporter module hierarchy after export: {len(exporter._module_hierarchy)}")
        
        # Check what's in the hierarchy
        if exporter._module_hierarchy:
            print(f"  Sample hierarchy entries:")
            for i, (path, metadata) in enumerate(exporter._module_hierarchy.items()):
                if i < 5:
                    print(f"    {path or 'ROOT'}: {metadata.get('class_name')} ({metadata.get('module_type')})")
        else:
            print(f"  ❌ Module hierarchy is empty!")
            
    except Exception as e:
        print(f"  ❌ Export failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    debug_optimized_exporter()