#!/usr/bin/env python3
"""
Experimental implementation of Enhanced Trace Module Map approach.
This demonstrates what we discovered in the notebook about PyTorch's
internal _trace_module_map during ONNX export.
"""

import json
import tempfile
from collections import defaultdict
from pathlib import Path

import torch
from transformers import AutoModel, AutoTokenizer


def establish_expected_results():
    """First, let's establish what the correct hierarchy should look like for bert-tiny."""
    
    print("=" * 80)
    print("📋 ESTABLISHING EXPECTED RESULTS FOR BERT-TINY")
    print("=" * 80)
    
    # Load model
    model_name = "prajjwal1/bert-tiny"
    model = AutoModel.from_pretrained(model_name)
    
    # Analyze the model structure
    print("\n🔍 Model Structure Analysis:")
    
    # Count modules by type
    module_types = defaultdict(int)
    hierarchy_examples = []
    
    for name, module in model.named_modules():
        module_type = type(module).__name__
        module_types[module_type] += 1
        
        # Collect examples of different hierarchy levels
        if len(hierarchy_examples) < 20 or "layer.0" in name or "layer.1" in name:
            hierarchy_examples.append({
                'name': name,
                'class': module_type,
                'level': name.count('.') if name else 0,
                'path': f"__module.{name}" if name else "__module"
            })
    
    # Print module type distribution
    print("\n📊 Module Type Distribution:")
    for module_type, count in sorted(module_types.items()):
        print(f"   {module_type:30s}: {count:2d} modules")
    
    # Print hierarchy examples
    print("\n🏗️ Expected Hierarchy Structure:")
    print("   (What we should see in _trace_module_map)")
    print()
    
    # Sort by hierarchy level for better visualization
    hierarchy_examples.sort(key=lambda x: (x['level'], x['name']))
    
    for example in hierarchy_examples[:25]:  # Show first 25
        indent = "  " * example['level']
        expected_scope = f"{example['class']}::{example['path']}"
        print(f"   {indent}{example['class']:20s} → {expected_scope}")
    
    print(f"\n✅ Expected Patterns in Enhanced Trace Module Map:")
    print("   1. Root: BertModel::__module")
    print("   2. Embeddings: BertEmbeddings::__module.embeddings")
    print("   3. Encoder layers: BertLayer::__module.encoder.layer.0")
    print("   4. Attention: BertAttention::__module.encoder.layer.0.attention")
    print("   5. Self-attention: BertSdpaSelfAttention::__module.encoder.layer.0.attention.self")
    
    return model, hierarchy_examples

def implement_enhanced_trace_map(model, tokenizer):
    """Implement the Enhanced Trace Module Map approach."""
    
    print("\n" + "=" * 80)
    print("🚀 IMPLEMENTING ENHANCED TRACE MODULE MAP")
    print("=" * 80)
    
    # Prepare inputs
    text = "Hello world"
    inputs = tokenizer(text, return_tensors="pt")
    input_ids = inputs["input_ids"]
    attention_mask = inputs["attention_mask"]
    
    # Store captured data
    captured_trace_maps = []
    
    # Store original function
    original_setup = getattr(torch.onnx.utils, '_setup_trace_module_map', None)
    
    def capture_enhanced_trace_map(*args, **kwargs):
        """Hook to capture the enhanced trace module map."""
        print("\n🎣 Hook called! Capturing trace module map...")
        
        # Call original
        result = None
        if original_setup:
            result = original_setup(*args, **kwargs)
        
        # Capture the map
        trace_map = getattr(torch.jit._trace, '_trace_module_map', None)
        
        if trace_map:
            print(f"✅ Captured trace module map with {len(trace_map)} entries!")
            
            # Convert to serializable format
            serializable_map = {}
            for module, scope_name in trace_map.items():
                module_key = f"{type(module).__name__}_{id(module)}"
                serializable_map[module_key] = {
                    'module_class': type(module).__name__,
                    'scope_name': scope_name,
                    'module_str': str(module)[:100] + '...' if len(str(module)) > 100 else str(module)
                }
            
            captured_trace_maps.append({
                'hook_point': 'setup_trace_module_map',
                'map_size': len(trace_map),
                'entries': serializable_map
            })
        else:
            print("⚠️ No trace module map found!")
        
        return result
    
    # Apply hook
    if original_setup:
        torch.onnx.utils._setup_trace_module_map = capture_enhanced_trace_map
        print("✅ Hook installed on _setup_trace_module_map")
    else:
        print("❌ Could not find _setup_trace_module_map!")
        return None
    
    try:
        # Perform ONNX export
        print("\n📦 Performing ONNX export to trigger trace module map creation...")
        
        with tempfile.NamedTemporaryFile(suffix='.onnx', delete=False) as tmp:
            onnx_path = tmp.name
            
            torch.onnx.export(
                model,
                (input_ids, attention_mask),
                onnx_path,
                input_names=['input_ids', 'attention_mask'],
                output_names=['last_hidden_state'],
                dynamic_axes={
                    'input_ids': {0: 'batch_size', 1: 'sequence'},
                    'attention_mask': {0: 'batch_size', 1: 'sequence'},
                    'last_hidden_state': {0: 'batch_size', 1: 'sequence'}
                },
                do_constant_folding=True,
                opset_version=17,
                verbose=False
            )
            
            print(f"✅ ONNX export completed: {Path(onnx_path).name}")
    
    finally:
        # Restore original
        if original_setup:
            torch.onnx.utils._setup_trace_module_map = original_setup
            print("✅ Original function restored")
    
    # Clean up ONNX file
    Path(onnx_path).unlink()
    
    return captured_trace_maps

def analyze_and_verify_results(captured_maps, expected_hierarchy):
    """Analyze the captured trace maps and verify against expected results."""
    
    print("\n" + "=" * 80)
    print("📊 ANALYZING CAPTURED RESULTS")
    print("=" * 80)
    
    if not captured_maps:
        print("❌ No trace maps were captured!")
        return
    
    for i, capture in enumerate(captured_maps):
        print(f"\n📋 Capture {i+1}: {capture['hook_point']}")
        print(f"   Map size: {capture['map_size']} entries")
        
        # Analyze the captured entries
        entries = capture['entries']
        
        # Group by module class
        by_class = defaultdict(list)
        for _key, data in entries.items():
            by_class[data['module_class']].append(data)
        
        print(f"\n   Module Class Distribution:")
        for module_class, items in sorted(by_class.items()):
            print(f"      {module_class:30s}: {len(items):2d} instances")
        
        # Show sample entries with enhanced scope names
        print(f"\n   Sample Enhanced Scope Names:")
        
        # Sort entries by scope name for better visualization
        sorted_entries = sorted(entries.items(), key=lambda x: x[1]['scope_name'])
        
        # Show first 30 entries
        for j, (_key, data) in enumerate(sorted_entries[:30]):
            scope_name = data['scope_name']
            module_class = data['module_class']
            
            # Determine quality of scope name
            if '::' in scope_name and '.' in scope_name:
                quality = "🟢 ENHANCED"  # Full hierarchy with class name
            elif '::' in scope_name:
                quality = "🟡 PARTIAL"   # Has class but no path
            else:
                quality = "🔴 BASIC"     # Just path, no class
            
            print(f"      {j+1:2d}. {quality} {module_class:20s} → {scope_name}")
        
        # Verify expected patterns
        print(f"\n   ✅ Verification Against Expected Patterns:")
        
        expected_patterns = [
            ("BertModel", "__module"),
            ("BertEmbeddings", "__module.embeddings"),
            ("BertEncoder", "__module.encoder"),
            ("BertLayer", "__module.encoder.layer.0"),
            ("BertAttention", "__module.encoder.layer.0.attention"),
            ("BertSdpaSelfAttention", "__module.encoder.layer.0.attention.self"),
            ("Linear", "__module.encoder.layer.0.attention.self.query"),
            ("BertSelfOutput", "__module.encoder.layer.0.attention.output"),
            ("LayerNorm", "__module.encoder.layer.0.attention.output.LayerNorm"),
            ("BertIntermediate", "__module.encoder.layer.0.intermediate"),
            ("BertOutput", "__module.encoder.layer.0.output"),
            ("BertPooler", "__module.pooler")
        ]
        
        for expected_class, expected_path in expected_patterns:
            # Look for this pattern in captured data
            found = False
            for _key, data in entries.items():
                if data['module_class'] == expected_class:
                    scope = data['scope_name']
                    # Check if scope contains expected path
                    if expected_path in scope:
                        found = True
                        if '::' in scope:
                            print(f"      ✅ Found {expected_class} with enhanced scope: {scope}")
                        else:
                            print(f"      ⚠️  Found {expected_class} but without enhancement: {scope}")
                        break
            
            if not found:
                print(f"      ❌ Missing expected pattern: {expected_class} at {expected_path}")
        
        # Check for enhanced scope names
        enhanced_count = sum(1 for d in entries.values() if '::' in d['scope_name'])
        basic_count = len(entries) - enhanced_count
        
        print(f"\n   📊 Scope Name Quality Summary:")
        print(f"      🟢 Enhanced (with ::): {enhanced_count} ({enhanced_count/len(entries)*100:.1f}%)")
        print(f"      🔴 Basic (no ::):      {basic_count} ({basic_count/len(entries)*100:.1f}%)")

def save_results(captured_maps):
    """Save the captured results for further analysis."""
    
    output_path = Path("enhanced_trace_map_results.json")
    
    with open(output_path, 'w') as f:
        json.dump({
            'experiment': 'Enhanced Trace Module Map',
            'model': 'prajjwal1/bert-tiny',
            'captures': captured_maps,
            'summary': {
                'total_captures': len(captured_maps),
                'total_modules': captured_maps[0]['map_size'] if captured_maps else 0
            }
        }, f, indent=2)
    
    print(f"\n💾 Results saved to: {output_path}")
    return output_path

def main():
    """Run the experimental implementation."""
    
    print("🔬 EXPERIMENTAL: Enhanced Trace Module Map Implementation")
    print("=" * 80)
    
    # Load model and tokenizer
    model_name = "prajjwal1/bert-tiny"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # Step 1: Establish expected results
    model, expected_hierarchy = establish_expected_results()
    
    # Step 2: Implement enhanced trace map capture
    captured_maps = implement_enhanced_trace_map(model, tokenizer)
    
    # Step 3: Analyze and verify results
    if captured_maps:
        analyze_and_verify_results(captured_maps, expected_hierarchy)
        
        # Step 4: Save results
        output_path = save_results(captured_maps)
        
        print(f"\n" + "=" * 80)
        print("🎉 EXPERIMENT COMPLETE!")
        print("=" * 80)
        print(f"\n📊 Summary:")
        print(f"   - Model: {model_name}")
        print(f"   - Captures: {len(captured_maps)}")
        print(f"   - Total modules tracked: {captured_maps[0]['map_size'] if captured_maps else 0}")
        print(f"   - Results saved to: {output_path}")
        
        # Key findings
        if captured_maps and captured_maps[0]['entries']:
            entries = captured_maps[0]['entries']
            enhanced_count = sum(1 for d in entries.values() if '::' in d['scope_name'])
            print(f"\n🔍 Key Finding:")
            print(f"   PyTorch creates enhanced scope names for {enhanced_count}/{len(entries)} modules!")
            print(f"   This validates that _trace_module_map contains rich hierarchy information.")
    else:
        print("\n❌ Experiment failed - no trace maps captured")

if __name__ == "__main__":
    main()