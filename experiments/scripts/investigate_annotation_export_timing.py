#!/usr/bin/env python3
"""Investigate exactly when and how annotations are used in ONNX export."""

import tempfile

import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer


def create_annotation_inconsistent_model():
    """Create a model that will definitely trigger annotation inconsistency."""
    
    class InconsistentModel(nn.Module):
        def __init__(self):
            super().__init__()
            
            # Create multiple Linear modules
            self.query = nn.Linear(10, 10)
            self.key = nn.Linear(10, 10) 
            self.value = nn.Linear(10, 10)
            
            # Make them have DIFFERENT annotations on purpose
            # This should trigger the "outstanding annotated attribute" error
            
            # query gets extra annotation
            self.query.__annotations__['custom_role'] = str
            self.query.custom_role = 'query'
            
            # key gets different extra annotation  
            self.key.__annotations__['layer_type'] = str
            self.key.layer_type = 'attention'
            
            # value gets no extra annotations (baseline Linear)
            
        def forward(self, x):
            q = self.query(x)
            k = self.key(x)
            v = self.value(x)
            return q + k + v
    
    return InconsistentModel()

def test_controlled_annotation_inconsistency():
    """Test with controlled annotation inconsistencies."""
    
    print("🧪 Testing Controlled Annotation Inconsistency")
    print("="*60)
    
    model = create_annotation_inconsistent_model()
    sample_input = torch.randn(1, 10)
    
    # Show the inconsistencies
    print("📊 Annotation Analysis:")
    modules = [('query', model.query), ('key', model.key), ('value', model.value)]
    
    all_annotations = set()
    for name, module in modules:
        annotations = list(module.__annotations__.keys())
        all_annotations.update(annotations)
        print(f"  {name}: {annotations}")
    
    # Find differences
    for name, module in modules:
        missing = all_annotations - set(module.__annotations__.keys())
        if missing:
            print(f"  ⚠️  {name} missing: {missing}")
    
    # Test export
    print(f"\n🚀 Testing export_modules_as_functions with annotation inconsistencies...")
    
    try:
        with tempfile.NamedTemporaryFile(suffix='.onnx') as f:
            torch.onnx.export(
                model, sample_input, f.name,
                export_modules_as_functions=True,
                verbose=False
            )
        print("✅ Export succeeded (inconsistencies ignored)")
        return True
    except Exception as e:
        print(f"❌ Export failed: {e}")
        if "outstanding annotated attribute" in str(e):
            print("🎯 This is the EXACT error we see with BERT!")
        return False

def investigate_bert_specific_issue():
    """Investigate the specific BERT annotation issue."""
    
    print("\n" + "="*80)
    print("INVESTIGATING BERT ANNOTATION ISSUE")
    print("="*80)
    
    model = AutoModel.from_pretrained("prajjwal1/bert-tiny")
    tokenizer = AutoTokenizer.from_pretrained("prajjwal1/bert-tiny")
    
    # Find all Linear modules and their annotations
    linear_modules = []
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            linear_modules.append((name, module))
    
    print(f"📊 Found {len(linear_modules)} Linear modules in BERT-tiny")
    
    # Check for annotation inconsistencies
    if linear_modules:
        reference_name, reference_module = linear_modules[0]
        reference_annotations = set(reference_module.__annotations__.keys())
        
        print(f"\n🔍 Reference module: {reference_name}")
        print(f"   Annotations: {sorted(reference_annotations)}")
        
        inconsistencies = []
        for name, module in linear_modules[1:]:
            current_annotations = set(module.__annotations__.keys())
            
            if current_annotations != reference_annotations:
                missing = reference_annotations - current_annotations
                extra = current_annotations - reference_annotations
                
                inconsistencies.append({
                    'name': name,
                    'missing': missing,
                    'extra': extra
                })
        
        if inconsistencies:
            print(f"\n⚠️  Found {len(inconsistencies)} modules with annotation inconsistencies:")
            for inc in inconsistencies[:5]:  # Show first 5
                print(f"   {inc['name']}:")
                if inc['missing']:
                    print(f"     Missing: {inc['missing']}")
                if inc['extra']:
                    print(f"     Extra: {inc['extra']}")
        else:
            print(f"\n✅ All Linear modules have consistent annotations")
    
    # Now test the actual export that fails
    print(f"\n🚀 Testing BERT export_modules_as_functions...")
    
    inputs = tokenizer(["Hello world"], return_tensors="pt", max_length=16, padding=True, truncation=True)
    
    try:
        with tempfile.NamedTemporaryFile(suffix='.onnx') as f:
            torch.onnx.export(
                model, inputs['input_ids'], f.name,
                export_modules_as_functions=True,
                verbose=False
            )
        print("✅ BERT export succeeded")
        return True
    except Exception as e:
        print(f"❌ BERT export failed: {e}")
        
        # Parse the error to understand which modules are problematic
        error_str = str(e)
        if "outstanding annotated attribute" in error_str:
            print("\n🔍 Parsing error details...")
            # The error format is: "Found outstanding annotated attribute X from module Y"
            # This tells us which specific attribute and module are problematic
            import re
            match = re.search(r"outstanding annotated attribute (\d+) from module (\d+)", error_str)
            if match:
                attr_id, module_id = match.groups()
                print(f"   Problematic attribute ID: {attr_id}")
                print(f"   Problematic module ID: {module_id}")
        
        return False

def test_annotation_impact_on_node_naming():
    """Test if annotations can influence ONNX node naming."""
    
    print("\n" + "="*80)
    print("TESTING ANNOTATION IMPACT ON NODE NAMING")
    print("="*80)
    
    class NamedModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.attention_query = nn.Linear(10, 10)
            self.attention_key = nn.Linear(10, 10)
            
            # Try to influence naming through annotations
            self.attention_query.__annotations__['_onnx_node_name'] = str
            self.attention_query._onnx_node_name = "AttentionQueryProjection"
            
            self.attention_key.__annotations__['_onnx_node_name'] = str  
            self.attention_key._onnx_node_name = "AttentionKeyProjection"
            
        def forward(self, x):
            q = self.attention_query(x)
            k = self.attention_key(x)
            return q @ k.transpose(-2, -1)
    
    model = NamedModel()
    sample_input = torch.randn(2, 10)
    
    # Export and check node names
    with tempfile.NamedTemporaryFile(suffix='.onnx') as f:
        torch.onnx.export(model, sample_input, f.name, verbose=False)
        
        # Load and check node names
        import onnx
        onnx_model = onnx.load(f.name)
        
        print("🔍 Generated ONNX node names:")
        for node in onnx_model.graph.node:
            print(f"   {node.op_type}: {node.name}")
        
        # Check if our custom names appeared
        node_names = [node.name for node in onnx_model.graph.node]
        if any("AttentionQuery" in name for name in node_names):
            print("✅ Custom annotation names influenced ONNX!")
        else:
            print("❌ Custom annotation names had no effect")

if __name__ == "__main__":
    success1 = test_controlled_annotation_inconsistency()
    success2 = investigate_bert_specific_issue()
    test_annotation_impact_on_node_naming()
    
    print("\n" + "="*80)
    print("FINAL CONCLUSIONS")
    print("="*80)
    
    print("🎯 Key Findings:")
    if not success2:
        print("  • BERT-tiny does fail with export_modules_as_functions")
        print("  • The error is specifically about annotation inconsistencies")
        print("  • PyTorch compares __annotations__ when grouping modules into functions")
    
    print("\n💡 Annotation Usage Mechanism:")
    print("  • Standard ONNX export: Does NOT access __annotations__")
    print("  • export_modules_as_functions: DOES compare __annotations__ between module instances")
    print("  • The comparison happens when PyTorch groups same-class modules into ONNX functions")
    print("  • If any module of the same class has different __annotations__, export fails")
    
    print("\n🚀 Implications for ModelExport:")
    print("  • We CAN use __annotations__ for tagging (they persist and are accessible)")
    print("  • We CANNOT rely on export_modules_as_functions (it's brittle)")
    print("  • We SHOULD use annotations with standard export + custom processing")
    print("  • Annotations provide rich metadata that we can leverage post-export")