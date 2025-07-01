#!/usr/bin/env python3
"""
Verify what we're chasing based on the requirements document.
"""

import torch
from transformers import AutoModel, AutoTokenizer
from modelexport.strategies.htp.htp_hierarchy_exporter import HierarchyExporter
import onnx
import json
from pathlib import Path

def analyze_requirements():
    """Analyze what the requirements are asking for."""
    
    print("=" * 80)
    print("📋 REQUIREMENTS ANALYSIS - What Are We Chasing?")
    print("=" * 80)
    
    print("\n🎯 CORE MISSION (from REQUIREMENTS.md):")
    print("Universal Hierarchy-Preserving ONNX Export that:")
    print("1. ✅ Preserves complete module hierarchy through intelligent tagging")
    print("2. ✅ Maintains IDENTICAL graph topology to baseline export")
    print("3. ✅ Enables subgraph extraction for any module")
    print("4. ✅ Supports operation-to-module attribution")
    
    print("\n🚨 CARDINAL RULES:")
    print("1. MUST-001: NO HARDCODED LOGIC - ✅ Using universal PyTorch principles")
    print("2. MUST-002: TORCH.NN FILTERING - ✅ Filter most torch.nn except whitelist")
    print("3. MUST-003: UNIVERSAL DESIGN - ✅ Works with ANY PyTorch model")
    
    print("\n🏗️ KEY REQUIREMENTS:")
    print("R10: Operation-to-Module Attribution")
    print("  - Map EVERY ONNX operation to source HF module class")
    print("  - This is what we're verifying with the trace module map!")
    
    print("R12: Instance-Specific Hierarchy Paths")
    print("  - Preserve instance numbers: BertLayer.0 vs BertLayer.1")
    print("  - Example: /BertModel/BertEncoder/BertLayer.0/BertAttention")
    
    print("\n📊 CURRENT STATUS (from requirements):")
    print("✅ Topology Preservation: 100% identical to baseline")
    print("✅ Multi-Consumer Tagging: 100% tensor coverage")
    print("✅ Contamination Reduction: 72% reduction achieved")
    print("✅ Performance: 29% improvement with built-in tracking")
    print("✅ Production Ready: 121 tests passing")

def test_current_implementation():
    """Test what the current HTP implementation actually produces."""
    
    print("\n" + "=" * 80)
    print("🔬 TESTING CURRENT HTP IMPLEMENTATION")
    print("=" * 80)
    
    # Load model
    model_name = "prajjwal1/bert-tiny"
    print(f"\n📦 Loading {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    
    # Prepare inputs
    text = "Hello world"
    inputs = tokenizer(text, return_tensors="pt", max_length=128, padding="max_length", truncation=True)
    input_ids = inputs["input_ids"]
    attention_mask = inputs["attention_mask"]
    
    # Create exporter
    print("\n🔧 Creating HTP exporter...")
    exporter = HierarchyExporter(
        strategy="htp",
        torch_nn_exceptions=['LayerNorm', 'Embedding']  # Allowed torch.nn modules
    )
    
    # Export to ONNX
    output_path = Path("temp/bert_tiny_htp_test.onnx")
    output_path.parent.mkdir(exist_ok=True)
    
    print(f"\n🚀 Exporting with HTP strategy...")
    try:
        result = exporter.export(
            model, 
            (input_ids, attention_mask),
            str(output_path),
            input_names=['input_ids', 'attention_mask'],
            output_names=['last_hidden_state'],
            dynamic_axes={
                'input_ids': {0: 'batch_size', 1: 'sequence'},
                'attention_mask': {0: 'batch_size', 1: 'sequence'}
            },
            opset_version=17
        )
        
        print(f"✅ Export successful!")
        print(f"   - Output: {output_path}")
        print(f"   - Export time: {result.get('export_time', 'N/A'):.2f}s")
        print(f"   - Tagged operations: {result.get('tagged_operations', 'N/A')}")
        print(f"   - Empty tags: {result.get('empty_tags', 'N/A')}")
        
    except Exception as e:
        print(f"❌ Export failed: {e}")
        return
    
    # Analyze the ONNX model
    print(f"\n📊 Analyzing ONNX model tags...")
    onnx_model = onnx.load(str(output_path))
    
    # Count nodes with tags
    tagged_nodes = 0
    empty_tags = 0
    tag_examples = []
    
    for node in onnx_model.graph.node:
        # Check for hierarchy tag
        for attr in node.attribute:
            if attr.name == "hf_hierarchy_tag":
                if attr.s:  # Has a tag
                    tag = attr.s.decode('utf-8')
                    tagged_nodes += 1
                    if len(tag_examples) < 10:
                        tag_examples.append({
                            'op_type': node.op_type,
                            'name': node.name,
                            'tag': tag
                        })
                else:
                    empty_tags += 1
                break
    
    total_nodes = len(onnx_model.graph.node)
    print(f"\n📈 Tagging Results:")
    print(f"   - Total ONNX nodes: {total_nodes}")
    print(f"   - Tagged nodes: {tagged_nodes} ({tagged_nodes/total_nodes*100:.1f}%)")
    print(f"   - Empty tags: {empty_tags} ({empty_tags/total_nodes*100:.1f}%)")
    
    print(f"\n🏷️ Sample Tags (what we're actually producing):")
    for i, example in enumerate(tag_examples, 1):
        print(f"   {i}. {example['op_type']:15s} → {example['tag']}")
    
    # Check if tags match requirements
    print(f"\n✅ VERIFICATION Against Requirements:")
    
    # R10: Operation-to-Module Attribution
    print(f"   R10 (Operation Attribution): {'✅ PASS' if tagged_nodes > 0 else '❌ FAIL'}")
    
    # R12: Instance-Specific Paths
    has_instance_numbers = any('.0' in ex['tag'] or '.1' in ex['tag'] for ex in tag_examples)
    print(f"   R12 (Instance Numbers): {'✅ PASS' if has_instance_numbers else '❌ FAIL'}")
    
    # MUST-002: torch.nn filtering
    has_torch_nn_filtered = not any('torch.nn' in ex['tag'] for ex in tag_examples 
                                   if 'LayerNorm' not in ex['tag'] and 'Embedding' not in ex['tag'])
    print(f"   MUST-002 (torch.nn filtering): {'✅ PASS' if has_torch_nn_filtered else '❌ FAIL'}")

def explain_what_we_need():
    """Explain what we're trying to achieve."""
    
    print("\n" + "=" * 80)
    print("💡 WHAT WE'RE CHASING - THE GOAL")
    print("=" * 80)
    
    print("\n🎯 The Goal is to tag ONNX operations with their source module paths:")
    
    print("\n📋 EXPECTED TAG FORMAT (from R12):")
    print("   /BertModel/BertEncoder/BertLayer.0/BertAttention")
    print("   ↑          ↑           ↑           ↑")
    print("   Root       Container   Instance    Leaf module")
    
    print("\n🏷️ WHAT EACH ONNX NODE SHOULD HAVE:")
    print("   <Node op_type='MatMul'>")
    print("     <attribute name='hf_hierarchy_tag'>")
    print("       /BertModel/BertEncoder/BertLayer.0/BertAttention/BertSelfAttention")
    print("     </attribute>")
    print("   </Node>")
    
    print("\n🔍 WHY WE INVESTIGATED TRACE MODULE MAP:")
    print("   - PyTorch's _trace_module_map contains: transformers.models.bert.BertLayer::layer.0")
    print("   - We need to convert this to: /BertModel/BertEncoder/BertLayer.0")
    print("   - The instance name 'layer.0' tells us it's the first layer")
    print("   - But we need the FULL hierarchy path, not just the instance name")
    
    print("\n✅ WHAT HTP v2 ALREADY DOES:")
    print("   1. Captures _trace_module_map during export")
    print("   2. Uses forward hooks to track which module executes each operation")
    print("   3. Tags ONNX nodes with the module hierarchy")
    print("   4. Filters torch.nn modules (except whitelist)")
    
    print("\n❓ THE QUESTION:")
    print("   Are the current tags meeting the requirements?")
    print("   - Do they have full hierarchy paths?")
    print("   - Do they preserve instance numbers?")
    print("   - Do they enable subgraph extraction?")

if __name__ == "__main__":
    # First, understand the requirements
    analyze_requirements()
    
    # Test current implementation
    test_current_implementation()
    
    # Explain what we need
    explain_what_we_need()
    
    print("\n" + "=" * 80)
    print("🎯 CONCLUSION")
    print("=" * 80)
    print("\nThe Enhanced Trace Module Map investigation revealed HOW PyTorch tracks modules.")
    print("HTP v2 already uses this mechanism effectively.")
    print("The question is: are the current tags in the format the requirements expect?")
    print("\nNext step: Analyze actual ONNX output to verify tag format matches requirements.")