"""
Test with an actual BERT model to reproduce the slice tagging issue.
"""

import torch
import tempfile
import json
from modelexport.hierarchy_exporter import HierarchyExporter


def test_real_bert_slice_issue():
    """Test with prajjwal1/bert-tiny to reproduce the actual issue."""
    
    print("=== Testing Real BERT Model Slice Context Issue ===")
    
    try:
        from transformers import AutoModel, AutoTokenizer
    except ImportError:
        print("Transformers not available, skipping real BERT test")
        return
    
    # Load the actual BERT model that was mentioned in the issue
    model_name = "prajjwal1/bert-tiny"
    print(f"Loading {model_name}...")
    
    model = AutoModel.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model.eval()
    
    # Create sample input
    text = "Hello world this is a test"
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
    
    print(f"Model: {model.__class__.__name__}")
    print(f"Input keys: {list(inputs.keys())}")
    print(f"Input shapes: {[(k, v.shape) for k, v in inputs.items()]}")
    
    # Show model structure
    print(f"\nModel structure (first few levels):")
    for name, module in model.named_modules():
        if name and name.count('.') <= 3:  # Limit depth for readability
            print(f"  {name}: {module.__class__.__name__}")
        if name.count('.') > 3:
            break
    
    # Test with HTP strategy 
    exporter = HierarchyExporter(strategy="htp")
    
    with tempfile.NamedTemporaryFile(suffix='.onnx') as tmp:
        print(f"\n--- Exporting {model_name} ---")
        result = exporter.export(
            model=model,
            example_inputs=inputs,
            output_path=tmp.name
        )
        
        print(f"Export result: {result}")
        
        # Load and analyze ONNX model
        import onnx
        onnx_model = onnx.load(tmp.name)
        
        # Find all slice operations
        slice_nodes = [node for node in onnx_model.graph.node if node.op_type == 'Slice']
        print(f"\nONNX Slice nodes found: {len(slice_nodes)}")
        
        # Get tag mapping
        tag_mapping = exporter.get_tag_mapping()
        
        # Analyze slice operation tags
        print(f"\n=== SLICE OPERATION ANALYSIS ===")
        slice_tagging_results = []
        
        for node_name, node_info in tag_mapping.items():
            if node_info.get('op_type') == 'Slice':
                tags = node_info.get('tags', [])
                slice_tagging_results.append({
                    'node_name': node_name,
                    'tags': tags
                })
        
        # Group by tag patterns
        root_level_slices = []
        pooler_slices = []
        attention_slices = []
        other_slices = []
        
        for result in slice_tagging_results:
            tags = result['tags']
            node_name = result['node_name']
            
            print(f"Node: {node_name}")
            print(f"  Tags: {tags}")
            
            # Categorize based on tags
            is_root_only = any(tag.strip('/').split('/')[-1] in ['BertModel', 'DistilBertModel'] for tag in tags)
            is_pooler = any('pooler' in tag.lower() for tag in tags)
            is_attention = any('attention' in tag.lower() for tag in tags)
            
            if is_root_only and not is_pooler and not is_attention:
                root_level_slices.append(result)
                print(f"  ❌ ROOT-LEVEL: Tagged only with model root")
            elif is_pooler:
                pooler_slices.append(result)
                print(f"  ⚠️  POOLER: Tagged with pooler context")
            elif is_attention:
                attention_slices.append(result)
                print(f"  ✅ ATTENTION: Tagged with attention context")
            else:
                other_slices.append(result)
                print(f"  ? OTHER: Unclear categorization")
        
        # Load hierarchy metadata
        hierarchy_file = tmp.name.replace('.onnx', '_hierarchy.json')
        try:
            with open(hierarchy_file, 'r') as f:
                hierarchy_data = json.load(f)
            
            print(f"\n=== HIERARCHY METADATA ANALYSIS ===")
            htp_metadata = hierarchy_data.get('htp_metadata', {})
            slice_ops = htp_metadata.get('slice_operations', [])
            
            print(f"Slice operations tracked in metadata: {len(slice_ops)}")
            
            # Analyze context patterns
            context_patterns = {}
            for slice_op in slice_ops:
                context = slice_op.get('context', 'None')
                if context not in context_patterns:
                    context_patterns[context] = 0
                context_patterns[context] += 1
            
            print(f"Context patterns:")
            for context, count in sorted(context_patterns.items(), key=lambda x: x[1], reverse=True):
                print(f"  {context}: {count} operations")
                
                # Check if this is problematic
                if context.strip('/').split('/')[-1] in ['BertModel', 'DistilBertModel']:
                    print(f"    ❌ ROOT-LEVEL context detected")
                elif 'pooler' in context.lower():
                    print(f"    ⚠️  Pooler context (may be correct)")
                elif 'attention' in context.lower():
                    print(f"    ✅ Attention context (likely correct)")
        
        except FileNotFoundError:
            print("No hierarchy JSON file found")
        
        # Summary
        print(f"\n=== SUMMARY ===")
        print(f"Total slice nodes: {len(slice_tagging_results)}")
        print(f"Root-level only: {len(root_level_slices)}")
        print(f"Pooler context: {len(pooler_slices)}")
        print(f"Attention context: {len(attention_slices)}")
        print(f"Other: {len(other_slices)}")
        
        if len(root_level_slices) > 0:
            print(f"❌ CONFIRMED ISSUE: {len(root_level_slices)} slice operations tagged with root-level context")
            print("Root-level slices:")
            for result in root_level_slices:
                print(f"  {result['node_name']}: {result['tags']}")
        else:
            print("✅ No root-level slice tagging issues found")
        
        if len(attention_slices) > 0:
            print(f"✅ {len(attention_slices)} slice operations correctly tagged with attention context")


if __name__ == "__main__":
    test_real_bert_slice_issue()