"""
Debug the real BERT model slice tagging issue with detailed tracing.
"""

import tempfile

import torch

from modelexport.hierarchy_exporter import HierarchyExporter


def create_detailed_debug_exporter():
    """Create an exporter with very detailed debugging."""
    
    exporter = HierarchyExporter(strategy="htp")
    
    # Store original methods
    original_patch_getitem = exporter._patch_tensor_getitem
    original_get_current_tag = exporter.get_current_tag
    original_tag_slice_operations = exporter._tag_slice_operations
    
    execution_counter = {'count': 0}
    
    def debug_get_current_tag():
        tag = original_get_current_tag()
        return tag
    
    def debug_patch_tensor_getitem():
        if exporter._original_getitem is None:
            exporter._original_getitem = torch.Tensor.__getitem__
            
            def debug_context_aware_getitem(tensor_self, key):
                is_slice = exporter._is_slice_operation(key)
                
                if is_slice:
                    execution_counter['count'] += 1
                    current_tag = debug_get_current_tag()
                    
                    print(f"[SLICE {execution_counter['count']:02d}] shape={tensor_self.shape}, key={key}")
                    print(f"  context='{current_tag}', stack_depth={len(exporter._tag_stack)}")
                    
                    if current_tag:
                        slice_info = {
                            'tensor_id': id(tensor_self),
                            'key': str(key),
                            'context': current_tag,
                            'order': len(exporter._slice_operations),
                            'type': 'slice',
                            'execution_order': execution_counter['count'],
                            'tensor_shape': list(tensor_self.shape)
                        }
                        exporter._slice_operations.append(slice_info)
                
                return exporter._original_getitem(tensor_self, key)
            
            torch.Tensor.__getitem__ = debug_context_aware_getitem
    
    def debug_tag_slice_operations(onnx_model, onnx_nodes_by_type):
        """Debug version of slice operation tagging."""
        print(f"\n=== TAGGING SLICE OPERATIONS ===")
        
        if 'Slice' not in onnx_nodes_by_type or not exporter._slice_operations:
            print("No slice operations to tag")
            return
        
        slice_nodes = onnx_nodes_by_type['Slice']
        print(f"ONNX Slice nodes: {len(slice_nodes)}")
        print(f"Captured slice operations: {len(exporter._slice_operations)}")
        
        slice_operation_idx = 0
        
        for i, node in enumerate(slice_nodes):
            node_name = node.name or f"{node.op_type}_{len(exporter._tag_mapping)}"
            
            print(f"\nONNX Slice node {i}: '{node_name}'")
            
            # Skip if already tagged
            if exporter._tag_mapping[node_name]["tags"]:
                print(f"  Already tagged: {exporter._tag_mapping[node_name]['tags']}")
                continue
            
            # Use the next tracked slice operation
            if slice_operation_idx < len(exporter._slice_operations):
                slice_op = exporter._slice_operations[slice_operation_idx]
                context = slice_op['context']
                
                print(f"  Mapping to captured slice {slice_operation_idx}:")
                print(f"    exec_order={slice_op.get('execution_order', 'unknown')}")
                print(f"    context='{context}'")
                print(f"    key='{slice_op['key']}'")
                print(f"    shape={slice_op.get('tensor_shape', 'unknown')}")
                
                # Tag the slice node with the captured context
                exporter._tag_mapping[node_name]["tags"] = [context]
                slice_operation_idx += 1
                
                print(f"  ✅ Tagged '{node_name}' with context: {context}")
            else:
                print(f"  ❌ No more captured slice operations available")
    
    # Replace methods
    exporter.get_current_tag = debug_get_current_tag
    exporter._patch_tensor_getitem = debug_patch_tensor_getitem
    exporter._tag_slice_operations = debug_tag_slice_operations
    
    return exporter


def debug_real_bert_slice_issue():
    """Debug the real BERT model slice tagging issue."""
    
    print("=== Debugging Real BERT Slice Issue ===")
    
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
    text = "Hello world this is a test sequence"
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
    
    print(f"Model: {model.__class__.__name__}")
    print(f"Input shapes: {[(k, v.shape) for k, v in inputs.items()]}")
    
    # Show relevant model structure
    print(f"\nAttention module structure:")
    for name, module in model.named_modules():
        if 'attention' in name.lower() and name.count('.') <= 4:
            print(f"  {name}: {module.__class__.__name__}")
    
    # Create debug exporter
    exporter = create_detailed_debug_exporter()
    
    with tempfile.NamedTemporaryFile(suffix='.onnx') as tmp:
        print(f"\n--- Starting export with detailed debugging ---")
        
        result = exporter.export(
            model=model,
            example_inputs=inputs,
            output_path=tmp.name
        )
        
        print(f"\n--- Export completed ---")
        print(f"Export result: {result}")
        
        # Analyze results
        print(f"\n=== FINAL ANALYSIS ===")
        print(f"Total slice operations captured: {len(exporter._slice_operations)}")
        
        # Group by context
        context_groups = {}
        for slice_op in exporter._slice_operations:
            context = slice_op['context']
            if context not in context_groups:
                context_groups[context] = []
            context_groups[context].append(slice_op)
        
        print(f"\nSlice operations by context:")
        for context, ops in context_groups.items():
            print(f"  '{context}': {len(ops)} operations")
            
            # Check if this is an attention context
            if 'attention' in context.lower():
                print(f"    ✅ Attention context")
            elif context.strip('/').endswith('BertModel'):
                print(f"    ❌ Root-level context")
            elif 'embeddings' in context.lower():
                print(f"    ⚠️  Embeddings context")
            elif 'pooler' in context.lower():
                print(f"    ⚠️  Pooler context")
            else:
                print(f"    ? Other context")
        
        # Check ONNX slice node tagging
        tag_mapping = exporter.get_tag_mapping()
        slice_nodes = {name: info for name, info in tag_mapping.items() 
                      if info.get('op_type') == 'Slice'}
        
        print(f"\nONNX Slice node tagging:")
        attention_tagged = 0
        root_tagged = 0
        other_tagged = 0
        
        for node_name, node_info in slice_nodes.items():
            tags = node_info.get('tags', [])
            print(f"  {node_name}: {tags}")
            
            if any('attention' in tag.lower() for tag in tags):
                attention_tagged += 1
                print(f"    ✅ Attention-tagged")
            elif any(tag.strip('/').endswith('BertModel') for tag in tags):
                root_tagged += 1
                print(f"    ❌ Root-tagged")
            else:
                other_tagged += 1
                print(f"    ? Other tagging")
        
        print(f"\nSummary:")
        print(f"  Attention-tagged slice nodes: {attention_tagged}")
        print(f"  Root-tagged slice nodes: {root_tagged}")
        print(f"  Other-tagged slice nodes: {other_tagged}")
        
        if root_tagged > 0:
            print(f"❌ ISSUE CONFIRMED: {root_tagged} slice nodes incorrectly tagged")
        else:
            print(f"✅ No incorrect tagging detected")


if __name__ == "__main__":
    debug_real_bert_slice_issue()