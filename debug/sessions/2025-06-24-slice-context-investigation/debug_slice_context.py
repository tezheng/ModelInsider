"""
Debug slice operation context capture to understand why slices get wrong tags.
"""

import tempfile

import torch
import torch.nn as nn

from modelexport.hierarchy_exporter import HierarchyExporter


class DebugAttentionModule(nn.Module):
    """Simple attention module for debugging slice context capture."""
    
    def __init__(self):
        super().__init__()
        self.query = nn.Linear(768, 768)
        
    def forward(self, x):
        print(f"[DEBUG] Entering DebugAttentionModule.forward, input shape: {x.shape}")
        
        # This slice should be captured with this module's context
        print(f"[DEBUG] About to perform slice operation...")
        sliced = x[:, 1:-1, :]  # This should be tagged with this module
        print(f"[DEBUG] Slice completed, output shape: {sliced.shape}")
        
        # Apply query transformation
        result = self.query(sliced)
        print(f"[DEBUG] Exiting DebugAttentionModule.forward, output shape: {result.shape}")
        return result


class DebugModel(nn.Module):
    """Model for debugging slice context issues."""
    
    def __init__(self):
        super().__init__()
        self.embeddings = nn.Embedding(100, 768)
        self.attention = DebugAttentionModule()
        self.pooler = nn.Linear(768, 768)
        
    def forward(self, input_ids):
        print(f"[DEBUG] Entering DebugModel.forward")
        
        # Embeddings
        print(f"[DEBUG] Computing embeddings...")
        hidden_states = self.embeddings(input_ids)
        print(f"[DEBUG] Embeddings shape: {hidden_states.shape}")
        
        # Attention (this is where slice should happen)
        print(f"[DEBUG] Calling attention module...")
        attention_output = self.attention(hidden_states)
        print(f"[DEBUG] Attention output shape: {attention_output.shape}")
        
        # Pooling
        print(f"[DEBUG] Computing pooler...")
        pooled = self.pooler(attention_output[:, 0, :])  # Another slice operation
        print(f"[DEBUG] Pooled shape: {pooled.shape}")
        
        print(f"[DEBUG] Exiting DebugModel.forward")
        return pooled


def debug_slice_context_capture():
    """Debug why slice operations get wrong context tags."""
    
    print("=== Debugging Slice Context Capture ===")
    
    model = DebugModel()
    model.eval()
    inputs = torch.randint(0, 100, (1, 5))  # Small input for debugging
    
    print(f"Model structure:")
    for name, module in model.named_modules():
        if name:
            print(f"  {name}: {module.__class__.__name__}")
    
    # Create custom exporter with debugging
    exporter = HierarchyExporter(strategy="htp")
    
    # Monkey patch some debug info into the exporter
    original_get_current_tag = exporter.get_current_tag
    original_patch_getitem = exporter._patch_tensor_getitem
    
    def debug_get_current_tag():
        tag = original_get_current_tag()
        print(f"[DEBUG] get_current_tag() -> {tag}")
        print(f"[DEBUG] Current tag stack: {exporter._tag_stack}")
        return tag
    
    def debug_patch_tensor_getitem():
        """Enhanced patch with debug output."""
        if exporter._original_getitem is None:  # Only patch once
            exporter._original_getitem = torch.Tensor.__getitem__
            
            def debug_context_aware_getitem(tensor_self, key):
                # Check if this is a slice operation
                is_slice = exporter._is_slice_operation(key)
                
                if is_slice:
                    print(f"[DEBUG] SLICE OPERATION DETECTED!")
                    print(f"  Key: {key}")
                    print(f"  Tensor shape: {tensor_self.shape}")
                    print(f"  Current tag stack: {exporter._tag_stack}")
                
                # Capture current module context from stack
                current_tag = debug_get_current_tag()
                
                # Record slice operation if we have context and it's a slice
                if current_tag and is_slice:
                    slice_info = {
                        'tensor_id': id(tensor_self),
                        'key': str(key),
                        'context': current_tag,
                        'order': len(exporter._slice_operations),
                        'type': 'slice'
                    }
                    exporter._slice_operations.append(slice_info)
                    print(f"[DEBUG] Recorded slice operation: {slice_info}")
                
                # Execute original __getitem__
                result = exporter._original_getitem(tensor_self, key)
                
                if is_slice:
                    print(f"[DEBUG] Slice result shape: {result.shape}")
                
                return result
            
            # Apply the patch
            torch.Tensor.__getitem__ = debug_context_aware_getitem
    
    # Apply debug patches
    exporter.get_current_tag = debug_get_current_tag
    exporter._patch_tensor_getitem = debug_patch_tensor_getitem
    
    with tempfile.NamedTemporaryFile(suffix='.onnx') as tmp:
        print(f"\n--- Starting export with debugging ---")
        
        result = exporter.export(
            model=model,
            example_inputs=inputs,
            output_path=tmp.name
        )
        
        print(f"\n--- Export completed ---")
        print(f"Export result: {result}")
        
        print(f"\n--- Analyzing captured slice operations ---")
        print(f"Total slice operations captured: {len(exporter._slice_operations)}")
        
        for i, slice_op in enumerate(exporter._slice_operations):
            print(f"  {i}: context='{slice_op['context']}', key='{slice_op['key']}'")
            
            # Check if the context is what we expect
            if 'DebugAttentionModule' in slice_op['context']:
                print(f"    ✅ Correct context captured")
            else:
                print(f"    ❌ Wrong context captured, expected DebugAttentionModule")
        
        # Check ONNX slice nodes and their tags
        tag_mapping = exporter.get_tag_mapping()
        slice_nodes = {name: info for name, info in tag_mapping.items() 
                      if info.get('op_type') == 'Slice'}
        
        print(f"\n--- ONNX Slice nodes and their tags ---")
        for node_name, node_info in slice_nodes.items():
            tags = node_info.get('tags', [])
            print(f"  {node_name}: {tags}")
            
            if any('DebugAttentionModule' in tag for tag in tags):
                print(f"    ✅ ONNX node correctly tagged")
            else:
                print(f"    ❌ ONNX node incorrectly tagged")


def test_stack_behavior_during_forward():
    """Test the hook stack behavior during forward pass."""
    
    print("\n\n=== Testing Hook Stack Behavior ===")
    
    model = DebugModel()
    model.eval()
    inputs = torch.randint(0, 100, (1, 5))
    
    exporter = HierarchyExporter(strategy="htp")
    exporter._reset_state()
    exporter._model = model
    
    # Monkey patch the hook creation to add debug output
    original_create_pre_hook = None
    original_create_post_hook = None
    
    def debug_create_pre_hook(module_name: str, module: torch.nn.Module):
        """Debug version of pre-hook creation."""
        def debug_pre_hook(module, inputs):
            hierarchical_tag = exporter._build_tag(module_name, module)
            print(f"[HOOK] PRE-HOOK: {module_name} -> pushing tag {hierarchical_tag}")
            print(f"[HOOK] Stack before push: {exporter._tag_stack}")
            exporter._tag_stack.append(hierarchical_tag)
            print(f"[HOOK] Stack after push: {exporter._tag_stack}")
            
            exporter._operation_context[module_name] = {
                "tag": hierarchical_tag,
                "module_class": module.__class__.__name__,
                "creates_hierarchy": True,
                "stack_depth": len(exporter._tag_stack),
            }
        return debug_pre_hook
    
    def debug_create_post_hook(module_name: str, module: torch.nn.Module):
        """Debug version of post-hook creation."""
        def debug_post_hook(module, inputs, outputs):
            print(f"[HOOK] POST-HOOK: {module_name} -> popping tag")
            print(f"[HOOK] Stack before pop: {exporter._tag_stack}")
            if exporter._tag_stack:
                popped = exporter._tag_stack.pop()
                print(f"[HOOK] Popped tag: {popped}")
            print(f"[HOOK] Stack after pop: {exporter._tag_stack}")
        return debug_post_hook
    
    # Monkey patch the exporter's hook creation methods
    import types
    exporter.debug_create_pre_hook = types.MethodType(debug_create_pre_hook, exporter)
    exporter.debug_create_post_hook = types.MethodType(debug_create_post_hook, exporter)
    
    # Register hooks manually with debug versions
    for name, module in model.named_modules():
        if name:  # Skip root module
            module_class = module.__class__.__module__
            should_tag = exporter._should_tag_module(module_class)
            
            if should_tag:
                creates_hierarchy = exporter._should_create_hierarchy_level(module)
                print(f"[SETUP] Module {name} ({module.__class__.__name__}): should_tag={should_tag}, creates_hierarchy={creates_hierarchy}")
                
                if creates_hierarchy:
                    pre_hook = module.register_forward_pre_hook(
                        exporter.debug_create_pre_hook(name, module)
                    )
                    exporter._pre_hooks.append(pre_hook)
                    
                    post_hook = module.register_forward_hook(
                        exporter.debug_create_post_hook(name, module)
                    )
                    exporter._post_hooks.append(post_hook)
    
    # Now run the forward pass and observe stack behavior
    print(f"\n--- Running forward pass with hook debugging ---")
    with torch.no_grad():
        output = model(inputs)
    
    print(f"\n--- Final stack state ---")
    print(f"Final tag stack: {exporter._tag_stack}")
    print(f"Final operation context keys: {list(exporter._operation_context.keys())}")
    
    exporter._remove_hooks()


if __name__ == "__main__":
    debug_slice_context_capture()
    test_stack_behavior_during_forward()