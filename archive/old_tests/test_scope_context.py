#!/usr/bin/env python3
"""
Test accessing torch nodes in different contexts to try to replicate
the user's scope information.
"""

import torch
import torch.jit
from transformers import AutoModel


def test_onnx_export_context():
    """Test if scopes are available during ONNX export process."""
    
    print("=" * 80)
    print("Testing scopeName() During ONNX Export Process")
    print("=" * 80)
    
    model = AutoModel.from_pretrained('prajjwal1/bert-tiny')
    dummy_input = torch.randint(0, 1000, (1, 4))
    
    # Try to capture the graph during ONNX export
    class GraphCapture:
        def __init__(self):
            self.captured_graph = None
            self.captured_nodes = []
        
        def capture_during_export(self):
            # Hook into the export process
            original_trace = torch.jit.trace
            
            def trace_with_capture(*args, **kwargs):
                result = original_trace(*args, **kwargs)
                self.captured_graph = result.graph
                self.captured_nodes = list(result.graph.nodes())
                return result
            
            # Temporarily replace trace function
            torch.jit.trace = trace_with_capture
            
            try:
                # This should trigger tracing
                torch.onnx.export(
                    model, 
                    dummy_input, 
                    'temp_scope_test.onnx',
                    verbose=False,
                    opset_version=17
                )
            finally:
                # Restore original function
                torch.jit.trace = original_trace
    
    capture = GraphCapture()
    capture.capture_during_export()
    
    if capture.captured_nodes:
        print(f"Captured {len(capture.captured_nodes)} nodes during ONNX export")
        
        scope_count = 0
        for i, node in enumerate(capture.captured_nodes):
            scope = node.scopeName()
            if scope:
                scope_count += 1
                print(f"  [{i:2d}] {node.kind()}: '{scope}'")
        
        print(f"Total nodes with scopes: {scope_count}")
    else:
        print("No nodes captured during export")


def test_direct_graph_access():
    """Test accessing the graph through different methods."""
    
    print(f"\n" + "=" * 80)
    print("Testing Direct Graph Access Methods")
    print("=" * 80)
    
    model = AutoModel.from_pretrained('prajjwal1/bert-tiny')
    dummy_input = torch.randint(0, 1000, (1, 4))
    
    # Method 1: Through torch.jit._get_trace_graph
    print("🔬 Method 1: torch.jit._get_trace_graph")
    try:
        if hasattr(torch.jit, '_get_trace_graph'):
            trace_graph = torch.jit._get_trace_graph(model, dummy_input)
            print(f"  Got trace graph: {trace_graph}")
        else:
            print("  _get_trace_graph not available")
    except Exception as e:
        print(f"  ERROR: {e}")
    
    # Method 2: Through torch.jit._trace
    print("\n🔬 Method 2: torch.jit._trace")  
    try:
        if hasattr(torch.jit, '_trace'):
            trace_result = torch.jit._trace(model, dummy_input)
            print(f"  Got trace result: {type(trace_result)}")
            
            if hasattr(trace_result, 'graph'):
                nodes = list(trace_result.graph.nodes())
                print(f"  Nodes: {len(nodes)}")
                
                for i, node in enumerate(nodes[:5]):
                    scope = node.scopeName()
                    if scope:
                        print(f"    [{i}] {node.kind()}: '{scope}'")
        else:
            print("  _trace not available")
    except Exception as e:
        print(f"  ERROR: {e}")
    
    # Method 3: Check if there are module-level hooks
    print("\n🔬 Method 3: Module-level scope capture")
    try:
        # Add module names to the trace
        def add_module_info(module, input, output):
            pass
        
        # Get the trace while hooks are active
        handles = []
        for name, module in model.named_modules():
            if name:  # Skip root module
                handle = module.register_forward_hook(add_module_info)
                handles.append(handle)
        
        try:
            traced = torch.jit.trace(model, dummy_input, strict=False)
            print(f"  Traced with {len(handles)} hooks active")
            
            nodes = list(traced.graph.nodes())
            scope_count = 0
            for node in nodes:
                scope = node.scopeName()
                if scope:
                    scope_count += 1
                    print(f"  Found scope: '{scope}'")
            
            print(f"  Scopes found: {scope_count}")
            
        finally:
            for handle in handles:
                handle.remove()
                
    except Exception as e:
        print(f"  ERROR: {e}")


def show_user_expected_format():
    """Show what the user's scope format tells us about the expected structure."""
    
    print(f"\n" + "=" * 80)
    print("Analysis of User's Scope Format")
    print("=" * 80)
    
    user_scope = "transformers.models.bert.modeling_bert.BertModel::/transformers.models.bert.modeling_bert.BertEmbeddings::embeddings"
    
    print(f"User's scope: '{user_scope}'")
    print(f"Length: {len(user_scope)} characters")
    
    # Parse the components
    parts = user_scope.split('::')
    print(f"\nParsed components:")
    for i, part in enumerate(parts):
        print(f"  [{i}] '{part}'")
    
    if len(parts) >= 3:
        parent_class = parts[0]
        current_class = parts[1].lstrip('/')
        module_name = parts[2]
        
        print(f"\nStructure analysis:")
        print(f"  Parent class: {parent_class}")
        print(f"  Current class: {current_class}")  
        print(f"  Module name: {module_name}")
        
        print(f"\nThis suggests:")
        print(f"  • Scopes contain full class paths")
        print(f"  • :: is used as separator")
        print(f"  • Module hierarchy is preserved")
        print(f"  • Both parent and current module info is included")
    
    # Show what we would expect for different modules
    model = AutoModel.from_pretrained('prajjwal1/bert-tiny')
    
    print(f"\nExpected scopes for our model:")
    for name, module in list(model.named_modules())[:5]:
        if name:
            module_class = f"transformers.models.bert.modeling_bert.{module.__class__.__name__}"
            expected_scope = f"transformers.models.bert.modeling_bert.BertModel::{module_class}::{name}"
            print(f"  {name}: '{expected_scope}'")


if __name__ == "__main__":
    try:
        test_onnx_export_context()
        test_direct_graph_access()
        show_user_expected_format()
        
        print(f"\n" + "=" * 80)
        print("CONCLUSION")
        print("=" * 80)
        print("If scopeName() still returns empty strings, this indicates:")
        print("1. Environment/version differences with user's setup")
        print("2. User may be accessing nodes from different context")
        print("3. The rich scope info exists but requires specific conditions")
        print("4. Our Enhanced Semantic Exporter's ONNX parsing approach remains valid")
        
    except Exception as e:
        print(f"Test failed: {e}")
        import traceback
        traceback.print_exc()