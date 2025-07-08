#!/usr/bin/env python3
"""
Test using torch.jit._get_trace_graph to access raw trace graph
and check for scope information.
"""

import torch
import torch.jit
from transformers import AutoModel


def test_raw_trace_graph():
    """Test accessing the raw trace graph for scope information."""
    
    print("=" * 80)
    print("Testing Raw Trace Graph for Scope Information")
    print("=" * 80)
    
    model = AutoModel.from_pretrained('prajjwal1/bert-tiny')
    dummy_input = torch.randint(0, 1000, (1, 4))
    
    model.eval()
    
    # Method 1: Direct _get_trace_graph
    print("🔬 Method 1: torch.jit._get_trace_graph")
    
    try:
        trace_graph = torch.jit._get_trace_graph(model, dummy_input)
        print(f"  Trace graph type: {type(trace_graph)}")
        
        if hasattr(trace_graph, 'nodes'):
            nodes = list(trace_graph.nodes())
            print(f"  Total nodes: {len(nodes)}")
            
            scope_count = 0
            scope_examples = []
            
            for i, node in enumerate(nodes):
                try:
                    scope = node.scopeName()
                    if scope:
                        scope_count += 1
                        scope_examples.append((i, node.kind(), scope))
                        
                        if len(scope_examples) <= 5:
                            print(f"    [{i:2d}] {node.kind()}: '{scope}'")
                except Exception as e:
                    print(f"    Error on node {i}: {e}")
            
            print(f"  Total scopes found: {scope_count}")
            
            if scope_examples:
                print(f"\n  ✅ SUCCESS! Found scopes in raw trace graph!")
                print(f"  This confirms scopeName() can provide rich information.")
            else:
                print(f"  Still no scopes in raw trace graph")
        else:
            print(f"  Trace graph has no nodes attribute")
            
    except Exception as e:
        print(f"  ERROR: {e}")
    
    # Method 2: Compare with regular trace
    print(f"\n🔬 Method 2: Compare with regular torch.jit.trace")
    
    try:
        traced_model = torch.jit.trace(model, dummy_input, strict=False)
        regular_nodes = list(traced_model.graph.nodes())
        
        print(f"  Regular trace nodes: {len(regular_nodes)}")
        
        regular_scope_count = 0
        for node in regular_nodes:
            scope = node.scopeName()
            if scope:
                regular_scope_count += 1
        
        print(f"  Regular trace scopes: {regular_scope_count}")
        
    except Exception as e:
        print(f"  ERROR: {e}")
    
    # Method 3: Use torch.jit._trace directly
    print(f"\n🔬 Method 3: torch.jit._trace")
    
    try:
        if hasattr(torch.jit, '_trace'):
            trace_result = torch.jit._trace(model, dummy_input)
            print(f"  _trace result type: {type(trace_result)}")
            
            if hasattr(trace_result, 'graph'):
                trace_nodes = list(trace_result.graph.nodes())
                print(f"  _trace nodes: {len(trace_nodes)}")
                
                trace_scope_count = 0
                for i, node in enumerate(trace_nodes):
                    scope = node.scopeName()
                    if scope:
                        trace_scope_count += 1
                        if trace_scope_count <= 3:
                            print(f"    [{i}] {node.kind()}: '{scope}'")
                
                print(f"  _trace scopes: {trace_scope_count}")
                
    except Exception as e:
        print(f"  ERROR: {e}")


def test_graph_node_analysis():
    """Detailed analysis of graph nodes from different sources."""
    
    print(f"\n" + "=" * 80)
    print("Detailed Graph Node Analysis")
    print("=" * 80)
    
    model = AutoModel.from_pretrained('prajjwal1/bert-tiny')
    dummy_input = torch.randint(0, 1000, (1, 4))
    
    graph_sources = []
    
    # Source 1: _get_trace_graph
    try:
        raw_graph = torch.jit._get_trace_graph(model, dummy_input)
        if hasattr(raw_graph, 'nodes'):
            graph_sources.append(("Raw trace graph", list(raw_graph.nodes())))
    except Exception as e:
        print(f"Raw trace graph failed: {e}")
    
    # Source 2: Regular trace
    try:
        traced = torch.jit.trace(model, dummy_input, strict=False)
        graph_sources.append(("Regular trace", list(traced.graph.nodes())))
    except Exception as e:
        print(f"Regular trace failed: {e}")
    
    # Source 3: _trace
    try:
        if hasattr(torch.jit, '_trace'):
            trace_result = torch.jit._trace(model, dummy_input)
            if hasattr(trace_result, 'graph'):
                graph_sources.append(("_trace result", list(trace_result.graph.nodes())))
    except Exception as e:
        print(f"_trace failed: {e}")
    
    # Analyze each source
    for source_name, nodes in graph_sources:
        print(f"\n📊 {source_name}:")
        print(f"  Total nodes: {len(nodes)}")
        
        scopes_found = []
        for i, node in enumerate(nodes):
            try:
                scope = node.scopeName()
                kind = node.kind() if hasattr(node, 'kind') else 'unknown'
                
                if scope:
                    scopes_found.append((i, kind, scope))
            except Exception as e:
                print(f"    Error analyzing node {i}: {e}")
        
        print(f"  Scopes found: {len(scopes_found)}")
        
        if scopes_found:
            print(f"  Examples:")
            for i, kind, scope in scopes_found[:3]:
                print(f"    [{i:2d}] {kind}: '{scope[:60]}{'...' if len(scope) > 60 else ''}'")
            
            # Check if any match user's pattern
            user_pattern_matches = [s for _, _, s in scopes_found if '::' in s and 'transformers.models.bert' in s]
            if user_pattern_matches:
                print(f"  ✅ MATCHES USER PATTERN! Found {len(user_pattern_matches)} matching scopes")
                for scope in user_pattern_matches[:2]:
                    print(f"    '{scope}'")


def test_hook_integration():
    """Test if forward hooks can enhance scope information."""
    
    print(f"\n" + "=" * 80)
    print("Testing Forward Hook Integration with Tracing")
    print("=" * 80)
    
    model = AutoModel.from_pretrained('prajjwal1/bert-tiny')
    dummy_input = torch.randint(0, 1000, (1, 4))
    
    # Set up forward hooks to inject scope information
    module_context = {}
    current_module = [None]
    
    def forward_pre_hook(module, input):
        # Record current module
        module_name = None
        for name, mod in model.named_modules():
            if mod is module:
                module_name = name
                break
        current_module[0] = module_name or 'unknown'
    
    def forward_hook(module, input, output):
        # This runs after forward
        pass
    
    # Register hooks
    handles = []
    for name, module in model.named_modules():
        if name:  # Skip root
            pre_handle = module.register_forward_pre_hook(forward_pre_hook)
            post_handle = module.register_forward_hook(forward_hook)
            handles.extend([pre_handle, post_handle])
    
    try:
        print(f"  Registered {len(handles)} hooks")
        
        # Test with hooks active
        trace_graph = torch.jit._get_trace_graph(model, dummy_input)
        if hasattr(trace_graph, 'nodes'):
            nodes = list(trace_graph.nodes())
            
            scope_count = 0
            for node in nodes:
                scope = node.scopeName()
                if scope:
                    scope_count += 1
            
            print(f"  Nodes with hooks active: {len(nodes)}")
            print(f"  Scopes with hooks: {scope_count}")
        
    finally:
        # Clean up hooks
        for handle in handles:
            handle.remove()


if __name__ == "__main__":
    try:
        test_raw_trace_graph()
        test_graph_node_analysis()
        test_hook_integration()
        
        print(f"\n" + "=" * 80)
        print("FINAL ANALYSIS")
        print("=" * 80)
        print("This test should reveal:")
        print("1. Whether raw trace graphs contain scope information")
        print("2. Differences between tracing methods")
        print("3. How to access the rich scope data user reported")
        
    except Exception as e:
        print(f"Test failed: {e}")
        import traceback
        traceback.print_exc()