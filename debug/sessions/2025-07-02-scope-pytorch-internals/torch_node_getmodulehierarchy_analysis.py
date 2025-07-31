#!/usr/bin/env python3
"""
Research on torch.Node.getModuleHierarchy() method.

This investigates what hierarchy information this method provides and how it
could potentially enhance semantic mapping approaches.
"""


import torch
import torch.jit
from transformers import AutoModel


def test_getmodulehierarchy_method():
    """Test torch.Node.getModuleHierarchy() method comprehensively."""
    
    print("=" * 80)
    print("torch.Node.getModuleHierarchy() Method Analysis")
    print("=" * 80)
    
    # Load model and create traced graph
    model = AutoModel.from_pretrained('prajjwal1/bert-tiny')
    dummy_input = torch.randint(0, 1000, (1, 4))
    
    # Get PyTorch JIT graph
    traced = torch.jit.trace(model, dummy_input, strict=False)
    torch_graph = traced.graph
    
    torch_nodes = list(torch_graph.nodes())
    print(f"\nTotal PyTorch nodes: {len(torch_nodes)}")
    
    # Check if getModuleHierarchy exists
    sample_node = torch_nodes[10] if len(torch_nodes) > 10 else torch_nodes[0]
    
    print(f"\nSample Node: {sample_node}")
    print(f"Node type: {type(sample_node)}")
    
    # Check for getModuleHierarchy method
    has_method = hasattr(sample_node, 'getModuleHierarchy')
    print(f"\nHas getModuleHierarchy method: {has_method}")
    
    if has_method:
        print("\n🔍 Testing getModuleHierarchy() method:")
        try:
            hierarchy = sample_node.getModuleHierarchy()
            print(f"  Return type: {type(hierarchy)}")
            print(f"  Return value: {hierarchy}")
            print(f"  Return repr: {hierarchy!r}")
        except Exception as e:
            print(f"  ERROR calling getModuleHierarchy(): {e}")
            import traceback
            traceback.print_exc()
    
    # Test method on multiple nodes
    print("\n📊 Testing getModuleHierarchy() on multiple nodes:")
    
    hierarchy_results = []
    for i, node in enumerate(torch_nodes):
        if i >= 20:  # Limit to first 20 nodes
            break
            
        if not hasattr(node, 'getModuleHierarchy'):
            continue
            
        try:
            hierarchy = node.getModuleHierarchy()
            node_info = {
                'node_index': i,
                'node_kind': node.kind() if hasattr(node, 'kind') else 'unknown',
                'scope_name': node.scopeName() if hasattr(node, 'scopeName') else '',
                'hierarchy': hierarchy,
                'hierarchy_type': str(type(hierarchy)),
                'hierarchy_len': len(hierarchy) if hasattr(hierarchy, '__len__') else 'no_len'
            }
            hierarchy_results.append(node_info)
            
        except Exception as e:
            print(f"    Node[{i:2d}]: ERROR - {e}")
    
    print(f"\nSuccessfully analyzed {len(hierarchy_results)} nodes with getModuleHierarchy()")
    
    if hierarchy_results:
        print("\n📋 Sample Results:")
        for result in hierarchy_results[:5]:
            print(f"  Node[{result['node_index']:2d}] {result['node_kind']:15}")
            print(f"    scope: '{result['scope_name']}'")
            print(f"    hierarchy: {result['hierarchy']}")
            print(f"    hierarchy type: {result['hierarchy_type']}")
            print(f"    hierarchy length: {result['hierarchy_len']}")
            print()
    
    # Analyze hierarchy patterns
    if hierarchy_results:
        print("\n🔍 Hierarchy Pattern Analysis:")
        
        unique_hierarchies = set()
        non_empty_hierarchies = []
        
        for result in hierarchy_results:
            hierarchy = result['hierarchy']
            unique_hierarchies.add(str(hierarchy))
            
            if hierarchy and hierarchy != [] and hierarchy != () and hierarchy != "":
                non_empty_hierarchies.append(result)
        
        print(f"  Unique hierarchy values: {len(unique_hierarchies)}")
        print(f"  Non-empty hierarchies: {len(non_empty_hierarchies)}")
        
        if non_empty_hierarchies:
            print(f"\n  Non-empty hierarchy examples:")
            for result in non_empty_hierarchies[:3]:
                print(f"    {result['node_kind']:15} -> {result['hierarchy']}")
        else:
            print(f"  All hierarchies appear to be empty")
    
    return hierarchy_results


def compare_with_scopename():
    """Compare getModuleHierarchy() results with scopeName() method."""
    
    print("\n" + "=" * 80)
    print("Comparison: getModuleHierarchy() vs scopeName()")
    print("=" * 80)
    
    model = AutoModel.from_pretrained('prajjwal1/bert-tiny')
    dummy_input = torch.randint(0, 1000, (1, 4))
    
    traced = torch.jit.trace(model, dummy_input, strict=False)
    torch_graph = traced.graph
    torch_nodes = list(torch_graph.nodes())
    
    comparison_results = []
    
    for i, node in enumerate(torch_nodes[:30]):  # First 30 nodes
        scope_name = node.scopeName() if hasattr(node, 'scopeName') else ''
        
        hierarchy = None
        hierarchy_error = None
        
        if hasattr(node, 'getModuleHierarchy'):
            try:
                hierarchy = node.getModuleHierarchy()
            except Exception as e:
                hierarchy_error = str(e)
        
        node_kind = node.kind() if hasattr(node, 'kind') else 'unknown'
        
        comparison_results.append({
            'index': i,
            'kind': node_kind,
            'scope_name': scope_name,
            'hierarchy': hierarchy,
            'hierarchy_error': hierarchy_error,
            'both_empty': not scope_name and not hierarchy
        })
    
    # Display comparison
    print(f"\nComparing scopeName() vs getModuleHierarchy() for {len(comparison_results)} nodes:\n")
    print(f"{'Idx':>3} {'Kind':15} {'scopeName()':30} {'getModuleHierarchy()':30}")
    print("-" * 80)
    
    for result in comparison_results:
        scope_display = f"'{result['scope_name']}'" if result['scope_name'] else 'empty'
        
        if result['hierarchy_error']:
            hierarchy_display = f"ERROR: {result['hierarchy_error'][:20]}"
        elif result['hierarchy']:
            hierarchy_display = str(result['hierarchy'])[:28]
        else:
            hierarchy_display = 'empty'
        
        print(f"{result['index']:>3} {result['kind']:15} {scope_display:30} {hierarchy_display:30}")
    
    # Statistics
    empty_scopes = sum(1 for r in comparison_results if not r['scope_name'])
    empty_hierarchies = sum(1 for r in comparison_results if not r['hierarchy'])
    both_empty = sum(1 for r in comparison_results if r['both_empty'])
    hierarchy_errors = sum(1 for r in comparison_results if r['hierarchy_error'])
    
    print(f"\n📊 Statistics:")
    print(f"  Empty scopeName(): {empty_scopes}/{len(comparison_results)}")
    print(f"  Empty getModuleHierarchy(): {empty_hierarchies}/{len(comparison_results)}")
    print(f"  Both empty: {both_empty}/{len(comparison_results)}")
    print(f"  getModuleHierarchy() errors: {hierarchy_errors}/{len(comparison_results)}")


def test_different_tracing_modes():
    """Test getModuleHierarchy() with different tracing modes."""
    
    print("\n" + "=" * 80)
    print("Testing getModuleHierarchy() with Different Tracing Modes")
    print("=" * 80)
    
    model = AutoModel.from_pretrained('prajjwal1/bert-tiny')
    dummy_input = torch.randint(0, 1000, (1, 4))
    
    tracing_modes = [
        ("Standard trace", lambda: torch.jit.trace(model, dummy_input, strict=False)),
        ("Strict trace", lambda: torch.jit.trace(model, dummy_input, strict=True)),
    ]
    
    # Also try script mode if possible
    try:
        scripted = torch.jit.script(model)
        tracing_modes.append(("Scripted model", lambda: scripted))
    except Exception as e:
        print(f"Note: Could not script model: {e}")
    
    for mode_name, trace_func in tracing_modes:
        print(f"\n🔬 {mode_name}:")
        
        try:
            traced_model = trace_func()
            graph = traced_model.graph
            nodes = list(graph.nodes())
            
            print(f"  Total nodes: {len(nodes)}")
            
            # Test getModuleHierarchy on first few nodes
            hierarchy_found = 0
            for i, node in enumerate(nodes[:10]):
                if hasattr(node, 'getModuleHierarchy'):
                    try:
                        hierarchy = node.getModuleHierarchy()
                        if hierarchy:
                            hierarchy_found += 1
                            print(f"    Node[{i}] {node.kind()}: {hierarchy}")
                    except:
                        pass
            
            if hierarchy_found == 0:
                print(f"    No non-empty hierarchies found in first 10 nodes")
                
        except Exception as e:
            print(f"    ERROR with {mode_name}: {e}")


def analyze_method_availability():
    """Check what methods are available on torch.Node objects."""
    
    print("\n" + "=" * 80)
    print("torch.Node Method Availability Analysis")
    print("=" * 80)
    
    model = AutoModel.from_pretrained('prajjwal1/bert-tiny')
    dummy_input = torch.randint(0, 1000, (1, 4))
    
    traced = torch.jit.trace(model, dummy_input, strict=False)
    sample_node = list(traced.graph.nodes())[0]
    
    print(f"Sample node: {sample_node}")
    print(f"Node type: {type(sample_node)}")
    
    # Get all methods and attributes
    all_attrs = dir(sample_node)
    methods = [attr for attr in all_attrs if callable(getattr(sample_node, attr, None))]
    
    print(f"\nAll methods on torch.Node:")
    for method in sorted(methods):
        print(f"  {method}")
    
    # Specifically check hierarchy-related methods
    hierarchy_methods = [method for method in methods if 'hierarchy' in method.lower() or 'scope' in method.lower() or 'module' in method.lower()]
    
    print(f"\nHierarchy/scope/module related methods:")
    for method in hierarchy_methods:
        print(f"  {method}")
        
        # Try to call the method if it looks safe
        if method in ['getModuleHierarchy', 'scopeName']:
            try:
                result = getattr(sample_node, method)()
                print(f"    -> {result!r}")
            except Exception as e:
                print(f"    -> ERROR: {e}")


if __name__ == "__main__":
    try:
        # Main analysis
        hierarchy_results = test_getmodulehierarchy_method()
        
        # Comparison analysis
        compare_with_scopename()
        
        # Different tracing modes
        test_different_tracing_modes()
        
        # Method availability
        analyze_method_availability()
        
        print("\n" + "=" * 80)
        print("Summary and Conclusions")
        print("=" * 80)
        
        print("\n🎯 Key Findings:")
        print("1. torch.Node.getModuleHierarchy() method availability and behavior")
        print("2. Comparison with existing scopeName() method")
        print("3. Effectiveness across different PyTorch tracing modes")
        print("4. Potential integration with Enhanced Semantic Exporter")
        
        print("\n💡 For Enhanced Semantic Exporter:")
        print("• Evaluate if getModuleHierarchy() provides richer context than scopeName()")
        print("• Consider integration for improved semantic mapping accuracy")
        print("• Test compatibility with current ONNX node scope parsing approach")
        
    except Exception as e:
        print(f"Analysis failed: {e}")
        import traceback
        traceback.print_exc()