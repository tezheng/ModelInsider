#!/usr/bin/env python3
"""
Detailed investigation of torch.Node.scopeName() method.

Based on user feedback that scopeName() returns rich information like:
'transformers.models.bert.modeling_bert.BertModel::/transformers.models.bert.modeling_bert.BertEmbeddings::embeddings'

This investigates why my previous analysis showed empty strings.
"""

import torch
import torch.jit
from transformers import AutoModel
import json
from typing import List, Dict, Any, Optional


def detailed_scope_analysis():
    """Comprehensive analysis of scopeName() across different scenarios."""
    
    print("=" * 80)
    print("Detailed torch.Node.scopeName() Investigation")
    print("=" * 80)
    
    model = AutoModel.from_pretrained('prajjwal1/bert-tiny')
    dummy_input = torch.randint(0, 1000, (1, 4))
    
    print(f"Model type: {type(model)}")
    print(f"Model class: {model.__class__.__name__}")
    
    # Test different tracing approaches
    tracing_configs = [
        {
            'name': 'Standard Trace (strict=False)',
            'func': lambda: torch.jit.trace(model, dummy_input, strict=False)
        },
        {
            'name': 'Standard Trace (strict=True)', 
            'func': lambda: torch.jit.trace(model, dummy_input, strict=True)
        },
        {
            'name': 'Trace with check_trace=False',
            'func': lambda: torch.jit.trace(model, dummy_input, check_trace=False)
        }
    ]
    
    for config in tracing_configs:
        print(f"\n🔬 {config['name']}:")
        
        try:
            traced_model = config['func']()
            graph = traced_model.graph
            nodes = list(graph.nodes())
            
            print(f"  Total nodes: {len(nodes)}")
            
            # Analyze scopeName() for all nodes
            scope_results = []
            for i, node in enumerate(nodes):
                try:
                    scope = node.scopeName() if hasattr(node, 'scopeName') else None
                    kind = node.kind() if hasattr(node, 'kind') else 'unknown'
                    
                    scope_results.append({
                        'index': i,
                        'kind': kind,
                        'scope': scope,
                        'scope_length': len(scope) if scope else 0,
                        'has_scope': bool(scope)
                    })
                except Exception as e:
                    print(f"    ERROR on node {i}: {e}")
            
            # Statistics
            non_empty_scopes = [r for r in scope_results if r['has_scope']]
            empty_scopes = [r for r in scope_results if not r['has_scope']]
            
            print(f"  Non-empty scopes: {len(non_empty_scopes)}/{len(scope_results)}")
            print(f"  Empty scopes: {len(empty_scopes)}/{len(scope_results)}")
            
            # Show examples of non-empty scopes
            if non_empty_scopes:
                print(f"  \n  📋 Non-empty scope examples:")
                for result in non_empty_scopes[:5]:
                    print(f"    [{result['index']:2d}] {result['kind']:15} scope: '{result['scope']}'")
                    
                # Show longest scopes
                longest_scopes = sorted(non_empty_scopes, key=lambda x: x['scope_length'], reverse=True)[:3]
                print(f"  \n  📏 Longest scopes:")
                for result in longest_scopes:
                    print(f"    [{result['index']:2d}] {result['kind']:15} ({result['scope_length']} chars)")
                    print(f"        '{result['scope']}'")
            else:
                print(f"  ❌ No non-empty scopes found")
                
        except Exception as e:
            print(f"  ERROR: {e}")


def test_different_model_states():
    """Test scopeName() on models in different states (eval vs train)."""
    
    print(f"\n" + "=" * 80)
    print("Testing scopeName() with Different Model States")
    print("=" * 80)
    
    model = AutoModel.from_pretrained('prajjwal1/bert-tiny')
    dummy_input = torch.randint(0, 1000, (1, 4))
    
    states = [
        ('eval mode', lambda m: m.eval()),
        ('train mode', lambda m: m.train()),
    ]
    
    for state_name, state_func in states:
        print(f"\n🔬 Model in {state_name}:")
        
        # Set model state
        state_func(model)
        
        try:
            traced = torch.jit.trace(model, dummy_input, strict=False)
            nodes = list(traced.graph.nodes())
            
            scopes_with_content = []
            for i, node in enumerate(nodes):
                scope = node.scopeName()
                if scope:
                    scopes_with_content.append((i, node.kind(), scope))
            
            print(f"  Nodes with scope info: {len(scopes_with_content)}/{len(nodes)}")
            
            for i, kind, scope in scopes_with_content[:3]:
                print(f"    [{i:2d}] {kind:15}: '{scope[:60]}{'...' if len(scope) > 60 else ''}'")
                
        except Exception as e:
            print(f"  ERROR: {e}")


def analyze_scope_parsing():
    """Analyze the structure of scope names to understand the format."""
    
    print(f"\n" + "=" * 80)
    print("Scope Name Structure Analysis")
    print("=" * 80)
    
    model = AutoModel.from_pretrained('prajjwal1/bert-tiny')
    dummy_input = torch.randint(0, 1000, (1, 4))
    
    model.eval()  # Set to eval mode
    traced = torch.jit.trace(model, dummy_input, strict=False)
    nodes = list(traced.graph.nodes())
    
    print(f"Total nodes: {len(nodes)}")
    
    # Collect all unique scopes
    all_scopes = set()
    scope_examples = []
    
    for i, node in enumerate(nodes):
        scope = node.scopeName()
        if scope:
            all_scopes.add(scope)
            scope_examples.append((i, node.kind(), scope))
    
    print(f"Unique scopes found: {len(all_scopes)}")
    print(f"Nodes with scopes: {len(scope_examples)}")
    
    if scope_examples:
        print(f"\n📋 All scope examples:")
        for i, kind, scope in scope_examples:
            print(f"  [{i:2d}] {kind:15}: {scope}")
        
        # Analyze scope patterns
        print(f"\n🔍 Scope Pattern Analysis:")
        
        # Look for common patterns
        class_patterns = set()
        module_patterns = set()
        
        for scope in all_scopes:
            # Split by :: to find patterns
            parts = scope.split('::')
            for part in parts:
                if '.' in part and 'transformers.' in part:
                    class_patterns.add(part)
                elif not '.' in part and part:
                    module_patterns.add(part)
        
        print(f"  Class patterns: {sorted(class_patterns)}")
        print(f"  Module patterns: {sorted(module_patterns)}")
        
        # Parse example scope like user provided
        example_scope = "transformers.models.bert.modeling_bert.BertModel::/transformers.models.bert.modeling_bert.BertEmbeddings::embeddings"
        print(f"\n📝 Parsing example scope:")
        print(f"  Full: '{example_scope}'")
        
        parts = example_scope.split('::')
        print(f"  Parts: {parts}")
        
        for i, part in enumerate(parts):
            if part.startswith('/'):
                print(f"    [{i}] Class path: {part[1:]}")
            elif '.' in part:
                print(f"    [{i}] Class path: {part}")
            else:
                print(f"    [{i}] Module name: {part}")


def debug_my_previous_analysis():
    """Debug why my previous analysis showed empty scopes."""
    
    print(f"\n" + "=" * 80)
    print("Debugging Previous Analysis Results")
    print("=" * 80)
    
    model = AutoModel.from_pretrained('prajjwal1/bert-tiny')
    dummy_input = torch.randint(0, 1000, (1, 4))
    
    # Replicate my previous approach exactly
    print("🔍 Replicating previous analysis approach:")
    
    traced = torch.jit.trace(model, dummy_input, strict=False)
    torch_graph = traced.graph
    torch_nodes = list(torch_graph.nodes())
    
    print(f"Total nodes: {len(torch_nodes)}")
    
    # Check first 30 nodes like I did before
    print(f"\nChecking first 30 nodes (like previous analysis):")
    
    for i in range(min(30, len(torch_nodes))):
        node = torch_nodes[i]
        scope = node.scopeName()
        kind = node.kind()
        
        print(f"  [{i:2d}] {kind:15} scope: '{scope}' (len={len(scope)})")
        
        if scope:  # If we find a non-empty scope, this is the difference!
            print(f"       ✅ FOUND NON-EMPTY SCOPE!")
            break
    
    # Check if there are scopes later in the graph
    print(f"\nScanning entire graph for scopes:")
    scope_positions = []
    for i, node in enumerate(torch_nodes):
        scope = node.scopeName()
        if scope:
            scope_positions.append((i, node.kind(), scope))
    
    print(f"Found {len(scope_positions)} nodes with scopes at positions:")
    for pos, kind, scope in scope_positions[:10]:
        print(f"  Position {pos}: {kind} -> '{scope[:50]}{'...' if len(scope) > 50 else ''}'")
    
    if not scope_positions:
        print("  ❌ Still no scopes found - investigating further...")
        
        # Try different approaches
        print(f"\n🔬 Trying different node access methods:")
        
        # Method 1: Direct iteration
        for i, node in enumerate(traced.graph.nodes()):
            scope = node.scopeName()
            if scope:
                print(f"  Direct iteration found scope at {i}: '{scope}'")
                break
        
        # Method 2: Using graph.findAllNodes
        try:
            all_nodes = traced.graph.findAllNodes("*")  # Find all nodes
            print(f"  findAllNodes returned {len(all_nodes)} nodes")
            for i, node in enumerate(all_nodes[:10]):
                scope = node.scopeName()
                if scope:
                    print(f"  findAllNodes found scope at {i}: '{scope}'")
                    break
        except:
            print(f"  findAllNodes method not available")


if __name__ == "__main__":
    try:
        # Run all analyses
        detailed_scope_analysis()
        test_different_model_states()
        analyze_scope_parsing()
        debug_my_previous_analysis()
        
        print(f"\n" + "=" * 80)
        print("SUMMARY")
        print("=" * 80)
        print("This investigation should reveal:")
        print("1. Whether scopeName() actually returns rich information")
        print("2. What conditions are needed to get non-empty scopes")
        print("3. Why my previous analysis may have missed this")
        print("4. How to properly extract hierarchy from scopeName()")
        
    except Exception as e:
        print(f"Investigation failed: {e}")
        import traceback
        traceback.print_exc()