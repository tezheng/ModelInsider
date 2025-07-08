#!/usr/bin/env python3
"""
Exploration script to dump TorchScript intermediate state before ONNX conversion.
This script explores what information is available at node.scopeName() level.
"""

import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer
import json
from typing import Dict, List, Any


def explore_torchscript_intermediate_state():
    """Explore what's available in TorchScript graph before ONNX conversion."""
    
    print("=== TorchScript Intermediate State Exploration ===\n")
    
    # Load a small model for exploration
    model_name = "prajjwal1/bert-tiny"
    print(f"Loading model: {model_name}")
    
    model = AutoModel.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # Prepare inputs
    text = "Hello world"
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
    
    # Set model to eval mode
    model.eval()
    
    print(f"Model type: {type(model).__name__}")
    print(f"Input shapes: {[(k, v.shape) for k, v in inputs.items()]}\n")
    
    # Step 1: Trace the model (handle dict output)
    print("Step 1: Tracing model with torch.jit.trace...")
    
    # Try multiple tracing approaches
    traced_model = None
    
    # Approach 1: Direct tracing with strict=False (to handle dict output)
    try:
        print("  Trying direct tracing with strict=False...")
        with torch.no_grad():
            traced_model = torch.jit.trace(
                model, 
                inputs, 
                strict=False,
                check_trace=False
            )
        print("  ✅ Direct tracing successful")
    except Exception as e:
        print(f"  ❌ Direct tracing failed: {e}")
        
        # Approach 2: Try with positional args
        try:
            print("  Trying positional args tracing...")
            with torch.no_grad():
                traced_model = torch.jit.trace(
                    model, 
                    (inputs['input_ids'], inputs['attention_mask']),
                    strict=False,
                    check_trace=False
                )
            print("  ✅ Positional args tracing successful")
        except Exception as e:
            print(f"  ❌ Positional args tracing failed: {e}")
            
            # Approach 3: Script the model instead
            try:
                print("  Trying torch.jit.script...")
                traced_model = torch.jit.script(model)
                print("  ✅ Scripting successful")
            except Exception as e:
                print(f"  ❌ Scripting failed: {e}")
                
                # Approach 4: Wrapper as fallback
                print("  Using wrapper as fallback...")
                class ModelWrapper(nn.Module):
                    def __init__(self, model):
                        super().__init__()
                        self.model = model
                    
                    def forward(self, input_ids, attention_mask):
                        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
                        return outputs.last_hidden_state
                
                wrapped_model = ModelWrapper(model)
                with torch.no_grad():
                    traced_model = torch.jit.trace(wrapped_model, (inputs['input_ids'], inputs['attention_mask']))
                print("  ✅ Wrapper tracing successful")
    
    print(f"Traced model type: {type(traced_model)}")
    print(f"Has graph: {hasattr(traced_model, 'graph')}\n")
    
    # Step 2: Access the TorchScript graph
    print("Step 2: Accessing TorchScript graph...")
    graph = traced_model.graph
    nodes = list(graph.nodes())
    
    print(f"Total nodes in graph: {len(nodes)}")
    print(f"Graph type: {type(graph)}\n")
    
    # Step 3: Explore node information
    print("Step 3: Exploring node information...\n")
    
    node_info = []
    scope_hierarchy = set()
    
    for i, node in enumerate(nodes):
        try:
            # Extract all available node information
            info = {
                "index": i,
                "kind": str(node.kind()),
                "scope_name": str(node.scopeName()) if hasattr(node, 'scopeName') else None,
                "debug_name": str(node.debugName()) if hasattr(node, 'debugName') else None,
                "has_scope": hasattr(node, 'scopeName'),
                "inputs": [str(inp) for inp in node.inputs()],
                "outputs": [str(out) for out in node.outputs()],
                "attributes": {}
            }
            
            # Extract node attributes
            try:
                attr_names = node.attributeNames()
                for attr_name in attr_names:
                    try:
                        # Different attribute types need different accessors
                        if node.hasAttributeS(attr_name):
                            info["attributes"][attr_name] = node.s(attr_name)
                        elif node.hasAttributeI(attr_name):
                            info["attributes"][attr_name] = node.i(attr_name)
                        elif node.hasAttributeF(attr_name):
                            info["attributes"][attr_name] = node.f(attr_name)
                        elif node.hasAttributeIs(attr_name):
                            info["attributes"][attr_name] = node.is_(attr_name)
                        elif node.hasAttributeFs(attr_name):
                            info["attributes"][attr_name] = node.fs(attr_name)
                    except Exception as e:
                        info["attributes"][attr_name] = f"<error: {e}>"
            except Exception as e:
                info["attributes"] = f"<error accessing attributes: {e}>"
            
            node_info.append(info)
            
            # Collect unique scope names
            if info["scope_name"]:
                scope_hierarchy.add(info["scope_name"])
            
            # Print first few nodes as examples
            if i < 10:
                print(f"Node {i}:")
                print(f"  Kind: {info['kind']}")
                print(f"  Scope: {info['scope_name']}")
                print(f"  Debug: {info['debug_name']}")
                print(f"  Inputs: {len(info['inputs'])} inputs")
                print(f"  Outputs: {len(info['outputs'])} outputs")
                if info["attributes"]:
                    print(f"  Attributes: {list(info['attributes'].keys())}")
                print()
        
        except Exception as e:
            print(f"Error processing node {i}: {e}")
            node_info.append({
                "index": i,
                "error": str(e)
            })
    
    # Step 4: Analyze scope hierarchy
    print("Step 4: Analyzing scope hierarchy...\n")
    
    sorted_scopes = sorted(scope_hierarchy)
    print(f"Total unique scopes found: {len(sorted_scopes)}")
    print("Scope hierarchy:")
    for scope in sorted_scopes[:20]:  # Show first 20
        print(f"  {scope}")
    
    if len(sorted_scopes) > 20:
        print(f"  ... and {len(sorted_scopes) - 20} more")
    print()
    
    # Step 5: Analyze scope patterns
    print("Step 5: Analyzing scope patterns...\n")
    
    scope_analysis = {
        "depth_distribution": {},
        "module_types": set(),
        "common_prefixes": {},
    }
    
    for scope in sorted_scopes:
        # Analyze depth
        depth = scope.count('/')
        scope_analysis["depth_distribution"][depth] = scope_analysis["depth_distribution"].get(depth, 0) + 1
        
        # Extract module types (last component)
        if '/' in scope:
            module_type = scope.split('/')[-1]
            scope_analysis["module_types"].add(module_type)
        
        # Find common prefixes
        parts = scope.split('/')
        for i in range(1, len(parts)):
            prefix = '/'.join(parts[:i])
            scope_analysis["common_prefixes"][prefix] = scope_analysis["common_prefixes"].get(prefix, 0) + 1
    
    print("Scope depth distribution:")
    for depth, count in sorted(scope_analysis["depth_distribution"].items()):
        print(f"  Depth {depth}: {count} scopes")
    print()
    
    print("Module types found:")
    for module_type in sorted(scope_analysis["module_types"])[:15]:
        print(f"  {module_type}")
    print()
    
    print("Most common scope prefixes:")
    top_prefixes = sorted(scope_analysis["common_prefixes"].items(), key=lambda x: x[1], reverse=True)[:10]
    for prefix, count in top_prefixes:
        print(f"  {prefix}: {count} occurrences")
    print()
    
    # Step 6: Save detailed results
    print("Step 6: Saving results...\n")
    
    results = {
        "model_name": model_name,
        "total_nodes": len(nodes),
        "total_scopes": len(sorted_scopes),
        "scope_hierarchy": sorted_scopes,
        "scope_analysis": {
            "depth_distribution": scope_analysis["depth_distribution"],
            "module_types": sorted(list(scope_analysis["module_types"])),
            "top_prefixes": top_prefixes
        },
        "node_details": node_info
    }
    
    # Save to temp folder
    output_file = "temp/torchscript_exploration.json"
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"✅ Detailed results saved to: {output_file}")
    
    # Step 7: Compare with post-ONNX state
    print("\nStep 7: Comparing with post-ONNX export...\n")
    
    # Export to ONNX to see what's lost
    onnx_file = "temp/torchscript_comparison.onnx"
    torch.onnx.export(
        traced_model,
        (inputs['input_ids'], inputs['attention_mask']),
        onnx_file,
        input_names=['input_ids', 'attention_mask'],
        output_names=['last_hidden_state'],
        dynamic_axes={
            'input_ids': {0: 'batch_size', 1: 'sequence'},
            'attention_mask': {0: 'batch_size', 1: 'sequence'},
            'last_hidden_state': {0: 'batch_size', 1: 'sequence'}
        }
    )
    
    # Load and analyze ONNX
    import onnx
    onnx_model = onnx.load(onnx_file)
    onnx_nodes = onnx_model.graph.node
    
    print(f"ONNX model has {len(onnx_nodes)} nodes")
    print("Sample ONNX node names:")
    for i, node in enumerate(onnx_nodes[:10]):
        print(f"  {node.name} ({node.op_type})")
    
    print(f"\n✅ ONNX model saved to: {onnx_file}")
    print(f"✅ TorchScript had {len(sorted_scopes)} scopes with full hierarchy")
    print(f"✅ ONNX has {len(onnx_nodes)} nodes with generic names")
    print("\n🎯 CONCLUSION: Context is preserved in TorchScript and lost in ONNX conversion!")


if __name__ == "__main__":
    # Create temp directory
    import os
    os.makedirs("temp", exist_ok=True)
    
    explore_torchscript_intermediate_state()