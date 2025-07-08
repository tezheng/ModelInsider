#!/usr/bin/env python3
"""
Analysis of torch.Node properties and their relationship to ONNX nodes.

This helps understand how to extract detailed information from torch.Node
objects during the PyTorch -> ONNX conversion process.
"""

import torch
import onnx
from transformers import AutoModel


def analyze_torch_node_properties():
    """Comprehensive analysis of torch.Node properties."""
    
    model = AutoModel.from_pretrained('prajjwal1/bert-tiny')
    dummy_input = torch.randint(0, 1000, (1, 4))
    
    # Get PyTorch JIT graph
    traced = torch.jit.trace(model, dummy_input, strict=False)
    torch_graph = traced.graph
    
    # Export to ONNX for comparison
    torch.onnx.export(model, dummy_input, 'node_analysis.onnx', verbose=False)
    onnx_model = onnx.load('node_analysis.onnx')
    
    print("=" * 80)
    print("torch.Node Property Analysis")
    print("=" * 80)
    
    # Analyze representative torch nodes
    torch_nodes = list(torch_graph.nodes())
    
    print(f"\nTotal PyTorch nodes: {len(torch_nodes)}")
    print(f"Total ONNX nodes: {len(onnx_model.graph.node)}")
    
    print("\n" + "="*50)
    print("Key torch.Node Properties and Methods:")
    print("="*50)
    
    # Find a representative node
    sample_node = None
    for node in torch_nodes:
        if hasattr(node, 'scopeName') and node.scopeName():
            sample_node = node
            break
    
    if not sample_node:
        sample_node = torch_nodes[10]  # Use a middle node
    
    print(f"\nSample Node: {sample_node}")
    print(f"Node type: {type(sample_node)}")
    
    # Key methods for getting node information
    key_methods = [
        ('kind', 'Operation type (e.g., aten::add, prim::Constant)'),
        ('scopeName', 'Module scope information'),
        ('outputsSize', 'Number of outputs'),
        ('inputsSize', 'Number of inputs'),
        ('attributeNames', 'Available attributes'),
        ('hasMultipleOutputs', 'Whether node has multiple outputs'),
    ]
    
    print("\n📋 Key Methods:")
    for method_name, description in key_methods:
        if hasattr(sample_node, method_name):
            try:
                result = getattr(sample_node, method_name)()
                print(f"  {method_name:18} -> {result!r:30} ({description})")
            except Exception as e:
                print(f"  {method_name:18} -> ERROR: {e}")
    
    # Output analysis
    print("\n📤 Output Analysis:")
    if hasattr(sample_node, 'outputs'):
        outputs = list(sample_node.outputs())
        print(f"  Number of outputs: {len(outputs)}")
        for i, output in enumerate(outputs[:3]):
            if hasattr(output, 'debugName'):
                debug_name = output.debugName()
                print(f"    output[{i}].debugName(): %{debug_name}")
            if hasattr(output, 'type'):
                output_type = output.type()
                print(f"    output[{i}].type(): {output_type}")
    
    # Scope analysis
    print("\n🔍 Scope Analysis:")
    scope_examples = []
    for i, node in enumerate(torch_nodes):
        scope = node.scopeName() if hasattr(node, 'scopeName') else ''
        kind = node.kind() if hasattr(node, 'kind') else ''
        
        if scope and len(scope_examples) < 5:
            outputs = list(node.outputs())
            output_name = outputs[0].debugName() if outputs else 'no_output'
            scope_examples.append((i, kind, scope, output_name))
    
    if scope_examples:
        print("  Nodes with scope information:")
        for idx, kind, scope, output_name in scope_examples:
            print(f"    [{idx:2d}] {kind:20} scope='{scope}' output=%{output_name}")
    else:
        print("  No nodes found with scope information in this model")
    
    # ONNX name mapping analysis
    print("\n🔗 ONNX Name Mapping:")
    print("  How torch.Node output names relate to ONNX node names:")
    
    # Collect torch output names
    torch_output_names = set()
    for node in torch_nodes:
        outputs = list(node.outputs())
        for output in outputs:
            if hasattr(output, 'debugName'):
                torch_output_names.add(output.debugName())
    
    # Show patterns
    torch_names = sorted(list(torch_output_names))[:10]
    onnx_names = [node.name for node in onnx_model.graph.node[:10]]
    
    print(f"  Sample PyTorch output names: {torch_names}")
    print(f"  Sample ONNX node names: {onnx_names}")
    
    print("\n📝 Key Insights:")
    print("  1. torch.Node.kind() gives the operation type")
    print("  2. torch.Node.scopeName() provides module hierarchy context")
    print("  3. torch.Node.outputs()[0].debugName() gives internal tensor names")
    print("  4. ONNX nodes get descriptive names like '/embeddings/Gather'")
    print("  5. The mapping from torch->ONNX involves scope resolution")
    
    print("\n🎯 For Enhanced Semantic Exporter:")
    print("  • ONNX node.name contains rich scope information")
    print("  • ONNX node.op_type maps to torch.Node.kind()")
    print("  • Scope parsing from ONNX names is more reliable than torch scope")
    print("  • torch.Node.scopeName() often empty in traced models")


def demonstrate_scope_extraction():
    """Show how scope information can be extracted from ONNX node names."""
    
    print("\n" + "="*80)
    print("ONNX Node Name Scope Extraction Demo")
    print("="*80)
    
    model = AutoModel.from_pretrained('prajjwal1/bert-tiny')
    dummy_input = torch.randint(0, 1000, (1, 4))
    
    torch.onnx.export(model, dummy_input, 'scope_demo.onnx', verbose=False)
    onnx_model = onnx.load('scope_demo.onnx')
    
    print("\nExtracting scope information from ONNX node names:")
    
    interesting_nodes = []
    for node in onnx_model.graph.node:
        name = node.name
        op_type = node.op_type
        
        # Look for nodes with rich scope information
        if '/' in name and any(keyword in name for keyword in ['encoder', 'attention', 'embeddings']):
            interesting_nodes.append((name, op_type))
    
    print(f"\nFound {len(interesting_nodes)} nodes with rich scope information:")
    for name, op_type in interesting_nodes[:10]:
        # Parse scope components
        parts = name.strip('/').split('/')
        print(f"  {op_type:12} | {name:50} -> {parts}")
    
    print("\n🔍 Scope Parsing Pattern:")
    if interesting_nodes:
        example_name, example_op = interesting_nodes[0]
        parts = example_name.strip('/').split('/')
        print(f"  Example: '{example_name}'")
        print(f"    Components: {parts}")
        print(f"    Module path: {'/'.join(parts[:-1]) if len(parts) > 1 else 'root'}")
        print(f"    Operation: {example_op}")


if __name__ == "__main__":
    try:
        analyze_torch_node_properties()
        demonstrate_scope_extraction()
    except Exception as e:
        print(f"Analysis failed: {e}")
        import traceback
        traceback.print_exc()