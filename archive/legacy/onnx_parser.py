#!/usr/bin/env python3
"""
ONNX Model Parser - Parse and show tags of each node
"""

import json
from collections import defaultdict

import onnx


def parse_onnx_model(onnx_path):
    """Parse ONNX model and show tags of each node"""
    print(f"=== Parsing ONNX Model: {onnx_path} ===")
    
    # Load ONNX model
    model = onnx.load(onnx_path)
    
    print(f"Model IR version: {model.ir_version}")
    print(f"Producer: {model.producer_name} {model.producer_version}")
    print(f"Graph name: {model.graph.name}")
    print(f"Nodes: {len(model.graph.node)}")
    print(f"Initializers: {len(model.graph.initializer)}")
    print(f"Inputs: {len(model.graph.input)}")
    print(f"Outputs: {len(model.graph.output)}")
    
    print(f"\n=== Model Metadata ===")
    for prop in model.metadata_props:
        print(f"{prop.key}: {len(prop.value)} characters")
        if prop.key in ['module_hierarchy', 'parameter_mapping', 'execution_trace']:
            try:
                data = json.loads(prop.value)
                if isinstance(data, dict | list):
                    print(f"  -> {len(data)} items")
            except:
                print(f"  -> (not JSON)")
    
    print(f"\n=== Initializers (Parameters) ===")
    for i, init in enumerate(model.graph.initializer):
        print(f"{i+1:2d}. {init.name}")
    
    print(f"\n=== Graph Inputs ===")
    for i, input_info in enumerate(model.graph.input):
        print(f"{i+1:2d}. {input_info.name}")
    
    print(f"\n=== Graph Outputs ===")
    for i, output_info in enumerate(model.graph.output):
        print(f"{i+1:2d}. {output_info.name}")
    
    print(f"\n=== Nodes with Tags ===")
    nodes_with_tags = 0
    nodes_without_tags = 0
    tag_distribution = defaultdict(int)
    
    for i, node in enumerate(model.graph.node):
        # Look for tag attributes
        tags = []
        source_module = None
        hierarchy_depth = None
        
        for attr in node.attribute:
            if attr.name == "source_module":
                source_module = attr.s.decode('utf-8') if attr.s else None
            elif attr.name == "hierarchy_depth":
                hierarchy_depth = attr.i
            elif attr.name == "tags":
                # If we stored tags as an attribute
                if attr.type == onnx.AttributeProto.STRINGS:
                    tags = [s.decode('utf-8') for s in attr.strings]
                elif attr.type == onnx.AttributeProto.STRING:
                    tags = [attr.s.decode('utf-8')]
        
        # Show node info
        node_name = node.name if node.name else f"node_{i}"
        print(f"{i+1:3d}. {node_name}")
        print(f"     Op: {node.op_type}")
        print(f"     Inputs: {list(node.input)}")
        print(f"     Outputs: {list(node.output)}")
        
        if source_module:
            print(f"     Source Module: {source_module}")
            tags.append(source_module)
            nodes_with_tags += 1
            tag_distribution[source_module] += 1
        
        if hierarchy_depth is not None:
            print(f"     Hierarchy Depth: {hierarchy_depth}")
        
        if tags:
            print(f"     Tags: {tags}")
        else:
            print(f"     Tags: None")
            nodes_without_tags += 1
        
        print()
    
    print(f"=== Tag Summary ===")
    print(f"Nodes with tags: {nodes_with_tags}")
    print(f"Nodes without tags: {nodes_without_tags}")
    print(f"Total nodes: {len(model.graph.node)}")
    
    if tag_distribution:
        print(f"\n=== Tag Distribution ===")
        for tag, count in sorted(tag_distribution.items()):
            print(f"{count:3d} nodes: {tag}")
    
    return model

def main():
    """Main function to parse BERT ONNX model"""
    onnx_path = "bert_tiny_dag.onnx"
    
    try:
        model = parse_onnx_model(onnx_path)
        print(f"\n✅ Successfully parsed {onnx_path}")
        return model
    except Exception as e:
        print(f"❌ Error parsing {onnx_path}: {e}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    main()