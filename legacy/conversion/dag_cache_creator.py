#!/usr/bin/env python3
"""
Create DAG-enhanced static cache with connection information
"""

import onnx
import json

def extract_dag_connections(onnx_file, module_name):
    """Extract DAG connections (input/output relationships) from ONNX model"""
    model = onnx.load(onnx_file)
    
    # Build DAG structure
    dag_info = {
        'nodes': [],
        'input_tensors': [inp.name for inp in model.graph.input],
        'output_tensors': [out.name for out in model.graph.output],
        'tensor_to_producer': {},  # tensor_name -> node_index that produces it
        'tensor_to_consumers': {}  # tensor_name -> [node_indices] that consume it
    }
    
    # Process each node to build the DAG
    core_node_count = 0
    for i, node in enumerate(model.graph.node):
        if is_core_model_operation(node, module_name):
            node_info = {
                'index': core_node_count,
                'original_index': i,
                'name': node.name,
                'op_type': node.op_type,
                'inputs': list(node.input),
                'outputs': list(node.output),
                'predecessor_nodes': [],  # nodes that feed into this node
                'successor_nodes': []     # nodes that this node feeds into
            }
            
            dag_info['nodes'].append(node_info)
            
            # Map outputs to this node (this node produces these tensors)
            for output_tensor in node.output:
                dag_info['tensor_to_producer'][output_tensor] = core_node_count
            
            # Track consumers for inputs (will be filled in second pass)
            for input_tensor in node.input:
                if input_tensor not in dag_info['tensor_to_consumers']:
                    dag_info['tensor_to_consumers'][input_tensor] = []
                dag_info['tensor_to_consumers'][input_tensor].append(core_node_count)
            
            core_node_count += 1
    
    # Second pass: build explicit predecessor/successor relationships
    for node in dag_info['nodes']:
        node_idx = node['index']
        
        # Find predecessors (nodes that produce our inputs)
        for input_tensor in node['inputs']:
            if input_tensor in dag_info['tensor_to_producer']:
                producer_idx = dag_info['tensor_to_producer'][input_tensor]
                if producer_idx != node_idx:  # Don't connect to self
                    node['predecessor_nodes'].append({
                        'node_index': producer_idx,
                        'tensor': input_tensor,
                        'connection_type': 'data_dependency'
                    })
        
        # Find successors (nodes that consume our outputs)
        for output_tensor in node['outputs']:
            if output_tensor in dag_info['tensor_to_consumers']:
                consumers = dag_info['tensor_to_consumers'][output_tensor]
                for consumer_idx in consumers:
                    if consumer_idx != node_idx:  # Don't connect to self
                        node['successor_nodes'].append({
                            'node_index': consumer_idx,
                            'tensor': output_tensor,
                            'connection_type': 'data_dependency'
                        })
    
    # Calculate DAG statistics
    total_edges = 0
    for node in dag_info['nodes']:
        total_edges += len(node['predecessor_nodes'])
    
    dag_info['summary'] = {
        'total_nodes': len(dag_info['nodes']),
        'total_edges': total_edges,
        'input_count': len(dag_info['input_tensors']),
        'output_count': len(dag_info['output_tensors']),
        'max_fan_in': max(len(node['predecessor_nodes']) for node in dag_info['nodes']) if dag_info['nodes'] else 0,
        'max_fan_out': max(len(node['successor_nodes']) for node in dag_info['nodes']) if dag_info['nodes'] else 0
    }
    
    return dag_info

def is_core_model_operation(node, module_name):
    """Filter for core operations"""
    core_ml_ops = {
        'MatMul', 'Gemm', 'Conv', 'Add', 'Mul', 'Sub', 'Div',
        'Softmax', 'Tanh', 'Relu', 'Gelu', 'Sigmoid', 'Erf',
        'ReduceMean', 'ReduceSum', 'Sqrt', 'Pow',
        'LayerNormalization', 'BatchNormalization',
        'Gather', 'Slice', 'Transpose', 'Reshape',
        'Cast', 'Where', 'Equal', 'Greater', 'Less'
    }
    
    return node.op_type in core_ml_ops

def create_full_dag_cache():
    """Create complete DAG cache for all pieces"""
    
    pieces = {
        'embeddings': 'bert_component_embeddings.onnx',
        'encoder.layer.0': 'bert_component_encoder_layer_0.onnx',
        'encoder.layer.0.attention': 'bert_component_encoder_layer_0_attention.onnx',
        'encoder.layer.0.attention.self': 'bert_component_encoder_layer_0_attention_self.onnx',
        'encoder.layer.0.intermediate': 'bert_component_encoder_layer_0_intermediate.onnx',
        'encoder.layer.1': 'bert_component_encoder_layer_1.onnx',
        'encoder.layer.1.attention': 'bert_component_encoder_layer_1_attention.onnx',
        'encoder.layer.1.attention.self': 'bert_component_encoder_layer_1_attention_self.onnx',
        'encoder.layer.1.intermediate': 'bert_component_encoder_layer_1_intermediate.onnx',
        'pooler': 'bert_component_pooler.onnx'
    }
    
    print('=== CREATING COMPLETE DAG CACHE ===')
    print()
    
    dag_cache = {
        'model_info': {
            'model_name': 'google/bert_uncased_L-2_H-128_A-2',
            'model_type': 'BertModel',
            'extraction_method': 'piece_by_piece_with_dag_connections',
            'description': 'Core ONNX operations with complete DAG connection information'
        },
        'pieces': {},
        'global_summary': {}
    }
    
    total_nodes = 0
    total_edges = 0
    
    for module_name, file_path in pieces.items():
        try:
            print(f'Processing {module_name}...')
            dag_info = extract_dag_connections(file_path, module_name)
            
            piece_info = {
                'module_name': module_name,
                'file_source': file_path,
                'dag_structure': dag_info
            }
            
            dag_cache['pieces'][module_name] = piece_info
            total_nodes += dag_info['summary']['total_nodes']
            total_edges += dag_info['summary']['total_edges']
            
            print(f'  Nodes: {dag_info["summary"]["total_nodes"]}')
            print(f'  Edges: {dag_info["summary"]["total_edges"]}')
            print(f'  Max fan-in/out: {dag_info["summary"]["max_fan_in"]}/{dag_info["summary"]["max_fan_out"]}')
            
        except Exception as e:
            print(f'Error processing {file_path}: {e}')
    
    # Global summary
    dag_cache['global_summary'] = {
        'total_pieces': len(dag_cache['pieces']),
        'total_nodes': total_nodes,
        'total_edges': total_edges,
        'avg_nodes_per_piece': total_nodes / len(dag_cache['pieces']) if dag_cache['pieces'] else 0,
        'avg_edges_per_piece': total_edges / len(dag_cache['pieces']) if dag_cache['pieces'] else 0
    }
    
    return dag_cache

def demonstrate_dag_structure():
    """Demonstrate DAG structure on a simple piece"""
    print('=== DAG STRUCTURE DEMONSTRATION ===')
    print()
    
    # Use pooler as simple example
    print('Analyzing pooler DAG structure...')
    dag_info = extract_dag_connections('bert_component_pooler.onnx', 'pooler')
    
    print(f'Pooler DAG Summary:')
    print(f'  Nodes: {dag_info["summary"]["total_nodes"]}')
    print(f'  Edges: {dag_info["summary"]["total_edges"]}')
    print(f'  Inputs: {dag_info["input_tensors"]}')
    print(f'  Outputs: {dag_info["output_tensors"]}')
    print()
    
    print('Node connections:')
    for i, node in enumerate(dag_info['nodes']):
        print(f'  Node {i}: {node["op_type"]} ({node["name"]})')
        print(f'    Inputs: {node["inputs"]}')
        print(f'    Outputs: {node["outputs"]}')
        print(f'    Predecessors: {[p["node_index"] for p in node["predecessor_nodes"]]}')
        print(f'    Successors: {[s["node_index"] for s in node["successor_nodes"]]}')
        print()
    
    # Show tensor flow
    print('Tensor dependencies:')
    for node in dag_info['nodes']:
        for pred in node['predecessor_nodes']:
            print(f'  Node {pred["node_index"]} -> Node {node["index"]} via {pred["tensor"]}')
    
    return dag_info

if __name__ == "__main__":
    # First demonstrate on simple piece
    demonstrate_dag_structure()
    
    print('=' * 60)
    
    # Create full cache
    cache = create_full_dag_cache()
    
    # Save to file
    with open('bert_dag_operations_cache.json', 'w') as f:
        json.dump(cache, f, indent=2)
    
    print()
    print('=== FINAL DAG CACHE SUMMARY ===')
    print(f'Total pieces: {cache["global_summary"]["total_pieces"]}')
    print(f'Total nodes: {cache["global_summary"]["total_nodes"]}')
    print(f'Total edges: {cache["global_summary"]["total_edges"]}')
    print(f'Avg nodes per piece: {cache["global_summary"]["avg_nodes_per_piece"]:.1f}')
    print(f'Avg edges per piece: {cache["global_summary"]["avg_edges_per_piece"]:.1f}')
    print()
    print('✅ DAG cache saved to: bert_dag_operations_cache.json')
    print('✅ Contains: Operations + DAG connections + tensor dependencies')
    print('✅ Preserves: Complete data flow structure within each piece')