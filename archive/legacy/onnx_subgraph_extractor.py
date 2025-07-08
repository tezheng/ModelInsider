#!/usr/bin/env python3
"""
ONNX Subgraph Extractor
Extract subgraphs from ONNX models by hierarchy tags to create standalone ONNX models
"""

import onnx
from onnx import helper, TensorProto, ValueInfoProto
import json
import os
from pathlib import Path
from typing import Dict, List, Set, Optional, Tuple
from collections import defaultdict, deque
import argparse


class ONNXSubgraphExtractor:
    """Extract subgraphs from ONNX models by hierarchy tags"""
    
    def __init__(self, onnx_model_path: str):
        self.onnx_model_path = Path(onnx_model_path)
        self.onnx_model = onnx.load(str(onnx_model_path))
        self.original_model = onnx.load(str(onnx_model_path))  # Keep original copy
        
        # Build node lookup tables
        self.node_by_name = {}
        self.nodes_by_tag = defaultdict(list)
        self.tensor_producers = {}  # tensor_name -> node_name
        self.tensor_consumers = defaultdict(list)  # tensor_name -> list of node_names
        
        self._build_lookup_tables()
    
    def _build_lookup_tables(self):
        """Build lookup tables for fast access"""
        print("Building lookup tables...")
        
        # Index nodes by name and tags
        for node in self.onnx_model.graph.node:
            node_name = node.name if node.name else f"{node.op_type}_{len(self.node_by_name)}"
            self.node_by_name[node_name] = node
            
            # Extract tags from node attributes
            tags = self._extract_node_tags(node)
            for tag in tags:
                self.nodes_by_tag[tag].append(node_name)
            
            # Build tensor flow maps
            for output in node.output:
                self.tensor_producers[output] = node_name
            
            for input_tensor in node.input:
                self.tensor_consumers[input_tensor].append(node_name)
        
        # Index initializers
        for init in self.onnx_model.graph.initializer:
            self.tensor_producers[init.name] = init.name  # Initializer produces itself
        
        print(f"Indexed {len(self.node_by_name)} nodes")
        print(f"Found {len(self.nodes_by_tag)} unique tags")
    
    def _extract_node_tags(self, node) -> List[str]:
        """Extract hierarchy tags from a node"""
        tags = []
        
        for attr in node.attribute:
            if attr.name == "source_module":
                tags.append(attr.s.decode('utf-8'))
            elif attr.name == "hierarchy_tags":
                tags.extend([tag.decode('utf-8') for tag in attr.strings])
        
        return tags
    
    def list_available_modules(self) -> Dict[str, int]:
        """List all available modules and their operation counts"""
        module_counts = {}
        for tag, nodes in self.nodes_by_tag.items():
            module_counts[tag] = len(nodes)
        
        return dict(sorted(module_counts.items(), key=lambda x: x[1], reverse=True))
    
    def extract_subgraph(self, target_tag: str, output_path: Optional[str] = None, 
                        include_dependencies: bool = True) -> onnx.ModelProto:
        """
        Extract a subgraph containing all operations with the specified tag
        
        Args:
            target_tag: Hierarchy tag to extract (e.g., "/BertModel/BertEmbeddings")
            output_path: Where to save the extracted model (optional)
            include_dependencies: Whether to include dependency operations
            
        Returns:
            Extracted ONNX model
        """
        print(f"\n{'='*60}")
        print(f"EXTRACTING SUBGRAPH: {target_tag}")
        print(f"{'='*60}")
        
        if target_tag not in self.nodes_by_tag:
            raise ValueError(f"Tag '{target_tag}' not found in model. Available tags: {list(self.nodes_by_tag.keys())}")
        
        # Get all nodes with the target tag
        target_nodes = set(self.nodes_by_tag[target_tag])
        print(f"Found {len(target_nodes)} nodes with tag '{target_tag}'")
        
        # Find all required tensors and operations
        required_nodes = set(target_nodes)
        required_tensors = set()
        required_initializers = set()
        
        # Collect all tensors used by target nodes
        for node_name in target_nodes:
            node = self.node_by_name[node_name]
            
            # Add input tensors
            for input_tensor in node.input:
                required_tensors.add(input_tensor)
                
                # If it's an initializer, add it
                if self._is_initializer(input_tensor):
                    required_initializers.add(input_tensor)
            
            # Add output tensors
            for output_tensor in node.output:
                required_tensors.add(output_tensor)
        
        # If including dependencies, find all producer operations
        if include_dependencies:
            required_nodes = self._find_dependencies(target_nodes, required_tensors)
        
        print(f"Total nodes in subgraph: {len(required_nodes)}")
        print(f"Required tensors: {len(required_tensors)}")
        print(f"Required initializers: {len(required_initializers)}")
        
        # Create new graph
        extracted_graph = self._create_extracted_graph(
            required_nodes, required_tensors, required_initializers, target_tag
        )
        
        # Create new model
        extracted_model = helper.make_model(
            extracted_graph,
            producer_name=f"ONNXSubgraphExtractor",
            producer_version="1.0",
            doc_string=f"Extracted subgraph for module: {target_tag}"
        )
        
        # Copy metadata
        for prop in self.original_model.metadata_props:
            new_prop = extracted_model.metadata_props.add()
            new_prop.key = prop.key
            new_prop.value = prop.value
        
        # Add extraction metadata
        extraction_meta = extracted_model.metadata_props.add()
        extraction_meta.key = "extracted_module"
        extraction_meta.value = target_tag
        
        # Save if output path provided
        if output_path:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            onnx.save(extracted_model, str(output_path))
            print(f"Saved extracted model to: {output_path}")
        
        # Verify the extracted model
        try:
            onnx.checker.check_model(extracted_model)
            print("✅ Extracted model is valid")
        except Exception as e:
            print(f"⚠️  Model validation warning: {e}")
        
        return extracted_model
    
    def _find_dependencies(self, target_nodes: Set[str], required_tensors: Set[str]) -> Set[str]:
        """Find all nodes that need to be included for dependencies"""
        all_required_nodes = set(target_nodes)
        
        # BFS to find all producer operations
        queue = deque(required_tensors)
        visited_tensors = set()
        
        while queue:
            tensor_name = queue.popleft()
            if tensor_name in visited_tensors:
                continue
            
            visited_tensors.add(tensor_name)
            
            # Find producer of this tensor
            if tensor_name in self.tensor_producers:
                producer = self.tensor_producers[tensor_name]
                
                # If producer is a node (not initializer), add it
                if producer in self.node_by_name and producer not in all_required_nodes:
                    all_required_nodes.add(producer)
                    
                    # Add producer's input tensors to queue
                    producer_node = self.node_by_name[producer]
                    for input_tensor in producer_node.input:
                        if input_tensor not in visited_tensors:
                            queue.append(input_tensor)
        
        return all_required_nodes
    
    def _create_extracted_graph(self, required_nodes: Set[str], required_tensors: Set[str], 
                              required_initializers: Set[str], target_tag: str) -> onnx.GraphProto:
        """Create the extracted graph"""
        
        # Collect nodes
        extracted_nodes = []
        for node_name in required_nodes:
            if node_name in self.node_by_name:
                extracted_nodes.append(self.node_by_name[node_name])
        
        # Collect initializers
        extracted_initializers = []
        for init in self.original_model.graph.initializer:
            if init.name in required_initializers:
                extracted_initializers.append(init)
        
        # Determine graph inputs and outputs
        graph_inputs = []
        graph_outputs = []
        
        # Find external inputs (tensors not produced by any node in subgraph)
        for tensor_name in required_tensors:
            if tensor_name not in self.tensor_producers or self.tensor_producers[tensor_name] not in required_nodes:
                # This is an external input
                if not self._is_initializer(tensor_name):
                    # Find the tensor info from original graph
                    tensor_info = self._find_tensor_info(tensor_name)
                    if tensor_info:
                        graph_inputs.append(tensor_info)
        
        # Find outputs (tensors produced by subgraph but used outside it)
        for node_name in required_nodes:
            node = self.node_by_name[node_name]
            for output_tensor in node.output:
                # Check if this tensor is used by nodes outside our subgraph
                is_external_output = False
                for consumer in self.tensor_consumers.get(output_tensor, []):
                    if consumer not in required_nodes:
                        is_external_output = True
                        break
                
                # Also include final outputs from target tag nodes
                if not is_external_output and node_name in self.nodes_by_tag[target_tag]:
                    # This might be a final output of the module
                    if not self.tensor_consumers.get(output_tensor, []):
                        is_external_output = True
                
                if is_external_output:
                    tensor_info = self._find_tensor_info(output_tensor)
                    if tensor_info:
                        graph_outputs.append(tensor_info)
        
        # If no explicit outputs found, use outputs from target nodes
        if not graph_outputs:
            for node_name in self.nodes_by_tag[target_tag]:
                if node_name in required_nodes:
                    node = self.node_by_name[node_name]
                    for output_tensor in node.output:
                        tensor_info = self._find_tensor_info(output_tensor)
                        if tensor_info:
                            graph_outputs.append(tensor_info)
        
        # Create graph
        graph_name = f"extracted_{target_tag.replace('/', '_')}"
        extracted_graph = helper.make_graph(
            extracted_nodes,
            graph_name,
            graph_inputs,
            graph_outputs,
            extracted_initializers
        )
        
        print(f"Graph inputs: {len(graph_inputs)}")
        print(f"Graph outputs: {len(graph_outputs)}")
        print(f"Graph nodes: {len(extracted_nodes)}")
        print(f"Graph initializers: {len(extracted_initializers)}")
        
        return extracted_graph
    
    def _is_initializer(self, tensor_name: str) -> bool:
        """Check if tensor is an initializer"""
        for init in self.original_model.graph.initializer:
            if init.name == tensor_name:
                return True
        return False
    
    def _find_tensor_info(self, tensor_name: str) -> Optional[ValueInfoProto]:
        """Find tensor info from original graph"""
        # Check inputs
        for input_info in self.original_model.graph.input:
            if input_info.name == tensor_name:
                return input_info
        
        # Check outputs
        for output_info in self.original_model.graph.output:
            if output_info.name == tensor_name:
                return output_info
        
        # Check value_info
        for value_info in self.original_model.graph.value_info:
            if value_info.name == tensor_name:
                return value_info
        
        # Create a generic tensor info
        return helper.make_tensor_value_info(
            tensor_name,
            TensorProto.FLOAT,
            []  # Unknown shape
        )
    
    def extract_multiple_modules(self, module_tags: List[str], output_dir: str):
        """Extract multiple modules and save them"""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        results = {}
        
        for tag in module_tags:
            try:
                safe_name = tag.replace('/', '_').replace('.', '_')
                output_path = output_dir / f"{safe_name}.onnx"
                
                extracted_model = self.extract_subgraph(tag, str(output_path))
                
                results[tag] = {
                    'status': 'SUCCESS',
                    'output_path': str(output_path),
                    'nodes': len(extracted_model.graph.node),
                    'inputs': len(extracted_model.graph.input),
                    'outputs': len(extracted_model.graph.output)
                }
                
            except Exception as e:
                results[tag] = {
                    'status': 'FAILED',
                    'error': str(e)
                }
        
        # Save summary
        summary_path = output_dir / "extraction_summary.json"
        with open(summary_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"\n{'='*60}")
        print("EXTRACTION SUMMARY")
        print(f"{'='*60}")
        
        successful = sum(1 for r in results.values() if r['status'] == 'SUCCESS')
        print(f"Successful extractions: {successful}/{len(module_tags)}")
        
        for tag, result in results.items():
            if result['status'] == 'SUCCESS':
                print(f"✅ {tag}: {result['nodes']} nodes -> {result['output_path']}")
            else:
                print(f"❌ {tag}: {result['error']}")
        
        return results


def main():
    """Command line interface for subgraph extraction"""
    parser = argparse.ArgumentParser(description="Extract subgraphs from ONNX models by hierarchy tags")
    parser.add_argument('model_path', help='Path to ONNX model with hierarchy tags')
    parser.add_argument('--list-modules', action='store_true', help='List available modules')
    parser.add_argument('--extract', type=str, help='Module tag to extract')
    parser.add_argument('--output', type=str, help='Output path for extracted model')
    parser.add_argument('--extract-all', action='store_true', help='Extract all modules')
    parser.add_argument('--output-dir', type=str, default='extracted_modules', help='Output directory for batch extraction')
    
    args = parser.parse_args()
    
    # Initialize extractor
    print(f"Loading ONNX model: {args.model_path}")
    extractor = ONNXSubgraphExtractor(args.model_path)
    
    if args.list_modules:
        print("\n📋 AVAILABLE MODULES:")
        modules = extractor.list_available_modules()
        for module, count in modules.items():
            print(f"   {module:50} ({count} operations)")
    
    elif args.extract:
        output_path = args.output or f"extracted_{args.extract.replace('/', '_')}.onnx"
        extractor.extract_subgraph(args.extract, output_path)
    
    elif args.extract_all:
        modules = list(extractor.list_available_modules().keys())
        print(f"Extracting {len(modules)} modules...")
        extractor.extract_multiple_modules(modules, args.output_dir)
    
    else:
        print("Please specify --list-modules, --extract <module>, or --extract-all")


if __name__ == "__main__":
    main()