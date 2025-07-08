#!/usr/bin/env python3
"""
DAG Extractor for BERT Hierarchy
Extract DAG for each nn.Module with detailed operation metadata
"""

import torch
import torch.nn as nn
import onnx
from onnx import helper, TensorProto
import json
from typing import Dict, List, Any, Set
from transformers import AutoModel
import numpy as np
from pathlib import Path
import inspect
from collections import defaultdict

class DAGExtractor:
    """Extract DAG for each nn.Module in hierarchy"""
    
    def __init__(self):
        self.module_hierarchy = {}
        self.execution_trace = []
        self.parameter_mapping = {}
        self.operation_metadata = {}
        self.module_operations = defaultdict(list)  # module -> list of operations
        
    def analyze_model_structure(self, model):
        """Analyze nn.Module hierarchy"""
        print(f"=== Analyzing {type(model).__name__} Module Hierarchy ===")
        
        # Get root class name for hierarchy paths
        root_class = type(model).__name__
        
        # Extract complete hierarchy with class names
        for name, module in model.named_modules():
            hierarchy_path = f"/{root_class}"
            if name:  # Not root module
                # Convert module path to hierarchy path with class names
                parts = name.split('.')
                current_module = model
                path_parts = [root_class]
                
                for part in parts:
                    if hasattr(current_module, part):
                        current_module = getattr(current_module, part)
                        class_name = type(current_module).__name__
                        
                        # Handle numbered modules (like layer.0, layer.1)
                        if part.isdigit():
                            path_parts[-1] = f"{path_parts[-1]}.{part}"
                        else:
                            path_parts.append(class_name)
                
                hierarchy_path = "/" + "/".join(path_parts)
            
            self.module_hierarchy[name if name else "root"] = {
                'hierarchy_path': hierarchy_path,
                'type': type(module).__name__,
                'depth': len(name.split('.')) if name else 0,
                'is_leaf': len(list(module.children())) == 0,
                'parameter_count': sum(p.numel() for p in module.parameters(recurse=False)),
                'children': [child_name for child_name, _ in module.named_children()]
            }
        
        print(f"Total modules: {len(self.module_hierarchy)}")
        return self.module_hierarchy
    
    def trace_execution_with_hooks(self, model, dummy_inputs):
        """Trace model execution to map operations to modules"""
        print("=== Tracing Execution with Hooks ===")
        
        execution_order = []
        
        def forward_hook(module, input, output):
            module_name = getattr(module, '_hierarchy_name', 'unknown')
            if module_name != 'unknown' and module_name in self.module_hierarchy:
                execution_order.append({
                    'module': module_name,
                    'hierarchy_path': self.module_hierarchy[module_name]['hierarchy_path'],
                    'type': type(module).__name__,
                    'order': len(execution_order)
                })
        
        # Register hooks for ALL modules
        hooks = []
        for name, module in model.named_modules():
            module._hierarchy_name = name if name else "root"
            hooks.append(module.register_forward_hook(forward_hook))
        
        # Run forward pass
        model.eval()
        with torch.no_grad():
            try:
                if isinstance(dummy_inputs, dict):
                    _ = model(**dummy_inputs)
                else:
                    _ = model(dummy_inputs)
            except Exception as e:
                print(f"Forward pass failed: {e}")
        
        # Clean up hooks
        for hook in hooks:
            hook.remove()
        
        self.execution_trace = execution_order
        print(f"Traced {len(execution_order)} module executions")
        return execution_order
    
    def create_parameter_mapping(self, model):
        """Create mapping from parameters to modules"""
        print("=== Creating Parameter Mapping ===")
        
        for name, module in model.named_modules():
            module_name = name if name else "root"
            if module_name in self.module_hierarchy:
                hierarchy_path = self.module_hierarchy[module_name]['hierarchy_path']
                
                for param_name, param in module.named_parameters(recurse=False):
                    onnx_param_name = f"{name}.{param_name}".replace('.', '_') if name else param_name
                    self.parameter_mapping[onnx_param_name] = {
                        'module': module_name,
                        'hierarchy_path': hierarchy_path,
                        'param_name': param_name,
                        'shape': list(param.shape),
                        'dtype': str(param.dtype)
                    }
        
        print(f"Mapped {len(self.parameter_mapping)} parameters")
        return self.parameter_mapping
    
    def export_and_analyze_onnx(self, model, dummy_inputs, output_path):
        """Export to ONNX and analyze operations"""
        print("=== Exporting to ONNX ===")
        
        # Export to ONNX
        if isinstance(dummy_inputs, dict):
            input_names = list(dummy_inputs.keys())
            dummy_input_tuple = tuple(dummy_inputs.values())
        else:
            input_names = ['input']
            dummy_input_tuple = (dummy_inputs,)
        
        torch.onnx.export(
            model,
            dummy_input_tuple,
            output_path,
            export_params=True,
            opset_version=14,
            do_constant_folding=False,
            input_names=input_names,
            output_names=['output'],
            verbose=False
        )
        
        # Load and analyze ONNX model
        onnx_model = onnx.load(output_path)
        self.analyze_onnx_operations(onnx_model)
        
        # CRITICAL: Inject tags into ONNX model
        enhanced_path = self.inject_tags_into_onnx(onnx_model, output_path)
        
        return onnx_model
    
    def analyze_onnx_operations(self, onnx_model):
        """Analyze ONNX operations and create metadata"""
        print("=== Analyzing ONNX Operations ===")
        
        # Get model inputs (to exclude them)
        model_inputs = set()
        for input_info in onnx_model.graph.input:
            model_inputs.add(input_info.name)
        
        # Get model outputs (to exclude them)
        model_outputs = set()
        for output_info in onnx_model.graph.output:
            model_outputs.add(output_info.name)
        
        # Process initializers (parameters)
        for init in onnx_model.graph.initializer:
            tags = self.get_parameter_tags(init.name)
            if tags:  # Only include if used by modules
                self.operation_metadata[init.name] = {
                    'op_type': 'Initializer',
                    'inputs': [],
                    'outputs': [init.name],
                    'tags': tags
                }
                # Add to module operations
                for tag in tags:
                    self.module_operations[tag].append(init.name)
        
        # Process nodes (operations)
        for node in onnx_model.graph.node:
            # Skip input/output operations
            if any(inp in model_inputs for inp in node.input) and len(node.input) == 1:
                continue  # Skip direct input operations
            if any(out in model_outputs for out in node.output) and len(node.output) == 1:
                continue  # Skip direct output operations
            
            # Get tags for this operation
            tags = self.get_operation_tags(node)
            
            if tags:  # Only include operations that belong to modules
                op_name = node.name if node.name else f"{node.op_type}_{len(self.operation_metadata)}"
                
                self.operation_metadata[op_name] = {
                    'op_type': node.op_type,
                    'inputs': list(node.input),
                    'outputs': list(node.output),
                    'tags': tags
                }
                
                # Add to module operations
                for tag in tags:
                    self.module_operations[tag].append(op_name)
        
        print(f"Analyzed {len(self.operation_metadata)} operations")
        print(f"Found operations for {len(self.module_operations)} modules")
    
    def inject_tags_into_onnx(self, onnx_model, original_path):
        """CRITICAL: Inject hierarchy tags as ONNX node attributes"""
        print("=== Injecting Tags into ONNX Model ===")
        
        nodes_tagged = 0
        nodes_without_tags = 0
        
        # Add metadata to model
        hierarchy_meta = onnx_model.metadata_props.add()
        hierarchy_meta.key = "module_hierarchy"
        hierarchy_meta.value = json.dumps(self.module_hierarchy, indent=2)
        
        param_meta = onnx_model.metadata_props.add()
        param_meta.key = "operation_metadata"
        param_meta.value = json.dumps(self.operation_metadata, indent=2)
        
        # Inject tags as node attributes
        for node in onnx_model.graph.node:
            node_name = node.name if node.name else None
            
            # Try to find this node in our operation metadata
            if node_name and node_name in self.operation_metadata:
                tags = self.operation_metadata[node_name].get('tags', [])
                if tags:
                    # Add primary source module
                    module_attr = onnx.AttributeProto()
                    module_attr.name = "source_module"
                    module_attr.type = onnx.AttributeProto.STRING
                    module_attr.s = tags[0].encode('utf-8')
                    node.attribute.append(module_attr)
                    
                    # Add all tags if multiple
                    if len(tags) > 1:
                        tags_attr = onnx.AttributeProto()
                        tags_attr.name = "hierarchy_tags"
                        tags_attr.type = onnx.AttributeProto.STRINGS
                        tags_attr.strings.extend([tag.encode('utf-8') for tag in tags])
                        node.attribute.append(tags_attr)
                    
                    nodes_tagged += 1
                else:
                    nodes_without_tags += 1
            else:
                # Try to find by matching operation characteristics
                found_match = False
                for op_name, op_data in self.operation_metadata.items():
                    # Match by op_type and inputs
                    if (op_data.get('op_type') == node.op_type and
                        len(op_data.get('inputs', [])) == len(node.input)):
                        
                        # Check if inputs match (at least some of them)
                        input_match = False
                        node_inputs = set(node.input)
                        op_inputs = set(op_data.get('inputs', []))
                        if node_inputs.intersection(op_inputs):
                            input_match = True
                        
                        if input_match:
                            tags = op_data.get('tags', [])
                            if tags:
                                # Add primary source module
                                module_attr = onnx.AttributeProto()
                                module_attr.name = "source_module"
                                module_attr.type = onnx.AttributeProto.STRING
                                module_attr.s = tags[0].encode('utf-8')
                                node.attribute.append(module_attr)
                                
                                # Add all tags if multiple
                                if len(tags) > 1:
                                    tags_attr = onnx.AttributeProto()
                                    tags_attr.name = "hierarchy_tags"
                                    tags_attr.type = onnx.AttributeProto.STRINGS
                                    tags_attr.strings.extend([tag.encode('utf-8') for tag in tags])
                                    node.attribute.append(tags_attr)
                                
                                nodes_tagged += 1
                                found_match = True
                                break
                
                if not found_match:
                    nodes_without_tags += 1
        
        print(f"Tagged {nodes_tagged} nodes")
        print(f"Nodes without tags: {nodes_without_tags}")
        print(f"Total nodes: {len(onnx_model.graph.node)}")
        
        # Save enhanced model
        enhanced_path = original_path.replace('.onnx', '_with_tags.onnx')
        onnx.save(onnx_model, enhanced_path)
        print(f"Saved enhanced model with tags: {enhanced_path}")
        
        return enhanced_path
    
    def get_parameter_tags(self, param_name):
        """Get tags for a parameter based on which modules use it"""
        tags = []
        
        # Convert ONNX parameter name to PyTorch parameter name
        # ONNX: "encoder.layer.0.attention.self.query.weight"
        # PyTorch mapping: "encoder_layer_0_attention_self_query_weight"
        pytorch_param_name = param_name.replace('.', '_')
        
        # Direct parameter mapping
        if pytorch_param_name in self.parameter_mapping:
            tags.append(self.parameter_mapping[pytorch_param_name]['hierarchy_path'])
        
        # Also try direct name
        if param_name in self.parameter_mapping:
            tags.append(self.parameter_mapping[param_name]['hierarchy_path'])
        
        # Check all parameter mappings to find matches
        for mapped_param, info in self.parameter_mapping.items():
            # Try both directions of name conversion
            if (mapped_param == pytorch_param_name or 
                mapped_param.replace('_', '.') == param_name or
                mapped_param == param_name):
                if info['hierarchy_path'] not in tags:
                    tags.append(info['hierarchy_path'])
        
        return tags
    
    def get_operation_tags(self, node):
        """Get tags for an operation based on parameters it uses and execution context"""
        tags = []
        
        # Parameter-based tagging
        for input_name in node.input:
            param_tags = self.get_parameter_tags(input_name)
            for tag in param_tags:
                if tag not in tags:
                    tags.append(tag)
        
        # Execution-based tagging (simplified - assign to deepest module that uses parameters)
        if not tags and node.input:
            # Try to infer from parameter names
            for input_name in node.input:
                # Look for parameter mapping by checking if input matches any parameter
                for param_key, param_info in self.parameter_mapping.items():
                    if input_name.replace('.', '_') == param_key or param_key in input_name:
                        if param_info['hierarchy_path'] not in tags:
                            tags.append(param_info['hierarchy_path'])
        
        return tags
    
    def extract_module_dag(self, hierarchy_path):
        """Extract DAG for a specific module"""
        if hierarchy_path not in self.module_operations:
            return {"nodes": [], "edges": []}
        
        operations = self.module_operations[hierarchy_path]
        
        # Build edges by analyzing operation dependencies
        edges = []
        
        # Create a map of all outputs to their producing operations (including all operations, not just this module)
        output_to_op = {}
        for op_name, op_data in self.operation_metadata.items():
            for output in op_data.get('outputs', []):
                output_to_op[output] = op_name
        
        # Now find dependencies for operations in this module
        for op_name in operations:
            if op_name in self.operation_metadata:
                op_data = self.operation_metadata[op_name]
                
                # Find dependencies (inputs that are outputs of other operations)
                for input_name in op_data.get('inputs', []):
                    if input_name in output_to_op:
                        producer_op = output_to_op[input_name]
                        # Only add edge if producer is also in this module
                        if producer_op in operations and producer_op != op_name:
                            edge = [producer_op, op_name]
                            if edge not in edges:
                                edges.append(edge)
        
        return {
            "nodes": operations,
            "edges": edges
        }
    
    def generate_all_module_dags(self):
        """Generate DAGs for all modules"""
        all_dags = {}
        
        for hierarchy_path in self.module_operations:
            dag = self.extract_module_dag(hierarchy_path)
            if dag["nodes"]:  # Only include modules with operations
                all_dags[hierarchy_path] = dag
        
        return all_dags


def main():
    """Main function to test DAG extraction with BERT tiny"""
    print("=== DAG Extraction Test ===\\n")
    
    # Load BERT tiny model
    model_name = "google/bert_uncased_L-2_H-128_A-2"
    print(f"Loading model: {model_name}")
    
    try:
        from input_generator import UniversalInputGenerator
        
        model = AutoModel.from_pretrained(model_name)
        input_generator = UniversalInputGenerator()
        dummy_inputs = input_generator.generate_inputs(model, model_name)
        
        # Initialize DAG extractor
        extractor = DAGExtractor()
        
        # Step 1: Analyze model structure
        hierarchy = extractor.analyze_model_structure(model)
        
        # Step 2: Trace execution
        trace = extractor.trace_execution_with_hooks(model, dummy_inputs)
        
        # Step 3: Create parameter mapping
        params = extractor.create_parameter_mapping(model)
        
        # Step 4: Export and analyze ONNX
        os.makedirs("temp/onnx_models", exist_ok=True)
        output_path = "temp/onnx_models/bert_tiny_dag.onnx"
        onnx_model = extractor.export_and_analyze_onnx(model, dummy_inputs, output_path)
        
        # Step 5: Generate DAGs for all modules
        all_dags = extractor.generate_all_module_dags()
        
        # Step 6: Save results to temp directory
        import os
        os.makedirs("temp/test_outputs", exist_ok=True)
        
        # Save operation metadata
        with open("temp/test_outputs/bert_operation_metadata.json", "w") as f:
            json.dump(extractor.operation_metadata, f, indent=2)
        
        # Save module DAGs
        with open("temp/test_outputs/bert_module_dags.json", "w") as f:
            json.dump(all_dags, f, indent=2)
        
        # Step 7: Show specific module data
        target_module = "/BertModel/BertEncoder/BertLayer.0/BertAttention/BertSelfAttention"
        if target_module in all_dags:
            print(f"\\n=== {target_module} DAG ===")
            print(json.dumps(all_dags[target_module], indent=2))
            
            print(f"\\n=== {target_module} Operations Metadata ===")
            module_ops = all_dags[target_module]["nodes"]
            module_metadata = {}
            for op_name in module_ops:
                if op_name in extractor.operation_metadata:
                    module_metadata[op_name] = extractor.operation_metadata[op_name]
            
            print(json.dumps(module_metadata, indent=2))
        else:
            print(f"\\nModule {target_module} not found in DAGs")
            print("Available modules:")
            for module_path in sorted(all_dags.keys()):
                print(f"  {module_path}")
        
        print(f"\\n=== Summary ===")
        print(f"Total modules: {len(hierarchy)}")
        print(f"Execution trace: {len(trace)} executions")
        print(f"Parameters: {len(params)}")
        print(f"Operations: {len(extractor.operation_metadata)}")
        print(f"Modules with operations: {len(all_dags)}")
        
        return extractor
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    main()