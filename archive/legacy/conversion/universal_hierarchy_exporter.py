#!/usr/bin/env python3
"""
Universal hierarchy-preserving ONNX export for any PyTorch nn.Module
Works with any Hugging Face model by leveraging nn.Module hierarchy
Simple approach: hooks + module hierarchy + ONNX functions
"""

import inspect
import json

import onnx
import torch
from onnx import AttributeProto
from transformers import AutoModel


class UniversalHierarchyExporter:
    """Universal hierarchy-preserving ONNX exporter for any nn.Module"""
    
    def __init__(self):
        self.execution_trace = []
        self.module_hierarchy = {}
        self.parameter_mapping = {}
        
    def analyze_model_structure(self, model):
        """Analyze nn.Module hierarchy - universal for any model"""
        print(f"=== Analyzing {type(model).__name__} Module Hierarchy ===")
        
        # Extract complete hierarchy - works for any nn.Module
        for name, module in model.named_modules():
            if name:  # Skip root module
                self.module_hierarchy[name] = {
                    'type': type(module).__name__,
                    'depth': len(name.split('.')),
                    'parent': '.'.join(name.split('.')[:-1]) if '.' in name else 'root',
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
            if module_name != 'unknown':
                execution_order.append({
                    'module': module_name,
                    'type': type(module).__name__,
                    'order': len(execution_order),
                    'input_shapes': [list(t.shape) if hasattr(t, 'shape') else None for t in input],
                    'output_shape': list(output.shape) if hasattr(output, 'shape') else None
                })
        
        # Register hooks for ALL modules (not just leaves)
        hooks = []
        for name, module in model.named_modules():
            if name:  # Skip root
                module._hierarchy_name = name
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
            if name:
                for param_name, param in module.named_parameters(recurse=False):
                    onnx_param_name = f"{name}.{param_name}".replace('.', '_')
                    self.parameter_mapping[onnx_param_name] = {
                        'module': name,
                        'param_name': param_name,
                        'shape': list(param.shape),
                        'dtype': str(param.dtype)
                    }
        
        print(f"Mapped {len(self.parameter_mapping)} parameters")
        return self.parameter_mapping
    
    def export_with_hierarchy(self, model, dummy_inputs, output_path):
        """Export with hierarchy preservation using ONNX functions"""
        print("=== Exporting with Hierarchy Preservation ===")
        
        # Standard export first
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
        
        # Enhance with hierarchy
        onnx_model = onnx.load(output_path)
        self.enhance_with_hierarchy(onnx_model, output_path)
        
        return onnx_model
    
    def enhance_with_hierarchy(self, onnx_model, output_path):
        """Enhance ONNX model with hierarchy information"""
        print("=== Enhancing with Hierarchy ===")
        
        # Add hierarchy metadata
        hierarchy_meta = onnx_model.metadata_props.add()
        hierarchy_meta.key = "module_hierarchy"
        hierarchy_meta.value = json.dumps(self.module_hierarchy, indent=2)
        
        param_meta = onnx_model.metadata_props.add()
        param_meta.key = "parameter_mapping"
        param_meta.value = json.dumps(self.parameter_mapping, indent=2)
        
        trace_meta = onnx_model.metadata_props.add()
        trace_meta.key = "execution_trace"
        trace_meta.value = json.dumps(self.execution_trace, indent=2)
        
        # Map nodes to modules
        enhanced_nodes = 0
        for i, node in enumerate(onnx_model.graph.node):
            source_module = self.infer_node_module(node)
            
            if source_module:
                # Add module path attribute
                module_attr = AttributeProto()
                module_attr.name = "source_module"
                module_attr.type = AttributeProto.STRING
                module_attr.s = source_module.encode('utf-8')
                node.attribute.append(module_attr)
                
                # Add depth attribute
                if source_module in self.module_hierarchy:
                    depth_attr = AttributeProto()
                    depth_attr.name = "hierarchy_depth"
                    depth_attr.type = AttributeProto.INT
                    depth_attr.i = self.module_hierarchy[source_module]['depth']
                    node.attribute.append(depth_attr)
                
                node.name = f"{source_module}.{node.op_type}_{i}"
                enhanced_nodes += 1
        
        print(f"Enhanced {enhanced_nodes}/{len(onnx_model.graph.node)} nodes")
        
        enhanced_path = output_path.replace('.onnx', '_with_hierarchy.onnx')
        onnx.save(onnx_model, enhanced_path)
        print(f"Saved enhanced model: {enhanced_path}")
        
        return enhanced_path
    
    def infer_node_module(self, node):
        """Infer which module a node belongs to based on parameters"""
        # Map based on parameter names
        for input_name in node.input:
            for param_key, param_info in self.parameter_mapping.items():
                if param_key in input_name or input_name.endswith(param_key):
                    return param_info['module']
        return None

def create_dummy_inputs(model):
    """Create dummy inputs by inspecting model's forward signature"""
    sig = inspect.signature(model.forward)
    
    batch_size = 1
    seq_length = 32
    vocab_size = getattr(model.config, 'vocab_size', 30000)
    
    inputs = {}
    
    for param_name in sig.parameters:
        if param_name in ['self', 'args', 'kwargs']:
            continue
            
        # Create inputs based on parameter names
        if 'input_ids' in param_name:
            inputs[param_name] = torch.randint(0, min(vocab_size, 1000), (batch_size, seq_length))
        elif 'attention_mask' in param_name:
            inputs[param_name] = torch.ones(batch_size, seq_length)
        elif 'token_type_ids' in param_name:
            inputs[param_name] = torch.zeros(batch_size, seq_length, dtype=torch.long)
        elif 'pixel_values' in param_name:
            inputs[param_name] = torch.randn(batch_size, 3, 224, 224)
        elif 'decoder' in param_name and 'input_ids' in param_name:
            inputs[param_name] = torch.randint(0, min(vocab_size, 1000), (batch_size, seq_length))
    
    # Fallback
    if not inputs:
        inputs['input_ids'] = torch.randint(0, min(vocab_size, 1000), (batch_size, seq_length))
    
    return inputs

def main(model_name="google/bert_uncased_L-2_H-128_A-2"):
    """Main test function - works with any HF model"""
    print(f"=== Universal Hierarchy Export Test ===\\n")
    print(f"Loading model: {model_name}")
    
    try:
        model = AutoModel.from_pretrained(model_name)
        dummy_inputs = create_dummy_inputs(model)
        
        # Initialize universal exporter
        exporter = UniversalHierarchyExporter()
        
        # Step 1: Analyze structure
        hierarchy = exporter.analyze_model_structure(model)
        
        # Step 2: Trace execution
        trace = exporter.trace_execution_with_hooks(model, dummy_inputs)
        
        # Step 3: Create parameter mapping
        params = exporter.create_parameter_mapping(model)
        
        # Step 4: Export with hierarchy
        safe_name = model_name.replace('/', '_').replace('-', '_')
        output_path = f"{safe_name}_universal.onnx"
        exporter.export_with_hierarchy(model, dummy_inputs, output_path)
        
        print(f"\\n=== Success ===")
        print(f"Modules: {len(hierarchy)}")
        print(f"Traced: {len(trace)} executions")
        print(f"Parameters: {len(params)}")
        
        return exporter
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    main()