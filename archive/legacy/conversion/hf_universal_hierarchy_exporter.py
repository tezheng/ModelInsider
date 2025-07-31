#!/usr/bin/env python3
"""
Universal hierarchy-preserving ONNX export for Hugging Face models
Supports BERT, GPT, T5, and other transformer architectures
Combining tracing hooks, ONNX functions, and comprehensive testing
"""

import json

import onnx
import torch
import torch.nn as nn
from onnx import AttributeProto
from transformers import AutoModel


class UniversalHFHierarchyExporter:
    """Universal hierarchy-preserving ONNX exporter for Hugging Face models"""
    
    def __init__(self):
        self.execution_trace = []
        self.module_hierarchy = {}
        self.parameter_mapping = {}
        self.function_registry = {}
        
    def analyze_model_structure(self, model):
        """Analyze HF model structure and identify function candidates"""
        print(f"=== Analyzing {type(model).__name__} Model Structure ===")
        
        # Extract complete hierarchy
        for name, module in model.named_modules():
            if name:
                self.module_hierarchy[name] = {
                    'type': type(module).__name__,
                    'depth': len(name.split('.')),
                    'parent': '.'.join(name.split('.')[:-1]) if '.' in name else 'root',
                    'is_leaf': len(list(module.children())) == 0,
                    'parameter_count': sum(p.numel() for p in module.parameters(recurse=False)),
                    'children': [child_name for child_name, _ in module.named_children()]
                }
        
        print(f"Total modules: {len(self.module_hierarchy)}")
        
        # Identify function candidates (major blocks)
        function_candidates = self.identify_function_candidates()
        print(f"Function candidates: {len(function_candidates)}")
        
        return function_candidates
    
    def identify_function_candidates(self):
        """Identify modules that should become ONNX functions for any HF model"""
        candidates = {}
        
        for name, info in self.module_hierarchy.items():
            # Universal blocks: embeddings, layers, attention, feed-forward
            universal_keywords = [
                'embeddings', 'embed_tokens', 'wte', 'word_embeddings',  # Embedding layers
                'layer.', 'layers.', 'h.', 'block.',  # Transformer layers
                'attention', 'attn', 'self_attn', 'cross_attn',  # Attention blocks
                'feed_forward', 'ffn', 'mlp', 'intermediate',  # Feed-forward blocks
                'encoder', 'decoder'  # Encoder/decoder blocks
            ]
            
            if any(keyword in name.lower() for keyword in universal_keywords):
                # Must be substantial (have parameters and children)
                if info['parameter_count'] > 50 and len(info['children']) > 1:
                    # Avoid too deep nesting - prefer layer-level functions
                    if name.count('.') <= 3:
                        candidates[name] = info
        
        return candidates
    
    def trace_execution_with_hooks(self, model, dummy_inputs):
        """Trace model execution to map operations to modules"""
        print("=== Tracing Execution with Hooks ===")
        
        execution_order = []
        module_contexts = {}
        
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
        
        # Register hooks for leaf modules
        hooks = []
        for name, module in model.named_modules():
            if name and len(list(module.children())) == 0:  # Leaf modules
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
        """Create mapping from parameters to modules for ONNX node matching"""
        print("=== Creating Parameter Mapping ===")
        
        for name, module in model.named_modules():
            if name:
                for param_name, param in module.named_parameters(recurse=False):
                    # ONNX parameter naming convention
                    onnx_param_name = f"{name}.{param_name}".replace('.', '_')
                    self.parameter_mapping[onnx_param_name] = {
                        'module': name,
                        'param_name': param_name,
                        'shape': list(param.shape),
                        'dtype': str(param.dtype)
                    }
        
        print(f"Mapped {len(self.parameter_mapping)} parameters")
        return self.parameter_mapping
    
    def export_with_standard_method(self, model, dummy_inputs, output_path):
        """Export using standard method with enhanced metadata"""
        print("=== Standard ONNX Export with Metadata ===")
        
        # Standard export
        if isinstance(dummy_inputs, dict):
            # Handle multiple inputs (BERT case)
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
            do_constant_folding=False,  # Preserve operations for better mapping
            input_names=input_names,
            output_names=['output'],
            verbose=False
        )
        
        # Enhance with metadata
        onnx_model = onnx.load(output_path)
        self.enhance_onnx_with_hierarchy(onnx_model, output_path)
        
        return onnx_model
    
    def enhance_onnx_with_hierarchy(self, onnx_model, output_path):
        """Enhance ONNX model with hierarchy information"""
        print("=== Enhancing ONNX with Hierarchy ===")
        
        # Add hierarchy metadata
        hierarchy_meta = onnx_model.metadata_props.add()
        hierarchy_meta.key = "module_hierarchy"
        hierarchy_meta.value = json.dumps(self.module_hierarchy, indent=2)
        
        # Add parameter mapping metadata
        param_meta = onnx_model.metadata_props.add()
        param_meta.key = "parameter_mapping"
        param_meta.value = json.dumps(self.parameter_mapping, indent=2)
        
        # Add execution trace metadata
        trace_meta = onnx_model.metadata_props.add()
        trace_meta.key = "execution_trace"
        trace_meta.value = json.dumps(self.execution_trace, indent=2)
        
        # Enhance node names and add module attributes
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
                
                # Enhance node name
                node.name = f"{source_module}.{node.op_type}_{i}"
                enhanced_nodes += 1
        
        print(f"Enhanced {enhanced_nodes}/{len(onnx_model.graph.node)} nodes")
        
        # Save enhanced model
        enhanced_path = output_path.replace('.onnx', '_with_hierarchy.onnx')
        onnx.save(onnx_model, enhanced_path)
        print(f"Saved enhanced model: {enhanced_path}")
        
        return enhanced_path
    
    def infer_node_module(self, node):
        """Infer which module a node belongs to based on parameters"""
        # Strategy 1: Direct parameter matching
        for input_name in node.input:
            for param_key, param_info in self.parameter_mapping.items():
                if param_key in input_name or input_name.endswith(param_key):
                    return param_info['module']
        
        # Strategy 2: Pattern matching for common operations
        if node.op_type in ['MatMul', 'Gemm']:
            for input_name in node.input:
                if 'weight' in input_name:
                    # Try to extract module name from weight name
                    # e.g., "encoder_layer_0_attention_self_query_weight" -> "encoder.layer.0.attention.self.query"
                    parts = input_name.replace('_weight', '').replace('_bias', '').split('_')
                    if len(parts) >= 2:
                        # Reconstruct module path
                        module_candidate = '.'.join(parts)
                        if module_candidate in self.module_hierarchy:
                            return module_candidate
        
        return None
    
    def export_components_separately(self, model, dummy_inputs, model_config=None):
        """Export individual components for validation"""
        print("=== Exporting Components Separately ===")
        
        components = self.extract_components(model)
        component_exports = {}
        
        for comp_name, component in components.items():
            try:
                comp_input = self.create_component_input(comp_name, dummy_inputs, model_config)
                if comp_input is not None:
                    export_result = self.export_single_component(component, comp_name, comp_input)
                    if export_result:
                        component_exports[comp_name] = export_result
            except Exception as e:
                print(f"Failed to export {comp_name}: {e}")
        
        return component_exports
    
    def extract_components(self, model):
        """Extract major components for separate export - works with any HF model"""
        components = {}
        model_type = type(model).__name__.lower()
        
        # Universal embedding extraction
        embedding_attrs = ['embeddings', 'embed_tokens', 'wte', 'word_embeddings', 'token_embedding']
        for attr in embedding_attrs:
            if hasattr(model, attr):
                components[f'{attr}'] = getattr(model, attr)
                break
        
        # Universal transformer layer extraction
        layer_containers = [
            ('encoder.layer', lambda m: hasattr(m, 'encoder') and hasattr(m.encoder, 'layer')),
            ('decoder.layer', lambda m: hasattr(m, 'decoder') and hasattr(m.decoder, 'layer')),
            ('h', lambda m: hasattr(m, 'h')),  # GPT-style
            ('layers', lambda m: hasattr(m, 'layers')),  # Some models use 'layers'
            ('transformer.h', lambda m: hasattr(m, 'transformer') and hasattr(m.transformer, 'h'))
        ]
        
        for container_name, check_func in layer_containers:
            if check_func(model):
                layer_container = self._get_nested_attr(model, container_name)
                if layer_container is not None:
                    for i, layer in enumerate(layer_container):
                        layer_prefix = container_name.replace('.', '_')
                        components[f'{layer_prefix}_{i}'] = layer
                        
                        # Extract attention sub-components
                        attention_attrs = ['attention', 'attn', 'self_attn', 'cross_attn']
                        for attn_attr in attention_attrs:
                            if hasattr(layer, attn_attr):
                                components[f'{layer_prefix}_{i}_{attn_attr}'] = getattr(layer, attn_attr)
                                
                                # Extract self-attention if nested
                                attn_module = getattr(layer, attn_attr)
                                if hasattr(attn_module, 'self'):
                                    components[f'{layer_prefix}_{i}_{attn_attr}_self'] = attn_module.self
                        
                        # Extract feed-forward components
                        ff_attrs = ['intermediate', 'ffn', 'mlp', 'feed_forward']
                        for ff_attr in ff_attrs:
                            if hasattr(layer, ff_attr):
                                components[f'{layer_prefix}_{i}_{ff_attr}'] = getattr(layer, ff_attr)
                        
                        # Extract output/dense layers
                        output_attrs = ['output', 'dense', 'out_proj']
                        for out_attr in output_attrs:
                            if hasattr(layer, out_attr):
                                components[f'{layer_prefix}_{i}_{out_attr}'] = getattr(layer, out_attr)
                    break
        
        # Extract other common components
        other_components = ['pooler', 'classifier', 'lm_head', 'prediction_head']
        for comp_name in other_components:
            if hasattr(model, comp_name):
                components[comp_name] = getattr(model, comp_name)
        
        print(f"Extracted {len(components)} components from {model_type} model")
        return components
    
    def _get_nested_attr(self, obj, attr_path):
        """Get nested attribute using dot notation"""
        try:
            for attr in attr_path.split('.'):
                obj = getattr(obj, attr)
            return obj
        except AttributeError:
            return None
    
    def create_component_input(self, comp_name, dummy_inputs, model_config=None):
        """Create appropriate input for component based on its position and model type"""
        # Handle embedding layers
        embedding_keywords = ['embeddings', 'embed_tokens', 'wte', 'word_embeddings', 'token_embedding']
        if any(keyword in comp_name for keyword in embedding_keywords):
            if isinstance(dummy_inputs, dict):
                return dummy_inputs.get('input_ids', list(dummy_inputs.values())[0])
            else:
                return dummy_inputs
        
        # Determine hidden size from model config or use reasonable defaults
        batch_size = 1
        seq_length = 32
        
        if model_config and hasattr(model_config, 'hidden_size'):
            hidden_size = model_config.hidden_size
        elif model_config and hasattr(model_config, 'd_model'):
            hidden_size = model_config.d_model  # T5 uses d_model
        elif model_config and hasattr(model_config, 'n_embd'):
            hidden_size = model_config.n_embd  # GPT uses n_embd
        else:
            hidden_size = 768  # Default transformer size
        
        # For transformer layers and attention components
        layer_keywords = ['layer', 'h_', 'attention', 'attn', 'intermediate', 'ffn', 'mlp', 'output', 'dense']
        if any(keyword in comp_name for keyword in layer_keywords):
            hidden_states = torch.randn(batch_size, seq_length, hidden_size)
            
            # Include attention mask if available and component needs it
            if ('attention' in comp_name or 'attn' in comp_name) and isinstance(dummy_inputs, dict):
                if 'attention_mask' in dummy_inputs:
                    return (hidden_states, dummy_inputs['attention_mask'])
            
            return (hidden_states,)
        
        # For classification heads and output layers
        if any(keyword in comp_name for keyword in ['pooler', 'classifier', 'lm_head', 'prediction_head']):
            if 'pooler' in comp_name:
                # Pooler typically takes hidden states
                return (torch.randn(batch_size, seq_length, hidden_size),)
            else:
                # Classification heads typically take pooled output
                return (torch.randn(batch_size, hidden_size),)
        
        return None
    
    def export_single_component(self, component, comp_name, comp_input):
        """Export a single component to ONNX"""
        class ComponentWrapper(nn.Module):
            def __init__(self, comp):
                super().__init__()
                self.comp = comp
            
            def forward(self, *args):
                return self.comp(*args)
        
        wrapper = ComponentWrapper(component)
        wrapper.eval()
        
        output_path = f"hf_component_{comp_name.replace('.', '_')}.onnx"
        
        with torch.no_grad():
            torch.onnx.export(
                wrapper,
                comp_input,
                output_path,
                export_params=True,
                opset_version=14,
                do_constant_folding=False,
                verbose=False
            )
        
        # Analyze exported component
        comp_onnx = onnx.load(output_path)
        node_count = len(comp_onnx.graph.node)
        
        return {
            'path': output_path,
            'node_count': node_count,
            'onnx_model': comp_onnx,
            'operations': [{'name': node.name, 'op_type': node.op_type} for node in comp_onnx.graph.node]
        }
    
    def validate_hierarchy_preservation(self, whole_model_path, component_exports):
        """Validate that hierarchy is preserved by comparing whole vs parts"""
        print("=== Validating Hierarchy Preservation ===")
        
        # Load whole model with hierarchy
        whole_model = onnx.load(whole_model_path)
        
        # Extract hierarchy metadata
        hierarchy_metadata = None
        for prop in whole_model.metadata_props:
            if prop.key == "module_hierarchy":
                hierarchy_metadata = json.loads(prop.value)
                break
        
        if not hierarchy_metadata:
            print("No hierarchy metadata found in whole model")
            return False
        
        # Group whole model operations by modules
        module_to_ops = self.group_operations_by_hierarchy(whole_model)
        
        # Compare with component exports
        validation_results = {}
        total_matches = 0
        total_components = 0
        
        for comp_name, comp_data in component_exports.items():
            if comp_name in module_to_ops:
                whole_ops = module_to_ops[comp_name]
                comp_ops = comp_data['operations']
                
                # Compare operation types
                whole_op_types = sorted([op['op_type'] for op in whole_ops])
                comp_op_types = sorted([op['op_type'] for op in comp_ops])
                
                match_ratio = self.calculate_operation_similarity(whole_op_types, comp_op_types)
                validation_results[comp_name] = {
                    'whole_ops': len(whole_ops),
                    'component_ops': len(comp_ops),
                    'match_ratio': match_ratio,
                    'matches': match_ratio > 0.7  # 70% similarity threshold
                }
                
                if match_ratio > 0.7:
                    total_matches += 1
                total_components += 1
                
                print(f"{comp_name}: {match_ratio:.2f} similarity ({len(whole_ops)} vs {len(comp_ops)} ops)")
        
        overall_success_rate = total_matches / total_components if total_components > 0 else 0
        print(f"\nOverall validation: {total_matches}/{total_components} components match ({overall_success_rate:.2f})")
        
        return validation_results
    
    def group_operations_by_hierarchy(self, onnx_model):
        """Group ONNX operations by their source modules"""
        module_to_ops = {}
        
        for node in onnx_model.graph.node:
            source_module = None
            
            # Check for source_module attribute
            for attr in node.attribute:
                if attr.name == "source_module":
                    source_module = attr.s.decode('utf-8')
                    break
            
            if source_module:
                if source_module not in module_to_ops:
                    module_to_ops[source_module] = []
                
                module_to_ops[source_module].append({
                    'name': node.name,
                    'op_type': node.op_type
                })
        
        return module_to_ops
    
    def calculate_operation_similarity(self, ops1, ops2):
        """Calculate similarity between two operation lists"""
        if not ops1 and not ops2:
            return 1.0
        if not ops1 or not ops2:
            return 0.0
        
        # Simple Jaccard similarity
        set1, set2 = set(ops1), set(ops2)
        intersection = len(set1.intersection(set2))
        union = len(set1.union(set2))
        
        return intersection / union if union > 0 else 0.0

def create_universal_dummy_inputs(model_config=None, model=None):
    """Create dummy inputs for any HF model using truly universal approach"""
    batch_size = 1
    seq_length = 32
    
    # Get vocab size from config if available
    vocab_size = 30000
    if model_config and hasattr(model_config, 'vocab_size'):
        vocab_size = model_config.vocab_size
    
    # Start with the most common input
    inputs = {
        'input_ids': torch.randint(0, min(vocab_size, 1000), (batch_size, seq_length))
    }
    
    # Universal approach: inspect the model's forward method signature
    # to determine what inputs it expects
    if model is not None:
        try:
            import inspect
            forward_signature = inspect.signature(model.forward)
            expected_params = list(forward_signature.parameters.keys())
            
            # Add common inputs based on what the model expects
            if 'attention_mask' in expected_params:
                inputs['attention_mask'] = torch.ones(batch_size, seq_length)
            
            if 'token_type_ids' in expected_params:
                inputs['token_type_ids'] = torch.zeros(batch_size, seq_length, dtype=torch.long)
            
            # For encoder-decoder models
            if 'decoder_input_ids' in expected_params:
                inputs['decoder_input_ids'] = torch.randint(0, min(vocab_size, 1000), (batch_size, seq_length))
            
            if 'decoder_attention_mask' in expected_params:
                inputs['decoder_attention_mask'] = torch.ones(batch_size, seq_length)
            
            # For models that expect pixel values (vision models)
            if 'pixel_values' in expected_params:
                # Assume common image size
                inputs['pixel_values'] = torch.randn(batch_size, 3, 224, 224)
            
        except Exception as e:
            print(f"Could not inspect model signature: {e}")
            # Fall back to common defaults
            inputs['attention_mask'] = torch.ones(batch_size, seq_length)
    
    return inputs

def main(model_name=None):
    """Main test function for universal HF model hierarchy preservation"""
    print("=== Universal HF Model Hierarchy Preservation Test ===\n")
    
    # Default models for testing different architectures
    if model_name is None:
        test_models = [
            "google/bert_uncased_L-2_H-128_A-2",  # BERT
            "gpt2",  # GPT-2
            "distilbert-base-uncased",  # DistilBERT
        ]
        model_name = test_models[0]  # Default to BERT for this run
    
    print(f"Loading model: {model_name}")
    
    try:
        model = AutoModel.from_pretrained(model_name)
        model_config = model.config
        model_type = type(model).__name__
        
        # Create appropriate dummy inputs based on model
        dummy_inputs = create_universal_dummy_inputs(model_config, model)
        
        # Initialize universal exporter
        exporter = UniversalHFHierarchyExporter()
        
        # Step 1: Analyze model structure
        function_candidates = exporter.analyze_model_structure(model)
        
        # Step 2: Trace execution
        execution_trace = exporter.trace_execution_with_hooks(model, dummy_inputs)
        
        # Step 3: Create parameter mapping
        parameter_mapping = exporter.create_parameter_mapping(model)
        
        # Step 4: Export whole model with hierarchy
        safe_model_name = model_name.replace('/', '_').replace('-', '_')
        whole_model_path = f"{safe_model_name}_whole_model.onnx"
        whole_onnx = exporter.export_with_standard_method(model, dummy_inputs, whole_model_path)
        
        # Step 5: Export components separately (pass model config for better input creation)
        component_exports = exporter.export_components_separately(model, dummy_inputs, model_config)
        
        # Step 6: Validate hierarchy preservation
        validation_results = exporter.validate_hierarchy_preservation(
            whole_model_path.replace('.onnx', '_with_hierarchy.onnx'),
            component_exports
        )
        
        # Step 7: Summary
        print(f"\n=== Test Summary ===")
        print(f"Model: {model_name} ({model_type})")
        print(f"Total modules: {len(exporter.module_hierarchy)}")
        print(f"Function candidates: {len(function_candidates)}")
        print(f"Execution trace: {len(execution_trace)} operations")
        print(f"Parameter mappings: {len(parameter_mapping)}")
        print(f"Component exports: {len(component_exports)}")
        
        # Show validation results
        if validation_results:
            successful_validations = sum(1 for r in validation_results.values() if r['matches'])
            print(f"Validation success: {successful_validations}/{len(validation_results)} components")
            
            print(f"\nValidation details:")
            for comp_name, result in validation_results.items():
                status = "✓" if result['matches'] else "✗"
                print(f"  {status} {comp_name}: {result['match_ratio']:.2f} similarity")
        
        return exporter, validation_results
        
    except Exception as e:
        print(f"Error processing model {model_name}: {e}")
        import traceback
        traceback.print_exc()
        return None, None

if __name__ == "__main__":
    main()