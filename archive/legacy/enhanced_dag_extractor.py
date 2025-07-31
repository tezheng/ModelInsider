#!/usr/bin/env python3
"""
Enhanced DAG Extractor with 100% Node Tagging
Tag ALL operations to enable subgraph extraction
"""

import json
import os
import traceback
from collections import defaultdict

import torch
from dag_extractor import DAGExtractor


class EnhancedDAGExtractor(DAGExtractor):
    """Enhanced DAG extractor that tags 100% of operations for subgraph extraction"""
    
    def __init__(self):
        super().__init__()
        self.tensor_to_module = {}  # Maps tensor names to the modules that created them
        self.operation_execution_order = []  # Track execution order for better tagging
        
    def trace_execution_with_hooks(self, model, dummy_inputs):
        """Enhanced execution tracing that tracks tensor flow"""
        print("=== Enhanced Execution Tracing ===")
        
        self.tensor_to_module = {}
        self.operation_execution_order = []
        
        def forward_hook(module, input, output):
            module_name = getattr(module, '_hierarchy_name', 'unknown')
            if module_name != 'unknown' and module_name in self.module_hierarchy:
                hierarchy_path = self.module_hierarchy[module_name]['hierarchy_path']
                
                # Track this module's execution
                self.operation_execution_order.append({
                    'module': module_name,
                    'hierarchy_path': hierarchy_path,
                    'type': type(module).__name__,
                    'order': len(self.operation_execution_order)
                })
                
                # Map output tensors to this module
                if isinstance(output, torch.Tensor):
                    tensor_id = id(output)
                    self.tensor_to_module[tensor_id] = hierarchy_path
                elif isinstance(output, tuple | list):
                    for _i, tensor in enumerate(output):
                        if isinstance(tensor, torch.Tensor):
                            tensor_id = id(tensor)
                            self.tensor_to_module[tensor_id] = hierarchy_path
        
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
        
        self.execution_trace = self.operation_execution_order
        print(f"Traced {len(self.operation_execution_order)} module executions")
        return self.operation_execution_order
    
    def analyze_onnx_operations(self, onnx_model):
        """Enhanced ONNX analysis that tags ALL operations"""
        print("=== Enhanced ONNX Operations Analysis ===")
        
        # Get model inputs and outputs
        model_inputs = set()
        for input_info in onnx_model.graph.input:
            model_inputs.add(input_info.name)
        
        model_outputs = set()
        for output_info in onnx_model.graph.output:
            model_outputs.add(output_info.name)
        
        # Build complete tensor flow graph
        tensor_producers = {}  # tensor_name -> operation_name that produces it
        operation_consumers = defaultdict(list)  # operation_name -> list of operations that consume its outputs
        
        # Process initializers (parameters)
        for init in onnx_model.graph.initializer:
            tags = self.get_parameter_tags(init.name)
            if tags:
                self.operation_metadata[init.name] = {
                    'op_type': 'Initializer',
                    'inputs': [],
                    'outputs': [init.name],
                    'tags': tags
                }
                tensor_producers[init.name] = init.name
                
                # Add to module operations
                for tag in tags:
                    self.module_operations[tag].append(init.name)
        
        # First pass: collect all operations and build tensor flow
        all_operations = []
        for node in onnx_model.graph.node:
            op_name = node.name if node.name else f"{node.op_type}_{len(all_operations)}"
            all_operations.append((op_name, node))
            
            # Track tensor producers
            for output in node.output:
                tensor_producers[output] = op_name
        
        # Second pass: build operation dependencies
        for op_name, node in all_operations:
            for input_tensor in node.input:
                if input_tensor in tensor_producers:
                    producer_op = tensor_producers[input_tensor]
                    operation_consumers[producer_op].append(op_name)
        
        # Third pass: tag operations using enhanced strategy
        for op_name, node in all_operations:
            # Skip tagging Constant nodes to avoid extracting embedded constants
            # Only tag if it's a genuine computational operation
            if node.op_type == 'Constant':
                # Don't tag constant nodes - let them remain as embedded values
                continue
            
            tags = self.get_enhanced_operation_tags(node, op_name, tensor_producers, model_inputs, model_outputs)
            
            # Every operation gets at least one tag
            if not tags:
                tags = self.get_fallback_tags(node, op_name, tensor_producers)
            
            if tags:  # Should always be true now
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
        
        # Verify coverage (excluding Constant nodes which we intentionally skip)
        total_nodes = len(onnx_model.graph.node)
        constant_nodes = len([node for node in onnx_model.graph.node if node.op_type == 'Constant'])
        taggable_nodes = total_nodes - constant_nodes
        tagged_operations = len(self.operation_metadata)
        coverage = tagged_operations / taggable_nodes if taggable_nodes > 0 else 0
        print(f"Operation coverage: {tagged_operations}/{taggable_nodes} ({coverage:.1%}) [Excluded {constant_nodes} constant nodes]")
        
        if coverage < 1.0:
            print("⚠️  Not all taggable operations were tagged - investigating...")
            untagged_count = 0
            for node in onnx_model.graph.node:
                if node.op_type == 'Constant':
                    continue  # Skip constants as expected
                op_name = node.name if node.name else f"{node.op_type}_missing"
                if op_name not in self.operation_metadata:
                    untagged_count += 1
                    print(f"   Untagged: {op_name} ({node.op_type})")
            print(f"Total untagged: {untagged_count}")
        
        # Skip parent module aggregation - using single tag strategy per new requirements
    
    def _create_universal_parent_modules(self):
        """Create parent module tags based on operation execution patterns"""
        print("=== Creating Universal Parent Module Tags ===")
        
        # Group operations by execution context and parameter relationships
        parent_candidates = {}
        
        for op_name, op_info in self.operation_metadata.items():
            if op_info.get('op_type') == 'Initializer':
                continue
                
            tags = op_info.get('tags', [])
            for tag in tags:
                # Look for potential parent modules (modules with child Linear/submodules)
                tag_parts = tag.split('/')
                
                # Try different parent levels
                for i in range(len(tag_parts) - 1, 1, -1):  # Don't go to root level
                    potential_parent = '/'.join(tag_parts[:i])
                    
                    if potential_parent not in parent_candidates:
                        parent_candidates[potential_parent] = {
                            'operations': [],
                            'children': set(),
                            'parameter_count': 0
                        }
                    
                    parent_candidates[potential_parent]['operations'].append(op_name)
                    parent_candidates[potential_parent]['children'].add(tag)
                    
                    # Count parameters for this potential parent
                    if any(param_info.get('hierarchy_path') == tag for param_info in self.parameter_mapping.values()):
                        parent_candidates[potential_parent]['parameter_count'] += 1
        
        # Create parent modules for candidates with sufficient complexity
        created_parents = 0
        for parent_path, parent_info in parent_candidates.items():
            # Universal criteria for parent module creation:
            # 1. Has multiple child modules (>=3 for attention-like patterns) 
            # 2. Has sufficient operations (>=10)
            # 3. Has parameters (indicates computational significance)
            
            child_count = len(parent_info['children'])
            op_count = len(parent_info['operations'])
            param_count = parent_info['parameter_count']
            
            if child_count >= 3 and op_count >= 10 and param_count > 0:
                # Create parent module tag
                if parent_path not in self.module_operations:
                    self.module_operations[parent_path] = []
                
                # Aggregate operations from children
                for op_name in parent_info['operations']:
                    if op_name in self.operation_metadata:
                        # Add parent tag to operation
                        current_tags = self.operation_metadata[op_name].get('tags', [])
                        if parent_path not in current_tags:
                            current_tags.append(parent_path)
                            self.operation_metadata[op_name]['tags'] = current_tags
                
                self.module_operations[parent_path].extend(parent_info['operations'])
                created_parents += 1
                print(f"   Created parent: {parent_path} ({op_count} ops, {child_count} children)")
        
        print(f"Created {created_parents} universal parent modules")
    
    def get_enhanced_operation_tags(self, node, op_name: str, tensor_producers: dict, 
                                  model_inputs: set, model_outputs: set) -> list[str]:
        """Enhanced tagging strategy that assigns exactly ONE tag per operation"""
        
        # Strategy 1: Name-based tagging for attention operations (highest priority)
        # Map attention operations by their ONNX operation names
        attention_tag = self.get_attention_tag_from_name(op_name)
        if attention_tag:
            return [attention_tag]
        
        # Strategy 2: Parameter-based tagging
        candidate_tags = []
        for input_name in node.input:
            param_tags = self.get_parameter_tags(input_name)
            candidate_tags.extend(param_tags)
        
        # Strategy 3: Execution-based tagging (if no parameters)
        if not candidate_tags:
            for input_name in node.input:
                if input_name in tensor_producers:
                    producer_op = tensor_producers[input_name]
                    if producer_op in self.operation_metadata:
                        producer_tags = self.operation_metadata[producer_op]['tags']
                        candidate_tags.extend(producer_tags)
        
        # Strategy 4: Fallback to execution context
        if not candidate_tags and self.operation_execution_order:
            # Get the most recently executed module
            recent_module = self.operation_execution_order[-1]['hierarchy_path']
            candidate_tags.append(recent_module)
        
        # Apply CRITICAL TAGGING RULES:
        # 1. Select most specific transformers class (exclude torch.nn leaf classes)
        # 2. Return exactly ONE tag
        
        if candidate_tags:
            # Remove duplicates while preserving order
            unique_candidate_tags = []
            for tag in candidate_tags:
                if tag not in unique_candidate_tags:
                    unique_candidate_tags.append(tag)
            
            # Select the most specific transformers class
            best_tag = self.select_best_transformers_tag(unique_candidate_tags)
            if best_tag:
                return [best_tag]
        
        # Fallback: assign to root transformers class
        root_modules = [path for path in self.module_hierarchy.values() 
                      if path['depth'] == 0]
        if root_modules:
            return [root_modules[0]['hierarchy_path']]
        
        return ["/UnknownModule"]
    
    def is_attention_operation(self, node, op_name: str) -> bool:
        """Check if this operation is part of an attention mechanism"""
        attention_ops = [
            'MatMul',  # Query-Key multiplication, Attention-Value multiplication
            'Softmax',  # Attention probabilities
            'Add',     # Attention mask addition
            'Mul',     # Scaling operations
            'Transpose',  # Tensor reshaping for attention
            'Reshape',    # Tensor reshaping for attention heads
            'Concat',     # Concatenating attention heads
        ]
        
        # Check if this is a known attention operation type
        if node.op_type not in attention_ops:
            return False
        
        # Additional heuristics: check if operation name suggests attention
        attention_keywords = [
            'attention', 'query', 'key', 'value', 'self', 'head'
        ]
        
        op_name_lower = op_name.lower()
        return any(keyword in op_name_lower for keyword in attention_keywords)
    
    def get_attention_module_from_name(self, op_name: str) -> str | None:
        """Determine if an operation belongs to a specific attention module based on its name"""
        op_name_lower = op_name.lower()
        
        # Look for attention patterns in operation names, but be more specific
        # Include: /encoder/layer.0/attention/self/query/MatMul
        # Include: /encoder/layer.0/attention/self/MatMul (attention computation)  
        # Exclude: /encoder/layer.0/attention/output/dense/MatMul (output projection)
        
        if 'encoder/layer.' in op_name_lower and 'attention/self' in op_name_lower:
            # Exclude output dense layer operations
            if '/attention/output/' in op_name_lower:
                return None
                
            # REMOVED: Hardcoded architecture mapping not allowed in universal exporter
            pass
        
        return None
    
    def is_within_attention_scope(self, op_name: str, attention_tag: str) -> bool:
        """REMOVED: Hardcoded architecture scope checking not allowed in universal exporter"""
        # Universal approach: rely on parameter-based and execution-based tagging
        return False
    
    def get_fallback_tags(self, node, op_name: str, tensor_producers: dict) -> list[str]:
        """Fallback tagging for operations that couldn't be tagged by other methods"""
        
        # Strategy 1: For activation functions, assign to the module whose output they process
        if node.op_type in ['Relu', 'Tanh', 'Sigmoid', 'Gelu', 'Softmax']:
            for input_name in node.input:
                if input_name in tensor_producers:
                    producer_op = tensor_producers[input_name]
                    if producer_op in self.operation_metadata:
                        producer_tags = self.operation_metadata[producer_op]['tags']
                        if producer_tags:
                            return [producer_tags[0]]  # Inherit from producer
        
        # Strategy 2: For reshape operations, assign to root module
        # NOTE: Exclude 'Constant' to avoid extracting embedded constants into separate nodes
        if node.op_type in ['Reshape', 'Transpose', 'Unsqueeze', 'Squeeze']:
            # Find the root module
            root_modules = [path for path in self.module_hierarchy.values() 
                          if path['depth'] == 0]
            if root_modules:
                return [root_modules[0]['hierarchy_path']]
        
        # Strategy 3: For math operations, assign to the most recent module in execution order
        if node.op_type in ['Add', 'Mul', 'Sub', 'Div', 'MatMul']:
            if self.operation_execution_order:
                # Get the most recently executed module
                recent_module = self.operation_execution_order[-1]['hierarchy_path']
                return [recent_module]
        
        # Strategy 4: Assign to root module as ultimate fallback
        root_modules = [path for path in self.module_hierarchy.values() 
                      if path['depth'] == 0]
        if root_modules:
            return [root_modules[0]['hierarchy_path']]
        
        # Strategy 5: Create a default module if nothing else works
        return ["/UnknownModule"]

    def save_node_tag_mapping(self, output_path: str):
        """Save complete node-to-tag mapping as JSON"""
        mapping = {
            "metadata": {
                "total_operations": len(self.operation_metadata),
                "total_modules": len(self.module_hierarchy),
                "generation_info": {
                    "excluded_constant_nodes": True,
                    "tagging_strategies": ["parameter_based", "execution_based", "fallback"]
                }
            },
            "node_tags": {},
            "tag_statistics": {},
            "untagged_operations": []
        }
        
        # Process all operations
        for op_name, op_info in self.operation_metadata.items():
            if op_info.get('op_type') == 'Initializer':
                continue  # Skip initializers
                
            tags = op_info.get('tags', [])
            mapping["node_tags"][op_name] = {
                "op_type": op_info.get('op_type', 'Unknown'),
                "tags": tags,
                "input_count": op_info.get('input_count', 0),
                "output_count": op_info.get('output_count', 0)
            }
            
            # Update statistics
            if tags:
                for tag in tags:
                    if tag not in mapping["tag_statistics"]:
                        mapping["tag_statistics"][tag] = 0
                    mapping["tag_statistics"][tag] += 1
            else:
                mapping["untagged_operations"].append(op_name)
        
        # Save to file
        import json
        from pathlib import Path
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w') as f:
            json.dump(mapping, f, indent=2)
        
        print(f"   Saved {len(mapping['node_tags'])} node mappings")
        print(f"   Unique tags: {len(mapping['tag_statistics'])}")
        print(f"   Untagged operations: {len(mapping['untagged_operations'])}")
    
    def get_attention_tag_from_name(self, op_name: str) -> str:
        """Universal operation-to-module mapping - NO hardcoded patterns allowed"""
        # Universal approach: NO hardcoded architecture-specific logic
        # Let parameter-based and execution-based tagging handle everything
        return None
    
    
    def select_best_transformers_tag(self, candidate_tags: list[str]) -> str:
        """Select the best transformers class tag according to the new rules"""
        if not candidate_tags:
            return None
        
        if len(candidate_tags) == 1:
            # Even single tags need torch.nn stripping
            tag = candidate_tags[0]
            return self._strip_torch_nn_suffix(tag)
        
        # Define torch.nn classes that should be excluded as final tags
        torch_nn_classes = {
            'Linear', 'LayerNorm', 'Dropout', 'Embedding', 'Conv1d', 'Conv2d', 
            'BatchNorm1d', 'BatchNorm2d', 'ReLU', 'GELU', 'Tanh', 'Sigmoid',
            'MultiheadAttention', 'TransformerEncoder', 'TransformerDecoder'
        }
        
        # Step 1: Strip torch.nn suffixes from ALL tags
        clean_tags = []
        for tag in candidate_tags:
            clean_tag = self._strip_torch_nn_suffix(tag)
            if clean_tag not in clean_tags:
                clean_tags.append(clean_tag)
        
        # Step 2: Select the most specific transformers class
        if clean_tags:
            # Find the deepest (most specific) transformers class
            best_tag = clean_tags[0]
            max_depth = len(best_tag.split('/'))
            
            for tag in clean_tags[1:]:
                depth = len(tag.split('/'))
                if depth > max_depth:
                    max_depth = depth
                    best_tag = tag
            
            return best_tag
        
        # Step 3: Fallback to the first tag (stripped)
        return self._strip_torch_nn_suffix(candidate_tags[0])
    
    def _strip_torch_nn_suffix(self, tag: str) -> str:
        """Strip torch.nn class suffixes from a tag"""
        torch_nn_classes = {
            'Linear', 'LayerNorm', 'Dropout', 'Embedding', 'Conv1d', 'Conv2d', 
            'BatchNorm1d', 'BatchNorm2d', 'ReLU', 'GELU', 'Tanh', 'Sigmoid',
            'MultiheadAttention', 'TransformerEncoder', 'TransformerDecoder'
        }
        
        for nn_class in torch_nn_classes:
            if tag.endswith(f'/{nn_class}'):
                # Remove the torch.nn class and return the parent
                return tag.rsplit('/', 1)[0]
        
        return tag
    
    def find_common_ancestor_module(self, module_paths: list[str]) -> str | None:
        """Find the deepest common ancestor of multiple module paths"""
        if not module_paths:
            return None
        
        if len(module_paths) == 1:
            return module_paths[0]
        
        # Split paths into components
        path_components = [path.split('/')[1:] for path in module_paths]  # Skip empty first element
        
        # Find common prefix
        common_components = []
        min_length = min(len(components) for components in path_components)
        
        for i in range(min_length):
            component = path_components[0][i]
            if all(components[i] == component for components in path_components):
                common_components.append(component)
            else:
                break
        
        if common_components:
            return '/' + '/'.join(common_components)
        
        return None


def test_enhanced_tagging(model_name: str = "resnet18"):
    """Test the enhanced tagging system"""
    print(f"=== Testing Enhanced Tagging: {model_name.upper()} ===")
    
    try:
        # Load model
        if model_name == "resnet18":
            import torchvision.models as models
            model = models.resnet18(pretrained=False)
            inputs = {'x': torch.randn(1, 3, 224, 224)}
        else:
            from input_generator import UniversalInputGenerator
            from transformers import AutoModel
            
            model = AutoModel.from_pretrained(model_name)
            generator = UniversalInputGenerator()
            inputs = generator.generate_inputs(model, model_name)
        
        # Initialize enhanced extractor
        extractor = EnhancedDAGExtractor()
        
        # Run analysis
        hierarchy = extractor.analyze_model_structure(model)
        trace = extractor.trace_execution_with_hooks(model, inputs)
        params = extractor.create_parameter_mapping(model)
        
        # Export and analyze ONNX
        os.makedirs("temp/enhanced_test", exist_ok=True)
        output_path = f"temp/enhanced_test/{model_name.replace('/', '_')}_enhanced.onnx"
        onnx_model = extractor.export_and_analyze_onnx(model, inputs, output_path)
        
        # Generate DAGs
        all_dags = extractor.generate_all_module_dags()
        
        # Save results
        with open(f"temp/enhanced_test/{model_name.replace('/', '_')}_enhanced_metadata.json", "w") as f:
            json.dump(extractor.operation_metadata, f, indent=2)
        
        with open(f"temp/enhanced_test/{model_name.replace('/', '_')}_enhanced_dags.json", "w") as f:
            json.dump(all_dags, f, indent=2)
        
        # Calculate coverage
        total_nodes = len(onnx_model.graph.node)
        tagged_operations = len([op for op in extractor.operation_metadata.values() 
                               if op['op_type'] != 'Initializer'])
        coverage = tagged_operations / total_nodes if total_nodes > 0 else 0
        
        print(f"\n✅ Enhanced Tagging Results for {model_name.upper()}:")
        print(f"   Total ONNX nodes: {total_nodes}")
        print(f"   Tagged operations: {tagged_operations}")
        print(f"   Coverage: {coverage:.1%}")
        print(f"   Modules with operations: {len(all_dags)}")
        
        return extractor, coverage
        
    except Exception as e:
        print(f"❌ Enhanced tagging failed: {e}")
        traceback.print_exc()
        return None, 0


def main():
    """Test enhanced tagging on different models"""
    models_to_test = [
        "resnet18",
        "google/bert_uncased_L-2_H-128_A-2"
    ]
    
    results = {}
    
    for model_name in models_to_test:
        extractor, coverage = test_enhanced_tagging(model_name)
        results[model_name] = coverage
        print(f"\n{'-'*60}")
    
    print(f"\n{'='*60}")
    print("ENHANCED TAGGING SUMMARY")
    print(f"{'='*60}")
    
    for model_name, coverage in results.items():
        status = "✅ SUCCESS" if coverage >= 0.95 else "⚠️  NEEDS IMPROVEMENT"
        print(f"{model_name:30} Coverage: {coverage:.1%} {status}")


if __name__ == "__main__":
    main()