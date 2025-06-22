"""
Universal Hierarchy-Preserving ONNX Exporter

This module implements a universal approach to ONNX export that preserves
PyTorch model hierarchy through usage-based tagging. The implementation
follows Option B design: tag operations only when they are actually used
during forward pass execution.

Key Principles:
1. NO HARDCODED LOGIC - works with any PyTorch model
2. Usage-based tagging - operations tagged only when traced
3. Recursive propagation - operations that produce inputs get tagged
4. Stack-based context - preserves module execution hierarchy
5. NO TORCH.NN MODULES in tags - only model-specific modules appear in hierarchy
"""

from __future__ import annotations

import torch
import torch.onnx
import onnx
from typing import Dict, List, Any, Optional, Union, Tuple
from pathlib import Path
import tempfile
from collections import defaultdict, deque


class HierarchyExporter:
    """
    Universal hierarchy-preserving ONNX exporter.

    This exporter works with ANY PyTorch model by leveraging the universal
    nn.Module hierarchy structure and forward hooks for execution tracing.
    """
    
    # Semantically important torch.nn modules that should create hierarchy levels
    TORCH_NN_HIERARCHY_EXCEPTIONS = {
        "LayerNorm",      # Normalization layers are architecturally significant
        "Embedding",      # Embedding layers represent major components
        "BatchNorm1d",    # Batch normalization variants
        "BatchNorm2d",
        "BatchNorm3d",
        "GroupNorm",
        "InstanceNorm1d",
        "InstanceNorm2d",
        "InstanceNorm3d",
    }

    def __init__(self, strategy: str = "usage_based", torch_nn_exceptions: Optional[List[str]] = None):
        """
        Initialize the HierarchyExporter.

        Args:
            strategy: Tagging strategy to use. Currently supports "usage_based"
            torch_nn_exceptions: Override default list of torch.nn modules that create hierarchy
        """
        if strategy != "usage_based":
            raise ValueError(
                f"Unsupported strategy: {strategy}. Only 'usage_based' is implemented."
            )

        self.strategy = strategy
        self._tag_mapping: Dict[str, Dict[str, Any]] = {}
        self._tag_stack: List[str] = []  # Stack for hierarchical tags
        self._operation_context: Dict[str, List[str]] = defaultdict(list)
        self._tensor_producers: Dict[str, str] = {}
        self._tensor_consumer_mapping: Dict[str, List[str]] = {}
        self._tensor_to_tag: Dict[str, List[str]] = {}  # Tensor-to-tag mapping for filtering
        self._pre_hooks = []  # Pre-forward hooks
        self._post_hooks = []  # Post-forward hooks
        self._model = None  # Track the root model
        
        # Allow customization of torch.nn exceptions
        self._torch_nn_exceptions = (
            set(torch_nn_exceptions) if torch_nn_exceptions 
            else self.TORCH_NN_HIERARCHY_EXCEPTIONS.copy()
        )

    def export(
        self,
        model: torch.nn.Module,
        example_inputs: Union[torch.Tensor, Tuple, Dict],
        output_path: str,
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Export PyTorch model to ONNX with hierarchy-preserving tags.

        Args:
            model: PyTorch model to export
            example_inputs: Example inputs for tracing
            output_path: Path to save the ONNX model
            **kwargs: Additional arguments passed to torch.onnx.export

        Returns:
            Dictionary containing export metadata
        """
        # Step 1: Prepare model for tracing
        model.eval()
        self._reset_state()
        self._model = model  # Store reference to root model

        # Step 2: Register hooks for execution tracing
        self._register_hooks(model)

        try:
            # Step 3: Perform tracing to build operation context
            with torch.no_grad():
                self._trace_model_execution(model, example_inputs)

            # Step 3.5: Remove hooks before ONNX export to ensure clean topology
            self._remove_hooks()

            # Step 4: Export to ONNX (standard PyTorch export - PRESERVE TOPOLOGY)
            self._export_to_onnx(model, example_inputs, output_path, **kwargs)

            # Step 5: Load exported ONNX and analyze its structure
            onnx_model = onnx.load(output_path)
            
            # Step 6: Build tag mapping based on the ACTUAL exported graph
            self._build_tag_mapping_from_onnx(onnx_model)

            # Step 7: Inject tags into EXISTING nodes (no topology changes)
            self._inject_tags_into_onnx(output_path)

            return {
                "output_path": output_path,
                "strategy": self.strategy,
                "total_operations": len(self._tag_mapping),
                "tagged_operations": len(
                    [op for op in self._tag_mapping.values() if op.get("tags", [])]
                ),
            }

        finally:
            # Clean up hooks
            self._remove_hooks()

    def get_tag_mapping(self) -> Dict[str, Dict[str, Any]]:
        """
        Get the current tag mapping for all operations.

        Returns:
            Dictionary mapping operation names to their metadata including tags
        """
        return self._tag_mapping.copy()

    def _reset_state(self):
        """Reset internal state for new export."""
        self._tag_mapping.clear()
        self._tag_stack.clear()
        self._operation_context.clear()
        self._tensor_producers.clear()
        self._tensor_consumer_mapping = {}
        self._tensor_to_tag = {}
        self._model = None
        self._remove_hooks()

    def _register_hooks(self, model: torch.nn.Module):
        """Register pre and post forward hooks for stack-based execution tracing."""

        def create_pre_hook(module_name: str, module: torch.nn.Module):
            """Create pre-forward hook that pushes tag onto stack."""
            def pre_hook(module, inputs):
                # Build hierarchical tag for this module
                hierarchical_tag = self._build_hierarchical_tag(module_name, module)
                # Push tag onto stack - any operations from now use this tag
                self._tag_stack.append(hierarchical_tag)
                
                # Also record in operation context for later mapping
                self._operation_context[module_name] = {
                    "tag": hierarchical_tag,
                    "module_class": module.__class__.__name__,
                    "creates_hierarchy": True,
                    "stack_depth": len(self._tag_stack),
                }
            return pre_hook

        def create_post_hook(module_name: str, module: torch.nn.Module):
            """Create post-forward hook that pops tag from stack."""
            def post_hook(module, inputs, outputs):
                # Pop the tag when module execution completes
                if self._tag_stack:
                    self._tag_stack.pop()
            return post_hook

        def create_tagging_hook(module_name: str, module: torch.nn.Module):
            """Create hook for non-hierarchy modules that still need operation tagging."""
            def tagging_hook(module, inputs, outputs):
                # Record execution context for operation tagging but don't affect stack
                # Get current tag from stack (from parent module)
                current_tag = self.get_current_tag()
                if current_tag:
                    self._operation_context[module_name] = {
                        "tag": current_tag,  # Use parent's tag
                        "module_class": module.__class__.__name__,
                        "creates_hierarchy": False,
                        "parent_tag": current_tag,
                    }
            return tagging_hook

        # Register hooks on all modules using universal criteria
        for name, module in model.named_modules():
            if name:  # Skip root module
                module_class = module.__class__.__module__
                should_tag = self._should_tag_module(module_class)
                
                if should_tag:
                    creates_hierarchy = self._should_create_hierarchy_level(module)
                    
                    if creates_hierarchy:
                        # HF modules and torch.nn exceptions: Register pre/post hooks (push/pop stack)
                        pre_hook = module.register_forward_pre_hook(
                            create_pre_hook(name, module)
                        )
                        self._pre_hooks.append(pre_hook)
                        
                        post_hook = module.register_forward_hook(
                            create_post_hook(name, module)
                        )
                        self._post_hooks.append(post_hook)
                    else:
                        # Other torch.nn modules: Register only tagging hook (no stack change)
                        tag_hook = module.register_forward_hook(
                            create_tagging_hook(name, module)
                        )
                        self._post_hooks.append(tag_hook)

    def _should_create_hierarchy_level(self, module: torch.nn.Module) -> bool:
        """Determine if module should create a hierarchy level (push/pop stack)."""
        module_class_path = module.__class__.__module__
        module_class_name = module.__class__.__name__
        
        # Skip low-level PyTorch implementation modules
        if "torch._C" in module_class_path:
            return False
            
        # Skip built-in Python modules
        if module_class_path.startswith("builtins"):
            return False
            
        # torch.nn modules only create hierarchy if in exception list
        if module_class_path.startswith("torch.nn"):
            return module_class_name in self._torch_nn_exceptions
            
        # All other modules (HF, custom) create hierarchy levels
        return True
    
    def _should_tag_module(self, module_class_path: str) -> bool:
        """Determine if we should tag a module (all modules except internals)."""
        # Skip low-level PyTorch implementation modules
        if "torch._C" in module_class_path:
            return False

        # Skip built-in Python modules
        if module_class_path.startswith("builtins"):
            return False

        # Tag all other modules - operations need attribution
        return True

    def _build_hierarchical_tag(self, module_name: str, module: torch.nn.Module) -> str:
        """
        Build hierarchical tag from module name and class.

        UNIVERSAL APPROACH: Build path using actual module classes only.
        No hardcoded architecture assumptions.
        """
        # Always build from module name by resolving class names
        return self._resolve_hierarchical_path(module_name, module)

    def _resolve_hierarchical_path(
        self, module_name: str, module: torch.nn.Module
    ) -> str:
        """
        Resolve hierarchical path from module name.

        UNIVERSAL APPROACH: Map dot-separated module names to hierarchical class paths.
        No hardcoded architecture assumptions.

        IMPORTANT: torch.nn modules should NOT appear in tags - only model-specific modules.
        """
        if not self._model:
            return f"/{module.__class__.__name__}"

        # Build path by traversing the actual module hierarchy
        path_segments = []

        # Add root model class
        path_segments.append(self._model.__class__.__name__)

        # Parse the module name and map each segment to its actual class
        if module_name:
            current_module = self._model
            name_parts = module_name.split(".")

            for part in name_parts:
                if hasattr(current_module, part):
                    current_module = getattr(current_module, part)
                    # Filter out torch.nn modules - only include model-specific modules
                    module_path = current_module.__class__.__module__
                    if not module_path.startswith(
                        "torch._C"
                    ) and not module_path.startswith("torch.nn"):
                        path_segments.append(current_module.__class__.__name__)

        # Build the final path
        return "/" + "/".join(path_segments)

    def get_current_tag(self) -> Optional[str]:
        """Get current execution context tag from stack."""
        return self._tag_stack[-1] if self._tag_stack else None

    def _remove_hooks(self):
        """Remove all registered hooks."""
        # Remove pre-hooks
        for hook in self._pre_hooks:
            hook.remove()
        self._pre_hooks.clear()
        
        # Remove post-hooks
        for hook in self._post_hooks:
            hook.remove()
        self._post_hooks.clear()

    def _trace_model_execution(self, model: torch.nn.Module, example_inputs):
        """Trace model execution to capture operation context."""
        # Convert inputs to proper format
        if hasattr(example_inputs, "keys") and hasattr(example_inputs, "values"):
            # For dict-like objects (including HuggingFace BatchEncoding)
            # Convert to regular dict and filter to only tensor values
            if hasattr(example_inputs, "data"):
                # BatchEncoding object - access the underlying data
                tensor_inputs = {
                    k: v
                    for k, v in example_inputs.data.items()
                    if isinstance(v, torch.Tensor)
                }
            else:
                # Regular dict
                tensor_inputs = {
                    k: v
                    for k, v in example_inputs.items()
                    if isinstance(v, torch.Tensor)
                }
            _ = model(**tensor_inputs)
        elif isinstance(example_inputs, (tuple, list)):
            # For models expecting multiple positional arguments
            _ = model(*example_inputs)
        else:
            # For models expecting single tensor input
            _ = model(example_inputs)

    def _export_to_onnx(
        self, model: torch.nn.Module, example_inputs, output_path: str, **kwargs
    ):
        """Export model to ONNX using caller-provided parameters."""
        # Prepare inputs for torch.onnx.export
        if hasattr(example_inputs, "keys") and hasattr(example_inputs, "values"):
            # For dict-like objects (including HuggingFace BatchEncoding)
            if hasattr(example_inputs, "data"):
                # BatchEncoding object - access the underlying data
                tensor_inputs = {
                    k: v
                    for k, v in example_inputs.data.items()
                    if isinstance(v, torch.Tensor)
                }
            else:
                # Regular dict
                tensor_inputs = {
                    k: v
                    for k, v in example_inputs.items()
                    if isinstance(v, torch.Tensor)
                }
            input_args = tuple(tensor_inputs.values())
        elif isinstance(example_inputs, (tuple, list)):
            input_args = tuple(example_inputs)
        else:
            input_args = (example_inputs,)

        # Filter out non-ONNX parameters from kwargs (like input_specs)
        valid_onnx_params = {
            "export_params",
            "verbose",
            "training",
            "input_names",
            "output_names",
            "aten_fallback",
            "operator_export_type",
            "opset_version",
            "do_constant_folding",
            "keep_initializers_as_inputs",
            "custom_opsets",
            "export_modules_as_functions",
            "dynamic_axes",
            "strip_doc_string",
            "example_outputs",
        }

        filtered_kwargs = {k: v for k, v in kwargs.items() if k in valid_onnx_params}

        # Use filtered export parameters
        torch.onnx.export(model, input_args, output_path, **filtered_kwargs)

    def _build_tag_mapping_from_onnx(self, onnx_model):
        """
        Analyze ONNX graph and build tag mapping.

        This implements our Option B design: map ONNX operations back to
        the module execution contexts we captured during forward hooks.
        
        IMPORTANT: This method takes the actual exported ONNX model to ensure
        we work with the exact topology that was exported.
        """

        # Build tensor producer mapping
        tensor_producers = {}
        for node in onnx_model.graph.node:
            for output in node.output:
                tensor_producers[output] = (
                    node.name or f"{node.op_type}_{len(self._tag_mapping)}"
                )

        # Initialize tag mapping with all operations
        for node in onnx_model.graph.node:
            node_name = node.name or f"{node.op_type}_{len(self._tag_mapping)}"

            self._tag_mapping[node_name] = {
                "op_type": node.op_type,
                "tags": [],
                "inputs": list(node.input),
                "outputs": list(node.output),
            }

        # NEW APPROACH: Use stack-based operation context from tracing
        # 1. Map ONNX operations to module execution contexts via stack-based tracing
        # 2. Forward propagate tags through the dataflow graph
        # 3. Apply multi-consumer propagation for complete subgraph extraction
        # 4. Tag tensors and operations for filtering capabilities
        
        self._map_operations_to_stack_context(onnx_model)
        self._forward_propagate_tags(onnx_model, tensor_producers)
        self._propagate_tags_with_multi_consumer_logic(onnx_model, tensor_producers)
        self._tag_tensor_inputs_for_filtering(onnx_model)

    def _map_operations_to_stack_context(self, onnx_model):
        """Map ONNX operations to module execution contexts from stack-based tracing."""
        # Strategy: Use parameter names and operation patterns to infer module context
        # This leverages the operation_context built during stack-based tracing
        
        # Method 1: Direct parameter-based mapping (most reliable)
        param_to_module_context = {}
        param_names = {init.name for init in onnx_model.graph.initializer}
        
        for param_name in param_names:
            if not param_name.startswith("onnx::"):
                # Extract module name from parameter (e.g., "encoder.layer.0.attention.self.query.weight")
                module_name = self._extract_module_name_from_param(param_name)
                
                # Find matching execution context from our stack-based tracing
                if module_name in self._operation_context:
                    param_to_module_context[param_name] = self._operation_context[module_name]["tag"]
                else:
                    # Try parent modules
                    parent_module = self._find_parent_module_in_context(module_name)
                    if parent_module:
                        param_to_module_context[param_name] = parent_module
        
        # Method 2: Tag operations based on parameter usage
        for node in onnx_model.graph.node:
            node_name = node.name or f"{node.op_type}_{len([n for n in self._tag_mapping.keys() if node.op_type in n])}"
            
            # Find parameters used by this operation
            operation_tags = set()
            for input_tensor in node.input:
                if input_tensor in param_to_module_context:
                    operation_tags.add(param_to_module_context[input_tensor])
            
            # Apply tags to operation
            if operation_tags:
                self._tag_mapping[node_name]["tags"] = list(operation_tags)
    
    def _find_parent_module_in_context(self, module_name: str) -> Optional[str]:
        """Find parent module context for a given module name."""
        parts = module_name.split(".")
        
        # Walk up the hierarchy to find a parent with execution context
        for i in range(len(parts) - 1, 0, -1):
            parent_name = ".".join(parts[:i])
            if parent_name in self._operation_context:
                return self._operation_context[parent_name]["tag"]
        
        return None
    
    def _forward_propagate_tags(self, onnx_model, tensor_producers):
        """Forward propagate tags through the dataflow graph."""
        # This fills in operations that don't use parameters but process tagged tensors
        
        max_iterations = 10  # Prevent infinite loops
        for iteration in range(max_iterations):
            tags_changed = False
            
            for node in onnx_model.graph.node:
                node_name = node.name or f"{node.op_type}_{len([n for n in self._tag_mapping.keys() if node.op_type in n])}"
                current_tags = set(self._tag_mapping[node_name].get('tags', []))
                
                # Collect tags from input tensors
                input_tags = set()
                for input_tensor in node.input:
                    # Find the operation that produces this tensor
                    producer_node = tensor_producers.get(input_tensor)
                    if producer_node and producer_node in self._tag_mapping:
                        producer_tags = self._tag_mapping[producer_node].get('tags', [])
                        input_tags.update(producer_tags)
                
                # If this operation doesn't have tags but its inputs do, inherit them
                if not current_tags and input_tags:
                    self._tag_mapping[node_name]['tags'] = list(input_tags)
                    tags_changed = True
                elif input_tags and not input_tags.issubset(current_tags):
                    # Add new tags from inputs
                    all_tags = current_tags.union(input_tags)
                    self._tag_mapping[node_name]['tags'] = list(all_tags)
                    tags_changed = True
            
            # Stop if no changes in this iteration
            if not tags_changed:
                break
        
    def _propagate_tags_with_multi_consumer_logic(self, onnx_model, tensor_producers):
        """Apply multi-consumer propagation on top of stack-based foundation."""
        # Build tensor -> consumer modules mapping
        tensor_consumers = defaultdict(set)
        
        for node in onnx_model.graph.node:
            node_name = node.name or f"{node.op_type}_{len([n for n in self._tag_mapping.keys() if node.op_type in n])}"
            node_tags = self._tag_mapping.get(node_name, {}).get('tags', [])
            
            # For each input tensor this operation uses
            for input_tensor in node.input:
                # Add ALL tags from this consuming operation
                for tag in node_tags:
                    tensor_consumers[input_tensor].add(tag)
        
        # Propagate consumer tags back to producing operations
        for node in onnx_model.graph.node:
            node_name = node.name or f"{node.op_type}_{len([n for n in self._tag_mapping.keys() if node.op_type in n])}"
            
            for output_tensor in node.output:
                if output_tensor in tensor_consumers:
                    consumer_tags = tensor_consumers[output_tensor]
                    # Add all consumer tags to this operation
                    existing_tags = set(self._tag_mapping[node_name].get('tags', []))
                    all_tags = existing_tags.union(consumer_tags)
                    self._tag_mapping[node_name]['tags'] = list(all_tags)
    
    def _tag_tensor_inputs_for_filtering(self, onnx_model):
        """Tag tensor inputs with their context for filtering capabilities."""
        # Create tensor-to-tag mapping for input filtering
        self._tensor_to_tag = {}
        
        for node in onnx_model.graph.node:
            node_name = node.name or f"{node.op_type}_{len([n for n in self._tag_mapping.keys() if node.op_type in n])}"
            node_tags = self._tag_mapping.get(node_name, {}).get('tags', [])
            
            # Tag all input tensors with the operation's tags
            for input_tensor in node.input:
                if input_tensor not in self._tensor_to_tag:
                    self._tensor_to_tag[input_tensor] = []
                self._tensor_to_tag[input_tensor].extend(node_tags)
            
            # Tag all output tensors with the operation's tags
            for output_tensor in node.output:
                if output_tensor not in self._tensor_to_tag:
                    self._tensor_to_tag[output_tensor] = []
                self._tensor_to_tag[output_tensor].extend(node_tags)

    def _map_parameters_to_modules(self, onnx_model):
        """Map ONNX parameters to PyTorch modules based on naming and usage."""
        # Build mapping from parameter names to module contexts
        param_to_module = {}

        # Get parameter names from ONNX initializers
        param_names = {init.name for init in onnx_model.graph.initializer}

        # Method 1: Direct parameter name mapping (for parameters that retain names)
        for param_name in param_names:
            # Skip generic ONNX names (they'll be handled in Method 2)
            if param_name.startswith("onnx::"):
                continue

            # Extract module name from parameter name (e.g., "encoder.layer.0.attention.self.query.weight")
            module_name = self._extract_module_name_from_param(param_name)

            # Try direct match first
            if module_name in self._operation_context:
                param_to_module[param_name] = self._operation_context[module_name]
            else:
                # Find parent module (walk up the hierarchy)
                parent_module = self._find_parent_module(module_name)
                if parent_module and parent_module in self._operation_context:
                    param_to_module[param_name] = self._operation_context[parent_module]

        # Method 2: Infer parameter ownership from operation patterns
        # Map generic parameter names (like onnx::MatMul_14) to modules based on operation context
        operation_to_module = {}

        # Build operation name to module context mapping
        for node in onnx_model.graph.node:
            node_name = (
                node.name
                or f"{node.op_type}_{len([n for n in self._tag_mapping.keys() if node.op_type in n])}"
            )

            # Try to infer module from operation name pattern
            inferred_module = self._infer_module_from_operation_name(node_name)
            if inferred_module and inferred_module in self._operation_context:
                operation_to_module[node_name] = self._operation_context[
                    inferred_module
                ]

        # Map generic parameters to modules based on which operations use them
        for node in onnx_model.graph.node:
            node_name = (
                node.name
                or f"{node.op_type}_{len([n for n in self._tag_mapping.keys() if node.op_type in n])}"
            )

            if node_name in operation_to_module:
                module_context = operation_to_module[node_name]

                # Map any generic parameters used by this operation to this module
                for input_name in node.input:
                    if input_name in param_names and input_name.startswith("onnx::"):
                        if input_name not in param_to_module:
                            param_to_module[input_name] = module_context

        self._param_to_module = param_to_module

    def _infer_module_from_operation_name(self, operation_name: str) -> Optional[str]:
        """Infer module name from ONNX operation name using universal approach."""
        # UNIVERSAL APPROACH: Only use actual module hierarchy captured during execution
        # No hardcoded patterns - rely entirely on execution context

        # Extract hierarchical path from operation name if present
        if "/" in operation_name:
            parts = operation_name.strip("/").split("/")
            # Try to find the most specific module name that exists in our context
            for i in range(len(parts) - 1, -1, -1):
                candidate = ".".join(parts[: i + 1]) if i > 0 else parts[0]
                if candidate in self._operation_context:
                    return candidate

        # If no hierarchical match, check for direct name match
        if operation_name in self._operation_context:
            return operation_name

        # No inference possible - rely on parameter-based mapping only
        return None

    def _find_parent_module(self, module_name: str) -> Optional[str]:
        """
        Find the nearest parent module for a given module name (universal approach).

        For example: 'encoder.layer.0.attention.self.query' -> 'encoder.layer.0.attention.self'
        """
        # Walk up the hierarchy by removing segments from the end
        parts = module_name.split(".")
        for i in range(len(parts) - 1, 0, -1):
            parent_candidate = ".".join(parts[:i])
            if parent_candidate in self._operation_context:
                return parent_candidate
        return None

    def _extract_module_name_from_param(self, param_name: str) -> str:
        """Extract module name from parameter name (universal approach)."""
        # Remove common parameter suffixes
        param_suffixes = [
            ".weight",
            ".bias",
            ".running_mean",
            ".running_var",
            ".num_batches_tracked",
        ]

        module_name = param_name
        for suffix in param_suffixes:
            if module_name.endswith(suffix):
                module_name = module_name[: -len(suffix)]
                break

        return module_name

    def _tag_operations_by_parameter_usage(self, onnx_model):
        """Tag operations that use parameters with their module context."""
        # Get parameter names
        param_names = {init.name for init in onnx_model.graph.initializer}

        # For each operation that uses parameters
        for node in onnx_model.graph.node:
            node_name = (
                node.name
                or f"{node.op_type}_{len([n for n in self._tag_mapping.keys() if node.op_type in n])}"
            )

            # Check if this operation uses any parameters
            used_params = [inp for inp in node.input if inp in param_names]

            if used_params:
                # Find which modules own these parameters
                module_tags = set()
                for param_name in used_params:
                    if param_name in self._param_to_module:
                        module_context = self._param_to_module[param_name]
                        module_tags.add(module_context["tag"])

                # Add tags to this operation
                self._tag_mapping[node_name]["tags"].extend(list(module_tags))

    def _propagate_tags_recursively(self, onnx_model, tensor_producers):
        """Multi-consumer tensor tagging: Tag tensors with ALL consuming modules."""
        # Build tensor consumer mapping
        tensor_consumers = defaultdict(list)
        for node in onnx_model.graph.node:
            node_name = (
                node.name
                or f"{node.op_type}_{len([n for n in self._tag_mapping.keys() if node.op_type in n])}"
            )
            for inp in node.input:
                tensor_consumers[inp].append(node_name)

        # NEW MULTI-CONSUMER APPROACH: Tag tensors with ALL consuming modules
        self._tag_tensors_by_all_consumers(onnx_model, tensor_consumers)
        
        # FALLBACK: Still run traditional propagation for coverage
        self._propagate_backward(tensor_producers)
        self._propagate_forward(tensor_consumers)
        self._propagate_support_operations(tensor_producers, tensor_consumers)

    def _tag_tensors_by_all_consumers(self, onnx_model, tensor_consumers):
        """
        Tag tensors with ALL modules that consume them.
        
        This implements the R13 multi-consumer tagging design for subgraph extraction.
        Each tensor will be tagged with all modules that consume it, enabling
        complete subgraph extraction for any module hierarchy.
        """
        
        # Step 1: Build tensor -> consumer modules mapping
        tensor_consumer_tags = defaultdict(set)
        
        for node in onnx_model.graph.node:
            node_name = (
                node.name
                or f"{node.op_type}_{len([n for n in self._tag_mapping.keys() if node.op_type in n])}"
            )
            
            # Get all tags for this consuming operation
            node_tags = self._tag_mapping.get(node_name, {}).get('tags', [])
            
            # For each input tensor this operation uses
            for input_tensor in node.input:
                # Add ALL tags from this consuming operation
                for tag in node_tags:
                    tensor_consumer_tags[input_tensor].add(tag)
        
        # Step 2: Propagate consumer tags back to producing operations
        tensor_producers = {}
        for node in onnx_model.graph.node:
            node_name = (
                node.name
                or f"{node.op_type}_{len([n for n in self._tag_mapping.keys() if node.op_type in n])}"
            )
            for output_tensor in node.output:
                tensor_producers[output_tensor] = node_name
        
        # Tag producing operations with all consumer tags
        for tensor_name, consumer_tags in tensor_consumer_tags.items():
            if tensor_name in tensor_producers:
                producer_node = tensor_producers[tensor_name]
                
                if producer_node in self._tag_mapping:
                    producer_tags = self._tag_mapping[producer_node]['tags']
                    
                    # Add all consumer tags to this producing operation
                    for consumer_tag in consumer_tags:
                        if consumer_tag not in producer_tags:
                            producer_tags.append(consumer_tag)
        
        # Step 3: Record tensor-level tagging metadata for subgraph extraction
        self._tensor_consumer_mapping = dict(tensor_consumer_tags)

    def _propagate_backward(self, tensor_producers):
        """Propagate tags backward to operations that produce inputs."""
        propagation_queue = deque()

        # Start with operations that already have tags
        for node_name, node_info in self._tag_mapping.items():
            if node_info["tags"]:
                propagation_queue.append((node_name, node_info["tags"], 0))

        propagated = set()
        MAX_DEPTH = 6  # Increased depth for better coverage

        while propagation_queue:
            current_node, current_tags, depth = propagation_queue.popleft()

            if current_node in propagated or depth > MAX_DEPTH:
                continue
            propagated.add(current_node)

            # Find producers of inputs for this node
            current_inputs = self._tag_mapping[current_node]["inputs"]

            for input_tensor in current_inputs:
                if input_tensor in tensor_producers:
                    producer_node = tensor_producers[input_tensor]

                    if producer_node in self._tag_mapping:
                        producer_tags = self._tag_mapping[producer_node]["tags"]

                        # Simplified propagation: be more permissive
                        for tag in current_tags:
                            # Only block propagation for clearly incompatible tags
                            if self._tags_are_incompatible(producer_tags, tag):
                                continue

                            if tag not in producer_tags:
                                producer_tags.append(tag)

                        # Continue propagation
                        if producer_tags and producer_node not in propagated:
                            propagation_queue.append(
                                (producer_node, producer_tags, depth + 1)
                            )

    def _propagate_forward(self, tensor_consumers):
        """Propagate tags forward to operations that consume outputs."""
        propagation_queue = deque()

        # Start with operations that have tags
        for node_name, node_info in self._tag_mapping.items():
            if node_info["tags"]:
                propagation_queue.append((node_name, node_info["tags"], 0))

        propagated = set()
        MAX_DEPTH = 4  # Forward propagation with moderate depth

        while propagation_queue:
            current_node, current_tags, depth = propagation_queue.popleft()

            if current_node in propagated or depth > MAX_DEPTH:
                continue
            propagated.add(current_node)

            # Find consumers of outputs from this node
            current_outputs = self._tag_mapping[current_node]["outputs"]

            for output_tensor in current_outputs:
                if output_tensor in tensor_consumers:
                    for consumer_node in tensor_consumers[output_tensor]:
                        if consumer_node in self._tag_mapping:
                            consumer_tags = self._tag_mapping[consumer_node]["tags"]

                            # Forward propagation: tag operations that directly use tagged outputs
                            for tag in current_tags:
                                if self._should_propagate_forward(tag, consumer_node, output_tensor):
                                    if tag not in consumer_tags:
                                        consumer_tags.append(tag)

                            # Continue forward propagation
                            if consumer_tags and consumer_node not in propagated:
                                propagation_queue.append(
                                    (consumer_node, consumer_tags, depth + 1)
                                )

    def _propagate_support_operations(self, tensor_producers, tensor_consumers):
        """Special pass to tag support operations like Shape, Constant, Gather that support tagged operations."""
        # Find operations that are clearly support operations for tagged operations
        support_ops = ['Shape', 'Constant', 'Gather', 'Unsqueeze', 'Slice', 'Concat']

        for node_name, node_info in self._tag_mapping.items():
            if node_info["op_type"] in support_ops and not node_info["tags"]:
                # Check if this support operation feeds into tagged operations
                node_outputs = node_info["outputs"]
                
                for output_tensor in node_outputs:
                    if output_tensor in tensor_consumers:
                        for consumer_node in tensor_consumers[output_tensor]:
                            if consumer_node in self._tag_mapping:
                                consumer_tags = self._tag_mapping[consumer_node]["tags"]
                                if consumer_tags:
                                    # Tag this support operation with same tags as its consumers
                                    for tag in consumer_tags:
                                        if tag not in node_info["tags"]:
                                            node_info["tags"].append(tag)
                                    break  # Found a tagged consumer, stop looking

    def _tags_are_incompatible(self, existing_tags: List[str], new_tag: str) -> bool:
        """Check if a new tag is incompatible with existing tags."""
        if not existing_tags:
            return False
        
        # Tags are incompatible if they're from completely different module hierarchies
        new_parts = new_tag.strip('/').split('/')
        
        for existing_tag in existing_tags:
            existing_parts = existing_tag.strip('/').split('/')
            
            # Check if they share the same root (first 2 levels)
            if len(new_parts) >= 2 and len(existing_parts) >= 2:
                if new_parts[0] == existing_parts[0] and new_parts[1] == existing_parts[1]:
                    return False  # Compatible - same major component
        
        # If no compatibility found, they're incompatible
        return len(existing_tags) > 0

    def _should_propagate_forward(self, tag: str, consumer_node: str, tensor_name: str) -> bool:
        """Determine if a tag should propagate forward to a consumer node."""
        # Be more permissive for forward propagation
        # Only block if consumer is clearly from a different major module
        
        tag_parts = tag.strip('/').split('/')
        consumer_parts = consumer_node.strip('/').split('/')
        
        # Allow forward propagation within reasonable bounds
        if len(tag_parts) >= 2:
            # Don't propagate too far forward (avoid crossing major boundaries)
            if len(tag_parts) > 4:  # Deep tags should be more restricted
                return len(consumer_parts) <= 2  # Only to shallow consumers
        
        return True  # Default: allow forward propagation

    def _are_tags_compatible(self, existing_tag: str, new_tag: str) -> bool:
        """Check if two tags are compatible for propagation."""
        # Extract the module hierarchy levels
        existing_parts = existing_tag.strip("/").split("/")
        new_parts = new_tag.strip("/").split("/")

        # Tags are compatible if they share a common prefix
        # (i.e., they belong to the same module hierarchy branch)
        min_len = min(len(existing_parts), len(new_parts))

        # Allow propagation within the same major module (first 2-3 levels)
        for i in range(min(3, min_len)):
            if existing_parts[i] != new_parts[i]:
                return False

        return True

    def _should_propagate_tag(
        self, tag: str, producer_node: str, tensor_name: str
    ) -> bool:
        """Determine if a tag should propagate to a producer node."""
        # Don't propagate across major module boundaries
        # Examples of boundaries: embeddings -> encoder, encoder -> pooler

        # Extract module path from tag
        tag_parts = tag.strip("/").split("/")
        producer_parts = producer_node.strip("/").split("/")

        # Check for major boundary violations using universal structural approach
        # Don't propagate across major semantic boundaries
        if len(tag_parts) >= 3 and len(producer_parts) >= 2:
            # Use hierarchy depth and class name analysis instead of hardcoded patterns
            tag_depth = len(tag_parts)
            producer_depth = len(producer_parts)

            # Don't propagate from deep modules to shallow modules (likely different components)
            if tag_depth - producer_depth > 2:
                return False

            # Use universal semantic analysis based on class names
            tag_major_component = tag_parts[1] if len(tag_parts) > 1 else ""
            producer_major_component = (
                producer_parts[0] if len(producer_parts) > 0 else ""
            )

            # Map components to their semantic equivalents using universal approach
            tag_semantic = self._get_semantic_component(tag_major_component)
            producer_semantic = self._get_semantic_component(producer_major_component)

            # Don't propagate across different major components
            if (
                tag_semantic != producer_semantic
                and tag_semantic != "unknown"
                and producer_semantic != "unknown"
            ):
                return False

        # Universal boundary check: Allow propagation within the same module hierarchy
        # Use depth-based heuristic instead of hardcoded patterns
        if len(tag_parts) >= 4:  # Deep module paths suggest hierarchical structure
            tag_depth = len(tag_parts)
            if tag_depth > 5:  # Only restrict very deep tags
                return False

            # Additional check: if producer node contains hierarchical path elements
            # that match the tag hierarchy, allow propagation
            producer_lower = producer_node.lower()

            # Look for matching hierarchy elements (but not just any substring)
            for i, part in enumerate(tag_parts[1:], 1):  # Skip root model name
                if (
                    i >= 2 and part.lower() in producer_lower
                ):  # Match deeper hierarchy elements
                    # Check it's a meaningful match, not just substring
                    part_lower = part.lower()
                    if (
                        part_lower in producer_lower and len(part_lower) > 3
                    ):  # Avoid matching short substrings
                        return True

        return True

    def _get_semantic_component(self, component_name: str) -> str:
        """Map component names to semantic categories using universal approach."""
        # UNIVERSAL APPROACH: Use structural depth and naming patterns
        # instead of hardcoded architecture-specific names
        component_lower = component_name.lower()

        # Use general naming patterns that apply across architectures
        if any(term in component_lower for term in ["embed", "input"]):
            return "input_processing"
        elif any(term in component_lower for term in ["layer", "block", "stage"]):
            return "processing_layer"
        elif any(
            term in component_lower for term in ["pool", "head", "classifier", "output"]
        ):
            return "output_processing"
        else:
            return "unknown"

    def _inject_tags_into_onnx(self, onnx_path: str):
        """
        Inject tags into ONNX using doc_string field and create sidecar file.

        This implements the ONNX-compliant hybrid approach:
        1. Store tags in node doc_string field (ONNX compliant)
        2. Create JSON sidecar file for tooling and debugging
        """
        from datetime import datetime
        import json

        # Load ONNX model
        onnx_model = onnx.load(onnx_path)

        # 1. Inject tags as node attributes
        nodes_with_tags = 0
        for node in onnx_model.graph.node:
            node_name = node.name or f"{node.op_type}_{hash(str(node))}"

            if node_name in self._tag_mapping:
                node_info = self._tag_mapping[node_name]
                tags = node_info.get("tags", [])

                if tags:
                    # Store hierarchy information in doc_string (ONNX-compliant approach)
                    primary_path = tags[0] if tags else ""
                    hierarchy_info = {
                        "hierarchy_tags": tags,
                        "hierarchy_path": primary_path,
                        "hierarchy_count": len(tags),
                        "hierarchy_method": "parameter_based",
                    }

                    # Use doc_string field for ONNX compliance
                    node.doc_string = json.dumps(hierarchy_info)

                    nodes_with_tags += 1

        # Save enhanced ONNX model
        onnx.save(onnx_model, onnx_path)

        # 2. Create sidecar JSON file
        sidecar_path = onnx_path.replace(".onnx", "_hierarchy.json")
        sidecar_data = {
            "version": "1.0",
            "format": "modelexport_hierarchy",
            "model_path": str(onnx_path),
            "generated_at": datetime.now().isoformat(),
            "exporter": {
                "name": "modelexport",
                "version": "0.1.0",
                "strategy": self.strategy,
            },
            "summary": {
                "total_operations": len(self._tag_mapping),
                "tagged_operations": len(
                    [op for op in self._tag_mapping.values() if op.get("tags", [])]
                ),
                "nodes_with_attributes": nodes_with_tags,
                "unique_tags": len(
                    set(
                        tag
                        for op in self._tag_mapping.values()
                        for tag in op.get("tags", [])
                    )
                ),
            },
            "tag_statistics": self._compute_tag_statistics(),
            "node_tags": self._tag_mapping,
            "schema": {
                "hierarchy_tags": {
                    "type": "repeated string",
                    "description": "List of hierarchical module paths that produced this operation",
                },
                "hierarchy_path": {
                    "type": "string",
                    "description": "Primary hierarchical path (first tag)",
                },
                "hierarchy_count": {
                    "type": "int",
                    "description": "Number of hierarchy tags for this operation",
                },
                "hierarchy_method": {
                    "type": "string",
                    "description": "Method used to assign tags (parameter_based, propagated, etc.)",
                },
            },
        }

        with open(sidecar_path, "w") as f:
            json.dump(sidecar_data, f, indent=2)

        return sidecar_path

    def extract_module_subgraph(self, onnx_path: str, target_module: str) -> Dict[str, Any]:
        """
        Extract complete subgraph for a specific module hierarchy.
        
        This implements the R13 subgraph extraction algorithm using multi-consumer
        tagging results to identify all operations needed for a target module.
        
        Args:
            onnx_path: Path to the tagged ONNX model
            target_module: Module hierarchy path (e.g., "/BertModel/BertEncoder/BertLayer/BertAttention/BertSdpaSelfAttention")
            
        Returns:
            Dictionary containing:
            - module: target module path
            - operations: list of operation names in the subgraph
            - external_inputs: tensors from outside the module
            - internal_tensors: tensors produced within module
            - boundary_operations: operations that provide inputs
            - tensor_mapping: tensor -> consumer modules mapping
        """
        
        # Load ONNX model for analysis
        onnx_model = onnx.load(onnx_path)
        
        # Step 1: Find all operations tagged with target module
        module_operations = set()
        for node_name, node_info in self._tag_mapping.items():
            if target_module in node_info.get('tags', []):
                module_operations.add(node_name)
        
        if not module_operations:
            return {
                'module': target_module,
                'operations': [],
                'external_inputs': [],
                'internal_tensors': [],
                'boundary_operations': [],
                'tensor_mapping': {},
                'error': f'No operations found for module {target_module}'
            }
        
        # Step 2: Collect all tensors used by these operations
        module_tensors = set()
        for node in onnx_model.graph.node:
            node_name = node.name or f"{node.op_type}_{hash(str(node))}"
            if node_name in module_operations:
                module_tensors.update(node.input)
                module_tensors.update(node.output)
        
        # Step 3: Find boundary operations (provide inputs to module)
        boundary_operations = set()
        for node in onnx_model.graph.node:
            node_name = node.name or f"{node.op_type}_{hash(str(node))}"
            if node_name not in module_operations:
                # Check if this operation produces tensors used by module
                for output in node.output:
                    if output in module_tensors:
                        boundary_operations.add(node_name)
        
        # Step 4: Determine external inputs (not produced within subgraph)
        produced_tensors = set()
        for node in onnx_model.graph.node:
            node_name = node.name or f"{node.op_type}_{hash(str(node))}"
            if node_name in module_operations or node_name in boundary_operations:
                produced_tensors.update(node.output)
        
        external_inputs = set()
        for node in onnx_model.graph.node:
            node_name = node.name or f"{node.op_type}_{hash(str(node))}"
            if node_name in module_operations:
                for input_tensor in node.input:
                    if input_tensor not in produced_tensors:
                        external_inputs.add(input_tensor)
        
        # Step 5: Identify internal tensors (produced within module)
        internal_tensors = set()
        for node in onnx_model.graph.node:
            node_name = node.name or f"{node.op_type}_{hash(str(node))}"
            if node_name in module_operations:
                internal_tensors.update(node.output)
        
        # Step 6: Build tensor consumer mapping for this subgraph
        subgraph_tensor_mapping = {}
        if hasattr(self, '_tensor_consumer_mapping'):
            for tensor, consumer_tags in self._tensor_consumer_mapping.items():
                if tensor in module_tensors:
                    subgraph_tensor_mapping[tensor] = list(consumer_tags)
        
        return {
            'module': target_module,
            'operations': list(module_operations),
            'external_inputs': list(external_inputs),
            'internal_tensors': list(internal_tensors),
            'boundary_operations': list(boundary_operations),
            'tensor_mapping': subgraph_tensor_mapping,
            'summary': {
                'total_operations': len(module_operations),
                'boundary_operations': len(boundary_operations),
                'external_dependencies': len(external_inputs),
                'internal_tensors': len(internal_tensors)
            }
        }

    def _compute_tag_statistics(self) -> Dict[str, int]:
        """Compute statistics about tag distribution."""
        tag_counts = {}
        for node_info in self._tag_mapping.values():
            for tag in node_info.get("tags", []):
                tag_counts[tag] = tag_counts.get(tag, 0) + 1
        return tag_counts
