"""
Tracing-based HF hierarchy builder using forward hooks.

This version builds module hierarchy lazily during tracing to avoid
including unused modules in the final hierarchy. Only processes modules
that are actually executed (18 vs 48 total modules for BERT-tiny).

Key insight: Execution order IS hierarchy order - parents always execute before children!
"""
from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn

from .base import should_include_in_hierarchy


class TracingHierarchyBuilder:
    """
    Tracing-based hierarchy builder that builds hierarchy lazily.

    Key improvements:
    - Only includes modules that are actually executed (18 vs 48 modules for BERT-tiny)
    - Leverages natural execution order (parents execute before children)
    - Ultra-simple implementation using tag stack
    - No complex parent tracking needed
    """

    def __init__(self, exceptions: list[str] | None = None):
        """
        Initialize the tracing hierarchy builder.
        
        Args:
            exceptions: List of torch.nn class names to include in hierarchy despite being
                       torch.nn modules. Passed to should_include_in_hierarchy.
                       Example: ["Conv2d", "BatchNorm2d"] to include these in hierarchy.
        """
        self.tag_stack = []
        self.execution_trace = []
        self.operation_context = {}
        self.hooks = []
        self.module_hierarchy = {}  # Only populated for executed modules
        self.traced_modules = set()  # Track which modules were traced
        self.exceptions = exceptions  # torch.nn exceptions to include
        self.model_outputs = None  # Store model outputs from execution

    def is_hf_class(self, module: nn.Module) -> bool:
        """Check if a module is a HuggingFace class - UNIVERSAL."""
        module_path = module.__class__.__module__
        return module_path.startswith("transformers")

    def should_create_hierarchy_level(self, module: nn.Module) -> bool:
        """
        Determine if module should create a new hierarchy level - UNIVERSAL.

        CARDINAL RULE #5 (MUST-002): Filter torch.nn modules from hierarchy
        Only include semantically important modules in hierarchy structure.
        """
        # Use should_include_in_hierarchy to filter torch.nn infrastructure modules
        # This ensures MUST-002 compliance - no torch.nn classes in hierarchy by default
        # But allows exceptions for specific cases where torch.nn modules are needed
        return should_include_in_hierarchy(module, exceptions=self.exceptions)

    def create_pre_hook(self, module_name: str, module: nn.Module):
        """Create pre-forward hook - ultra simple version."""

        def pre_hook(module_ref, inputs):
            # Extract class name and check for index
            class_name = module.__class__.__name__
            name_parts = module_name.split(".") if module_name else []

            # Handle indexed modules (e.g., layer.0)
            if name_parts and name_parts[-1].isdigit():
                current_class_name = f"{class_name}.{name_parts[-1]}"
            else:
                current_class_name = class_name

            # Build hierarchical tag
            if self.tag_stack:  # Has parent
                parent_tag = self.tag_stack[-1]
                hierarchical_tag = f"{parent_tag}/{current_class_name}"
            else:  # Root module
                hierarchical_tag = f"/{current_class_name}"
            
            self.tag_stack.append(hierarchical_tag)

            # Add to hierarchy (parents guaranteed to exist due to execution order!)
            if module_name not in self.module_hierarchy:
                # Determine module type universally
                module_type = "huggingface" if self.is_hf_class(module) else "pytorch"
                
                self.module_hierarchy[module_name] = {
                    "name": module_name,
                    "class_name": class_name,
                    "module_type": module_type,  # Universal module typing
                    "traced_tag": hierarchical_tag,
                    "execution_order": sum(
                        1
                        for trace in self.execution_trace
                        if trace["action"] == "enter"
                    ),  # Count only "enter" actions for execution order
                }

            # Record execution trace
            self.execution_trace.append(
                {
                    "module_name": module_name,
                    "tag": hierarchical_tag,
                    "action": "enter",
                    "stack_depth": len(self.tag_stack),
                }
            )

            # Mark as traced
            self.traced_modules.add(module_name)

            # Operation context for compatibility
            self.operation_context[module_name] = {
                "tag": hierarchical_tag,
                "module_class": class_name,
            }

        return pre_hook

    def create_post_hook(self, module_name: str):
        """Create post-forward hook - record exit and pop the stack."""

        def post_hook(module, inputs, outputs):
            # Record exit action
            if self.tag_stack:
                self.execution_trace.append(
                    {
                        "module_name": module_name,
                        "tag": self.tag_stack[-1],
                        "action": "exit",
                        "stack_depth": len(self.tag_stack),
                    }
                )
                self.tag_stack.pop()

        return post_hook

    def register_hooks(self, model: nn.Module) -> None:
        """Register hooks - ultra simple version."""
        # Initialize with empty tag stack (root will add itself)
        self.tag_stack = []

        # Register hooks on ALL PyTorch modules including root - UNIVERSAL APPROACH
        for name, module in model.named_modules():
            if self.should_create_hierarchy_level(module):  # Include ALL nn.Module instances
                pre_hook = module.register_forward_pre_hook(
                    self.create_pre_hook(name, module)
                )
                self.hooks.append(pre_hook)

                post_hook = module.register_forward_hook(self.create_post_hook(name))
                self.hooks.append(post_hook)

    def remove_hooks(self) -> None:
        """Remove all registered hooks."""
        for hook in self.hooks:
            hook.remove()
        self.hooks.clear()

    def trace_model_execution(
        self, model: nn.Module, example_inputs: tuple[torch.Tensor, ...]
    ) -> None:
        """Trace model execution to build hierarchy mapping - UNIVERSAL."""
        self.register_hooks(model)

        try:
            # Run model forward pass to trigger hooks
            model.eval()
            with torch.no_grad():
                # Handle both dict inputs (keyword args) and list/tuple inputs (positional args)
                if isinstance(example_inputs, dict):
                    self.model_outputs = model(**example_inputs)
                else:
                    self.model_outputs = model(*example_inputs)
        finally:
            self.remove_hooks()

    def get_hierarchy_mapping(self) -> dict[str, str]:
        """Get the traced hierarchy mapping."""
        hierarchy_mapping = {}

        for module_name, context in self.operation_context.items():
            hierarchy_mapping[module_name] = context["tag"]

        return hierarchy_mapping

    def get_complete_hierarchy(self) -> dict[str, dict[str, Any]]:
        """
        Get the complete module hierarchy with traced tags.

        Optimized version: Only includes modules that were actually executed.
        """
        # Add expected_tag for backward compatibility if needed
        result = {}
        for module_name, metadata in self.module_hierarchy.items():
            result[module_name] = metadata.copy()
            # Add expected_tag for backward compatibility
            result[module_name]["expected_tag"] = metadata.get("traced_tag")
        
        return result

    def get_execution_summary(self) -> dict[str, Any]:
        """Get execution summary."""
        return {
            "total_modules_traced": len(self.traced_modules),
            "total_modules": len(self.module_hierarchy),
            "execution_steps": len(self.execution_trace),
            "hierarchy_mapping": self.get_hierarchy_mapping(),
            "module_hierarchy": self.get_complete_hierarchy(),
        }

    def get_outputs(self) -> Any:
        """
        Get the model outputs captured during trace_model_execution.
        
        Returns:
            The model outputs from the last execution, or None if not executed yet.
        """
        return self.model_outputs

    def clear(self) -> None:
        """Clear all internal state."""
        self.tag_stack.clear()
        self.execution_trace.clear()
        self.operation_context.clear()
        self.module_hierarchy.clear()
        self.traced_modules.clear()
        self.remove_hooks()
        self.model_outputs = None
