"""
Usage-based Hierarchy Exporter (Legacy)

This module implements the original usage-based tagging strategy for backward
compatibility and baseline comparisons. It captures hierarchy based on module
usage during execution using a simpler approach than modern FX/HTP strategies.

Note: This strategy is maintained for compatibility. Consider using FX or HTP
strategies for new applications as they provide better coverage and accuracy.
"""

from __future__ import annotations

import json
import logging
from collections import defaultdict
from typing import Any

import onnx
import torch
import torch.onnx

from ...core.base import BaseHierarchyExporter, build_hierarchy_path, should_tag_module
from ...core.onnx_utils import ONNXUtils

logger = logging.getLogger(__name__)


class UsageBasedExporter(BaseHierarchyExporter):
    """
    Legacy usage-based hierarchy-preserving ONNX exporter.
    
    This strategy uses basic hook-based module usage tracking during model
    execution to capture hierarchy information. It's simpler than FX/HTP
    but provides lower coverage and accuracy.
    """
    
    def __init__(self, torch_nn_exceptions: list[str] | None = None):
        """
        Initialize usage-based exporter.
        
        Args:
            torch_nn_exceptions: List of torch.nn modules to include in hierarchy
        """
        super().__init__()
        self._torch_nn_exceptions = torch_nn_exceptions or []  # MUST-002: No torch.nn classes should appear in hierarchy tags
        self._usage_tracking: dict[str, Any] = {}
        self._module_usage_count: dict[str, int] = defaultdict(int)
        
    def export(
        self,
        model: torch.nn.Module,
        example_inputs: torch.Tensor | tuple | dict,
        output_path: str,
        **kwargs
    ) -> dict[str, Any]:
        """
        Export PyTorch model to ONNX with usage-based hierarchy preservation.
        
        Args:
            model: PyTorch model to export
            example_inputs: Example inputs for tracing
            output_path: Path to save ONNX model
            **kwargs: Additional arguments for torch.onnx.export
            
        Returns:
            Export metadata with hierarchy information
        """
        logger.info("Starting usage-based hierarchy-preserving ONNX export")
        
        self._model_root = model
        model.eval()
        
        # Step 1: Track module usage during forward pass
        logger.info("Phase 1: Tracking module usage")
        self._track_module_usage(model, example_inputs)
        
        # Step 2: Standard ONNX export
        logger.info("Phase 2: Standard ONNX export")
        torch.onnx.export(
            model,
            example_inputs,
            output_path,
            **kwargs
        )
        
        # Step 3: Create basic hierarchy mapping
        logger.info("Phase 3: Creating hierarchy mapping")
        hierarchy_mapping = self._create_hierarchy_mapping()
        
        # Step 4: Inject hierarchy into ONNX model
        logger.info("Phase 4: Injecting hierarchy metadata")
        onnx_model = ONNXUtils.load_and_validate(output_path)
        tagged_count = ONNXUtils.inject_hierarchy_metadata(
            onnx_model, hierarchy_mapping, "usage_based"
        )
        
        # Save enhanced model
        onnx.save(onnx_model, output_path)
        
        # Step 5: Create sidecar file
        sidecar_path = ONNXUtils.create_sidecar_file(
            output_path, hierarchy_mapping, {
                "strategy": "usage_based",
                "torch_nn_exceptions": self._torch_nn_exceptions,
                "usage_stats": self._module_usage_count
            }
        )
        
        # Build results
        results = {
            'onnx_path': output_path,
            'sidecar_path': sidecar_path,
            'strategy': 'usage_based',
            'hierarchy_nodes': tagged_count,
            'unique_modules': len(set(self._usage_tracking.keys())),
            'topology_preserved': True,
            'usage_stats': dict(self._module_usage_count)
        }
        
        self._export_stats = results
        logger.info("Usage-based hierarchy export completed successfully")
        return results
    
    def _track_module_usage(
        self, 
        model: torch.nn.Module, 
        example_inputs: Any
    ) -> None:
        """Track which modules are used during forward pass."""
        self._usage_tracking.clear()
        self._module_usage_count.clear()
        
        # Register hooks on all modules
        hooks = []
        all_modules = dict(model.named_modules())
        
        for name, module in all_modules.items():
            if should_tag_module(module, self._torch_nn_exceptions):
                # Create hook for this module
                def create_usage_hook(module_name: str, module_ref: torch.nn.Module):
                    def usage_hook(module, inputs, outputs):
                        # Record module usage
                        self._module_usage_count[module_name] += 1
                        
                        if module_name not in self._usage_tracking:
                            hierarchy_path = build_hierarchy_path(
                                self._model_root, module_name, all_modules
                            )
                            self._usage_tracking[module_name] = {
                                "hierarchy_path": hierarchy_path,
                                "module_class": module_ref.__class__.__name__,
                                "usage_count": 0
                            }
                        
                        self._usage_tracking[module_name]["usage_count"] += 1
                    
                    return usage_hook
                
                hook = module.register_forward_hook(create_usage_hook(name, module))
                hooks.append(hook)
        
        try:
            # Run forward pass to trigger hooks
            with torch.no_grad():
                if isinstance(example_inputs, dict):
                    model(**example_inputs)
                elif isinstance(example_inputs, (list, tuple)):
                    model(*example_inputs)
                else:
                    model(example_inputs)
        finally:
            # Remove hooks
            for hook in hooks:
                hook.remove()
    
    def _create_hierarchy_mapping(self) -> dict[str, dict[str, Any]]:
        """Create hierarchy mapping from usage tracking."""
        hierarchy_mapping = {}
        
        # Create synthetic node mapping based on usage
        for i, (module_name, usage_info) in enumerate(self._usage_tracking.items()):
            # Create synthetic node name (usage-based strategy doesn't have direct ONNX mapping)
            node_name = f"usage_node_{i}_{module_name.replace('.', '_')}"
            
            hierarchy_mapping[node_name] = {
                "tags": [usage_info["hierarchy_path"]],
                "primary_path": usage_info["hierarchy_path"],
                "op_type": "unknown",  # Usage-based doesn't track operation types
                "source": "module_usage",
                "usage_count": usage_info["usage_count"],
                "module_class": usage_info["module_class"]
            }
        
        return hierarchy_mapping
    
    def extract_subgraph(
        self, 
        onnx_path: str, 
        target_module: str
    ) -> dict[str, Any]:
        """
        Extract subgraph for specific module hierarchy (limited implementation).
        
        Note: Usage-based strategy has limited subgraph extraction capabilities
        compared to FX/HTP strategies due to lack of detailed operation mapping.
        """
        logger.warning("Usage-based strategy has limited subgraph extraction capabilities")
        
        # Basic implementation that identifies related nodes
        sidecar_path = onnx_path.replace('.onnx', '_hierarchy.json')
        
        try:
            with open(sidecar_path) as f:
                sidecar_data = json.load(f)
            
            node_tags = sidecar_data.get("node_tags", {})
            matching_nodes = []
            
            for node_name, node_info in node_tags.items():
                tags = node_info.get("tags", [])
                if any(target_module in tag for tag in tags):
                    matching_nodes.append(node_name)
            
            return {
                'target_module': target_module,
                'strategy': 'usage_based',
                'matching_nodes': matching_nodes,
                'operation_count': len(matching_nodes),
                'note': 'Limited subgraph extraction - consider using FX or HTP strategies'
            }
            
        except FileNotFoundError:
            return {
                'target_module': target_module,
                'strategy': 'usage_based', 
                'error': 'Sidecar file not found',
                'matching_nodes': []
            }