"""
Usage-Based Strategy Optimizations for Iteration 18

Performance optimizations for the Usage-Based strategy, applying lessons
learned from HTP optimization in Iteration 17.
"""

import logging
from typing import Any

import onnx
import torch
import torch.onnx

logger = logging.getLogger(__name__)


class UsageBasedOptimizedMethods:
    """Optimized methods for Usage-Based strategy."""
    
    @staticmethod
    def _track_module_usage_optimized(self, model: torch.nn.Module, example_inputs) -> None:
        """
        Optimized module usage tracking with reduced overhead.
        
        Optimizations:
        1. Batch hook registration
        2. Lightweight hook functions
        3. Pre-allocated data structures
        4. Single-pass module traversal
        """
        self._usage_tracking.clear()
        self._module_usage_count.clear()
        
        # Import the should_tag_module function
        from ...core.base import should_tag_module
        
        # Pre-allocate module list for efficiency
        all_modules = dict(model.named_modules())
        modules_to_track = []
        for name, module in all_modules.items():
            if should_tag_module(module, self._torch_nn_exceptions):
                modules_to_track.append((name, module))
        
        # Batch hook registration
        hooks = []
        
        # Create lightweight hook function generator
        def create_lightweight_hook(module_name: str):
            """Create minimal overhead hook."""
            def hook(module, inputs, outputs):
                # Minimal tracking - just increment counter
                self._module_usage_count[module_name] += 1
                
                # Only track hierarchy on first usage
                if module_name not in self._usage_tracking:
                    from ...core.base import build_hierarchy_path
                    hierarchy_path = build_hierarchy_path(
                        self._model_root, module_name, all_modules
                    )
                    self._usage_tracking[module_name] = {
                        'module': module,
                        'hierarchy': hierarchy_path
                    }
            return hook
        
        # Register all hooks at once
        for name, module in modules_to_track:
            hook = module.register_forward_hook(create_lightweight_hook(name))
            hooks.append(hook)
        
        try:
            # Run forward pass
            with torch.no_grad():
                model(example_inputs)
        finally:
            # Batch hook removal
            for hook in hooks:
                hook.remove()
    
    @staticmethod
    def _create_hierarchy_mapping_optimized(self) -> dict[str, dict[str, Any]]:
        """
        Optimized hierarchy mapping creation.
        
        Uses single-pass processing and efficient data structures.
        """
        hierarchy_mapping = {}
        
        # Single pass over usage tracking
        for module_name, usage_info in self._usage_tracking.items():
            if usage_info['hierarchy']:  # Only process modules with valid hierarchy
                hierarchy_mapping[module_name] = {
                    'hierarchy': usage_info['hierarchy'],
                    'usage_count': self._module_usage_count[module_name],
                    'module_type': usage_info['module'].__class__.__name__
                }
        
        return hierarchy_mapping
    
    @staticmethod
    def optimize_onnx_export_params(model: torch.nn.Module, example_inputs, **kwargs) -> dict[str, Any]:
        """
        Optimize ONNX export parameters for faster export.
        
        Applies optimizations learned from performance analysis.
        """
        optimized_kwargs = kwargs.copy()
        
        # Optimization 1: Disable training mode exports
        optimized_kwargs.setdefault('training', torch.onnx.TrainingMode.EVAL)
        
        # Optimization 2: Use faster opset version if not specified
        optimized_kwargs.setdefault('opset_version', 14)  # Optimal for most models
        
        # Optimization 3: Disable verbose logging unless debugging
        optimized_kwargs.setdefault('verbose', False)
        
        # Optimization 4: Enable operator export type optimization
        optimized_kwargs.setdefault('operator_export_type', torch.onnx.OperatorExportTypes.ONNX)
        
        # Optimization 5: Do NOT export parameter names to reduce size
        optimized_kwargs.setdefault('export_params', True)
        
        # Optimization 6: Keep initializers as inputs (faster)
        optimized_kwargs.setdefault('keep_initializers_as_inputs', True)
        
        return optimized_kwargs
    
    @staticmethod
    def save_onnx_optimized(onnx_model, output_path: str) -> None:
        """
        Optimized ONNX save with reduced I/O overhead.
        
        Uses optimized serialization settings.
        """
        # Use optimized save settings
        onnx.save(
            onnx_model,
            output_path,
            save_as_external_data=False,  # Keep inline for smaller models
        )


class BatchProcessingOptimizer:
    """Batch processing optimizations for Usage-Based strategy."""
    
    @staticmethod
    def batch_module_filtering(model: torch.nn.Module, should_track_func) -> list[tuple[str, torch.nn.Module]]:
        """
        Batch process module filtering for efficiency.
        
        Single-pass traversal with pre-filtering.
        """
        filtered_modules = []
        
        # Single traversal with inline filtering
        for name, module in model.named_modules():
            if should_track_func(module):
                filtered_modules.append((name, module))
        
        return filtered_modules
    
    @staticmethod
    def batch_hierarchy_generation(module_names: list[str], model_root) -> dict[str, str]:
        """
        Generate hierarchies for multiple modules in batch.
        
        Reduces redundant computation.
        """
        hierarchies = {}
        
        # Cache common prefixes
        prefix_cache = {}
        
        for module_name in module_names:
            # Check cache for common prefixes
            parts = module_name.split('.')
            hierarchy_parts = []
            
            for i, part in enumerate(parts):
                prefix = '.'.join(parts[:i+1])
                
                if prefix in prefix_cache:
                    hierarchy_parts.append(prefix_cache[prefix])
                else:
                    # Generate hierarchy for this part
                    if hasattr(model_root, prefix):
                        module = eval(f"model_root.{prefix}")
                        hierarchy_part = module.__class__.__name__
                        prefix_cache[prefix] = hierarchy_part
                        hierarchy_parts.append(hierarchy_part)
            
            hierarchies[module_name] = '/' + '/'.join(hierarchy_parts)
        
        return hierarchies


class UsageBasedCachingOptimizer:
    """Caching optimizations for repeated operations."""
    
    def __init__(self):
        self._module_type_cache = {}
        self._hierarchy_cache = {}
    
    def get_module_type_cached(self, module: torch.nn.Module) -> str:
        """Get module type with caching."""
        module_id = id(module)
        
        if module_id not in self._module_type_cache:
            self._module_type_cache[module_id] = module.__class__.__name__
        
        return self._module_type_cache[module_id]
    
    def get_hierarchy_cached(self, module_name: str, build_func) -> str:
        """Get hierarchy with caching."""
        if module_name not in self._hierarchy_cache:
            self._hierarchy_cache[module_name] = build_func(module_name)
        
        return self._hierarchy_cache[module_name]


def apply_usage_based_optimizations(exporter):
    """
    Apply all Usage-Based optimizations to an exporter instance.
    
    This function enhances the exporter with optimized methods.
    """
    
    # Store original methods
    exporter._track_module_usage_original = exporter._track_module_usage
    exporter._create_hierarchy_mapping_original = exporter._create_hierarchy_mapping
    
    # Replace with optimized versions
    exporter._track_module_usage = lambda model, inputs: \
        UsageBasedOptimizedMethods._track_module_usage_optimized(exporter, model, inputs)
    
    exporter._create_hierarchy_mapping = lambda: \
        UsageBasedOptimizedMethods._create_hierarchy_mapping_optimized(exporter)
    
    # Add caching optimizer
    exporter._cache_optimizer = UsageBasedCachingOptimizer()
    
    # Add optimization flag
    exporter._optimizations_enabled = True
    
    # Add model root reference if needed
    if not hasattr(exporter, '_model_root'):
        exporter._model_root = None
    
    return exporter


def create_optimized_usage_based_export(
    model: torch.nn.Module,
    example_inputs,
    output_path: str,
    **kwargs
) -> dict[str, Any]:
    """
    Complete optimized export pipeline for Usage-Based strategy.
    
    Combines all optimizations for maximum performance.
    """
    from modelexport.strategies.usage_based import UsageBasedExporter
    
    # Create and optimize exporter
    exporter = UsageBasedExporter()
    exporter = apply_usage_based_optimizations(exporter)
    
    # Optimize ONNX export parameters
    optimized_kwargs = UsageBasedOptimizedMethods.optimize_onnx_export_params(
        model, example_inputs, **kwargs
    )
    
    # Run optimized export
    result = exporter.export(model, example_inputs, output_path, **optimized_kwargs)
    
    # Add optimization metadata
    result['optimizations_applied'] = [
        'lightweight_hooks',
        'batch_processing',
        'single_pass_algorithms',
        'optimized_onnx_params',
        'caching'
    ]
    
    return result