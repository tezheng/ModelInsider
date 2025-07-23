"""
Unified Export Interface for ModelExport

This module provides a high-level, user-friendly interface for exporting PyTorch
models to ONNX with automatic strategy selection and optimization.
"""

import logging
from pathlib import Path
from typing import Any

import torch

from .core.strategy_selector import (
    ExportStrategy,
    select_best_strategy,
)
from .core.unified_optimizer import (
    PerformanceMonitor,
    create_optimized_exporter,
)

logger = logging.getLogger(__name__)


class UnifiedExporter:
    """
    High-level interface for unified model export with intelligent defaults.
    
    Features:
    - Automatic strategy selection based on model analysis
    - Built-in optimizations from iterations 17-18
    - Performance monitoring and reporting
    - Fallback handling for maximum reliability
    """
    
    def __init__(
        self,
        strategy: str | ExportStrategy = ExportStrategy.AUTO,
        enable_optimizations: bool = True,
        enable_monitoring: bool = True,
        verbose: bool = False
    ):
        """
        Initialize unified exporter.
        
        Args:
            strategy: Export strategy to use (default: AUTO for automatic selection)
            enable_optimizations: Apply performance optimizations (default: True)
            enable_monitoring: Enable performance monitoring (default: True)
            verbose: Enable verbose logging (default: False)
        """
        self.strategy = ExportStrategy(strategy) if isinstance(strategy, str) else strategy
        self.enable_optimizations = enable_optimizations
        self.enable_monitoring = enable_monitoring
        self.verbose = verbose
        
        if verbose:
            logging.getLogger('modelexport').setLevel(logging.DEBUG)
        
        self.monitor = PerformanceMonitor() if enable_monitoring else None
        self.last_export_report = None
    
    def export(
        self,
        model: torch.nn.Module,
        example_inputs: torch.Tensor | tuple | dict,
        output_path: str | Path,
        **kwargs
    ) -> dict[str, Any]:
        """
        Export a PyTorch model to ONNX with intelligent strategy selection.
        
        Args:
            model: PyTorch model to export
            example_inputs: Example inputs for tracing
            output_path: Path to save ONNX model
            **kwargs: Additional arguments for torch.onnx.export
            
        Returns:
            Export report with detailed information
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Start monitoring
        if self.monitor:
            export_timer = self.monitor.time_operation("total_export")
            @export_timer
            def monitored_export():
                return self._perform_export(model, example_inputs, output_path, **kwargs)
            
            report = monitored_export()
        else:
            report = self._perform_export(model, example_inputs, output_path, **kwargs)
        
        self.last_export_report = report
        return report
    
    def _perform_export(
        self,
        model: torch.nn.Module,
        example_inputs: torch.Tensor | tuple | dict,
        output_path: Path,
        **kwargs
    ) -> dict[str, Any]:
        """Perform the actual export with strategy selection and optimization."""
        
        report = {
            "output_path": str(output_path),
            "model_info": {
                "type": model.__class__.__name__,
                "module_count": sum(1 for _ in model.named_modules())
            },
            "strategy_selection": {},
            "export_result": {},
            "performance_metrics": {},
            "warnings": [],
            "errors": []
        }
        
        try:
            # Step 1: Strategy selection
            if self.strategy == ExportStrategy.AUTO:
                strategy, recommendation = select_best_strategy(
                    model,
                    example_inputs,
                    prioritize_speed=True
                )
                report["strategy_selection"] = {
                    "selected": strategy.value,
                    "reasoning": recommendation.reasoning,
                    "confidence": recommendation.confidence,
                    "expected_performance": recommendation.expected_performance
                }
                
                if recommendation.warnings:
                    report["warnings"].extend(recommendation.warnings)
            else:
                strategy = self.strategy
                report["strategy_selection"] = {
                    "selected": strategy.value,
                    "reasoning": ["User specified strategy"],
                    "confidence": 1.0
                }
            
            # Step 2: Create exporter with optimizations
            if self.enable_optimizations:
                exporter = create_optimized_exporter(strategy.value, **kwargs)
                report["optimizations_applied"] = exporter._optimization_profile.optimizations_applied
            else:
                # Create unoptimized exporter
                exporter = self._create_basic_exporter(strategy.value, **kwargs)
                report["optimizations_applied"] = []
            
            # Step 3: Perform export with fallback handling
            export_success = False
            strategies_tried = []
            
            # Try primary strategy
            try:
                if self.monitor:
                    @self.monitor.time_operation(f"export_{strategy.value}")
                    def export_with_monitoring():
                        return exporter.export(model, example_inputs, str(output_path), **kwargs)
                    
                    export_result = export_with_monitoring()
                else:
                    export_result = exporter.export(model, example_inputs, str(output_path), **kwargs)
                
                export_success = True
                strategies_tried.append((strategy.value, "success"))
                report["export_result"] = export_result
                
            except Exception as e:
                strategies_tried.append((strategy.value, f"failed: {str(e)}"))
                logger.warning(f"Primary strategy {strategy.value} failed: {e}")
                
                # Try fallback strategy if available
                if strategy == ExportStrategy.FX:
                    # FX failed, try Usage-Based as fallback
                    logger.info("Trying Usage-Based strategy as fallback...")
                    
                    fallback_exporter = create_optimized_exporter("usage_based")
                    try:
                        export_result = fallback_exporter.export(
                            model, example_inputs, str(output_path), **kwargs
                        )
                        export_success = True
                        strategies_tried.append(("usage_based", "success"))
                        report["export_result"] = export_result
                        report["warnings"].append(f"Fell back from {strategy.value} to usage_based")
                    except Exception as e2:
                        strategies_tried.append(("usage_based", f"failed: {str(e2)}"))
                        report["errors"].append(f"All strategies failed: {strategies_tried}")
                        raise RuntimeError(f"Export failed with all strategies: {strategies_tried}")
                else:
                    report["errors"].append(f"Export failed: {str(e)}")
                    raise
            
            # Step 4: Collect performance metrics
            if self.monitor:
                report["performance_metrics"] = self.monitor.get_metrics()
            
            # Step 5: Add export summary
            report["summary"] = {
                "success": export_success,
                "strategies_tried": strategies_tried,
                "final_strategy": strategies_tried[-1][0] if strategies_tried else None,
                "export_time": report["performance_metrics"].get("total_export_avg", 0),
                "file_size": output_path.stat().st_size if output_path.exists() else 0
            }
            
            # Log summary
            if export_success:
                logger.info(f"Export successful using {report['summary']['final_strategy']} strategy")
                logger.info(f"Output saved to: {output_path}")
                if self.verbose:
                    logger.info(f"Export time: {report['summary']['export_time']:.3f}s")
                    logger.info(f"File size: {report['summary']['file_size'] / 1024 / 1024:.2f} MB")
            
        except Exception as e:
            report["errors"].append(str(e))
            report["summary"] = {"success": False, "error": str(e)}
            raise
        
        return report
    
    def _create_basic_exporter(self, strategy: str, **kwargs):
        """Create a basic exporter without optimizations."""
        if strategy == "usage_based":
            from .strategies.usage_based import UsageBasedExporter
            return UsageBasedExporter(**kwargs)
        elif strategy == "htp":
            from .strategies.htp_new import HTPExporter
            return HTPExporter(**kwargs)
        elif strategy in ["fx_graph", "fx"]:
            from .strategies.fx import FXHierarchyExporter
            return FXHierarchyExporter(**kwargs)
        else:
            raise ValueError(f"Unknown strategy: {strategy}")
    
    def get_performance_report(self) -> dict[str, Any] | None:
        """Get detailed performance report from last export."""
        return self.last_export_report
    
    def benchmark_strategies(
        self,
        model: torch.nn.Module,
        example_inputs: torch.Tensor | tuple | dict,
        strategies: list[str] | None = None
    ) -> dict[str, Any]:
        """
        Benchmark multiple strategies on the same model.
        
        Args:
            model: Model to benchmark
            example_inputs: Example inputs
            strategies: List of strategies to test (default: all compatible)
            
        Returns:
            Benchmark results comparing strategies
        """
        import tempfile
        import time
        
        if strategies is None:
            # Determine compatible strategies
            _, recommendation = select_best_strategy(model, example_inputs)
            strategies = ["usage_based", "htp"]  # Always test these
            
            # Only test FX if not warned against
            if not any("FX" in w for w in recommendation.warnings):
                strategies.append("fx_graph")
        
        results = {
            "model": model.__class__.__name__,
            "strategies": {},
            "ranking": []
        }
        
        for strategy in strategies:
            logger.info(f"Benchmarking {strategy} strategy...")
            
            try:
                with tempfile.NamedTemporaryFile(suffix='.onnx', delete=False) as tmp:
                    # Create new exporter for each test
                    exporter = UnifiedExporter(
                        strategy=strategy,
                        enable_optimizations=self.enable_optimizations,
                        enable_monitoring=True,
                        verbose=False
                    )
                    
                    start_time = time.time()
                    report = exporter.export(model, example_inputs, tmp.name)
                    elapsed = time.time() - start_time
                    
                    results["strategies"][strategy] = {
                        "success": report["summary"]["success"],
                        "export_time": elapsed,
                        "file_size": report["summary"]["file_size"],
                        "optimizations": report.get("optimizations_applied", [])
                    }
                    
                    # Clean up
                    Path(tmp.name).unlink(missing_ok=True)
                    
            except Exception as e:
                results["strategies"][strategy] = {
                    "success": False,
                    "error": str(e),
                    "export_time": None
                }
        
        # Rank strategies by speed
        successful_strategies = [
            (name, data["export_time"]) 
            for name, data in results["strategies"].items() 
            if data["success"]
        ]
        successful_strategies.sort(key=lambda x: x[1])
        results["ranking"] = [s[0] for s in successful_strategies]
        
        return results


def export_model(
    model: torch.nn.Module,
    example_inputs: torch.Tensor | tuple | dict,
    output_path: str | Path,
    strategy: str | ExportStrategy = "auto",
    optimize: bool = True,
    verbose: bool = False,
    **kwargs
) -> dict[str, Any]:
    """
    Convenience function to export a model with intelligent defaults.
    
    This is the recommended entry point for most users.
    
    Args:
        model: PyTorch model to export
        example_inputs: Example inputs for tracing
        output_path: Path to save ONNX model
        strategy: Export strategy (default: "auto" for automatic selection)
        optimize: Apply performance optimizations (default: True)
        verbose: Enable verbose output (default: False)
        **kwargs: Additional arguments for torch.onnx.export
        
    Returns:
        Export report with details
        
    Example:
        >>> import modelexport
        >>> report = modelexport.export_model(
        ...     model,
        ...     torch.randn(1, 3, 224, 224),
        ...     "model.onnx"
        ... )
        >>> print(f"Exported using {report['summary']['final_strategy']} strategy")
    """
    exporter = UnifiedExporter(
        strategy=strategy,
        enable_optimizations=optimize,
        enable_monitoring=True,
        verbose=verbose
    )
    
    return exporter.export(model, example_inputs, output_path, **kwargs)