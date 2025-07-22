"""
HTP Export Monitor - Restored IO/ABC Design.

This module provides the main orchestrator for HTP export monitoring,
coordinating multiple writers using the decorator pattern.
"""

from __future__ import annotations

import contextlib
import io
import time
from pathlib import Path

from rich.console import Console

from .base_writer import ExportData, ExportStep, StepAwareWriter
from .console_writer import ConsoleWriter
from .metadata_writer import MetadataWriter
from .markdown_report_writer import MarkdownReportWriter
from .step_data import (
    HierarchyData,
    InputGenData,
    ModelPrepData,
    ModuleInfo,
    NodeTaggingData,
    ONNXExportData,
    TagInjectionData,
    TensorInfo,
)


class HTPExportMonitor:
    """
    Central monitor that coordinates data updates and writer dispatch.
    
    This class maintains backward compatibility with the existing API
    while using the new IO/ABC-based writer architecture internally.
    """
    
    def __init__(
        self,
        output_path: str,
        model_name: str = "",
        verbose: bool = True,
        enable_report: bool = True,
        embed_hierarchy: bool = True,
    ):
        """
        Initialize the export monitor.
        
        Args:
            output_path: Path for the output ONNX model
            model_name: Name of the model being exported
            verbose: Whether to output verbose console information
            enable_report: Whether to generate a text report
            embed_hierarchy: Whether to embed hierarchy tags in ONNX
        """
        self.output_path = output_path
        self.model_name = model_name
        self.verbose = verbose
        self.enable_report = enable_report
        self.embed_hierarchy = embed_hierarchy
        
        # Initialize shared data
        self.data = ExportData(
            model_name=model_name,
            output_path=output_path,
            embed_hierarchy=embed_hierarchy,
        )
        
        # Initialize writers
        self.writers: list[StepAwareWriter] = []
        
        # Console writer (output to terminal when verbose)
        if verbose:
            self.console_writer = ConsoleWriter()
            self.writers.append(self.console_writer)
        else:
            self.console_writer = None
        
        # Metadata writer (always enabled)
        self.metadata_writer = MetadataWriter(output_path)
        self.writers.append(self.metadata_writer)
        
        # Report writer (optional) - now using MarkdownReportWriter
        if enable_report:
            self.report_writer = MarkdownReportWriter(output_path)
            self.writers.append(self.report_writer)
        else:
            self.report_writer = None
        
        # For backward compatibility - store as attributes
        self.console = self.console_writer.console if self.console_writer else Console()
        
        # Track start time
        self._start_time = time.time()
    
    def update(self, step: ExportStep, **kwargs) -> None:
        """
        Update monitoring with step data.
        
        This method maintains backward compatibility by accepting
        the step enum and arbitrary kwargs, then converting them
        to the appropriate typed data structures.
        
        Args:
            step: The current export step
            **kwargs: Step-specific data
        """
        # Update typed step data based on the step
        self._update_step_data(step, kwargs)
        
        # Notify all writers
        for writer in self.writers:
            try:
                writer.write(step, self.data)
            except Exception as e:
                print(f"Error in {writer.__class__.__name__}: {e}")
    
    def _update_step_data(self, step: ExportStep, kwargs: dict) -> None:
        """Convert kwargs to typed step data."""
        if step == ExportStep.MODEL_PREP:
            self.data.model_prep = ModelPrepData(
                model_class=kwargs.get("model_class", ""),
                total_modules=kwargs.get("total_modules", 0),
                total_parameters=kwargs.get("total_parameters", 0),
            )
        
        elif step == ExportStep.INPUT_GEN:
            # Convert input dict to TensorInfo objects
            inputs = {}
            if "inputs" in kwargs:
                for name, info in kwargs["inputs"].items():
                    if isinstance(info, dict):
                        inputs[name] = TensorInfo(
                            shape=info.get("shape", []),
                            dtype=info.get("dtype", "float32"),
                        )
            
            self.data.input_gen = InputGenData(
                method=kwargs.get("method", "auto_generated"),
                model_type=kwargs.get("model_type"),
                task=kwargs.get("task"),
                inputs=inputs,
            )
        
        elif step == ExportStep.HIERARCHY:
            # Convert hierarchy dict to ModuleInfo objects
            hierarchy = {}
            if "hierarchy" in kwargs:
                for path, info in kwargs["hierarchy"].items():
                    if isinstance(info, dict):
                        hierarchy[path] = ModuleInfo(
                            class_name=info.get("class_name", "Unknown"),
                            traced_tag=info.get("traced_tag", ""),
                            execution_order=info.get("execution_order"),
                            source=info.get("source"),
                        )
            
            self.data.hierarchy = HierarchyData(
                hierarchy=hierarchy,
                execution_steps=kwargs.get("execution_steps", 0),
                module_list=kwargs.get("module_list", []),
            )
        
        elif step == ExportStep.ONNX_EXPORT:
            self.data.onnx_export = ONNXExportData(
                opset_version=kwargs.get("opset_version", 17),
                do_constant_folding=kwargs.get("do_constant_folding", True),
                verbose=kwargs.get("verbose", False),
                input_names=kwargs.get("input_names", []),
                output_names=kwargs.get("output_names"),
                onnx_size_mb=kwargs.get("onnx_size_mb", 0.0),
            )
        
        elif step == ExportStep.NODE_TAGGING:
            self.data.node_tagging = NodeTaggingData(
                total_nodes=kwargs.get("total_nodes", 0),
                tagged_nodes=kwargs.get("tagged_nodes", {}),
                tagging_stats=kwargs.get("tagging_stats", {}),
                coverage=kwargs.get("coverage", 0.0),
                op_counts=kwargs.get("op_counts", {}),
            )
        
        elif step == ExportStep.TAG_INJECTION:
            self.data.tag_injection = TagInjectionData(
                tags_injected=self.embed_hierarchy,
                tags_stripped=not self.embed_hierarchy,
            )
    
    def _print_summary(self) -> None:
        """Print export summary to console."""
        console = self.console
        
        console.print("\n" + "=" * ConsoleWriter.SEPARATOR_LENGTH)
        console.print("✅ [bold green]EXPORT COMPLETE[/bold green]")
        console.print("=" * ConsoleWriter.SEPARATOR_LENGTH)
        
        # Summary stats
        total_time = self.data.export_time
        traced_modules = len(self.data.hierarchy.hierarchy) if self.data.hierarchy else 0
        total_modules = self.data.model_prep.total_modules if self.data.model_prep else 0
        nodes = self.data.node_tagging.total_nodes if self.data.node_tagging else 0
        tagged = len(self.data.node_tagging.tagged_nodes) if self.data.node_tagging else 0
        coverage = self.data.node_tagging.coverage if self.data.node_tagging else 0.0
        
        console.print("📊 Export Summary:")
        console.print(f"   • Total time: [bold cyan]{total_time:.2f}s[/bold cyan]")
        console.print(f"   • Hierarchy modules: [bold cyan]{total_modules}[/bold cyan]")
        console.print(f"   • Traced modules: [bold cyan]{traced_modules}/{total_modules}[/bold cyan]")
        console.print(f"   • ONNX nodes: [bold cyan]{nodes}[/bold cyan]")
        console.print(
            f"   • Tagged nodes: [bold cyan]{tagged}[/bold cyan] "
            f"([bold cyan]{coverage:.1f}%[/bold cyan] coverage)"
        )
        
        console.print("\n📁 Output files:")
        console.print(f"   • ONNX model: [bold magenta]{self.output_path}[/bold magenta]")
        
        # Metadata is always written
        base_path = Path(self.output_path).with_suffix("")
        metadata_path = f"{base_path}_htp_metadata.json"
        console.print(f"   • Metadata: [bold magenta]{metadata_path}[/bold magenta]")
        
        if self.enable_report:
            report_path = f"{base_path}_htp_export_report.md"
            console.print(f"   • Report: [bold magenta]{report_path}[/bold magenta]")
    
    def __enter__(self):
        """Context manager entry."""
        # Print header if verbose
        if self.verbose and self.console_writer:
            self._print_header()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit - finalize all writers."""
        if exc_type is None:
            # Success - calculate final export time and finalize
            self.data.export_time = self.data.elapsed_time
            
            # Print summary if verbose
            if self.verbose and self.console_writer:
                self._print_summary()
            
            # Close all writers normally
            for writer in self.writers:
                try:
                    writer.close()
                except Exception as e:
                    print(f"Error closing {writer.__class__.__name__}: {e}")
        else:
            # Error - still try to close writers
            for writer in self.writers:
                with contextlib.suppress(Exception):
                    writer.close()
        
        # Always add empty line at the end for visual separation
        if self.verbose and self.console_writer:
            self.console.print()
    
    def _print_header(self) -> None:
        """Print export header."""
        console = self.console
        
        console.print("\n" + "=" * ConsoleWriter.SEPARATOR_LENGTH)
        console.print("🚀 [bold cyan]HTP ONNX EXPORT PROCESS[/bold cyan]")
        console.print("=" * ConsoleWriter.SEPARATOR_LENGTH)
        
        # Timestamp
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
        console.print(f"📅 Export Time: [bold green]{timestamp}[/bold green]")
        console.print(f"🔄 Loading model and exporting: [bold magenta]{self.model_name}[/bold magenta]")
        
        # Strategy info
        if self.embed_hierarchy:
            console.print(
                "🎯 Strategy: [bold cyan]HTP[/bold cyan] (Hierarchical Tracing and Projection)"
            )
        else:
            console.print(
                "🎯 Strategy: [bold cyan]HTP[/bold cyan] (Hierarchical Tracing and Projection) - "
                "[bold red]DISABLED[/bold red] (--clean-onnx)"
            )
        console.print("=" * ConsoleWriter.SEPARATOR_LENGTH)