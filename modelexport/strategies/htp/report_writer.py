"""
Report writer for HTP export monitoring.

This module provides full text report writing with complete console capture,
no truncation, and all module/node details.
"""

from __future__ import annotations

import io
import re
import time
from pathlib import Path

from .base_writer import ExportData, ExportStep, StepAwareWriter, step


class ReportWriter(StepAwareWriter):
    """Full text report writer with complete console capture."""
    
    def __init__(self, output_path: str, console_buffer: io.StringIO | None = None):
        """
        Initialize report writer.
        
        Args:
            output_path: Base output path for the ONNX model
            console_buffer: Buffer containing console output to include
        """
        super().__init__()
        self.output_path = Path(output_path).with_suffix("").as_posix()
        self.report_path = f"{self.output_path}_htp_export_report.txt"
        self.buffer = io.StringIO()
        self.console_buffer = console_buffer
        
        # Write header
        self._write_header()
        
        # Store data for complete sections
        self._hierarchy_data = None
        self._tagging_data = None
        self._model_name = ""
        self._export_time = 0.0
    
    def _write_header(self) -> None:
        """Write report header."""
        self.buffer.write("=" * 80 + "\n")
        self.buffer.write("HTP ONNX EXPORT REPORT\n")
        self.buffer.write("=" * 80 + "\n")
        self.buffer.write(f"Timestamp: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
    
    def _write_default(self, export_step: ExportStep, data: ExportData) -> int:
        """Default handler - do nothing."""
        return 0
    
    @step(ExportStep.MODEL_PREP)
    def write_model_prep(self, export_step: ExportStep, data: ExportData) -> int:
        """Record model info for header."""
        self._model_name = data.model_name
        self.buffer.write(f"Model: {data.model_name}\n")
        self.buffer.write(f"Output: {data.output_path}\n")
        self.buffer.write("=" * 80 + "\n\n")
        return 1
    
    @step(ExportStep.HIERARCHY)
    def write_hierarchy(self, export_step: ExportStep, data: ExportData) -> int:
        """Store hierarchy data for later."""
        if data.hierarchy:
            self._hierarchy_data = data.hierarchy
        return 0
    
    @step(ExportStep.NODE_TAGGING)
    def write_node_tagging(self, export_step: ExportStep, data: ExportData) -> int:
        """Store tagging data for later."""
        if data.node_tagging:
            self._tagging_data = data.node_tagging
        return 0
    
    @step(ExportStep.TAG_INJECTION)
    def write_tag_injection(self, export_step: ExportStep, data: ExportData) -> int:
        """Final step - write complete report."""
        self._export_time = data.export_time
        
        # First, include the console output if available
        if self.console_buffer:
            console_content = self.console_buffer.getvalue()
            if console_content:
                self.buffer.write("CONSOLE OUTPUT\n")
                self.buffer.write("-" * 80 + "\n")
                # Strip ANSI codes from console output
                clean_content = self._strip_ansi_codes(console_content)
                self.buffer.write(clean_content)
                if not clean_content.endswith("\n"):
                    self.buffer.write("\n")
                self.buffer.write("\n")
        
        # Add complete module hierarchy (no truncation)
        if self._hierarchy_data:
            self._write_complete_hierarchy()
        
        # Add complete node-to-hierarchy mapping
        if self._tagging_data:
            self._write_complete_node_mapping()
        
        # Add summary
        self._write_summary()
        
        return 1
    
    def _strip_ansi_codes(self, text: str) -> str:
        """Remove ANSI escape codes from text."""
        # Pattern to match ANSI escape sequences
        ansi_pattern = re.compile(r'\x1b\[[0-9;]*m')
        return ansi_pattern.sub('', text)
    
    def _write_complete_hierarchy(self) -> None:
        """Write complete module hierarchy without truncation."""
        self.buffer.write("=" * 80 + "\n")
        self.buffer.write("COMPLETE MODULE HIERARCHY\n")
        self.buffer.write("=" * 80 + "\n")
        
        if not self._hierarchy_data:
            self.buffer.write("No hierarchy data available\n")
            return
        
        # Sort modules by path for consistent output
        sorted_modules = sorted(self._hierarchy_data.hierarchy.items())
        
        for path, module_info in sorted_modules:
            # Format as per spec: Module: <path>
            display_path = "[ROOT]" if path == "" else path
            self.buffer.write(f"\nModule: {display_path}\n")
            self.buffer.write(f"  Class: {module_info.class_name}\n")
            self.buffer.write(f"  Tag: {module_info.traced_tag}\n")
            if module_info.execution_order is not None:
                self.buffer.write(f"  Execution Order: {module_info.execution_order}\n")
            if module_info.source:
                self.buffer.write(f"  Source: {module_info.source}\n")
        
        self.buffer.write(f"\nTotal Modules: {len(self._hierarchy_data.hierarchy)}\n")
        self.buffer.write(f"Execution Steps: {self._hierarchy_data.execution_steps}\n")
    
    def _write_complete_node_mapping(self) -> None:
        """Write complete node-to-hierarchy mapping without truncation."""
        self.buffer.write("\n" + "=" * 80 + "\n")
        self.buffer.write("COMPLETE NODE-TO-HIERARCHY MAPPING\n")
        self.buffer.write("=" * 80 + "\n")
        
        if not self._tagging_data:
            self.buffer.write("No tagging data available\n")
            return
        
        # Sort nodes by name for consistent output
        sorted_nodes = sorted(self._tagging_data.tagged_nodes.items())
        
        for node_name, hierarchy_tag in sorted_nodes:
            self.buffer.write(f"{node_name} -> {hierarchy_tag}\n")
        
        self.buffer.write(f"\nTotal ONNX Nodes: {self._tagging_data.total_nodes}\n")
        self.buffer.write(f"Tagged Nodes: {len(self._tagging_data.tagged_nodes)}\n")
        self.buffer.write(f"Coverage: {self._tagging_data.coverage:.1f}%\n")
        
        # Add tagging statistics
        if self._tagging_data.tagging_stats:
            self.buffer.write("\nTagging Statistics:\n")
            for stat_name, stat_value in sorted(self._tagging_data.tagging_stats.items()):
                self.buffer.write(f"  {stat_name}: {stat_value}\n")
        
        # Add operation counts if available
        if self._tagging_data.op_counts:
            self.buffer.write("\nTop Operations:\n")
            sorted_ops = sorted(
                self._tagging_data.op_counts.items(),
                key=lambda x: x[1],
                reverse=True
            )[:10]  # Top 10 operations
            for op_type, count in sorted_ops:
                self.buffer.write(f"  {op_type}: {count} nodes\n")
    
    def _write_summary(self) -> None:
        """Write export summary."""
        self.buffer.write("\n" + "=" * 80 + "\n")
        self.buffer.write("EXPORT SUMMARY\n")
        self.buffer.write("=" * 80 + "\n")
        self.buffer.write(f"Export Time: {self._export_time:.2f}s\n")
        self.buffer.write(f"Model: {self._model_name}\n")
        self.buffer.write(f"Output Path: {self.output_path}\n")
        
        if self._hierarchy_data:
            self.buffer.write(f"Total Modules: {len(self._hierarchy_data.hierarchy)}\n")
        
        if self._tagging_data:
            self.buffer.write(f"Total ONNX Nodes: {self._tagging_data.total_nodes}\n")
            self.buffer.write(f"Tagged Nodes: {len(self._tagging_data.tagged_nodes)}\n")
            self.buffer.write(f"Final Coverage: {self._tagging_data.coverage:.1f}%\n")
        
        # Add completion message
        self.buffer.write("\n" + "=" * 80 + "\n")
        self.buffer.write("Export completed successfully!\n")
    
    def flush(self) -> None:
        """Write buffer to file."""
        with open(self.report_path, 'w', encoding='utf-8') as f:
            f.write(self.buffer.getvalue())
    
    def close(self) -> None:
        """Close buffer and write file."""
        if not self.buffer.closed:
            self.flush()
            self.buffer.close()
        # Don't call super().close() as it would try to flush again