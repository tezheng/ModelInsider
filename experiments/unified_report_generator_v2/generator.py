"""
Main report generator implementation.
"""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from interfaces import (
    IReportGenerator, IReportSection, IDataProvider,
    ReportFormat, StepInfo, StatisticsInfo
)
from sections import (
    ModelInfoSection, StepSection, StatisticsSection,
    HierarchyTreeSection, SummarySection
)


class UnifiedReportGenerator(IReportGenerator):
    """Main report generator that coordinates sections."""
    
    def __init__(self, data_provider: IDataProvider):
        self.data_provider = data_provider
        self.sections: List[IReportSection] = []
        self._setup_default_sections()
    
    def _setup_default_sections(self):
        """Set up default report sections."""
        # Add sections in order
        self.add_section(ModelInfoSection(self.data_provider))
        
        # Add step sections
        steps = [
            ("input_generation", 2, "INPUT GENERATION & VALIDATION", "🔧"),
            ("hierarchy_building", 3, "HIERARCHY BUILDING", "🏗️"),
            ("onnx_export", 4, "ONNX EXPORT", "📦"),
            ("node_tagger_creation", 5, "NODE TAGGER CREATION", "🏷️"),
            ("node_tagging", 6, "ONNX NODE TAGGING", "🔗"),
            ("tag_injection", 7, "TAG INJECTION", "🏷️"),
            ("metadata_generation", 8, "METADATA GENERATION", "📄"),
        ]
        
        for step_name, num, title, icon in steps:
            # Create step info if needed
            if self.data_provider.get_step_info(step_name) is None:
                # This would be handled by the data provider
                pass
            self.add_section(StepSection(step_name, self.data_provider))
        
        # Add other sections
        self.add_section(HierarchyTreeSection(self.data_provider))
        self.add_section(StatisticsSection(self.data_provider))
        self.add_section(SummarySection(self.data_provider))
    
    def add_section(self, section: IReportSection) -> None:
        """Add a section to the report."""
        self.sections.append(section)
    
    def generate(self, format: ReportFormat, **context) -> Any:
        """Generate the complete report in the specified format."""
        if format == ReportFormat.CONSOLE:
            return self._generate_console(context)
        elif format == ReportFormat.TEXT:
            return self._generate_text(context)
        elif format == ReportFormat.METADATA:
            return self._generate_metadata(context)
        else:
            raise ValueError(f"Unsupported format: {format}")
    
    def _generate_console(self, context: Dict[str, Any]) -> str:
        """Generate console output."""
        # Set console-specific context
        context.setdefault('width', 80)
        context.setdefault('color', True)
        context.setdefault('max_tree_lines', 30)  # Truncate trees
        
        parts = []
        for section in self.sections:
            rendered = section.render(ReportFormat.CONSOLE, context)
            if rendered:
                parts.append(rendered)
        
        return "".join(parts)
    
    def _generate_text(self, context: Dict[str, Any]) -> str:
        """Generate text report."""
        # Set text-specific context
        context.setdefault('width', 80)
        context['max_tree_lines'] = None  # No truncation in text report
        
        parts = []
        
        # Add header
        parts.append("=" * 80)
        parts.append("\nHTP EXPORT FULL REPORT\n")
        parts.append("=" * 80)
        parts.append("\n")
        
        # Add sections
        for section in self.sections:
            rendered = section.render(ReportFormat.TEXT, context)
            if rendered:
                parts.append(rendered)
        
        # Add complete listings at the end
        parts.append(self._generate_complete_listings())
        
        return "".join(parts)
    
    def _generate_complete_listings(self) -> str:
        """Generate complete module and node listings for text report."""
        parts = []
        
        # Complete module hierarchy
        hierarchy_data = self.data_provider.get_hierarchy_data()
        parts.append("\n" + "=" * 80 + "\n")
        parts.append("COMPLETE MODULE HIERARCHY\n")
        parts.append("=" * 80 + "\n\n")
        
        for module_path, module_data in sorted(hierarchy_data.items()):
            parts.append(f"Module: {module_path or '[ROOT]'}\n")
            parts.append(f"  Class: {module_data.get('class_name', 'Unknown')}\n")
            parts.append(f"  Tag: {module_data.get('traced_tag', '')}\n")
            if 'parameters' in module_data:
                parts.append(f"  Parameters: {module_data['parameters']:,}\n")
            parts.append("\n")
        
        # Complete node mappings
        tagged_nodes = self.data_provider.get_tagged_nodes()
        parts.append("\n" + "=" * 80 + "\n")
        parts.append("COMPLETE NODE MAPPINGS\n")
        parts.append("=" * 80 + "\n\n")
        
        for node_name, tag in sorted(tagged_nodes.items()):
            parts.append(f"{node_name} -> {tag}\n")
        
        return "".join(parts)
    
    def _generate_metadata(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Generate metadata dictionary."""
        metadata = {}
        
        # Collect from all sections
        for section in self.sections:
            rendered = section.render(ReportFormat.METADATA, context)
            if rendered and isinstance(rendered, dict):
                # Merge dictionaries
                for key, value in rendered.items():
                    if key in metadata and isinstance(metadata[key], dict) and isinstance(value, dict):
                        # Merge nested dicts
                        metadata[key].update(value)
                    else:
                        metadata[key] = value
        
        # Add additional metadata structure
        self._structure_metadata(metadata)
        
        return metadata
    
    def _structure_metadata(self, metadata: Dict[str, Any]):
        """Structure metadata according to expected format."""
        # Ensure required top-level keys exist
        if "export_context" not in metadata:
            export_config = self.data_provider.get_export_config()
            metadata["export_context"] = {
                "timestamp": export_config.get("timestamp", ""),
                "strategy": "htp",
                "version": "1.0",
                "exporter": "HTPExporter",
                "embed_hierarchy_attributes": export_config.get("embed_hierarchy_attributes", True)
            }
        
        # Restructure nodes if needed
        if "nodes" not in metadata and "tagged_nodes" in metadata:
            metadata["nodes"] = metadata.pop("tagged_nodes")
        
        # Add outputs section
        if "outputs" not in metadata:
            file_info = self.data_provider.get_file_info()
            metadata["outputs"] = {
                "onnx_model": {
                    "path": Path(file_info.get("onnx_path", "")).name,
                    "size_mb": file_info.get("onnx_size_mb", 0),
                    "opset_version": self.data_provider.get_export_config().get("opset_version", 17)
                }
            }
            
            if file_info.get("metadata_path"):
                metadata["outputs"]["metadata"] = {
                    "path": Path(file_info["metadata_path"]).name
                }
            
            if file_info.get("report_path"):
                metadata["outputs"]["report"] = {
                    "path": Path(file_info["report_path"]).name
                }
        
        # Ensure report section exists with proper structure
        if "report" not in metadata:
            metadata["report"] = {}
        
        # Move statistics to report.node_tagging if needed
        if "statistics" in metadata and "node_tagging" not in metadata["report"]:
            metadata["report"]["node_tagging"] = {
                "statistics": metadata.pop("statistics")
            }
    
    def save(self, path: str, format: ReportFormat) -> None:
        """Save the report to a file."""
        content = self.generate(format)
        
        if format == ReportFormat.METADATA:
            # Save as JSON
            with open(path, 'w') as f:
                json.dump(content, f, indent=2)
        else:
            # Save as text
            with open(path, 'w', encoding='utf-8') as f:
                f.write(content)