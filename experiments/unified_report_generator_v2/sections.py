"""
Concrete report section implementations with shared logic.
"""

from typing import Any, Dict, List, Optional
from dataclasses import asdict

from interfaces import (
    IReportSection, IReportFormatter, IDataProvider,
    ReportFormat, StepInfo, StatisticsInfo
)
from formatters import get_formatter


class ModelInfoSection(IReportSection):
    """Section for model information."""
    
    def __init__(self, data_provider: IDataProvider):
        self.data_provider = data_provider
    
    def get_data(self) -> Dict[str, Any]:
        """Get model info data."""
        return self.data_provider.get_model_info()
    
    def render(self, format: ReportFormat, context: Dict[str, Any]) -> Any:
        """Render model info in specified format."""
        formatter = get_formatter(format, **context)
        data = self.get_data()
        
        elements = []
        
        # Different rendering based on format
        if format == ReportFormat.CONSOLE:
            elements.append(formatter.format_header("📋 STEP 1/8: MODEL PREPARATION"))
            elements.append(formatter.format_key_value(
                "✅ Model loaded",
                f"{data['class_name']} ({data['total_modules']} modules, {data['total_parameters']/1e6:.1f}M parameters)"
            ))
            elements.append(formatter.format_key_value("🎯 Export target", data['output_path']))
            elements.append(formatter.format_key_value("⚙️ Strategy", "HTP (Hierarchy-Preserving)"))
            if data.get('eval_mode'):
                elements.append("✅ Model set to evaluation mode\n")
                
        elif format == ReportFormat.TEXT:
            elements.append(formatter.format_header("MODEL INFORMATION", level=2))
            elements.append(formatter.format_key_value("Model Class", data['class_name']))
            elements.append(formatter.format_key_value("Total Modules", data['total_modules']))
            elements.append(formatter.format_key_value("Total Parameters", f"{data['total_parameters']:,}"))
            elements.append(formatter.format_key_value("Output Path", data['output_path']))
            
        elif format == ReportFormat.METADATA:
            return {
                "model": {
                    "name_or_path": data.get('name_or_path', ''),
                    "class": data['class_name'],
                    "framework": "transformers",
                    "total_modules": data['total_modules'],
                    "total_parameters": data['total_parameters']
                }
            }
        
        return formatter.join_elements(elements)


class StepSection(IReportSection):
    """Generic section for export steps."""
    
    def __init__(self, step_name: str, data_provider: IDataProvider):
        self.step_name = step_name
        self.data_provider = data_provider
    
    def get_data(self) -> Optional[StepInfo]:
        """Get step data."""
        return self.data_provider.get_step_info(self.step_name)
    
    def render(self, format: ReportFormat, context: Dict[str, Any]) -> Any:
        """Render step in specified format."""
        step_info = self.get_data()
        if not step_info:
            return None
            
        formatter = get_formatter(format, **context)
        elements = []
        
        if format == ReportFormat.CONSOLE:
            # Console rendering with icons and formatting
            elements.append(formatter.format_header(
                f"{step_info.icon} STEP {step_info.number}/{step_info.total}: {step_info.title}"
            ))
            
            # Render step-specific details
            if self.step_name == "input_generation":
                self._render_input_generation_console(step_info, formatter, elements)
            elif self.step_name == "hierarchy_building":
                self._render_hierarchy_building_console(step_info, formatter, elements)
            # Add more step-specific renderers...
            
        elif format == ReportFormat.TEXT:
            # Plain text rendering
            elements.append(formatter.format_header(step_info.title, level=2))
            elements.append(formatter.format_key_value("Status", step_info.status))
            for key, value in step_info.details.items():
                elements.append(formatter.format_key_value(key, value))
                
        elif format == ReportFormat.METADATA:
            # Metadata rendering
            return {
                self.step_name: {
                    "status": step_info.status,
                    **step_info.details
                }
            }
        
        return formatter.join_elements(elements)
    
    def _render_input_generation_console(self, step_info: StepInfo, formatter: IReportFormatter, elements: List[Any]):
        """Render input generation step for console."""
        details = step_info.details
        elements.append(formatter.format_key_value(
            "🤖 Auto-generating inputs for",
            self.data_provider.get_model_info().get('name_or_path', 'unknown')
        ))
        
        sub_items = [
            f"Model type: {details.get('model_type', 'unknown')}",
            f"Auto-detected task: {details.get('task', 'unknown')}"
        ]
        elements.append(formatter.format_list(sub_items))
        
        elements.append(f"✅ Created onnx export config for {details.get('model_type')} with task {details.get('task')}\n")
        
        inputs = details.get('inputs', {})
        elements.append(f"🔧 Generated {len(inputs)} input tensors:\n")
        input_items = [
            f"{name}: {info['shape']} ({info['dtype']})"
            for name, info in inputs.items()
        ]
        elements.append(formatter.format_list(input_items))
    
    def _render_hierarchy_building_console(self, step_info: StepInfo, formatter: IReportFormatter, elements: List[Any]):
        """Render hierarchy building step for console."""
        details = step_info.details
        hierarchy_data = self.data_provider.get_hierarchy_data()
        
        elements.append(f"✅ Hierarchy building completed with {details.get('builder', 'TracingHierarchyBuilder')}\n")
        elements.append(f"📈 Traced {len(hierarchy_data)} modules\n")
        elements.append(f"🔄 Execution steps: {details.get('execution_steps', 0)}\n")


class StatisticsSection(IReportSection):
    """Section for tagging statistics."""
    
    def __init__(self, data_provider: IDataProvider):
        self.data_provider = data_provider
    
    def get_data(self) -> StatisticsInfo:
        """Get statistics data."""
        return self.data_provider.get_statistics()
    
    def render(self, format: ReportFormat, context: Dict[str, Any]) -> Any:
        """Render statistics in specified format."""
        stats = self.get_data()
        formatter = get_formatter(format, **context)
        
        if format == ReportFormat.CONSOLE:
            return formatter.format_statistics(stats)
        elif format == ReportFormat.TEXT:
            elements = [
                formatter.format_header("TAGGING STATISTICS", level=2),
                formatter.format_statistics(stats)
            ]
            return formatter.join_elements(elements)
        elif format == ReportFormat.METADATA:
            return {
                "statistics": formatter.format_statistics(stats)
            }


class HierarchyTreeSection(IReportSection):
    """Section for module hierarchy tree."""
    
    def __init__(self, data_provider: IDataProvider, include_nodes: bool = False):
        self.data_provider = data_provider
        self.include_nodes = include_nodes
    
    def get_data(self) -> Dict[str, Dict[str, Any]]:
        """Get hierarchy data."""
        return self.data_provider.get_hierarchy_data()
    
    def render(self, format: ReportFormat, context: Dict[str, Any]) -> Any:
        """Render hierarchy tree in specified format."""
        hierarchy_data = self.get_data()
        formatter = get_formatter(format, **context)
        
        # Build tree structure
        tree_data = self._build_tree_structure(hierarchy_data)
        
        # Get root info
        root_info = hierarchy_data.get("", {})
        root_name = root_info.get("class_name", "Model")
        
        if self.include_nodes:
            # Add node counts
            tagged_nodes = self.data_provider.get_tagged_nodes()
            node_counts = self._calculate_node_counts(tagged_nodes)
            root_name = f"{root_name} ({len(tagged_nodes)} ONNX nodes)"
        
        # Render based on format
        if format in [ReportFormat.CONSOLE, ReportFormat.TEXT]:
            # Check for truncation settings
            max_depth = context.get('max_tree_depth')
            max_lines = context.get('max_tree_lines')
            
            if format == ReportFormat.CONSOLE:
                elements = [
                    "\n🌳 Module Hierarchy:\n",
                    formatter.format_separator(60),
                    formatter.format_tree(root_name, tree_data, max_depth)
                ]
            else:
                elements = [
                    formatter.format_header("MODULE HIERARCHY", level=2),
                    formatter.format_tree(root_name, tree_data, max_depth)
                ]
            
            result = formatter.join_elements(elements)
            
            # Handle line truncation for console
            if max_lines and format == ReportFormat.CONSOLE:
                lines = result.splitlines()
                if len(lines) > max_lines:
                    truncated = lines[:max_lines]
                    truncated.append(f"... and {len(lines) - max_lines} more lines (truncated for console)")
                    truncated.append(f"(showing {max_lines}/{len(lines)} lines)")
                    result = "\n".join(truncated) + "\n"
            
            return result
            
        elif format == ReportFormat.METADATA:
            return {
                "modules": hierarchy_data
            }
    
    def _build_tree_structure(self, hierarchy_data: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        """Build nested tree structure from flat hierarchy data."""
        tree = {}
        
        # Sort paths to ensure parents come before children
        sorted_paths = sorted(hierarchy_data.keys())
        
        for path in sorted_paths:
            if not path:  # Skip root
                continue
                
            parts = path.split('.')
            current = tree
            
            # Navigate/create path
            for i, part in enumerate(parts):
                if part not in current:
                    current[part] = {}
                
                # Add info at leaf
                if i == len(parts) - 1:
                    info = hierarchy_data[path]
                    current[part]["_info"] = {
                        "class": info.get("class_name", "Unknown"),
                        "tag": info.get("traced_tag", "")
                    }
                
                current = current[part]
        
        return tree
    
    def _calculate_node_counts(self, tagged_nodes: Dict[str, str]) -> Dict[str, int]:
        """Calculate node counts per hierarchy tag."""
        counts = {}
        for node, tag in tagged_nodes.items():
            counts[tag] = counts.get(tag, 0) + 1
        return counts


class SummarySection(IReportSection):
    """Section for final export summary."""
    
    def __init__(self, data_provider: IDataProvider):
        self.data_provider = data_provider
    
    def get_data(self) -> Dict[str, Any]:
        """Get summary data."""
        model_info = self.data_provider.get_model_info()
        stats = self.data_provider.get_statistics()
        file_info = self.data_provider.get_file_info()
        export_config = self.data_provider.get_export_config()
        
        return {
            "export_time": export_config.get("export_time", 0),
            "hierarchy_modules": len(self.data_provider.get_hierarchy_data()),
            "statistics": stats,
            "files": file_info
        }
    
    def render(self, format: ReportFormat, context: Dict[str, Any]) -> Any:
        """Render summary in specified format."""
        data = self.get_data()
        formatter = get_formatter(format, **context)
        stats = data["statistics"]
        
        elements = []
        
        if format == ReportFormat.CONSOLE:
            elements.append(formatter.format_header("📋 FINAL EXPORT SUMMARY"))
            elements.append(f"🎉 HTP Export completed successfully in {data['export_time']:.2f}s!\n")
            elements.append("📊 Export Statistics:\n")
            
            stat_items = [
                f"Export time: {data['export_time']:.2f}s",
                f"Hierarchy modules: {data['hierarchy_modules']}",
                f"ONNX nodes: {stats.total_onnx_nodes}",
                f"Tagged nodes: {stats.tagged_nodes}",
                f"Coverage: {stats.coverage_percentage:.1f}%",
                f"Empty tags: {stats.empty_tags} {'✅' if stats.empty_tags == 0 else '⚠️'}"
            ]
            elements.append(formatter.format_list(stat_items))
            
            elements.append("\n📁 Output Files:\n")
            file_items = []
            for file_type, file_path in data['files'].items():
                if file_path:
                    file_items.append(f"{file_type}: {file_path}")
            elements.append(formatter.format_list(file_items))
            
        elif format == ReportFormat.TEXT:
            elements.append(formatter.format_header("EXPORT SUMMARY"))
            elements.append(formatter.format_key_value("Export Time", f"{data['export_time']:.2f}s"))
            elements.append(formatter.format_key_value("Total Modules", data['hierarchy_modules']))
            elements.append(formatter.format_statistics(stats))
            
        elif format == ReportFormat.METADATA:
            return {
                "summary": {
                    "export_time_seconds": data['export_time'],
                    "hierarchy_modules": data['hierarchy_modules'],
                    **formatter.format_statistics(stats)
                }
            }
        
        return formatter.join_elements(elements)