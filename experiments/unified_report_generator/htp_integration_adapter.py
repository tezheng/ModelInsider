"""
Adapter to integrate the unified report generator into HTP exporter.

This shows how to modify the HTP exporter to use the unified approach.
"""

from pathlib import Path

from unified_report_generator import ExportSession, UnifiedReportGenerator


class HTPExporterAdapter:
    """Adapter to show how HTP exporter would be modified."""
    
    def __init__(self, exporter):
        """Initialize with HTP exporter instance."""
        self.exporter = exporter
        self.session = ExportSession(
            strategy="htp",
            verbose=exporter.verbose,
            enable_reporting=exporter.enable_reporting,
            embed_hierarchy_attributes=exporter.embed_hierarchy_attributes
        )
    
    def update_session_from_exporter(self):
        """Update session with data from exporter."""
        # Model info
        if hasattr(self.exporter, '_export_report'):
            model_info = self.exporter._export_report.get("model_info", {})
            self.session.model_name_or_path = model_info.get("model_name_or_path", "")
            self.session.model_class = model_info.get("model_class", "")
            self.session.total_modules = model_info.get("total_modules", 0)
            self.session.total_parameters = model_info.get("total_parameters", 0)
        
        # Copy data collections
        self.session.hierarchy_data = self.exporter._hierarchy_data.copy()
        self.session.tagged_nodes = self.exporter._tagged_nodes.copy()
        self.session.tagging_statistics = self.exporter._tagging_stats.copy()
        
        # Export statistics
        self.session.export_time = self.exporter._export_stats.get("export_time", 0.0)
        self.session.onnx_nodes_count = self.exporter._export_stats.get("onnx_nodes", 0)
        self.session.tagged_nodes_count = self.exporter._export_stats.get("tagged_nodes", 0)
        self.session.coverage_percentage = self.exporter._export_stats.get("coverage_percentage", 0.0)
        self.session.empty_tags = self.exporter._export_stats.get("empty_tags", 0)
        
        # Copy steps from _export_report
        if hasattr(self.exporter, '_export_report') and "export_report" in self.exporter._export_report:
            for step_name, step_data in self.exporter._export_report["export_report"].items():
                self.session.add_step(
                    step_name,
                    step_data.get("status", "pending"),
                    **step_data.get("details", {})
                )
    
    def add_step(self, name: str, status: str = "in_progress", **details):
        """Add a step to the session."""
        return self.session.add_step(name, status, **details)
    
    def generate_all_reports(self, output_path: str) -> dict[str, str]:
        """Generate all report formats."""
        # Update session with latest data
        self.update_session_from_exporter()
        
        # Set file paths
        self.session.output_path = output_path
        self.session.onnx_file_path = output_path
        self.session.metadata_file_path = str(output_path).replace('.onnx', '_htp_metadata.json')
        self.session.report_file_path = str(output_path).replace('.onnx', '_full_report.txt')
        
        # Get file size
        if Path(output_path).exists():
            self.session.onnx_file_size_mb = Path(output_path).stat().st_size / (1024 * 1024)
        
        # Create generator
        generator = UnifiedReportGenerator(self.session)
        
        # Generate console output if verbose
        if self.exporter.verbose:
            console_output = generator.generate_console_output(truncate_trees=True)
            # Instead of printing directly, we would integrate with _output_message
            # For now, just print it
            print(console_output)
        
        # Generate and save metadata
        metadata = generator.generate_metadata()
        with open(self.session.metadata_file_path, 'w') as f:
            import json
            json.dump(metadata, f, indent=2)
        
        # Generate and save full report
        full_report = generator.generate_text_report()
        with open(self.session.report_file_path, 'w') as f:
            f.write(full_report)
        
        return {
            "metadata_path": self.session.metadata_file_path,
            "report_path": self.session.report_file_path
        }


# Example of how to modify HTP exporter methods:

def new_print_model_preparation(self, model, output_path):
    """Modified version that updates session."""
    # Original functionality
    module_count = len(list(model.modules()))
    param_count = sum(p.numel() for p in model.parameters())
    
    # Update session
    self.report_adapter.session.model_class = type(model).__name__
    self.report_adapter.session.total_modules = module_count
    self.report_adapter.session.total_parameters = param_count
    self.report_adapter.session.output_path = output_path
    
    # Add step
    self.report_adapter.add_step(
        "model_preparation",
        "completed",
        model_class=type(model).__name__,
        module_count=module_count,
        parameter_count=param_count,
        eval_mode=True
    )
    
    # If verbose, the console output will be generated later


def new_export_method_ending(self, output_path, metadata_path):
    """Modified export method ending."""
    # Generate all reports using the unified generator
    paths = self.report_adapter.generate_all_reports(output_path)
    
    # Return export stats as before
    return self._export_stats.copy()


# Example integration in __init__:
def init_with_adapter(self, *args, **kwargs):
    """Initialize with report adapter."""
    # Original init
    original_init(self, *args, **kwargs)
    
    # Add adapter
    self.report_adapter = HTPExporterAdapter(self)