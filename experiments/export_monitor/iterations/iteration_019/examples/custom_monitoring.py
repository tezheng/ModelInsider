#!/usr/bin/env python3
"""Custom monitoring configuration example."""

from modelexport.strategies.htp import HTPExporter

# Custom configuration
config = {
    "max_tree_depth": 15,      # Limit hierarchy display depth
    "style_numbers": True,     # Enable number styling
    "batch_console": True,     # Batch console writes
    "enable_cache": True,      # Enable style caching
}

# Export with custom config
exporter = HTPExporter(verbose=True, monitor_config=config)
result = exporter.export(model, "model.onnx")

# Access individual outputs
print(f"Console log saved to: {result['console_log_path']}")
print(f"Metadata saved to: {result['metadata_path']}")
print(f"Report saved to: {result['report_path']}")
