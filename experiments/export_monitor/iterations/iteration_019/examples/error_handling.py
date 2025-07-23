#!/usr/bin/env python3
"""Error handling example."""

from modelexport.strategies.htp import HTPExporter

try:
    exporter = HTPExporter(verbose=True)
    result = exporter.export(
        model=model,
        output_path="model.onnx",
        model_name_or_path="my-model"
    )
except Exception as e:
    print(f"Export failed: {e}")
    # The monitor will have saved partial outputs
    # Check model_console.log for debugging info
