#!/usr/bin/env python3
"""Basic HTP export example."""

from transformers import AutoModel
from modelexport.strategies.htp import HTPExporter

# Load model
model = AutoModel.from_pretrained("bert-base-uncased")

# Export with HTP
exporter = HTPExporter(verbose=True)
exporter.export(
    model=model,
    output_path="bert.onnx",
    model_name_or_path="bert-base-uncased"
)

print("✅ Export complete!")
