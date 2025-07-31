#!/usr/bin/env python3
"""Test the integrated HTP exporter."""

import sys
from pathlib import Path

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent))
# Add project root to path
project_root = Path(__file__).parent.parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from htp_exporter_standalone import HTPExporter
from transformers import AutoModel


def test_export():
    """Test integrated export."""
    print("Loading model...")
    model = AutoModel.from_pretrained("prajjwal1/bert-tiny")
    
    print("\nExporting with integrated HTP exporter...")
    exporter = HTPExporter(verbose=True, embed_hierarchy_attributes=True)
    
    output_path = "test_integrated.onnx"
    export_params = {
        "model_name": "prajjwal1/bert-tiny",
        "enable_report": True,
        "opset_version": 17,
        "do_constant_folding": True
    }
    
    result = exporter.export(model, output_path, export_params=export_params)
    print(f"\n✅ Export complete: {result}")
    
    # Check outputs
    outputs = ["test_integrated.onnx", "test_integrated_htp_metadata.json", "test_integrated_full_report.txt"]
    for output in outputs:
        if Path(output).exists():
            print(f"✓ {output} created")
        else:
            print(f"✗ {output} missing")


if __name__ == "__main__":
    test_export()
