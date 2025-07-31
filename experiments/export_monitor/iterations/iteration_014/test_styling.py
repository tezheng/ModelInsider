#!/usr/bin/env python3
"""Test the fixed export monitor."""

import sys
from pathlib import Path

# Add paths
project_root = Path(__file__).parent.parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from transformers import AutoModel

from modelexport.strategies.htp.htp_exporter import HTPExporter


def test_export():
    """Test export with fixed styling."""
    print("Loading model...")
    model = AutoModel.from_pretrained("prajjwal1/bert-tiny")
    
    print("\nExporting with HTP exporter...")
    exporter = HTPExporter(verbose=True)
    
    result = exporter.export(
        model=model,
        output_path="test_styled_output.onnx",
        model_name_or_path="prajjwal1/bert-tiny"
    )
    
    print(f"\n✅ Export complete: {result}")
    
    # Check if console output has ANSI codes
    # This would need to be captured during export
    print("\n📊 Check console output for ANSI styling codes!")


if __name__ == "__main__":
    test_export()
