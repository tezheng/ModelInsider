#!/usr/bin/env python3
"""
Simple test to verify export works from experiments directory.
Run this from experiments/tez-144_onnx_automodel_infer/
"""

import subprocess
from pathlib import Path

print("🧪 Testing Export from Experiments Directory")
print("=" * 50)

# Test export from this directory
model_name = "prajjwal1/bert-tiny"
output_dir = Path("models/test-export")
output_dir.mkdir(parents=True, exist_ok=True)

onnx_path = output_dir / "model.onnx"

print(f"Model: {model_name}")
print(f"Output: {onnx_path}")
print(f"Current directory: {Path.cwd()}")

# Test command
cmd = ["uv", "run", "modelexport", "export",
       "--model", model_name,
       "--output", str(onnx_path)]
       # Clean ONNX is now the default, no flag needed

print(f"\n📤 Running: {' '.join(cmd)}")

# Run from project root (../..)
result = subprocess.run(cmd, capture_output=True, text=True, cwd="../..")

print(f"\nReturn code: {result.returncode}")
print(f"STDOUT:\n{result.stdout}")
if result.stderr:
    print(f"STDERR:\n{result.stderr}")

if result.returncode == 0 and onnx_path.exists():
    size = onnx_path.stat().st_size / (1024 * 1024)
    print(f"\n✅ SUCCESS! ONNX exported: {size:.1f} MB")
else:
    print(f"\n❌ FAILED!")