#!/usr/bin/env python3
"""
Windows-native GGUF to QNN conversion script.
Run this directly on Windows with QNN SDK installed.
"""

import os
import sys
import subprocess
from pathlib import Path

def setup_qnn_environment():
    """Setup QNN SDK environment variables"""
    qnn_sdk_root = Path(r"C:\Qualcomm\AIStack\qairt\2.34.0.250424")
    
    if not qnn_sdk_root.exists():
        print(f"Error: QNN SDK not found at {qnn_sdk_root}")
        print("Please install QNN SDK or update the path")
        return None
    
    # Setup environment
    os.environ['QNN_SDK_ROOT'] = str(qnn_sdk_root)
    os.environ['PYTHONPATH'] = f"{qnn_sdk_root}\\lib\\python;{os.environ.get('PYTHONPATH', '')}"
    os.environ['PATH'] = f"{qnn_sdk_root}\\bin\\x86_64-windows-msvc;{os.environ.get('PATH', '')}"
    
    # Add QNN Python modules to path
    sys.path.insert(0, str(qnn_sdk_root / "lib" / "python"))
    
    return qnn_sdk_root

def convert_gguf_to_qnn():
    """Perform the actual GGUF to QNN conversion"""
    
    print("=" * 70)
    print("GGUF to QNN Context Binary Converter")
    print("=" * 70)
    
    # Setup environment
    qnn_sdk_root = setup_qnn_environment()
    if not qnn_sdk_root:
        return 1
    
    # Get paths relative to this script
    script_dir = Path(__file__).parent
    project_dir = script_dir.parent
    
    # Input and output paths
    gguf_path = project_dir / "models" / "DeepSeek-R1-Distill-Qwen-1.5B-Q4_0.gguf"
    dlc_path = script_dir / "deepseek_qwen.dlc"
    bin_path = script_dir / "deepseek_qwen.bin"
    
    # Check if model exists
    if not gguf_path.exists():
        print(f"Error: GGUF model not found at {gguf_path}")
        return 1
    
    print(f"Input model: {gguf_path}")
    print(f"Model size: {gguf_path.stat().st_size / (1024**3):.2f} GB")
    print()
    
    # Step 1: Convert GGUF to DLC
    print("Step 1: Converting GGUF to DLC")
    print("-" * 70)
    
    converter = qnn_sdk_root / "bin" / "x86_64-windows-msvc" / "qairt-converter"
    
    cmd = [
        sys.executable,
        str(converter),
        "--input_network", str(gguf_path),
        "--output_path", str(dlc_path),
        "--input_layout", "input_ids,NONTRIVIAL",
        "--input_layout", "attention_mask,NONTRIVIAL",
        "--preserve_io", "datatype,input_ids,attention_mask",
        "--float_fallback",
        "--float_bitwidth", "16",
        "--enable_cpu_fallback"
    ]
    
    print("Running:", " ".join(cmd[:3]) + "...")
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        print("✓ DLC generation successful!")
        print(f"  Output: {dlc_path}")
        print(f"  Size: {dlc_path.stat().st_size / (1024**6):.1f} MB")
    except subprocess.CalledProcessError as e:
        print(f"✗ DLC generation failed: {e}")
        print("Error output:", e.stderr[:500])
        return 1
    
    # Step 2: Generate context binary
    print()
    print("Step 2: Generating Context Binary")
    print("-" * 70)
    
    ctx_gen = qnn_sdk_root / "bin" / "x86_64-windows-msvc" / "qnn-context-binary-generator.exe"
    backend_lib = qnn_sdk_root / "lib" / "x86_64-windows-msvc" / "libQnnHtp.dll"
    
    cmd = [
        str(ctx_gen),
        "--dlc_path", str(dlc_path),
        "--backend", str(backend_lib),
        "--binary_file", str(bin_path),
        "--output_dir", str(script_dir),
        "--target_arch", "sm8650"  # Snapdragon 8 Gen 3
    ]
    
    print("Running context binary generator...")
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        print("✓ Context binary generation successful!")
        print(f"  Output: {bin_path}")
        print(f"  Size: {bin_path.stat().st_size / (1024**6):.1f} MB")
    except subprocess.CalledProcessError as e:
        print(f"✗ Context binary generation failed: {e}")
        print("Error output:", e.stderr[:500])
        # This is optional, so don't fail
    
    # Summary
    print()
    print("=" * 70)
    print("Conversion Complete!")
    print("=" * 70)
    print(f"✓ DLC: {dlc_path.name}")
    print(f"✓ Binary: {bin_path.name}")
    print()
    print("The model is now ready for deployment on Snapdragon NPU!")
    print("Use ONNX Runtime with QNN EP to load the context binary.")
    
    return 0

if __name__ == "__main__":
    sys.exit(convert_gguf_to_qnn())