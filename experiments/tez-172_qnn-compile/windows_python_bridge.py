#!/usr/bin/env python3
"""
Windows Python Bridge - Use Windows Python to run QNN conversion
This creates a bridge between WSL and Windows Python with QNN SDK.
"""

import os
import sys
import subprocess
from pathlib import Path
import tempfile

def create_windows_python_script():
    """Create a Python conversion script for Windows execution"""
    
    script_content = '''
import sys
import os
from pathlib import Path

# Setup QNN environment
qnn_sdk_root = Path(r"C:\\Qualcomm\\AIStack\\qairt\\2.34.0.250424")
os.environ['QNN_SDK_ROOT'] = str(qnn_sdk_root)
os.environ['PYTHONPATH'] = f"{qnn_sdk_root}\\\\lib\\\\python;{os.environ.get('PYTHONPATH', '')}"

sys.path.insert(0, str(qnn_sdk_root / "lib" / "python"))

def convert_gguf_to_qnn():
    """Perform the conversion using Windows Python + QNN"""
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', required=True, help='Input GGUF file path')
    parser.add_argument('--output-dlc', required=True, help='Output DLC path')
    parser.add_argument('--output-bin', required=True, help='Output binary path')
    args = parser.parse_args()
    
    print("=" * 70)
    print("🚀 QNN Conversion (Windows Python)")
    print("=" * 70)
    print(f"Input: {args.input}")
    print(f"DLC Output: {args.output_dlc}")
    print(f"Binary Output: {args.output_bin}")
    print()
    
    # Method 1: Try using subprocess to call qairt-converter
    import subprocess
    
    converter = qnn_sdk_root / "bin" / "x86_64-windows-msvc" / "qairt-converter"
    
    cmd = [
        sys.executable,
        str(converter),
        "--input_network", args.input,
        "--output_path", args.output_dlc,
        "--input_layout", "input_ids,NONTRIVIAL",
        "--input_layout", "attention_mask,NONTRIVIAL", 
        "--preserve_io", "datatype,input_ids,attention_mask",
        "--float_fallback",
        "--float_bitwidth", "16",
        "--enable_cpu_fallback"
    ]
    
    print("🔄 Running qairt-converter...")
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
        
        if result.returncode == 0:
            print("✅ DLC generation successful!")
            dlc_path = Path(args.output_dlc)
            if dlc_path.exists():
                print(f"   Size: {dlc_path.stat().st_size / (1024**2):.1f} MB")
        else:
            print(f"❌ DLC generation failed: {result.stderr}")
            return 1
            
    except Exception as e:
        print(f"❌ Conversion error: {e}")
        return 1
    
    # Method 2: Generate context binary
    ctx_gen = qnn_sdk_root / "bin" / "x86_64-windows-msvc" / "qnn-context-binary-generator.exe"
    backend_lib = qnn_sdk_root / "lib" / "x86_64-windows-msvc" / "libQnnHtp.dll"
    
    cmd_ctx = [
        str(ctx_gen),
        "--dlc_path", args.output_dlc,
        "--backend", str(backend_lib),
        "--binary_file", args.output_bin,
        "--output_dir", str(Path(args.output_bin).parent),
        "--target_arch", "sm8650"
    ]
    
    print("\\n🔄 Generating context binary...")
    try:
        result = subprocess.run(cmd_ctx, capture_output=True, text=True, timeout=300)
        
        if result.returncode == 0:
            print("✅ Context binary generation successful!")
            bin_path = Path(args.output_bin)
            if bin_path.exists():
                print(f"   Size: {bin_path.stat().st_size / (1024**2):.1f} MB")
        else:
            print(f"⚠️ Context binary failed (optional): {result.stderr}")
            
    except Exception as e:
        print(f"⚠️ Context binary error: {e}")
    
    print("\\n✅ Conversion process completed!")
    return 0

if __name__ == "__main__":
    sys.exit(convert_gguf_to_qnn())
'''
    
    # Save to Windows temp directory via WSL path
    windows_script = Path("/mnt/c/temp/qnn_conversion.py")
    windows_script.parent.mkdir(exist_ok=True)
    
    with open(windows_script, 'w') as f:
        f.write(script_content)
    
    return windows_script

def run_windows_conversion():
    """Run conversion using Windows Python"""
    
    print("=" * 70)
    print("🌉 Windows Python Bridge Conversion")
    print("=" * 70)
    print("Creating bridge between WSL and Windows Python with QNN SDK")
    print()
    
    # Paths
    script_dir = Path(__file__).parent
    gguf_path = script_dir / "models" / "DeepSeek-R1-Distill-Qwen-1.5B-Q4_0.gguf"
    temp_dir = script_dir / "temp"
    temp_dir.mkdir(exist_ok=True)
    
    if not gguf_path.exists():
        print(f"❌ GGUF model not found: {gguf_path}")
        return 1
    
    # Create Windows Python script
    print("🔧 Creating Windows Python conversion script...")
    windows_script = create_windows_python_script()
    print(f"   Script: {windows_script}")
    
    # Convert paths to Windows format
    def wsl_to_windows_path(wsl_path):
        """Convert WSL path to Windows path"""
        path_str = str(wsl_path)
        if path_str.startswith('/home/'):
            return f"\\\\wsl.localhost\\Ubuntu{path_str}"
        elif path_str.startswith('/mnt/c/'):
            return path_str.replace('/mnt/c/', 'C:/')
        else:
            return f"\\\\wsl.localhost\\Ubuntu{path_str}"
    
    # Output paths
    dlc_path = temp_dir / "deepseek_real.dlc"
    bin_path = temp_dir / "deepseek_real.bin"
    
    # Convert to Windows paths
    win_gguf = wsl_to_windows_path(gguf_path)
    win_dlc = wsl_to_windows_path(dlc_path)
    win_bin = wsl_to_windows_path(bin_path)
    
    print(f"📦 Input: {win_gguf}")
    print(f"📄 DLC: {win_dlc}")
    print(f"🔧 Binary: {win_bin}")
    print()
    
    # Run Windows Python
    print("🚀 Launching Windows Python...")
    cmd = [
        "cmd.exe", "/c",
        f"python C:\\temp\\qnn_conversion.py "
        f"--input \"{win_gguf}\" "
        f"--output-dlc \"{win_dlc}\" "
        f"--output-bin \"{win_bin}\""
    ]
    
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=900  # 15 minutes
        )
        
        print("Windows Python output:")
        print("-" * 50)
        print(result.stdout)
        
        if result.stderr:
            print("Errors:")
            print(result.stderr)
        
        # Check results
        success = dlc_path.exists()
        
        if success:
            print()
            print("=" * 70)
            print("✅ CONVERSION SUCCESSFUL!")
            print("=" * 70)
            print(f"DLC: {dlc_path} ({dlc_path.stat().st_size / (1024**2):.1f} MB)")
            
            if bin_path.exists():
                print(f"Binary: {bin_path} ({bin_path.stat().st_size / (1024**2):.1f} MB)")
            
            print()
            print("🎉 Real QNN model files generated!")
            print("   Ready for deployment on Snapdragon NPU")
            return 0
        else:
            print("❌ Conversion failed - no output files generated")
            return 1
            
    except subprocess.TimeoutExpired:
        print("⚠️ Conversion timed out after 15 minutes")
        return 1
    except Exception as e:
        print(f"❌ Bridge error: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(run_windows_conversion())