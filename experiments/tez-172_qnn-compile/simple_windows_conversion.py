#!/usr/bin/env python3
"""
Simple Windows conversion without emoji and UNC path issues.
"""

import os
import sys
import subprocess
from pathlib import Path
import shutil

def create_simple_conversion_script():
    """Create a simple Python script for Windows without emojis"""
    
    script_content = r'''# -*- coding: utf-8 -*-
import sys
import os
import subprocess
from pathlib import Path

# Setup QNN environment
qnn_sdk_root = Path(r"C:\Qualcomm\AIStack\qairt\2.34.0.250424")
os.environ['QNN_SDK_ROOT'] = str(qnn_sdk_root)
os.environ['PYTHONPATH'] = f"{qnn_sdk_root}\\lib\\python;{os.environ.get('PYTHONPATH', '')}"

def convert_model():
    """Convert GGUF to QNN using Windows tools"""
    
    # Input arguments
    input_model = sys.argv[1] if len(sys.argv) > 1 else r"\\wsl.localhost\Ubuntu\home\zhengte\modelexport_tez47\experiments\tez-172_qnn-compile\models\DeepSeek-R1-Distill-Qwen-1.5B-Q4_0.gguf"
    output_dlc = sys.argv[2] if len(sys.argv) > 2 else r"\\wsl.localhost\Ubuntu\home\zhengte\modelexport_tez47\experiments\tez-172_qnn-compile\temp\deepseek_final.dlc"
    output_bin = sys.argv[3] if len(sys.argv) > 3 else r"\\wsl.localhost\Ubuntu\home\zhengte\modelexport_tez47\experiments\tez-172_qnn-compile\temp\deepseek_final.bin"
    
    print("=" * 60)
    print("QNN GGUF to DLC/Binary Conversion")
    print("=" * 60)
    print(f"Input: {input_model}")
    print(f"DLC: {output_dlc}")
    print(f"Binary: {output_bin}")
    print()
    
    # Step 1: GGUF to DLC
    converter = qnn_sdk_root / "bin" / "x86_64-windows-msvc" / "qairt-converter"
    
    cmd = [
        "python",
        str(converter),
        "--input_network", input_model,
        "--output_path", output_dlc,
        "--input_layout", "input_ids,NONTRIVIAL",
        "--input_layout", "attention_mask,NONTRIVIAL",
        "--preserve_io", "datatype,input_ids,attention_mask",
        "--float_fallback",
        "--float_bitwidth", "16",
        "--enable_cpu_fallback"
    ]
    
    print("Step 1: Converting GGUF to DLC...")
    print("Command: " + " ".join(cmd[:3]) + " [with args]")
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
        
        if result.returncode == 0:
            print("SUCCESS: DLC generated")
            # Check file size
            try:
                dlc_size = Path(output_dlc).stat().st_size / (1024**2)
                print(f"DLC size: {dlc_size:.1f} MB")
            except:
                print("DLC file created")
        else:
            print("ERROR: DLC generation failed")
            print("STDERR:", result.stderr[:500])
            return 1
            
    except subprocess.TimeoutExpired:
        print("ERROR: Conversion timed out")
        return 1
    except Exception as e:
        print(f"ERROR: {e}")
        return 1
    
    # Step 2: DLC to Binary
    ctx_gen = qnn_sdk_root / "bin" / "x86_64-windows-msvc" / "qnn-context-binary-generator.exe"
    backend = qnn_sdk_root / "lib" / "x86_64-windows-msvc" / "libQnnHtp.dll"
    output_dir = str(Path(output_bin).parent)
    
    cmd_bin = [
        str(ctx_gen),
        "--dlc_path", output_dlc,
        "--backend", str(backend),
        "--binary_file", output_bin,
        "--output_dir", output_dir,
        "--target_arch", "sm8650"
    ]
    
    print("\nStep 2: Generating context binary...")
    
    try:
        result = subprocess.run(cmd_bin, capture_output=True, text=True, timeout=300)
        
        if result.returncode == 0:
            print("SUCCESS: Context binary generated")
            try:
                bin_size = Path(output_bin).stat().st_size / (1024**2)
                print(f"Binary size: {bin_size:.1f} MB")
            except:
                print("Binary file created")
        else:
            print("WARNING: Context binary failed (optional)")
            print("STDERR:", result.stderr[:300])
            
    except Exception as e:
        print(f"WARNING: Context binary error: {e}")
    
    print("\n" + "=" * 60)
    print("Conversion completed!")
    print("Check the temp folder for output files")
    return 0

if __name__ == "__main__":
    exit(convert_model())
'''
    
    # Save to C:\temp
    windows_script = Path("/mnt/c/temp/simple_qnn_convert.py")
    with open(windows_script, 'w', encoding='utf-8') as f:
        f.write(script_content)
    
    return windows_script

def run_simple_conversion():
    """Run the simple conversion"""
    
    print("=" * 60)
    print("Simple Windows QNN Conversion")
    print("=" * 60)
    print()
    
    # Check model exists
    script_dir = Path(__file__).parent
    gguf_path = script_dir / "models" / "DeepSeek-R1-Distill-Qwen-1.5B-Q4_0.gguf"
    
    if not gguf_path.exists():
        print(f"ERROR: GGUF model not found at {gguf_path}")
        return 1
    
    # Create script
    print("Creating Windows conversion script...")
    windows_script = create_simple_conversion_script()
    print(f"Script created: {windows_script}")
    
    # Run using Windows Python directly in C:\temp directory
    print("\nRunning Windows Python conversion...")
    
    cmd = [
        "cmd.exe", "/c", 
        f"cd C:\\temp && python simple_qnn_convert.py"
    ]
    
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=900,
            encoding='utf-8',
            errors='replace'  # Handle encoding issues
        )
        
        print("Windows Python Output:")
        print("-" * 40)
        if result.stdout:
            print(result.stdout)
        
        if result.stderr:
            print("Errors/Warnings:")
            print(result.stderr)
        
        # Check for output files
        temp_dir = script_dir / "temp"
        dlc_file = temp_dir / "deepseek_final.dlc"
        bin_file = temp_dir / "deepseek_final.bin"
        
        dlc_exists = dlc_file.exists()
        bin_exists = bin_file.exists()
        
        print("\n" + "=" * 60)
        print("RESULTS:")
        print("=" * 60)
        
        if dlc_exists:
            size_mb = dlc_file.stat().st_size / (1024**2)
            print(f"SUCCESS: DLC created - {dlc_file.name} ({size_mb:.1f} MB)")
        else:
            print("FAILED: No DLC file created")
        
        if bin_exists:
            size_mb = bin_file.stat().st_size / (1024**2)
            print(f"SUCCESS: Binary created - {bin_file.name} ({size_mb:.1f} MB)")
        else:
            print("INFO: No binary file (optional)")
        
        if dlc_exists:
            print("\nSUCCESS: Real QNN model conversion completed!")
            print("Files ready for Snapdragon NPU deployment")
            return 0
        else:
            print("\nFAILED: Conversion did not produce output files")
            return 1
            
    except subprocess.TimeoutExpired:
        print("ERROR: Conversion timed out after 15 minutes")
        return 1
    except Exception as e:
        print(f"ERROR: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(run_simple_conversion())