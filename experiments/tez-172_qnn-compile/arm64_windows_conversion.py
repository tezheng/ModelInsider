#!/usr/bin/env python3
"""
ARM64 Windows QNN Conversion - Use the correct ARM64 tools for Windows ARM64.
"""

import os
import sys
import subprocess
from pathlib import Path

def create_arm64_conversion_script():
    """Create conversion script using ARM64 Windows tools"""
    
    script_content = r'''# -*- coding: utf-8 -*-
import sys
import os
import subprocess
from pathlib import Path

# Setup QNN environment for ARM64 Windows
qnn_sdk_root = Path(r"C:\Qualcomm\AIStack\qairt\2.34.0.250424")
os.environ['QNN_SDK_ROOT'] = str(qnn_sdk_root)
os.environ['PYTHONPATH'] = f"{qnn_sdk_root}\\lib\\python;{os.environ.get('PYTHONPATH', '')}"

def convert_model_arm64():
    """Convert using ARM64 Windows tools"""
    
    # Paths - using ARM64 Windows tools
    converter = qnn_sdk_root / "bin" / "aarch64-windows-msvc" / "qairt-converter"
    if not converter.exists():
        converter = qnn_sdk_root / "bin" / "x86_64-windows-msvc" / "qairt-converter"
        print("WARNING: ARM64 converter not found, trying x86_64")
    
    ctx_gen = qnn_sdk_root / "bin" / "aarch64-windows-msvc" / "qnn-context-binary-generator.exe"
    if not ctx_gen.exists():
        ctx_gen = qnn_sdk_root / "bin" / "x86_64-windows-msvc" / "qnn-context-binary-generator.exe"
        print("WARNING: ARM64 context generator not found, trying x86_64")
    
    backend = qnn_sdk_root / "lib" / "aarch64-windows-msvc" / "QnnHtp.dll"
    if not backend.exists():
        backend = qnn_sdk_root / "lib" / "x86_64-windows-msvc" / "libQnnHtp.dll"
        print("WARNING: ARM64 backend not found, trying x86_64")
    
    print("=" * 60)
    print("ARM64 Windows QNN Conversion")
    print("=" * 60)
    print(f"Converter: {converter}")
    print(f"Context Gen: {ctx_gen}")
    print(f"Backend: {backend}")
    print()
    
    # Input/output paths
    input_model = r"\\wsl.localhost\Ubuntu\home\zhengte\modelexport_tez47\experiments\tez-172_qnn-compile\models\DeepSeek-R1-Distill-Qwen-1.5B-Q4_0.gguf"
    output_dlc = r"\\wsl.localhost\Ubuntu\home\zhengte\modelexport_tez47\experiments\tez-172_qnn-compile\temp\deepseek_arm64.dlc"
    output_bin = r"\\wsl.localhost\Ubuntu\home\zhengte\modelexport_tez47\experiments\tez-172_qnn-compile\temp\deepseek_arm64.bin"
    
    # Step 1: Check if we can import QNN modules first
    print("Testing QNN module imports...")
    try:
        # Try to add ARM64 libraries to path
        arm64_py_dir = qnn_sdk_root / "lib" / "python" / "qti" / "aisw" / "converters" / "common" / "windows-arm64ec"
        if arm64_py_dir.exists():
            sys.path.insert(0, str(arm64_py_dir))
            print(f"Added ARM64 Python libs: {arm64_py_dir}")
        
        # Try importing
        from qti.aisw.converters.common import ir_graph
        print("SUCCESS: QNN ARM64 modules imported!")
        
        # Now try the conversion
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
        
        print("\nStep 1: Converting GGUF to DLC...")
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
        
        if result.returncode == 0:
            print("SUCCESS: DLC generated")
            
            # Step 2: Context binary
            cmd_bin = [
                str(ctx_gen),
                "--dlc_path", output_dlc,
                "--backend", str(backend),
                "--binary_file", output_bin,
                "--output_dir", str(Path(output_bin).parent),
                "--target_arch", "sm8650"
            ]
            
            print("\nStep 2: Generating context binary...")
            result = subprocess.run(cmd_bin, capture_output=True, text=True, timeout=300)
            
            if result.returncode == 0:
                print("SUCCESS: Context binary generated")
            else:
                print("WARNING: Context binary failed")
                
        else:
            print("ERROR: DLC generation failed")
            print("STDERR:", result.stderr[:500])
            return 1
        
    except ImportError as e:
        print(f"ERROR: Cannot import QNN ARM64 modules: {e}")
        print("This confirms we need proper ARM64 support")
        
        # Alternative: Try direct execution with proper environment
        print("\nTrying alternative approach...")
        
        # Set up environment variables for ARM64
        env = os.environ.copy()
        env['PYTHONPATH'] = f"{qnn_sdk_root}\\lib\\python\\qti\\aisw\\converters\\common\\windows-arm64ec;{qnn_sdk_root}\\lib\\python"
        
        cmd = [
            "python",
            str(converter),
            "--input_network", input_model,
            "--output_path", output_dlc,
            "--float_fallback",
            "--enable_cpu_fallback"
        ]
        
        result = subprocess.run(cmd, env=env, capture_output=True, text=True, timeout=600)
        
        if result.returncode == 0:
            print("SUCCESS: Alternative approach worked!")
        else:
            print("ERROR: Alternative approach also failed")
            print("STDERR:", result.stderr[:500])
            
    except Exception as e:
        print(f"ERROR: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(convert_model_arm64())
'''
    
    # Save to C:\temp
    windows_script = Path("/mnt/c/temp/arm64_qnn_convert.py")
    with open(windows_script, 'w', encoding='utf-8') as f:
        f.write(script_content)
    
    return windows_script

def run_arm64_conversion():
    """Run ARM64 conversion"""
    
    print("=" * 60)
    print("ARM64 Windows QNN Conversion")
    print("=" * 60)
    print("Using ARM64 Windows QNN tools for proper architecture match")
    print()
    
    # Check what ARM64 tools we have
    qnn_sdk = Path("/mnt/c/Qualcomm/AIStack/qairt/2.34.0.250424")
    arm64_bin = qnn_sdk / "bin" / "aarch64-windows-msvc"
    arm64_lib = qnn_sdk / "lib" / "aarch64-windows-msvc"
    
    print("ARM64 Windows Tools Check:")
    print(f"  Binaries: {arm64_bin} ({'EXISTS' if arm64_bin.exists() else 'MISSING'})")
    if arm64_bin.exists():
        tools = list(arm64_bin.glob("*"))
        print(f"    Found {len(tools)} tools: {', '.join([t.name for t in tools[:5]])}")
    
    print(f"  Libraries: {arm64_lib} ({'EXISTS' if arm64_lib.exists() else 'MISSING'})")
    if arm64_lib.exists():
        libs = list(arm64_lib.glob("*.dll"))
        print(f"    Found {len(libs)} DLLs: {', '.join([l.name for l in libs[:5]])}")
    
    # Create and run script
    print("\nCreating ARM64 conversion script...")
    windows_script = create_arm64_conversion_script()
    print(f"Script: {windows_script}")
    
    print("\nRunning ARM64 Windows conversion...")
    
    cmd = [
        "cmd.exe", "/c", 
        f"cd C:\\temp && python arm64_qnn_convert.py"
    ]
    
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=900,
            encoding='utf-8',
            errors='replace'
        )
        
        print("ARM64 Windows Output:")
        print("-" * 40)
        if result.stdout:
            print(result.stdout)
        
        if result.stderr:
            print("Errors/Warnings:")
            print(result.stderr)
        
        # Check results
        temp_dir = Path(__file__).parent / "temp"
        dlc_file = temp_dir / "deepseek_arm64.dlc"
        bin_file = temp_dir / "deepseek_arm64.bin"
        
        print("\n" + "=" * 60)
        print("ARM64 CONVERSION RESULTS:")
        print("=" * 60)
        
        if dlc_file.exists():
            size_mb = dlc_file.stat().st_size / (1024**2)
            print(f"SUCCESS: DLC created - {dlc_file.name} ({size_mb:.1f} MB)")
        else:
            print("FAILED: No DLC file created")
        
        if bin_file.exists():
            size_mb = bin_file.stat().st_size / (1024**2)
            print(f"SUCCESS: Binary created - {bin_file.name} ({size_mb:.1f} MB)")
        else:
            print("INFO: No binary file")
        
        return 0 if dlc_file.exists() else 1
        
    except Exception as e:
        print(f"ERROR: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(run_arm64_conversion())