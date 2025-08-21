#!/usr/bin/env python3
"""
Linux-native GGUF to QNN conversion script.
Uses the Linux x86_64 binaries and libraries from QNN SDK.
"""

import os
import sys
import subprocess
from pathlib import Path

def setup_qnn_environment_linux():
    """Setup QNN SDK environment for Linux"""
    qnn_sdk_root = Path("/mnt/c/Qualcomm/AIStack/qairt/2.34.0.250424")
    
    if not qnn_sdk_root.exists():
        print(f"Error: QNN SDK not found at {qnn_sdk_root}")
        return None
    
    # Setup environment for Linux
    os.environ['QNN_SDK_ROOT'] = str(qnn_sdk_root)
    os.environ['PYTHONPATH'] = f"{qnn_sdk_root}/lib/python:{os.environ.get('PYTHONPATH', '')}"
    os.environ['PATH'] = f"{qnn_sdk_root}/bin/x86_64-linux-clang:{os.environ.get('PATH', '')}"
    os.environ['LD_LIBRARY_PATH'] = f"{qnn_sdk_root}/lib/x86_64-linux-clang:{os.environ.get('LD_LIBRARY_PATH', '')}"
    
    # Add Linux-specific Python library paths
    linux_lib_dir = qnn_sdk_root / "lib" / "python" / "qti" / "aisw" / "converters" / "common" / "linux-x86_64"
    if linux_lib_dir.exists():
        sys.path.insert(0, str(linux_lib_dir))
    
    sys.path.insert(0, str(qnn_sdk_root / "lib" / "python"))
    
    return qnn_sdk_root

def convert_gguf_to_qnn_linux():
    """Perform GGUF to QNN conversion using Linux tools"""
    
    print("=" * 70)
    print("🐧 Linux GGUF to QNN Context Binary Converter")
    print("=" * 70)
    
    # Setup environment
    qnn_sdk_root = setup_qnn_environment_linux()
    if not qnn_sdk_root:
        return 1
    
    # Verify Linux tools exist
    converter_path = qnn_sdk_root / "bin" / "x86_64-linux-clang" / "qairt-converter"
    ctx_gen_path = qnn_sdk_root / "bin" / "x86_64-linux-clang" / "qnn-context-binary-generator"
    htp_lib_path = qnn_sdk_root / "lib" / "x86_64-linux-clang" / "libQnnHtp.so"
    
    print(f"✅ QNN SDK Root: {qnn_sdk_root}")
    print(f"✅ Converter: {converter_path} ({'✓' if converter_path.exists() else '✗'})")
    print(f"✅ Context Gen: {ctx_gen_path} ({'✓' if ctx_gen_path.exists() else '✗'})")
    print(f"✅ HTP Library: {htp_lib_path} ({'✓' if htp_lib_path.exists() else '✗'})")
    
    # Check for Python libraries
    linux_py_libs = qnn_sdk_root / "lib" / "python" / "qti" / "aisw" / "converters" / "common" / "linux-x86_64"
    print(f"✅ Python Libs: {linux_py_libs} ({'✓' if linux_py_libs.exists() else '✗'})")
    
    if linux_py_libs.exists():
        ir_graph_lib = linux_py_libs / "libPyIrGraph.so"
        print(f"✅ IrGraph Lib: {ir_graph_lib.name} ({'✓' if ir_graph_lib.exists() else '✗'})")
    
    # Paths
    script_dir = Path(__file__).parent
    gguf_path = script_dir / "models" / "DeepSeek-R1-Distill-Qwen-1.5B-Q4_0.gguf"
    output_dir = script_dir / "temp"
    output_dir.mkdir(exist_ok=True)
    
    dlc_path = output_dir / "deepseek_qwen_linux.dlc"
    bin_path = output_dir / "deepseek_qwen_linux.bin"
    
    # Check model
    if not gguf_path.exists():
        print(f"❌ GGUF model not found at: {gguf_path}")
        return 1
    
    print()
    print(f"📦 Input model: {gguf_path}")
    print(f"📊 Model size: {gguf_path.stat().st_size / (1024**3):.2f} GB")
    print(f"📁 Output directory: {output_dir}")
    print()
    
    # Step 1: Convert GGUF to DLC using Linux converter
    print("🔄 Step 1: Converting GGUF to DLC (Linux)")
    print("-" * 70)
    
    cmd = [
        sys.executable,
        str(converter_path),
        "--input_network", str(gguf_path),
        "--output_path", str(dlc_path),
        "--input_layout", "input_ids,NONTRIVIAL",
        "--input_layout", "attention_mask,NONTRIVIAL", 
        "--preserve_io", "datatype,input_ids,attention_mask",
        "--float_fallback",    # Preserve Q4_0 quantization
        "--float_bitwidth", "16",  # Use FP16 for dequantized values
        "--enable_cpu_fallback"    # Handle unsupported ops
    ]
    
    print("Command:")
    cmd_str = " \\\\\n    ".join([cmd[0]] + [f'"{arg}"' if ' ' in arg else arg for arg in cmd[1:]])
    print(f"    {cmd_str}")
    print()
    
    try:
        print("🚀 Running conversion...")
        result = subprocess.run(
            cmd, 
            capture_output=True, 
            text=True, 
            timeout=600  # 10 minute timeout
        )
        
        if result.returncode == 0:
            print("✅ DLC conversion successful!")
            if dlc_path.exists():
                size_mb = dlc_path.stat().st_size / (1024**2)
                print(f"   📄 DLC file: {dlc_path}")
                print(f"   📊 DLC size: {size_mb:.1f} MB")
            else:
                print("⚠️  DLC file not found, but command succeeded")
                
            # Show some output
            if result.stdout:
                print("   Output preview:", result.stdout[:200] + "..." if len(result.stdout) > 200 else result.stdout)
        else:
            print(f"❌ DLC conversion failed (exit code: {result.returncode})")
            print("Error output:")
            print(result.stderr[:1000] if result.stderr else "No error output")
            return 1
            
    except subprocess.TimeoutExpired:
        print("⚠️ Conversion timed out after 10 minutes")
        return 1
    except Exception as e:
        print(f"⚠️ Error running converter: {e}")
        return 1
    
    # Step 2: Generate context binary (if DLC was created)
    if dlc_path.exists():
        print()
        print("🔄 Step 2: Generating Context Binary (Linux)")
        print("-" * 70)
        
        cmd = [
            str(ctx_gen_path),
            "--dlc_path", str(dlc_path),
            "--backend", str(htp_lib_path),
            "--binary_file", str(bin_path),
            "--output_dir", str(output_dir),
            "--target_arch", "sm8650"  # Snapdragon 8 Gen 3
        ]
        
        print("Command:")
        cmd_str = " \\\\\n    ".join([f'"{arg}"' if ' ' in arg else arg for arg in cmd])
        print(f"    {cmd_str}")
        print()
        
        try:
            print("🚀 Running context binary generation...")
            result = subprocess.run(
                cmd, 
                capture_output=True, 
                text=True, 
                timeout=300  # 5 minute timeout
            )
            
            if result.returncode == 0:
                print("✅ Context binary generation successful!")
                if bin_path.exists():
                    size_mb = bin_path.stat().st_size / (1024**2)
                    print(f"   📄 Binary file: {bin_path}")
                    print(f"   📊 Binary size: {size_mb:.1f} MB")
                else:
                    print("⚠️  Binary file not found, but command succeeded")
            else:
                print(f"⚠️ Context binary generation failed (exit code: {result.returncode})")
                print("This is optional, continuing...")
                if result.stderr:
                    print("Error:", result.stderr[:500])
                    
        except subprocess.TimeoutExpired:
            print("⚠️ Context binary generation timed out")
        except Exception as e:
            print(f"⚠️ Error running context generator: {e}")
    
    # Summary
    print()
    print("=" * 70)
    print("📋 Linux Conversion Summary")
    print("=" * 70)
    
    print(f"🔹 Method: Native GGUF Support (Linux x86_64)")
    print(f"🔹 Input: {gguf_path.name} ({gguf_path.stat().st_size / (1024**3):.2f} GB)")
    
    if dlc_path.exists():
        print(f"✅ DLC: {dlc_path.name} ({dlc_path.stat().st_size / (1024**2):.1f} MB)")
    else:
        print("❌ DLC: Not generated")
        
    if bin_path.exists():
        print(f"✅ Binary: {bin_path.name} ({bin_path.stat().st_size / (1024**2):.1f} MB)")
    else:
        print("⚠️  Binary: Not generated")
    
    print()
    if dlc_path.exists():
        print("🎉 Conversion successful! Model ready for NPU deployment.")
        print("   Files available in temp/ directory")
        return 0
    else:
        print("⚠️ Partial success - check error messages above")
        return 1

if __name__ == "__main__":
    sys.exit(convert_gguf_to_qnn_linux())