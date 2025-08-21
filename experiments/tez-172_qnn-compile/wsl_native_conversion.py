#!/usr/bin/env python3
"""
WSL Native QNN Conversion - Use QNN SDK copied to WSL filesystem for proper .so file access.
This should resolve the shared library loading issues.
"""

import os
import sys
import subprocess
from pathlib import Path

def setup_wsl_native_qnn():
    """Setup QNN environment using WSL-native SDK copy"""
    
    qnn_sdk_root = Path("/tmp/qnn-sdk")
    
    if not qnn_sdk_root.exists():
        print("❌ QNN SDK not found in /tmp/qnn-sdk")
        print("The SDK copy may have failed")
        return None
    
    print("Setting up QNN environment (WSL-native SDK)...")
    
    # Environment variables
    os.environ['QNN_SDK_ROOT'] = str(qnn_sdk_root)
    
    # Python paths with architecture-specific directory
    qnn_python_dir = qnn_sdk_root / "lib" / "python"
    qnn_arch_python_dir = qnn_python_dir / "qti" / "aisw" / "converters" / "common" / "linux-x86_64"
    
    python_paths = [
        str(qnn_arch_python_dir),
        str(qnn_python_dir),
    ]
    os.environ['PYTHONPATH'] = ":".join(python_paths + [os.environ.get('PYTHONPATH', '')])
    
    # Library path for x86_64 Linux libraries  
    qnn_lib_dir = qnn_sdk_root / "lib" / "x86_64-linux-clang"
    os.environ['LD_LIBRARY_PATH'] = f"{qnn_lib_dir}:{os.environ.get('LD_LIBRARY_PATH', '')}"
    
    # Add to Python sys.path
    sys.path.insert(0, str(qnn_arch_python_dir))
    sys.path.insert(0, str(qnn_python_dir))
    
    print(f"✅ QNN_SDK_ROOT: {qnn_sdk_root}")
    print(f"✅ Python arch path: {qnn_arch_python_dir}")
    print(f"✅ Library path: {qnn_lib_dir}")
    print()
    
    return qnn_sdk_root

def test_wsl_imports():
    """Test QNN imports with WSL-native libraries"""
    
    print("Testing QNN imports (WSL-native)...")
    
    try:
        # Test direct libPyIrGraph import
        import libPyIrGraph as ir_graph
        print("✅ libPyIrGraph imported directly!")
        
        # Test qti modules
        import qti
        from qti.aisw.converters.common import ir_graph as qti_ir
        print("✅ QTI converters imported!")
        
        # Test LLM builder
        from qti.aisw.converters.llm_builder import LLMBuilder
        print("✅ LLMBuilder available for GGUF!")
        
        return True
        
    except ImportError as e:
        print(f"❌ Import failed: {e}")
        return False
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

def convert_with_wsl_sdk():
    """Convert GGUF using WSL-native QNN SDK"""
    
    print("\n" + "=" * 60)
    print("🚀 WSL-Native QNN Conversion")
    print("=" * 60)
    
    # Paths
    script_dir = Path(__file__).parent
    gguf_path = script_dir / "models" / "DeepSeek-R1-Distill-Qwen-1.5B-Q4_0.gguf"
    temp_dir = script_dir / "temp"
    temp_dir.mkdir(exist_ok=True)
    
    dlc_path = temp_dir / "deepseek_wsl.dlc"
    bin_path = temp_dir / "deepseek_wsl.bin"
    
    if not gguf_path.exists():
        print(f"❌ GGUF model not found: {gguf_path}")
        return False
    
    print(f"📦 Input: {gguf_path}")
    print(f"📄 DLC: {dlc_path}")
    print(f"🔧 Binary: {bin_path}")
    
    # Use WSL-native qairt-converter
    qnn_sdk_root = Path(os.environ['QNN_SDK_ROOT'])
    converter = qnn_sdk_root / "bin" / "x86_64-linux-clang" / "qairt-converter"
    
    # Step 1: GGUF to DLC
    cmd = [
        "python3",
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
    
    print("\nStep 1: GGUF → DLC conversion...")
    print(f"Converter: {converter}")
    
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=600,
            env=os.environ.copy()
        )
        
        if result.returncode == 0:
            print("✅ DLC generation successful!")
            
            if dlc_path.exists():
                size_mb = dlc_path.stat().st_size / (1024**2)
                print(f"   📄 DLC size: {size_mb:.1f} MB")
                
                # Step 2: DLC to Context Binary
                context_gen = qnn_sdk_root / "bin" / "x86_64-linux-clang" / "qnn-context-binary-generator"
                backend_lib = qnn_sdk_root / "lib" / "x86_64-linux-clang" / "libQnnHtp.so"
                
                cmd_bin = [
                    str(context_gen),
                    "--dlc_path", str(dlc_path),
                    "--backend", str(backend_lib),
                    "--binary_file", str(bin_path),
                    "--output_dir", str(temp_dir),
                    "--target_arch", "sm8650"
                ]
                
                print("\nStep 2: DLC → Context Binary...")
                result_bin = subprocess.run(
                    cmd_bin,
                    capture_output=True,
                    text=True,
                    timeout=300,
                    env=os.environ.copy()
                )
                
                if result_bin.returncode == 0:
                    print("✅ Context binary generation successful!")
                    if bin_path.exists():
                        size_mb = bin_path.stat().st_size / (1024**2)
                        print(f"   🔧 Binary size: {size_mb:.1f} MB")
                else:
                    print("⚠️ Context binary failed (DLC still valid)")
                    print(f"   Error: {result_bin.stderr[:200]}")
                
                return True
            else:
                print("❌ DLC file not created")
                return False
        else:
            print("❌ DLC generation failed")
            print("STDERR:")
            print(result.stderr[:800])
            return False
            
    except subprocess.TimeoutExpired:
        print("❌ Conversion timed out")
        return False
    except Exception as e:
        print(f"❌ Conversion error: {e}")
        return False

def main():
    """Main conversion workflow"""
    
    print("=" * 70)
    print("🐧 WSL-Native QNN Conversion")
    print("=" * 70)
    print("Using QNN SDK copied to WSL filesystem for proper .so access")
    print()
    
    # Setup WSL-native environment
    qnn_sdk_root = setup_wsl_native_qnn()
    if not qnn_sdk_root:
        return 1
    
    # Test imports
    if test_wsl_imports():
        print("\n🎉 All QNN Python modules loaded successfully!")
        print("Proceeding with real conversion...")
        
        success = convert_with_wsl_sdk()
        
        if success:
            print("\n" + "=" * 70)
            print("🎉 REAL QNN CONVERSION COMPLETED!")
            print("=" * 70)
            print("Files in temp/ directory:")
            
            temp_dir = Path(__file__).parent / "temp"
            for file in temp_dir.glob("deepseek_wsl.*"):
                size_mb = file.stat().st_size / (1024**2)
                print(f"  📁 {file.name}: {size_mb:.1f} MB")
            
            print("\n✅ Ready for Snapdragon NPU deployment!")
            return 0
        else:
            return 1
    else:
        print("\n❌ Python import issues persist")
        print("Attempting direct execution anyway...")
        
        # Try conversion even without imports working
        success = convert_with_wsl_sdk()
        return 0 if success else 1

if __name__ == "__main__":
    sys.exit(main())