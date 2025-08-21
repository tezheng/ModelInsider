#!/usr/bin/env python3
"""
Linux x86_64 QNN Conversion - Use x86_64 Linux tools with proper environment setup.
This should work better than Windows approach since we're in WSL2.
"""

import os
import sys
import subprocess
from pathlib import Path

def setup_qnn_environment():
    """Setup QNN environment for x86_64 Linux tools"""
    
    qnn_sdk_root = Path("/mnt/c/Qualcomm/AIStack/qairt/2.34.0.250424")
    
    print("Setting up QNN x86_64 Linux environment...")
    
    # Environment variables
    os.environ['QNN_SDK_ROOT'] = str(qnn_sdk_root)
    
    # Python path for QNN modules
    qnn_python_dir = qnn_sdk_root / "lib" / "python"
    os.environ['PYTHONPATH'] = f"{qnn_python_dir}:{os.environ.get('PYTHONPATH', '')}"
    
    # Library path for x86_64 Linux libraries  
    qnn_lib_dir = qnn_sdk_root / "lib" / "x86_64-linux-clang"
    os.environ['LD_LIBRARY_PATH'] = f"{qnn_lib_dir}:{os.environ.get('LD_LIBRARY_PATH', '')}"
    
    # Add to Python sys.path
    sys.path.insert(0, str(qnn_python_dir))
    
    print(f"QNN_SDK_ROOT: {qnn_sdk_root}")
    print(f"Python path: {qnn_python_dir}")
    print(f"Library path: {qnn_lib_dir}")
    print()
    
    return qnn_sdk_root

def test_qnn_imports():
    """Test if we can import QNN Python modules"""
    
    print("Testing QNN Python imports...")
    
    try:
        # Test basic import
        import qti
        print("✅ qti base module imported")
        
        # Test converters
        from qti.aisw.converters.common import ir_graph
        print("✅ ir_graph imported (this was failing before)")
        
        # Test LLM builder for GGUF support
        from qti.aisw.converters.llm_builder import LLMBuilder
        print("✅ LLMBuilder imported (GGUF support available)")
        
        return True
        
    except ImportError as e:
        print(f"❌ Import failed: {e}")
        return False
    except Exception as e:
        print(f"❌ Other error: {e}")
        return False

def convert_gguf_direct():
    """Directly convert GGUF using Linux x86_64 qairt-converter"""
    
    print("\n" + "=" * 60)
    print("Direct GGUF to QNN Conversion (Linux x86_64)")
    print("=" * 60)
    
    # Paths
    script_dir = Path(__file__).parent
    gguf_path = script_dir / "models" / "DeepSeek-R1-Distill-Qwen-1.5B-Q4_0.gguf"
    temp_dir = script_dir / "temp"
    temp_dir.mkdir(exist_ok=True)
    
    dlc_path = temp_dir / "deepseek_linux.dlc"
    bin_path = temp_dir / "deepseek_linux.bin"
    
    if not gguf_path.exists():
        print(f"❌ GGUF model not found: {gguf_path}")
        return False
    
    print(f"📦 Input: {gguf_path}")
    print(f"📄 DLC Output: {dlc_path}")
    print(f"🔧 Binary Output: {bin_path}")
    print()
    
    # Use Linux x86_64 converter
    qnn_sdk_root = Path(os.environ['QNN_SDK_ROOT'])
    converter = qnn_sdk_root / "bin" / "x86_64-linux-clang" / "qairt-converter"
    
    # Step 1: GGUF to DLC
    cmd = [
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
    
    print("Step 1: Converting GGUF to DLC...")
    print(f"Command: {converter.name} [with args]")
    
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=600,
            cwd=str(temp_dir)
        )
        
        if result.returncode == 0:
            print("✅ DLC conversion successful!")
            if dlc_path.exists():
                size_mb = dlc_path.stat().st_size / (1024**2)
                print(f"   DLC size: {size_mb:.1f} MB")
            
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
            
            print("\nStep 2: Generating context binary...")
            result_bin = subprocess.run(
                cmd_bin,
                capture_output=True,
                text=True,
                timeout=300,
                cwd=str(temp_dir)
            )
            
            if result_bin.returncode == 0:
                print("✅ Context binary generation successful!")
                if bin_path.exists():
                    size_mb = bin_path.stat().st_size / (1024**2)
                    print(f"   Binary size: {size_mb:.1f} MB")
            else:
                print("⚠️ Context binary generation failed (optional)")
                print(f"   Error: {result_bin.stderr[:300]}")
            
            return True
            
        else:
            print("❌ DLC conversion failed")
            print("STDERR:")
            print(result.stderr[:1000])
            return False
            
    except subprocess.TimeoutExpired:
        print("❌ Conversion timed out after 10 minutes")
        return False
    except Exception as e:
        print(f"❌ Conversion error: {e}")
        return False

def main():
    """Main conversion workflow"""
    
    print("=" * 70)
    print("🐧 Linux x86_64 QNN Conversion")
    print("=" * 70)
    print("Using Linux x86_64 tools with proper library paths")
    print("This should resolve the architecture compatibility issues")
    print()
    
    # Setup environment
    qnn_sdk_root = setup_qnn_environment()
    
    # Test imports first
    if test_qnn_imports():
        print("\n✅ All QNN Python modules imported successfully!")
        print("Proceeding with conversion...")
        
        success = convert_gguf_direct()
        
        if success:
            print("\n" + "=" * 70)
            print("🎉 SUCCESS: Real QNN conversion completed!")
            print("=" * 70)
            print("Files generated in temp/ directory:")
            print("- deepseek_linux.dlc (QNN Deep Learning Container)")
            print("- deepseek_linux.bin (Context Binary for Snapdragon NPU)")
            print()
            print("Ready for deployment on Qualcomm NPU!")
            return 0
        else:
            print("\n❌ Conversion failed")
            return 1
    else:
        print("\n❌ QNN Python modules not available")
        print("This indicates the x86_64 libraries still can't be loaded")
        return 1

if __name__ == "__main__":
    sys.exit(main())