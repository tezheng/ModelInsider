#!/usr/bin/env python3
"""
Python 3.10 QNN Conversion - Using uv-installed Python 3.10 for QNN SDK compatibility.
"""

import os
import sys
import subprocess
from pathlib import Path

def setup_python310_qnn_env():
    """Setup QNN environment with Python 3.10"""
    
    qnn_sdk_root = Path("/tmp/qnn-sdk")
    
    print("Setting up QNN environment with Python 3.10...")
    
    # Environment variables
    os.environ['QNN_SDK_ROOT'] = str(qnn_sdk_root)
    
    # Python paths
    qnn_python_dir = qnn_sdk_root / "lib" / "python"
    qnn_arch_python_dir = qnn_python_dir / "qti" / "aisw" / "converters" / "common" / "linux-x86_64"
    
    python_paths = [
        str(qnn_arch_python_dir),
        str(qnn_python_dir),
    ]
    os.environ['PYTHONPATH'] = ":".join(python_paths)
    
    # Library path
    qnn_lib_dir = qnn_sdk_root / "lib" / "x86_64-linux-clang"
    os.environ['LD_LIBRARY_PATH'] = f"{qnn_lib_dir}:{os.environ.get('LD_LIBRARY_PATH', '')}"
    
    print(f"✅ QNN_SDK_ROOT: {qnn_sdk_root}")
    print(f"✅ Python paths configured")
    print(f"✅ Library path: {qnn_lib_dir}")
    
    return qnn_sdk_root

def find_uv_python310():
    """Find the uv-installed Python 3.10"""
    
    print("Locating uv-installed Python 3.10...")
    
    try:
        # Get uv python path
        result = subprocess.run(["uv", "python", "find", "3.10"], capture_output=True, text=True)
        if result.returncode == 0:
            python310_path = result.stdout.strip()
            print(f"✅ Found Python 3.10: {python310_path}")
            
            # Verify version
            version_result = subprocess.run([python310_path, "--version"], capture_output=True, text=True)
            if version_result.returncode == 0:
                print(f"✅ Version: {version_result.stdout.strip()}")
                return python310_path
        
        print("❌ uv Python 3.10 not found")
        return None
        
    except Exception as e:
        print(f"❌ Error finding Python 3.10: {e}")
        return None

def test_python310_qnn_imports(python310_path):
    """Test QNN imports with Python 3.10"""
    
    print("\nTesting QNN imports with Python 3.10...")
    
    test_script = """
import sys
import os

# Setup paths
sys.path.insert(0, "/tmp/qnn-sdk/lib/python/qti/aisw/converters/common/linux-x86_64")
sys.path.insert(0, "/tmp/qnn-sdk/lib/python")

try:
    import libPyIrGraph as ir_graph
    print("✅ libPyIrGraph imported successfully!")
    
    import qti
    from qti.aisw.converters.common import ir_graph as qti_ir
    print("✅ QTI converters imported!")
    
    from qti.aisw.converters.llm_builder import LLMBuilder
    print("✅ LLMBuilder available!")
    
    print("SUCCESS: All QNN modules imported with Python 3.10!")
    
except Exception as e:
    print(f"ERROR: {e}")
    sys.exit(1)
"""
    
    try:
        result = subprocess.run(
            [python310_path, "-c", test_script],
            env=os.environ.copy(),
            capture_output=True,
            text=True,
            timeout=30
        )
        
        print("Python 3.10 Import Test:")
        print("-" * 40)
        print(result.stdout)
        if result.stderr:
            print("Errors:")
            print(result.stderr)
        
        return result.returncode == 0
        
    except Exception as e:
        print(f"❌ Import test failed: {e}")
        return False

def convert_with_python310(python310_path):
    """Convert GGUF using Python 3.10 and QNN SDK"""
    
    print("\n" + "=" * 60)
    print("🚀 Python 3.10 QNN Conversion")
    print("=" * 60)
    
    # Paths
    script_dir = Path(__file__).parent
    gguf_path = script_dir / "models" / "DeepSeek-R1-Distill-Qwen-1.5B-Q4_0.gguf"
    temp_dir = script_dir / "temp"
    temp_dir.mkdir(exist_ok=True)
    
    dlc_path = temp_dir / "deepseek_py310.dlc"
    bin_path = temp_dir / "deepseek_py310.bin"
    
    if not gguf_path.exists():
        print(f"❌ GGUF not found: {gguf_path}")
        return False
    
    print(f"📦 Input: {gguf_path}")
    print(f"📄 DLC: {dlc_path}")
    print(f"🔧 Binary: {bin_path}")
    print(f"🐍 Python: {python310_path}")
    
    # Use qairt-converter with Python 3.10
    qnn_sdk_root = Path(os.environ['QNN_SDK_ROOT'])
    converter = qnn_sdk_root / "bin" / "x86_64-linux-clang" / "qairt-converter"
    
    cmd = [
        python310_path,
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
    
    print("\nStep 1: GGUF → DLC (Python 3.10)...")
    
    try:
        result = subprocess.run(
            cmd,
            env=os.environ.copy(),
            capture_output=True,
            text=True,
            timeout=600,
            cwd=str(temp_dir)
        )
        
        if result.returncode == 0:
            print("✅ DLC conversion successful!")
            
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
                    env=os.environ.copy(),
                    capture_output=True,
                    text=True,
                    timeout=300
                )
                
                if result_bin.returncode == 0:
                    print("✅ Context binary successful!")
                    if bin_path.exists():
                        size_mb = bin_path.stat().st_size / (1024**2)
                        print(f"   🔧 Binary size: {size_mb:.1f} MB")
                else:
                    print("⚠️ Context binary failed (DLC still valid)")
                    if result_bin.stderr:
                        print(f"   Error: {result_bin.stderr[:200]}")
                
                return True
            else:
                print("❌ No DLC file generated")
                return False
        else:
            print("❌ DLC conversion failed")
            print("STDOUT:", result.stdout[:500] if result.stdout else "None")
            print("STDERR:", result.stderr[:500] if result.stderr else "None")
            return False
            
    except subprocess.TimeoutExpired:
        print("❌ Conversion timed out")
        return False
    except Exception as e:
        print(f"❌ Conversion error: {e}")
        return False

def main():
    """Main Python 3.10 conversion workflow"""
    
    print("=" * 70)
    print("🐍 Python 3.10 QNN Conversion")
    print("=" * 70)
    print("Using uv-installed Python 3.10 for QNN SDK compatibility")
    print()
    
    # Setup environment
    setup_python310_qnn_env()
    
    # Find Python 3.10
    python310_path = find_uv_python310()
    if not python310_path:
        return 1
    
    # Test imports with Python 3.10
    if test_python310_qnn_imports(python310_path):
        print("\n🎉 QNN modules load successfully with Python 3.10!")
        print("Proceeding with real conversion...")
        
        success = convert_with_python310(python310_path)
        
        if success:
            print("\n" + "=" * 70)
            print("🎉 REAL QNN CONVERSION SUCCESS!")
            print("=" * 70)
            print("Python 3.10 + QNN SDK compatibility resolved!")
            print()
            
            temp_dir = Path(__file__).parent / "temp"
            for file in temp_dir.glob("deepseek_py310.*"):
                size_mb = file.stat().st_size / (1024**2)
                print(f"✅ {file.name}: {size_mb:.1f} MB")
            
            print("\n🚀 Ready for Snapdragon NPU deployment!")
            return 0
        else:
            return 1
    else:
        print("\n❌ QNN imports still failing with Python 3.10")
        print("Attempting conversion anyway...")
        
        success = convert_with_python310(python310_path)
        return 0 if success else 1

if __name__ == "__main__":
    sys.exit(main())