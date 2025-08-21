#!/usr/bin/env python3
"""
QNN Conversion with Python 3.10 + NumPy environment
"""

import os
import sys
import subprocess
from pathlib import Path

def setup_qnn_environment():
    """Setup QNN environment"""
    
    qnn_sdk_root = Path("/tmp/qnn-sdk")
    
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
    
    print(f"✅ QNN environment configured")
    
    return qnn_sdk_root

def convert_with_numpy_env():
    """Convert using Python 3.10 environment with numpy"""
    
    print("=" * 70)
    print("🔧 QNN Conversion with NumPy Environment")
    print("=" * 70)
    
    # Setup environment
    setup_qnn_environment()
    
    # Use Python 3.10 with numpy
    python310_path = "/tmp/qnn-py310-env/bin/python"
    
    if not Path(python310_path).exists():
        print(f"❌ Python 3.10 environment not found: {python310_path}")
        return False
    
    # Verify numpy is available
    try:
        result = subprocess.run([python310_path, "-c", "import numpy; print('NumPy version:', numpy.__version__)"], 
                              capture_output=True, text=True)
        if result.returncode == 0:
            print(f"✅ NumPy available: {result.stdout.strip()}")
        else:
            print("❌ NumPy not available")
            return False
    except Exception as e:
        print(f"❌ Error checking NumPy: {e}")
        return False
    
    # Paths
    script_dir = Path(__file__).parent
    gguf_path = script_dir / "models" / "DeepSeek-R1-Distill-Qwen-1.5B-Q4_0.gguf"
    temp_dir = script_dir / "temp"
    temp_dir.mkdir(exist_ok=True)
    
    dlc_path = temp_dir / "deepseek_numpy.dlc"
    
    if not gguf_path.exists():
        print(f"❌ GGUF not found: {gguf_path}")
        return False
    
    print(f"\n📦 Input: {gguf_path}")
    print(f"📄 Output: {dlc_path}")
    print(f"🐍 Python: {python310_path}")
    
    # Run conversion
    qnn_sdk_root = Path(os.environ['QNN_SDK_ROOT'])
    converter = qnn_sdk_root / "bin" / "x86_64-linux-clang" / "qairt-converter"
    
    cmd = [
        python310_path,
        str(converter),
        "--input_network", str(gguf_path),
        "--output_path", str(dlc_path),
        "--float_fallback",
        "--enable_cpu_fallback"
    ]
    
    print("\n🚀 Running QNN conversion...")
    
    try:
        result = subprocess.run(
            cmd,
            env=os.environ.copy(),
            capture_output=True,
            text=True,
            timeout=600,
            cwd=str(temp_dir)
        )
        
        print("Conversion Output:")
        print("-" * 50)
        if result.stdout:
            print("STDOUT:")
            print(result.stdout[:1000])
        
        if result.stderr:
            print("STDERR:")  
            print(result.stderr[:1000])
        
        if result.returncode == 0:
            print("\n✅ Conversion successful!")
            if dlc_path.exists():
                size_mb = dlc_path.stat().st_size / (1024**2)
                print(f"📄 DLC created: {dlc_path.name} ({size_mb:.1f} MB)")
                return True
        else:
            print(f"\n❌ Conversion failed (return code: {result.returncode})")
        
        return False
        
    except subprocess.TimeoutExpired:
        print("❌ Conversion timed out")
        return False
    except Exception as e:
        print(f"❌ Conversion error: {e}")
        return False

def main():
    """Main function"""
    
    success = convert_with_numpy_env()
    
    if success:
        print("\n" + "=" * 70)
        print("🎉 CONVERSION SUCCESS!")
        print("=" * 70)
        print("Real .dlc file generated using Python 3.10 + NumPy!")
        return 0
    else:
        print("\n❌ Conversion failed")
        return 1

if __name__ == "__main__":
    sys.exit(main())