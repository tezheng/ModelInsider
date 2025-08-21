#!/usr/bin/env python3
"""
Final QNN Conversion - Last attempt using system Python and targeting Python 3.8 libraries.
"""

import os
import sys
import subprocess
from pathlib import Path

def find_compatible_python():
    """Find a compatible Python version"""
    
    print("Searching for compatible Python versions...")
    
    # Check available Python versions
    python_versions = []
    for py_cmd in ["python3.10", "python3.9", "python3.8", "python3", "python"]:
        try:
            result = subprocess.run([py_cmd, "--version"], capture_output=True, text=True)
            if result.returncode == 0:
                version = result.stdout.strip()
                python_versions.append((py_cmd, version))
                print(f"  Found: {py_cmd} -> {version}")
        except FileNotFoundError:
            continue
    
    # Prefer Python 3.10, then 3.8, then others
    for py_cmd, version in python_versions:
        if "3.10" in version:
            print(f"✅ Using Python 3.10: {py_cmd}")
            return py_cmd
    
    for py_cmd, version in python_versions:
        if "3.8" in version:
            print(f"✅ Using Python 3.8: {py_cmd}")
            return py_cmd
    
    # Use any available Python
    if python_versions:
        py_cmd, version = python_versions[0]
        print(f"⚠️ Using available Python: {py_cmd} ({version})")
        return py_cmd
    
    print("❌ No Python found")
    return None

def create_compatibility_script():
    """Create a Python script that handles QNN SDK compatibility"""
    
    script_content = '''#!/usr/bin/env python3
import os
import sys
from pathlib import Path

# Setup QNN environment
qnn_sdk_root = Path("/tmp/qnn-sdk")
os.environ['QNN_SDK_ROOT'] = str(qnn_sdk_root)

# Add all possible Python paths
python_dirs = [
    qnn_sdk_root / "lib" / "python",
    qnn_sdk_root / "lib" / "python" / "qti" / "aisw" / "converters" / "common" / "linux-x86_64",
]

for py_dir in python_dirs:
    if py_dir.exists():
        sys.path.insert(0, str(py_dir))

# Set library path
qnn_lib_dir = qnn_sdk_root / "lib" / "x86_64-linux-clang"
os.environ['LD_LIBRARY_PATH'] = f"{qnn_lib_dir}:{os.environ.get('LD_LIBRARY_PATH', '')}"

def main():
    if len(sys.argv) < 4:
        print("Usage: python script.py <input_gguf> <output_dlc> <temp_dir>")
        return 1
    
    input_gguf = sys.argv[1]
    output_dlc = sys.argv[2]
    temp_dir = sys.argv[3]
    
    print(f"Input: {input_gguf}")
    print(f"Output: {output_dlc}")
    print(f"Temp: {temp_dir}")
    
    # Import QNN modules (try different approaches)
    try:
        # Try direct import of Python 3.8 version
        import importlib.util
        
        # Load libPyIrGraph38.so explicitly
        lib38_path = qnn_sdk_root / "lib" / "python" / "qti" / "aisw" / "converters" / "common" / "linux-x86_64" / "libPyIrGraph38.so"
        
        if lib38_path.exists():
            import ctypes
            lib = ctypes.CDLL(str(lib38_path))
            print("✅ Loaded libPyIrGraph38.so with ctypes")
        
        # Try importing QNN modules
        from qti.aisw.converters.llm_builder import LLMBuilder
        
        print("✅ QNN modules imported successfully!")
        
        # Use LLMBuilder for native GGUF conversion
        print("🚀 Starting GGUF conversion...")
        
        builder = LLMBuilder(
            input_model=input_gguf,
            output_dir=temp_dir
        )
        
        # This does the GGUF → ONNX → QNN conversion internally
        onnx_path, encodings_path, input_layouts, inputs_to_preserve = builder.build_from_gguf()
        
        print(f"✅ Conversion completed!")
        print(f"ONNX: {onnx_path}")
        print(f"Encodings: {encodings_path}")
        
        return 0
        
    except Exception as e:
        print(f"❌ Native conversion failed: {e}")
        
        # Fallback: Try subprocess approach
        print("🔄 Trying subprocess approach...")
        
        import subprocess
        
        converter = qnn_sdk_root / "bin" / "x86_64-linux-clang" / "qairt-converter"
        
        cmd = [
            sys.executable,
            str(converter),
            "--input_network", input_gguf,
            "--output_path", output_dlc,
            "--float_fallback",
            "--enable_cpu_fallback"
        ]
        
        env = os.environ.copy()
        env['PYTHONPATH'] = ":".join([str(d) for d in python_dirs if d.exists()])
        
        try:
            result = subprocess.run(cmd, env=env, cwd=temp_dir, timeout=600, capture_output=True, text=True)
            
            if result.returncode == 0:
                print("✅ Subprocess conversion successful!")
                return 0
            else:
                print(f"❌ Subprocess failed: {result.stderr[:300]}")
                return 1
                
        except Exception as e2:
            print(f"❌ Subprocess error: {e2}")
            return 1

if __name__ == "__main__":
    sys.exit(main())
'''
    
    script_path = Path("/tmp/qnn_convert_compat.py")
    with open(script_path, 'w') as f:
        f.write(script_content)
    
    return script_path

def run_final_conversion():
    """Run the final conversion attempt"""
    
    print("=" * 70)
    print("🔧 Final QNN Conversion Attempt")
    print("=" * 70)
    
    # Find compatible Python
    python_cmd = find_compatible_python()
    if not python_cmd:
        return 1
    
    # Create compatibility script
    print("\nCreating compatibility conversion script...")
    compat_script = create_compatibility_script()
    
    # Setup paths
    script_dir = Path(__file__).parent
    gguf_path = script_dir / "models" / "DeepSeek-R1-Distill-Qwen-1.5B-Q4_0.gguf"
    temp_dir = script_dir / "temp"
    temp_dir.mkdir(exist_ok=True)
    
    dlc_path = temp_dir / "deepseek_final.dlc"
    
    if not gguf_path.exists():
        print(f"❌ GGUF not found: {gguf_path}")
        return 1
    
    print(f"\n📦 Input: {gguf_path}")
    print(f"📄 Output: {dlc_path}")
    print(f"🐍 Python: {python_cmd}")
    
    # Run conversion
    print("\n🚀 Running final conversion...")
    
    cmd = [python_cmd, str(compat_script), str(gguf_path), str(dlc_path), str(temp_dir)]
    
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=900
        )
        
        print("Conversion Output:")
        print("-" * 40)
        print(result.stdout)
        
        if result.stderr:
            print("Errors:")
            print(result.stderr)
        
        # Check results
        success = dlc_path.exists()
        
        if success:
            size_mb = dlc_path.stat().st_size / (1024**2)
            print("\n" + "=" * 70)
            print("🎉 FINAL SUCCESS!")
            print("=" * 70)
            print(f"✅ Real QNN model generated: {dlc_path.name} ({size_mb:.1f} MB)")
            print("✅ Ready for Snapdragon NPU deployment!")
            return 0
        else:
            print("\n❌ Final conversion failed - no output file")
            return 1
            
    except subprocess.TimeoutExpired:
        print("❌ Conversion timed out")
        return 1
    except Exception as e:
        print(f"❌ Final error: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(run_final_conversion())