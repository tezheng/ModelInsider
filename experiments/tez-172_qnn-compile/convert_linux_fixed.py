#!/usr/bin/env python3
"""
Linux GGUF to QNN conversion with proper library path setup.
Fixed version that handles the libPyIrGraph import issue.
"""

import os
import sys
import subprocess
from pathlib import Path

def setup_qnn_environment_fixed():
    """Setup QNN SDK environment with proper library loading"""
    qnn_sdk_root = Path("/mnt/c/Qualcomm/AIStack/qairt/2.34.0.250424")
    
    if not qnn_sdk_root.exists():
        print(f"❌ QNN SDK not found at {qnn_sdk_root}")
        return None
    
    print(f"🔧 Setting up QNN environment for Linux...")
    
    # Key insight: Set LD_LIBRARY_PATH for shared library loading
    linux_lib_dir = qnn_sdk_root / "lib" / "x86_64-linux-clang"
    
    # Setup critical environment variables
    env_vars = {
        'QNN_SDK_ROOT': str(qnn_sdk_root),
        'PYTHONPATH': f"{qnn_sdk_root}/lib/python:{os.environ.get('PYTHONPATH', '')}",
        'PATH': f"{qnn_sdk_root}/bin/x86_64-linux-clang:{os.environ.get('PATH', '')}",
        'LD_LIBRARY_PATH': f"{linux_lib_dir}:{os.environ.get('LD_LIBRARY_PATH', '')}",
    }
    
    for key, value in env_vars.items():
        os.environ[key] = value
        print(f"   {key}={value[:80]}{'...' if len(value) > 80 else ''}")
    
    # Critical: Add the platform-specific library directory to Python path
    # This matches what QNN's __init__.py does
    common_dir = qnn_sdk_root / "lib" / "python" / "qti" / "aisw" / "converters" / "common"
    linux_py_dir = common_dir / "linux-x86_64"
    
    if linux_py_dir.exists():
        # Insert at beginning to override any conflicts
        sys.path.insert(0, str(linux_py_dir))
        sys.path.insert(0, str(common_dir))
        print(f"   Added to Python path: {linux_py_dir}")
    
    # Also add main QNN Python directory
    qnn_py_dir = qnn_sdk_root / "lib" / "python"
    sys.path.insert(0, str(qnn_py_dir))
    
    return qnn_sdk_root

def test_qnn_imports():
    """Test if we can import QNN modules"""
    print("🧪 Testing QNN module imports...")
    
    try:
        # Try importing the problematic module directly
        from qti.aisw.converters.common import ir_graph
        print("   ✅ ir_graph imported successfully")
        return True
    except ImportError as e:
        print(f"   ❌ ir_graph import failed: {e}")
        
        # Try manual import approach
        try:
            import sys
            # Try importing the library directly
            py_version = sys.version_info
            if py_version.major == 3 and py_version.minor == 12:
                # For Python 3.12, try the generic version
                import libPyIrGraph as ir_graph_direct
                print("   ✅ libPyIrGraph imported directly (Python 3.12)")
                return True
            elif py_version.major == 3 and py_version.minor == 8:
                import libPyIrGraph38 as ir_graph_direct
                print("   ✅ libPyIrGraph38 imported directly")
                return True
            else:
                import libPyIrGraph as ir_graph_direct
                print(f"   ✅ libPyIrGraph imported directly (Python {py_version.major}.{py_version.minor})")
                return True
        except ImportError as e2:
            print(f"   ❌ Direct import also failed: {e2}")
            return False
    

def create_wrapper_script(qnn_sdk_root):
    """Create a wrapper script that bypasses Python import issues"""
    
    script_dir = Path(__file__).parent
    wrapper_path = script_dir / "temp" / "qnn_wrapper.sh"
    
    # Create wrapper script that properly sets up environment
    wrapper_content = f'''#!/bin/bash
# QNN Converter Wrapper Script
# Auto-generated to handle library path issues

export QNN_SDK_ROOT="{qnn_sdk_root}"
export PYTHONPATH="{qnn_sdk_root}/lib/python:$PYTHONPATH"
export PATH="{qnn_sdk_root}/bin/x86_64-linux-clang:$PATH"
export LD_LIBRARY_PATH="{qnn_sdk_root}/lib/x86_64-linux-clang:$LD_LIBRARY_PATH"

# Add platform-specific Python library to path
export PYTHONPATH="{qnn_sdk_root}/lib/python/qti/aisw/converters/common/linux-x86_64:$PYTHONPATH"

echo "🚀 Running QNN Converter with proper environment..."
echo "   QNN_SDK_ROOT: $QNN_SDK_ROOT"
echo "   Using converter: {qnn_sdk_root}/bin/x86_64-linux-clang/qairt-converter"

# Run the actual converter
exec python3 "{qnn_sdk_root}/bin/x86_64-linux-clang/qairt-converter" "$@"
'''
    
    wrapper_path.parent.mkdir(exist_ok=True)
    with open(wrapper_path, 'w') as f:
        f.write(wrapper_content)
    
    # Make executable
    wrapper_path.chmod(0o755)
    
    return wrapper_path

def convert_using_wrapper():
    """Convert using shell wrapper approach"""
    
    print("=" * 70)
    print("🔧 Linux GGUF to QNN Converter (Wrapper Method)")
    print("=" * 70)
    
    qnn_sdk_root = setup_qnn_environment_fixed()
    if not qnn_sdk_root:
        return 1
    
    # Test imports first
    imports_work = test_qnn_imports()
    
    script_dir = Path(__file__).parent
    gguf_path = script_dir / "models" / "DeepSeek-R1-Distill-Qwen-1.5B-Q4_0.gguf"
    output_dir = script_dir / "temp"
    output_dir.mkdir(exist_ok=True)
    
    dlc_path = output_dir / "deepseek_qwen_linux.dlc"
    
    if not gguf_path.exists():
        print(f"❌ GGUF model not found: {gguf_path}")
        return 1
    
    print()
    print(f"📦 Input: {gguf_path} ({gguf_path.stat().st_size / (1024**3):.2f} GB)")
    print(f"📁 Output: {dlc_path}")
    print()
    
    if imports_work:
        print("🎯 Method 1: Direct Python execution (imports working)")
        print("-" * 70)
        
        converter_path = qnn_sdk_root / "bin" / "x86_64-linux-clang" / "qairt-converter"
        
        cmd = [
            sys.executable,
            str(converter_path),
            "--input_network", str(gguf_path),
            "--output_path", str(dlc_path),
            "--input_layout", "input_ids,NONTRIVIAL",
            "--input_layout", "attention_mask,NONTRIVIAL", 
            "--preserve_io", "datatype,input_ids,attention_mask",
            "--float_fallback",
            "--float_bitwidth", "16",
            "--enable_cpu_fallback"
        ]
        
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
            
            if result.returncode == 0:
                print("✅ Direct conversion successful!")
                print(f"   DLC: {dlc_path} ({dlc_path.stat().st_size / (1024**2):.1f} MB)")
                return 0
            else:
                print(f"⚠️ Direct conversion failed: {result.stderr[:500]}")
                
        except Exception as e:
            print(f"⚠️ Direct conversion error: {e}")
    
    print()
    print("🎯 Method 2: Shell wrapper approach")
    print("-" * 70)
    
    # Create and use wrapper script
    wrapper_path = create_wrapper_script(qnn_sdk_root)
    print(f"✅ Created wrapper: {wrapper_path}")
    
    cmd = [
        str(wrapper_path),
        "--input_network", str(gguf_path),
        "--output_path", str(dlc_path),
        "--input_layout", "input_ids,NONTRIVIAL",
        "--input_layout", "attention_mask,NONTRIVIAL", 
        "--preserve_io", "datatype,input_ids,attention_mask",
        "--float_fallback",
        "--float_bitwidth", "16", 
        "--enable_cpu_fallback"
    ]
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
        
        if result.returncode == 0:
            print("✅ Wrapper conversion successful!")
            if dlc_path.exists():
                print(f"   DLC: {dlc_path} ({dlc_path.stat().st_size / (1024**2):.1f} MB)")
                return 0
        else:
            print(f"⚠️ Wrapper conversion failed: {result.stderr[:500]}")
            
    except Exception as e:
        print(f"⚠️ Wrapper conversion error: {e}")
    
    print()
    print("🎯 Method 3: Manual command for reference")
    print("-" * 70)
    
    manual_cmd = f'''
# Manual command to run in terminal:
export QNN_SDK_ROOT="{qnn_sdk_root}"
export PYTHONPATH="{qnn_sdk_root}/lib/python:$PYTHONPATH"
export LD_LIBRARY_PATH="{qnn_sdk_root}/lib/x86_64-linux-clang:$LD_LIBRARY_PATH"

python3 "{qnn_sdk_root}/bin/x86_64-linux-clang/qairt-converter" \\
    --input_network "{gguf_path}" \\
    --output_path "{dlc_path}" \\
    --input_layout "input_ids,NONTRIVIAL" \\
    --input_layout "attention_mask,NONTRIVIAL" \\
    --preserve_io "datatype,input_ids,attention_mask" \\
    --float_fallback \\
    --float_bitwidth 16 \\
    --enable_cpu_fallback
'''
    
    print(manual_cmd)
    
    # Save manual command to file
    cmd_file = output_dir / "manual_conversion_command.sh"
    with open(cmd_file, 'w') as f:
        f.write("#!/bin/bash\n")
        f.write(manual_cmd)
    cmd_file.chmod(0o755)
    
    print(f"💾 Manual command saved to: {cmd_file}")
    print()
    print("=" * 70)
    print("📋 Summary")
    print("=" * 70)
    print("Linux conversion attempted with multiple methods.")
    print("The QNN SDK libraries exist and should work on Linux.")
    print("The issue appears to be Python module import resolution.")
    print()
    print("Next steps:")
    print(f"1. Run the wrapper script: {wrapper_path}")
    print(f"2. Or execute manually: bash {cmd_file}")
    print("3. Check temp/ directory for output files")
    
    return 1

if __name__ == "__main__":
    sys.exit(convert_using_wrapper())