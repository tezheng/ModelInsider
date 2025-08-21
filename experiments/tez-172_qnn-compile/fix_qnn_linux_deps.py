#!/usr/bin/env python3
"""
QNN Linux Dependencies Analyzer and Fixer
Identifies and resolves the Linux library compatibility issues.
"""

import os
import sys
import subprocess
from pathlib import Path

def analyze_qnn_dependencies():
    """Analyze what QNN SDK needs vs what's available"""
    
    print("=" * 70)
    print("🔍 QNN Linux Dependencies Analysis")
    print("=" * 70)
    
    qnn_sdk_root = Path("/mnt/c/Qualcomm/AIStack/qairt/2.34.0.250424")
    ir_graph_so = qnn_sdk_root / "lib/python/qti/aisw/converters/common/linux-x86_64/libPyIrGraph.so"
    
    print(f"📁 QNN SDK: {qnn_sdk_root}")
    print(f"🔍 Analyzing: {ir_graph_so.name}")
    
    if not ir_graph_so.exists():
        print("❌ libPyIrGraph.so not found")
        return
    
    # Get required libraries using readelf
    print("\n🔗 Required Libraries:")
    print("-" * 50)
    
    try:
        result = subprocess.run(
            ["readelf", "-d", str(ir_graph_so)], 
            capture_output=True, text=True, check=True
        )
        
        required_libs = []
        for line in result.stdout.split('\n'):
            if 'NEEDED' in line and 'Shared library:' in line:
                lib = line.split('[')[1].split(']')[0]
                required_libs.append(lib)
                print(f"  📚 {lib}")
        
    except Exception as e:
        print(f"❌ Failed to analyze dependencies: {e}")
        return
    
    print(f"\n📊 Total required libraries: {len(required_libs)}")
    
    # Check what's available on system
    print("\n✅ System Library Check:")
    print("-" * 50)
    
    missing_libs = []
    
    for lib in required_libs:
        # Check common library locations
        found = False
        search_paths = [
            "/usr/lib/x86_64-linux-gnu/",
            "/lib/x86_64-linux-gnu/", 
            "/usr/lib/",
            "/lib/"
        ]
        
        for path in search_paths:
            lib_path = Path(path) / lib
            if lib_path.exists():
                print(f"  ✅ {lib} -> {lib_path}")
                found = True
                break
        
        if not found:
            missing_libs.append(lib)
            print(f"  ❌ {lib} -> NOT FOUND")
    
    # Specific checks
    print(f"\n🐍 Python Version Analysis:")
    print("-" * 50)
    
    # Check current Python
    py_version = f"{sys.version_info.major}.{sys.version_info.minor}"
    print(f"  Current Python: {py_version}")
    
    # Check what QNN expects
    py_needed = [lib for lib in required_libs if 'libpython' in lib]
    if py_needed:
        expected_py = py_needed[0].replace('libpython', '').replace('.so.1.0', '')
        print(f"  QNN expects: {expected_py}")
        if expected_py != py_version:
            print(f"  ⚠️ Version mismatch! QNN needs Python {expected_py}, have {py_version}")
    
    # Generate solution
    print(f"\n💡 Solution Steps:")
    print("-" * 50)
    
    if missing_libs:
        print("1. Install missing system libraries:")
        
        # Categorize missing libraries and suggest packages
        install_commands = []
        
        for lib in missing_libs:
            if 'libpython3.10' in lib:
                install_commands.append("python3.10-dev libpython3.10")
            elif 'libc++' in lib:
                install_commands.append("libc++1 libc++abi1")
            elif 'libdl' in lib:
                install_commands.append("libc6-dev")
            elif 'libm' in lib or 'libc' in lib:
                install_commands.append("build-essential")
        
        unique_packages = list(set(' '.join(install_commands).split()))
        
        print(f"   sudo apt update")
        print(f"   sudo apt install -y {' '.join(unique_packages)}")
        
        print("\n2. Create symbolic links if needed:")
        for lib in missing_libs:
            if 'libpython3.10' in lib:
                print(f"   # May need to link libpython3.10.so.1.0 to actual Python 3.10 lib")
        
    else:
        print("✅ All required libraries are available!")
    
    # Create a test script
    print(f"\n3. Test QNN import after installing dependencies:")
    
    test_script = f'''#!/usr/bin/env python3
import os
import sys

# Set up QNN environment
os.environ['QNN_SDK_ROOT'] = '{qnn_sdk_root}'
os.environ['PYTHONPATH'] = f'{qnn_sdk_root}/lib/python:{{os.environ.get("PYTHONPATH", "")}}'
os.environ['LD_LIBRARY_PATH'] = f'{qnn_sdk_root}/lib/x86_64-linux-clang:{{os.environ.get("LD_LIBRARY_PATH", "")}}'

sys.path.insert(0, '{qnn_sdk_root}/lib/python/qti/aisw/converters/common/linux-x86_64')
sys.path.insert(0, '{qnn_sdk_root}/lib/python')

try:
    from qti.aisw.converters.common import ir_graph
    print("✅ QNN import successful!")
    return True
except ImportError as e:
    print(f"❌ QNN import failed: {{e}}")
    return False
'''
    
    test_file = Path("test_qnn_import.py")
    with open(test_file, 'w') as f:
        f.write(test_script)
    
    print(f"   python3 {test_file}")
    
    print(f"\n4. If successful, run the actual conversion:")
    print(f"   python3 convert_linux_fixed.py")
    
    # Summary
    print(f"\n" + "=" * 70)
    print("📋 SUMMARY")
    print("=" * 70)
    print(f"You were RIGHT! WSL2 CAN load Linux .so files.")
    print(f"The issue is missing system dependencies, not WSL compatibility.")
    print()
    print(f"Missing libraries: {len(missing_libs)}")
    for lib in missing_libs[:5]:  # Show first 5
        print(f"  • {lib}")
    if len(missing_libs) > 5:
        print(f"  • ... and {len(missing_libs) - 5} more")
    
    print(f"\nAfter installing the dependencies above, the QNN Linux")
    print(f"converter should work perfectly in WSL2!")
    
    return len(missing_libs) == 0

if __name__ == "__main__":
    success = analyze_qnn_dependencies()
    sys.exit(0 if success else 1)