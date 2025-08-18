#!/usr/bin/env python3
"""
Quick profiling test for Windows ARM64 with Qualcomm NPU.
Minimal dependencies - uses only QNN command-line tools.
"""

import os
import sys
import json
import subprocess
from pathlib import Path
import numpy as np

def find_qnn_sdk():
    """Find QNN SDK installation"""
    # Check environment variable
    if "QNN_SDK_ROOT" in os.environ:
        return Path(os.environ["QNN_SDK_ROOT"])
    
    # Check common Windows locations
    common_paths = [
        r"C:\Qualcomm\AIStack\qairt\2.34.0.250424",
        r"C:\Qualcomm\AIStack\QAIRT\2.34.0.250424",
        r"C:\Program Files\Qualcomm\AIStack\qairt",
    ]
    
    for path in common_paths:
        if Path(path).exists():
            return Path(path)
    
    return None

def test_qnn_profiling():
    """Quick test of QNN profiling capabilities"""
    
    print("QNN Quick Profiling Test")
    print("=" * 50)
    
    # Find SDK
    sdk_path = find_qnn_sdk()
    if not sdk_path:
        print("ERROR: QNN SDK not found!")
        print("Please set QNN_SDK_ROOT environment variable")
        return False
    
    print(f"✓ Found QNN SDK: {sdk_path}")
    
    # Check for required binaries
    binaries_to_check = {
        "qnn-net-run": ["bin", "aarch64-windows-msvc", "qnn-net-run.exe"],
        "qnn-profile-viewer": ["bin", "aarch64-windows-msvc", "qnn-profile-viewer.exe"],
        "HTP Backend": ["lib", "aarch64-windows-msvc", "libQnnHtp.dll"],
    }
    
    # Also check x86_64 paths as fallback
    alt_paths = {
        "qnn-net-run": ["bin", "x86_64-windows-msvc", "qnn-net-run.exe"],
        "qnn-profile-viewer": ["bin", "x86_64-windows-msvc", "qnn-profile-viewer.exe"],
        "HTP Backend": ["lib", "x86_64-windows-msvc", "QnnHtp.dll"],
    }
    
    found_binaries = {}
    for name, path_parts in binaries_to_check.items():
        full_path = sdk_path.joinpath(*path_parts)
        if not full_path.exists():
            # Try alternative path
            alt_path_parts = alt_paths.get(name, [])
            if alt_path_parts:
                full_path = sdk_path.joinpath(*alt_path_parts)
        
        if full_path.exists():
            print(f"✓ Found {name}: {full_path.name}")
            found_binaries[name] = full_path
        else:
            print(f"✗ Missing {name}")
    
    if len(found_binaries) < 3:
        print("\nERROR: Required QNN binaries not found")
        return False
    
    # Test profiling capability
    print("\n" + "=" * 50)
    print("Testing Profiling Capability")
    print("=" * 50)
    
    # Create test directory
    test_dir = Path("quick_test_output")
    test_dir.mkdir(exist_ok=True)
    
    # Check if we can query backend capabilities
    qnn_net_run = found_binaries.get("qnn-net-run")
    htp_backend = found_binaries.get("HTP Backend")
    
    if qnn_net_run and htp_backend:
        # Test backend query
        cmd = [str(qnn_net_run), "--version"]
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=5)
            if result.returncode == 0:
                print("✓ QNN tools are executable")
                print(f"  Version info: {result.stdout.strip()[:100]}...")
            else:
                print("✗ QNN tools execution failed")
        except Exception as e:
            print(f"✗ Error running QNN tools: {e}")
    
    # Check Python SDK availability
    print("\n" + "=" * 50)
    print("Checking Python SDK")
    print("=" * 50)
    
    python_sdk_path = sdk_path / "lib" / "python"
    if python_sdk_path.exists():
        sys.path.insert(0, str(python_sdk_path))
        try:
            import qairt
            print(f"✓ Python SDK available: qairt version {getattr(qairt, '__version__', 'unknown')}")
            
            # Try to import profiling modules
            from qairt.api.profiler import ProfilerContext
            print("✓ Profiling API available")
            
            # List available profiling levels
            from qairt.constants import ProfilingLevel
            levels = [attr for attr in dir(ProfilingLevel) if not attr.startswith('_')]
            print(f"  Available profiling levels: {', '.join(levels)}")
            
        except ImportError as e:
            print(f"✗ Python SDK import failed: {e}")
            print("  You can still use command-line tools for profiling")
    else:
        print("✗ Python SDK not found")
        print("  Command-line tools can still be used")
    
    # Generate summary
    print("\n" + "=" * 50)
    print("PROFILING CAPABILITY SUMMARY")
    print("=" * 50)
    
    print("\n📋 Available Profiling Methods:")
    if "qnn-net-run" in found_binaries:
        print("  1. Command-line: qnn-net-run.exe with --profiling_level")
    if python_sdk_path.exists():
        print("  2. Python SDK: qairt.api.profiler")
    
    print("\n📊 Profiling Outputs:")
    print("  • Profiling log: qnn-profiling-data_0.log")
    print("  • Chrome trace: Use qnn-profile-viewer to generate")
    print("  • QHAS summary: Performance analysis report")
    
    print("\n🎯 Next Steps:")
    print("  1. Prepare your model (ONNX or DLC format)")
    print("  2. Run: python run_real_profiling_windows.py")
    print("  3. View results in Chrome: chrome://tracing")
    
    # Save configuration for later use
    config = {
        "qnn_sdk_root": str(sdk_path),
        "qnn_net_run": str(found_binaries.get("qnn-net-run", "")),
        "profile_viewer": str(found_binaries.get("qnn-profile-viewer", "")),
        "htp_backend": str(found_binaries.get("HTP Backend", "")),
        "python_sdk_available": python_sdk_path.exists()
    }
    
    config_file = test_dir / "qnn_config.json"
    with open(config_file, 'w') as f:
        json.dump(config, f, indent=2)
    
    print(f"\n✓ Configuration saved to: {config_file}")
    
    return True

if __name__ == "__main__":
    success = test_qnn_profiling()
    sys.exit(0 if success else 1)