#!/usr/bin/env python3
"""
Fixed Linux x86_64 QNN Conversion - Include architecture-specific Python paths.
"""

import os
import sys
import subprocess
from pathlib import Path

def setup_qnn_environment_fixed():
    """Setup QNN environment with correct architecture-specific paths"""
    
    qnn_sdk_root = Path("/mnt/c/Qualcomm/AIStack/qairt/2.34.0.250424")
    
    print("Setting up QNN x86_64 Linux environment (with arch-specific paths)...")
    
    # Environment variables
    os.environ['QNN_SDK_ROOT'] = str(qnn_sdk_root)
    
    # Python path - include architecture-specific directory
    qnn_python_dir = qnn_sdk_root / "lib" / "python"
    qnn_arch_python_dir = qnn_python_dir / "qti" / "aisw" / "converters" / "common" / "linux-x86_64"
    
    python_paths = [
        str(qnn_python_dir),
        str(qnn_arch_python_dir),
    ]
    os.environ['PYTHONPATH'] = ":".join(python_paths + [os.environ.get('PYTHONPATH', '')])
    
    # Library path for x86_64 Linux libraries  
    qnn_lib_dir = qnn_sdk_root / "lib" / "x86_64-linux-clang"
    os.environ['LD_LIBRARY_PATH'] = f"{qnn_lib_dir}:{os.environ.get('LD_LIBRARY_PATH', '')}"
    
    # Add to Python sys.path - architecture-specific first
    sys.path.insert(0, str(qnn_arch_python_dir))
    sys.path.insert(0, str(qnn_python_dir))
    
    print(f"QNN_SDK_ROOT: {qnn_sdk_root}")
    print(f"Python paths:")
    print(f"  Base: {qnn_python_dir}")
    print(f"  Arch: {qnn_arch_python_dir} {'(EXISTS)' if qnn_arch_python_dir.exists() else '(MISSING)'}")
    print(f"Library path: {qnn_lib_dir}")
    print()
    
    return qnn_sdk_root

def test_qnn_imports_detailed():
    """Test imports with detailed error reporting"""
    
    print("Testing QNN Python imports (detailed)...")
    
    try:
        # Test architecture-specific import first
        arch_dir = Path("/mnt/c/Qualcomm/AIStack/qairt/2.34.0.250424/lib/python/qti/aisw/converters/common/linux-x86_64")
        if arch_dir.exists():
            lib_files = list(arch_dir.glob("libPyIrGraph*.so"))
            print(f"Found {len(lib_files)} PyIrGraph libraries in {arch_dir.name}:")
            for lib in lib_files:
                print(f"  - {lib.name}")
        
        # Test direct import of libPyIrGraph
        print("\nTesting direct libPyIrGraph import...")
        import libPyIrGraph as ir_graph
        print("✅ libPyIrGraph imported directly!")
        
        # Test through qti module
        print("\nTesting qti module imports...")
        import qti
        print("✅ qti base module")
        
        from qti.aisw.converters.common import ir_graph as qti_ir
        print("✅ qti ir_graph imported!")
        
        # Test LLM builder
        from qti.aisw.converters.llm_builder import LLMBuilder
        print("✅ LLMBuilder available!")
        
        return True
        
    except ImportError as e:
        print(f"❌ Import failed: {e}")
        
        # Try to diagnose the issue
        try:
            import ctypes
            arch_lib = "/mnt/c/Qualcomm/AIStack/qairt/2.34.0.250424/lib/python/qti/aisw/converters/common/linux-x86_64/libPyIrGraph.so"
            print(f"\nTrying to load {arch_lib} with ctypes...")
            lib = ctypes.CDLL(arch_lib)
            print("✅ Library loaded with ctypes - this is a Python binding issue, not architecture")
        except Exception as e2:
            print(f"❌ ctypes load failed: {e2}")
        
        return False
    except Exception as e:
        print(f"❌ Other error: {e}")
        return False

def run_qairt_converter_directly():
    """Run qairt-converter directly without Python imports"""
    
    print("\n" + "=" * 60)
    print("Direct qairt-converter Execution (Bypass Python Imports)")
    print("=" * 60)
    print("Running the qairt-converter Python script directly")
    print()
    
    # Paths
    script_dir = Path(__file__).parent
    gguf_path = script_dir / "models" / "DeepSeek-R1-Distill-Qwen-1.5B-Q4_0.gguf"
    temp_dir = script_dir / "temp"
    temp_dir.mkdir(exist_ok=True)
    
    dlc_path = temp_dir / "deepseek_direct.dlc"
    
    if not gguf_path.exists():
        print(f"❌ GGUF model not found: {gguf_path}")
        return False
    
    print(f"📦 Input: {gguf_path}")
    print(f"📄 DLC Output: {dlc_path}")
    
    # Run qairt-converter as Python script directly
    qnn_sdk_root = Path(os.environ['QNN_SDK_ROOT'])
    converter = qnn_sdk_root / "bin" / "x86_64-linux-clang" / "qairt-converter"
    
    # Set up environment for the subprocess
    env = os.environ.copy()
    
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
    
    print("Running qairt-converter directly...")
    print(f"Command: python3 {converter.name} [args]")
    
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=600,
            env=env,
            cwd=str(temp_dir)
        )
        
        if result.returncode == 0:
            print("✅ Direct conversion successful!")
            if dlc_path.exists():
                size_mb = dlc_path.stat().st_size / (1024**2)
                print(f"   DLC size: {size_mb:.1f} MB")
                return True
            else:
                print("❌ No output file generated")
                return False
        else:
            print("❌ Direct conversion failed")
            print("STDOUT:", result.stdout[:500])
            print("STDERR:", result.stderr[:500])
            return False
            
    except subprocess.TimeoutExpired:
        print("❌ Conversion timed out")
        return False
    except Exception as e:
        print(f"❌ Execution error: {e}")
        return False

def main():
    """Main execution"""
    
    print("=" * 70)
    print("🔧 Fixed Linux x86_64 QNN Conversion")
    print("=" * 70)
    print("Using architecture-specific Python paths and direct execution")
    print()
    
    # Setup environment
    setup_qnn_environment_fixed()
    
    # Test imports (diagnostic)
    print("=" * 50)
    print("DIAGNOSTIC: Testing Python Import Path")
    print("=" * 50)
    import_success = test_qnn_imports_detailed()
    
    # Always try direct execution regardless of import results
    print("\n" + "=" * 50)
    print("EXECUTION: Direct qairt-converter Run")
    print("=" * 50)
    direct_success = run_qairt_converter_directly()
    
    if direct_success:
        print("\n" + "=" * 70)
        print("🎉 SUCCESS: QNN conversion completed!")
        print("=" * 70)
        print("Real .dlc file generated in temp/ directory")
        return 0
    else:
        print(f"\n❌ Conversion failed")
        print("Import success:", import_success)
        print("Direct execution success:", direct_success)
        return 1

if __name__ == "__main__":
    sys.exit(main())