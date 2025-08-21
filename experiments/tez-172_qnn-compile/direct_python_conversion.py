#!/usr/bin/env python3
"""
Direct Python approach to QNN conversion using native bindings.
Bypass qairt-converter wrapper and use QNN Python modules directly.
"""

import os
import sys
from pathlib import Path

def setup_qnn_python_environment():
    """Setup QNN Python environment for direct module usage"""
    qnn_sdk_root = Path("/mnt/c/Qualcomm/AIStack/qairt/2.34.0.250424")
    
    print("🔧 Setting up QNN Python environment...")
    
    # Environment variables
    os.environ['QNN_SDK_ROOT'] = str(qnn_sdk_root)
    
    # Add QNN Python libraries to path
    qnn_python_dir = qnn_sdk_root / "lib" / "python"
    sys.path.insert(0, str(qnn_python_dir))
    
    # Try to work around the architecture issue by using available libraries
    common_dir = qnn_python_dir / "qti" / "aisw" / "converters" / "common"
    
    # Check what we have available
    available_dirs = []
    for dir_name in ['linux-x86_64', 'linux-aarch64', 'windows-x86_64', 'windows-arm64ec']:
        arch_dir = common_dir / dir_name
        if arch_dir.exists():
            available_dirs.append(str(arch_dir))
            print(f"   Found: {dir_name}")
    
    # Try each available directory
    for arch_dir in available_dirs:
        sys.path.insert(0, arch_dir)
    
    return qnn_sdk_root

def test_direct_imports():
    """Test if we can import QNN modules directly"""
    print("\n🧪 Testing direct QNN Python imports...")
    
    # Try importing each component we need
    imports_success = {}
    
    # Test basic QNN imports
    try:
        import qti
        imports_success['qti'] = True
        print("   ✅ qti base module")
    except Exception as e:
        imports_success['qti'] = False
        print(f"   ❌ qti: {e}")
    
    # Test converters module
    try:
        from qti.aisw import converters
        imports_success['converters'] = True
        print("   ✅ converters module")
    except Exception as e:
        imports_success['converters'] = False
        print(f"   ❌ converters: {e}")
    
    # Test specific converter components
    try:
        from qti.aisw.converters.common import converter_ir
        imports_success['converter_ir'] = True
        print("   ✅ converter_ir")
    except Exception as e:
        imports_success['converter_ir'] = False
        print(f"   ❌ converter_ir: {e}")
    
    # Test ONNX frontend (what we need for GGUF)
    try:
        from qti.aisw.converters import onnx
        imports_success['onnx_frontend'] = True
        print("   ✅ onnx frontend")
    except Exception as e:
        imports_success['onnx_frontend'] = False
        print(f"   ❌ onnx frontend: {e}")
    
    # Test backend
    try:
        from qti.aisw.converters.backend import qnn_backend
        imports_success['qnn_backend'] = True
        print("   ✅ qnn_backend")
    except Exception as e:
        imports_success['qnn_backend'] = False
        print(f"   ❌ qnn_backend: {e}")
    
    # Test if we can import the LLM builder for GGUF support
    try:
        from qti.aisw.converters.llm_builder import LLMBuilder
        imports_success['llm_builder'] = True
        print("   ✅ LLMBuilder (GGUF support)")
    except Exception as e:
        imports_success['llm_builder'] = False
        print(f"   ❌ LLMBuilder: {e}")
    
    return imports_success

def attempt_direct_conversion():
    """Attempt direct conversion using Python API"""
    print("\n🔄 Attempting direct Python conversion...")
    
    # Paths
    script_dir = Path(__file__).parent
    gguf_path = script_dir / "models" / "DeepSeek-R1-Distill-Qwen-1.5B-Q4_0.gguf"
    output_dir = script_dir / "temp"
    output_dir.mkdir(exist_ok=True)
    
    if not gguf_path.exists():
        print(f"❌ GGUF model not found: {gguf_path}")
        return False
    
    print(f"📦 Input: {gguf_path}")
    
    try:
        # Try using LLMBuilder directly (if available)
        from qti.aisw.converters.llm_builder import LLMBuilder
        
        print("🚀 Using LLMBuilder for native GGUF conversion...")
        
        builder = LLMBuilder(
            input_model=str(gguf_path),
            output_dir=str(output_dir)
        )
        
        # This should do the GGUF parsing and ONNX generation internally
        onnx_path, encodings_path, input_layouts, inputs_to_preserve = builder.build_from_gguf()
        
        print(f"✅ LLMBuilder conversion successful!")
        print(f"   ONNX: {onnx_path}")
        print(f"   Encodings: {encodings_path}")
        
        # Now we'd need to convert ONNX to QNN DLC
        # This would require the QNN backend
        from qti.aisw.converters.backend.qnn_backend import QnnBackend
        
        backend = QnnBackend()
        # Continue with DLC generation...
        
        return True
        
    except ImportError as e:
        print(f"❌ LLMBuilder not available: {e}")
        
        # Try alternative approach using ONNX converter
        try:
            from qti.aisw.converters.onnx.onnx_to_ir import OnnxConverterFrontend
            print("Trying ONNX converter approach...")
            
            # This would require first converting GGUF to ONNX externally
            return False
            
        except ImportError as e2:
            print(f"❌ ONNX converter also not available: {e2}")
            return False
    
    except Exception as e:
        print(f"❌ Conversion failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Main conversion workflow using direct Python approach"""
    
    print("=" * 70)
    print("🐍 Direct Python QNN Conversion (Native Bindings)")
    print("=" * 70)
    print("Attempting to use QNN Python modules directly")
    print("This bypasses qairt-converter wrapper and architecture issues")
    print()
    
    # Setup environment
    qnn_sdk_root = setup_qnn_python_environment()
    if not qnn_sdk_root:
        return 1
    
    # Test imports
    imports = test_direct_imports()
    
    # Count successful imports
    successful = sum(imports.values())
    total = len(imports)
    
    print(f"\n📊 Import Results: {successful}/{total} successful")
    
    if successful == 0:
        print("❌ No QNN modules could be imported")
        print("   This confirms the architecture compatibility issue")
        return 1
    
    if imports.get('llm_builder', False):
        print("✅ LLMBuilder available - can proceed with GGUF conversion")
        success = attempt_direct_conversion()
        return 0 if success else 1
    else:
        print("⚠️ LLMBuilder not available - would need ONNX intermediate step")
        return 1

if __name__ == "__main__":
    sys.exit(main())